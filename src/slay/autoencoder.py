from typing import Any

import numpy as np
import scipy.spatial.distance as dist
import torch
import torch.nn as nn
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split
from spikeinterface.core import SortingAnalyzer, get_template_extremum_channel
from spikeinterface.postprocessing import compute_template_similarity
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from tqdm import tqdm

from .utils import get_channels_by_distance


def extract_spike_snippets(
    sorting_analyzer: SortingAnalyzer,
    autoencoder_params: dict[str, Any],
) -> tuple[torch.Tensor, NDArray[np.int_]]:
    """
    Extracts spike snippets from a SpikeInterface SortingAnalyzer for training
    a convolutional autoencoder that learns waveform shape features.

    The first channel in each snippet is the channel with peak amplitude, and the
    remaining channels are ordered by distance from the peak channel. This removes
    info about the absolute position of the spikes, so the autoencoder can focus only
    on waveform shape.

    Spike waveforms are extracted directly from the recording using the spike times
    from the "random_spikes" extension.

    Args:
        sorting_analyzer (SortingAnalyzer): A SpikeInterface SortingAnalyzer object.
            Must have the following extensions computed:
                - "templates": Average waveforms per unit (used for channel selection)
                - "random_spikes": Random spike selection for training
        autoencoder_params (dict): Autoencoder configuration parameters:
            - n_chan (int): Total number of channels on the probe.
            - num_chan (int): Number of channels to include in each spike snippet.
            - ms_before (float, optional): Milliseconds before spike peak. Defaults to 1.0.
            - ms_after (float, optional): Milliseconds after spike peak. Defaults to 2.0.

    Returns:
        spikes (torch.Tensor): Extracted spike snippets with shape
            (# snippets, 1, # channels, # timepoints). This Tensor lives on the GPU
            if available.
        unit_ids (NDArray): Unit labels of spike snippets with shape (# snippets).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get required extensions
    templates_ext = sorting_analyzer.get_extension("templates")
    random_spikes_ext = sorting_analyzer.get_extension("random_spikes")

    if templates_ext is None:
        raise ValueError("SortingAnalyzer must have 'templates' extension computed.")
    if random_spikes_ext is None:
        raise ValueError(
            "SortingAnalyzer must have 'random_spikes' extension computed."
        )

    # Get recording, sorting, and templates
    recording = sorting_analyzer.recording
    unit_ids_list = sorting_analyzer.unit_ids

    # Calculate timing parameters
    sampling_frequency = recording.get_sampling_frequency()
    ms_before = autoencoder_params.get("ms_before", 1 / 3)
    ms_after = autoencoder_params.get("ms_after", 1.0)
    num_samples_before = int(ms_before * sampling_frequency / 1000)
    num_samples_after = int(ms_after * sampling_frequency / 1000)
    last_sample = sorting_analyzer.recording.get_total_samples()
    num_samples = num_samples_before + num_samples_after

    # Pre-compute the set of closest channels for each unit ordered by distance from peak and the total number of snippets
    chans = {}
    peak_chans = get_template_extremum_channel(
        sorting_analyzer, peak_sign="both", outputs="index"
    )
    num_snippets = 0
    random_spikes = random_spikes_ext.get_random_spikes()
    spikes_in_bounds = (random_spikes["sample_index"] >= num_samples_before) & (
        random_spikes["sample_index"] <= last_sample - num_samples_after - 1
    )
    for unit_idx in range(len(unit_ids_list)):
        chans[unit_idx] = get_channels_by_distance(
            peak_chans[unit_ids_list[unit_idx]],
            sorting_analyzer,
            autoencoder_params["num_chan"],
        )

        num_spikes = np.sum(
            (random_spikes["unit_index"] == unit_idx) & spikes_in_bounds
        )
        num_snippets += num_spikes

    # Pre-allocate memory for the snippets
    spikes = np.zeros(
        (
            num_snippets,
            1,
            autoencoder_params["num_chan"],
            num_samples,
        ),
        dtype=np.float32,
    )
    spike_labels = np.zeros(num_snippets, dtype="int32")

    # Extract waveforms from recording traces for each unit
    snip_idx = 0
    for unit_idx in tqdm(range(len(unit_ids_list)), desc="Extracting snippets"):
        spike_times = random_spikes["sample_index"][
            (random_spikes["unit_index"] == unit_idx) & spikes_in_bounds
        ]
        desired_channels = chans[unit_idx]
        n_spikes_unit = len(spike_times)

        # Extract waveforms for all spikes
        snippets = np.zeros(
            (n_spikes_unit, autoencoder_params["num_chan"], num_samples)
        )

        for i, spike_time in enumerate(spike_times):
            start_frame = int(spike_time - num_samples_before)
            end_frame = int(spike_time + num_samples_after)

            snippets[i] = recording.get_traces(
                start_frame=start_frame,
                end_frame=end_frame,
                channel_ids=desired_channels,
                return_in_uV=False,
            ).T

        # Store unit labels and waveforms
        spike_labels[snip_idx : snip_idx + n_spikes_unit] = unit_idx
        spikes[snip_idx : snip_idx + n_spikes_unit, 0, :, :] = snippets

        snip_idx += n_spikes_unit

    return torch.Tensor(spikes).to(device), spike_labels


class SpikeDataset(Dataset):
    """
    A Dataset class to hold spike snippets.

    Attributes:
        spikes (torch.Tensor): Spike snippets.
        labels (NDArray): Cluster ids for each snippet.
        transform (function, optional): Transform to apply to snippets.
        target_transform (function, optional): Transform to apply to labels.
    """

    def __init__(
        self,
        spikes: torch.Tensor,
        labels: NDArray[np.int_],
        transform=None,
        target_transform=None,
    ) -> None:
        self.spikes: torch.Tensor = spikes
        self.labels: NDArray[np.int_] = labels
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return self.spikes.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        spk: torch.Tensor = self.spikes[idx]
        label: int = self.labels[idx]
        if self.transform:
            spk = self.transform(spk)
        if self.target_transform:
            label = self.target_transform(label)
        return spk, label, idx


class CN_AE(nn.Module):
    """
    A convolutional autoencoder for spike snippet feature extraction.

    This model consists of three convolutional layers with pooling operations in the encoder,
    followed by a fully connected layer as the bottleneck, and then three convolutional transpose
    layers with upsampling operations in the decoder. Each layer is followed by a ReLU activation
    except the output layer.

    Attributes:
        num_chan (int): number of channels in snippets
        num_samp (int): number of timepoints per channel in snippets
        n_filt (int): number of convolutional filters in the bottleneck
        half_filt (int): number of filters for the second convolutional layer in the encoder
        qrt_filt (int): number of filters for the first convolutional layer in the encoder
        featureDim: number of features in the input to the fully connected bottleneck
    """

    def __init__(
        self,
        imgChannels: int = 1,
        n_filt: int = 256,
        zDim: int = 15,
        num_chan: int = 8,
        num_samp: int = 40,
    ) -> None:
        super(CN_AE, self).__init__()
        self.num_chan = num_chan
        self.num_samp = num_samp
        self.n_filt = n_filt
        self.half_filt = n_filt // 2
        self.qrt_filt = n_filt // 4
        self.featureDim = (n_filt * num_chan // 8) * (num_samp // 8)

        self.encoder_conv = nn.Sequential()

        self.encoder_conv.append(
            nn.Conv2d(imgChannels, self.qrt_filt, kernel_size=3, padding="same")
        )
        self.encoder_conv.append(nn.BatchNorm2d(self.qrt_filt))
        self.encoder_conv.append(nn.ReLU())
        self.encoder_conv.append(nn.MaxPool2d(kernel_size=2))

        self.encoder_conv.append(
            nn.Conv2d(self.qrt_filt, self.half_filt, kernel_size=3, padding="same")
        )
        self.encoder_conv.append(nn.BatchNorm2d(self.half_filt))
        self.encoder_conv.append(nn.ReLU())
        self.encoder_conv.append(nn.MaxPool2d(kernel_size=2))

        self.encoder_conv.append(
            nn.Conv2d(self.half_filt, n_filt, kernel_size=3, padding="same")
        )
        self.encoder_conv.append(nn.BatchNorm2d(n_filt))
        self.encoder_conv.append(nn.ReLU())
        self.encoder_conv.append(nn.MaxPool2d(kernel_size=2))

        self.encoder_fc = nn.Sequential()
        self.encoder_fc.append(nn.Linear(self.featureDim, zDim))
        self.encoder_fc.append(nn.ReLU())

        self.decoder_fc = nn.Sequential()
        self.decoder_fc.append(nn.Linear(zDim, self.featureDim))

        self.decoder_conv = nn.Sequential()
        self.decoder_conv.append(nn.BatchNorm2d(n_filt))
        self.decoder_conv.append(nn.ReLU())

        self.decoder_conv.append(nn.Upsample(size=(num_chan // 4, num_samp // 4)))
        self.decoder_conv.append(
            nn.ConvTranspose2d(n_filt, self.half_filt, kernel_size=3, padding=1)
        )
        self.decoder_conv.append(nn.BatchNorm2d(self.half_filt))
        self.decoder_conv.append(nn.ReLU())

        self.decoder_conv.append(nn.Upsample(size=(num_chan // 2, num_samp // 2)))
        self.decoder_conv.append(
            nn.ConvTranspose2d(self.half_filt, self.qrt_filt, kernel_size=3, padding=1)
        )
        self.decoder_conv.append(nn.BatchNorm2d(self.qrt_filt))
        self.decoder_conv.append(nn.ReLU())

        self.decoder_conv.append(nn.Upsample(size=(num_chan, num_samp)))
        self.decoder_conv.append(
            nn.ConvTranspose2d(self.qrt_filt, imgChannels, kernel_size=3, padding=1)
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder_conv(x)
        x = x.view(-1, self.featureDim)
        x = self.encoder_fc(x)

        return x

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.decoder_fc(z)
        x = x.view(-1, self.n_filt, self.num_chan // 8, self.num_samp // 8)
        x = self.decoder_conv(x)

        return x

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z: torch.Tensor = self.encode(x)
        out: torch.Tensor = self.decode(z)

        return out, z


def train_ae(
    spikes: torch.Tensor,
    cl_ids: NDArray[np.int_],
    n_filt: int = 256,
    num_epochs: int = 25,
    zDim: int = 15,
    lr: float = 1e-4,
    model=None,
    batch_size: int = 128,
    return_inds=False,
    verbose=True,
):
    """
    Creates and trains an autoencoder on the given spike dataset.

    Args:
        spikes (torch.Tensor): Spike snippets.
        cl_ids (NDArray): Cluster ids of spike snippets.
        n_filt (int): Number of filters in the last convolutional layer before
            the bottleneck. Defaults to 256, values larger than 1024 cause
            CUDA to run out of memory on most GPUs.
        num_epochs (int): number of training epochs. Defaults to 25.
        zDi (int): latent dimensionality of CN_AE. Defaults to 15.
        lr (float): optimizer learning rate. Defaults to 1e-3.
        pre_samples (int): number of samples included before spike time. Defaults
            to 10.
        post_samples (int): number of samples included after spike time.
            Defaults to 30.
        do_shft (bool): True if training samples should be randomly time-shifted to
            explicitly induce time shift invariance. Note that the architecture is
            implicitly invariant to time shifts due to the convolutional layers.
        model (nn.Module, optional): Pre-trained model, if using.
    Returns:
        net (CN_AE): The trained network.
        spk_data (SpikeDataset): Dataset containing snippets used for training.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Construct cluster-balanced dataloaders
    spk_data = SpikeDataset(spikes, cl_ids)
    labels = cl_ids

    train_indices, test_indices, _, _ = train_test_split(
        np.arange(len(spk_data)),
        labels,
        stratify=labels,
        test_size=0.2,
        random_state=42,
    )
    train_labels, train_counts = np.unique(labels[train_indices], return_counts=True)
    test_labels, test_counts = np.unique(labels[test_indices], return_counts=True)

    label_to_train_count = dict(zip(train_labels, train_counts))
    label_to_test_count = dict(zip(test_labels, test_counts))

    train_weights = [1 / label_to_train_count[label] for label in labels[train_indices]]
    test_weights = [1 / label_to_test_count[label] for label in labels[test_indices]]
    train_split = Subset(spk_data, train_indices)
    test_split = Subset(spk_data, test_indices)
    sampler = WeightedRandomSampler(
        weights=train_weights, num_samples=len(train_split), replacement=True
    )
    test_sampler = WeightedRandomSampler(
        weights=test_weights, num_samples=len(test_split), replacement=True
    )

    BATCH_SIZE = batch_size
    train_loader = DataLoader(train_split, batch_size=BATCH_SIZE, sampler=sampler)
    test_loader = DataLoader(test_split, batch_size=BATCH_SIZE, sampler=test_sampler)

    net = (
        model
        if model
        else CN_AE(zDim=zDim, n_filt=n_filt, num_samp=spikes.shape[-1]).to(device)
    )
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    # TRAIN/TEST LOOP
    for epoch in tqdm(range(num_epochs), desc="Epoch"):
        net.train()
        running_mse = 0

        # TRAINING ITERATION
        for spks, _, _ in tqdm(train_loader, desc="Training", leave=False):
            spks = spks.to(device)

            out, z_batch = net(spks)
            loss = torch.nn.functional.mse_loss(out, spks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_mse += loss.item()

        avg_train_mse = running_mse / len(train_loader)

        # TESTING ITERATION
        net.eval()
        running_tloss = 0
        with torch.no_grad():
            for spks, _, _ in tqdm(test_loader, desc="Testing", leave=False):
                spks = spks.to(device)

                out, _ = net(spks)
                tloss = torch.nn.functional.mse_loss(out, spks)
                running_tloss += tloss.item()

        avg_test_loss = running_tloss / len(test_loader)
        if verbose:
            tqdm.write(
                f"Epoch {epoch + 1:2d}/{num_epochs} | Train MSE: {avg_train_mse:.4f} | Test MSE: {avg_test_loss:.4f}"
            )

    if return_inds:
        return net, spk_data, test_indices
    else:
        return net, spk_data


def compute_autoencoder_similarity(
    sorting_analyzer: SortingAnalyzer,
    autoencoder: nn.Module,
    autoencoder_params: dict[str, Any] = {
        "num_chan": 8,
    },
    spike_dataset: SpikeDataset = None,
    zDim: int = 15,
) -> NDArray[np.float64]:
    """
    Calculates autoencoder-based unit similarity using latent space representations.

    Args:
        sorting_analyzer (SortingAnalyzer): SpikeInterface SortingAnalyzer containing
            unit templates and metadata.
        spike_dataset (SpikeDataset): Dataset containing spike snippets and unit labels.
        autoencoder (nn.Module): Trained autoencoder model in eval mode.
        zDim (int, optional): Latent dimensionality of the autoencoder. Defaults to 15.

    Returns:
        ae_sim (NDArray): Pairwise autoencoder-based similarity matrix with shape
            (# units, # units). ae_sim[i,j] = 1 indicates maximal similarity.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if spike_dataset is None:
        spike_snippets, unit_ids = extract_spike_snippets(
            sorting_analyzer, autoencoder_params
        )
        spike_dataset = SpikeDataset(spike_snippets, unit_ids)

    # Get templates and peak channels from sorting analyzer
    templates_ext = sorting_analyzer.get_extension("templates")
    if templates_ext is None:
        raise ValueError("SortingAnalyzer must have 'templates' extension computed.")

    templates = templates_ext.get_templates()
    unit_ids = sorting_analyzer.unit_ids
    n_units = len(unit_ids)

    peak_chans = get_template_extremum_channel(
        sorting_analyzer, peak_sign="both", outputs="index"
    )

    # Calculate latent representations of spikes
    dataloader = DataLoader(spike_dataset, batch_size=128)
    spike_latents = np.zeros((len(spike_dataset), zDim))
    spike_labels = np.zeros(len(spike_dataset), dtype="int32")
    loss_fn = nn.MSELoss()

    autoencoder.eval()
    loss = 0
    with torch.no_grad():
        for idx, data in enumerate(
            tqdm(dataloader, desc="Calculating latent representations")
        ):
            spikes, labels, idx = data[0].to(device), data[1].to(device), data[2]

            reconstructions, _ = autoencoder(spikes)
            loss += loss_fn(spikes, reconstructions).item()

            batch_latents = autoencoder.encode(spikes)
            spike_latents[idx] = batch_latents.cpu().detach().numpy()
            spike_labels[idx] = labels.cpu().detach().numpy()

    tqdm.write(f"\nAverage Loss: {loss / len(dataloader):.4f}")

    # Calculate cluster centroids (mean latent representation for each unit)
    unit_centroids = np.zeros((n_units, zDim))
    for unit_idx in range(n_units):
        unit_mask = spike_labels == unit_idx
        if np.sum(unit_mask) > 0:
            unit_centroids[unit_idx] = np.mean(spike_latents[unit_mask], axis=0)

    # Calculate pairwise euclidean distances between unit centroids
    centroid_distances = dist.squareform(dist.pdist(unit_centroids, "euclidean"))

    # Calibrate similarity threshold to 0.6 template cosine similarity
    template_similarity = compute_template_similarity(
        sorting_analyzer, method="cosine", save=False
    )
    ref_inds = np.argsort(np.abs(template_similarity.flatten() - 0.6))[-10:]
    ref_dist = np.mean(centroid_distances.flatten()[ref_inds])

    # Calculate similarity -- ref_dist is scaled to 0.6 similarity
    autoencoder_similarity = np.exp(-0.5 * centroid_distances / ref_dist)

    # Zero out self-similarity
    np.fill_diagonal(autoencoder_similarity, 0)

    # Penalize pairs with different peak channels
    amplitudes = np.max(templates, 1) - np.min(templates, 1)
    for i in tqdm(range(n_units), desc="Calculating similarity metric"):
        for j in range(n_units):
            if i >= j:
                continue

            unit_i_peak_chan = peak_chans[unit_ids[i]]
            unit_j_peak_chan = peak_chans[unit_ids[j]]

            # Penalize by sigmoided version of geometric mean of cross-decay
            if (
                (amplitudes[i, unit_j_peak_chan] == 0)
                or (amplitudes[i, unit_i_peak_chan] == 0)
                or (amplitudes[j, unit_i_peak_chan] == 0)
                or (amplitudes[j, unit_j_peak_chan] == 0)
            ):
                autoencoder_similarity[i, j] = 0
                continue
            else:
                decay_pen_raw = np.sqrt(
                    amplitudes[i, unit_j_peak_chan]
                    / amplitudes[i, unit_i_peak_chan]
                    * amplitudes[j, unit_i_peak_chan]
                    / amplitudes[j, unit_j_peak_chan]
                )

            # decay_pen = 1 / (1 + np.exp(-10 * (decay_pen_raw - 0.5)))

            autoencoder_similarity[i, j] *= decay_pen_raw
            autoencoder_similarity[j, i] = autoencoder_similarity[i, j]

    return autoencoder_similarity
