from typing import Any

import numpy as np
import scipy.spatial.distance as dist
import torch
import torch.nn as nn
from numpy.typing import NDArray
from scipy.sparse import lil_array
from scipy.sparse.csgraph import dijkstra
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from spikeinterface.core import SortingAnalyzer, get_template_extremum_channel
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
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
            autoencoder_params["num_chan"] * num_samples,
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
            (n_spikes_unit, autoencoder_params["num_chan"] * num_samples)
        )

        for i, spike_time in enumerate(spike_times):
            start_frame = int(spike_time - num_samples_before)
            end_frame = int(spike_time + num_samples_after)

            snippets[i] = recording.get_traces(
                start_frame=start_frame,
                end_frame=end_frame,
                channel_ids=desired_channels,
                return_in_uV=False,
            ).flatten()

        # Store unit labels and waveforms
        spike_labels[snip_idx : snip_idx + n_spikes_unit] = unit_idx
        spikes[snip_idx : snip_idx + n_spikes_unit, :] = snippets

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


class AE(nn.Module):
    def __init__(
        self,
        zDim: int = 15,
        num_chan: int = 8,
        num_samp: int = 40,
        n_units_l1: int = 600,
        n_units_l2: int = 300,
    ):
        super(AE, self).__init__()
        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()

        self.encoder.append(nn.Linear(num_chan * num_samp, n_units_l1))
        self.encoder.append(nn.GELU())
        self.encoder.append(nn.Linear(n_units_l1, n_units_l2))
        self.encoder.append(nn.GELU())
        self.encoder.append(nn.Linear(n_units_l2, zDim))
        self.encoder.append(nn.GELU())

        self.decoder.append(nn.Linear(zDim, n_units_l2))
        self.decoder.append(nn.GELU())
        self.decoder.append(nn.Linear(n_units_l2, n_units_l1))
        self.decoder.append(nn.GELU())
        self.decoder.append(nn.Linear(n_units_l1, num_chan * num_samp))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z: torch.Tensor = self.encode(x)
        out: torch.Tensor = self.decode(z)

        return out, z


def train_autoencoder(
    spikes: torch.Tensor,
    cl_ids: NDArray[np.int_],
    model: nn.Module,
    num_epochs: int = 25,
    lr: float = 1e-4,
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
    device, spk_data, train_indices, test_indices, train_loader, test_loader = (
        _create_dataloaders(spikes, cl_ids, batch_size)
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in tqdm(range(num_epochs), desc="Epoch"):
        model.train()
        running_mse = 0

        for spks, _, _ in tqdm(train_loader, desc="Training", leave=False):
            spks = spks.to(device)

            out, z_batch = model(spks)
            loss = torch.nn.functional.mse_loss(out, spks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_mse += loss.item()

        average_mse = running_mse / len(train_loader)

        model.eval()
        running_test_mse = 0
        with torch.no_grad():
            for spks, _, _ in tqdm(test_loader, desc="Testing", leave=False):
                spks = spks.to(device)

                out, _ = model(spks)
                mse = torch.nn.functional.mse_loss(out, spks)
                running_test_mse += mse.item()

        average_test_mse = running_test_mse / len(test_loader)
        if verbose:
            tqdm.write(
                f"Epoch {epoch + 1:2d}/{num_epochs} | Train MSE: {average_mse:.4f} | Test MSE: {average_test_mse:.4f}"
            )

    if return_inds:
        return model, spk_data, test_indices
    else:
        return model, spk_data


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
            # NumPy 2 treats a 1-D torch.Tensor of length 1 as ambiguous with a
            # 0-D scalar when used as a fancy index, yielding
            # ``ValueError: setting an array element with a sequence`` on the
            # last batch whenever ``len(dataset) % batch_size == 1``. Convert
            # the index tensor to numpy up front — cheap, works in every batch
            # size, and decouples from torch/numpy array-api quirks.
            idx_np = idx.cpu().numpy() if isinstance(idx, torch.Tensor) else np.asarray(idx)

            reconstructions, _ = autoencoder(spikes)
            loss += loss_fn(spikes, reconstructions).item()

            batch_latents = autoencoder.encode(spikes)
            spike_latents[idx_np] = batch_latents.cpu().detach().numpy()
            spike_labels[idx_np] = labels.cpu().detach().numpy()

    tqdm.write(f"\nAverage Loss: {loss / len(dataloader):.4f}")

    # Calculate cluster centroids (mean latent representation for each unit)
    unit_centroids = np.zeros((n_units, zDim))
    unit_spreads = np.zeros(n_units)
    for unit_idx in range(n_units):
        unit_mask = spike_labels == unit_idx
        if np.sum(unit_mask) > 0:
            unit_centroids[unit_idx] = np.mean(spike_latents[unit_mask], axis=0)
        if np.sum(unit_mask) > 0:
            diffs = spike_latents[unit_mask] - unit_centroids[unit_idx]
            unit_spreads[unit_idx] = np.mean(np.sqrt(np.sum(diffs**2, axis=1)))

    # Calculate pairwise euclidean distances between unit centroids
    centroid_distances = dist.squareform(dist.pdist(unit_centroids, "euclidean"))

    # Calculate similarity -- ref_dist is scaled to 0.6 similarity
    ref_dists = unit_spreads[:, None] + unit_spreads[None, :]
    autoencoder_similarity = np.exp(-centroid_distances / (2 * ref_dists))

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

            # Penalize by geometric mean of cross-decay
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

            autoencoder_similarity[i, j] *= decay_pen_raw
            autoencoder_similarity[j, i] = autoencoder_similarity[i, j]

    return autoencoder_similarity


def _create_dataloaders(spikes, cl_ids, batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    labels = cl_ids

    train_indices, test_indices, _, _ = train_test_split(
        np.arange(len(spikes)),
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

    train_dataset = SpikeDataset(spikes[train_indices], labels[train_indices])
    test_dataset = SpikeDataset(spikes[test_indices], labels[test_indices])

    sampler = WeightedRandomSampler(
        weights=train_weights, num_samples=len(train_indices), replacement=True
    )
    test_sampler = WeightedRandomSampler(
        weights=test_weights, num_samples=len(test_indices), replacement=True
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler)
    return device, train_dataset, train_indices, test_indices, train_loader, test_loader
