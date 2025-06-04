from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray
from scipy.sparse import lil_array
from scipy.sparse.csgraph import dijkstra
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from tqdm import tqdm

import slay


def generate_train_data(
    data: NDArray[np.int_],
    ci: dict[str, Any],
    channel_pos: NDArray[np.float64],
    ext_params: dict[str, Any],
    params: dict[str, Any],
) -> tuple[torch.Tensor, NDArray[np.int_]]:
    """
    Generates a dataset of spike snippets from an ephys recording that can be used
    to train an convolutional autoencoder that learns waveform shape features.

    The first channel in the snippet is the channel with peak amplitude, and the
    remaining channels are ordered by distance from the peak channel. This removes
    info about the absolute position of the spikes, so the autoencoder can focus only
    on its shape. To reduce training time, we cap the number of snippets that
    are generated per cluster.

    Args:
        data (NDArray): ephys data with shape (# of timepoints, # of channels).
            Should be passed in as an np.memmap for large datasets.
        ci (dict): Cluster information --
            times_multi (list): Spike times indexed by cluster id.
            counts (dict): Spike counts per cluster.
            good_ids (NDArray): IDs of clusters that passed quality and min_spikes threshold.
            mean_wf (NDArray): Cluster mean waveforms with shape
                (# of clusters, # channels, # timepoints).
        channel_pos (NDArray): XY coordinates of each channel on the probe.
        ext_params (dict): Snippet extraction parameters --
            pre_samples (int): Number of samples to include before spike time.
            post_samples (int): Number of samples to after before spike time.
            num_chan (int): Number of channels to include in spike waveform.
            for_shft (bool): True if dataset will be used to train a shift-invariant autoencoder.
       params (dict): General SpECtr params
    Returns:
        spikes (torch.Tensor): Extracted spike snippets with shape
            (# snippets, 1, # channels, # timepoints). This Tensor lives on the GPU
            if available.
        cl_ids (NDArray): Cluster labels of spike snippets with shape (# snippets)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Pre-compute the set of closest channels for each channel.
    chans = {}
    for id in ci["good_ids"]:
        chs, peak = slay.find_best_channels(
            ci["mean_wf"][id], channel_pos, params["n_chan"], ext_params["num_chan"]
        )
        dists = slay.get_dists(channel_pos, peak, chs)
        chans[id] = chs[np.argsort(dists)].tolist()

    if ext_params["for_shft"]:
        ext_params["pre_samples"] += 5
        ext_params["post_samples"] += 5

    # Pre-allocate memory for the snippets for good clusters.
    if params["max_spikes"] == -1:
        n_snip = np.sum(ci["counts"][ci["good_ids"]])
    else:
        n_snip = np.sum(np.minimum(ci["counts"][ci["good_ids"]], params["max_spikes"]))

    spikes = torch.zeros(
        (
            n_snip,
            1,
            ext_params["num_chan"],
            ext_params["pre_samples"] + ext_params["post_samples"],
        ),
        device=device,
    )
    cl_ids = np.zeros(n_snip, dtype="int16")

    snip_idx = 0
    for id in tqdm(ci["good_ids"], desc="Generating snippets"):
        cl_times = ci["times_multi"][id].astype("int64")

        if cl_times.shape[0] > params["max_spikes"]:
            np.random.shuffle(cl_times)
            cl_times = cl_times[: params["max_spikes"]]

        start_times = cl_times - ext_params["pre_samples"]
        end_times = cl_times + ext_params["post_samples"]
        for j, (start, end) in enumerate(zip(start_times, end_times)):
            cl_ids[snip_idx + j] = id
            spike = np.nan_to_num(data[start:end, chans[id]].T)
            spikes[snip_idx + j] = torch.Tensor(spike).unsqueeze(dim=0)
        snip_idx += cl_times.shape[0]

    return spikes, cl_ids


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
        transform: Callable = None,
        target_transform: Callable = None,
    ) -> None:
        self.spikes: torch.Tensor = spikes
        self.labels: NDArray[np.int_] = labels
        self.transform: Callable = transform
        self.target_transform: Callable = target_transform

    def __len__(self) -> int:
        return self.spikes.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
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

        # encoder
        self.encoder_conv = nn.Sequential()
        self.encoder_conv.append(
            nn.Conv2d(imgChannels, self.qrt_filt, kernel_size=4, padding=1, stride=2)
        )
        self.encoder_conv.append(nn.ReLU())
        self.encoder_conv.append(
            nn.Conv2d(self.qrt_filt, self.half_filt, kernel_size=4, padding=1, stride=2)
        )
        self.encoder_conv.append(nn.ReLU())
        self.encoder_conv.append(
            nn.Conv2d(self.half_filt, n_filt, kernel_size=4, padding=1, stride=2)
        )

        self.encoder_fc = nn.Sequential()
        self.encoder_fc.append(nn.Linear(self.featureDim, zDim))
        self.encoder_fc.append(nn.ReLU())

        # decoder
        self.decoder_fc = nn.Sequential()
        self.decoder_fc.append(nn.Linear(zDim, self.featureDim // 2))
        self.decoder_fc.append(nn.ReLU())
        self.decoder_fc.append(nn.Linear(self.featureDim // 2, self.featureDim))

        self.decoder_conv = nn.Sequential()
        self.decoder_conv.append(
            nn.ConvTranspose2d(
                n_filt,
                self.half_filt,
                kernel_size=4,
                padding=1,
                stride=2,
            )
        )
        self.decoder_conv.append(nn.ReLU())
        self.decoder_conv.append(
            nn.ConvTranspose2d(
                self.half_filt,
                self.qrt_filt,
                kernel_size=4,
                padding=1,
                stride=2,
            )
        )
        self.decoder_conv.append(nn.ReLU())
        self.decoder_conv.append(
            nn.ConvTranspose2d(
                self.qrt_filt,
                imgChannels,
                kernel_size=4,
                padding=1,
                stride=2,
            )
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder_conv(x)
        x = x.view(-1, self.featureDim)
        x = self.encoder_fc(x)

        return x

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x: torch.Tensor = self.decoder_fc(z)
        x = x.view(-1, self.n_filt, int(self.num_chan / 8), int(self.num_samp / 8))
        x = self.decoder_conv(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z: torch.Tensor = self.encode(x)
        out: torch.Tensor = self.decode(z)

        return out, z


def train_ae(
    spikes: torch.Tensor,
    cl_ids: NDArray[np.int_],
    counts,
    n_filt: int = 256,
    num_epochs: int = 25,
    zDim: int = 15,
    lr: float = 1e-3,
    model: CN_AE = None,
    batch_size: int = 128,
    max_snips: int = 500,
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

    # Initialize the dataset and dataloaders
    spk_data = SpikeDataset(spikes, cl_ids)
    labels = cl_ids

    train_indices, test_indices, _, _ = train_test_split(
        np.arange(len(spk_data)),
        labels,
        stratify=labels,
        test_size=0.2,
        random_state=42,
    )
    sample_weights = [
        1 / max(counts[int(label)], max_snips) for label in labels[train_indices]
    ]
    train_split = Subset(spk_data, train_indices)
    test_split = Subset(spk_data, test_indices)
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(train_split), replacement=True
    )

    BATCH_SIZE = batch_size
    train_loader = DataLoader(train_split, batch_size=BATCH_SIZE, sampler=sampler)
    test_loader = DataLoader(test_split, batch_size=BATCH_SIZE)

    net = model if model else CN_AE(zDim=zDim, n_filt=n_filt).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    # TRAIN/TEST LOOP
    for epoch in tqdm(range(num_epochs), desc="Epoch"):
        net.train()
        running_mse = 0

        # TRAINING ITERATION
        for spks, _, idx in tqdm(train_loader, desc="Training", leave=False):
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
