import logging
import os
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

import burst_detector as bd

logger = logging.getLogger("burst-detector")


def generate_train_data(
    data: NDArray[np.int_],
    ci: dict[str, Any],
    channel_pos: NDArray[np.float_],
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
            clusters (NDArray): Spike cluster assignments.
            counts (dict): Spike counts per cluster.
            labels (pd.DataFrame): Cluster quality labels.
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
    # Load existing spike snippets
    spikes_path = os.path.join(params["KS_folder"], "automerge", "spikes.pt")
    if os.path.exists(spikes_path):
        logger.info("Loading existing spike snippets...")
        torch_data = torch.load(spikes_path)
        return torch_data["spikes"], torch_data["cl_ids"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Pre-compute the set of closest channels for each channel.
    good_ids = np.argwhere(ci["good_ids"]).flatten()
    chans = {}
    for id in good_ids:
        chs, peak = bd.find_best_channels(
            ci["mean_wf"][id], channel_pos, params["n_chan"], ext_params["num_chan"]
        )
        dists = bd.get_dists(channel_pos, peak, chs)
        chans[id] = chs[np.argsort(dists)].tolist()

    if ext_params["for_shft"]:
        ext_params["pre_samples"] += 5
        ext_params["post_samples"] += 5

    # Pre-allocate memory for the snippets for good cluster
    if params["max_spikes"] == -1:
        n_snip = np.sum(ci["counts"][good_ids])
    else:
        n_snip = np.sum(np.minimum(ci["counts"][good_ids], params["max_spikes"]))

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
    for id in tqdm(good_ids, desc="Generating snippets"):
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

    # Save the snippets to disk
    torch.save({"spikes": spikes, "cl_ids": cl_ids}, spikes_path)
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
        return spk, label


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

        # Encoder layers
        self.encConv1 = nn.Conv2d(
            imgChannels, self.qrt_filt, kernel_size=3, padding="same"
        )
        self.encBN1 = nn.BatchNorm2d(self.qrt_filt)
        self.encPool1 = nn.MaxPool2d(kernel_size=2)

        self.encConv2 = nn.Conv2d(
            self.qrt_filt, self.half_filt, kernel_size=3, padding="same"
        )
        self.encBN2 = nn.BatchNorm2d(self.half_filt)
        self.encPool2 = nn.MaxPool2d(kernel_size=2)

        self.encConv3 = nn.Conv2d(self.half_filt, n_filt, kernel_size=3, padding="same")
        self.encBN3 = nn.BatchNorm2d(n_filt)
        self.encPool3 = nn.MaxPool2d(kernel_size=2)

        self.encFC1 = nn.Linear(self.featureDim, zDim)
        self.decFC1 = nn.Linear(zDim, self.featureDim)
        self.decBN1 = nn.BatchNorm2d(n_filt)

        self.decUpSamp1 = nn.Upsample(size=(num_chan // 4, num_samp // 4))
        self.decConv1 = nn.ConvTranspose2d(
            n_filt, self.half_filt, kernel_size=3, padding=1
        )
        self.decBN2 = nn.BatchNorm2d(self.half_filt)

        self.decUpSamp2 = nn.Upsample(size=(num_chan // 2, num_samp // 2))
        self.decConv2 = nn.ConvTranspose2d(
            self.half_filt, self.qrt_filt, kernel_size=3, padding=1
        )
        self.decBN3 = nn.BatchNorm2d(self.qrt_filt)

        self.decUpSamp3 = nn.Upsample(size=(num_chan, num_samp))
        self.decConv3 = nn.ConvTranspose2d(
            self.qrt_filt, imgChannels, kernel_size=3, padding=1
        )

    def encoder(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.encBN1(self.encConv1(x)))
        x = self.encPool1(x)

        x = F.relu(self.encBN2(self.encConv2(x)))
        x = self.encPool2(x)

        x = F.relu(self.encBN3(self.encConv3(x)))
        x = self.encPool3(x)

        x = x.view(-1, self.featureDim)
        x = self.encFC1(x)

        return x

    def decoder(self, z: torch.Tensor) -> torch.Tensor:
        x: torch.Tensor = self.decFC1(z)
        x = x.view(-1, self.n_filt, int(self.num_chan / 8), int(self.num_samp / 8))
        x = F.relu(self.decBN1(x))

        x = self.decUpSamp1(x)
        x = F.relu(self.decBN2(self.decConv1(x)))

        x = self.decUpSamp2(x)
        x = F.relu(self.decBN3(self.decConv2(x)))

        x = self.decUpSamp3(x)
        x = self.decConv3(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z: torch.Tensor = self.encoder(x)
        out: torch.Tensor = self.decoder(z)

        return out


def _apply_time_shift(
    spks: torch.Tensor, pre_samples: int, post_samples: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies random time shifts to a subset of spike snippets for data augmentation.

    Args:
        spks (torch.Tensor): Original spike snippets.
        pre_samples (int): Number of pre-samples before spike time.
        post_samples (int): Number of post-samples after spike time.

    Returns:
        spks (torch.Tensor): Time-shifted spike snippets.
        targ (torch.Tensor): Target snippets (unshifted).
    """
    B = spks.shape[0]
    targ = spks[:, :, :, 5:-5].clone()
    shft_spks = targ.clone()
    shft_ind = np.random.choice(B, int(B * 0.3), replace=False)

    for ind in shft_ind:
        shifts = np.arange(-5, 6)
        shifts = shifts[shifts != 0]
        shift = np.random.choice(shifts, 1)[0]
        shft_spks[ind] = spks[
            ind, :, :, 5 + shift : pre_samples + post_samples + 5 + shift
        ].clone()

    return shft_spks, targ


def train_ae(
    spikes: torch.Tensor,
    cl_ids: NDArray[np.int_],
    n_filt: int = 256,
    num_epochs: int = 25,
    zDim: int = 15,
    lr: float = 1e-3,
    pre_samples: int = 10,
    post_samples: int = 30,
    do_shft: bool = False,
    model: CN_AE = None,
) -> tuple[CN_AE, SpikeDataset]:
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
        range(len(spk_data)), labels, stratify=labels, test_size=0.2, random_state=42
    )
    train_split = Subset(spk_data, train_indices)
    test_split = Subset(spk_data, test_indices)

    BATCH_SIZE = 128
    train_loader = DataLoader(train_split, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_split, batch_size=BATCH_SIZE)

    net = model if model else CN_AE(zDim=zDim, n_filt=n_filt).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # TRAIN/TEST LOOP
    for epoch in tqdm(range(num_epochs), desc="Epoch"):
        net.train()
        running_loss = 0

        # TRAINING ITERATION
        for spks, _ in tqdm(train_loader, desc="Training", leave=False):
            spks = spks.to(device)

            # If explicitly training time-shift invariance, we randomly shift 30% of the
            # input snippets.
            if do_shft:
                spks, targ = _apply_time_shift(spks, pre_samples, post_samples)
            else:
                targ = spks.clone()

            out: torch.Tensor = net(spks)
            loss: torch.Tensor = loss_fn(out, targ)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        # TESTING ITERATION
        net.eval()
        running_tloss = 0
        with torch.no_grad():
            for spks, _ in tqdm(test_loader, desc="Testing", leave=False):
                spks = spks.to(device)
                # If explicitly training time-shift invariance, we randomly shift 30% of
                # the input snippets.
                if do_shft:
                    spks, targ = _apply_time_shift(spks, pre_samples, post_samples)
                else:
                    targ = spks.clone()

                out = net(spks)
                tloss: torch.Tensor = loss_fn(out, targ)
                running_tloss += tloss.item()

        avg_test_loss = running_tloss / len(test_loader)
        tqdm.write(
            f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f}"
        )
    return net, spk_data
