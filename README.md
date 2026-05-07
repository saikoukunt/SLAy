# SLAy oversplitting errors in your spike sorting!

Manually merging spike sorted units is time-consuming, subjective, and lacks reproducibility. SLAy uses quantitative metrics to automatically identify oversplit units, which can then be merged automatically or reviewed further manually. Specifically, we developed novel metrics for **(1)** calculating waveform similarity using an autoencoder, and **(2)** capturing structure in cross-correlograms. When used with automatic unit quality labelling (e.g., via [SpikeInterface](https://github.com/SpikeInterface/spikeinterface) or [BombCell](https://github.com/Julie-Fabre/Bombcell)), SLAy can fully automate the manual curation process, making it feasible to quickly process many single- or multi- probe ephys recordings. Read our [paper](https://www.biorxiv.org/content/10.1101/2025.06.20.660590v1) to learn more about SLAy.

- [Installation](#installation)
  - [uv (recommended)](#uv-recommended)
  - [pip + conda (slower)](#pip--conda-slower)
- [(Prerequisite) SpikeInterface SortingAnalyzer](#prerequisite-spikeinterface-sortinganalyzer)
  - [Create a SortingAnalyzer](#create-a-sortinganalyzer)
  - [Compute extensions](#compute-extensions)
  - [Remove noise and multi-units](#remove-noise-and-multi-units)
- [Using SLAy](#using-slay)
  - [(default) with autoencoder and automatic parameter selection](#default-with-autoencoder-and-automatic-parameter-selection)
  - [with manual parameter selection](#with-manual-parameter-selection)
  - [with L2 waveform similarity](#with-l2-waveform-similarity)
  - [directly through SpikeInterface](#directly-through-spikeinterface)
- [Questions/Issues](#questionsissues)
- [Citing](#citing)


## Installation
We recommend using [uv](https://docs.astral.sh/uv/) to install SLAy. If you want to use your GPU to train SLAy's autoencoder, which is recommended, you will also need CUDA (if on an NVIDIA GPU) and the GPU version of pytorch. The instructions below show how to install pytorch for CUDA 12.6, but you should modify it to match your CUDA version.

### uv (recommended)
0. Install CUDA if you don't already have it.
1. Install pytorch for GPU following the instructions [here](https://docs.astral.sh/uv/guides/integration/pytorch/#using-a-pytorch-index).
1. Install SLAy: 
    ```shell
    uv add slay
    ```

### pip + conda (slower)
0. Install CUDA if you don't already have it.
1. Create a new conda environment.
    ```shell
    conda create -n slay python=3.11
    conda activate slay

    # optional but highly recommended, for GPU autoencoder training
    pip install torch --index-url https://download.pytorch.org/whl/cu126
    ```

2. Install SLAy.

    ``` shell
    pip install slay
    ```


## (Prerequisite) SpikeInterface SortingAnalyzer

### Create a SortingAnalyzer
To ensure compatibility with any recording hardware and spike sorter, SLAy operates on SpikeInterface's `SortingAnalyzer`. If your pipeline already uses SpikeInterface for preprocessing and sorting, you can create a `SortingAnalyzer` like this:

```python
sorting_analyzer = si.create_sorting_analyzer(
    sorting, 
    recording_preprocessed,
    format="binary_folder",         # change if desired
    folder="/my_sorting_analyzer",
    **job_kwargs
)
```

Otherwise, you can use SpikeInterface's `read_XXX` functions (see a [full list](https://spikeinterface.readthedocs.io/en/stable/modules/extractors.html#compatible-formats)) to load in your *pre-processed* recording and spike sorter output. For example, to load in a SpikeGLX recording processed with CatGT and sorted with Kilosort:

```python
import spikeinterface.full as si

recording_preprocessed = si.read_spikeglx(
    recording_folder,
    stream_id="imec0.ap"    
)
sorting = si.read_phy(phy_folder, exclude_cluster_groups=["noise", "mua"])

sorting_analyzer = si.create_sorting_analyzer(
    sorting, 
    recording_preprocessed,
    format="binary_folder",         # change if desired
    folder="/my_sorting_analyzer",
    **job_kwargs
)
```

### Compute extensions
SLAy's metrics rely on your `SortingAnalyzer` to access spike waveforms and cross-correlograms. SLAy will use your `SortingAnalyzer` to compute these automatically if they are not already present, but we recommend precomputing them for use in other processing steps and so SLAy runs faster:

```python
job_kwargs = dict(n_jobs=-1, progress_bar=True, chunk_duration="1s")

sorting_analyzer.compute("random_spikes", method="uniform", max_spikes_per_unit=500)
sorting_analyzer.compute("waveforms", **job_kwargs)
sorting_analyzer.compute("templates", **job_kwargs)

sorting_analyzer.compute("correlograms", window_ms=100, bin_ms=1.) # change to match your brain region/animal model, we recommend <= 2 ms and >= 0.5 ms

sorting_analyzer.compute("quality_metrics", metric_names=["snr", "firing_rate"])
```

You should set the cross-correlogram parameters based on the standard range for your brain region(s) and animal model.

### Remove noise and multi-units
To avoid extra computations, you should give SLAy a `SortingAnalyzer` that does not contain noise or multi-units. This can be done with `si.select_units()`.


## Using SLAy

### (default) with autoencoder and automatic parameter selection
You do not need a pre-trained autoencoder to run SLAy. To run SLAy with autoencoder-based similarity and automatic parameter selection:

```python
from slay import compute_slay_merges

merges, sorting_analyzer, slay_metrics = compute_slay_merges(
    sorting_analyzer,
    model_path="save/path/for/autoencoder.pt",
)
```
This will extract spike snippets from your recording, train and save the autoencoder, compute the SLAy merge metrics, and output a list of suggested merges. By default, SLAy will automatically select the appropriate parameters (coefficients and merge threshold) for your recording. You can use SpikeInterface to merge the suggestions:

```python
merged_sorting_analyzer = sorting_analyzer.merge_units(
    merges,
    censor_ms=5/30000,      # remove duplicate spikes
    format="binary_folder",
    folder="/my_merged_sorting_analyzer"
)
```

or review them manually in [SpikeInterface-GUI](https://github.com/SpikeInterface/spikeinterface-gui) or [Phy](https://github.com/cortex-lab/phy).

> Note: If you have a pretrained model, you can point `model_path` to the `.pt` file and SLAy will automatically use it! SLAy's autoencoder usually trains in under 10 minutes, but if you are processing many recordings from the same animal with the same site map, you can save time by reusing the same autoencoder.


### with manual parameter selection

If you want manual control over SLAy's parameters,  you can specify custom parameters through the `merge_parameters` argument. You must specify a dictionary with values for "k1" (coefficient for CCG structure metric), "k2" (coefficient for refractory period penalty), and "merge_threshold". Below is a reasonable starting point for manual adjustment:

```python
merges, sorting_analyzer, slay_metrics = compute_slay_merges(
    sorting_analyzer,
    merge_parameters={"k1": 0.25, "k2": 1, "merge_threshold": 0.5},
    model_path="save/path/for/autoencoder.pt",
)
```

### with L2 waveform similarity
If you don't want to use autoencoder-based waveform similarity (e.g., if you don't have a GPU and CPU-training would be too slow), you can run SLAy with a simpler measure of waveform similarity:

```python
merges, sorting_analyzer, slay_metrics = compute_slay_merges(
    sorting_analyzer,
    model_path="save/path/for/autoencoder.pt",
    similarity_type="l2"
)
```

We found that the L2 similarity performs similarly to the autoencoder for some recordings, but has false positives and negatives for others (see [paper](https://www.biorxiv.org/content/10.1101/2025.06.20.660590v1) to learn more about SLAy).

### directly through SpikeInterface
If you do not want to use autoencoder-based waveform similarity or automatic parameter selection, you can run an L2-similarity version of SLAy directly through SpikeInterface:

```python
from spikeinterface.curation import compute_merge_unit_groups

merges = compute_merge_unit_groups(
    sorting_analyzer, 
    preset="slay",
    steps_params={"slay_score": {"k1": 0.25, "k2": 1.0, "slay_threshold": 0.5}},
    resolve_graph=True      # find multi-way merges (>2 units)
)
merged_sorting_analyzer = sorting_analyzer.merge_units(
    merges,
    censor_ms=5/30000,      # remove duplicate spikes
    format="binary_folder",
    folder="/my_merged_sorting_analyzer"
)
```

<!-- ### Advanced

#### Train an autoencoder with spikes from multiple recordings
Under construction!


#### Use a custom autoencoder architecture or training function
Under construction! -->

## Questions/Issues
This codebase is in beta --- if you have questions or run into any errors, open a GitHub issue or [shoot me an email](mailto:sai.koukunt@gmail.com)!

## Citing
If you use SLAy in your own work, please cite our paper!

> S. Koukuntla, T. Deweese, A. Cheng, R. Mildren, A. Lawrence, A. Graves, K.E. Cullen, J. Colonell, T.D. Harris, and A.S. Charles. SLAy-ing errors in high-density electrophysiology spike sorting. *bioRxiv*, doi: 10.1101/2025.06.20.660590., 2025 [https://www.biorxiv.org/content/10.1101/2025.06.20.660590v1](https://www.biorxiv.org/content/10.1101/2025.06.20.660590v1).

