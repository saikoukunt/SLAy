# Spike-sorting Lapse Amelioration System (SLAy) <!-- omit from toc -->
- [Running SLAy](#running-slay)
  - [Setting default parameters](#setting-default-parameters)
  - [Option 1: Run from a python script (recommended)](#option-1-run-from-a-python-script-recommended)
  - [Option 2: Run from the command line](#option-2-run-from-the-command-line)
- [Installation Instructions](#installation-instructions)
- [Citing](#citing)



## Running SLAy

### Setting default parameters
[`slay/schemas.py`](./slay/schemas.py) contains the names and descriptions of all parameters, and their default values can be changed there (or parameters may be passed in when calling the `run_slay` function). The default parameters work pretty well across a variety of brain regions and animal models, but you can adjust the defaults for your own preferences. Some parameters you may want to change are:

- `'auto_accept_merges'` : If True, all the suggested merges will be automatically accepted and the KS files (`spike_clusters.npy`) will be overwritten. If False, the merges will be plotted in the `{ks_dir}/automerge/merges` folder. **Defaults to False**.

- `'good_lbls'`: The quality labels in `cluster_group.tsv` of the clusters that should be considered for merging. **Defaults to `["good"]`.**

- `'max_viol'`: Acceptable cross-correlogram (CCG) refractory period violations as a proportion of baseline CCG event rate. **Defaults to 0.15.**

- `'xcorr_coeff'`: Coefficient applied to cross-correlation metric for final metric calculation. Decrease this if you want merging to rely more on waveform similarity and less on CCG shape. **Defaults to 0.25.**

- `'ref_pen_coeff'`: Coefficient applied to refractory period violation penalty for final metric calculation. Decrease this if you want merging to be more lenient on cluster pairs with refractory period violations. **Defaults to 1.**

- `'final_thresh'`: Threshold on the final metric (between 0 and 1) to decide whether to merge a cluster pair. **The default value of 0.5** works well, but increase (decrease) it if you want less (more) merge suggestions. Would highly recommend turning off `auto_accept_merges` if you decrease the final threshold.

If you don't want to edit the default parameters, you can also pass in parameters for each recording you run, either through a python dictionary or the command line (see below).


### Option 1: Run from a python script (recommended)

To run SLAy from a python script, you can call `slay.run.main()` and pass in a dictionary containing the `'KS_folder'` (this is the folder containing params.py for Phy), and any non-default parameters you want to set.\
For example, in the snippet below from [`scripts/auto_slay.py`](./scripts/auto_slay.py), we set the `plot_merges`, `max_spikes`, and `auto_accept_merges` parameters and run SLAy on all KS4-sorted recordings in the `data_dir` folder:

```python
import os

import slay

if __name__ == "__main__":
    merge_params = {
        "plot_merges": True,
        "max_spikes": 500,
        "auto_accept_merges": False,
    }

    data_dir = "D:/SLAY_data/"

    # Automatically finds all KS 4 folders in the data directory
    for root, dirs, files in os.walk(data_dir):
        for dir_name in dirs:
            if "ks4" in dir_name:
                ks_dir = os.path.join(root, dir_name)
                print(ks_dir)
                slay.run_slay({"KS_folder": ks_dir, **merge_params})
```

### Option 2: Run from the command line

You can also run SLAy from the command line by passing in the path to the KS_folder with the flag `--KS_folder`, and any non-default parameters with the flag `--{parameter name}`. For example,

```bash
python slay/run.py --KS_folder path/to/ks/folder --max_spikes 500
```

## Installation Instructions
If you want to run SLAy as a standalone tool, use the installation instructions below.

1. Create a new conda environment
   ```bash
   conda create -n slay python=3.10
   conda activate slay
   ```
1. Install packages
   ```bash
    conda install pytorch pytorch-cuda=12.4 -c pytorch -c nvidia
    conda install scipy argparse numpy pandas scikit-learn matplotlib
    pip install cupy-cuda12x marshmallow
    ```
1. Install SLAy
   ```bash
   *navigate to desired folder*
   git clone https://github.com/saikoukunt/SLAy.git
   cd slay
   pip install -e .
   ```

## Citing
If you use SLAy in your own work, please cite our paper!

> S. Koukuntla, T. Deweese, A. Cheng, R. Mildren, A. Lawrence, A. Graves, K.E. Cullen, J. Colonell, T.D. Harris, and A.S. Charles. SLAy-ing errors in high-density electrophysiology spike sorting. *bioRxiv*, doi: 10.1101/2025.06.20.660590., 2025 [https://www.biorxiv.org/content/10.1101/2025.06.20.660590v1](https://www.biorxiv.org/content/10.1101/2025.06.20.660590v1).




