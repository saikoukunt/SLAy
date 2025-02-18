# Spike-sorting Lapse Amelioration System (SLAy)

## Running SLAy


## Installation Instructions
To fully automate your spike sorting pipeline and avoid switching environments between spike sorting and postprocessing, we recommend using SLAy with the CILANtro pipeline (installation instructions [here](https://github.com/NP-HarrisLab/CILANtro)). If you want to run SLAy as a standalone tool, use the installation instructions below.

1. Create a new conda environment
   ```bash
   conda create -n slay python=3.10
   conda activate slay
   ```
2. Install packages
   ```bash
    conda install pytorch pytorch-cuda=12.4 -c pytorch -c nvidia
    conda install scipy argparse numpy pandas scikit-learn matplotlib
    pip install cupy-cuda12x marshmallow
    ```
3. Install [npx_utils]()
   ```bash
   *navigate to desired folder*
   git clone https://github.com/NP-HarrisLab/npx_utils.git
   cd npx_utils
   pip install -e .
   ```
4. Install SLAy
   ```bash
   *navigate to desired folder*
   git clone https://github.com/saikoukunt/SLAy.git
   cd slay
   pip install -e .
   ```



