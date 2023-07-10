## Background
The following repository contains Axels work from my summer internship 2023. A detailed log of my progress and steps taken/decisions can be found here: https://docs.google.com/document/d/1GLqgyrLra6IkQDmGH3Nt_NHsXbvvfJEm1sujFL1-Qis/edit#heading=h.1a9j6dmma3eq
## Environment
The code requires Python `3.8.x` and CUDA `11.1` to work. The superresolution i.e. RVRT doesnt run on Windows without major rewrites. Windows should work for everything else after possibly changing some of the paths in the code. Everything works on Linux.
## Important files
The most important files are:
- `PCT/clean.ipynb` contains example usage of the model api
- `PCT/model_api.py` the "backend" code

## Setup instructions
clone this repo (make sure you have signed in / added ssh to your terminal so you can clone it https://github.com/settings/ssh/new)
```bash
git clone https://github.com/naynasa/SailingPoseEstimation
cd SailingPoseEstimation
```
create the conda environment, `PCT_linux` and install cudatoolkit in the env
```bash
conda create -n PCT_linux python=3.8
conda activate PCT_linux
conda config --add channels conda-forge
conda install cudatoolkit=11.1
```
install the packages, note it will work with building the wheel for mmcv-full for a long times, 20+ min. This is normal and seems to be a known issue with mmcv but we cant avoid it, only affects installation time though.
```bash
python -m pip install --upgrade pip
pip uninstall torch torchvision torchaudio
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

Dont forget to add the videos to the videos folder before running the code.


