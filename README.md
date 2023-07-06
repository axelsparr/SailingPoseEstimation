## Background
The following repository contains Axels work from my summer internship 2023. A detailed log of my progress and steps taken/decisions can be found here: https://docs.google.com/document/d/1GLqgyrLra6IkQDmGH3Nt_NHsXbvvfJEm1sujFL1-Qis/edit#heading=h.1a9j6dmma3eq

## Important files
The most important files are:
- `PCT/clean.ipynb` contains the usage/experimentation with the model api
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
install the packages, note it will work with building the wheel for mmcv-full for a long times, 20+ min. This is normal and seems to be a known issue with mmcv but we cant avoid it, only effects installation time though.
```bash
python -m pip install --upgrade pip
pip uninstall torch torchvision torchaudio
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

dont forget to add the videos to the videos folder.


