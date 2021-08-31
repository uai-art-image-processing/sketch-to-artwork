#!/bin/bash
# Simple script to install miniconda and its dependencies. Needed for VQ-GAN.

# Install miniconda 3.7
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-py37_4.10.3-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash

# Install required dependencies
source ~/.bashrc
conda env create -f environment.yaml
conda activate taming
python -e .