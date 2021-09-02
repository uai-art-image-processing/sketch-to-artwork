# Artwork generation based on Sketches

## Prerequisites

- Linux or macOS
- NVIDIA GPU with at least 12GB VRAM
- Python 3.7 or higher

## Installation

1. On Linux, run `install_requirements.sh`, this will install miniconda with Python 3.

## Training on custom data

1. Install the conda environment if you haven't before:
```{python}
conda env create -f environment.yaml
conda activate taming
python -e .
```
2. Put your .jpg files in a `<dataset>` folder.
3. Create 2 text files, a `<dataset>_train.txt` and `<dataset>_test.txt` that point to the files in your training and test set respectively (for example find `$(pwd)/<dataset>/train -name "*.jpg" > <dataser>_train.txt`).
4. Create a new config file based on `configs/custom_vqgan.yaml` to point to these 2 files.
5. Run `python main.py --base configs/<your_config>_vqgan.yaml -t True --gpus 0,1` to train on two GPUs. Use `--gpus 0,` (with a trailing comma) to train on a single GPU.
