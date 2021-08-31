# Arwork generation based on Sketches

## Prerequisites

- Linux or macOS
- GPU with at least 12GB VRAM
- Python 3.7 or higher

## Installation

1. On Linux, run `install_requirements.sh`, this will install miniconda with python 3.7 and create a new conda environmet (taming) for this repo.

## Training on custom data

These steps assume you have installed the repo dependencies with `install_requirements.sh`.

1. Put your .jpg files in a `<dataset>` folder.
2. Create 2 text files, a `<dataset>_train.txt` and `<dataset>_test.txt` that point to the files in your training and test set respectively (for example find `$(pwd)/<dataset>/train -name "*.jpg" > <dataser>_train.txt`).
3. Create a new config file based on `configs/custom_vqgan.yaml` to point to these 2 files.
4. Run `python main.py --base configs/custom_vqgan.yaml -t True --gpus 0,1` to train on two GPUs. Use `--gpus 0,` (with a trailing comma) to train on a single GPU.
