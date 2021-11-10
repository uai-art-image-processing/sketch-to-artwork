# Artwork generation based on Sketches

## Prerequisites

- Linux or macOS
- NVIDIA GPU with at least 12GB VRAM and cuDNN
- Python 3.7 or higher

## Installation

1. On Linux, run `install_requirements.sh`, this will install miniconda with Python 3.
2. Install the conda environment if you haven't before:
    
    ```{python}
    conda env create -f environment.yaml
    conda activate taming
    python -e .
    ```

## Training on custom data

1. Put your .jpg files in a `data` folder.
2. Create 2 text files, a `<dataset>_train.txt` and `<dataset>_test.txt` that point to the files in your training and test set respectively (for example `find $(pwd)/<dataset>/train -name "*.jpg" > <dataset>_train.txt`).
3. Create a new config file based on `configs/custom_vqgan.yaml` to point to these 2 files.
4. Run `python main.py --base configs/<your_config>_vqgan.yaml -t True --gpus 0,1` to train on two GPUs. Use `--gpus 0,` (with a trailing comma) to train on a single GPU.

## Custom Datasets

### Wikiart

1. Run `bash scripts/wikiart.sh` to download and preprocess the dataset
2. Run `python main.py --base configs/wikiart_vqgan.yaml -t True --gpus 0,`
3. Run `python main.py --base configs/wikiart_edges_vqgan.yaml -t True --gpus 0,`
4. Run `python main.py --base configs/wikiart_edges_transformer.yaml -t True --gpus 0,`

## Continue training

1. (Optional) Update the config file
2. (Optional) Run `python -m pytorch_lightning.utilities.upgrade_checkpoint --file logs/<model>/checkpoints/last.ckpt`
2. Run `python main.py -t True --gpus <gpus> --resume logs/<model>` and the training proccess should be started :

## For Fine-tuning

2. Create directories `./logs/<some name>/configs` and `./logs/<some name>/checkpoints`
3. Copy `last.ckpt` file into newly created checkpoints directory
4. Rename copied `model.yaml` file into `<some name>-project.yaml` and put it into configs directory
5. (Optional) Add these lines to the end of `<some name>-project.yaml` file. Don't forget to adapt some values like you did when training a model from scratch
6. (Optional) Run python -m pytorch_lightning.utilities.upgrade_checkpoint --file logs/<some name>/checkpoints/last.ckpt
7. Run python main.py -t True --gpus <gpus> --resume logs/<some name> and the training proccess should be started :