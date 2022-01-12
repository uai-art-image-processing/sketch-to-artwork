# Artwork generation based on Sketches

Implementation of VQ-GAN architecture and transformers for Image-to-Image Translation (Edge-to-Artwork)

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
    ```

## Pretrained Models

Coming Soon

## Training

For Edge-to-Image Translation, we need first to train two VQ-GANs: one VQ-GAN for the edge images and another for the actual images/artworks. After training both autoencoders, we train a transformer usin both pretrained models checkpoints. The edge extraction in this implementation occurs mid training so there is no need for an edge dataset (See `taming/data/wikiart.py`).


The training  process should look like this:

1. Run `bash scripts/wikiart.sh` to get and preprocess the Wikiart dataset
2. Run `python main.py --base configs/edge_vqgan.yaml -t True --gpus <gpus>` to train the Edge Autoencoder
3. Run `python main.py --base configs/wikiart_vqgan.yaml -t True --gpus <gpus>` to train the Artwork Autoencoder
4. Create a config file for the transformer that point to both checkpoints. The parameters of the first and conditional stage should match with the original config files (see `configs/transformer.yaml`, `configs/wikiart_vqgan.yaml` and `configs/edge_vqgan.yaml`). It should look something like this:

```
model:
  base_learning_rate: 4.5e-06
  target: taming.models.cond_transformer.Net2NetTransformer
  params:
    cond_stage_key: lr
    transformer_config:
      target: taming.modules.transformer.mingpt.GPT
      ...
    first_stage_config:
      target: taming.models.vqgan.VQModel
      params:
        ckpt_path: # logs/<wikiart_vqgan>/checkpoints/last.ckpt
        embed_dim: 256
        n_embed: 1024
        image_key: image
        ... 
      lossconfig:
          target: taming.modules.losses.DummyLoss
    cond_stage_config:
      target: taming.models.vqgan.VQModel
      params:
        ckpt_path: # logs/<edge_vqgan>/checkpoints/last.ckpt
        embed_dim: 256
        n_embed: 1024
        ...
      lossconfig:
          target: taming.modules.losses.DummyLoss
          ...
```
5. Run `python main.py --base configs/transformer.yaml -t True --gpus <gpus>` to train the Transformer

Replace `<gpus>` with ` 0,` (with a trailing comma) to train on a single GPU. `0,1` for two GPUs and so on.

## Continue training

1. (Optional) Update the config file
2. (Optional) Run `python -m pytorch_lightning.utilities.upgrade_checkpoint --file logs/<model>/checkpoints/last.ckpt` to update checkpoint
2. Run `python main.py -t True --gpus <gpus> --resume logs/<model>` and the training proccess should be started

## Fine-tuning

2. Create directories `./logs/<some name>/configs` and `./logs/<some name>/checkpoints`
3. Copy `last.ckpt` file into newly created checkpoints directory
4. Rename copied `model.yaml` file into `<some name>-project.yaml` and put it into configs directory
5. (Optional) Add these lines to the end of `<some name>-project.yaml` file. Don't forget to adapt some values like you did when training a model from scratch
6. (Optional) `Run python -m pytorch_lightning.utilities.upgrade_checkpoint --file logs/<some name>/checkpoints/last.ckpt` to update checkpoint
7. Run `python main.py -t True --gpus <gpus> --resume logs/<some name>` and the training proccess should be started

## Custom Datasets

1. Put your `.jpg` files in a `dataset` folder.
2. Create 2 text files, a `<dataset>_train.txt` and `<dataset>_test.txt` that point to the files in your training and test set respectively (for example `find $(pwd)/<dataset>/train -name "*.jpg" > <dataset>_train.txt`).
3. Create a new config file based on `configs/custom_vqgan.yaml` to point to these 2 files. 

## Acknowledgement

  - Thanks to [Patrick Esser, Robin Rombach, Björn Ommer](https://github.com/CompVis) for the VQ-GAN architecture
    - [Github](https://github.com/CompVis/taming-transformers)
    - [Paper](https://arxiv.org/abs/2012.09841)
    - [Licence](taming/License.txt)
  - Thanks to [Chee Seng Chan](https://github.com/cs-chan) for the WikiArt dataset
    - [Github](https://github.com/cs-chan/ArtGAN/tree/master/WikiArt%20Dataset)