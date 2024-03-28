<div align="center">

# Graph Net

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)

</div>

## Description

Machine learning with Graph Neural Network

## Installation

#### Pip

```bash
# clone project
git clone https://github.com/avansp/graph-net
cd graph-net

# [OPTIONAL] create conda environment
conda create -n gnn python=3.9
conda activate gnn

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

#### Conda

```bash
# clone project
git clone https://github.com/avansp/graph-net
cd graph-net

# create conda environment and install dependencies
conda env create -f environment.yaml -n gnn

# activate conda environment
conda activate gnn
```

## How to run

Train model with default configuration

```bash
# train on CPU
python src/train.py trainer=cpu

# train on GPU
python src/train.py trainer=gpu
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train.py experiment=cora_gcn.yaml
```

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=200 data.batch_size=1
```

For evaluation, a pre-trained checkpoint is required

```bash
python src/eval.py ckpt_path=logs/train/runs/2024-03-28_11-23-24/checkpoints/epoch_033.ckpt
```

