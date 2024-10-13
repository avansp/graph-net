<div align="center">

# Graph Net

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a>

</div>

## Description

Graph Neural Network framework with [pytorch geometric](https://pytorch-geometric.readthedocs.io/en/latest/).

## Installation

<details><summary>Pip</summary>

```bash
# clone project
git clone https://github.com/avansp/graph-net
cd graph-net

# [OPTIONAL] create conda environment
conda create -n graphnet python=3.11
conda activate graphnet

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

</details>

<details><summary>Conda</summary>

```bash
# clone project
git clone https://github.com/avansp/graph-net
cd graph-net

# create conda environment and install dependencies
conda env create -f environment.yaml -n graphnet

# activate conda environment
conda activate graphnet
```

</details>

## Quick run

### Training

<details><summary>Default training</summary>

The default configuration for training is a graph classification task with [MUTAG](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.TUDataset.html) dataset by using GPU accelerator.

```bash
python src/train.py
```

which is defined in [`train.yaml`](configs/train.yaml) file, listing the following sub configs:

- data: [`mutag.yaml`](configs/data/mutag.yaml)
- model: [`graph_pred.yaml`](configs/model/graph_pred.yaml)
- callbacks: [`default.yaml`](configs/callbacks/default.yaml)
- logger: [`mlflow.yaml`](configs/logger/mlflow.yaml)
- trainer: [`default.yaml`](configs/trainer/default.yaml)
- paths: [`default.yaml`](configs/paths/default.yaml)
- extras: [`default.yaml`](configs/extras/default.yaml)
- hydra: [`default.yaml`](configs/hydra/default.yaml)

</details>

<details><summary>Overriding parameters</summary>

The script is highly configured by YAML files in the [configs](configs) directory.
You can override all parameters from the command line, for examples:

* Change maximum epochs in `configs/trainer/default.yaml` and the batch size in `configs/data/mutag.yaml`:
    ```bash
    python src/train.py trainer.max_epcohs=50 data.batch_size=16
    ```
  
* Use experiment configuration, e.g. prediction on IMDB-BINARY dataset defined in [`imdb.yaml`](configs/experiment/imdb.yaml)
    ```bash
    python src/train.py experiment=imdb
    ```
  
</details>

### Evaluation

Provide the checkpoint for evaluation
```bash
python src/eval.py ckpt_path=./logs/train/runs/2024-10-12_18-17-37/checkpoints/epoch_024.ckpt
```

