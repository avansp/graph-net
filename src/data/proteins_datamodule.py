from lightning import LightningDataModule
import torch_geometric as geom
from torch_geometric.loader import DataLoader
import torch

"""
A `LightningDataModule` implements 7 key methods:

```python
    def prepare_data(self):
    # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
    # Download data, pre-process, split, save to disk, etc...

    def setup(self, stage):
    # Things to do on every process in DDP.
    # Load data, set variables, etc...

    def train_dataloader(self):
    # return train dataloader

    def val_dataloader(self):
    # return validation dataloader

    def test_dataloader(self):
    # return test dataloader

    def predict_dataloader(self):
    # return predict dataloader

    def teardown(self, stage):
    # Called on every process in DDP.
    # Clean up after fit or test.
```

This allows you to share a full dataset without explaining how to download,
split, transform and process the data.

Read the docs:
    https://lightning.ai/docs/pytorch/latest/data/datamodule.html
"""


class ProteinsDataModule(LightningDataModule):
    """`LightningDataModule` for the PROTEINS dataset from torch.

    The PROTEINS dataset consists of 1,113 protein molecules defined by the TUDataset.

    Read the docs:
        https://chrsmrrs.github.io/datasets/docs/datasets/
        https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.TUDataset.html

    Some information about the PROTEINS dataset:
    * Features for each node is a 3-element one-hot vectors
    * There are two types of molecule: enzyme (y=1) and non-enzyme (y=0)
    * The common task is binary graph classification
    """
    def __init__(self,
                 data_dir: str,
                 batch_size: int,
                 data_split: list,
                 num_workers: int = 0,
                 pin_memory: bool = False) -> None:
        """Initialise the object.

        :param data_dir: a folder to save the data.
        :param batch_size: mini batch size
        :param data_split: [train, val, test] fractional split; sum must be 1.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # initialise the dataset
        self.train_data = None
        self.val_data = None
        self.test_data = None

    def prepare_data(self) -> None:
        """Download the data (if necessary), then split."""
        ds = geom.datasets.TUDataset(root=self.hparams.data_dir, name="PROTEINS")

        # random split
        self.train_data, self.val_data, self.test_data = torch.utils.data.random_split(ds, self.hparams.data_split)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_data, batch_size=1)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_data, batch_size=1)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_data, batch_size=1)
