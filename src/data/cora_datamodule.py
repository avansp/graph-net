from typing import Any, Dict, Optional, Tuple

from lightning import LightningDataModule
import torch_geometric as tg
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


class CoraDataModule(LightningDataModule):
    """`LightningDataModule` for the Cora dataset.

    The Cora dataset consists of 2,708 scientific publications with link between
    each others representing a citation from one paper to another.

    * Each publication belongs to one out of seven classes.
    * Each publication is represented by a bag-of-words vector
      containing 1,433 binary elements, where 1 at $i$-th element means
      that the $i$-th word in a pre-defined dictionary is in that publication.

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

    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False
    ) -> None:
        """Initialise a `CoraDataModule`.

        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data sets
        self.cora = None
        self.data_train: Optional[Data] = None
        self.data_val: Optional[Data] = None
        self.data_test: Optional[Data] = None

    def prepare_data(self) -> None:
        """
        Download data if needed.
        """
        # Download Cora dataset if needed
        self.cora = tg.datasets.Planetoid(root=self.hparams.data_dir, name="Cora")

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # no need to setup, everything has been fixed during prepare_data
        pass

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.data_train, batch_size=1)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.data_val, batch_size=1)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.data_test, batch_size=1)
