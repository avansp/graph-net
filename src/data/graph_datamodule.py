from torch_geometric.data import Dataset
from lightning import LightningDataModule
from typing import Optional
from torch_geometric.loader import DataLoader
from src.utils import (RankedLogger)
from typing import List
from sklearn.model_selection import StratifiedShuffleSplit
import torch

log = RankedLogger()


class GraphDataModule(LightningDataModule):
    """Abstract class of `LightningDataModule` for the graph dataset.

    Note: you must implement these methods:
    - load_data

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
            data_dir: str,
            dataset: Dataset,
            split: List[float] = None,
            split_seed: float = 1234,
            batch_size: int = 64,
            num_workers: int = 0,
            pin_memory: bool = False,
    ) -> None:
        super().__init__()
        self._num_classes = -1   # will be defined after prepare_data
        self._num_features = -1  # will be defined after prepare_data
        self.data_train = self.data_test = self.data_val = None

        # the split
        if split is None:
            split = [0.6, 0.2, 0.2]
        assert sum(split) == 1.0, f"The sum of split is not 1.0"
        assert len(split) == 3, f"The list must be [train_part, val_part, test_part], with sum == 1.0"
        self._split = split
        self._split_seed = split_seed

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def num_features(self) -> int:
        return self._num_features

    def load_data(self) -> Dataset:
        """Helper to consistently load data."""
        return self.hparams.dataset

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        # this will download if necessary
        dt = self.load_data()
        self._num_classes = dt.num_classes
        self._num_features = dt.num_features

        # some assertion checks
        assert self._num_classes > 0, f"Number of graph class is not positive (={self._num_classes})"
        assert self._num_features > 0, f"Number of features is not defined (={self._num_features})"

        log.info(f"Data is ready: num_graphs={len(dt)}, {self.num_classes=}, {self.num_features=}.")

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        log.debug(f"Setup at {stage=}. Has the test data been split? {self.data_test is not None}")

        if stage == "fit":
            if self.data_test is not None:
                log.warning(f"The test data has been created. This will reshuffled & split the data for train & test")

            dataset = self.load_data()
            dataset = dataset.shuffle()

            # get class list
            y = torch.stack([ds.y for ds in dataset])

            # we're going to use StratifiedShuffleSplit twice
            # for the test, we use the split directly
            sss_test = StratifiedShuffleSplit(n_splits=1, test_size=self._split[-1], random_state=self._split_seed)
            trainval_idx, test_idx = next(sss_test.split(X=torch.zeros(len(dataset)), y=y))

            # sanity check
            assert len(set(trainval_idx).intersection(test_idx)) == 0, "There is a mixed index between trainval_idx & test_idx"

            # for the validation split, we need to re-calculate it again
            val_size = self._split[1] / (self._split[0] + self._split[1])
            sss_val = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=self._split_seed)
            y_trainval = y[trainval_idx]
            train_idx, val_idx = next(sss_val.split(X=torch.zeros(len(y_trainval)), y=y_trainval))

            # sanity check
            assert len(set(train_idx).intersection(val_idx)) == 0, "There is a mixed index between train_idx & val_idx"
            assert len(train_idx) + len(val_idx) + len(test_idx) == len(dataset)

            # create train, val & test datasets
            self.data_test = dataset[test_idx]
            self.data_val = dataset[val_idx]
            self.data_train = dataset[train_idx]

        if stage == "test":
            if self.data_test is None:
                log.error(f"Test data is undefined. Call fit() first to define train & test datasets.")

    def train_dataloader(self) -> DataLoader:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        assert self.data_train, "Train dataset is not initiated"

        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        assert self.data_test, "Test dataset is not initiated"

        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def val_dataloader(self) -> DataLoader:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        assert self.data_val, "Validation dataset is not initiated"

        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

