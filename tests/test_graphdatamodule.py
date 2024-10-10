from pathlib import Path
from src.data.graph_datamodule import GraphDataModule
from torch_geometric.datasets import TUDataset
from collections import Counter
import torch


def test_graph_datamodule() -> None:
    """Tests with MUTAG to verify that it can be downloaded & split correctly."""
    data_dir = "data/"

    # dataset
    dataset = TUDataset(data_dir, name="MUTAG", cleaned=True)

    # create the module with mandatory parameters - and check the default parameters
    dm = GraphDataModule(data_dir=data_dir, dataset=dataset)
    split = dm.hparams.split
    assert len(split) == 3
    assert sum(split) == 1.0

    # # check the availability of the data
    dm.prepare_data()
    assert Path(data_dir, "MUTAG").exists()
    assert Path(data_dir, "MUTAG", "processed_cleaned").exists()
    assert Path(data_dir, "MUTAG", "raw_cleaned").exists()

    # this is MUNTAG
    assert dm.num_classes == 2
    assert dm.num_features == 7

    # check data after splitting
    dm.setup(stage="fit")

    # collect classes in train_loader
    train_loader = dm.train_dataloader()
    assert train_loader is not None

    y = []
    for ds in train_loader:
        y += torch.stack([d.y for d in ds.to_data_list()]).flatten().tolist()
    y_train = Counter(y)
    prop_train = [y_train[0] / len(y), y_train[1] / len(y)]
    assert len(y_train) == dm.num_classes

    # collect classes in test_loader
    test_loader = dm.test_dataloader()
    assert test_loader is not None

    y = []
    for ds in test_loader:
        y += torch.stack([d.y for d in ds.to_data_list()]).flatten().tolist()
    y_test = Counter(y)
    prop_test = [y_test[0] / len(y), y_test[1] / len(y)]
    assert len(y_test) == dm.num_classes

    # collect classes in val_loader
    val_loader = dm.val_dataloader()
    assert val_loader is not None

    y = []
    for ds in val_loader:
        y += torch.stack([d.y for d in ds.to_data_list()]).flatten().tolist()
    y_val = Counter(y)
    prop_val = [y_val[0] / len(y), y_val[1] / len(y)]
    assert len(y_val) == dm.num_classes

    prop_0 = [prop_train[0], prop_val[0], prop_test[0]]
    prop_1 = [prop_train[1], prop_val[1], prop_test[1]]

    mean_0 = sum(prop_0) / 3.0
    mean_1 = sum(prop_1) / 3.0

    diff_0 = [abs(x-mean_0) < 0.1 for x in prop_0]
    diff_1 = [abs(x-mean_1) < 0.1 for x in prop_1]

    assert all(diff_0)
    assert all(diff_1)


