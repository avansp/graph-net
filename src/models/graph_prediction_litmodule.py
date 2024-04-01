from typing import Any, Dict
from lightning import LightningModule
import torch
import torch_geometric.data as geom_data


"""
A `LightningModule` implements 8 key methods:

```python
def __init__(self):
# Define initialization code here.

def setup(self, stage):
# Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
# This hook is called on every process when using DDP.

def training_step(self, batch, batch_idx):
# The complete training step.

def validation_step(self, batch, batch_idx):
# The complete validation step.

def test_step(self, batch, batch_idx):
# The complete test step.

def predict_step(self, batch, batch_idx):
# The complete predict step.

def configure_optimizers(self):
# Define and configure optimizers and LR schedulers.
```

Docs:
    https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
"""


class GraphPredictionLitModule(LightningModule):
    """Graph-level classification lightning module."""
    def __init__(self,
                 net: torch.nn.Module,
                 optimizer: torch.optim.Optimizer):
        """Initialise the object.

        :params net: The graph network
        :params optimizer: PyTorch's optimizer
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.model = net
        self.loss_module = torch.nn.BCEWithLogitsLoss() if self.model.c_out == 1 else torch.nn.CrossEntropyLoss()

    def forward(self, data: geom_data.Data, mode="train") -> tuple:
        """Lightning forward step"""
        # get the forward passing value
        x = self.model(data.x, data.edge_index, data.batch)
        x = x.squeeze(dim=-1)

        # binary vs multi-class
        if self.model.c_out == 1:
            preds = (x > 0).float()
            data.y = data.y.float()
        else:
            preds = x.argmax(dim=-1)

        # calculate losses
        loss = self.loss_module(x, data.y)
        acc = (preds == data.y).sum().float() / preds.shape[0]

        return loss, acc

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.
        """
        # perform one step training and get loss & accuracy values
        loss, acc = self.forward(batch, mode="train")

        # update log metrics
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        # must return loss to allow backpropagation
        return loss

    def validation_step(self, batch, batch_idx) -> None:
        """Perform a single validation step on a batch of data from the validation set."""
        _, acc = self.forward(batch, mode="val")

        # update log metrics
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx) -> None:
        """Perform a test validation step on a batch of data from the test set."""
        loss, acc = self.forward(batch, mode="test")

        # update log metrics
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizer only"""
        return {
            "optimizer": self.hparams.optimizer(params=self.trainer.model.parameters())
        }

