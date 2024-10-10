from lightning import LightningModule
import torch
from torch_geometric.typing import (Adj, OptTensor)
from typing import Dict, Any, Tuple
import torch_geometric.nn as tgnn
from torchmetrics import Accuracy, MeanMetric, MaxMetric
from src.utils import RankedLogger


log = RankedLogger(__name__, rank_zero_only=True)


class GraphPrediction(LightningModule):
    """A `LightningModule` for graph-level prediction.

    Classifier is fixed with MLP.
    """
    def __init__(self,
                 num_features: int,
                 num_classes: int,
                 graph_net: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler = None,
                 graph_net_out_channels: int = 64,
                 pooling: str | Any = "add",
                 mlp_dropout: float = 0.5,
                 mlp_norm: str = "batch_norm",
                 compile: bool = False):
        """Initialise the module."""
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # layers
        self.graph_net = graph_net(in_channels=num_features)
        self.classifier = tgnn.MLP(
            [graph_net_out_channels, graph_net_out_channels, num_classes],
            norm=mlp_norm, dropout=mlp_dropout
        )

        # pooling
        match pooling:
            case "add":
                self.pooling = tgnn.global_add_pool
            case "max":
                self.pooling = tgnn.global_max_pool
            case "mean":
                self.pooling = tgnn.global_mean_pool
            case _:
                # should be a class with forward (x, batch) -> 1 channel
                self.pooling = pooling

        # loss function
        self.loss = torch.nn.CrossEntropyLoss()

        # create the metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

    def forward(self, x: torch.Tensor, edge_index: Adj, batch) -> torch.Tensor:
        """Perform a forward pass through the model: graph_net + pooling + classifier"""
        x = self.graph_net(x, edge_index)
        x = self.pooling(x, batch)
        x = self.classifier(x)
        return x

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins.

        By default lightning executes validation step sanity checks before training starts,
        therefore it's worth to make sure validation metrics don't store results from these checks.
        """
        # reset all validation metrics
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def model_step(self, data) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param data: A graph data batch

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        # perform forward calculation & retrieve the probability value of the graph classification result
        logits = self(data.x, data.edge_index, data.batch).softmax(dim=-1)

        # Cross Entropy loss calculation
        loss = torch.nn.functional.cross_entropy(logits, data.y)

        # pick the class (maximum probability)
        preds = torch.argmax(logits, dim=1)

        return loss, preds, data.y

    def training_step(self, data, batch_idx) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        # run the model_step, which we get the loss probability, predicted class, and the reference class
        loss, preds, targets = self.model_step(data)

        # update and log training metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log('train/acc', self.train_acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train/loss', self.train_loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, data, batch_idx) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        The same logic with the training_step, except that we calculate the validation metrics,
        """
        # run the model_step, which we get the loss probability, predicted class, and the reference class
        loss, preds, targets = self.model_step(data)

        # update and log validation metrics, except the best model
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log('val/acc', self.val_acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val/loss', self.val_loss, prog_bar=True, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self) -> None:
        """Lightning hook that is called when a validation epoch ends.

        At the end of each epoch, we update the best validation model.
        """
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, data, batch_idx) -> None:
        """Perform a test validation step on a batch of data from the test set.

        The same logic as the training_step, except that we update the test metrics.
        """
        # run the model_step, which we get the loss probability, predicted class, and the reference class
        loss, preds, targets = self.model_step(data)

        # update and log test metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log('test/acc', self.test_acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log('test/loss', self.test_loss, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizer only"""
        # Hint: we're using self.trainer.model instead of self.net because the self.net hasn't been initiated.
        conf = {"optimizer": self.hparams.optimizer(params=self.trainer.model.parameters())}

        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=conf['optimizer'])
            conf['lr_scheduler'] = {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                }

        return conf

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.graph_net = torch.compile(self.graph_net)
            self.classifier = torch.compile(self.classifier)

