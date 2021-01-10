from sklearn.metrics import accuracy_score
import pytorch_lightning as pl
import torch.nn.functional as F
import torch
from template_utils.initializers import load_class

# import custom architectures
from pytorch_modules.architectures.simple_mnist import SimpleMNISTClassifier


class LitModel(pl.LightningModule):
    """
    This is example of lightning model for MNIST classification.
    This class enables you to specify what happens during training, validation and test step.
    You can just remove 'validation_step()' or 'test_step()' methods if you don't want to have them during training.
    To learn how to create lightning models visit:
        https://pytorch-lightning.readthedocs.io/en/latest/lightning_module.html
    """

    def __init__(self, model_config, optimizer_config):
        super().__init__()
        hparams = {**model_config["args"], **optimizer_config["args"]}
        self.save_hyperparameters(hparams)
        self.optimizer_config = optimizer_config

        self.model = SimpleMNISTClassifier(hparams=self.hparams)

    def forward(self, x):
        return self.model(x)

    # logic for a single training step
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = F.nll_loss(logits, y)

        # training metrics
        preds = torch.argmax(logits, dim=1)
        acc = accuracy_score(preds.cpu(), y.cpu())
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here anything and then read it in some callback
        return {"loss": loss, "acc": acc, "preds": preds, "y": y}

    # logic for a single validation step
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = F.nll_loss(logits, y)

        # validation metrics
        preds = torch.argmax(logits, dim=1)
        acc = accuracy_score(preds.cpu(), y.cpu())
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "acc": acc, "preds": preds, "y": y}

    # logic for a single testing step
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = F.nll_loss(logits, y)

        # test metrics
        preds = torch.argmax(logits, dim=1)
        acc = accuracy_score(preds.cpu(), y.cpu())
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_acc', acc, on_step=False, on_epoch=True)

        return {"loss": loss}

    def configure_optimizers(self):
        Optimizer = load_class(self.optimizer_config["class"])
        return Optimizer(self.parameters(), **self.optimizer_config["args"])
