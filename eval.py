import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from torchmetrics import Accuracy
from swin_base import SwinTransformer
from dataloader import create_dataloaders
from augmentations import get_val_transforms
from metrics import calculate_metrics
from configs.config import Config


class DeepfakeDetectionModel(pl.LightningModule):
    def __init__(self, config):
        super(DeepfakeDetectionModel, self).__init__()
        self.config = config
        self.model = SwinTransformer(config)
        self.criterion = getattr(F, config.loss_function)()
        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.train_accuracy(logits, y)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "train_acc",
            self.train_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.val_accuracy(logits, y)
        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "val_acc",
            self.val_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.config.optimizer)(
            self.parameters(), lr=self.config.learning_rate
        )
        return optimizer

    def train_dataloader(self):
        train_transforms = get_train_transforms(self.config)
        train_loader, _ = create_dataloaders(self.config, train_transforms, None)
        return train_loader

    def val_dataloader(self):
        val_transforms = get_val_transforms(self.config)
        _, val_loader = create_dataloaders(self.config, None, val_transforms)
        return val_loader


if __name__ == "__main__":
    config = Config("configs/config.yaml")
    model = DeepfakeDetectionModel.load_from_checkpoint(
        checkpoint_path="path/to/your/best_checkpoint.ckpt", config=config
    )
    model.eval()

    val_transforms = get_val_transforms(config)
    _, val_loader = create_dataloaders(config, None, val_transforms)

    trainer = pl.Trainer()
    predictions = trainer.predict(model, val_loader)
    predictions = torch.cat(predictions)
    targets = torch.cat([y for _, y in val_loader])

    metrics = calculate_metrics(predictions, targets)
    print(metrics)
