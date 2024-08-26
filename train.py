from torch.nn import CrossEntropyLoss
import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from torchmetrics import Accuracy
from swin_base import SwinTransformer
from dataloader import create_dataloaders
from augmentations import get_train_transforms, get_val_transforms
import sys
import os
import yaml

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Config:
    def __init__(self, config_path):
        with open(config_path, "r") as f:
            self.__dict__ = yaml.safe_load(f)


class DeepfakeDetectionModel(pl.LightningModule):
    def __init__(self, config):
        super(DeepfakeDetectionModel, self).__init__()
        self.config = config
        self.model = SwinTransformer(config)
        # self.criterion = getattr(F, config.loss_function)()
        self.criterion = CrossEntropyLoss()
        self.train_accuracy = Accuracy(task="binary")  # Specify task="binary"
        self.val_accuracy = Accuracy(task="binary")  # Specify task="binary"

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        # Convert logits to predicted class labels
        _, preds = torch.max(logits, 1)

        self.train_accuracy(preds, y)  # Pass predicted labels to accuracy metric
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

        # Convert logits to predicted class labels
        _, preds = torch.max(logits, 1)

        self.val_accuracy(preds, y)  # Pass predicted labels to accuracy metric
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "val_acc",
            self.val_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
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
    model = DeepfakeDetectionModel(config)

    # Logging and Checkpointing
    logger = pl.loggers.TensorBoardLogger(
        save_dir=config.log_dir, name="deepfake_detection"
    )
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=config.checkpoint_dir,
        filename="{epoch}-{val_loss:.2f}-{val_acc:.2f}",
        monitor="val_loss",
        mode="min",
        save_top_k=config.save_top_k,
    )

    trainer = pl.Trainer(
        max_epochs=config.num_epochs, logger=logger, callbacks=[checkpoint_callback]
    )
    trainer.fit(model)
