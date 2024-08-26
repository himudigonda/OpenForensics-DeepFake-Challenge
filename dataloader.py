import numpy as np
import os
from torch.utils.data import DataLoader, Dataset
from PIL import Image


class DeepfakeDataset(Dataset):
    def __init__(self, root, split, transform=None):
        self.root = root
        self.split = split  # 'train', 'val', or 'test'
        self.transform = transform
        self.image_paths = []
        self.labels = []

        split_dir = os.path.join(self.root, self.split)
        for label_folder in ["Fake", "Real"]:
            label_dir = os.path.join(split_dir, label_folder)
            for filename in os.listdir(label_dir):
                if filename.endswith(".jpg"):
                    self.image_paths.append(os.path.join(label_dir, filename))
                    self.labels.append(0 if label_folder == "Real" else 1)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)

        label = self.labels[index]

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        return image, label


def create_dataloaders(config, train_transforms, val_transforms):
    """
    Creates train and validation dataloaders.

    Args:
        config: Configuration object.
        train_transforms: Transformations to apply to training images.
        val_transforms: Transformations to apply to validation images.

    Returns:
        train_loader, val_loader: PyTorch DataLoaders for training and validation.
    """
    train_dataset = DeepfakeDataset(
        root=config.data_root,
        split=config.train_dir,
        transform=train_transforms,
    )
    val_dataset = DeepfakeDataset(
        root=config.data_root, split=config.val_dir, transform=val_transforms
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    return train_loader, val_loader
