import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(config):
    """
    Returns a composition of image transformations for training.
    """

    transforms_list = [
        A.Resize(config.image_size, config.image_size),
    ]

    # Add augmentations from config
    for aug_config in config.train_augmentations:
        aug_class = getattr(A, aug_config["name"])
        aug_params = aug_config.copy()
        del aug_params["name"]
        transforms_list.append(aug_class(**aug_params))

    # Additional augmentations suitable for deepfake detection
    transforms_list.extend(
        [
            A.RandomBrightnessContrast(p=0.2),
            A.GaussNoise(p=0.2),
            A.MotionBlur(p=0.2),
            A.Normalize(),
            ToTensorV2(),
        ]
    )

    return A.Compose(transforms_list)


def get_val_transforms(config):
    """
    Returns a composition of image transformations for validation.
    """
    return A.Compose(
        [
            A.Resize(config.image_size, config.image_size),
            A.Normalize(),
            ToTensorV2(),
        ]
    )
