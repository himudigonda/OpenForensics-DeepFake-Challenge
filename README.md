# Deepfake Detection with Swin Transformer

This repository contains code for deepfake detection using the Swin Transformer architecture. The project leverages PyTorch and PyTorch Lightning for training and evaluation.

## Dataset Structure

The dataset should be organized as follows:

```
/data/
    train/
        Fake/
            fake_0.jpg
            fake_1.jpg
            ...
        Real/
            real_0.jpg
            real_1.jpg
            ...
    val/
        Fake/
            ...
        Real/
            ...
    test/
        Fake/
            ...
        Real/
            ...
```

## Requirements

* Python 3.9+
* PyTorch 
* PyTorch Lightning
* timm (PyTorch Image Models)
* albumentations
* torchmetrics
* Other dependencies listed in `requirements.txt`

Install the required packages using:

```bash
pip install -r requirements.txt
```

## Configuration

The project's settings are defined in `configs/config.yaml`. You can modify parameters such as:

* `data_root`: Path to the dataset root directory
* `model_name`: Swin Transformer model name from `timm`
* `batch_size`, `num_epochs`, `learning_rate`, etc.
* `train_augmentations` and `val_augmentations`: Albumentations transformations

## Training

To train the model, run:

```bash
python3 train.py
```

Training progress and metrics will be logged to TensorBoard. Checkpoints will be saved in the `checkpoints` directory.

## Evaluation

To evaluate a trained model on the validation set, use:

```bash
python3 eval.py
```

Evaluation metrics will be printed to the console.

## Key Files

* `train.py`: Main training script.
* `eval.py`: Evaluation script.
* `swin_base.py`: Defines the Swin Transformer model.
* `dataloader.py`: Handles dataset loading and augmentations.
* `augmentations.py`: Defines image transformations.
* `metrics.py`: Calculates evaluation metrics.
* `configs/config.yaml`: Project configuration file.

## Acknowledgements

* The Swin Transformer implementation is based on the `timm` library.
* Albumentations is used for image augmentations.
* PyTorch Lightning simplifies training and experimentation.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
