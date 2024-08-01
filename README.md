# CIL Road Segmentation Project
This project is completed as a part of the Computational Intelligence Lab course during the Spring 2024 semester. This GitHub repository contains the code for the Road Segmentation project.

## Installation

First things first, you must open this repository on your local machine or cluster.

```bash
cd road-segmentation/
```

## Virtual Environment/Requirement Installation

Now, you must have all the required libraries installed. The easiest way to do this is with a virtual environment.

```bash
python3 -m venv roadsegenv
source roadsegenv/bin/activate
pip install -r requirements.txt
```
## Training
To train on the student-cluster we provide a custom script 'train_segment.py' which you can run with the following command:
You will have to manually change the desired config in the train_segment.py file. 
```bash
sbatch train_segment.sh
```
We provide the configs for the models that we have used in the config_loader.py file, but 
they can also be changed

| Parameter             | Example Value          | Description                                                             |
|-----------------------|------------------------|-------------------------------------------------------------------------|
| `decoder_channels`    | [256, 128, 64, 32, 16] | List of channels for each decoder block in the network.                 |
| `backbone`            | efficientnet-b7        | Pre-trained model backbone used for feature extraction.                 |
| `epochs`              | 100                    | Number of training epochs.                                              |
| `use_epfl`            | True                   | Whether to use the EPFL dataset.                                        |
| `use_deepglobe`       | True                   | Whether to use the DeepGlobe dataset.                                   |
| `augmentation_factor` | 8                      | Factor by which to augment the original ETH data.                       |
| `transformation`      | minimal                | Type of transformation to apply during data augmentation.               |
| `resize`              | 416                    | Resize dimension for the input images.                                  |
| `validation_size`     | 0.2                    | Proportion of the dataset to be used for validation.                    |
| `seed`                | 42                     | Random seed for reproducibility.                                        |
| `batch_size`          | 8                      | Number of samples per batch.                                            |
| `lr`                  | 0.001                  | Learning rate for the optimizer.                                        |
| `device`              | cuda                   | Device to be used for training (e.g., 'cuda' for GPU or 'cpu' for CPU). |
| `metric`              | iou_score              | Metric used for evaluating the model performance.                       |
| `model_save_path`     | custom_save_path       | Path where the trained model will be saved.                             |
| `model_name`          | UnetPlusPlus           | Name of the model architecture.                                         |
| `loss_function`       | SoftBCEWithLogitsLoss  | Loss function used for training the model.                              |
| `optimizer`           | AdamW                  | Optimizer used for training the model.                                  |
| `show_val`            | True                   | Whether to show validation results during training.                     |


Each config is named. To train a model, you can run the following command:

```bash
python train_smp.py --config 'config_name'
```

## Prediction

To predict the models need to be in the model_checkpoints folder. You can download the models that we used for the final prediction from the following links:

UNet++: https://polybox.ethz.ch/index.php/s/ia04RfduAomnGfY

DeepLabV3+: https://polybox.ethz.ch/index.php/s/CsxmE1UssTIxhwj

PSPNet: https://polybox.ethz.ch/index.php/s/1enDvq10AekQZ6a

You can either specify multiple models or just a single one to get the prediction. The length of the models, the checkpoints and the backbones has to match.

The prediction to recreate our results can be done with the following command:

```bash
python submit_smp.py --models UnetPlusPlus DeepLabV3Plus PSPNet --checkpoints model_checkpoints/UNetpp_B7_Final.pt model_checkpoints/DeepLab_regnetx_final.pt model_checkpoints/Psp_resnet200e_final.pt --backbones efficientnet-b7 timm-regnetx_160 timm-resnest200e --device cpu --data-dir data --submission-dir submissions/Ensemble.csv
```


## Team/Contributors
- Sebastian Hönig
- Kerem Güra
- Samuel-Moor Smith
