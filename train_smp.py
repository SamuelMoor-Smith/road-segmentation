"""
Main training script for training a model using segmentation_models_pytorch. The available models are UnetPlusPlus,
PSPNet and DeepLabV3Plus. The script uses the TrainEpoch and ValidEpoch from the deprecated utils module from smp
and thus we have copied the code here.

How to use:
    1. Create a config file in the config_loader.py file
    2. Activate a virtual environment with python version 3.10
    3. pip install -r requirements.txt
    4. Run the script with the following command:
        python train_smp.py --config defined_config --data_dir /path/to/data
"""

import segmentation_models_pytorch as smp
from preprocess.augment import smp_get_preprocessing
import torch
from torch.utils.data import DataLoader
from data.dataset import ImageDataset
from smp_utils import TrainEpoch, ValidEpoch
import argparse
from config_loader import get_config
from models.UNet_provided import UNet
from models.ResNet34_deeper import ResNetBackbone as ResNetBackboneDeeper
from models.ResNet34 import ResNetBackbone
from models.LinkNet import LinkNet
from models.DLinkNet import LinkNet as DLinkNet
from models.NLLinkNet import NL34_LinkNet as NLLinkNet

ENCODER_WEIGHTS = 'imagenet'
#DATA_DIR = "/Users/sebastian/University/Master/second_term/cil/road-segmentation/data/training"


def get_loss_function(loss_function: str):
    if loss_function == 'SoftBCEWithLogitsLoss':
        return smp.losses.SoftBCEWithLogitsLoss()
    elif loss_function == 'DiceLoss':
        return smp.losses.DiceLoss(mode='binary')
    elif loss_function == 'JaccardLoss':
        return smp.losses.JaccardLoss(mode='binary')
    elif loss_function == 'TverskyLoss':
        return smp.losses.TverskyLoss(mode='binary')
    else:
        raise ValueError(f"Loss function {loss_function} not recognized")


def get_optimizer(optimizer: str, model, lr: float):
    if optimizer == 'Adam':
        return torch.optim.Adam([
            dict(params=model.parameters(), lr=lr),
        ])
    elif optimizer == 'AdamW':
        return torch.optim.AdamW([
            dict(params=model.parameters(), lr=lr),
        ])
    elif optimizer == 'NAdam':
        return torch.optim.NAdam([
            dict(params=model.parameters(), lr=lr),
        ])
    elif optimizer == 'SGD':
        return torch.optim.SGD([
            dict(params=model.parameters(), lr=lr),
        ])
    else:
        raise ValueError(f"Optimizer {optimizer} not recognized")


def get_scheduler(scheduler: str, optimizer):
    if scheduler == 'ReduceLROnPlateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', verbose=True)
    elif scheduler == 'OneCycleLR':
        return torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, verbose=True)
    elif scheduler == 'CosineAnnealingLR':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, verbose=True)
    elif scheduler == 'CycleLR':
        return torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.01, verbose=True)
    else:
        raise ValueError(f"Scheduler {scheduler} not recognized")


def train_smp(config, data_dir: str):
    decoder_channels = config['decoder_channels']
    backbone = config['backbone']
    epochs = config['epochs']
    use_epfl = config['use_epfl']
    use_deepglobe = config['use_deepglobe']
    augmentation_factor = config['augmentation_factor']
    transformation = config['transformation']
    resize = config['resize']
    validation_size = config['validation_size']
    seed = config['seed']
    batch_size = config['batch_size']
    lr = config['lr']
    device = config['device']
    model_save_path = config['model_save_path']
    model_name = config['model_name']
    metric = config['metric']

    if model_name in ['UnetPlusPlus', 'DeepLabV3Plus', 'PSPNet']:
        encoder_weights = ENCODER_WEIGHTS
        if 'encoder_weights' in config:
            encoder_weights = config['encoder_weights']

        preprocessing_fn = smp.encoders.get_preprocessing_fn(backbone, encoder_weights)
        preprocessing_fn = smp_get_preprocessing(preprocessing_fn)

        if 'model' in config and config['model'] is not None:  # if we want to continue training
            model = config['model']
        else:
            if model_name == 'UnetPlusPlus':
                model = smp.UnetPlusPlus(
                    encoder_name=backbone,
                    encoder_weights=encoder_weights,
                    decoder_channels=decoder_channels,
                    decoder_attention_type=None,
                    classes=1,
                    activation=None,
                )
            elif model_name == 'PSPNet':
                model = smp.PSPNet(
                    encoder_name=backbone,
                    encoder_weights=encoder_weights,
                    classes=1,
                    activation=None,
                )
            elif model_name == 'DeepLabV3Plus':
                model = smp.DeepLabV3Plus(
                    encoder_name=backbone,
                    encoder_weights=encoder_weights,
                    classes=1,
                    activation=None,
                )
            else:
                raise ValueError(f"Model name {model_name} not recognized")
    else:
        if model_name == 'UNet':
            model = UNet()
        elif model_name == 'ResNet34':
            model = ResNetBackbone()
        elif model_name == 'ResNet34Deeper':
            model = ResNetBackboneDeeper()
        elif model_name == 'LinkNet':
            model = LinkNet()
        elif model_name == 'DLinkNet':
            model = DLinkNet()
        elif model_name == 'NLLinkNet':
            model = NLLinkNet()
        else:
            raise ValueError(f"Model name {model_name} not recognized")
        preprocessing_fn = None

    model.to(device)
    loss = get_loss_function(config['loss_function'])
    optimizer = get_optimizer(config['optimizer'], model, lr)
    if 'scheduler' in config:
        scheduler = get_scheduler(config['scheduler'], optimizer)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', verbose=True)

    train_dataset = ImageDataset(
        data_dir=data_dir,
        is_train=True,
        device=device,
        use_epfl=use_epfl,
        use_deepglobe=use_deepglobe,
        validation_size=validation_size,
        seed=seed,
        transforms=transformation,
        preprocess=preprocessing_fn,
        augmentation_factor=augmentation_factor,
        resize=resize)

    valid_dataset = ImageDataset(
        data_dir=data_dir,
        is_train=False,
        device=device,
        use_epfl=False,
        use_deepglobe=False,
        validation_size=validation_size,
        seed=seed,
        transforms='validation',
        preprocess=preprocessing_fn,
        augmentation_factor=augmentation_factor,
        resize=resize,)

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=0,
                              pin_memory=True,
                              worker_init_fn=42,
                              )
    valid_loader = DataLoader(valid_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=0,
                              pin_memory=True,
                              worker_init_fn=42,
                              )

    train_epoch = TrainEpoch(
        model,
        loss=loss,
        metrics=["f1_score", "iou_score", "accuracy"],
        optimizer=optimizer,
        device=device,
        verbose=True,
    )

    valid_epoch = ValidEpoch(
        model,
        loss=loss,
        metrics=["f1_score", "iou_score", "accuracy"],
        device=device,
        verbose=True,
    )

    best_score = 0

    for i in range(epochs):
        print(f"Epoch {i+1}/{epochs}")
        train_logs = train_epoch.run(train_loader, i, config)
        valid_logs = valid_epoch.run(valid_loader, i, config)
        scheduler.step(valid_logs[metric])

        if valid_logs[metric] > best_score:
            best_score = valid_logs[metric]
            torch.save(model.state_dict(), model_save_path)

    return model


if __name__ == "__main__":
    """parser = argparse.ArgumentParser()

    parser.add_argument("--config", default="unetpp_b5")
    parser.add_argument("--data_dir", default="/home/shoenig/road-segmentation/data")

    args = parser.parse_args()

    config = get_config(args.config)
    data_dir = args.data_dir
    train_smp(config, data_dir=data_dir)"""
    config = {
        'decoder_channels': [256, 128, 64, 32, 16],
        'backbone': 'efficientnet-b7',
        'epochs': 100,
        'use_epfl': True,
        'use_deepglobe': True,
        'augmentation_factor': 4,
        'transformation': 'minimal',
        'resize': 416,
        'validation_size': 0.01,
        'seed': 42,
        'batch_size': 8,
        'lr': 0.001,
        'device': 'cpu',
        'metric': 'iou_score',
        'model_save_path': '/content/drive/MyDrive/models/Minimal',
        'model_name': 'UnetPlusPlus',
        'loss_function': 'SoftBCEWithLogitsLoss',
        'optimizer': 'AdamW',
        'show_val': False,
    }
    train_smp(config, data_dir=data_dir)
