import segmentation_models_pytorch as smp
from preprocess.augment import smp_get_preprocessing
import torch
from torch.utils.data import DataLoader
from data.dataset import ImageDataset
from smp_utils import TrainEpoch, ValidEpoch
import argparse
from config_loader import get_config

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


def get_scheduler(scheduler: str, optimizer, verbose: bool):
    if scheduler == 'ReduceLROnPlateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', verbose=verbose)
    elif scheduler == 'OneCycleLR':
        return torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, verbose=verbose)
    elif scheduler == 'CosineAnnealingLR':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, verbose=verbose)
    elif scheduler == 'CycleLR':
        return torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.1, verbose=verbose)
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

    preprocessing_fn = smp.encoders.get_preprocessing_fn(backbone, ENCODER_WEIGHTS)
    preprocessing_fn = smp_get_preprocessing(preprocessing_fn)
    encoder_weights = ENCODER_WEIGHTS

    # Look into aux_params - maybe also MA NET

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

    model.to(device)
    loss = get_loss_function(config['loss_function'])
    optimizer = get_optimizer(config['optimizer'], model, lr)

    metrics = ["f1_score",
               "iou_score",
               "accuracy",
               ]

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', verbose=True) # can play around with patience, factor, etc.

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
        metrics=metrics,
        optimizer=optimizer,
        device=device,
        verbose=True,
    )

    valid_epoch = ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=device,
        verbose=True,
    )

    best_iou = 0
    for i in range(epochs):
        print(f"Epoch {i+1}/{epochs}")
        train_logs = train_epoch.run(train_loader, i, config)
        valid_logs = valid_epoch.run(valid_loader, i, config)
        scheduler.step(valid_logs["iou_score"])

        if valid_logs["iou_score"] > best_iou:
            best_iou = valid_logs["iou_score"]
            torch.save(model.state_dict(), model_save_path)

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", default="unetpp_b5")

    args = parser.parse_args()

    config = get_config(args.config)
    train_smp(config, data_dir="/Users/sebastian/University/Master/second_term/cil/road-segmentation/data")
