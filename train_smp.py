import segmentation_models_pytorch as smp
from preprocess.augment import smp_get_preprocessing
import torch
from torch.utils.data import DataLoader
from data.dataset import ImageDataset
from smp_utils import TrainEpoch, ValidEpoch

ENCODER_WEIGHTS = 'imagenet'
#DATA_DIR = "/Users/sebastian/University/Master/second_term/cil/road-segmentation/data/training"

def get_loss_function(loss_function: str):
    if loss_function == 'SoftBCEWithLogitsLoss':
        return smp.losses.SoftBCEWithLogitsLoss()
    elif loss_function == 'DiceLoss':
        return smp.losses.DiceLoss(mode='binary', from_logits=False)
    elif loss_function == 'FocalLoss':
        return smp.losses.FocalLoss(mode='binary')
    elif loss_function == 'JaccardLoss':
        return smp.losses.JaccardLoss(mode='binary', from_logits=False)
    elif loss_function == 'TverskyLoss':
        return smp.losses.TverskyLoss(mode='binary', from_logits=False)
    else:
        raise ValueError(f"Loss function {loss_function} not recognized")

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
    #loss_function = config['loss_function']
    activation = config['activation']

    preprocessing_fn = smp.encoders.get_preprocessing_fn(backbone, ENCODER_WEIGHTS)
    preprocessing_fn = smp_get_preprocessing(preprocessing_fn)
    encoder_weights = ENCODER_WEIGHTS

    # Look into aux_params

    if model_name == 'UnetPlusPlus':
        model = smp.UnetPlusPlus(
            encoder_name=backbone,
            encoder_weights=encoder_weights,
            decoder_channels=decoder_channels,
            decoder_attention_type=None,
            classes=1,
            activation=activation if activation != 'None' else None,
        )
    elif model_name == 'PSPNet':
        model = smp.PSPNet(
            encoder_name=backbone,
            encoder_weights=encoder_weights,
            classes=1,
            activation=activation if activation != 'None' else None,
        )
    elif model_name == 'DeepLabV3Plus':
        model = smp.DeepLabV3Plus(
            encoder_name=backbone,
            encoder_weights=encoder_weights,
            classes=1,
            activation=activation if activation != 'None' else None,
        )
    else:
        raise ValueError(f"Model name {model_name} not recognized")

    model.to(device)
    loss = smp.losses.SoftBCEWithLogitsLoss()
    metrics = ["f1_score",
               "iou_score",
               "accuracy",
               ]
    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=lr),
    ])

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
        train_logs = train_epoch.run(train_loader, i)
        valid_logs = valid_epoch.run(valid_loader, i)
        scheduler.step(valid_logs["iou_score"])

        if valid_logs["iou_score"] > best_iou:
            best_iou = valid_logs["iou_score"]
            torch.save(model.state_dict(), model_save_path)

    return model


if __name__ == "__main__":
    smp_config = {
        'decoder_channels': [256, 128, 64, 32, 16],
        'backbone': 'efficientnet-b5',
        'epochs': 150,
        'use_epfl': True,
        'use_deepglobe': False,
        'augmentation_factor': 2,
        'transformation': 'advanced-satellite-augmentation',
        'resize': 416,
        'validation_size': 0.15,
        'seed': 42,
        'batch_size': 4,
        'lr': 0.0005,
        'device': 'cpu',
        'metric': 'iou_score',
        'model_save_path': '/content/gdrive/My Drive/models/UNetpp_EPFL_Adv_Aug'
    }
    train_smp(smp_config, data_dir="/Users/sebastian/University/Master/second_term/cil/road-segmentation/data")



