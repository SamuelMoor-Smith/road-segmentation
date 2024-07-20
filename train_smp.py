import segmentation_models_pytorch as smp
from preprocess.augment import smp_get_preprocessing
import torch
from torch.utils.data import DataLoader
from data.dataset import ImageDataset, TestDataset
from smp_utils import TrainEpoch, ValidEpoch

ENCODER_WEIGHTS = 'imagenet'
#DATA_DIR = "/Users/sebastian/University/Master/second_term/cil/road-segmentation/data/training"


def train_smp(decoder_channels: list[int], backbone: str, device: str, n_epochs: int, data_dir: str):
    preprocessing_fn = smp.encoders.get_preprocessing_fn(backbone, ENCODER_WEIGHTS)
    preprocessing_fn = smp_get_preprocessing(preprocessing_fn)
    encoder_weights = ENCODER_WEIGHTS

    model = smp.UnetPlusPlus(
        encoder_name=backbone,
        encoder_weights=encoder_weights,
        decoder_channels=decoder_channels,
        decoder_attention_type=None,
        classes=1,
        activation='sigmoid',
    )

    model.to(device)
    loss = smp.losses.SoftBCEWithLogitsLoss()
    metrics = ["f1_score",
               "iou_score",
               "accuracy",
               ]
    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=0.0005),
    ])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', verbose=True) # can play around with patience, factor, etc.

    train_dataset = ImageDataset(
        data_dir=data_dir,
        is_train=True,
        device=device,
        use_epfl=False,
        use_deepglobe=False,
        validation_size=0.15,
        seed=42,
        transforms='minimal',
        preprocess=preprocessing_fn,
        augmentation_factor=1,
        resize=416)

    valid_dataset = ImageDataset(
        data_dir=data_dir,
        is_train=False,
        device=device,
        use_epfl=False,
        use_deepglobe=False,
        validation_size=0.15,
        seed=42,
        transforms='validation',
        preprocess=preprocessing_fn,
        augmentation_factor=1,
        resize=416)

    train_loader = DataLoader(train_dataset,
                              batch_size=4,
                              shuffle=True,
                              num_workers=0,
                              pin_memory=True,
                              worker_init_fn=42,
                              )
    valid_loader = DataLoader(valid_dataset,
                              batch_size=4,
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

    for i in range(n_epochs):
        print(f"Epoch {i+1}/{n_epochs}")
        train_logs = train_epoch.run(train_loader, i)
        valid_logs = valid_epoch.run(valid_loader, i)
        scheduler.step(valid_logs["iou_score"])

    return model


if __name__ == "__main__":
    decoder_channels = [256, 128, 64, 32, 16]
    backbone = 'efficientnet-b5'
    device = 'cpu'
    epochs = 5
    train_smp(decoder_channels, backbone, device, epochs, data_dir="/Users/sebastian/University/Master/second_term/cil/road-segmentation/data")


