

def get_config(config_name: str):

    if config_name == 'unetpp_b7':
        config = {
            'decoder_channels': [256, 128, 64, 32, 16],
            'backbone': 'efficientnet-b7',
            'epochs': 100,
            'use_epfl': True,
            'use_deepglobe': True,
            'augmentation_factor': 8,
            'transformation': 'advanced-satellite-augmentation-two',
            'resize': 416,
            'validation_size': 0.2,
            'seed': 42,
            'batch_size': 8,
            'lr': 0.001,
            'device': 'cuda',
            'metric': 'iou_score',
            'model_save_path': 'model_save/UNetpp_B7_Final',
            'model_name': 'UnetPlusPlus',
            'loss_function': 'SoftBCEWithLogitsLoss',
            'optimizer': 'AdamW',
            'show_val': False,
        }
    elif config_name == 'deeplabv3plus':
        config = {
            'decoder_channels': [256, 128, 64, 32, 16],
            'backbone': 'timm-regnetx_160',
            'epochs': 100,
            'use_epfl': True,
            'use_deepglobe': True,
            'augmentation_factor': 8,
            'transformation': 'advanced-satellite-augmentation-two',
            'resize': 416,
            'validation_size': 0.2,
            'seed': 42,
            'batch_size': 8,
            'lr': 0.001,
            'device': 'cuda',
            'metric': 'iou_score',
            'model_save_path': 'model_save/DeepLab_regnetx_final',
            'model_name': 'DeepLabV3Plus',
            'loss_function': 'SoftBCEWithLogitsLoss',
            'optimizer': 'AdamW',
            'show_val': False,
        }
    elif config_name == 'pspnet':
        config = {
            'decoder_channels': [256, 128, 64, 32, 16],
            'backbone': 'timm-resnest200e',
            'epochs': 100,
            'use_epfl': True,
            'use_deepglobe': True,
            'augmentation_factor': 4,
            'transformation': 'advanced-satellite-augmentation-two',
            'resize': 416,
            'validation_size': 0.2,
            'seed': 42,
            'batch_size': 8,
            'lr': 0.001,
            'device': 'cuda',
            'metric': 'iou_score',
            'model_save_path': 'model_save/Psp_resnet200e_final',
            'model_name': 'PSPNet',
            'loss_function': 'SoftBCEWithLogitsLoss',
            'optimizer': 'AdamW',
            'show_val': False,
        }
    else:
        raise ValueError(f'Unknown config name: {config_name}')
    return config
