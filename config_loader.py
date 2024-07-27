

def get_config(config_name: str):

    if config_name == 'unetpp_b5':
        config = {
            'decoder_channels': [256, 128, 64, 32, 16],
            'backbone': 'efficientnet-b5',
            'epochs': 300,
            'use_epfl': True,
            'use_deepglobe': True,
            'augmentation_factor': 10,
            'transformation': 'advanced-satellite-augmentation',
            'resize': 416,
            'validation_size': 0.15,
            'seed': 42,
            'batch_size': 12,
            'lr': 0.0005,
            'device': 'cpu',
            'metric': 'accuracy',
            'model_save_path': 'model_save/UNetpp_B5_BS12',
            'model_name': 'UnetPlusPlus',  # DeepLabV3+, PspNet
            'loss_function': 'SoftBCEWithLogitsLoss',
            'optimizer': 'Adam',
            'show_val': False
        }
    elif config_name == 'unetpp_b5_2':
        config = {
            'decoder_channels': [256, 128, 64, 32, 16],
            'backbone': 'efficientnet-b5',
            'epochs': 300,
            'use_epfl': True,
            'use_deepglobe': True,
            'augmentation_factor': 6,
            'transformation': 'advanced-satellite-augmentation',
            'resize': 416,
            'validation_size': 0.5,
            'seed': 42,
            'batch_size': 12,
            'lr': 0.0005,
            'device': 'cuda',
            'metric': 'accuracy',
            'model_save_path': 'model_save/UNetpp_B5_BS12',
            'model_name': 'UnetPlusPlus',  # DeepLabV3+, PspNet
            'loss_function': 'SoftBCEWithLogitsLoss',
            'optimizer': 'Adam',
            'show_val': False
        }
    elif config_name == 'unetpp_b5_3':
        config = {
            'decoder_channels': [256, 128, 64, 32, 16],
            'backbone': 'efficientnet-b5',
            'epochs': 300,
            'use_epfl': True,
            'use_deepglobe': True,
            'augmentation_factor': 6,
            'transformation': 'advanced-satellite-augmentation',
            'resize': 416,
            'validation_size': 0.5,
            'seed': 42,
            'batch_size': 12,
            'lr': 0.0005,
            'device': 'cuda',
            'metric': 'iou_score',
            'model_save_path': 'model_save/UNetpp_B5_BS12',
            'model_name': 'UnetPlusPlus',  # DeepLabV3+, PspNet
            'loss_function': 'SoftBCEWithLogitsLoss',
            'optimizer': 'Adam',
            'show_val': False
        }
    elif config_name == 'unetpp_b7':
        config = {
            'decoder_channels': [256, 128, 64, 32, 16],
            'backbone': 'efficientnet-b7',
            'epochs': 100,
            'use_epfl': True,
            'use_deepglobe': True,
            'augmentation_factor': 1,
            'transformation': 'advanced-satellite-augmentation-two',
            'resize': 416,
            'validation_size': 1,
            'seed': 42,
            'batch_size': 8,
            'lr': 0.0005,
            'device': 'cuda',
            'metric': 'iou_score',
            'model_save_path': 'model_save/UNetpp_B7_Full_Val',
            'model_name': 'UnetPlusPlus',
            'loss_function': 'SoftBCEWithLogitsLoss',
            'optimizer': 'AdamW',
            'show_val': False,
        }
    else:
        raise ValueError(f'Unknown config name: {config_name}')
    return config
