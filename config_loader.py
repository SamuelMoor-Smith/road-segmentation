

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
            'device': 'cuda',
            'metric': 'accuracy',
            'model_save_path': 'model_save/UNetpp_B5_BS12',
            'model_name': 'UnetPlusPlus',  # DeepLabV3+, PspNet
            'loss_function': 'SoftBCEWithLogitsLoss',
            'optimizer': 'Adam',
            'show_val': False
        }
    else:
        raise ValueError(f'Unknown config name: {config_name}')
    return config
