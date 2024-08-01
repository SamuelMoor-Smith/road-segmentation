"""
Main testing and submission script for testing a model using segmentation_models_pytorch. The available models are UnetPlusPlus,
PSPNet and DeepLabV3Plus and are available on the public drive: ___. 

How to use:
    1. Create a config file in the config_loader.py file
    2. Activate a virtual environment with python version 3.10
    3. pip install -r requirements.txt
    4. Run the script with the following command:
        python run_model_create_submission_smp.py --config unetpp_b5 --data_dir /path/to/data
"""

from submit_smp import make_submission
import torch

ENCODER_WEIGHTS = 'imagenet'
#DATA_DIR = "/Users/sebastian/University/Master/second_term/cil/road-segmentation/data/training"

def create_model(config, data_dir: str):
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

    model.to(device)

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", default="unetpp_b5")
    parser.add_argument("--model_filename", default="___") # In public drive
    parser.add_argument("--data_dir", default="/home/shoenig/road-segmentation/data")
    parser.add_argument("--submission_path", default="/home/shoenig/road-segmentation/submission_for_model")

    args = parser.parse_args()

    PUBLIC_DRIVE_PATH = ____

    config = get_config(args.config)
    data_dir = args.data_dir
    model_filename = args.model_filename
    submission_path = args.submission_path

    create_model(config, data_dir=data_dir)
    state_dict = torch.load(PUBLIC_DRIVE_PATH + model_filename)
    model.load_state_dict(state_dict)
    make_submission(model, config, 'data_dir', submission_path)
