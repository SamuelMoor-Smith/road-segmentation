# CIL Road Segmentation Project
This project is completed as a part of the Computational Intelligence Lab course during the Spring 2024 semester. This GitHub repository contains the code for the Road Segmentation project.

## Installation

First things first, you must clone this repository on your local machine or cluster.

```bash
git clone https://github.com/SamuelMoor-Smith/road-segmentation.git
cd road-segmentation/
```

## Virtual Environment/Requirement Installation

Now, you must have all the required libraries installed. The easiest way to do this is with a virtual environment. You can either do this with conda or by:

```bash
python3 -m venv roadsegenv
source roadsegenv/bin/activate
pip install -r requirements.txt
```

## Usage

The training script for training a model using segmentation_models_pytorch is train_smp.py. The available models are UnetPlusPlus,
PSPNet and DeepLabV3Plus. The script uses the TrainEpoch and ValidEpoch from the deprecated utils module from smp
and thus we have copied the code here.

How to use:
    1. Create a config file in the config_loader.py file
    2. Activate a virtual environment with python version 3.10
    3. pip install -r requirements.txt
    4. Run the script with the following command:
    ```bash
    python train_smp.py --config unetpp_b5 --data_dir /path/to/data
    ```

In order to test your submission, run:

## Team/Contributors
- Sebastian Hönig
- Kerem Güra
- Samuel-Moor Smith
