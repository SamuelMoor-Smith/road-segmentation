#!/bin/bash
#SBATCH --job-name=road-segmentation    # Job name
#SBATCH --account=cil                   # Account name
#SBATCH --time=00:10:00                 # Time limit hh:mm:ss
#SBATCH --output=train.out              # Standard output and error log

# Load the modules environment
. /etc/profile.d/modules.sh

# Load necessary modules
module load cuda/12.4.1

# Set up and activate the virtual environment
cd ~/road-segmentation
python3 -m venv venv
source venv/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Run the Python script with command-line arguments
python train_smp.py --config 'deeplabv3plus'
