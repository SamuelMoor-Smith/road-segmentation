"""This script downloads the DeepGlobe dataset and extracts the images and masks into the data directory."""

import os

import numpy as np
from kaggle.api.kaggle_api_extended import KaggleApi
from tqdm import tqdm
import cv2
import shutil

api = KaggleApi()
api.authenticate()

# Change these paths to your own data directory and the directory where you want to save the images and masks
INIT_DIR = '/Users/sebastian/University/Master/second_term/cil/road-segmentation/data/deepglobe'
IMG_DIR = '/Users/sebastian/University/Master/second_term/cil/road-segmentation/data/training/images/deepglobe'
MASK_DIR = '/Users/sebastian/University/Master/second_term/cil/road-segmentation/data/training/groundtruth/deepglobe'

FILTERED_IMG_DIR = '/Users/sebastian/University/Master/second_term/cil/road-segmentation/data/training/images/deepglobe_filtered'
FILTERED_MASK_DIR = '/Users/sebastian/University/Master/second_term/cil/road-segmentation/data/training/groundtruth/deepglobe_filtered'

REJECTED_IMG_DIR = '/Users/sebastian/University/Master/second_term/cil/road-segmentation/data/training/images/deepglobe_rejected'
REJECTED_MASK_DIR = '/Users/sebastian/University/Master/second_term/cil/road-segmentation/data/training/groundtruth/deepglobe_rejected'

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(MASK_DIR, exist_ok=True)

os.makedirs(FILTERED_IMG_DIR, exist_ok=True)
os.makedirs(FILTERED_MASK_DIR, exist_ok=True)

os.makedirs(REJECTED_IMG_DIR, exist_ok=True)
os.makedirs(REJECTED_MASK_DIR, exist_ok=True)

# Download the dataset - its a bit messy, so just download this and then only take the train and val folders and
# remove all others and put the imgs into the INIT_DIR

# api.dataset_download_files('balraj98/deepglobe-road-extraction-dataset', path=INIT_DIR, unzip=True)

# Print the amount of files in the directory

print(f"Total images and masks downloaded: {len(os.listdir(INIT_DIR))/2}")

all_files = list(sorted(os.listdir(INIT_DIR)))
img_files = [f for f in all_files if 'sat' in f]
mask_files = [f for f in all_files if 'mask' in f]

# The images are of size 1024x1024, so we get 9 patches of size 400x400 from each image

i = 0
"""for mask, img in tqdm(zip(mask_files, img_files), total=len(img_files)):
    locations = [(0, 0), (0, 312), (0, 624), (312, 0), (312, 312), (312, 624), (624, 0), (624, 312), (624, 624)]
    img_path = os.path.join(INIT_DIR, img)
    mask_path = os.path.join(INIT_DIR, mask)

    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    for (x, y) in locations:
        img_patch = img[x:x+400, y:y+400]
        mask_patch = mask[x:x+400, y:y+400]

        cv2.imwrite(os.path.join(IMG_DIR, f'deeplglobe_img_{i}.png'), img_patch)
        cv2.imwrite(os.path.join(MASK_DIR, f'deepglobe_mask_{i}.png'), mask_patch)
        i += 1
"""

print(f"Total images and masks extracted: {len(os.listdir(IMG_DIR))}")

# Filter the files that are mostly black

sliced_img_files = list(sorted(os.listdir(IMG_DIR)))
sliced_mask_files = list(sorted(os.listdir(MASK_DIR)))

i = 0
for img, mask in tqdm(zip(sliced_img_files, sliced_mask_files), total=len(sliced_img_files)):
    img_path = os.path.join(IMG_DIR, img)
    mask_path = os.path.join(MASK_DIR, mask)

    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if np.mean(mask) > 10:
        shutil.move(img_path, os.path.join(FILTERED_IMG_DIR, f'deeplglobe_img_{i}.png'))
        shutil.move(mask_path, os.path.join(FILTERED_MASK_DIR, f'deepglobe_mask_{i}.png'))
    else:
        shutil.move(img_path, os.path.join(REJECTED_IMG_DIR, f'deeplglobe_img_{i}.png'))
        shutil.move(mask_path, os.path.join(REJECTED_MASK_DIR, f'deepglobe_mask_{i}.png'))
    i += 1


print(f"Total images and masks filtered: {len(os.listdir(FILTERED_IMG_DIR))}")
print(f"Total images and masks rejected: {len(os.listdir(REJECTED_IMG_DIR))}")




