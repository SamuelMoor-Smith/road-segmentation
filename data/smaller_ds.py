import os

import numpy as np
from tqdm import tqdm
import cv2
import shutil

IMG_DIR = '/Users/sebastian/University/Master/second_term/cil/road-segmentation/data/training/images/deepglobe_filtered'
MASK_DIR = '/Users/sebastian/University/Master/second_term/cil/road-segmentation/data/training/groundtruth/deepglobe_filtered'

REJ_IMG_DIR = '/Users/sebastian/University/Master/second_term/cil/road-segmentation/data/training/images/deepglobe_rejected'
REJ_MASK_DIR = '/Users/sebastian/University/Master/second_term/cil/road-segmentation/data/training/groundtruth/deepglobe_rejected'

os.makedirs(REJ_IMG_DIR, exist_ok=True)
os.makedirs(REJ_MASK_DIR, exist_ok=True)

sliced_img_files = list(sorted(os.listdir(IMG_DIR)))
sliced_mask_files = list(sorted(os.listdir(MASK_DIR)))

# remove all non png files
sliced_img_files = [f for f in sliced_img_files if f.endswith('.png')]
sliced_mask_files = [f for f in sliced_mask_files if f.endswith('.png')]

for img_file, mask_file in tqdm(zip(sliced_img_files, sliced_mask_files), total=len(sliced_img_files)):
    img_path = os.path.join(IMG_DIR, img_file)
    mask_path = os.path.join(MASK_DIR, mask_file)

    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if np.mean(mask) < 34:
        shutil.move(img_path, os.path.join(REJ_IMG_DIR, img_file))
        shutil.move(mask_path, os.path.join(REJ_MASK_DIR, mask_file))