"""
The ResNet model is trained on 224x224 images. The original images are 400x400.
This script extracts four 224x224 patches from the original images and saves them in a new directory.
We do this for both training and test images and the training masks.
"""

import os
import cv2

# Change this location whether you want to extract patches from the training or test images
inp_dir = '/Users/sebastian/University/Master/second_term/cil/road-segmentation/data/training/images/epfl'
outp_dir = '/Users/sebastian/University/Master/second_term/cil/road-segmentation/data/training/224_patches/images/epfl'


output_images_dir = os.path.join(outp_dir, 'images_patches')

os.makedirs(output_images_dir, exist_ok=True)
#os.makedirs(output_masks_dir, exist_ok=True)

for img_name in os.listdir(inp_dir):
    # Skip the .DS_Store file
    if img_name.startswith('.'):
        continue
    img_path = os.path.join(inp_dir, img_name)
    #mask_path = os.path.join(masks_dir, img_name)

    img = cv2.imread(img_path)
    #mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    positions = [(0, 0), (0, 400 - 224), (400 - 224, 0), (400 - 224, 400 - 224)]

    for i in range(len(positions)):
        top_left_x, top_left_y = positions[i]
        img_patch = img[top_left_y:top_left_y+224, top_left_x:top_left_x+224]
        #mask_patch = mask[top_left_y:top_left_y+224, top_left_x:top_left_x+224]

        cv2.imwrite(os.path.join(output_images_dir, f'{img_name[:-4]}_{i}.png'), img_patch)
        #cv2.imwrite(os.path.join(output_masks_dir, f'{img_name[:-4]}_{i}.png'), mask_patch)
