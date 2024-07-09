import os
import torch
import numpy as np
from utils import image_to_patches, np_to_tensor
import cv2
from sklearn.model_selection import train_test_split
from PIL import Image
from glob import glob
import random
from torchvision import transforms


def apply_transforms(image, mask, augment_probability=0.75):
    # Apply random affine transformations
    random_val = random.random()
    if random_val < augment_probability:
        if random_val < 0.25:
            image = transforms.functional.hflip(image)
            mask = transforms.functional.hflip(mask)

        # Random vertical flip
        elif random_val > 0.25 and random_val < 0.5:
            image = transforms.functional.vflip(image)
            mask = transforms.functional.vflip(mask)
        # Random rotation
        else:
            angle = random.uniform(-10, 10)
            image = transforms.functional.rotate(image, angle)
            mask = transforms.functional.rotate(mask, angle)
    return image, mask


class ImageDataset(torch.utils.data.Dataset):
    # dataset class that deals with loading the data and making it available by index.

    def __init__(self, data_dir, is_train, device, use_patches=True, resize_to=(400, 400), test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.data_dir = data_dir
        self.is_train = is_train
        self.device = device
        self.use_patches = use_patches
        self.resize_to=resize_to
        self.x, self.y, self.n_samples = None, None, None
        self._load_data()

    def _load_data(self):  # not very scalable, but good enough for now
        if self.is_train:
            images = load_all_from_path(os.path.join(self.data_dir, 'images', 'eth'))[:, :, :, :3]
            masks = load_all_from_path(os.path.join(self.data_dir, 'groundtruth', 'eth'))

            images_epfl = load_all_from_path(os.path.join(self.data_dir, 'images', 'epfl'))
            masks_epfl = load_all_from_path(os.path.join(self.data_dir, 'groundtruth', 'epfl'))

            images = np.concatenate([images, images_epfl], 0)
            masks = np.concatenate([masks, masks_epfl], 0)
        else:
            images = load_all_from_path(os.path.join(self.data_dir, 'images', 'val'))[:, :, :, :3]
            masks = load_all_from_path(os.path.join(self.data_dir, 'groundtruth', 'val'))
        # Split into training and validation sets
        #train_images, val_images, train_masks, val_masks = train_test_split(
        #    images, masks, test_size=self.test_size, random_state=self.random_state
        #)

        #self.x = train_images if self.is_train else val_images
        #self.y = train_masks if self.is_train else val_masks
        self.x = images
        self.y = masks

        if self.use_patches:  # split each image into patches
            self.x, self.y = image_to_patches(self.x, self.y)
        elif self.resize_to != (self.x.shape[1], self.x.shape[2]):  # resize images
            self.x = np.stack([cv2.resize(img, dsize=self.resize_to) for img in self.x], 0)
            self.y = np.stack([cv2.resize(mask, dsize=self.resize_to) for mask in self.y], 0)

        self.x = np.moveaxis(self.x, -1, 1)  # pytorch works with CHW format instead of HWC
        self.n_samples = len(self.x)

    def _preprocess(self, x, y):
        if self.is_train:
            x, y = apply_transforms(
                x,
                y,
                augment_probability=0.75,
            )

        return x, y

    def __getitem__(self, item):
        return self._preprocess(np_to_tensor(self.x[item], self.device), np_to_tensor(self.y[[item]], self.device))

    def __len__(self):
        return self.n_samples


def load_all_from_path(path):
    # loads all HxW .pngs contained in path as a 4D np.array of shape (n_images, H, W, 3)
    # images are loaded as floats with values in the interval [0., 1.]
    return np.stack([np.array(Image.open(f)) for f in sorted(glob(path + '/*.png'))]).astype(np.float32) / 255.
