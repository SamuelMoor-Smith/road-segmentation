import os
import torch
import numpy as np
from utils import image_to_patches, np_to_tensor
import cv2
from sklearn.model_selection import train_test_split
from PIL import Image
from glob import glob
import random
import torchvision.transforms.functional as TF

def apply_transforms(image, mask, augment_probability):

    # Apply random affine transformations
    if random.random() < augment_probability:
        angle = random.uniform(-10, 10)  # degrees
        translate = [random.uniform(-image.size(2) * 0.2, image.size(2) * 0.2), random.uniform(-image.size(1) * 0.2, image.size(1) * 0.2)]
        scale = random.uniform(0.9, 1.1)
        shear = random.uniform(-10, 10)

        image = TF.affine(image, angle=angle, translate=translate, scale=scale, shear=shear)
        mask = TF.affine(mask, angle=angle, translate=translate, scale=scale, shear=shear)

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
        images = load_all_from_path(os.path.join(self.data_dir, 'images'))[:, :, :, :3]
        masks = load_all_from_path(os.path.join(self.data_dir, 'groundtruth'))

        # Split into training and validation sets
        train_images, val_images, train_masks, val_masks = train_test_split(
            images, masks, test_size=self.test_size, random_state=self.random_state
        )

        self.x = train_images if self.is_train else val_images
        self.y = train_masks if self.is_train else val_masks
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
                augment_probability=0.5,
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
