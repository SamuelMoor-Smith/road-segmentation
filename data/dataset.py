import os
import torch
import numpy as np
from utils import image_to_patches, np_to_tensor
import cv2
from PIL import Image
from glob import glob
from preprocess import preprocess


class ImageDataset(torch.utils.data.Dataset):
    # dataset class that deals with loading the data and making it available by index.

    def __init__(self, data_dir, is_train, device, use_patches=True, resize_to=(400, 400)):
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
            x, y = preprocess.apply_transforms(
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