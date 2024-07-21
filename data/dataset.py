import os
import torch
import numpy as np
import cv2
from PIL import Image
from glob import glob
from preprocess import augment
import random


def preprocess_mask(mask: torch.FloatTensor) -> torch.FloatTensor:
    "This function takes a mask and converts it to a uint8 and float32,"
    "then sets the bits that are 1.0 and 0.0, respectively."
    mask = (mask > 0).astype(np.uint8) * 255
    mask = mask.astype(np.float32)
    mask[mask == 255.0] = 1.0
    mask[mask == 0.0] = 0.0
    return mask


class ImageDataset(torch.utils.data.Dataset):
    # dataset class that deals with loading the data and making it available by index.

    def __init__(self, data_dir: str, is_train: bool, device: str, use_epfl: bool = False, use_deepglobe: bool = False,
                 validation_size: float = 0.15, seed: int = 42, transforms=None, preprocess=None,
                 augmentation_factor: int = 1, resize: int = 384):
        self.data_dir = data_dir
        self.is_train = is_train
        self.device = device
        self.use_epfl = use_epfl
        self.use_deepglobe = use_deepglobe
        self.validation_size = validation_size
        self.seed = seed
        self.transforms = augment.augment(resize, transforms) if transforms is not None else None
        self.preprocess = preprocess
        self.augmentation_factor = augmentation_factor # By how much we increase dataset size

        eth_files = self._get_data()  # make distinction because we only want val img paths from eth imgs
        if self.is_train:
            additional = self._get_external_data()
            self.filenames = eth_files + additional
        else:
            self.filenames = eth_files

    def _get_data(self):
        image_dir = os.path.join(self.data_dir, 'training', 'images', 'eth')
        filenames = [os.path.join(image_dir, fname) for fname in sorted(os.listdir(image_dir))]

        filenames = [f for f in filenames if f.endswith('.png')]

        random.seed(42)
        random.shuffle(filenames)

        train, val = filenames[:int(len(filenames) * (1 - self.validation_size))], filenames[int(len(filenames) * (1 - self.validation_size)):]

        if self.is_train:
            train = train * self.augmentation_factor
            return train
        else:
            return val

    def _get_external_data(self):
        filenames = []
        if self.use_epfl:
            epfl_dir = os.path.join(self.data_dir, 'training', 'images', 'epfl')
            filenames += [os.path.join(epfl_dir, fname) for fname in sorted(os.listdir(epfl_dir))]
        if self.use_deepglobe:
            dg_dir = os.path.join(self.data_dir, 'training', 'images', 'deepglobe_filtered')
            filenames += [os.path.join(dg_dir, fname) for fname in sorted(os.listdir(dg_dir))]

        filenames = [f for f in filenames if f.endswith('.png')]
        return filenames

    def __getitem__(self, item: int):
        img_path = self.filenames[item]
        orig_image = cv2.imread(img_path)
        orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)

        image = orig_image # maybe need to kick last channel
        mask_path = img_path.replace('images', 'groundtruth')

        if 'dg' in mask_path:
            mask_path = mask_path.replace('sat', 'mask')

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) # or unchanged - anschauen ob preprocess nötig
        mask = preprocess_mask(mask)
        if self.transforms is not None:
            transformation = self.transforms(image=image, mask=mask)
            image = transformation['image']
            mask = transformation['mask']

        if self.preprocess is not None:
            preprocessed = self.preprocess(image=image, mask=mask)
            image = preprocessed['image']
            mask = preprocessed['mask']
        else:
            tensors = augment.to_tensor()(image=image, mask=mask)
            image = tensors['image']
            mask = tensors['mask']
        image = image.type(torch.float32) # check if I need any of that
        mask = mask.type(torch.float32)
        if len(mask.size()) == 2:
            mask = torch.unsqueeze(mask, 0)
        return image, mask, orig_image

    #return self._preprocess(np_to_tensor(self.x[item], self.device), np_to_tensor(self.y[[item]], self.device))

    def __len__(self):
        return len(self.filenames)


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir:str, transforms=None, preprocess=None, seed:int = 42, resize:int = 384):
        self.data_dir = os.path.join(data_dir, 'test', 'images')
        self.transforms = augment.augment(resize, transforms) if transforms is not None else None
        self.preprocess = preprocess
        self.seed = seed
        self.resize = resize

        filenames = list(sorted(os.listdir(self.data_dir)))
        self.filenames = [os.path.join(self.data_dir, fname) for fname in filenames if fname.endswith('.png')]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, item: int):
        img_path = self.filenames[item]
        orig_image = cv2.imread(img_path)
        orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        image = orig_image

        if self.transforms is not None:
            transformed = self.transforms(image=image)
            image = transformed["image"]
        # apply preprocessing
        if self.preprocess is not None:
            sample = self.preprocess(image=image)
            image = sample['image']
        else:
            tensors = augment.to_tensor()(image=image)
            image = tensors["image"]
        image = image.to(torch.float32)

        return image

