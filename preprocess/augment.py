# # import albumentations as A


# def affine():
#     return A.Compose([
#         A.HorizontalFlip(p=0.5),
#         A.VerticalFlip(p=0.5),
#         A.RandomRotate90(p=0.5),
#         A.Transpose(p=0.5),
#     ], p=0.75)

import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random

def apply_transforms(image, mask, size=(384, 384), augment_probability=0.5):

    # Apply random affine transformations
    if random.random() < augment_probability:
        angle = random.uniform(-90, 90)  # degrees
        translate = [random.uniform(-image.size(2) * 0.2, image.size(2) * 0.2), random.uniform(-image.size(1) * 0.2, image.size(1) * 0.2)]
        scale = random.uniform(0.5, 1.5)
        shear = random.uniform(-10, 10)
        image = TF.affine(image, angle=angle, translate=translate, scale=scale, shear=shear)
        mask = TF.affine(mask, angle=angle, translate=translate, scale=scale, shear=shear)

    return image, mask
