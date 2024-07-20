import albumentations as A
import torchvision.transforms.functional as TF
import random
from albumentations.pytorch import ToTensorV2


def smp_get_preprocessing(preprocessing_fn):
    # from https://github.com/qubvel-org/segmentation_models.pytorch/blob/main/examples/cars%20segmentation%20(camvid).ipynb
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        A.Lambda(image=preprocessing_fn),
        ToTensorV2()
    ]
    return A.Compose(_transform)


def augment(img_size: int = 384, augmentation: str = 'standard'):
    standard = A.Compose([
        A.PadIfNeeded(min_height=img_size, min_width=img_size, p=1.0),
        A.Resize(height=img_size, width=img_size)])
    if augmentation == 'validation':
        return standard
    elif augmentation == 'minimal':
        return A.Compose([
            standard,
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Transpose(p=0.5),
        ], p=1)


def to_tensor():
    return A.Compose([ToTensorV2()])


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
