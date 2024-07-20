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
    elif augmentation == 'advanced-satellite-augmentation':
        A.Compose([
            A.OneOf([
                A.RandomRotate90(p=1),
                A.HorizontalFlip(p=1),
                A.VerticalFlip(p=1),
                A.Transpose(p=1),
                A.Rotate(limit=45, p=1)  # Random rotations up to 45 degrees
            ], p=0.75),
            A.OneOf([
                A.RandomResizedCrop(img_size, img_size, p=1, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                A.GridDistortion(p=1, distort_limit=0.2),
                A.ElasticTransform(p=1, alpha=50, sigma=50 * 0.05, alpha_affine=50 * 0.03)
            ], p=0.75),
            A.OneOf([
                A.RandomShadow(p=1, num_shadows_lower=1, num_shadows_upper=3, shadow_dimension=5),
                A.CoarseDropout(max_holes=8, max_height=32, max_width=32, min_holes=1, min_height=8, min_width=8,
                                fill_value=0, p=1),
                A.GaussNoise(p=1, var_limit=(10.0, 30.0))
            ], p=0.75),
            A.OneOf([
                A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=1),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
                A.RandomGamma(p=1, gamma_limit=(80, 120)),
                A.ColorJitter(p=1)
            ], p=0.75),
            standard
        ], p=1)
    elif augmentation == 'advanced-satellite-augmentation-two':
        A.Compose([
            A.OneOf([
                A.RandomRotate90(p=1),
                A.HorizontalFlip(p=1),
                A.VerticalFlip(p=1),
                A.Transpose(p=1),
                A.Rotate(limit=30, p=1)
            ], p=0.75),
            A.OneOf([
                A.RandomResizedCrop(img_size, img_size, p=1, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                A.GridDistortion(p=1, distort_limit=0.2),
                A.ElasticTransform(p=1, alpha=50, sigma=50 * 0.05, alpha_affine=50 * 0.03),
                A.Perspective(p=1, scale=(0.05, 0.1))
            ], p=0.75),
            A.OneOf([
                A.RandomShadow(p=1, num_shadows_lower=1, num_shadows_upper=3, shadow_dimension=5),
                A.CoarseDropout(max_holes=8, max_height=32, max_width=32, min_holes=1, min_height=8, min_width=8,
                                fill_value=0, p=1),
                A.GaussNoise(p=1, var_limit=(10.0, 30.0))
            ], p=0.75),
            A.OneOf([
                A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=1),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
                A.RandomGamma(p=1, gamma_limit=(80, 120)),
                A.ColorJitter(p=1, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
            ], p=0.75),
            standard
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
