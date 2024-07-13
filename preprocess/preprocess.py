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
