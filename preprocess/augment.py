import albumentations as A


def transformations(method: str):
    if method == 'affine':
        return A.Compose([
                     A.HorizontalFlip(p=0.5),
                     A.VerticalFlip(p=0.5),
                     A.RandomRotate90(p=0.5),
                     A.Transpose(p=0.5),
                 ], p=0.75)
    elif method == 'spatial':
        return A.OneOf([
            A.RandomResizedCrop(384, 384, p=1, scale=(0.6, 0.9), ratio=(0.6, 0.9)),
            A.GridDistortion(p=1, distort_limit=0.4),
            A.ElasticTransform(p=1, alpha=10, sigma=50),
        ], p=0.75)
    elif method == 'appearance':
        return A.OneOf([
            A.RandomShadow(p=1, num_shadows_lower=1, num_shadows_upper=3, shadow_dimension=5),
            A.CoarseDropout(max_holes=8, max_height=50, max_width=50, min_holes=2, min_height=16, min_width=16,
                            fill_value=0, p=1),
            A.GaussNoise(p=1, var_limit=(0.1, 0.3))
        ], p=0.75)
    elif method == 'color':
        return A.OneOf([
            # A.RGBShift(r_shift_limit=50, g_shift_limit=50, b_shift_limit=50, p=1),
            A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.35, p=1),
            A.RandomGamma(p=1, gamma_limit=(80, 120)),
            A.ColorJitter(p=1)
        ], p=0.75)
    elif method == 'all':
        return A.Compose([
            A.OneOf([
                A.RandomRotate90(p=1),
                A.HorizontalFlip(p=1),
                A.VerticalFlip(p=1),
                A.Transpose(p=1),
                A.Rotate(limit=30, p=1)
            ], p=0.75),
            A.OneOf([
                A.RandomResizedCrop(384, 384, p=1, scale=(0.6, 0.9), ratio=(0.6, 0.9)),
                A.GridDistortion(p=1, distort_limit=0.4),
                A.ElasticTransform(p=1, alpha=10, sigma=50),
                # A.RandomResizedCrop(384, 384, p=1, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                # A.GridDistortion(p=1, distort_limit=0.2),
                # A.ElasticTransform(p=1, alpha=3, sigma=50),
                # A.Perspective(p=1, scale=(0.05, 0.1))
            ], p=0.75),
            A.OneOf([
                A.RandomShadow(p=1, num_shadows_lower=1, num_shadows_upper=3, shadow_dimension=5),
                A.CoarseDropout(max_holes=8, max_height=50, max_width=50, min_holes=2, min_height=16, min_width=16,
                            fill_value=0, p=1),
                A.GaussNoise(p=1, var_limit=(0.1, 0.3))
                # A.RandomShadow(p=1, num_shadows_lower=1, num_shadows_upper=3, shadow_dimension=5),
                # A.CoarseDropout(max_holes=8, max_height=32, max_width=32, min_holes=1, min_height=8, min_width=8,
                #                 fill_value=0, p=1),
                # A.GaussNoise(p=1, var_limit=(10.0, 30.0))
            ], p=0.75),
            A.OneOf([
                # A.RGBShift(r_shift_limit=50, g_shift_limit=50, b_shift_limit=50, p=1),
                A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.35, p=1),
                A.RandomGamma(p=1, gamma_limit=(80, 120)),
                A.ColorJitter(p=1)
                # A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=1),
                # A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
                # A.RandomGamma(p=1, gamma_limit=(80, 120)),
                # A.ColorJitter(p=1, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
            ], p=0.75),
            A.PadIfNeeded(min_height=384, min_width=384, p=1.0),
            A.Resize(height=384, width=384, p=1.0)
        ], p=1)

