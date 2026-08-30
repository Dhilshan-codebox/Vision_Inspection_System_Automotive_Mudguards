import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_advanced_training_augmentations(image_size: int = 640) -> A.Compose:
    """
    Returns an Albumentations composition for advanced robust training.
    Specifically designed to counter False Positives related to dust/reflections
    and False Negatives related to subtle dents on curved surfaces.
    """
    return A.Compose([
        # Basic geometry (to prevent edge artifacts, FP-002)
        A.RandomResizedCrop(height=image_size, width=image_size, scale=(0.8, 1.0), p=1.0),
        A.HorizontalFlip(p=0.5),
        
        # Color & Lighting robustness (to counter FP-001)
        A.OneOf([
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05, p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0)
        ], p=0.7),
        
        # Noise & Artifact robustness
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            A.ISONoise(color_shift=0.01, intensity=(0.1, 0.5), p=1.0),
            A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0)
        ], p=0.4),
        
        # Blur (simulating motion or out-of-focus during conveyor belt movement)
        A.OneOf([
            A.MotionBlur(blur_limit=5, p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0)
        ], p=0.2),

        # Normalization
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

def get_validation_augmentations(image_size: int = 640) -> A.Compose:
    """
    Returns an Albumentations composition for validation (no destructive transforms).
    """
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
