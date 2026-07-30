"""
OAI radiograph dataset loader (ImageFolder-style: <root>/<split>/<KL grade>/*.png).
Manuscript Section 3.1: images resized to 224x224, ImageNet normalization,
random flip/rotation/jitter/affine augmentation on the training split only.
"""
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(image_size: int = 224, train: bool = True):
    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def power_oversample_weights(targets, beta: float = 0.7):
    """Sampling weight ~ n_c^-beta (manuscript Section 3.1)."""
    counts = {}
    for t in targets:
        counts[t] = counts.get(t, 0) + 1
    class_weight = {c: n ** (-beta) for c, n in counts.items()}
    return [class_weight[t] for t in targets]


def build_dataloader(root: str, image_size: int = 224, batch_size: int = 32,
                      train: bool = True, oversample_beta: float = 0.7):
    ds = datasets.ImageFolder(root, transform=build_transforms(image_size, train))
    if train:
        weights = power_oversample_weights(ds.targets, beta=oversample_beta)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        return DataLoader(ds, batch_size=batch_size, sampler=sampler)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)
