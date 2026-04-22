import os
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets
from torchvision.models import Inception_V3_Weights, inception_v3


@dataclass
class EvalConfig:
    generated_dir: str = "./inference"
    dataset_dir: str = "./dataset"
    split: str = "train"  # "train" or "test"
    batch_size: int = 64
    num_workers: int = 2
    max_generated: int = 0  # 0 means use all generated images
    max_real: int = 0  # 0 means match generated sample count
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


CONFIG = EvalConfig()


class GeneratedImageDataset(Dataset):
    def __init__(self, image_dir: str):
        if not os.path.isdir(image_dir):
            raise FileNotFoundError(f"Generated image directory not found: {image_dir}")

        valid_ext = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        files = [
            os.path.join(image_dir, name)
            for name in sorted(os.listdir(image_dir))
            if os.path.splitext(name.lower())[1] in valid_ext
        ]

        if not files:
            raise ValueError(f"No image files found in directory: {image_dir}")

        self.files: List[str] = files

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.files[idx]
        with Image.open(path) as img:
            img = img.convert("L")
            arr = np.asarray(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).unsqueeze(0)
        return tensor


class RealMNISTDataset(Dataset):
    def __init__(self, dataset_dir: str, split: str):
        train = split == "train"
        self.dataset = datasets.MNIST(root=dataset_dir, train=train, download=True)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img, _ = self.dataset[idx]
        arr = np.asarray(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).unsqueeze(0)
        return tensor


def build_inception(device: torch.device) -> nn.Module:
    weights = Inception_V3_Weights.DEFAULT
    model = inception_v3(weights=weights, aux_logits=True)
    model.fc = nn.Identity()
    model.eval()
    model.to(device)
    return model


def preprocess_for_inception(batch: torch.Tensor) -> torch.Tensor:
    # Convert grayscale [B,1,H,W] in [0,1] to normalized ImageNet-like [B,3,299,299].
    batch = batch.repeat(1, 3, 1, 1)
    batch = F.interpolate(batch, size=(299, 299), mode="bilinear", align_corners=False)

    mean = torch.tensor([0.485, 0.456, 0.406], device=batch.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=batch.device).view(1, 3, 1, 1)
    return (batch - mean) / std


@torch.no_grad()
def extract_features(
    data_loader: DataLoader,
    feature_net: nn.Module,
    device: torch.device,
) -> np.ndarray:
    all_features: List[np.ndarray] = []
    for batch in data_loader:
        batch = batch.to(device, non_blocking=True)
        batch = preprocess_for_inception(batch)
        features = feature_net(batch)
        if isinstance(features, tuple):
            features = features[0]
        all_features.append(features.detach().cpu().numpy())

    return np.concatenate(all_features, axis=0)


def covariance_sqrt_trace(sigma1: np.ndarray, sigma2: np.ndarray) -> float:
    # Uses eigendecomposition of symmetric PSD matrices for numerical stability.
    eigvals1, eigvecs1 = np.linalg.eigh(sigma1)
    eigvals1 = np.clip(eigvals1, a_min=0.0, a_max=None)
    sqrt_sigma1 = eigvecs1 @ np.diag(np.sqrt(eigvals1)) @ eigvecs1.T

    middle = sqrt_sigma1 @ sigma2 @ sqrt_sigma1
    middle = (middle + middle.T) * 0.5
    eigvals_mid, _ = np.linalg.eigh(middle)
    eigvals_mid = np.clip(eigvals_mid, a_min=0.0, a_max=None)
    return float(np.sum(np.sqrt(eigvals_mid)))


def calculate_fid(features_real: np.ndarray, features_gen: np.ndarray) -> float:
    mu_real = np.mean(features_real, axis=0)
    mu_gen = np.mean(features_gen, axis=0)

    sigma_real = np.cov(features_real, rowvar=False)
    sigma_gen = np.cov(features_gen, rowvar=False)

    mean_diff = mu_real - mu_gen
    mean_norm = mean_diff @ mean_diff

    tr_cov_sqrt = covariance_sqrt_trace(sigma_real, sigma_gen)

    fid = mean_norm + np.trace(sigma_real) + np.trace(sigma_gen) - (2.0 * tr_cov_sqrt)
    return float(fid)


def maybe_subset(dataset: Dataset, max_items: int, seed: int) -> Dataset:
    if max_items <= 0 or max_items >= len(dataset):
        return dataset

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:max_items].tolist()
    return Subset(dataset, indices)


def main() -> None:
    config = CONFIG

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    device = torch.device(config.device)

    gen_dataset = GeneratedImageDataset(config.generated_dir)
    if config.max_generated > 0:
        gen_dataset = maybe_subset(gen_dataset, config.max_generated, config.seed)

    real_dataset = RealMNISTDataset(config.dataset_dir, config.split)

    if config.max_real > 0:
        real_target_count = config.max_real
    else:
        real_target_count = len(gen_dataset)

    real_dataset = maybe_subset(real_dataset, real_target_count, config.seed)

    if len(gen_dataset) < 2 or len(real_dataset) < 2:
        raise ValueError("Need at least 2 generated and 2 real images to compute covariance for FID.")

    gen_loader = DataLoader(
        gen_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    real_loader = DataLoader(
        real_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    feature_net = build_inception(device)

    print(f"Extracting generated features from {len(gen_dataset)} images...")
    gen_features = extract_features(gen_loader, feature_net, device)

    print(f"Extracting real MNIST features from {len(real_dataset)} images ({config.split} split)...")
    real_features = extract_features(real_loader, feature_net, device)

    fid = calculate_fid(real_features, gen_features)

    print("\nFID evaluation complete")
    print(f"Generated dir: {config.generated_dir}")
    print(f"MNIST dir: {config.dataset_dir}")
    print(f"Real split: {config.split}")
    print(f"Generated samples used: {len(gen_dataset)}")
    print(f"Real samples used: {len(real_dataset)}")
    print(f"FID: {fid:.6f}")


if __name__ == "__main__":
    main()
