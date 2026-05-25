"""
One-step sampling and visualization for Drifting Models.
Includes FID computation and sample generation utilities.
"""

import argparse
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import numpy as np

from model import DriftDiT_models
from utils import (
    load_checkpoint,
    save_image_grid,
    visualize_samples,
    set_seed,
)


@torch.no_grad()
def generate_samples(
    model: nn.Module,
    num_samples: int,
    in_channels: int,
    img_size: int,
    num_classes: int,
    device: torch.device,
    labels: Optional[torch.Tensor] = None,
    alpha: float = 1.5,
    use_cfg: bool = True,
) -> torch.Tensor:
    """
    Generate samples using one-step inference.

    Args:
        model: The generator model
        num_samples: Number of samples to generate
        in_channels: Number of image channels
        img_size: Image size
        num_classes: Number of classes
        device: Device to generate on
        labels: Optional class labels. If None, generates uniformly across classes.
        alpha: CFG scale
        use_cfg: Whether to use classifier-free guidance

    Returns:
        Generated images, shape (num_samples, in_channels, img_size, img_size)
    """
    model.eval()

    # Sample noise
    z = torch.randn(num_samples, in_channels, img_size, img_size, device=device)

    # Generate labels if not provided
    if labels is None:
        labels = torch.randint(0, num_classes, (num_samples,), device=device)
    else:
        labels = labels.to(device)

    # Generate samples
    if use_cfg:
        x = model.forward_with_cfg(z, labels, alpha=alpha)
    else:
        alpha_tensor = torch.ones(num_samples, device=device)
        x = model(z, labels, alpha_tensor)

    return x.clamp(-1, 1)


@torch.no_grad()
def generate_class_grid(
    model: nn.Module,
    in_channels: int,
    img_size: int,
    num_classes: int,
    device: torch.device,
    samples_per_class: int = 10,
    alpha: float = 1.5,
) -> torch.Tensor:
    """
    Generate a grid of samples with each row being one class.

    Args:
        model: The generator model
        in_channels: Number of image channels
        img_size: Image size
        num_classes: Number of classes
        device: Device to generate on
        samples_per_class: Number of samples per class
        alpha: CFG scale

    Returns:
        Grid of samples, shape (num_classes * samples_per_class, in_channels, img_size, img_size)
    """
    model.eval()

    samples = []
    for c in range(num_classes):
        # Sample noise
        z = torch.randn(samples_per_class, in_channels, img_size, img_size, device=device)
        labels = torch.full((samples_per_class,), c, device=device, dtype=torch.long)

        # Generate
        x = model.forward_with_cfg(z, labels, alpha=alpha)
        samples.append(x)

    return torch.cat(samples, dim=0).clamp(-1, 1)


@torch.no_grad()
def generate_alpha_sweep(
    model: nn.Module,
    in_channels: int,
    img_size: int,
    device: torch.device,
    label: int = 0,
    alphas: list = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
    samples_per_alpha: int = 8,
) -> tuple:
    """
    Generate samples across different CFG alpha values.

    Args:
        model: The generator model
        in_channels: Number of image channels
        img_size: Image size
        device: Device to generate on
        label: Class label to generate
        alphas: List of alpha values to sweep
        samples_per_alpha: Number of samples per alpha value

    Returns:
        Tuple of (samples tensor, alpha values)
    """
    model.eval()

    # Use same noise for fair comparison
    z = torch.randn(samples_per_alpha, in_channels, img_size, img_size, device=device)
    labels = torch.full((samples_per_alpha,), label, device=device, dtype=torch.long)

    samples = []
    for alpha in alphas:
        x = model.forward_with_cfg(z, labels, alpha=alpha)
        samples.append(x)

    return torch.cat(samples, dim=0).clamp(-1, 1), alphas


def compute_fid_score(
    model: nn.Module,
    real_images: torch.Tensor,
    in_channels: int,
    img_size: int,
    num_classes: int,
    device: torch.device,
    num_samples: int = 10000,
    batch_size: int = 256,
    alpha: float = 1.5,
) -> float:
    """
    Compute FID score between generated and real images.

    Args:
        model: The generator model
        real_images: Real images for comparison
        in_channels: Number of image channels
        img_size: Image size
        num_classes: Number of classes
        device: Device to use
        num_samples: Number of samples to generate for FID
        batch_size: Batch size for generation
        alpha: CFG scale

    Returns:
        FID score
    """
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
        fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    except ImportError:
        print("torchmetrics not available, skipping FID computation")
        return float("nan")

    model.eval()

    # Add real images
    print("Processing real images for FID...")
    for i in range(0, len(real_images), batch_size):
        batch = real_images[i:i+batch_size].to(device)
        # Convert to [0, 1] range and replicate channels if grayscale
        batch = (batch + 1) / 2
        if in_channels == 1:
            batch = batch.repeat(1, 3, 1, 1)
        # Resize to 299x299 for Inception
        batch = torch.nn.functional.interpolate(batch, size=(299, 299), mode="bilinear", align_corners=False)
        fid_metric.update(batch, real=True)

    # Generate and add fake images
    print("Generating samples for FID...")
    num_generated = 0
    while num_generated < num_samples:
        current_batch = min(batch_size, num_samples - num_generated)
        samples = generate_samples(
            model,
            current_batch,
            in_channels,
            img_size,
            num_classes,
            device,
            alpha=alpha,
        )
        # Convert to [0, 1] range
        samples = (samples + 1) / 2
        if in_channels == 1:
            samples = samples.repeat(1, 3, 1, 1)
        samples = torch.nn.functional.interpolate(samples, size=(299, 299), mode="bilinear", align_corners=False)
        fid_metric.update(samples, real=False)
        num_generated += current_batch

    # Compute FID
    fid_score = fid_metric.compute().item()
    return fid_score


def sample_and_save(
    checkpoint_path: str,
    output_dir: str,
    dataset: str = "mnist",
    num_samples: int = 100,
    samples_per_class: int = 10,
    alpha: float = 1.5,
    seed: int = 42,
    compute_fid: bool = False,
):
    """
    Load model and generate samples.

    Args:
        checkpoint_path: Path to model checkpoint
        output_dir: Directory to save samples
        dataset: Dataset name ("mnist" or "cifar10")
        num_samples: Number of random samples to generate
        samples_per_class: Number of samples per class for grid
        alpha: CFG scale
        seed: Random seed
        compute_fid: Whether to compute FID score
    """
    set_seed(seed)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})

    # Determine model config
    if dataset.lower() == "mnist":
        model_name = config.get("model", "DriftDiT-Tiny")
        in_channels = 1
        img_size = 32
        num_classes = 10
    else:
        model_name = config.get("model", "DriftDiT-Small")
        in_channels = 3
        img_size = 32
        num_classes = 10

    # Create model
    model_fn = DriftDiT_models[model_name]
    model = model_fn(
        img_size=img_size,
        in_channels=in_channels,
        num_classes=num_classes,
    ).to(device)

    # Load weights (prefer EMA if available)
    if "ema" in checkpoint:
        print("Loading EMA weights...")
        model.load_state_dict(checkpoint["ema"])
    else:
        model.load_state_dict(checkpoint["model"])

    model.eval()
    print(f"Model loaded: {model_name}")

    # Generate class grid
    print(f"Generating class grid ({num_classes} classes x {samples_per_class} samples)...")
    grid = generate_class_grid(
        model,
        in_channels,
        img_size,
        num_classes,
        device,
        samples_per_class=samples_per_class,
        alpha=alpha,
    )
    grid_path = output_dir / f"class_grid_alpha{alpha:.1f}.png"
    save_image_grid(grid, str(grid_path), nrow=samples_per_class)
    print(f"Saved class grid to {grid_path}")

    # Generate random samples
    print(f"Generating {num_samples} random samples...")
    samples = generate_samples(
        model,
        num_samples,
        in_channels,
        img_size,
        num_classes,
        device,
        alpha=alpha,
    )
    samples_path = output_dir / f"random_samples_alpha{alpha:.1f}.png"
    nrow = int(np.ceil(np.sqrt(num_samples)))
    save_image_grid(samples, str(samples_path), nrow=nrow)
    print(f"Saved random samples to {samples_path}")

    # Alpha sweep
    print("Generating alpha sweep...")
    alphas = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    for label in range(min(3, num_classes)):  # First 3 classes
        sweep_samples, _ = generate_alpha_sweep(
            model,
            in_channels,
            img_size,
            device,
            label=label,
            alphas=alphas,
            samples_per_alpha=8,
        )
        sweep_path = output_dir / f"alpha_sweep_class{label}.png"
        save_image_grid(sweep_samples, str(sweep_path), nrow=8)
        print(f"Saved alpha sweep for class {label} to {sweep_path}")

    # Compute FID if requested
    if compute_fid:
        print("Computing FID score...")
        from torchvision import datasets, transforms

        if dataset.lower() == "mnist":
            transform = transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])
            test_dataset = datasets.MNIST("./data", train=False, download=True, transform=transform)
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])
            test_dataset = datasets.CIFAR10("./data", train=False, download=True, transform=transform)

        real_images = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))])

        fid = compute_fid_score(
            model,
            real_images,
            in_channels,
            img_size,
            num_classes,
            device,
            num_samples=10000,
            alpha=alpha,
        )
        print(f"FID Score: {fid:.2f}")

        # Save FID result
        with open(output_dir / "fid_score.txt", "w") as f:
            f.write(f"FID Score (alpha={alpha}): {fid:.2f}\n")

    print("\nSampling complete!")


def main():
    parser = argparse.ArgumentParser(description="Sample from Drifting Models")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./samples",
        help="Output directory for samples",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=["mnist", "cifar10"],
        help="Dataset (for determining model config)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of random samples to generate",
    )
    parser.add_argument(
        "--samples_per_class",
        type=int,
        default=10,
        help="Number of samples per class for grid",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.5,
        help="CFG scale",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--compute_fid",
        action="store_true",
        help="Whether to compute FID score",
    )

    args = parser.parse_args()

    sample_and_save(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        dataset=args.dataset,
        num_samples=args.num_samples,
        samples_per_class=args.samples_per_class,
        alpha=args.alpha,
        seed=args.seed,
        compute_fid=args.compute_fid,
    )


if __name__ == "__main__":
    main()
