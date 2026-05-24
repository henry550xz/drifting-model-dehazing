import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


FOG_PRESETS: Dict[str, Dict[str, Tuple[float, float]]] = {
    "mild": {
        "beta_range": (1.0, 2.8),
        "a_range": (0.75, 1.0),
        "blur_sigma_range": (0.0, 0.15),
    },
    "medium": {
        "beta_range": (2.0, 4.5),
        "a_range": (0.75, 1.0),
        "blur_sigma_range": (0.1, 0.6),
    },
    "heavy": {
        "beta_range": (3.5, 7.0),
        "a_range": (0.75, 1.0),
        "blur_sigma_range": (0.5, 1.8),
    },
}


def resolve_fog_config(dataset: str, fog_preset: Optional[str]) -> Tuple[str, Dict[str, Tuple[float, float]]]:
    """Resolve dataset-aware fog preset while keeping old MNIST defaults."""
    name = dataset.lower()
    resolved = fog_preset
    if resolved is None:
        resolved = "mild" if name in ("cifar", "cifar10") else "heavy"
    config = FOG_PRESETS[resolved].copy()
    if name == "mnist" and resolved == "heavy":
        # Preserve the original MNIST toy haze while still exposing preset names.
        config["beta_range"] = (2.0, 4.5)
        config["a_range"] = (0.85, 1.0)
        config["blur_sigma_range"] = (0.5, 1.8)
    return resolved, config


def _make_depth_map(
    batch_size: int,
    height: int,
    width: int,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Random smooth depth maps in [0, 1] for artificial fog."""
    d = torch.zeros(batch_size, 1, height, width, device=device)
    weight, total = 1.0, 0.0
    for res in (2, 4, 8):
        octave = torch.rand(batch_size, 1, res, res, device=device, generator=generator)
        d += weight * F.interpolate(octave, size=(height, width), mode="bilinear", align_corners=False)
        total += weight
        weight *= 0.5

    # Add broad fog blobs so the haze varies locally instead of looking flat.
    ys = torch.linspace(0.0, 1.0, height, device=device).view(1, 1, height, 1)
    xs = torch.linspace(0.0, 1.0, width, device=device).view(1, 1, 1, width)
    n_blobs = int(torch.randint(2, 5, (1,), generator=generator).item())
    blobs = torch.zeros_like(d)
    for _ in range(n_blobs):
        cy = torch.empty(batch_size, 1, 1, 1, device=device).uniform_(0.0, 1.0, generator=generator)
        cx = torch.empty(batch_size, 1, 1, 1, device=device).uniform_(0.0, 1.0, generator=generator)
        sigma = torch.empty(batch_size, 1, 1, 1, device=device).uniform_(0.2, 0.55, generator=generator)
        blobs += torch.exp(-((ys - cy) ** 2 + (xs - cx) ** 2) / (2.0 * sigma ** 2))

    d = 0.65 * (d / total) + 0.35 * (blobs / n_blobs)
    d_min = d.amin(dim=(2, 3), keepdim=True)
    d_max = d.amax(dim=(2, 3), keepdim=True)
    return (d - d_min) / (d_max - d_min + 1e-6)


def _gaussian_kernel1d(sigma: float, device: torch.device) -> torch.Tensor:
    """1D Gaussian kernel, radius = ceil(3 * sigma)."""
    radius = max(1, int(math.ceil(3.0 * sigma)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    k = torch.exp(-(x ** 2) / (2.0 * sigma * sigma))
    return k / k.sum()


def gaussian_blur(
    image: torch.Tensor,
    sigma: float,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Per-channel separable Gaussian blur with random anisotropy."""
    if sigma <= 0.0:
        return image
    c = image.shape[1]
    jitter = torch.empty(2, device=image.device).uniform_(0.75, 1.25, generator=generator)
    sigma_x = max(0.05, sigma * float(jitter[0].item()))
    sigma_y = max(0.05, sigma * float(jitter[1].item()))

    kx_1d = _gaussian_kernel1d(sigma_x, image.device)
    ky_1d = _gaussian_kernel1d(sigma_y, image.device)
    radius_x = (kx_1d.numel() - 1) // 2
    radius_y = (ky_1d.numel() - 1) // 2
    kx = kx_1d.view(1, 1, 1, -1).expand(c, 1, 1, -1)
    ky = ky_1d.view(1, 1, -1, 1).expand(c, 1, -1, 1)

    out = F.pad(image, (radius_x, radius_x, 0, 0), mode="reflect")
    out = F.conv2d(out, kx, groups=c)
    out = F.pad(out, (0, 0, radius_y, radius_y), mode="reflect")
    out = F.conv2d(out, ky, groups=c)
    return out


def apply_fog(
    image: torch.Tensor,
    beta_range: Tuple[float, float] = (3.5, 7),
    a_range: Tuple[float, float] = (0.85, 1.0),
    blur_sigma_range: Tuple[float, float] = (0.5, 1.8),
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Apply artificial fog to a batch of images.

    The model is the standard atmospheric scattering equation
        I = J * t + A * (1 - t),    t = exp(-beta * d)
    followed by a per-image Gaussian blur that mimics small-angle scattering.

    Args:
        image:             Clean image batch in [-1, 1], shape (B, C, H, W).
        beta_range:        Per-image range for haze density beta (larger = denser).
        a_range:           Per-image range for atmospheric light A (closer to 1 = whiter).
        blur_sigma_range:  Per-image range for Gaussian blur sigma (pixels).
                           Set (0.0, 0.0) to disable.
        generator:         Optional torch.Generator for reproducible fog.

    Returns:
        Hazy image batch in [-1, 1], same shape as input.
    """
    if image.ndim != 4:
        raise ValueError(f"apply_fog expects (B, C, H, W), got {tuple(image.shape)}")
    b, _, h, w = image.shape
    if image.shape[1] not in (1, 3):
        raise ValueError(f"apply_fog supports grayscale/RGB images, got C={image.shape[1]}")
    device = image.device

    # Work in [0, 1] so the physical scattering model has its standard form.
    j = (image + 1.0) / 2.0

    beta = torch.empty(b, 1, 1, 1, device=device)
    beta.uniform_(beta_range[0], beta_range[1], generator=generator)

    a = torch.empty(b, 1, 1, 1, device=device)
    a.uniform_(a_range[0], a_range[1], generator=generator)

    d = _make_depth_map(b, h, w, device, generator=generator)
    t = torch.exp(-beta * d)

    i = j * t + a * (1.0 - t)

    # Each image gets its own scattering softness.
    sigma_lo, sigma_hi = blur_sigma_range
    if sigma_hi > 0.0:
        sigmas = torch.empty(b, device=device).uniform_(sigma_lo, sigma_hi, generator=generator)
        i = torch.cat(
            [
                gaussian_blur(i[n : n + 1], float(sigmas[n].item()), generator=generator)
                for n in range(b)
            ],
            dim=0,
        )

    return (i * 2.0 - 1.0).clamp(-1.0, 1.0)
