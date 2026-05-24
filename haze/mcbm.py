from typing import Any, Optional, Tuple

import torch

from .asm import _make_depth_map, gaussian_blur


def generate_mcbm_beta_map(
    batch_size: int,
    height: int,
    width: int,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
    num_particles: int = 16,
    num_steps: int = 24,
    step_scale: float = 0.08,
    blur_sigma: float = 2.0,
) -> torch.Tensor:
    """Generate non-homogeneous fog density maps in [0, 1]."""
    if batch_size <= 0 or height <= 0 or width <= 0:
        raise ValueError(
            f"Invalid beta map shape: batch={batch_size}, height={height}, width={width}"
        )
    if num_particles <= 0 or num_steps <= 0:
        raise ValueError(
            f"num_particles and num_steps must be positive, got {num_particles}, {num_steps}"
        )

    starts = torch.rand(batch_size, num_particles, 1, 2, device=device, generator=generator)
    increments = torch.randn(
        batch_size,
        num_particles,
        num_steps,
        2,
        device=device,
        generator=generator,
    ) * step_scale
    positions = (starts + increments.cumsum(dim=2)).clamp(0.0, 1.0)

    ys = (positions[..., 0] * float(height - 1)).round().long()
    xs = (positions[..., 1] * float(width - 1)).round().long()
    flat_idx = (ys * width + xs).reshape(batch_size, -1)

    density = torch.zeros(batch_size, height * width, device=device)
    density.scatter_add_(1, flat_idx, torch.ones_like(flat_idx, dtype=density.dtype))
    density = density.view(batch_size, 1, height, width)

    if blur_sigma > 0.0:
        density = gaussian_blur(density, blur_sigma, generator=generator)

    d_min = density.amin(dim=(2, 3), keepdim=True)
    d_max = density.amax(dim=(2, 3), keepdim=True)
    return (density - d_min) / (d_max - d_min + 1e-6)


def apply_mcbm_fog(
    image: torch.Tensor,
    beta_range: Tuple[float, float] = (3.5, 7.0),
    a_range: Tuple[float, float] = (0.85, 1.0),
    blur_sigma_range: Tuple[float, float] = (0.5, 1.8),
    generator: Optional[torch.Generator] = None,
    return_meta: bool = False,
) -> Any:
    """Apply MCBM-inspired non-homogeneous fog to a batch in [-1, 1]."""
    if image.ndim != 4:
        raise ValueError(f"apply_mcbm_fog expects (B, C, H, W), got {tuple(image.shape)}")
    b, c, h, w = image.shape
    if c not in (1, 3):
        raise ValueError(f"apply_mcbm_fog supports grayscale/RGB images, got C={c}")
    device = image.device

    j = (image + 1.0) / 2.0

    beta = torch.empty(b, 1, 1, 1, device=device)
    beta.uniform_(beta_range[0], beta_range[1], generator=generator)

    alpha = torch.empty(b, 1, 1, 1, device=device)
    alpha.uniform_(0.5, 1.0, generator=generator)

    a = torch.empty(b, 1, 1, 1, device=device)
    a.uniform_(a_range[0], a_range[1], generator=generator)

    beta_tilde = generate_mcbm_beta_map(b, h, w, device, generator=generator)
    depth = _make_depth_map(b, h, w, device, generator=generator)
    t_mcbm = torch.exp(-(beta + alpha * beta_tilde) * depth)

    i = j * t_mcbm + a * (1.0 - t_mcbm)

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

    x_hazy = (i * 2.0 - 1.0).clamp(-1.0, 1.0)
    if not return_meta:
        return x_hazy

    meta = {
        "depth": depth.clamp(0.0, 1.0),
        "transmission": t_mcbm.clamp(0.0, 1.0),
        "beta": beta,
        "atmospheric_light": a,
        "beta_tilde": beta_tilde.clamp(0.0, 1.0),
        "alpha": alpha,
        "fog_type": "mcbm",
        "fog_config": {
            "beta_range": beta_range,
            "a_range": a_range,
            "blur_sigma_range": blur_sigma_range,
        },
    }
    return x_hazy, meta
