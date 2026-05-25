from typing import Callable, List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from drifting import compute_V

FeatureOutput = Union[torch.Tensor, Sequence[torch.Tensor]]


class ModelEncodeFeatureExtractor(nn.Module):
    """Adapter for future models that expose encode(x)."""

    def __init__(self, model: nn.Module):
        super().__init__()
        if not callable(getattr(model, "encode", None)):
            raise ValueError("feature_encoder=none with drift_space=feature requires model.encode(x).")
        self.model = model

    def forward(self, x: torch.Tensor) -> FeatureOutput:
        return self.model.encode(x)


def make_multiscale_feature_encoder(in_channels: int) -> nn.Module:
    from feature_encoder import MultiScaleFeatureEncoder

    return MultiScaleFeatureEncoder(
        in_channels=in_channels,
        base_width=32,
        blocks_per_stage=1,
        feature_dim=512,
        multi_scale=True,
    )


def _flatten_feature_output(features: FeatureOutput) -> torch.Tensor:
    if isinstance(features, torch.Tensor):
        return features.flatten(start_dim=1)
    if isinstance(features, (list, tuple)):
        if len(features) == 0:
            raise ValueError("Feature encoder returned no features.")
        flattened: List[torch.Tensor] = []
        for feature in features:
            if not isinstance(feature, torch.Tensor):
                raise TypeError("Feature encoder outputs must be tensors.")
            if feature.ndim == 4:
                feature = F.adaptive_avg_pool2d(feature, 1)
            flattened.append(feature.flatten(start_dim=1))
        return torch.cat(flattened, dim=1)
    raise TypeError("Feature encoder must return a tensor or a sequence of tensors.")


def conditional_drift_loss_from_features(
    feat_gen: torch.Tensor,
    feat_pos: torch.Tensor,
    temperatures: Sequence[float] = (0.05, 0.2, 0.5),
    positive_mode: str = "batch",
) -> torch.Tensor:
    if feat_gen.shape != feat_pos.shape:
        raise RuntimeError(f"Generated feature shape {tuple(feat_gen.shape)} != positive {tuple(feat_pos.shape)}")

    feat_gen_norm = F.normalize(feat_gen, p=2, dim=1)
    feat_pos_norm = F.normalize(feat_pos, p=2, dim=1)

    v_total = torch.zeros_like(feat_gen_norm)
    if positive_mode == "batch":
        for tau in temperatures:
            # Negatives = other generated samples (mask_self=True drops the diagonal).
            v_tau = compute_V(feat_gen_norm, feat_pos_norm, feat_gen_norm, tau, mask_self=True)
            v_norm = torch.sqrt(torch.mean(v_tau ** 2) + 1e-8)
            v_total = v_total + v_tau / (v_norm + 1e-8)
    elif positive_mode == "paired":
        batch_size = feat_gen_norm.shape[0]
        for tau in temperatures:
            v_tau = torch.zeros_like(feat_gen_norm)
            for i in range(batch_size):
                if batch_size > 1:
                    neg_mask = torch.ones(batch_size, dtype=torch.bool, device=feat_gen_norm.device)
                    neg_mask[i] = False
                    y_neg = feat_gen_norm[neg_mask]
                    mask_self = False
                else:
                    y_neg = feat_gen_norm
                    mask_self = True
                v_tau[i : i + 1] = compute_V(
                    feat_gen_norm[i : i + 1],
                    feat_pos_norm[i : i + 1],
                    y_neg,
                    tau,
                    mask_self=mask_self,
                )
            v_norm = torch.sqrt(torch.mean(v_tau ** 2) + 1e-8)
            v_total = v_total + v_tau / (v_norm + 1e-8)
    else:
        raise ValueError(f"Unknown drift positive mode: {positive_mode}")

    target = (feat_gen_norm + v_total).detach()
    return F.mse_loss(feat_gen_norm, target)


def conditional_drift_loss(
    x_hat: torch.Tensor,
    x_clean: torch.Tensor,
    temperatures: Sequence[float] = (0.05, 0.2, 0.5),
    positive_mode: str = "batch",
) -> torch.Tensor:
    """Per-example conditional drifting loss in flattened pixel space."""
    feat_gen = x_hat.flatten(start_dim=1)
    feat_pos = x_clean.flatten(start_dim=1)
    return conditional_drift_loss_from_features(
        feat_gen,
        feat_pos,
        temperatures=temperatures,
        positive_mode=positive_mode,
    )


def feature_drift_loss(
    x_hat: torch.Tensor,
    x_clean: torch.Tensor,
    feature_extractor: Callable[[torch.Tensor], FeatureOutput],
    temperatures: Sequence[float] = (0.05, 0.2, 0.5),
    positive_mode: str = "batch",
) -> torch.Tensor:
    feat_gen = _flatten_feature_output(feature_extractor(x_hat))
    with torch.no_grad():
        feat_pos = _flatten_feature_output(feature_extractor(x_clean))
    return conditional_drift_loss_from_features(
        feat_gen,
        feat_pos,
        temperatures=temperatures,
        positive_mode=positive_mode,
    )


def compute_drift_loss(
    x_hat: torch.Tensor,
    x_clean: torch.Tensor,
    drift_space: str = "pixel",
    positive_mode: str = "batch",
    feature_extractor: Optional[Callable[[torch.Tensor], FeatureOutput]] = None,
    temperatures: Sequence[float] = (0.05, 0.2, 0.5),
) -> torch.Tensor:
    if drift_space == "pixel":
        return conditional_drift_loss(
            x_hat,
            x_clean,
            temperatures=temperatures,
            positive_mode=positive_mode,
        )
    if drift_space == "feature":
        if feature_extractor is None:
            raise ValueError("drift_space=feature requires a feature extractor.")
        return feature_drift_loss(
            x_hat,
            x_clean,
            feature_extractor=feature_extractor,
            temperatures=temperatures,
            positive_mode=positive_mode,
        )
    raise ValueError(f"Unknown drift space: {drift_space}")
