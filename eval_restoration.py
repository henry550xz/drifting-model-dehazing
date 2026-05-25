#!/usr/bin/env python
"""Evaluate paired restoration predictions with simple full-reference metrics."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from PIL import Image


MetricDict = dict[str, float]


class EvalError(RuntimeError):
    """Raised for user-facing evaluation errors."""


def parse_extensions(value: str) -> list[str]:
    extensions = [item.strip().lower() for item in value.split(",")]
    if not extensions or any(not item for item in extensions):
        raise argparse.ArgumentTypeError("extension list must be comma-separated and non-empty")

    seen: set[str] = set()
    parsed: list[str] = []
    for ext in extensions:
        if not ext.startswith(".") or ext == ".":
            raise argparse.ArgumentTypeError(f"invalid extension {ext!r}; extensions must start with '.'")
        if ext not in seen:
            seen.add(ext)
            parsed.append(ext)
    return parsed


def non_negative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid integer value: {value!r}") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("crop_border must be >= 0")
    return parsed


def scan_images(root: Path, recursive: bool, extensions: Sequence[str]) -> list[Path]:
    if not root.is_dir():
        raise EvalError(f"image directory does not exist: {root}")

    paths: Iterable[Path]
    paths = root.rglob("*") if recursive else root.iterdir()
    images = [path for path in paths if path.is_file() and path.suffix.lower() in extensions]
    return sorted(images, key=lambda path: path.relative_to(root).as_posix())


def display_rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def build_stem_index(root: Path, recursive: bool, extensions: Sequence[str], label: str) -> dict[str, Path]:
    paths = scan_images(root, recursive, extensions)
    index: dict[str, Path] = {}
    duplicates: dict[str, list[Path]] = {}

    for path in paths:
        stem = path.stem
        if stem in index:
            duplicates.setdefault(stem, [index[stem]]).append(path)
        else:
            index[stem] = path

    if duplicates:
        details = []
        for stem in sorted(duplicates):
            dup_paths = ", ".join(display_rel(path, root) for path in duplicates[stem])
            details.append(f"{stem!r}: {dup_paths}")
        raise EvalError(f"duplicate filename stems in {label}: " + "; ".join(details))

    return index


def find_relative_match(source_path: Path, source_root: Path, target_root: Path, extensions: Sequence[str]) -> Path:
    rel_path = source_path.relative_to(source_root)
    exact = target_root / rel_path
    if exact.is_file():
        return exact

    rel_stem = rel_path.with_suffix("")
    for ext in extensions:
        candidate = target_root / rel_stem.with_suffix(ext)
        if candidate.is_file():
            return candidate

    raise EvalError(
        f"missing match for {display_rel(source_path, source_root)} under {target_root} "
        f"(tried same relative path/stem)"
    )


def load_rgb_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        data = torch.frombuffer(bytearray(rgb.tobytes()), dtype=torch.uint8)
    return data.reshape(height, width, 3).to(dtype=torch.float32).div_(255.0)


def crop_tensor(image: torch.Tensor, crop_border: int, path: Path) -> torch.Tensor:
    if crop_border == 0:
        return image

    height, width = image.shape[:2]
    if crop_border * 2 >= height or crop_border * 2 >= width:
        raise EvalError(
            f"crop_border={crop_border} removes entire image for {path} "
            f"with size {width}x{height}"
        )
    return image[crop_border : height - crop_border, crop_border : width - crop_border, :]


def load_pair_tensors(clean_path: Path, pred_path: Path, hazy_path: Path | None, crop_border: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    clean = load_rgb_tensor(clean_path)
    pred = load_rgb_tensor(pred_path)
    hazy = load_rgb_tensor(hazy_path) if hazy_path is not None else None

    if clean.shape != pred.shape:
        raise EvalError(
            f"image size mismatch for clean/pred: {clean_path} has {tuple(clean.shape[:2])}, "
            f"{pred_path} has {tuple(pred.shape[:2])}"
        )
    if hazy is not None and clean.shape != hazy.shape:
        raise EvalError(
            f"image size mismatch for clean/hazy: {clean_path} has {tuple(clean.shape[:2])}, "
            f"{hazy_path} has {tuple(hazy.shape[:2])}"
        )

    clean = crop_tensor(clean, crop_border, clean_path)
    pred = crop_tensor(pred, crop_border, pred_path)
    hazy = crop_tensor(hazy, crop_border, hazy_path) if hazy is not None and hazy_path is not None else None
    return clean, pred, hazy


def compute_metrics(clean: torch.Tensor, other: torch.Tensor) -> MetricDict:
    diff = other - clean
    mse = torch.mean(diff.square()).item()
    mae = torch.mean(diff.abs()).item()
    psnr = math.inf if mse == 0.0 else 10.0 * math.log10(1.0 / mse)
    return {"mse": mse, "mae": mae, "psnr": psnr}


def finite_mean_std(values: Sequence[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return mean, math.sqrt(variance)


def mean_std(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        raise EvalError("empty metric list")

    if all(math.isfinite(value) for value in values):
        return finite_mean_std(values)

    if any(math.isnan(value) for value in values):
        return math.nan, math.nan

    unique_non_finite = {value for value in values if not math.isfinite(value)}
    finite_values = [value for value in values if math.isfinite(value)]
    if len(unique_non_finite) == 1 and not finite_values:
        return next(iter(unique_non_finite)), 0.0
    if math.inf in unique_non_finite and -math.inf in unique_non_finite:
        return math.nan, math.nan
    return next(iter(unique_non_finite)), math.inf


def summarize(prefix: str, rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    summary: dict[str, float] = {}
    for metric in ("mse", "mae", "psnr"):
        values = [float(row[f"{prefix}_{metric}"]) for row in rows]
        mean, std = mean_std(values)
        summary[f"{metric}_mean"] = mean
        summary[f"{metric}_std"] = std
    return summary


def sanitize_json(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return value
    if isinstance(value, dict):
        return {key: sanitize_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_json(item) for item in value]
    return value


def format_metric(value: float) -> str:
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:.6g}"


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    clean_dir = Path(args.clean_dir)
    pred_dir = Path(args.pred_dir)
    hazy_dir = Path(args.hazy_dir) if args.hazy_dir is not None else None

    clean_paths = scan_images(clean_dir, args.recursive, args.extensions)
    if not clean_paths:
        raise EvalError(f"no clean images found in {clean_dir}")

    if args.max_images is not None:
        clean_paths = clean_paths[: args.max_images]
    if not clean_paths:
        raise EvalError("empty evaluation set after applying max_images")

    pred_index: dict[str, Path] | None = None
    hazy_index: dict[str, Path] | None = None
    if args.match_mode == "stem":
        pred_index = build_stem_index(pred_dir, args.recursive, args.extensions, "pred_dir")
        if hazy_dir is not None:
            hazy_index = build_stem_index(hazy_dir, args.recursive, args.extensions, "hazy_dir")

    rows: list[dict[str, Any]] = []
    for clean_path in clean_paths:
        if args.match_mode == "relative":
            pred_path = find_relative_match(clean_path, clean_dir, pred_dir, args.extensions)
            hazy_path = (
                find_relative_match(clean_path, clean_dir, hazy_dir, args.extensions)
                if hazy_dir is not None
                else None
            )
        else:
            assert pred_index is not None
            stem = clean_path.stem
            if stem not in pred_index:
                raise EvalError(f"missing pred match for stem {stem!r} from {display_rel(clean_path, clean_dir)}")
            pred_path = pred_index[stem]
            if hazy_dir is not None:
                assert hazy_index is not None
                if stem not in hazy_index:
                    raise EvalError(f"missing hazy match for stem {stem!r} from {display_rel(clean_path, clean_dir)}")
                hazy_path = hazy_index[stem]
            else:
                hazy_path = None

        clean, pred, hazy = load_pair_tensors(clean_path, pred_path, hazy_path, args.crop_border)
        pred_metrics = compute_metrics(clean, pred)
        hazy_metrics = compute_metrics(clean, hazy) if hazy is not None else None

        row: dict[str, Any] = {
            "clean": display_rel(clean_path, clean_dir),
            "pred": display_rel(pred_path, pred_dir),
            "hazy": display_rel(hazy_path, hazy_dir) if hazy_path is not None and hazy_dir is not None else None,
            "pred_mse": pred_metrics["mse"],
            "pred_mae": pred_metrics["mae"],
            "pred_psnr": pred_metrics["psnr"],
            "hazy_mse": hazy_metrics["mse"] if hazy_metrics is not None else None,
            "hazy_mae": hazy_metrics["mae"] if hazy_metrics is not None else None,
            "hazy_psnr": hazy_metrics["psnr"] if hazy_metrics is not None else None,
        }
        rows.append(row)

        if args.verbose:
            print(
                f"{row['clean']}: pred mse={format_metric(row['pred_mse'])} "
                f"mae={format_metric(row['pred_mae'])} psnr={format_metric(row['pred_psnr'])}"
            )

    pred_summary = summarize("pred", rows)
    summary: dict[str, Any] = {"num_images": len(rows), "pred": pred_summary}
    if hazy_dir is not None:
        hazy_summary = summarize("hazy", rows)
        summary["hazy"] = hazy_summary
        summary["improvement"] = {
            "delta_mse_mean": hazy_summary["mse_mean"] - pred_summary["mse_mean"],
            "delta_mae_mean": hazy_summary["mae_mean"] - pred_summary["mae_mean"],
            "delta_psnr_mean": pred_summary["psnr_mean"] - hazy_summary["psnr_mean"],
        }

    return {
        "summary": summary,
        "config": {
            "clean_dir": clean_dir.as_posix(),
            "pred_dir": pred_dir.as_posix(),
            "hazy_dir": hazy_dir.as_posix() if hazy_dir is not None else None,
            "match_mode": args.match_mode,
            "recursive": args.recursive,
            "extensions": list(args.extensions),
            "crop_border": args.crop_border,
            "max_images": args.max_images,
        },
        "per_image": rows,
    }


def print_summary(result: dict[str, Any], output_path: Path) -> None:
    summary = result["summary"]
    pred = summary["pred"]
    print(f"Evaluated images: {summary['num_images']}")
    print(
        "Pred mean: "
        f"MSE={format_metric(pred['mse_mean'])} "
        f"MAE={format_metric(pred['mae_mean'])} "
        f"PSNR={format_metric(pred['psnr_mean'])}"
    )
    if "hazy" in summary:
        hazy = summary["hazy"]
        improvement = summary["improvement"]
        print(
            "Hazy mean: "
            f"MSE={format_metric(hazy['mse_mean'])} "
            f"MAE={format_metric(hazy['mae_mean'])} "
            f"PSNR={format_metric(hazy['psnr_mean'])}"
        )
        print(
            "Improvement: "
            f"delta_MSE={format_metric(improvement['delta_mse_mean'])} "
            f"delta_PSNR={format_metric(improvement['delta_psnr_mean'])}"
        )
    print(f"Output JSON: {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate clean/dehazed image pairs with MSE, MAE, and PSNR.")
    parser.add_argument("--clean_dir", required=True, type=Path, help="Directory containing clean reference images.")
    parser.add_argument("--pred_dir", required=True, type=Path, help="Directory containing predicted/dehazed images.")
    parser.add_argument("--output", required=True, type=Path, help="Path to write metrics JSON.")
    parser.add_argument("--hazy_dir", type=Path, default=None, help="Optional directory containing hazy input images.")
    parser.add_argument("--match_mode", choices=("relative", "stem"), default="relative", help="Image matching strategy.")
    parser.add_argument("--recursive", action="store_true", help="Scan image directories recursively.")
    parser.add_argument(
        "--extensions",
        type=parse_extensions,
        default=parse_extensions(".png,.jpg,.jpeg,.webp"),
        help="Comma-separated image extensions to include.",
    )
    parser.add_argument("--crop_border", type=non_negative_int, default=0, help="Pixels to crop from every border.")
    parser.add_argument("--max_images", type=int, default=None, help="Maximum number of sorted clean images to evaluate.")
    parser.add_argument("--verbose", action="store_true", help="Print per-image prediction metrics.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.max_images is not None and args.max_images < 0:
        parser.error("--max_images must be >= 0")

    try:
        result = evaluate(args)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(sanitize_json(result), handle, indent=2, allow_nan=False)
            handle.write("\n")
        print_summary(result, output_path)
    except EvalError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
