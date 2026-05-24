# Drifting Dehazing

Drifting Dehazing is a PyTorch dehazing project built around a DiT-style conditional generator and on-the-fly haze synthesis. It supports toy MNIST/CIFAR-10 synthetic dehazing, real RGB folder input, ASM and MCBM fog generation, supervised or drifting losses, and optional fog debug visualizations.

The training entry point for this dehazing work is `TrainingDeHazing.py`.

## Current Support

- MNIST and CIFAR-10 synthetic dehazing.
- Real RGB image folders with `--dataset folder`.
- ASM fog and MCBM fog with `--fog_type {asm,mcbm}`.
- Synthetic, flat, and external depth maps with `--depth_mode {synthetic,flat,external}`.
- Drift, supervised, and mixed training losses with `--loss_mode {drift,supervised,mixed}`.
- Batch or paired positive examples for drift loss with `--drift_positive_mode {batch,paired}`.
- Optional fog debug grids and metadata with `--save_fog_debug`.

## Setup

Create and activate a virtual environment, then install the project dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision einops matplotlib tqdm pillow numpy
```

Use the venv Python for project commands:

```bash
.venv/bin/python TrainingDeHazing.py --help
```

## Training Examples

### CIFAR Smoke Test

This runs one CPU step and writes outputs under `/tmp`.

```bash
.venv/bin/python TrainingDeHazing.py \
  --dataset cifar10 \
  --fog_type asm \
  --depth_mode synthetic \
  --batch_size 2 \
  --epochs 1 \
  --max_steps 1 \
  --save_dir /tmp/dehaze_smoke_cifar \
  --device cpu \
  --num_workers 0
```

### Folder Supervised Training

`--dataset folder` treats each clean RGB image as a supervised clean target and creates hazy inputs on the fly.

```bash
.venv/bin/python TrainingDeHazing.py \
  --dataset folder \
  --data_dir ./data/real_clean \
  --img_size 128 \
  --fog_type asm \
  --depth_mode synthetic \
  --loss_mode supervised \
  --lambda_l1 1.0 \
  --batch_size 8 \
  --epochs 20 \
  --save_dir ./outputs/folder_supervised \
  --device auto
```

### MCBM Fog Training

```bash
.venv/bin/python TrainingDeHazing.py \
  --dataset cifar10 \
  --fog_type mcbm \
  --depth_mode synthetic \
  --loss_mode mixed \
  --lambda_l1 1.0 \
  --batch_size 64 \
  --epochs 10 \
  --save_dir ./outputs/cifar_mcbm_mixed \
  --device auto
```

### Flat Depth Ablation

Flat depth uses a constant normalized depth map for each sample. This is useful when you want haze variation from fog parameters and MCBM density, but not from spatial synthetic depth.

```bash
.venv/bin/python TrainingDeHazing.py \
  --dataset cifar10 \
  --fog_type asm \
  --depth_mode flat \
  --batch_size 64 \
  --epochs 10 \
  --save_dir ./outputs/cifar_flat_depth \
  --device auto
```

### Fog Debug Visualization

Debug mode saves clean, hazy, depth, transmission, and density grids before training starts. It is opt-in and capped by `--fog_debug_max_images`.

```bash
.venv/bin/python TrainingDeHazing.py \
  --dataset cifar10 \
  --fog_type mcbm \
  --save_fog_debug \
  --fog_debug_dir /tmp/dehaze_fog_debug_mcbm \
  --fog_debug_max_images 4 \
  --batch_size 2 \
  --epochs 1 \
  --max_steps 1 \
  --save_dir /tmp/dehaze_smoke_debug \
  --device cpu \
  --num_workers 0
```

### External Depth Folder Training

External depth is currently supported for `--dataset folder` only.

```bash
.venv/bin/python TrainingDeHazing.py \
  --dataset folder \
  --data_dir ./data/real_clean \
  --depth_mode external \
  --depth_dir ./data/real_depth \
  --fog_type mcbm \
  --img_size 128 \
  --loss_mode supervised \
  --lambda_l1 1.0 \
  --batch_size 8 \
  --epochs 20 \
  --save_dir ./outputs/folder_external_depth \
  --device auto
```

## Important CLI Flags

- `--dataset {mnist,cifar10,folder}`: selects the input source.
- `--data_dir PATH`: dataset root. For folder mode this is the clean RGB image folder.
- `--recursive`: recursively scan `--data_dir` for folder images.
- `--img_size N`: output crop size for MNIST and folder datasets. CIFAR-10 uses fixed `32`.
- `--fog_type {asm,mcbm}`: selects atmospheric scattering model fog or MCBM non-homogeneous fog.
- `--fog_preset {mild,medium,heavy}`: selects fog strength ranges. Defaults are dataset-aware.
- `--depth_mode {synthetic,flat,external}`: selects the depth source for fog generation.
- `--depth_dir PATH`: required for `--depth_mode external`.
- `--loss_mode {drift,supervised,mixed}`: selects drift-only, reconstruction-only, or combined loss.
- `--drift_positive_mode {batch,paired}`: selects batch-level or paired positives for drift loss.
- `--lambda_l1`, `--lambda_l2`: supervised reconstruction weights.
- `--noise_mode {random,zero}`: input noise mode for the conditional generator.
- `--prediction_mode {direct,residual}`: whether the model predicts the clean image directly or a residual from the hazy input.
- `--save_fog_debug`: writes fog-generation debug grids and metadata before training.
- `--max_steps N`: hard cap for short checks and smoke tests.
- `--save_dir PATH`: output directory for samples and checkpoints.

## Folder Dataset Layout

For `--dataset folder`, place clean RGB images under `--data_dir`.

Non-recursive example:

```text
data/real_clean/
  image_0001.jpg
  image_0002.png
  image_0003.jpeg
```

Recursive example with `--recursive`:

```text
data/real_clean/
  scene_a/
    frame_0001.jpg
  scene_b/
    frame_0001.jpg
```

Supported RGB extensions are `.jpg`, `.jpeg`, `.png`, and `.webp`. Folder samples still return `(x_clean, x_hazy, -1)`, matching the existing training loop.

## Depth Modes

### `--depth_mode synthetic`

This is the default. ASM and MCBM use the existing random smooth synthetic depth generator. Existing commands that do not specify `--depth_mode` keep the previous behavior.

### `--depth_mode flat`

Flat mode uses a constant `[B, 1, H, W]` depth tensor with value `0.5`, clamped to `[0, 1]`. It works for MNIST, CIFAR-10, and folder datasets.

### `--depth_mode external`

External mode loads a precomputed depth map for each RGB image and passes it into ASM or MCBM instead of creating synthetic depth. It is currently supported only with `--dataset folder`; using it with MNIST or CIFAR-10 raises a clear error.

External depth files are matched from each RGB image to `--depth_dir` with this deterministic logic:

1. Try the same relative path under `--depth_dir`.
2. Try the same relative path with depth extensions `.png`, `.jpg`, `.jpeg`, `.npy`, `.pt`.
3. Try a root-level file in `--depth_dir` with the same RGB stem and one of those extensions.

For example:

```text
data/real_clean/
  scene_a/frame_0001.jpg
  scene_a/frame_0002.jpg

data/real_depth/
  scene_a/frame_0001.png
  scene_a/frame_0002.npy
```

Image depth maps are converted to grayscale tensors. `.npy` files may be shaped as `H x W`, `H x W x C`, or `C x H x W`. `.pt` files may be tensors or dictionaries with a `depth`, `depth_map`, or `map` key. Depth values are converted to float, normalized or clamped to `[0, 1]`, resized to the RGB resized shape, and cropped with the exact same crop parameters as the RGB image.

## Outputs

Each run writes under `--save_dir`:

```text
samples/
  latest.png
  latest.txt
checkpoints/
  latest.pt
  final.pt
```

Epoch checkpoints and samples are named `epoch_XXX.pt` and `epoch_XXX.png` when `--epoch_save_interval` enables them. CIFAR runs also write compatibility files such as `cifar_dehaze_samples.png` and `cifar_dehaze_samples.txt`.

Sample text files include paired hazy/dehazed metrics for the synthetic pair created by the dataset. Checkpoints include the model, EMA state, and run configuration, including `depth_mode` and `depth_dir`.

When `--save_fog_debug` is enabled, fog debug output includes:

```text
clean_grid.png
hazy_grid.png
depth_grid.png
transmission_grid.png
density_grid.png
metadata_summary.json
```

`density_grid.png` is present when the selected fog type exposes a density map, such as MCBM.

## Development Notes

- Use `/tmp/...` for smoke-test outputs while developing.
- Do not commit `outputs/`, `data/`, `__pycache__/`, or `.pyc` files.
- Keep smoke checks short. Prefer `--max_steps 1`, `--batch_size 2`, `--device cpu`, and `--num_workers 0`.
- Do not run long training when checking plumbing changes.

## Roadmap

Possible future work includes broader real-depth formats, richer validation workflows for real paired datasets, and more explicit evaluation scripts. These are not implemented benchmark claims.

## Acknowledgements

This project started as a fork/adaptation of [`tyfeld/drifting-model`](https://github.com/tyfeld/drifting-model), an unofficial PyTorch implementation of *Generative Modeling via Drifting*. The original repository provided the base drifting-model components, including the DiT-style generator and drifting-field training idea.

This repository has since been extended toward synthetic and real-image dehazing, including RGB folder datasets, ASM/MCBM fog generation, supervised dehazing baselines, paired-positive drift variants, fog metadata/debug visualization, and external-depth haze generation.
