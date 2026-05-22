# Drifting Models

Unofficial PyTorch implementation of ["Generative Modeling via Drifting"](https://arxiv.org/abs/2602.04770).

Just for fun and learning. Feel free to test and pull request~

**One-step generation** - no iterative sampling required (1-NFE).

## Method

Drifting Models train a generator by computing a *drifting field* V that points generated samples toward real data and away from other generated samples. The loss is simply `||V||^2`, which reaches zero when generated samples are indistinguishable from real data.

Key components:
- **DiT-style generator** with adaLN-Zero conditioning
- **Drifting field V** computed via soft assignment matrices (Algorithm 2)
- **Multi-temperature** V computation for multi-scale structure
- **Classifier-free guidance (CFG)** support

## Requirements

```bash
pip install torch torchvision einops
```

## TODO
- [x] Support MNIST and CIFAR-10
- [ ] Support ImageNet

## Usage

### Training

```bash
# MNIST (pixel space, ~20 min on GPU)
python train.py --dataset mnist

# CIFAR-10 (with feature encoder)
python train.py --dataset cifar10
```

### Synthetic Dehazing Toy

`TrainingDeHazing.py` supports the original hazy MNIST toy path and a first
paired synthetic CIFAR-10 dehazing run. CIFAR images are normalized to `[-1, 1]`,
fog is generated on the fly, and the dehazing model concatenates noise/degraded
input with the hazy image, so CIFAR uses 3 output channels and 6 internal input
channels at the patch embed.

```bash
# MNIST dehazing
python TrainingDeHazing.py --dataset mnist --data_dir ./data/mnist --batch_size 128 --epochs 10 --save_dir ./outputs/dehaze

# CIFAR-10 dehazing
python TrainingDeHazing.py --dataset cifar10 --data_dir ./data --batch_size 64 --epochs 5 --lr 2e-4 --save_dir ./outputs/cifar_dehaze --device auto

# Tiny CIFAR smoke test
python TrainingDeHazing.py --dataset cifar10 --data_dir ./data --batch_size 4 --epochs 1 --max_steps 1 --save_dir ./outputs/cifar_dehaze_smoke --device cpu --num_workers 0
```

CIFAR outputs include `samples/latest.png`, `samples/latest.txt`,
`cifar_dehaze_samples.png`, `cifar_dehaze_samples.txt`, and
`checkpoints/latest.pt` under `--save_dir`.

### CIFAR-10 Synthetic Dehazing Ablations

The unified `TrainingDeHazing.py` can compare fog strength, noise input,
prediction target, supervised reconstruction weight, and model size.

Mild fog, random noise, direct prediction, drift only:

```bash
python TrainingDeHazing.py \
  --dataset cifar10 \
  --fog_preset mild \
  --noise_mode random \
  --prediction_mode direct \
  --lambda_l1 0.0 \
  --model_preset small \
  --batch_size 128 \
  --epochs 20 \
  --save_dir ./outputs/abl_A_mild_random_direct_drift \
  --device cuda
```

Medium fog, zero noise, residual prediction, drift only:

```bash
python TrainingDeHazing.py \
  --dataset cifar10 \
  --fog_preset medium \
  --noise_mode zero \
  --prediction_mode residual \
  --lambda_l1 0.0 \
  --model_preset small \
  --batch_size 128 \
  --epochs 20 \
  --save_dir ./outputs/abl_B_medium_zero_residual_drift \
  --device cuda
```

Mild fog, zero noise, residual prediction, drift plus L1:

```bash
python TrainingDeHazing.py \
  --dataset cifar10 \
  --fog_preset mild \
  --noise_mode zero \
  --prediction_mode residual \
  --lambda_l1 1.0 \
  --lambda_l2 0.0 \
  --model_preset small \
  --batch_size 128 \
  --epochs 20 \
  --save_dir ./outputs/abl_C_mild_zero_residual_l1 \
  --device cuda
```

Model-size comparisons:

```bash
python TrainingDeHazing.py \
  --dataset cifar10 \
  --fog_preset mild \
  --noise_mode zero \
  --prediction_mode residual \
  --lambda_l1 1.0 \
  --model_preset medium \
  --batch_size 128 \
  --epochs 20 \
  --save_dir ./outputs/abl_D_medium_model \
  --device cuda

python TrainingDeHazing.py \
  --dataset cifar10 \
  --fog_preset mild \
  --noise_mode zero \
  --prediction_mode residual \
  --lambda_l1 1.0 \
  --model_preset large \
  --batch_size 128 \
  --epochs 20 \
  --save_dir ./outputs/abl_E_large_model \
  --device cuda
```

Each run writes `samples/latest.png`, `samples/latest.txt`,
`checkpoints/latest.pt`, and `checkpoints/final.pt` under `--save_dir`.

### Sampling

```bash
python sample.py --checkpoint outputs/mnist/checkpoint_final.pt --dataset mnist
```

## Results

MNIST samples after ~8000 steps:

![MNIST Samples](assets/samples_step8000.png)

CIFAR-10 samples after ~8500 steps:

![CIFAR-10 Samples](assets/samples_step8500.png)

## Project Structure

```
├── model.py            # DiT-style generator
├── drifting.py         # Drifting field V computation
├── feature_encoder.py  # CNN feature encoder (for CIFAR)
├── train.py            # Training loop
├── sample.py           # Sampling script
└── utils.py            # EMA, utilities
```

## Key Hyperparameters

| Parameter | MNIST | CIFAR-10 |
|-----------|-------|----------|
| Model | DriftDiT-Tiny (9M) | DriftDiT-Small (27M) |
| Epochs | 100 | 200 |
| LR | 2e-4 | 2e-4 |
| Temperatures | [0.02, 0.05, 0.2] | [0.02, 0.05, 0.2] |
| Feature space | Pixel (L2 norm) | CNN encoder |
