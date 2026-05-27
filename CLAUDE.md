# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DynamicPFL is a research project studying **Discrete Fractional Logistic (DFL) chaotic noise** in Differential Privacy Federated Learning (DP-FL). Core question: can pure DFL noise outperform Gaussian noise in privacy-utility tradeoff?

## Critical Platform Constraints

- **Python path**: Always use `/d/anaconda/envs/pp/python.exe` — the system `python` has no torch
- **GBK encoding**: Windows console uses GBK — **NEVER use emojis in print statements**
- **No parallel experiments**: Let one experiment finish before starting the next

## Common Commands

```bash
# Batch comparison (DFL then Gaussian, sequential)
/d/anaconda/envs/pp/python.exe run_experiments.py --datasets CIFAR10 --dp_methods dfl gaussian --num_runs 1 --epochs 30

# Single dataset, single method
/d/anaconda/envs/pp/python.exe ours.py --dataset CIFAR10 --dp_method dfl --global_epoch 30 ...

# Generate comparison chart
/d/anaconda/envs/pp/python.exe plot_experiment_results.py --dataset CIFAR10 --run_id 1

# Noise distribution plot (Ziggurat vs Inverse-CDF vs theory)
/d/anaconda/envs/pp/python.exe plot_chaotic_vs_gaussian_distribution.py --num_samples 10000
```

## Architecture

```
Global Model → Local Training (Client)
              ↓
    Gradient Clipping (optional, --no_clip to skip)
              ↓
    Noise Injection (DFL chaotic or Gaussian)
              ↓
    Aggregation (FedAvg) → Global Model Update
```

### Key Modules

**utils.py** — Noise Engine
- `_get_pure_dfl_sequences(a, b, k, seed_val, needed_length)`: Pure DFL map `x(n+1) = mod(a*x(n)*(1-x(n)) + b*x(n-k), 1.0)`. Pre-computes sequence once (cache key: `(a,b,k)`), then deterministic hash-avalanche start_idx per call. No repeats, no golden ratio, no jitter.
- `generate_dfl_gaussian_noise(shape, a, b, k, x0, burn_in, decimation, device)`: Extracts DFL sequence (5x oversampling), applies burn-in + decimation, degeneracy guard, adjacent-pair coupling, feeds to Gaussian sampler. If `device` is set, runs sampler on GPU.
- `generate_ziggurat_gaussian_noise(shape, uniform_sequence)`: True Marsaglia-Tsang (2000) Ziggurat — 256-layer lookup tables, float64 internally, float32 output. Tables move to input tensor's device automatically.
- `generate_inverse_cdf_gaussian_noise(shape, uniform_sequence)`: Classic erfinv inverse-CDF — kept for ablation comparison. Simpler and faster than Ziggurat, identical accuracy in practice.
- `generate_random_gaussian_noise(shape, device, dtype)`: Gaussian path — uses `torch.rand` as uniform source, feeds to the active Gaussian sampler.
- `generate_random_gaussian_noise_like(reference_tensor)`: Convenience wrapper, matches device/dtype.

**ours.py** — Training Pipeline
- `local_update_with_dp()`: Client-side training with DP. Manual cosine LR decay: `lr = eta_min + 0.5*(lr_init - eta_min)*(1 + cos(π*epoch/T))`.
- `generate_chaotic_noise_v2()`: Thin wrapper calling `generate_dfl_gaussian_noise()` with args from CLI.
- GIR simulator runs at most once per client update (controlled by `gir_max_evals_per_client_update`).

**gradient_inversion_risk_simulator.py** — Privacy Evaluator
- `simulate_gradient_inversion_risk()`: Deep-copies the model (no hooks), runs gradient-matching inversion attacks, returns leakage/PSNR/MSE scores. "Anti-Inversion Ability" = defense score (higher = better).

**run_experiments.py** — Batch Runner
- Per-dataset, per-method parameter dicts. Validates sigma/epsilon equality between methods.
- CIFAR10 params: lr=0.004, sigma=0.01, clip=2.0, k=3, gap=11, sparsity=0.0, epochs=30.
- Other datasets use sigma=0.30, gap=8, sparsity=0.4 (not yet aligned with CIFAR10 tuning).

**data.py** — Federated Data Loading
- MNIST: IID partition. CIFAR10/SVHN/FashionMNIST: heterogeneous Dirichlet (`hetero_dir_partition`).
- FEMNIST support is broken.

**net.py** — Models: `cifar10Net` (3 conv+3 fc), `mnistNet`, `SVHNNet`, `fashionmnistNet`.

**plot_experiment_results.py** — Reads `experiment_results/` output, generates accuracy + anti-inversion charts.

## Key Parameters (CIFAR10, current)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `sigma_factor` | 0.01 | Same for Gaussian and DFL |
| `clipping_bound` | 2.0 | |
| `lr` | 0.004 | Cosine decay to 1e-5 over T epochs |
| `dfl_a, dfl_b, dfl_k` | 4.0, 501.0, 3 | DFL map parameters |
| `dfl_decimation` | 11 | Gap factor for decorrelation |
| `dfl_burn_in` | 2048 | Burn-in steps before collecting samples |
| `chaotic_factor` | 1.0 | Pure DFL (0.0 = pure Gaussian) |
| `target_epsilon` | 8.0 | |
| `epochs` | 30 | |

## Sampling Comparison (2026-05-27)

| Method | Ziggurat | Inverse-CDF |
|--------|----------|-------------|
| DFL accuracy | 65.42% | 65.60% |
| Gaussian accuracy | 76.27% | 76.02% |
| Gaussian time | 14522s | 4305s |

**Conclusion**: Sampling method has negligible effect on accuracy (<0.3% difference). Inverse-CDF is 3x faster. The Gaussian-vs-DFL accuracy gap (~10%) is from the noise source, not the sampler.
