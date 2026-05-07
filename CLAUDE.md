# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DynamicPFL is a research project studying **Discrete Fractional Logistic (DFL) chaotic noise** in Differential Privacy Federated Learning (DP-FL). The core research question: **Can pure DFL noise outperform Gaussian noise in privacy-utility tradeoff?**

Current focus: **Pure DFL (chaotic_factor=1.0) vs Gaussian noise comparison**

## Critical Platform Constraints

- **Python path**: Always use `/d/anaconda/envs/pp/python.exe` — the system `python` has no torch
- **GBK encoding**: Windows console uses GBK — **NEVER use emojis in print statements**, they will crash experiments with `'gbk' codec can't encode character`
- **No parallel experiments**: Always let one experiment finish before starting the next

## Common Commands

```bash
# Batch comparison (DFL then Gaussian, sequential)
/d/anaconda/envs/pp/python.exe run_experiments.py --datasets MNIST --dp_methods dfl gaussian --num_runs 1 --epochs 60

# Single dataset, single method
/d/anaconda/envs/pp/python.exe ours.py --dataset MNIST --dp_method dfl --global_epoch 60 ...

# Generate comparison chart
/d/anaconda/envs/pp/python.exe plot_experiment_results.py --dataset MNIST --run_id 1
```

## Architecture

```
Global Model → Local Training (Client)
              ↓
    Gradient Clipping
              ↓
    Noise Injection (DFL or Gaussian)
              ↓
    Gradient Sparsification
              ↓
    Aggregation → Global Model Update
```

### Key Modules

**utils.py** — Chaotic Noise Engine
- `_get_dfl_sequences()`: Core DFL map. **Dynamic elastic reservoir** — pool size = `max(20M, needed_length * 3)` to eliminate sequence overlap contamination. Single sequence with deterministic pseudo-random start_idx from x0/x1. Cache key: `(mu, alpha)`.
- `generate_dfl_gaussian_noise()`: Gets 1 sequence via `_get_dfl_sequences`, then mixes with golden ratio roll: `seq + 0.618 * torch.roll(seq, 7, 0)`. Applies burn-in, decimation, jitter, degeneracy check, Ziggurat sampling. Block sign flip for zero-mean on tiled segments.
- `generate_ziggurat_gaussian_noise()`: Vectorized inverse-CDF via `torch.erfinv()`
- `add_adaptive_gaussian_noise()`: Client-side DP noise injection with adaptive clipping

**ours.py** — Main Training Pipeline
- `local_update_with_dp()`: Client training with DP defense
- `generate_chaotic_noise_v2()`: Thin wrapper calling `generate_dfl_gaussian_noise()`
- Post-norm clipping after noise injection to prevent gradient explosion
- Integrates with `gradient_inversion_risk_simulator.py`

**gradient_inversion_risk_simulator.py** — Privacy Evaluator
- `simulate_gradient_inversion_risk()`: Gradient inversion attack simulator
- "Anti-Inversion Ability" — defense score (higher = better privacy)

**run_experiments.py** — Batch Experiment Runner
- Dataset-specific param dicts (all 4 datasets aligned for fair comparison)
- Validates sigma/epsilon equality between gaussian and dfl groups
- Outputs to `experiment_results/`

**data.py** — Federated Data Loading
- Datasets: MNIST (IID partition), FashionMNIST/SVHN/CIFAR10 (hetero_dir_partition)
- FEMNIST support is broken (TensorFlow dependency removed)

**net.py** — Model Architectures
- MNIST/fEMNIST: `mnistNet` (2 conv + 2 fc, 62-class output)
- CIFAR10: `cifar10Net` (3 conv + 3 fc)
- SVHN: `SVHNNet` (2 conv + 2 fc)
- FashionMNIST: `fashionmnistNet` (2 conv + 2 fc, grayscale 10-class)

**plot_experiment_results.py** — Visualization
- Reads from `experiment_results/` output files
- Generates accuracy + anti-inversion comparison charts
- `parse_output_file()` extracts training history, `parse_live_log()` as fallback

## Current Experiment Parameters

All 4 datasets use aligned parameters for fair DFL vs Gaussian comparison:

| Parameter | Value |
|-----------|-------|
| `sigma_factor` | 0.10 (both gaussian and dfl) |
| `clipping_bound` | 2.0 |
| `dfl_alpha` | 0.85 |
| `dfl_decimation` (gap) | 12 |
| `dfl_mu` | 3.99 |
| `dfl_burn_in` | 2048 |
| `chaotic_factor` | 1.0 (pure DFL) |
| `target_epsilon` | 8.0 |
| `epochs` | 60 |

## DFL Reservoir Details

The elastic reservoir (`utils.py:_get_dfl_sequences`):

1. Pool size dynamically computed: `max(20M, needed_length * 3)`
2. Pure Python list generation (fast, ~2s for 20M points)
3. Single sequence — no dual-sequence correlation issues
4. Deterministic pseudo-random start_idx from `(x0*13579 + x1*97531) * 1e6`
5. `start_idx` has at least `2 * needed_length` random space — eliminates overlap
6. Golden ratio mixing via `torch.roll(seq, 7, 0)` for uniform distribution

## Output Files

- `experiment_results/{dataset}_{method}_run{N}_output.txt`: Full metrics + training history
- `experiment_results/{dataset}_{method}_run{N}_live.log`: Live training log
- `experiment_results/comparison_{dataset}_*.png`: Comparison charts (ignored by gitignore — use `-f` to add)
