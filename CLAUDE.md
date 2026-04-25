# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DynamicPFL is a research project studying **Discrete Fractional Logistic (DFL) chaotic noise** in Differential Privacy Federated Learning (DP-FL). The core research question: **Can pure DFL noise outperform Gaussian noise in privacy-utility tradeoff?**

Current focus: **Pure DFL (chaotic_factor=1.0) vs Gaussian noise comparison**

## Architecture

### Federated Learning Pipeline
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

**utils.py** - Chaotic Noise Engine
- `generate_dfl_sequence()`: Core DFL map iteration
- `generate_dfl_gaussian_noise()`: Combines two DFL sequences, applies Ziggurat sampling
- `generate_ziggurat_gaussian_noise()`: Vectorized inverse-CDF sampling via `torch.erfinv()`
- **Security fix**: Block sign flip for zero-mean noise

**ours.py** - Main Training Loop
- `local_update_with_dp()`: Client-side training with DP defense
- DFL noise mixing: `sqrt(chaotic_factor) * DFL_noise + sqrt(1-chaotic_factor) * Gaussian_noise`
- **Secondary clipping (line 359-366)**: After adding DFL noise, applies post-norm clipping to prevent gradient explosion
- Integrates with `gradient_inversion_risk_simulator.py` for real-time risk evaluation

**gradient_inversion_risk_simulator.py** - Privacy Evaluator
- `simulate_gradient_inversion_risk()`: Gradient inversion attack simulator
- Computes "Anti-Inversion Ability" / defense_score (higher = better privacy)

**run_experiments.py** - Batch Experiment Runner
- Runs comparative experiments across datasets
- Outputs to `experiment_results/`

## Common Commands

### Batch Comparison
```bash
python run_experiments.py --datasets CIFAR10 --dp_methods dfl gaussian --num_runs 1 --epochs 60
```

## Key Parameters

| Parameter | Range | Purpose |
|-----------|-------|---------|
| `--chaotic_factor` | [0, 1] | Mix ratio: 0=Gaussian, 1=DFL |
| `--sigma_factor_dfl` | > 0 | Noise intensity |
| `--dfl_alpha` | (0, 1] | Memory depth (higher=more memory) |
| `--dfl_mu` | [3.57, 4.0] | Chaos level (higher=more chaotic) |
| `--dfl_decimation` | ≥1 | Gap factor for correlation breaking |
| `--clipping_bound` | > 0 | Gradient clipping threshold |

## DFL Implementation Details

The DFL noise generation (`utils.py:generate_dfl_gaussian_noise`):

1. **Generate two DFL sequences** with different seeds (x0, x1)
2. **Mix sequences**: `u = (seq1 + 0.618 * seq2) % 1.0` (golden ratio)
3. **Apply burn-in**: discard initial samples
4. **Apply decimation (gap)**: break fractional-order correlation
5. **Add jitter**: prevent ties
6. **Degeneracy check**: inject noise if too few bins occupied
7. **Block sign flip**: ensure zero-mean for tiled segments
8. **Ziggurat sampling**: `sqrt(2) * erfinv(2u-1)` → Gaussian

## Output Files

- `experiment_results/{dataset}_{method}_run{N}_output.txt`: Full metrics
- `experiment_results/{dataset}_{method}_run{N}_live.log`: Live training log
