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
- `generate_dfl_sequence()`: Core DFL map iteration (`x[n+1] = alpha*mu*x[n]*(1-x[n]) + (1-alpha)*x[n-1]`)
- `generate_dfl_gaussian_noise()`: Combines two DFL sequences with different seeds, applies Ziggurat sampling
- `generate_ziggurat_gaussian_noise()`: Vectorized inverse-CDF sampling via `torch.erfinv()`
- **Security fix**: Line 368 uses `torch.rand(reps)` for phase generation (not deterministic `k*c`) to prevent iDLG attack correlation

**ours.py** - Main Training Loop
- `local_update_with_dp()`: Client-side training with DP defense
- DFL noise mixing: `flat_unit_noise = sqrt(chaotic_factor) * DFL_noise + sqrt(1-chaotic_factor) * Gaussian_noise`
- Integrates with `gradient_inversion_risk_simulator.py` for real-time risk evaluation

**gradient_inversion_risk_simulator.py** - Privacy Evaluator
- `simulate_gradient_inversion_risk()`: Paper-style gradient inversion attack
- Measures PSNR/MSE between reconstructed and original images
- Computes "Anti-Inversion Ability" / defense_score (higher = better privacy)

**run_experiments.py** - Batch Experiment Runner
- Runs comparative experiments across datasets
- **CIFAR10 config**: sigma=0.06, chaotic_factor=1.0 (pure DFL), dfl_alpha=0.99
- Outputs to `experiment_results/`

## Gradient Inversion Attack

The project implements a **gradient inversion attack simulator** to evaluate privacy protection effectiveness.

### Attack Mechanism

1. **Observe Gradients**: Compute clean gradients, apply defense (clipping + noise + sparsity)
2. **Run Inversion Trials**: For each trial:
   - Initialize dummy input `x_dummy` with random uniform noise
   - Optimize `x_dummy` to match observed gradients using gradient descent
   - Objective: `gradient_matching_loss + tv_weight * total_variation(x_dummy)`
3. **Evaluate Reconstruction**: Compare `x_dummy` to true input using PSNR and MSE
4. **Combine Results**: Average and worst-case leakage scores combined (50/50)

### Leakage Score Calculation

```
psnr_score = 1 / (1 + exp(-(psnr - 18.0) / 3.0))
mse_score = 1 / (1 + exp((mse - 0.02) / 0.01))
leakage = 0.7 * psnr_score + 0.3 * mse_score
combined_risk = 0.5 * avg_leakage + 0.5 * worst_leakage
defense_score = 1.0 - combined_risk
```

## Common Commands

### Single Experiment
```bash
# Pure DFL (chaotic_factor=1.0)
python ours.py --dataset CIFAR10 --dp_method dfl --global_epoch 60 --sigma_factor_dfl 0.06 --chaotic_factor 1.0

# Gaussian baseline
python ours.py --dataset CIFAR10 --dp_method gaussian --global_epoch 60 --sigma_factor_gaussian 0.06
```

### Batch Comparison
```bash
# DFL vs Gaussian on CIFAR10
python run_experiments.py --datasets CIFAR10 --dp_methods dfl gaussian --num_runs 1 --epochs 60
```

### Parameter Optimization
```bash
python grid_search.py --target balanced
python grid_search.py --quick
```

## DFL Implementation Details

The DFL noise generation (`utils.py:generate_dfl_gaussian_noise`):

1. **Generate two DFL sequences** with different seeds (x0, x1)
2. **Mix sequences**: `u = (seq1 + 0.618 * seq2) % 1.0` (golden ratio)
3. **Apply burn-in**: discard initial samples
4. **Apply decimation (gap)**: break fractional-order correlation
5. **Add jitter**: prevent ties
6. **Degeneracy check**: inject noise if too few bins occupied
7. **Phase expansion**: repeat with **random** phase shifts (security fix)
8. **Ziggurat sampling**: `sqrt(2) * erfinv(2u-1)` → Gaussian

## Key Parameters

| Parameter | Range | Current CIFAR10 | Purpose |
|-----------|-------|-----------------|---------|
| `--chaotic_factor` | [0, 1] | 1.0 (pure DFL) | Mix ratio: 0=Gaussian, 1=DFL |
| `--sigma_factor_dfl` | > 0 | 0.06 | Noise intensity |
| `--dfl_alpha` | (0, 1] | 0.99 | Memory depth (higher=more memory) |
| `--dfl_mu` | [3.57, 4.0] | 3.99 | Chaos level (higher=more chaotic) |
| `--dfl_decimation` | ≥1 | 2 | Gap factor for correlation breaking |
| `--clipping_bound` | > 0 | 2.0 | Gradient clipping threshold |
| `--target_epsilon` | > 0 | 8.0 | Privacy budget |

## Experimental Results Summary

| Config | sigma | chaotic_factor | DFL Acc | Gaussian Acc | DFL Defense | Gaussian Defense |
|--------|-------|----------------|---------|--------------|-------------|-----------------|
| Hard (博弈区) | 0.15 | 0.2 | 54.58% | 49.45% | 0.7072 | 0.5878 |
| Medium | 0.08 | 0.2 | 57.34% | 53.46% | 0.7763 | 0.5685 |
| Current | 0.06 | 1.0 (pure) | pending | pending | pending | pending |

**Key findings**:
- DFL consistently outperforms Gaussian in both accuracy and defense
- Higher sigma (harder privacy regime) amplifies DFL's advantages

## Data Handling

- MNIST: Simple CNN (2 conv layers)
- CIFAR10: ResNet-like architecture
- SVHN: Similar to CIFAR10
- Datasets stored in `./data/`, downloaded on first run
- Non-IID distribution via Dirichlet (`--dir_alpha`, default 100)

## Console Output

The code includes Windows GBK encoding fixes. Output files use UTF-8.

## Output Files

- `experiment_results/{dataset}_{method}_run{N}_output.txt`: Full metrics
- `experiment_results/{dataset}_{method}_run{N}_live.log`: Live training log
