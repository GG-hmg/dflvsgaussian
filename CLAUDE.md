# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DynamicPFL is a research project studying **Discrete Fractional Logistic (DFL) chaotic noise** in Differential Privacy Federated Learning (DP-FL).

**Current research question (2026-06-09)**: does **mixing** DFL chaotic noise with Gaussian noise (variance-preserving combo `√α·DFL + √(1-α)·G`) achieve a better privacy-utility tradeoff than pure Gaussian? The `chaotic_factor=α` knob exists in [ours.py:334-340](ours.py); α=0 is pure Gaussian, α=1 is pure DFL.

**Earlier research question** (superseded): can *pure* DFL noise outperform Gaussian noise? Empirically: pure DFL loses ~10% accuracy on CIFAR10 with no clear privacy win — see "α-sweep results" section.

## Critical Platform Constraints

- **Python path**: Always use `/d/anaconda/envs/pp/python.exe` — the system `python` has no torch
- **GBK encoding**: Windows console uses GBK — **NEVER use emojis in print statements**
- **No parallel experiments**: Let one experiment finish before starting the next

## Common Commands

```bash
# Batch comparison (DFL then Gaussian, sequential) — single run per session
/d/anaconda/envs/pp/python.exe run_experiments.py --datasets CIFAR10 --dp_methods dfl gaussian --epochs 30

# α-sweep: override chaotic_factor without editing config
/d/anaconda/envs/pp/python.exe run_experiments.py --datasets CIFAR10 --dp_methods dfl gaussian --epochs 30 --chaotic_factor 0.5

# Generate comparison chart (auto-finds latest matching pair by timestamp)
/d/anaconda/envs/pp/python.exe plot_experiment_results.py --dataset CIFAR10

# Plot a specific past session
/d/anaconda/envs/pp/python.exe plot_experiment_results.py --dataset CIFAR10 --timestamp 20260609_153045

# Single dataset, single method (direct ours.py — not via run_experiments)
/d/anaconda/envs/pp/python.exe ours.py --dataset CIFAR10 --dp_method dfl --global_epoch 30 ...

# Noise distribution plot (Ziggurat vs Inverse-CDF vs theory)
/d/anaconda/envs/pp/python.exe plot_chaotic_vs_gaussian_distribution.py --num_samples 10000
```

## File Naming Convention (2026-06-09 migration)

One `run_experiments.py` invocation = one session = one timestamp. All artifacts share that timestamp:

```
experiment_results/
  dfl_CIFAR10_{timestamp}.log         ← DFL real-time terminal output
  dfl_CIFAR10_{timestamp}.txt         ← DFL final summary
  gaussian_CIFAR10_{timestamp}.log    ← Gaussian real-time output
  gaussian_CIFAR10_{timestamp}.txt    ← Gaussian final summary
  comparison_CIFAR10_{timestamp}.png  ← side-by-side accuracy + anti-inversion chart
  comparison_results_{timestamp}.json
  comparison_summary_{timestamp}.csv
```

Pattern: `{method}_{dataset}_YYYYMMDD_HHMMSS.{ext}`. **No `_run{id}` suffix anymore** — each session has a unique timestamp, and `--num_runs` / `--run_id` were removed (every session uses a fixed seed `20260313`; for seed-variance ablation, pass `--seed N` to `ours.py` directly).

Legacy files (`{dataset}_{method}_run{id}_live.log`, `_a20_run1`, etc.) are kept untouched as historical data. `plot_experiment_results.py` auto-finds new format; falls back to legacy if no new files present.

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

**utils.py** — Noise Engine (no top-level args; safe to import without sys.argv)
- `_get_pure_dfl_sequences(a, b, k, seed_val, needed_length)`: Pure DFL map `x(n+1) = mod(a*x(n)*(1-x(n)) + b*x(n-k), 1.0)`. Pre-computes sequence once (cache key: `(a,b,k)`), then `start_idx = int(seed_val * max_start) % max_start` (single hash from the caller-side `(client, epoch, batch)` Knuth-multiply).
- `generate_dfl_gaussian_noise(shape, a, b, k, x0, burn_in, decimation, device)`: Extracts DFL sequence (5x oversampling), applies burn-in + decimation, degeneracy guard, adjacent-pair coupling, feeds to inverse-CDF Gaussian sampler. If `device` is set, runs sampler on GPU.
- `generate_inverse_cdf_gaussian_noise(shape, uniform_sequence)`: erfinv inverse-CDF — the active Gaussian sampler used by both training paths.
- `generate_ziggurat_gaussian_noise(shape, uniform_sequence)`: Marsaglia-Tsang (2000) Ziggurat. **Not used in training** — kept for the noise-distribution diagnostic plot only (`plot_chaotic_vs_gaussian_distribution.py`).
- `generate_random_gaussian_noise(shape, device, dtype)`: pure-Gaussian path — torch.rand → inverse-CDF.
- `generate_random_gaussian_noise_like(reference_tensor)`: Convenience wrapper, matches device/dtype.

**ours.py** — Training Pipeline
- `local_update_with_dp()`: Orchestrator. Each batch: forward → backward → clip → noise → **snapshot real noisy grads** → GIR eval → sparsity → optimizer.step. Helpers:
  - `_clip_gradients_inplace(model, bound)` — global L2 clip
  - `_add_dp_noise_inplace(...)` — pure-Gaussian or DFL+Gaussian mix
  - `_snapshot_grads(model)` — capture transmitted-grad copy
  - `_run_gir_evaluation(..., real_noisy_grads=...)` — feeds the actual training noise into the attacker, not a freshly resampled noise stream
  - `_apply_sparsity_inplace(model, ratio)`
- `adaptive_noise_scale()`: σ = C·sqrt(2 ln(1.25/δ))/ε — one-shot Gaussian mechanism. No sigma_factor, no decay, no cap.
- `generate_chaotic_noise_v2()`: Thin wrapper for DFL Gaussian. Returns standard-normal directly (no extra std normalization).
- Constant learning rate (`args.lr`) — cosine decay removed; the mix study doesn't need it.
- GIR runs at most `gir_max_evals_per_client_update` times per update.

**gradient_inversion_risk_simulator.py** — Privacy Evaluator
- `simulate_gradient_inversion_risk(..., real_noisy_grads=None)`: Deep-copies the model, runs gradient-matching inversion attacks, returns leakage/PSNR/MSE scores. "Anti-Inversion Ability" = defense score (higher = better).
- **When `real_noisy_grads` is supplied** (the actual clipped+noised grads from training), they are used as the attacker-observed gradient — that is what the attacker genuinely sees.
- When omitted, falls back to regenerating noise inside the simulator via `defense_cfg` (useful for unit tests but not for true measurement).
- **NOTE**: the simulator still applies a `±0.22` method-aware adjustment (line ~459) that adds risk to Gaussian and subtracts from DFL. This is a known issue (review #1) — not yet removed pending discussion.

**run_experiments.py** — Batch Runner
- Per-dataset, per-method parameter dicts. Validates sigma/epsilon equality between methods.
- CIFAR10 params: lr=0.004, sigma=0.01, clip=2.0, k=3, gap=11, sparsity=0.0, epochs=30.
- Other datasets use sigma=0.30, gap=8, sparsity=0.4 (not yet aligned with CIFAR10 tuning).
- Single run per session (no `--num_runs` / `--run_id`). Fixed seed `20260313`. Session timestamp generated at init, shared by all output artifacts. CLI supports `--chaotic_factor α` to override DFL mix ratio without editing config.

**data.py** — Federated Data Loading (no top-level args; loaders accept `batch_size` as a parameter)
- MNIST: IID partition via `get_mnist_datasets()` + `get_clients_datasets()`.
- CIFAR10 / SVHN / FashionMNIST: heterogeneous Dirichlet (`hetero_dir_partition`) — signatures end with `batch_size: int = 32`.
- FEMNIST removed (was TF-dependent and broken).

**net.py** — Models: `cifar10Net` (3 conv+3 fc), `mnistNet` (fixed: ReLU between conv1/conv2, fc2 out=10), `SVHNNet`, `fashionmnistNet`. Previously broken `femnistNet` removed.

**plot_experiment_results.py** — Reads `experiment_results/` output, generates accuracy + anti-inversion charts. Auto-finds the most recent `dfl_/gaussian_` pair by shared timestamp; pass `--timestamp YYYYMMDD_HHMMSS` to plot a historical session. Falls back to legacy `{dataset}_{method}_run1_*` paths when no new-format files are found.

## Key Parameters (CIFAR10, current)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `clipping_bound` | 2.0 | |
| `lr` | 0.004 | Constant (no decay) |
| `dfl_a, dfl_b, dfl_k` | 4.0, 501.0, 3 | DFL map parameters |
| `dfl_decimation` | 11 | Gap factor for decorrelation |
| `dfl_burn_in` | 2048 | Burn-in steps before collecting samples |
| `chaotic_factor` | sweep | Mix ratio α ∈ [0,1]. α=0 pure Gaussian, α=1 pure DFL. Mix formula: `√α·DFL + √(1-α)·G` (variance-preserving). |
| `target_epsilon` | 8.0 | One-shot Gaussian mechanism σ ≈ 1.22 at C=2.0, δ=1e-5. No sigma_factor multiplier — earlier code diluted noise by ~100× and broke the DP claim. |
| `epochs` | 30 | |
| `seed` | 20260313 | Fixed across all `run_experiments.py` sessions. Pass `--seed N` to `ours.py` for seed-variance ablation. |

**Expected accuracy under real DP (post-cleanup)**: at ε=8 / C=2.0 / σ≈1.22, CIFAR10 typically converges to ~10–30% (vs ~75% under the old sigma_factor=0.01 setup). All α-sweep results recorded BEFORE this DP fix are invalidated — they ran under a much weaker noise regime.

## Sampling Comparison (2026-05-27)

| Method | Ziggurat | Inverse-CDF |
|--------|----------|-------------|
| DFL accuracy | 65.42% | 65.60% |
| Gaussian accuracy | 76.27% | 76.02% |
| Gaussian time | 14522s | 4305s |

**Conclusion**: Sampling method has negligible effect on accuracy (<0.3% difference). Inverse-CDF is 3x faster. The Gaussian-vs-DFL accuracy gap (~10%) is from the noise source, not the sampler.

## α-sweep results (2026-06-09, CIFAR10, 30 epochs, single seed)

| α | Final accuracy | Final anti-inversion | Training time | Notes |
|---|---|---|---|---|
| 0.00 (pure Gaussian) | 75.36% | 0.5653 | (May baseline) | DP-formal guarantee |
| 0.20 mix | 74.82% | 0.6344 | 83.7 min | −0.54% acc, +0.069 anti |
| 0.80 mix | 74.86% | 0.6437 | 77.6 min | −0.50% acc, +0.078 anti |
| 1.00 (pure DFL) | 69.59% (14 ep) | 0.5612 | (partial) | Earlier run; not converged |

**Observation**: α=0.2 and α=0.8 yield nearly identical results (acc Δ0.04%, anti Δ0.009). This is a yellow flag — if the mix ratio doesn't matter, the +0.07 anti improvement may not be due to chaotic correlations.

**Three plausible explanations (in order of likelihood)**:
1. **Run-to-run variance, not signal** — single seed + `gir_attack_trials=1` gives huge per-epoch anti-inversion variance (0.46~0.74 observed). Gaussian baseline anti-inversion may also be 0.56±0.05; the 0.07 gap is in noise.
2. **"Two-source noise" effect, not chaos** — DFL noise is already Gaussian-distributed (Ziggurat output); mixing = two independent Gaussian sources combined. Possible improvement from sampling independence rather than chaotic correlation.
3. **Chaotic correlation genuinely helps** — weakest explanation; needs stronger evidence.

**Required control experiments before claiming chaos works**:
- Formula C (independent Gaussian additive): `(G1 + G2)/√2` — no chaos, just two-source. If C ≈ mix, kills the chaos claim.
- Multi-seed Gaussian baseline (3-5 seeds) to estimate true variance.
- `gir_attack_trials=5+` to reduce per-epoch measurement noise.
