# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DynamicPFL is a research project studying **chaotic noise mechanisms** in Differential Privacy Federated Learning (DP-FL), built around the Discrete Fractional Logistic (DFL) map `x(n+1) = mod(a·x(n)·(1-x(n)) + b·x(n-k), 1)`.

**Current research question (2026-06-09)**: how do four different chaotic-noise mechanisms compare against a pure-Gaussian baseline? The study isolates two variables — **distribution shape** (Gaussian vs uniform) and **sample correlation** (iid vs chaotic) — across these comparisons:

| # | Treatment vs Gaussian | Variable being isolated |
|---|---|---|
| 1 | `dfl_uniform` — raw DFL, mean-0 std-1, bounded support | distribution shape (uniform vs Gaussian, with chaotic correlation) |
| 2 | `dfl_gaussian` — DFL via inverse-CDF → N(0,1) | sample correlation alone (same distribution) |
| 3 | `mix_dgauss_gauss` — √α·dfl_gaussian + √(1-α)·gaussian | within-family mixing (two Gaussian sources) |
| 4 | `mix_dgauss_dchaos` — √α·dfl_gaussian + √(1-α)·dfl_uniform | heterogeneous mixing (Gaussian + bounded chaotic) |

**Earlier directions** (superseded): pure DFL vs Gaussian (#2 absorbed it) and an α-sweep of `√α·DFL + √(1-α)·G` (#3 supersedes it). Earlier results were also tainted by a hard-coded `±0.22` method bias in the GIR evaluator that was removed on 2026-06-09.

## DP framing (read this before writing about ε)

The Gaussian mechanism's `σ = C·√(2·ln(1.25/δ))/ε` gives **(ε, δ)-DP only when the noise is iid Gaussian and applied once**. In this codebase the picture is more complicated; the following three caveats must all be acknowledged in any paper/report:

### Caveat 1 — only `noise_kind=gaussian` even fits the mechanism

- ✅ `noise_kind=gaussian` carries the formal DP guarantee at the stated ε.
- ⚠️ The other four kinds (`dfl_uniform`, `dfl_gaussian`, `mix_dgauss_gauss`, `mix_dgauss_dchaos`) **reuse the same σ magnitude for a fair noise-budget comparison but make no formal DP claim** — they are non-iid, non-Gaussian, or both, and fall outside the Gaussian-mechanism theorem entirely.

### Caveat 2 — the formula is per-application, not cumulative

Training applies noise on **every batch of every local epoch of every global round**, roughly `local_epoch × batches_per_epoch × global_epoch ≈ 4 × 500 × 30 = 60 000` applications per client. The textbook formula gives (ε, δ)-DP for **one** application; cumulative privacy loss across all applications composes (RDP / moment accountant) and the true total ε is **much larger** than the stated `target_epsilon`. Without an RDP accountant we cannot quote a meaningful cumulative ε.

### Caveat 3 — `adaptive_privacy_budget` isn't standard composition

`utils.py:adaptive_privacy_budget` allocates ε per client as `ε_target · √(s_i / Σs) · n` (clipped to `[0.3ε, 2ε]`), then scales so the budgets average to `ε_target`. For 3 equal-size clients each client ends up with budget ≈ ε_target. This is **per-client**, not a global system ε: the total privacy spend across all clients is roughly `n · ε_target`, not `ε_target`. Standard FL-DP composition would use a privacy accountant; this codebase does not.

### Practical implication

The numeric `target_epsilon` setting in `run_experiments.py` is best read as a **noise-magnitude knob** (it determines σ), not as a formal DP guarantee. The current default `target_epsilon=1000` is too loose to count as DP at all — it was chosen empirically so models can learn under matched noise magnitude across the 5 noise_kinds. **For this codebase as of 2026-06-10, do not claim any DP guarantee in the paper**; frame the work as "empirical comparison of noise injection mechanisms at matched magnitude". If a DP-claiming variant of the experiment is needed later, plug in an RDP accountant (Opacus / TF-Privacy) and pick a meaningful ε.

## Critical Platform Constraints

- **Python path**: Always use `/d/anaconda/envs/pp/python.exe` — the system `python` has no torch
- **GBK encoding**: Windows console uses GBK — **NEVER use emojis in print statements**
- **No parallel experiments**: Let one experiment finish before starting the next

## Common Commands

```bash
# Batch comparison: any subset of the 5 noise_kinds (gaussian + treatments)
/d/anaconda/envs/pp/python.exe run_experiments.py --datasets CIFAR10 --noise_kinds gaussian dfl_gaussian --epochs 30

# Comparison with a mix variant — also specify alpha
/d/anaconda/envs/pp/python.exe run_experiments.py --datasets CIFAR10 --noise_kinds gaussian mix_dgauss_dchaos --mix_alpha 0.5 --epochs 30

# Plot one treatment vs gaussian (auto-finds latest matching pair)
/d/anaconda/envs/pp/python.exe plot_experiment_results.py --dataset CIFAR10
/d/anaconda/envs/pp/python.exe plot_experiment_results.py --dataset CIFAR10 --noise_kind mix_dgauss_dchaos
/d/anaconda/envs/pp/python.exe plot_experiment_results.py --dataset CIFAR10 --timestamp 20260609_153045

# Single run direct via ours.py (no batch driver)
/d/anaconda/envs/pp/python.exe ours.py --dataset CIFAR10 --noise_kind dfl_uniform --global_epoch 30 ...

# Noise distribution plot (Ziggurat vs Inverse-CDF vs theory)
/d/anaconda/envs/pp/python.exe plot_chaotic_vs_gaussian_distribution.py --num_samples 10000
```

## File Naming Convention

One `run_experiments.py` invocation = one session = one timestamp. All artifacts share that timestamp:

```
experiment_results/
  {noise_kind}_{dataset}_{timestamp}.log    real-time training log (per noise_kind)
  {noise_kind}_{dataset}_{timestamp}.txt    final summary (per noise_kind)
  comparison_results_{timestamp}.json       aggregated JSON across noise_kinds
  comparison_summary_{timestamp}.csv        aggregated CSV across noise_kinds
  comparison_{dataset}_{treatment}_vs_gaussian_{timestamp}.png   chart from plot_experiment_results.py
```

Pattern: `{noise_kind}_{dataset}_YYYYMMDD_HHMMSS.{ext}`. Fixed seed `20260313` across every session; pass `--seed N` to `ours.py` directly for seed-variance ablation.

Legacy files from earlier naming schemes (`{dataset}_{method}_run{id}_live.log`, `dfl_{dataset}_*.log`, `_a20_run1`, etc.) are kept untouched as historical data but are no longer read by the current scripts.

## Architecture

```
Global Model → Local Training (Client)
              ↓
    Gradient Clipping (optional, --no_clip to skip)
              ↓
    Noise Injection (selected by --noise_kind)
              ↓
    Aggregation (FedAvg) → Global Model Update
```

### Key Modules

**utils.py** — Noise Engine (no top-level args; safe to import without sys.argv)
- `_get_pure_dfl_sequences(a, b, k, seed_val, needed_length)`: Pure DFL map. Pre-computes one sequence per `(a,b,k)` (cached); each call picks a start_idx hashed from `seed_val`.
- `_extract_dfl_uniform_sequence(numel, ...)`: shared preprocessing (burn-in + decimation + degeneracy guard + adjacent-pair coupling) — returns DFL samples in `[0, 1)`.
- `generate_dfl_uniform_noise(shape, ...)`: `(u - 0.5) * sqrt(12)` → mean-0 std-1 bounded chaotic noise. Used by `dfl_uniform` and the second term of `mix_dgauss_dchaos`.
- `generate_dfl_gaussian_noise(shape, ...)`: DFL → inverse-CDF Gaussian. Used by `dfl_gaussian` and the first term of both `mix_*` kinds.
- `generate_random_gaussian_noise(shape, ...)`: pure iid Gaussian via `torch.rand` → inverse-CDF. Used by `gaussian` and the second term of `mix_dgauss_gauss`.
- `generate_random_gaussian_noise_like(reference_tensor)`: convenience wrapper.
- `generate_inverse_cdf_gaussian_noise(shape, uniform_sequence)`: erfinv inverse-CDF — the active Gaussian sampler.

**ours.py** — Training Pipeline
- `_build_unit_noise(numel, noise_kind, mix_alpha, ...)`: noise dispatcher — produces a unit-variance flat tensor for any of the 5 noise_kinds. For `mix_dgauss_dchaos`, the two DFL sources get independent x0 via the `_x0_from_seed(..., salt=0|1)` helper so they pull from different start positions in the shared `(a,b,k)` DFL cache.
- `_add_dp_noise_inplace(model, sigma, noise_kind, mix_alpha, ...)`: builds the unit noise, multiplies by sigma, splits across parameters in-place.
- `local_update_with_dp()`: orchestrator. Each batch: forward → backward → clip → noise → **snapshot real noisy grads** → GIR eval → sparsity → optimizer.step. Reads `args.noise_kind` and `args.mix_alpha` directly (no per-call params).
- `adaptive_noise_scale(client_epsilon, delta, clipping_bound)`: `σ = C·√(2 ln(1.25/δ))/ε` — only valid as a formal DP guarantee for `noise_kind=gaussian`; other kinds reuse this σ for fair comparison.
- Constant learning rate `args.lr`, no decay.

**gradient_inversion_risk_simulator.py** — Privacy Evaluator
- `simulate_gradient_inversion_risk(..., real_noisy_grads=None)`: runs a gradient-matching inversion attack on a deep-copied model. Returns leakage / PSNR / MSE; "anti-inversion ability" = `perturb_score * (1 - leakage_risk)`.
- **When `real_noisy_grads` is supplied** (the actual clipped+noised grads from training), they are used verbatim as the attacker-observed gradient.
- The fallback path (no `real_noisy_grads`) applies a generic Gaussian-sigma defense — used only for unit tests.
- `DefenseSimulationConfig` no longer carries `dp_method` / `use_chaotic` / `chaotic_factor` / `dfl_*` fields: the simulator is now fully noise-kind agnostic.
- **No method-aware biases**: the `±0.22` post-hoc adjustment and the Gaussian-only Wiener-denoising stronger-attack path were both removed (they systematically tilted comparisons in favour of DFL).

**run_experiments.py** — Batch Runner
- Single per-dataset training config shared across all noise_kinds (no per-method nesting) — only the noise injection logic changes between runs.
- CIFAR10 defaults: lr=0.004, clip=2.0, ε=8.0, k=3, gap=11, sparsity=0.0, epochs=30.
- One session = one timestamp = one fixed seed `20260313`. `--noise_kinds` controls which mechanisms run; `--mix_alpha` controls α for `mix_*` variants.

**data.py** — Federated Data Loading (no top-level args; loaders accept `batch_size` as a parameter)
- MNIST: IID partition via `get_mnist_datasets()` + `get_clients_datasets()`.
- CIFAR10 / SVHN / FashionMNIST: heterogeneous Dirichlet (`hetero_dir_partition`).
- FEMNIST removed (was TF-dependent and broken).

**net.py** — Models: `cifar10Net` (3 conv+3 fc), `mnistNet`, `SVHNNet`, `fashionmnistNet`.

**plot_experiment_results.py** — Plots one treatment noise_kind vs gaussian baseline. Auto-finds the most recent session where both files exist; pass `--noise_kind X` to pick which treatment, `--timestamp YYYYMMDD_HHMMSS` to plot a historical session.

## Key Parameters (CIFAR10, current)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `noise_kind` | sweep | One of `gaussian` / `dfl_uniform` / `dfl_gaussian` / `mix_dgauss_gauss` / `mix_dgauss_dchaos`. Only `gaussian` has formal DP. |
| `mix_alpha` | 0.5 | α ∈ [0,1] for `mix_*` variants. Mix formula: `√α·A + √(1-α)·B` (variance-preserving when A, B independent unit-variance). |
| `clipping_bound` | 2.0 | |
| `lr` | 0.004 | Constant (no decay) |
| `dfl_a, dfl_b, dfl_k` | 4.0, 501.0, 3 | DFL map parameters |
| `dfl_decimation` | 11 | Gap factor for decorrelation |
| `dfl_burn_in` | 2048 | Burn-in steps before collecting samples |
| `target_epsilon` | 1000.0 | One-shot Gaussian mechanism σ ≈ 0.0097 at C=2.0, δ=1e-5. **Effectively no DP guarantee.** Settings of ε ≤ 200 made σ·√N (noise norm ≈ 283 at ε=20 for the 343k-param CIFAR10 net) hugely overwhelm the clipped signal norm (= 2.0), so weights diverged and loss exploded to 100+ within 5 epochs. ε=1000 puts noise norm ≈ 5.7, low enough that the model actually learns. The research goal is comparing 5 noise mechanisms at matched magnitude — not proving a DP bound. |
| `gir_attack_trials` | 5 | Per-epoch anti-inversion score is averaged over this many inversion attacks. With trials=1 the single-attack noise dominated (~0.46–0.74 swings); trials=5 cuts that to roughly ±0.05. |
| `epochs` | 30 | |
| `seed` | 20260313 | Fixed across all `run_experiments.py` sessions. |

**Expected accuracy under current settings**: at ε=1000 / C=2.0 / σ≈0.0097, CIFAR10 should reach roughly the same ceiling as a clean (non-DP) run (~70-75%), with mechanism differences showing up as small but measurable accuracy/anti-inversion deltas across the 5 noise_kinds.

## Sampling Comparison (2026-05-27, historical)

| Method | Ziggurat | Inverse-CDF |
|--------|----------|-------------|
| DFL accuracy | 65.42% | 65.60% |
| Gaussian accuracy | 76.27% | 76.02% |
| Gaussian time | 14522s | 4305s |

**Conclusion**: Sampling method has negligible effect on accuracy (<0.3% difference). Inverse-CDF is 3x faster — the active sampler in training. These numbers predate the DP fix and the GIR bias removal.

## Pending control experiments

To rule out alternative explanations for any future "treatment beats gaussian" finding:
- **Multi-seed gaussian baseline** (3–5 seeds) to estimate the run-to-run variance band of the anti-inversion score under iid noise. Required because single-seed `gir_attack_trials=1` per-epoch variance is large (~0.46–0.74 observed in older runs).
- **`gir_attack_trials=5+`** to lower per-epoch measurement noise.
- **#3 vs an iid-only mix `(G1+G2)/√2`** to isolate the "two-source independence" effect from the chaotic correlation effect — i.e. if `mix_dgauss_gauss` improves over `gaussian`, does an iid two-source mix also? If yes, chaos isn't the cause.
