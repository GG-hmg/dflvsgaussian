import argparse
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import torch

# utils.py imports parse_args() at module import time.
# Temporarily reset argv to avoid consuming this script's CLI args.
_argv_backup = sys.argv[:]
sys.argv = [sys.argv[0]]
from utils import generate_dfl_gaussian_noise, generate_ziggurat_gaussian_noise
sys.argv = _argv_backup


def build_normal_gaussian_with_rand(num_samples: int) -> np.ndarray:
    """Use Python rand (random.random) + Ziggurat Gaussian sampler."""
    uniform_vals = torch.tensor(
        [random.random() for _ in range(2 * num_samples)],
        dtype=torch.float32
    )
    gaussian = generate_ziggurat_gaussian_noise(torch.Size([num_samples]), uniform_vals)
    return gaussian.detach().cpu().numpy()


def build_chaotic_gaussian(num_samples: int) -> np.ndarray:
    """Use existing DFL chaotic Gaussian generator (internally uses Ziggurat)."""
    chaotic = generate_dfl_gaussian_noise(
        shape=torch.Size([num_samples]),
        mu=3.99,
        alpha=0.98,
        x0=random.random(),
        x1=random.random(),
    )
    return chaotic.detach().cpu().numpy()


def build_chaotic_gaussian_decorrelated(
        num_samples: int,
        dfl_mu: float,
        dfl_alpha: float,
        dfl_burn_in: int,
        dfl_jitter: float,
        dfl_max_direct_uniform: int
) -> np.ndarray:
    """Use fixed DFL generator from utils.py with decorrelation parameters."""
    chaotic = generate_dfl_gaussian_noise(
        shape=torch.Size([num_samples]),
        mu=dfl_mu,
        alpha=dfl_alpha,
        x0=random.random(),
        x1=random.random(),
        burn_in=dfl_burn_in,
        jitter=dfl_jitter,
        max_direct_uniform=dfl_max_direct_uniform,
    )
    return chaotic.detach().cpu().numpy()


def lag1_autocorr(seq: np.ndarray) -> float:
    if len(seq) < 2:
        return float("nan")
    x0 = seq[:-1]
    x1 = seq[1:]
    s0 = np.std(x0)
    s1 = np.std(x1)
    if s0 < 1e-12 or s1 < 1e-12:
        return float("nan")
    return float(np.corrcoef(x0, x1)[0, 1])


def print_diagnostics(name: str, seq: np.ndarray) -> None:
    ks_stat, ks_p = stats.kstest(seq, "norm", args=(0, 1))
    print(
        f"{name}: mean={seq.mean():.6f}, std={seq.std():.6f}, "
        f"lag1={lag1_autocorr(seq):.6f}, KS_stat={ks_stat:.6f}, KS_p={ks_p:.6f}"
    )


def plot_distributions(normal_seq: np.ndarray, chaotic_seq: np.ndarray, output_path: str) -> None:
    x = np.linspace(-5, 5, 500)
    theory_pdf = stats.norm.pdf(x, 0, 1)
    bins = np.linspace(-5, 5, 80)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].hist(normal_seq, bins=bins, density=True, alpha=0.55,
                 label="Normal (rand + Ziggurat)")
    axes[0].hist(chaotic_seq, bins=bins, density=True, alpha=0.55,
                 label="Chaotic Gaussian (DFL + Ziggurat)")
    axes[0].plot(x, theory_pdf, "k--", linewidth=2, label="N(0,1) PDF")
    axes[0].set_title("Histogram PDF Comparison")
    axes[0].set_xlabel("Value")
    axes[0].set_ylabel("Density")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    normal_kde = stats.gaussian_kde(normal_seq)
    chaotic_kde = stats.gaussian_kde(chaotic_seq)
    axes[1].plot(x, normal_kde(x), linewidth=2, label="Normal KDE")
    axes[1].plot(x, chaotic_kde(x), linewidth=2, label="Chaotic KDE")
    axes[1].plot(x, theory_pdf, "k--", linewidth=2, label="N(0,1) PDF")
    axes[1].set_title("KDE Comparison")
    axes[1].set_xlabel("Value")
    axes[1].set_ylabel("Density")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    fig.suptitle("Probability Distribution: Chaotic Gaussian vs Normal Gaussian", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot probability distributions of chaotic Gaussian and normal Gaussian sequences."
    )
    parser.add_argument("--num_samples", type=int, default=10000, help="Number of samples, default=10000")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="chaotic_vs_normal_gaussian_distribution.png",
                        help="Output image path")
    parser.add_argument("--use_legacy_chaotic", action="store_true",
                        help="Use old chaotic generator without decorrelation")
    parser.add_argument("--dfl_mu", type=float, default=3.99, help="DFL control parameter mu")
    parser.add_argument("--dfl_alpha", type=float, default=0.98, help="DFL fractional order alpha")
    parser.add_argument("--dfl_burn_in", type=int, default=5000,
                        help="Burn-in steps before collecting chaotic samples")
    parser.add_argument("--dfl_jitter", type=float, default=1e-4,
                        help="Small uniform jitter amplitude in [0,1) domain")
    parser.add_argument("--dfl_max_direct_uniform", type=int, default=50000,
                        help="Max direct DFL uniforms before phase expansion (plot mode)")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    normal_seq = build_normal_gaussian_with_rand(args.num_samples)
    if args.use_legacy_chaotic:
        chaotic_seq = build_chaotic_gaussian(args.num_samples)
    else:
        chaotic_seq = build_chaotic_gaussian_decorrelated(
            num_samples=args.num_samples,
            dfl_mu=args.dfl_mu,
            dfl_alpha=args.dfl_alpha,
            dfl_burn_in=args.dfl_burn_in,
            dfl_jitter=args.dfl_jitter,
            dfl_max_direct_uniform=args.dfl_max_direct_uniform,
        )

    print_diagnostics("Normal sequence", normal_seq)
    print_diagnostics("Chaotic sequence", chaotic_seq)

    plot_distributions(normal_seq, chaotic_seq, args.output)
    print(f"Saved figure to: {args.output}")


if __name__ == "__main__":
    main()
