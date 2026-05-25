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
from utils import generate_delayed_feedback_noise, generate_inverse_cdf_gaussian_noise
sys.argv = _argv_backup


def build_normal_gaussian_with_rand(num_samples: int) -> np.ndarray:
    """Use Python rand (random.random) + inverse-CDF Gaussian sampler."""
    uniform_vals = torch.tensor(
        [random.random() for _ in range(2 * num_samples)],
        dtype=torch.float32
    )  
    gaussian = generate_inverse_cdf_gaussian_noise(torch.Size([num_samples]), uniform_vals)
    return gaussian.detach().cpu().numpy()


def build_delayed_feedback_sequence(
        num_samples: int,
        a: float,
        b: float,
        k: int,
        burn_in: int,
        decimation: int,
        q_bits: int,
        q_mode: str,
        map_type: str,
) -> np.ndarray:
    """Use the delayed-feedback generator from utils.py."""
    feedback = generate_delayed_feedback_noise(
        shape=torch.Size([num_samples]),
        a=a,
        b=b,
        x0=random.random(),
        k=k,
        burn_in=burn_in,
        decimation=decimation,
        q_bits=q_bits,
        q_mode=q_mode,
        map_type=map_type,
    )
    return feedback.detach().cpu().numpy()


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


def plot_distributions(normal_seq: np.ndarray, feedback_seq: np.ndarray, output_path: str) -> None:
    x = np.linspace(-5, 5, 500)
    theory_pdf = stats.norm.pdf(x, 0, 1)
    bins = np.linspace(-5, 5, 80)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].hist(normal_seq, bins=bins, density=True, alpha=0.55,
                 label="Pure Gaussian")
    axes[0].hist(feedback_seq, bins=bins, density=True, alpha=0.55,
                 label="Delayed Feedback")
    axes[0].plot(x, theory_pdf, "k--", linewidth=2, label="N(0,1) PDF")
    axes[0].set_title("Pure Gaussian vs Delayed Feedback")
    axes[0].set_xlabel("Value")
    axes[0].set_ylabel("Density")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    normal_kde = stats.gaussian_kde(normal_seq)
    feedback_kde = stats.gaussian_kde(feedback_seq)
    axes[1].plot(x, normal_kde(x), linewidth=2, label="Pure Gaussian KDE")
    axes[1].plot(x, feedback_kde(x), linewidth=2, label="Delayed Feedback KDE")
    axes[1].plot(x, theory_pdf, "k--", linewidth=2, label="N(0,1) PDF")
    axes[1].set_title("Distribution Comparison")
    axes[1].set_xlabel("Value")
    axes[1].set_ylabel("Density")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    fig.suptitle("Probability Distribution: Delayed Feedback vs Pure Gaussian", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot probability distributions of delayed-feedback and pure Gaussian sequences."
    )
    parser.add_argument("--num_samples", type=int, default=10000, help="Number of samples, default=10000")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="delayed_feedback_vs_pure_gaussian_distribution.png",
                        help="Output image path")
    parser.add_argument("--map_type", type=str, default="logistic", choices=["logistic", "tent", "sine"],
                        help="Seed map f(x) inside delayed feedback")

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    normal_seq = build_normal_gaussian_with_rand(args.num_samples)
    feedback_seq = build_delayed_feedback_sequence(
        num_samples=args.num_samples,
        a=4.0,
        b=501.0,
        k=4,
        burn_in=2048,
        decimation=12,
        q_bits=32,
        q_mode="round",
        map_type=args.map_type,
    )

    print_diagnostics("Normal sequence", normal_seq)
    print_diagnostics("Delayed feedback sequence", feedback_seq)

    plot_distributions(normal_seq, feedback_seq, args.output)
    print(f"Saved figure to: {args.output}")


if __name__ == "__main__":
    main()

