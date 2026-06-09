"""
Plot one noise_kind (treatment) vs gaussian (baseline) for a given dataset.

Reads files written by run_experiments.py:
    {noise_kind}_{dataset}_{timestamp}.log    real-time log
    {noise_kind}_{dataset}_{timestamp}.txt    final summary

Usage:
    python plot_experiment_results.py --dataset CIFAR10
        Auto-find latest session; plot the first non-gaussian noise_kind vs gaussian.
    python plot_experiment_results.py --dataset CIFAR10 --noise_kind dfl_gaussian
        Pick a specific treatment noise_kind.
    python plot_experiment_results.py --dataset CIFAR10 --timestamp 20260609_120000
        Plot a specific historical session.
"""

import os
import re
import glob
import argparse
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np


VALID_NOISE_KINDS = [
    "gaussian", "dfl_uniform", "dfl_gaussian",
    "mix_dgauss_gauss", "mix_dgauss_dchaos",
]
TREATMENT_KINDS = [k for k in VALID_NOISE_KINDS if k != "gaussian"]


def parse_output_file(output_path):
    """Parse accuracy + anti-inversion history from a summary .txt file."""
    if not os.path.exists(output_path):
        return [], [], []

    with open(output_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    accs = []
    m = re.search(r"平均准确率历程:\s*\[([^\]]+)\]", content)
    if m:
        s = m.group(1).replace('%', '').replace("'", "")
        accs = [float(x.strip()) for x in s.split(',') if x.strip()]

    antis = []
    m = re.search(r"抗梯度反演能力历程:\s*\[([^\]]+)\]", content)
    if m:
        s = m.group(1).replace('%', '').replace("'", "")
        antis = [float(x.strip()) for x in s.split(',') if x.strip()]

    epochs = list(range(1, len(accs) + 1)) if accs else []
    return epochs, accs, antis


def parse_live_log(log_path):
    """Parse per-epoch metrics from a live log (fallback when .txt missing/empty)."""
    if not os.path.exists(log_path):
        return [], [], []

    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # 轮次 N 客户端准确率: ['44.13%', ...]
    #         平均抗梯度反演能力: 0.4596
    pattern = r'轮次\s+(\d+)\s+客户端准确率:\s+\[(.*?)\].*?抗梯度反演能力:\s+([\d.]+)'
    matches = re.findall(pattern, content, re.DOTALL)
    epochs, accs, antis = [], [], []
    for ep_str, acc_list_str, anti_str in matches:
        acc_str = acc_list_str.replace('%', '').replace("'", "").replace(' ', '')
        vals = [float(x) for x in acc_str.split(',') if x]
        if vals:
            epochs.append(int(ep_str))
            accs.append(float(np.mean(vals)))
            antis.append(float(anti_str))
    return epochs, accs, antis


def read_experiment_output(output_path):
    """Pull final metrics + key params from a summary .txt file for the info bar."""
    results = {}
    if not os.path.exists(output_path):
        return results

    with open(output_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    for key, pat in [
        ('final_accuracy', r'最终准确率:\s*([\d.]+)%'),
        ('anti_inversion', r'最终抗梯度反演能力:\s*([\d.]+)'),
        ('training_time', r'总训练时间:\s*([\d.]+)秒'),
    ]:
        m = re.search(pat, content)
        if m:
            results[key] = float(m.group(1))

    # CLI args echoed in the log header
    for key, pat in [
        ('clipping_bound', r'裁剪边界:\s*([\d.]+)'),
        ('noise_kind', r'--noise_kind\s+(\S+)'),
        ('mix_alpha', r'--mix_alpha\s+([\d.]+)'),
        ('dfl_a', r'--dfl_a\s+([\d.]+)'),
        ('dfl_b', r'--dfl_b\s+([\d.]+)'),
        ('dfl_k', r'--dfl_k\s+(\d+)'),
        ('dfl_burn_in', r'--dfl_burn_in\s+(\d+)'),
        ('dfl_decimation', r'--dfl_decimation\s+(\d+)'),
        ('target_epsilon', r'隐私预算ε:\s*([\d.]+)'),
    ]:
        m = re.search(pat, content)
        if m:
            try:
                results[key] = float(m.group(1)) if '.' in m.group(1) else int(m.group(1))
            except ValueError:
                results[key] = m.group(1)
    return results


def find_session_files(results_dir, dataset, treatment_kind, timestamp=None):
    """
    Return (timestamp, treatment_log, gaussian_log, treatment_txt, gaussian_txt).

    If `timestamp` is None: auto-find the most recent session where both
    `{treatment_kind}_{dataset}_*.log` and `gaussian_{dataset}_*.log` exist.
    """
    if timestamp is None:
        pattern = os.path.join(results_dir, f'{treatment_kind}_{dataset}_*.log')
        candidates = []
        for path in glob.glob(pattern):
            m = re.search(rf'{re.escape(treatment_kind)}_{re.escape(dataset)}_(\d{{8}}_\d{{6}})\.log$',
                          os.path.basename(path))
            if not m:
                continue
            ts = m.group(1)
            if os.path.exists(os.path.join(results_dir, f'gaussian_{dataset}_{ts}.log')):
                candidates.append(ts)
        if not candidates:
            return None, None, None, None, None
        timestamp = max(candidates)

    return (
        timestamp,
        os.path.join(results_dir, f'{treatment_kind}_{dataset}_{timestamp}.log'),
        os.path.join(results_dir, f'gaussian_{dataset}_{timestamp}.log'),
        os.path.join(results_dir, f'{treatment_kind}_{dataset}_{timestamp}.txt'),
        os.path.join(results_dir, f'gaussian_{dataset}_{timestamp}.txt'),
    )


def autodetect_treatment_kind(results_dir, dataset, timestamp=None):
    """Find any treatment noise_kind that has a matching gaussian baseline."""
    for kind in TREATMENT_KINDS:
        ts, t_log, g_log, _, _ = find_session_files(results_dir, dataset, kind, timestamp)
        if ts is not None:
            return kind, ts
    return None, None


def plot_comparison(dataset, treatment_kind, t_log, g_log, t_txt, g_txt, save_path, params):
    t_epochs, t_accs, t_antis = parse_output_file(t_txt)
    g_epochs, g_accs, g_antis = parse_output_file(g_txt)
    if not t_accs:
        t_epochs, t_accs, t_antis = parse_live_log(t_log)
    if not g_accs:
        g_epochs, g_accs, g_antis = parse_live_log(g_log)

    t_results = read_experiment_output(t_txt)
    g_results = read_experiment_output(g_txt)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plt.rcParams['font.size'] = 10

    # ax1: accuracy
    if t_epochs:
        axes[0].plot(t_epochs, t_accs, 'b-', label=treatment_kind, linewidth=1, marker='o', markersize=2)
    if g_epochs:
        axes[0].plot(g_epochs, g_accs, 'r-', label='gaussian', linewidth=1, marker='s', markersize=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title(f'{dataset} - Accuracy Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    all_accs = t_accs + g_accs
    if all_accs:
        axes[0].set_ylim([max(0, min(all_accs) - 5), min(100, max(all_accs) + 5)])

    # ax2: anti-inversion
    if t_epochs:
        axes[1].plot(t_epochs, t_antis, 'b-', label=treatment_kind, linewidth=1, marker='o', markersize=2)
    if g_epochs:
        axes[1].plot(g_epochs, g_antis, 'r-', label='gaussian', linewidth=1, marker='s', markersize=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Anti-Inversion Ability')
    axes[1].set_title(f'{dataset} - Privacy Protection')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1.2])

    # Info bars: parameters at bottom, final results at top
    mix_alpha = params.get('mix_alpha')
    mix_part = f"mix_alpha={mix_alpha} | " if mix_alpha is not None and treatment_kind.startswith("mix_") else ""
    param_text = (
        f"noise_kind={treatment_kind} | {mix_part}"
        f"ε={params.get('target_epsilon', 'N/A')} | "
        f"clip={params.get('clipping_bound', 'N/A')} | "
        f"a={params.get('dfl_a', 'N/A')} | b={params.get('dfl_b', 'N/A')} | "
        f"k={params.get('dfl_k', 'N/A')} | "
        f"burn_in={params.get('dfl_burn_in', 'N/A')} | gap={params.get('dfl_decimation', 'N/A')}"
    )
    result_text = (
        f"{treatment_kind}: Acc={t_results.get('final_accuracy', 'N/A')}% | "
        f"Anti-Inv={t_results.get('anti_inversion', 'N/A')}  |  "
        f"gaussian: Acc={g_results.get('final_accuracy', 'N/A')}% | "
        f"Anti-Inv={g_results.get('anti_inversion', 'N/A')}"
    )
    fig.text(0.5, 0.02, param_text, ha='center', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    fig.text(0.5, 0.94, result_text, ha='center', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.tight_layout(rect=[0, 0.05, 1, 0.92])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved to: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot treatment noise_kind vs gaussian baseline')
    parser.add_argument('--dataset', type=str, default='CIFAR10')
    parser.add_argument('--noise_kind', type=str, default=None, choices=TREATMENT_KINDS,
                        help='Which non-gaussian noise_kind to plot vs gaussian (auto-detect if omitted)')
    parser.add_argument('--timestamp', type=str, default=None,
                        help='Session timestamp YYYYMMDD_HHMMSS (auto-find latest if omitted)')
    parser.add_argument('--results_dir', type=str, default='experiment_results')
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    treatment_kind = args.noise_kind
    if treatment_kind is None:
        treatment_kind, _ = autodetect_treatment_kind(args.results_dir, args.dataset, args.timestamp)
        if treatment_kind is None:
            print(f"No treatment noise_kind found with matching gaussian baseline for {args.dataset}.")
            return
        print(f"Auto-detected treatment noise_kind: {treatment_kind}")

    ts, t_log, g_log, t_txt, g_txt = find_session_files(
        args.results_dir, args.dataset, treatment_kind, args.timestamp,
    )
    if ts is None:
        print(f"No matching session files for {args.dataset}/{treatment_kind}.")
        return

    if args.output is None:
        args.output = os.path.join(
            args.results_dir,
            f'comparison_{args.dataset}_{treatment_kind}_vs_gaussian_{ts}.png',
        )

    params = read_experiment_output(t_txt) or read_experiment_output(g_txt)
    plot_comparison(args.dataset, treatment_kind, t_log, g_log, t_txt, g_txt, args.output, params)


if __name__ == '__main__':
    main()
