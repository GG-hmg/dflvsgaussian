"""
实验结果对比绘图脚本
从experiment_results目录读取CSV文件，绘制准确率和抗反演能力对比图
"""

import os
import re
import glob
import argparse
from datetime import datetime
from scipy.ndimage import gaussian_filter1d

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def parse_live_log(log_path):
    """从live log文件解析训练过程数据"""
    epochs = []
    accuracies = []
    anti_inversion_abilities = []

    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

        # 匹配轮次数据行
        pattern = r'轮次\s+(\d+)\s+客户端准确率:\s+\[(.*?)\].*?抗梯度反演能力:\s+([\d.]+)'
        matches = re.findall(pattern, content, re.DOTALL)

        for match in matches:
            epoch = int(match[0])
            # 解析准确率列表
            acc_str = match[1].replace('%', '').replace("'", "").replace(' ', '')
            accs = [float(x) for x in acc_str.split(',')]
            avg_acc = np.mean(accs)

            anti_inv = float(match[2])

            epochs.append(epoch)
            accuracies.append(avg_acc)
            anti_inversion_abilities.append(anti_inv)

    return epochs, accuracies, anti_inversion_abilities


def read_experiment_output(output_path):
    """从output文件读取最终结果"""
    results = {}
    with open(output_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

        # 解析最终准确率
        acc_match = re.search(r'最终准确率:\s*([\d.]+)%', content)
        if acc_match:
            results['final_accuracy'] = float(acc_match.group(1))

        # 解析最终抗反演能力
        anti_match = re.search(r'最终抗梯度反演能力:\s*([\d.]+)', content)
        if anti_match:
            results['anti_inversion'] = float(anti_match.group(1))

        # 解析训练时间
        time_match = re.search(r'总训练时间:\s*([\d.]+)秒', content)
        if time_match:
            results['training_time'] = float(time_match.group(1))

        # 解析参数
        sigma_match = re.search(r'sigma_factor[_-](?:gaussian|dfl)\s+([\d.]+)', content)
        if sigma_match:
            results['sigma'] = float(sigma_match.group(1))

        chaotic_match = re.search(r'chaotic_factor[:\s]+([\d.]+)', content)
        if chaotic_match:
            results['chaotic_factor'] = float(chaotic_match.group(1))

        alpha_match = re.search(r'dfl_alpha[:\s=]+([\d.]+)', content)
        if alpha_match:
            results['dfl_alpha'] = float(alpha_match.group(1))

        clip_match = re.search(r'裁剪边界[:\s]+([\d.]+)', content)
        if clip_match:
            results['clipping_bound'] = float(clip_match.group(1))

    return results


def plot_comparison(dataset, dfl_log, gaussian_log, dfl_output, gaussian_output, save_path, params):
    """绘制对比图"""

    # 解析数据
    dfl_epochs, dfl_accs, dfl_anti = parse_live_log(dfl_log) if os.path.exists(dfl_log) else ([], [], [])
    gauss_epochs, gauss_accs, gauss_anti = parse_live_log(gaussian_log) if os.path.exists(gaussian_log) else ([], [], [])

    # 读取最终结果
    dfl_results = read_experiment_output(dfl_output) if os.path.exists(dfl_output) else {}
    gauss_results = read_experiment_output(gaussian_output) if os.path.exists(gaussian_output) else {}

    # 创建图表
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 设置中文字体（如果可用）
    plt.rcParams['font.size'] = 10

    # 图1: 准确率对比
    ax1 = axes[0]
    if dfl_epochs:
        ax1.plot(dfl_epochs, dfl_accs, 'b-', label='DFL', linewidth=1, marker='o', markersize=2)
    if gauss_epochs:
        ax1.plot(gauss_epochs, gauss_accs, 'r-', label='Gaussian', linewidth=1, marker='s', markersize=2)

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title(f'{dataset} - Accuracy Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 100])

    # 图2: 抗反演能力对比
    ax2 = axes[1]
    if dfl_epochs:
        ax2.plot(dfl_epochs, dfl_anti, 'b-', label='DFL', linewidth=1, marker='o', markersize=2)
    if gauss_epochs:
        ax2.plot(gauss_epochs, gauss_anti, 'r-', label='Gaussian', linewidth=1, marker='s', markersize=2)

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Anti-Inversion Ability')
    ax2.set_title(f'{dataset} - Privacy Protection')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])

    # 添加参数信息到左上角
    param_text = f"Parameters:\n"
    param_text += f"sigma: {params.get('sigma', 'N/A')}\n"
    param_text += f"chaotic_factor: {params.get('chaotic_factor', 'N/A')}\n"
    param_text += f"dfl_alpha: {params.get('dfl_alpha', 'N/A')}\n"
    param_text += f"clipping_bound: {params.get('clipping_bound', 'N/A')}"

    fig.text(0.02, 0.98, param_text, ha='left', va='top', fontsize=8,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # 添加最终结果标注到右上角
    result_text = f"Final Results:\n"
    result_text += f"DFL Acc: {dfl_results.get('final_accuracy', 'N/A')}%\n"
    result_text += f"DFL Anti-Inversion: {dfl_results.get('anti_inversion', 'N/A')}\n"
    result_text += f"Gaussian Acc: {gauss_results.get('final_accuracy', 'N/A')}%\n"
    result_text += f"Gaussian Anti-Inversion: {gauss_results.get('anti_inversion', 'N/A')}"

    fig.text(0.98, 0.98, result_text, ha='right', va='top', fontsize=8,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
             transform=fig.transFigure)

    plt.tight_layout(rect=[0, 0, 1, 1])

    # 保存
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved to: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot experiment results comparison')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='Dataset name')
    parser.add_argument('--run_id', type=int, default=1, help='Run ID')
    parser.add_argument('--results_dir', type=str, default='experiment_results', help='Results directory')
    parser.add_argument('--output', type=str, default=None, help='Output figure path')
    args = parser.parse_args()

    # 生成带时间的文件名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.output is None:
        args.output = os.path.join(args.results_dir, f'comparison_{args.dataset}_{timestamp}.png')

    # 构建文件路径
    dfl_log = os.path.join(args.results_dir, f'{args.dataset}_dfl_run{args.run_id}_live.log')
    gaussian_log = os.path.join(args.results_dir, f'{args.dataset}_gaussian_run{args.run_id}_live.log')
    dfl_output = os.path.join(args.results_dir, f'{args.dataset}_dfl_run{args.run_id}_output.txt')
    gaussian_output = os.path.join(args.results_dir, f'{args.dataset}_gaussian_run{args.run_id}_output.txt')

    # 读取参数
    params = {}
    if os.path.exists(dfl_output):
        params = read_experiment_output(dfl_output)

    # 绘图
    plot_comparison(args.dataset, dfl_log, gaussian_log, dfl_output, gaussian_output, args.output, params)


if __name__ == '__main__':
    main()
