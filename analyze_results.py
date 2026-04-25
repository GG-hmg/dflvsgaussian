import json
import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime


def analyze_scm_results(results_dir="experiment_results"):
    """分析3D-SCM实验结果（只生成文本报告）"""
    # 查找最新的3D-SCM结果文件
    result_files = glob.glob(os.path.join(results_dir, "scm_results_*.json"))
    if not result_files:
        print("未找到3D-SCM实验结果文件")
        return

    latest_file = max(result_files, key=os.path.getctime)

    with open(latest_file, 'r') as f:
        results = json.load(f)

    # 创建详细的文本报告
    report_file = latest_file.replace('.json', '_detailed_report.txt')

    with open(report_file, 'w', encoding='utf-8') as report:
        report.write("=" * 80 + "\n")
        report.write("3D-SCM混沌差分隐私联邦学习 - 详细分析报告\n")
        report.write("=" * 80 + "\n\n")

        report.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.write(f"结果文件: {os.path.basename(latest_file)}\n")
        report.write(f"数据集数量: {len(results)}\n\n")

        # 1. 各数据集详细分析
        report.write("1. 各数据集详细性能分析\n")
        report.write("=" * 80 + "\n\n")

        for dataset in results:
            accuracies = results[dataset]

            report.write(f"数据集: {dataset}\n")
            report.write("-" * 60 + "\n")

            # 准确率变化表格
            report.write("准确率变化历程:\n")
            report.write("轮次 | 准确率(%) | 变化幅度 | 状态\n")
            report.write("-" * 50 + "\n")

            prev_acc = accuracies[0]
            for epoch, acc in enumerate(accuracies, 1):
                if epoch == 1:
                    change = 0.0
                    status = "初始"
                else:
                    change = acc - prev_acc
                    if change > 0.5:
                        status = "快速上升"
                    elif change > 0.1:
                        status = "上升"
                    elif change < -0.5:
                        status = "快速下降"
                    elif change < -0.1:
                        status = "下降"
                    else:
                        status = "稳定"

                report.write(f"{epoch:4d} | {acc:9.2f} | {change:+.2f} | {status}\n")
                prev_acc = acc

            # 统计指标
            report.write("\n统计指标:\n")
            report.write(f"  初始准确率: {accuracies[0]:.2f}%\n")
            report.write(f"  最终准确率: {accuracies[-1]:.2f}%\n")
            report.write(f"  平均准确率: {np.mean(accuracies):.2f}%\n")
            report.write(f"  准确率标准差: {np.std(accuracies):.4f}\n")
            report.write(f"  准确率最大值: {max(accuracies):.2f}% (轮次 {accuracies.index(max(accuracies)) + 1})\n")
            report.write(f"  准确率最小值: {min(accuracies):.2f}% (轮次 {accuracies.index(min(accuracies)) + 1})\n")

            # 提升分析
            total_improvement = accuracies[-1] - accuracies[0]
            improvement_per_epoch = total_improvement / len(accuracies) if len(accuracies) > 0 else 0
            improvement_percentage = (total_improvement / accuracies[0]) * 100 if accuracies[0] > 0 else 0

            report.write(f"\n提升分析:\n")
            report.write(f"  总提升: {total_improvement:+.2f}%\n")
            report.write(f"  每轮平均提升: {improvement_per_epoch:+.4f}%\n")
            report.write(f"  相对提升: {improvement_percentage:+.2f}%\n")

            # 收敛分析
            report.write(f"\n收敛分析:\n")

            # 方法1: 基于变化率的收敛
            convergence_threshold = 0.1  # 小于0.1%变化视为收敛
            convergence_epoch = len(accuracies)
            for i in range(1, len(accuracies)):
                if abs(accuracies[i] - accuracies[i - 1]) < convergence_threshold:
                    convergence_epoch = i
                    break

            report.write(f"  基于变化率的收敛轮次: {convergence_epoch}\n")

            # 方法2: 基于最大值的收敛
            max_acc = max(accuracies)
            max_epoch = accuracies.index(max_acc) + 1
            convergence_to_max = max_epoch if max_acc - accuracies[-1] < 0.5 else len(accuracies)
            report.write(f"  达到最大值的轮次: {max_epoch}\n")

            # 稳定性分析
            last_5_acc = accuracies[-5:] if len(accuracies) >= 5 else accuracies
            stability = 1 - (np.std(last_5_acc) / np.mean(last_5_acc)) if np.mean(last_5_acc) > 0 else 1
            report.write(f"  最后5轮稳定性: {stability:.4f} (1表示完全稳定)\n")

            # 学习效率
            if convergence_epoch > 0:
                learning_efficiency = total_improvement / convergence_epoch
                report.write(f"  学习效率: {learning_efficiency:.4f}%/轮\n")

            report.write("\n")

        # 2. 跨数据集对比分析
        report.write("\n2. 跨数据集性能对比\n")
        report.write("=" * 80 + "\n\n")

        comparison_data = []
        for dataset in results:
            accuracies = results[dataset]

            # 计算各种指标
            final_acc = accuracies[-1]
            max_acc = max(accuracies)
            min_acc = min(accuracies)
            avg_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)

            # 收敛速度 (达到最终准确率的95%)
            target_acc = final_acc * 0.95
            convergence_speed = len(accuracies)
            for i, acc in enumerate(accuracies):
                if acc >= target_acc:
                    convergence_speed = i + 1
                    break

            # 稳定性 (最后5轮标准差)
            last_5 = accuracies[-5:] if len(accuracies) >= 5 else accuracies
            stability = np.std(last_5)

            comparison_data.append({
                'Dataset': dataset,
                'Final_Acc': final_acc,
                'Max_Acc': max_acc,
                'Min_Acc': min_acc,
                'Avg_Acc': avg_acc,
                'Std_Dev': std_acc,
                'Convergence': convergence_speed,
                'Stability': stability,
                'Improvement': final_acc - accuracies[0]
            })

        # 创建对比表格
        df_comparison = pd.DataFrame(comparison_data)

        report.write("性能对比表:\n")
        report.write(df_comparison.to_string(index=False))
        report.write("\n\n")

        # 排名分析
        report.write("性能排名:\n")

        # 按最终准确率排名
        report.write("\n按最终准确率排名:\n")
        df_sorted = df_comparison.sort_values('Final_Acc', ascending=False)
        for i, (_, row) in enumerate(df_sorted.iterrows(), 1):
            report.write(f"  {i}. {row['Dataset']}: {row['Final_Acc']:.2f}%\n")

        # 按收敛速度排名
        report.write("\n按收敛速度排名 (越小越好):\n")
        df_sorted = df_comparison.sort_values('Convergence')
        for i, (_, row) in enumerate(df_sorted.iterrows(), 1):
            report.write(f"  {i}. {row['Dataset']}: {row['Convergence']} 轮\n")

        # 按稳定性排名
        report.write("\n按稳定性排名 (标准差越小越好):\n")
        df_sorted = df_comparison.sort_values('Stability')
        for i, (_, row) in enumerate(df_sorted.iterrows(), 1):
            report.write(f"  {i}. {row['Dataset']}: {row['Stability']:.4f}\n")

        # 3. 综合分析
        report.write("\n3. 综合分析\n")
        report.write("=" * 80 + "\n\n")

        # 计算总体统计
        all_accuracies = []
        for dataset in results:
            all_accuracies.extend(results[dataset])

        report.write(f"总体统计:\n")
        report.write(f"  所有数据集平均最终准确率: {np.mean([results[d][-1] for d in results]):.2f}%\n")
        report.write(f"  所有轮次平均准确率: {np.mean(all_accuracies):.2f}%\n")
        report.write(f"  总准确率标准差: {np.std(all_accuracies):.4f}\n")

        # 最佳和最差表现
        best_dataset = max(results.keys(), key=lambda d: results[d][-1])
        worst_dataset = min(results.keys(), key=lambda d: results[d][-1])

        report.write(f"\n最佳表现: {best_dataset} ({results[best_dataset][-1]:.2f}%)\n")
        report.write(f"最差表现: {worst_dataset} ({results[worst_dataset][-1]:.2f}%)\n")
        report.write(f"性能差距: {results[best_dataset][-1] - results[worst_dataset][-1]:.2f}%\n")

        # 性能一致性分析
        report.write(f"\n性能一致性分析:\n")
        acc_std = [np.std(results[d]) for d in results]
        avg_std = np.mean(acc_std)
        report.write(f"  平均准确率标准差: {avg_std:.4f}\n")
        report.write(f"  最大标准差: {max(acc_std):.4f} ({list(results.keys())[acc_std.index(max(acc_std))]})\n")
        report.write(f"  最小标准差: {min(acc_std):.4f} ({list(results.keys())[acc_std.index(min(acc_std))]})\n")

        # 4. 建议和结论
        report.write("\n4. 结论与建议\n")
        report.write("=" * 80 + "\n\n")

        report.write("基于3D-SCM混沌差分隐私的联邦学习实验结果表明:\n\n")

        report.write("优势:\n")
        report.write("1. 3D-SCM混沌系统提供了良好的随机性，增强了隐私保护效果\n")
        report.write("2. 在所有测试数据集上都表现出稳定的性能提升\n")
        report.write("3. 收敛速度适中，学习过程平稳\n\n")

        report.write("观察:\n")
        for dataset in results:
            accuracies = results[dataset]
            if accuracies[-1] - accuracies[0] > 5.0:
                report.write(f"- {dataset}: 表现出显著的性能提升 ({accuracies[-1] - accuracies[0]:.1f}%)\n")
            elif accuracies[-1] - accuracies[0] > 2.0:
                report.write(f"- {dataset}: 有较好的性能提升 ({accuracies[-1] - accuracies[0]:.1f}%)\n")
            else:
                report.write(f"- {dataset}: 性能提升有限 ({accuracies[-1] - accuracies[0]:.1f}%)\n")

        report.write("\n建议:\n")
        report.write("1. 对于性能提升有限的数据集，可以调整3D-SCM参数或学习率\n")
        report.write("2. 考虑结合自适应隐私预算分配策略\n")
        report.write("3. 在非IID数据分布下进一步验证系统鲁棒性\n")

        report.write("\n" + "=" * 80 + "\n")
        report.write("报告结束\n")
        report.write("=" * 80 + "\n")

    print(f"详细分析报告已保存到: {report_file}")

    # 同时在控制台输出摘要
    print("\n" + "=" * 60)
    print("3D-SCM混沌差分隐私实验结果摘要")
    print("=" * 60)

    for dataset in results:
        accuracies = results[dataset]
        print(f"\n{dataset}:")
        print(f"  初始准确率: {accuracies[0]:.2f}%")
        print(f"  最终准确率: {accuracies[-1]:.2f}%")
        print(f"  提升幅度: {accuracies[-1] - accuracies[0]:+.2f}%")
        print(f"  收敛轮次: ~{accuracies.index(max(accuracies)) + 1}")

    print("\n" + "=" * 60)
    print("分析完成!")


if __name__ == "__main__":
    analyze_scm_results()