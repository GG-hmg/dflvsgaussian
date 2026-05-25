import glob
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd


def _display_method_name(method: str) -> str:
    if method == "gaussian":
        return "Pure Gaussian"
    if method == "delayed_feedback":
        return "Delayed Feedback"
    if method == "none":
        return "No Noise"
    return method


def _find_latest_results_file(results_dir: str) -> str | None:
    preferred = glob.glob(os.path.join(results_dir, "comparison_results_*.json"))
    if preferred:
        return max(preferred, key=os.path.getctime)

    fallback = glob.glob(os.path.join(results_dir, "delayed_feedback_results_*.json"))
    if fallback:
        return max(fallback, key=os.path.getctime)

    return None


def analyze_results(results_dir="experiment_results"):
    """Generate a clean text report for pure Gaussian vs delayed feedback experiments."""
    latest_file = _find_latest_results_file(results_dir)
    if not latest_file:
        print("No comparison results file found.")
        return

    with open(latest_file, "r", encoding="utf-8") as f:
        results = json.load(f)

    report_file = latest_file.replace(".json", "_report.txt")

    with open(report_file, "w", encoding="utf-8") as report:
        report.write("=" * 80 + "\n")
        report.write("Pure Gaussian vs Delayed Feedback - Detailed Report\n")
        report.write("=" * 80 + "\n\n")
        report.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.write(f"Source file : {os.path.basename(latest_file)}\n")
        report.write(f"Datasets    : {len(results)}\n\n")

        comparison_data = []
        for dataset, methods in results.items():
            report.write(f"Dataset: {dataset}\n")
            report.write("-" * 72 + "\n")
            report.write("Method | Epoch | Accuracy | Delta | Status\n")
            report.write("-" * 72 + "\n")

            for method, metrics in methods.items():
                accuracies = metrics.get("accuracies", [])
                if not accuracies:
                    continue

                display_method = _display_method_name(method)
                prev_acc = accuracies[0]
                for epoch, acc in enumerate(accuracies, 1):
                    if epoch == 1:
                        delta = 0.0
                        status = "start"
                    else:
                        delta = acc - prev_acc
                        if delta > 0.5:
                            status = "strong rise"
                        elif delta > 0.1:
                            status = "rise"
                        elif delta < -0.5:
                            status = "strong drop"
                        elif delta < -0.1:
                            status = "drop"
                        else:
                            status = "stable"
                    report.write(f"{display_method:18s} | {epoch:5d} | {acc:8.2f} | {delta:+6.2f} | {status}\n")
                    prev_acc = acc

                final_acc = accuracies[-1]
                max_acc = max(accuracies)
                min_acc = min(accuracies)
                avg_acc = float(np.mean(accuracies))
                std_acc = float(np.std(accuracies))
                improvement = final_acc - accuracies[0]
                last_5 = accuracies[-5:] if len(accuracies) >= 5 else accuracies
                stability = 1 - (np.std(last_5) / np.mean(last_5)) if np.mean(last_5) > 0 else 1.0

                target_acc = final_acc * 0.95
                convergence_epoch = len(accuracies)
                for i, acc in enumerate(accuracies):
                    if acc >= target_acc:
                        convergence_epoch = i + 1
                        break

                report.write("\n")
                report.write(f"  Final accuracy: {final_acc:.2f}%\n")
                report.write(f"  Best accuracy : {max_acc:.2f}%\n")
                report.write(f"  Mean accuracy : {avg_acc:.2f}%\n")
                report.write(f"  Std dev       : {std_acc:.4f}\n")
                report.write(f"  Improvement   : {improvement:+.2f}%\n")
                report.write(f"  Convergence   : {convergence_epoch}\n")
                report.write(f"  Stability     : {stability:.4f}\n\n")

                comparison_data.append(
                    {
                        "Dataset": dataset,
                        "Method": display_method,
                        "Final_Acc": final_acc,
                        "Max_Acc": max_acc,
                        "Min_Acc": min_acc,
                        "Avg_Acc": avg_acc,
                        "Std_Dev": std_acc,
                        "Convergence": convergence_epoch,
                        "Stability": stability,
                        "Improvement": improvement,
                    }
                )

        report.write("=" * 80 + "\n")
        report.write("Summary Table\n")
        report.write("=" * 80 + "\n\n")

        df = pd.DataFrame(comparison_data)
        if not df.empty:
            report.write(df.to_string(index=False))
            report.write("\n\n")

            report.write("Ranking by final accuracy:\n")
            for i, (_, row) in enumerate(df.sort_values("Final_Acc", ascending=False).iterrows(), 1):
                report.write(f"  {i}. {row['Dataset']} / {row['Method']}: {row['Final_Acc']:.2f}%\n")

            report.write("\nRanking by convergence speed:\n")
            for i, (_, row) in enumerate(df.sort_values("Convergence").iterrows(), 1):
                report.write(f"  {i}. {row['Dataset']} / {row['Method']}: {row['Convergence']} epochs\n")

            report.write("\nRanking by stability:\n")
            for i, (_, row) in enumerate(df.sort_values("Stability").iterrows(), 1):
                report.write(f"  {i}. {row['Dataset']} / {row['Method']}: {row['Stability']:.4f}\n")

        report.write("\n" + "=" * 80 + "\n")
        report.write("Interpretation\n")
        report.write("=" * 80 + "\n\n")
        report.write("1. Pure Gaussian is the control baseline.\n")
        report.write("2. Delayed Feedback isolates the effect of the delayed-feedback generator without Gaussian mixing.\n")
        report.write("3. Compare final accuracy, convergence, and stability together to judge the tradeoff.\n")

    print(f"Saved report: {report_file}")

    print("\n" + "=" * 72)
    print("Pure Gaussian vs Delayed Feedback Summary")
    print("=" * 72)
    if not df.empty:
        print(df.to_string(index=False))


def analyze_delayed_feedback_results(results_dir="experiment_results"):
    analyze_results(results_dir)


if __name__ == "__main__":
    analyze_results()
