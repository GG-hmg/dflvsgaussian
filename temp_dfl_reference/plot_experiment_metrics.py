import argparse
import csv
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHOD_ORDER = ["none", "gaussian", "delayed_feedback"]


@dataclass
class MethodMetrics:
    dataset: str
    method: str
    display_name: str
    accuracies: List[float] = field(default_factory=list)
    ability: Optional[float] = None
    risk: Optional[float] = None
    ability_series: List[float] = field(default_factory=list)
    final_accuracy: Optional[float] = None
    paper_metrics: Dict[str, float] = field(default_factory=dict)
    client_metrics: Dict[str, float] = field(default_factory=dict)
    attack_metrics: Dict[str, float] = field(default_factory=dict)
    convergence_epoch: Optional[float] = None
    stability: Optional[float] = None
    training_time: Optional[float] = None
    avg_epoch_time: Optional[float] = None
    avg_psnr: Optional[float] = None
    avg_ssim: Optional[float] = None
    avg_lpips: Optional[float] = None
    attack_success_rate: Optional[float] = None
    total_comm_mb: Optional[float] = None
    worst_client_accuracy: Optional[float] = None
    jain_fairness: Optional[float] = None
    source_files: List[str] = field(default_factory=list)
    mtime: float = 0.0

    def derive(self) -> None:
        if self.accuracies:
            self.final_accuracy = self.accuracies[-1]
        if self.final_accuracy is None and self.accuracies:
            self.final_accuracy = self.accuracies[-1]
        if self.ability is None and self.ability_series:
            self.ability = self.ability_series[-1]
        if self.risk is None and self.ability is not None:
            self.risk = max(0.0, min(1.0, 1.0 - self.ability))
        if self.paper_metrics:
            self.final_accuracy = float(self.paper_metrics.get("final_accuracy", self.final_accuracy or 0.0))
            self.convergence_epoch = float(self.paper_metrics.get("convergence_epoch", self.convergence_epoch or 0.0))
            self.stability = float(self.paper_metrics.get("stability", self.stability or 0.0))
            self.training_time = float(self.paper_metrics.get("training_time", self.training_time or 0.0))
            self.avg_epoch_time = float(self.paper_metrics.get("avg_epoch_time", self.avg_epoch_time or 0.0))
            self.total_comm_mb = float(self.paper_metrics.get("total_comm_mb", self.total_comm_mb or 0.0))
        if self.attack_metrics:
            if self.ability is None and "anti_inversion_ability" in self.attack_metrics:
                self.ability = float(self.attack_metrics["anti_inversion_ability"])
            if self.risk is None and "leakage_risk" in self.attack_metrics:
                self.risk = float(self.attack_metrics["leakage_risk"])
            self.avg_psnr = float(self.attack_metrics.get("avg_psnr", self.avg_psnr or 0.0))
            self.avg_ssim = float(self.attack_metrics.get("avg_ssim", self.avg_ssim or 0.0))
            self.avg_lpips = float(self.attack_metrics.get("avg_lpips", self.avg_lpips or 0.0))
            self.attack_success_rate = float(self.attack_metrics.get("attack_success_rate", self.attack_success_rate or 0.0))
        if self.client_metrics:
            self.worst_client_accuracy = float(self.client_metrics.get("worst_client_accuracy", self.worst_client_accuracy or 0.0))
            self.jain_fairness = float(self.client_metrics.get("jain_fairness", self.jain_fairness or 0.0))


def display_method_name(method: str) -> str:
    if method == "none":
        return "No Noise"
    if method == "gaussian":
        return "Pure Gaussian"
    if method == "delayed_feedback":
        return "Delayed Feedback"
    return method


def normalize_method_name(method: str) -> str:
    text = method.strip()
    lowered = text.lower()
    if lowered in ("none", "no noise"):
        return "none"
    if lowered in ("gaussian", "pure gaussian"):
        return "gaussian"
    if "gaussian" in lowered:
        return "gaussian"
    if "no noise" in lowered:
        return "none"
    if "delayed" in lowered or "feedback" in lowered:
        return "delayed_feedback"
    return lowered.replace(" ", "_")


def percent_to_float(value: str) -> Optional[float]:
    cleaned = value.strip().replace("%", "").replace("+", "").replace("s", "")
    cleaned = re.sub(r"[^0-9.\-]", "", cleaned)
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def method_sort_key(method: str) -> Tuple[int, str]:
    if method in METHOD_ORDER:
        return METHOD_ORDER.index(method), method
    return len(METHOD_ORDER), method


def dataset_sort_key(dataset: str) -> Tuple[int, str]:
    preferred = ["MNIST", "FashionMNIST", "CIFAR10", "SVHN"]
    if dataset in preferred:
        return preferred.index(dataset), dataset
    return len(preferred), dataset


def read_text(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "gbk"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def result_json_files(results_dir: Path, latest_only: bool) -> List[Path]:
    files = sorted(results_dir.glob("comparison_results_*.json"))
    if not files:
        files = sorted(results_dir.glob("delayed_feedback_results_*.json"))
    if latest_only and files:
        return [max(files, key=lambda p: p.stat().st_mtime)]
    return sorted(files, key=lambda p: p.stat().st_mtime)


def put_record(records: Dict[str, Dict[str, MethodMetrics]], metric: MethodMetrics) -> None:
    metric.derive()
    records.setdefault(metric.dataset, {})[metric.method] = metric


def load_json_results(results_dir: Path, latest_only: bool) -> Dict[str, Dict[str, MethodMetrics]]:
    records: Dict[str, Dict[str, MethodMetrics]] = {}
    for path in result_json_files(results_dir, latest_only):
        try:
            payload = json.loads(read_text(path))
        except json.JSONDecodeError as exc:
            print("Skip invalid JSON %s: %s" % (path, exc))
            continue

        mtime = path.stat().st_mtime
        for dataset, methods in payload.items():
            if not isinstance(methods, dict):
                continue
            for raw_method, values in methods.items():
                if not isinstance(values, dict):
                    continue
                method = normalize_method_name(raw_method)
                accs = [float(x) for x in values.get("accuracies", []) if x is not None]
                ability = values.get("anti_inversion_ability")
                if ability is None:
                    ability = values.get("gradient_inversion_ability")
                risk = values.get("risk")
                if risk is None:
                    risk = values.get("gradient_inversion_risk")

                metric = MethodMetrics(
                    dataset=dataset,
                    method=method,
                    display_name=display_method_name(method),
                    accuracies=accs,
                    ability=float(ability) if ability is not None else None,
                    risk=float(risk) if risk is not None else None,
                    final_accuracy=float(values.get("final_accuracy")) if values.get("final_accuracy") is not None else None,
                    paper_metrics=values.get("paper_metrics", {}) or {},
                    client_metrics=values.get("client_metrics", {}) or {},
                    attack_metrics=values.get("attack_metrics", {}) or {},
                    source_files=[path.name],
                    mtime=mtime,
                )
                put_record(records, metric)
    return records


def load_csv_results(results_dir: Path, latest_only: bool) -> Dict[str, Dict[str, MethodMetrics]]:
    files = sorted(results_dir.glob("comparison_summary_*.csv"), key=lambda p: p.stat().st_mtime)
    if latest_only and files:
        files = [max(files, key=lambda p: p.stat().st_mtime)]

    records: Dict[str, Dict[str, MethodMetrics]] = {}
    for path in files:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                dataset = row.get("Dataset", "").strip()
                if not dataset:
                    continue
                method = normalize_method_name(row.get("Method", ""))
                final_accuracy = percent_to_float(row.get("Final_Accuracy", ""))
                ability = percent_to_float(row.get("Anti_Inversion_Ability", ""))
                risk = percent_to_float(row.get("Leakage_Risk", ""))
                metric = MethodMetrics(
                    dataset=dataset,
                    method=method,
                    display_name=display_method_name(method),
                    final_accuracy=final_accuracy,
                    ability=ability,
                    risk=risk,
                    convergence_epoch=percent_to_float(row.get("Convergence_Epoch", "")),
                    stability=percent_to_float(row.get("Stability", "")),
                    training_time=percent_to_float(row.get("Training_Time", "")),
                    avg_epoch_time=percent_to_float(row.get("Avg_Epoch_Time", "")),
                    avg_psnr=percent_to_float(row.get("Avg_PSNR", "")),
                    avg_ssim=percent_to_float(row.get("Avg_SSIM", "")),
                    avg_lpips=percent_to_float(row.get("Avg_LPIPS", "")),
                    attack_success_rate=percent_to_float(row.get("Attack_Success_Rate", "")),
                    total_comm_mb=percent_to_float(row.get("Total_Communication_MB", "")),
                    worst_client_accuracy=percent_to_float(row.get("Worst_Client_Accuracy", "")),
                    jain_fairness=percent_to_float(row.get("Jain_Fairness", "")),
                    source_files=[path.name],
                    mtime=path.stat().st_mtime,
                )
                put_record(records, metric)
    return records


def parse_numeric_list(candidate: str) -> List[float]:
    values: List[float] = []
    for item in candidate.split(","):
        cleaned = re.sub(r"[^0-9.\-]", "", item.strip())
        if not cleaned:
            continue
        try:
            values.append(float(cleaned))
        except ValueError:
            continue
    return values


def parse_output_lists(text: str) -> Tuple[List[float], List[float]]:
    bracket_lists = re.findall(r"\[([^\]]+)\]", text, flags=re.S)

    accuracy_candidates: List[List[float]] = []
    for candidate in bracket_lists:
        if "%" not in candidate:
            continue
        values = parse_numeric_list(candidate)
        if len(values) >= 3 and all(0.0 <= value <= 100.0 for value in values):
            accuracy_candidates.append(values)

    accuracies = max(accuracy_candidates, key=len) if accuracy_candidates else []

    ability_candidates: List[List[float]] = []
    for candidate in bracket_lists:
        if "%" in candidate:
            continue
        values = parse_numeric_list(candidate)
        if len(values) < 3:
            continue
        if all(0.0 <= value <= 1.0 for value in values):
            if not accuracies or len(values) == len(accuracies):
                ability_candidates.append(values)

    ability_series = ability_candidates[-1] if ability_candidates else []
    return accuracies, ability_series


def merge_output_logs(records: Dict[str, Dict[str, MethodMetrics]], results_dir: Path) -> None:
    grouped: Dict[Tuple[str, str], Dict[str, List[List[float]]]] = {}
    source_names: Dict[Tuple[str, str], List[str]] = {}

    for path in sorted(results_dir.glob("*_run*_output.txt"), key=lambda p: p.stat().st_mtime):
        match = re.match(r"(.+?)_(.+)_run\d+_output$", path.stem)
        if not match:
            continue
        dataset, raw_method = match.group(1), match.group(2)
        method = normalize_method_name(raw_method)
        accuracies, ability_series = parse_output_lists(read_text(path))
        key = (dataset, method)
        grouped.setdefault(key, {"accuracies": [], "ability_series": []})
        source_names.setdefault(key, [])
        if accuracies:
            grouped[key]["accuracies"].append(accuracies)
        if ability_series:
            grouped[key]["ability_series"].append(ability_series)
        source_names[key].append(path.name)

    for (dataset, method), values in grouped.items():
        rec = records.setdefault(
            dataset,
            {},
        ).setdefault(
            method,
            MethodMetrics(dataset=dataset, method=method, display_name=display_method_name(method)),
        )

        if not rec.accuracies and values["accuracies"]:
            rec.accuracies = average_aligned(values["accuracies"])
        if values["ability_series"]:
            rec.ability_series = average_aligned(values["ability_series"])
            rec.ability = rec.ability_series[-1]
            rec.risk = max(0.0, min(1.0, 1.0 - rec.ability))
        for source in source_names.get((dataset, method), []):
            if source not in rec.source_files:
                rec.source_files.append(source)
        rec.derive()


def average_aligned(series_list: Sequence[Sequence[float]]) -> List[float]:
    if not series_list:
        return []
    min_len = min(len(series) for series in series_list)
    if min_len <= 0:
        return []
    averaged: List[float] = []
    for idx in range(min_len):
        averaged.append(sum(series[idx] for series in series_list) / len(series_list))
    return averaged


def load_results(results_dir: Path, latest_only: bool) -> Dict[str, Dict[str, MethodMetrics]]:
    records = load_json_results(results_dir, latest_only)
    if not records:
        records = load_csv_results(results_dir, latest_only)
    merge_output_logs(records, results_dir)
    return records


def iter_datasets(records: Dict[str, Dict[str, MethodMetrics]]) -> List[str]:
    return sorted(records.keys(), key=dataset_sort_key)


def iter_methods(records: Dict[str, Dict[str, MethodMetrics]]) -> List[str]:
    methods = set()
    for method_map in records.values():
        methods.update(method_map.keys())
    return sorted(methods, key=method_sort_key)


def subplot_grid(count: int) -> Tuple[int, int]:
    cols = min(2, max(1, count))
    rows = int(math.ceil(count / cols))
    return rows, cols


def finish_empty_axes(axes: Iterable[plt.Axes], used_count: int) -> None:
    for idx, ax in enumerate(axes):
        if idx >= used_count:
            ax.axis("off")


def plot_accuracy_curves(records: Dict[str, Dict[str, MethodMetrics]], output_dir: Path, dpi: int) -> Optional[Path]:
    datasets = iter_datasets(records)
    if not datasets:
        return None

    rows, cols = subplot_grid(len(datasets))
    fig, axes = plt.subplots(rows, cols, figsize=(6.8 * cols, 4.4 * rows), squeeze=False)
    flat_axes = list(axes.ravel())

    for ax, dataset in zip(flat_axes, datasets):
        method_map = records[dataset]
        has_epoch_series = any(metric.accuracies for metric in method_map.values())

        if has_epoch_series:
            for method in sorted(method_map.keys(), key=method_sort_key):
                metric = method_map[method]
                if not metric.accuracies:
                    continue
                epochs = list(range(1, len(metric.accuracies) + 1))
                ax.plot(epochs, metric.accuracies, marker="o", linewidth=2, markersize=4, label=metric.display_name)
            ax.set_xlabel("Global Round")
            ax.set_ylabel("Accuracy (%)")
        else:
            method_names = []
            final_values = []
            for method in sorted(method_map.keys(), key=method_sort_key):
                metric = method_map[method]
                if metric.final_accuracy is None:
                    continue
                method_names.append(metric.display_name)
                final_values.append(metric.final_accuracy)
            ax.plot(method_names, final_values, marker="o", linewidth=2)
            ax.tick_params(axis="x", rotation=20)
            ax.set_xlabel("Method")
            ax.set_ylabel("Final Accuracy (%)")

        ax.set_title("%s Accuracy" % dataset)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    finish_empty_axes(flat_axes, len(datasets))
    fig.suptitle("Accuracy Curves by Dataset", fontsize=14)
    fig.tight_layout()
    output_path = output_dir / "accuracy_curves_by_dataset.png"
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_ability_curves(records: Dict[str, Dict[str, MethodMetrics]], output_dir: Path, dpi: int) -> Optional[Path]:
    datasets = iter_datasets(records)
    if not datasets:
        return None

    rows, cols = subplot_grid(len(datasets))
    fig, axes = plt.subplots(rows, cols, figsize=(6.8 * cols, 4.4 * rows), squeeze=False)
    flat_axes = list(axes.ravel())

    for ax, dataset in zip(flat_axes, datasets):
        method_map = records[dataset]
        has_epoch_series = any(metric.ability_series for metric in method_map.values())

        if has_epoch_series:
            max_len = max((len(metric.ability_series) for metric in method_map.values()), default=0)
            for method in sorted(method_map.keys(), key=method_sort_key):
                metric = method_map[method]
                if metric.ability_series:
                    epochs = list(range(1, len(metric.ability_series) + 1))
                    ax.plot(epochs, metric.ability_series, marker="o", linewidth=2, markersize=4, label=metric.display_name)
                elif metric.ability is not None and max_len:
                    ax.plot(
                        list(range(1, max_len + 1)),
                        [metric.ability] * max_len,
                        linestyle="--",
                        linewidth=1.5,
                        label="%s (final)" % metric.display_name,
                    )
            ax.set_xlabel("Global Round")
            ax.set_ylabel("Anti-attack Ability")
        else:
            method_names = []
            ability_values = []
            for method in sorted(method_map.keys(), key=method_sort_key):
                metric = method_map[method]
                if metric.ability is None:
                    continue
                method_names.append(metric.display_name)
                ability_values.append(metric.ability)
            ax.plot(method_names, ability_values, marker="o", linewidth=2)
            ax.tick_params(axis="x", rotation=20)
            ax.set_xlabel("Method")
            ax.set_ylabel("Final Anti-attack Ability")

        ax.set_ylim(0.0, 1.0)
        ax.set_title("%s Anti-attack Ability" % dataset)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    finish_empty_axes(flat_axes, len(datasets))
    fig.suptitle("Anti-attack Ability Curves by Dataset", fontsize=14)
    fig.tight_layout()
    output_path = output_dir / "anti_attack_ability_curves_by_dataset.png"
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_paper_summary(records: Dict[str, Dict[str, MethodMetrics]], output_dir: Path) -> Path:
    output_path = output_dir / "paper_metrics_summary.csv"
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "Dataset",
                "Method",
                "Final_Accuracy",
                "Convergence_Epoch",
                "Stability",
                "Anti_Inversion_Ability",
                "Leakage_Risk",
                "Avg_PSNR",
                "Avg_SSIM",
                "Avg_LPIPS",
                "Attack_Success_Rate",
                "Training_Time",
                "Avg_Epoch_Time",
                "Total_Communication_MB",
                "Worst_Client_Accuracy",
                "Jain_Fairness",
            ]
        )
        for dataset in iter_datasets(records):
            for method in sorted(records[dataset].keys(), key=method_sort_key):
                metric = records[dataset][method]
                writer.writerow(
                    [
                        dataset,
                        metric.display_name,
                        "" if metric.final_accuracy is None else f"{metric.final_accuracy:.4f}",
                        "" if metric.convergence_epoch is None else f"{metric.convergence_epoch:.0f}",
                        "" if metric.stability is None else f"{metric.stability:.6f}",
                        "" if metric.ability is None else f"{metric.ability:.6f}",
                        "" if metric.risk is None else f"{metric.risk:.6f}",
                        "" if metric.avg_psnr is None else f"{metric.avg_psnr:.6f}",
                        "" if metric.avg_ssim is None else f"{metric.avg_ssim:.6f}",
                        "" if metric.avg_lpips is None else f"{metric.avg_lpips:.6f}",
                        "" if metric.attack_success_rate is None else f"{metric.attack_success_rate:.6f}",
                        "" if metric.training_time is None else f"{metric.training_time:.4f}",
                        "" if metric.avg_epoch_time is None else f"{metric.avg_epoch_time:.4f}",
                        "" if metric.total_comm_mb is None else f"{metric.total_comm_mb:.6f}",
                        "" if metric.worst_client_accuracy is None else f"{metric.worst_client_accuracy:.4f}",
                        "" if metric.jain_fairness is None else f"{metric.jain_fairness:.6f}",
                    ]
                )
    return output_path


def print_summary(records: Dict[str, Dict[str, MethodMetrics]], output_paths: Sequence[Path]) -> None:
    print("Loaded datasets: %s" % ", ".join(iter_datasets(records)))
    print("Loaded methods : %s" % ", ".join(display_method_name(method) for method in iter_methods(records)))
    print("")
    for dataset in iter_datasets(records):
        print("[%s]" % dataset)
        for method in sorted(records[dataset].keys(), key=method_sort_key):
            metric = records[dataset][method]
            acc_text = "-" if metric.final_accuracy is None else "%.2f%%" % metric.final_accuracy
            ability_text = "-" if metric.ability is None else "%.4f" % metric.ability
            stability_text = "-" if metric.stability is None else "%.4f" % metric.stability
            psnr_text = "-" if metric.avg_psnr is None else "%.3f" % metric.avg_psnr
            print(
                "  %-24s final_acc=%s  anti_attack=%s  stability=%s  avg_psnr=%s"
                % (metric.display_name, acc_text, ability_text, stability_text, psnr_text)
            )
    print("")
    for output_path in output_paths:
        print("Saved: %s" % output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot accuracy and anti-attack ability curves from experiment_results."
    )
    parser.add_argument("--results_dir", default="experiment_results", help="Directory containing experiment result files.")
    parser.add_argument("--output_dir", default=None, help="Directory for generated figures. Defaults to ./plots.")
    parser.add_argument("--latest_only", action="store_true", help="Use only the newest comparison result file.")
    parser.add_argument("--dpi", type=int, default=200, help="Output image DPI.")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise FileNotFoundError("Results directory does not exist: %s" % results_dir)

    output_dir = Path(args.output_dir) if args.output_dir else Path("plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"

    records = load_results(results_dir, args.latest_only)
    if not records:
        raise RuntimeError("No usable experiment results found in %s" % results_dir)

    output_paths: List[Path] = []
    for output_path in (
        plot_accuracy_curves(records, output_dir, args.dpi),
        plot_ability_curves(records, output_dir, args.dpi),
    ):
        if output_path is not None:
            output_paths.append(output_path)
    output_paths.append(save_paper_summary(records, output_dir))

    print_summary(records, output_paths)


if __name__ == "__main__":
    main()


