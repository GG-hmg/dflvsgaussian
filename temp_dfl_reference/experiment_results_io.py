import json
import os
import re
from datetime import datetime

import numpy as np


def parse_kv_line(output: str, prefix: str):
    pattern = re.escape(prefix) + r"\s*(.+)"
    match = re.search(pattern, output)
    if not match:
        return {}
    payload = match.group(1).strip()
    result = {}
    for part in payload.split(","):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        key = key.strip()
        value = value.strip()
        try:
            result[key] = float(value)
        except ValueError:
            result[key] = value
    return result


def aggregate_metric_dicts(metric_dicts):
    if not metric_dicts:
        return {}
    aggregated = {}
    keys = set()
    for item in metric_dicts:
        keys.update(item.keys())
    for key in sorted(keys):
        values = [item.get(key) for item in metric_dicts if key in item]
        numeric_values = [float(v) for v in values if isinstance(v, (int, float))]
        if numeric_values:
            aggregated[key] = float(np.mean(numeric_values))
        elif values:
            aggregated[key] = values[0]
    return aggregated


def calculate_convergence_epoch(accuracies):
    if not accuracies:
        return 0
    final_acc = accuracies[-1]
    threshold = final_acc * 0.95
    for i, acc in enumerate(accuracies):
        if acc >= threshold:
            return i + 1
    return len(accuracies)


def parse_experiment_output(output: str, elapsed_time: float):
    metrics = {
        "accuracies": [],
        "training_time": elapsed_time,
        "final_accuracy": 0,
        "convergence_epoch": 0,
        "peak_accuracy": 0,
        "accuracy_std": 0,
        "improvement": 0,
        "gradient_inversion_ability": 0,
        "gradient_inversion_risk": 0,
        "paper_metrics": {},
        "client_metrics": {},
        "attack_metrics": {},
    }

    if not output:
        return metrics

    match = re.search(r"平均准确率历程:\s*\[([^\]]+)\]", output)
    if match:
        vals = []
        for item in match.group(1).split(","):
            cleaned = re.sub(r"[^0-9\.\-]", "", item.strip())
            if cleaned:
                try:
                    vals.append(float(cleaned))
                except ValueError:
                    pass
        if vals:
            metrics["accuracies"] = vals

    acc_pattern = r"平均准确率历程:\s*\[([^\]]+)\]"
    match = re.search(acc_pattern, output)
    if match:
        items = match.group(1).split(",")
        vals = []
        for item in items:
            cleaned = re.sub(r"[^0-9\.\-]", "", item.strip())
            if cleaned:
                try:
                    vals.append(float(cleaned))
                except ValueError:
                    pass
        if vals:
            metrics["accuracies"] = vals

    if not metrics["accuracies"]:
        list_candidates = re.findall(r"\[([0-9\.,\s%\-]+)\]", output)
        for candidate in reversed(list_candidates):
            if "%" in candidate:
                vals = []
                for item in candidate.split(","):
                    cleaned = re.sub(r"[^0-9\.\-]", "", item.strip())
                    if cleaned:
                        try:
                            vals.append(float(cleaned))
                        except ValueError:
                            pass
                if vals:
                    metrics["accuracies"] = vals
                    break

    if not metrics["accuracies"]:
        list_candidates = re.findall(r"\[([0-9\.,\s\-]+)\]", output)
        for candidate in reversed(list_candidates):
            vals = []
            for item in candidate.split(","):
                cleaned = re.sub(r"[^0-9\.\-]", "", item.strip())
                if cleaned:
                    try:
                        num = float(cleaned)
                        if num > 1000:
                            continue
                        vals.append(num)
                    except ValueError:
                        pass
            if vals:
                metrics["accuracies"] = vals
                break

    ability_patterns = [
        r"平均抗梯度反演能力:\s*(\d+\.\d+)",
        r"最终抗梯度反演能力:\s*(\d+\.\d+)",
        r"抗梯度反演能力[^\d]*(\d+\.\d+)",
    ]
    for pattern in ability_patterns:
        matches = re.findall(pattern, output)
        if matches:
            try:
                metrics["gradient_inversion_ability"] = float(matches[-1])
                metrics["gradient_inversion_risk"] = max(0.0, min(1.0, 1.0 - metrics["gradient_inversion_ability"]))
                break
            except ValueError:
                pass

    if metrics["gradient_inversion_ability"] == 0:
        risk_patterns = [
            r"平均梯度反演风险:\s*(\d+\.\d+)",
            r"最终梯度反演风险:\s*(\d+\.\d+)",
            r"risk[^\d]*(\d+\.\d+)",
            r"Risk[^\d]*(\d+\.\d+)",
            r"风险[^\d]*(\d+\.\d+)"
        ]
        for pattern in risk_patterns:
            matches = re.findall(pattern, output)
            if matches:
                try:
                    risk_value = float(matches[-1])
                    metrics["gradient_inversion_risk"] = risk_value
                    metrics["gradient_inversion_ability"] = max(0.0, min(1.0, 1.0 - risk_value))
                    break
                except ValueError:
                    pass

    accs = metrics["accuracies"]
    if accs:
        metrics["final_accuracy"] = accs[-1]
        metrics["peak_accuracy"] = max(accs)
        metrics["accuracy_std"] = float(np.std(accs))
        metrics["improvement"] = accs[-1] - accs[0] if len(accs) > 1 else 0.0
        metrics["convergence_epoch"] = calculate_convergence_epoch(accs)

    metrics["paper_metrics"] = parse_kv_line(output, "PAPER_METRICS_FINAL:")
    metrics["client_metrics"] = parse_kv_line(output, "PAPER_CLIENT_FINAL:")
    metrics["attack_metrics"] = parse_kv_line(output, "PAPER_ATTACK_FINAL:")

    attack_ability = metrics["attack_metrics"].get("anti_inversion_ability")
    attack_risk = metrics["attack_metrics"].get("leakage_risk")
    if isinstance(attack_ability, (int, float)):
        metrics["gradient_inversion_ability"] = float(attack_ability)
    if isinstance(attack_risk, (int, float)):
        metrics["gradient_inversion_risk"] = float(attack_risk)

    return metrics


def save_experiment_results(
    results_dir,
    datasets,
    dataset_params,
    display_method_name,
    all_results,
    all_abilities,
    all_times,
    all_meta=None,
    tag="baseline",
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_meta = all_meta or {}

    json_file = os.path.join(results_dir, f"comparison_results_{tag}_{timestamp}.json")
    with open(json_file, "w", encoding="utf-8") as f:
        payload = {}
        for dataset, methods in all_results.items():
            payload[dataset] = {}
            for method, accs in methods.items():
                method_cfg = dataset_params.get(dataset, {}).get(method, {})
                ability = all_abilities.get(dataset, {}).get(method, 0.0)
                method_meta = all_meta.get(dataset, {}).get(method, {})
                paper_metrics = dict(method_meta.get("paper_metrics", {}))
                client_metrics = dict(method_meta.get("client_metrics", {}))
                attack_metrics = dict(method_meta.get("attack_metrics", {}))
                final_accuracy = float(
                    paper_metrics.get("final_accuracy", accs[-1] if accs else 0.0)
                )
                ability = float(attack_metrics.get("anti_inversion_ability", ability))
                risk = float(
                    attack_metrics.get(
                        "leakage_risk",
                        max(0.0, min(1.0, 1.0 - ability)),
                    )
                )
                payload[dataset][method] = {
                    "accuracies": accs,
                    "anti_inversion_ability": ability,
                    "risk": risk,
                    "training_time": all_times.get(dataset, {}).get(method, 0.0),
                    "final_accuracy": final_accuracy,
                    "convergence_epoch": float(
                        paper_metrics.get("convergence_epoch", calculate_convergence_epoch(accs))
                    ),
                    "target_epsilon": float(method_cfg.get("target_epsilon", 0.0)),
                    "sigma_factor_gaussian": float(method_cfg.get("sigma_factor_gaussian", 0.005)),
                    "sigma_factor_delayed_feedback": float(method_cfg.get("sigma_factor_delayed_feedback", method_cfg.get("sigma_factor_gaussian", 0.005))),
                    "map_type": method_cfg.get("map_type", ""),
                    "paper_metrics": paper_metrics,
                    "client_metrics": client_metrics,
                    "attack_metrics": attack_metrics,
                }
        json.dump(payload, f, indent=2, ensure_ascii=False)

    csv_file = os.path.join(results_dir, f"comparison_summary_{tag}_{timestamp}.csv")
    with open(csv_file, "w", encoding="utf-8") as f:
        f.write(
            "Dataset,Method,Initial_Accuracy,Final_Accuracy,Best_Accuracy,Mean_Accuracy,Improvement,"
            "Training_Time,Avg_Epoch_Time,Convergence_Epoch,Stability,Accuracy_Variance,Final_Loss,Mean_Loss,"
            "Anti_Inversion_Ability,Leakage_Risk,Avg_PSNR,Best_PSNR,Avg_SSIM,Best_SSIM,Avg_LPIPS,Best_LPIPS,"
            "Attack_Success_Rate,Communication_MB_Per_Round,Total_Communication_MB,Worst_Client_Accuracy,"
            "Jain_Fairness,Dir_Alpha,Target_Epsilon,Peak_Accuracy\n"
        )
        for dataset in datasets:
            for method in all_results.get(dataset, {}):
                accs = all_results.get(dataset, {}).get(method, [])
                if not accs:
                    continue
                init_acc = accs[0]
                peak_acc = max(accs)
                conv = calculate_convergence_epoch(accs)
                ability = all_abilities.get(dataset, {}).get(method, 0.0)
                risk = max(0.0, min(1.0, 1.0 - ability))
                t = all_times.get(dataset, {}).get(method, 0.0)
                improvement = accs[-1] - init_acc
                method_meta = all_meta.get(dataset, {}).get(method, {})
                paper_metrics = method_meta.get("paper_metrics", {})
                client_metrics = method_meta.get("client_metrics", {})
                attack_metrics = method_meta.get("attack_metrics", {})
                final_acc = float(paper_metrics.get("final_accuracy", accs[-1]))
                best_acc = float(paper_metrics.get("best_accuracy", peak_acc))
                mean_acc = float(paper_metrics.get("mean_accuracy", np.mean(accs)))
                conv = float(paper_metrics.get("convergence_epoch", conv))
                stability = float(paper_metrics.get("stability", 0.0))
                accuracy_variance = float(paper_metrics.get("accuracy_variance", np.var(accs)))
                final_loss = float(paper_metrics.get("final_loss", 0.0))
                mean_loss = float(paper_metrics.get("mean_loss", 0.0))
                avg_epoch_time = float(paper_metrics.get("avg_epoch_time", 0.0))
                comm_per_round_mb = float(paper_metrics.get("comm_per_round_mb", 0.0))
                total_comm_mb = float(paper_metrics.get("total_comm_mb", 0.0))
                dir_alpha = float(paper_metrics.get("dir_alpha", 0.0))
                target_epsilon = float(paper_metrics.get("target_epsilon", 0.0))
                ability = float(attack_metrics.get("anti_inversion_ability", ability))
                risk = float(attack_metrics.get("leakage_risk", risk))
                avg_psnr = float(attack_metrics.get("avg_psnr", 0.0))
                best_psnr = float(attack_metrics.get("best_psnr", 0.0))
                avg_ssim = float(attack_metrics.get("avg_ssim", 0.0))
                best_ssim = float(attack_metrics.get("best_ssim", 0.0))
                avg_lpips = float(attack_metrics.get("avg_lpips", 0.0))
                best_lpips = float(attack_metrics.get("best_lpips", 0.0))
                attack_success_rate = float(attack_metrics.get("attack_success_rate", 0.0))
                worst_client_accuracy = float(client_metrics.get("worst_client_accuracy", 0.0))
                jain_fairness = float(client_metrics.get("jain_fairness", 0.0))
                display_method = display_method_name(method)
                f.write(
                    f"{dataset},{display_method},{init_acc:.2f}%,{final_acc:.2f}%,{best_acc:.2f}%,{mean_acc:.2f}%,"
                    f"{improvement:+.2f}%,{t:.1f}s,{avg_epoch_time:.2f}s,{conv:.0f},{stability:.4f},{accuracy_variance:.6f},"
                    f"{final_loss:.6f},{mean_loss:.6f},{ability:.4f},{risk:.4f},{avg_psnr:.4f},{best_psnr:.4f},"
                    f"{avg_ssim:.4f},{best_ssim:.4f},{avg_lpips:.4f},{best_lpips:.4f},{attack_success_rate:.4f},"
                    f"{comm_per_round_mb:.4f},{total_comm_mb:.4f},{worst_client_accuracy:.2f},{jain_fairness:.4f},"
                    f"{dir_alpha:.4f},{target_epsilon:.4f},{peak_acc:.2f}%\n"
                )

    return json_file, csv_file
