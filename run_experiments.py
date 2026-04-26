import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime

import numpy as np


class ExperimentRunner:
    def __init__(self):
        self.results_dir = "experiment_results"
        os.makedirs(self.results_dir, exist_ok=True)

        self.dp_methods = ["none", "gaussian", "dfl"]
        self.datasets = ["CIFAR10", "MNIST", "SVHN"]
        self.num_runs = 3
        self.global_epochs = {"CIFAR10": 40, "MNIST": 30, "SVHN": 40}
        self.local_epochs = {"CIFAR10": 4, "MNIST": 3, "SVHN": 4}
        self.gir_common = {
            "gir_attack_steps": "12",
            "gir_attack_trials": "1",
            "gir_attack_batch_size": "1",
            "gir_attack_lr": "0.1",
            "gir_eval_interval": "0",
            "gir_max_evals_per_client_update": "1",
        }

        # Pure mechanism comparison:
        # keep epsilon/sigma and core training settings aligned across DP methods.
        # Gaussian vs SCM only differ in noise generation mechanism.
        self.dataset_params = {
            "CIFAR10": {
                "none": {"num_clients": "3", "batch_size": "32", "lr": "0.002", "target_epsilon": "8.0", "clipping_bound": "1.5", "local_epoch": "4", "noise_decay": "0.9", "chaotic_factor": "0.0", "sparsity_ratio": "0.0"},
                "gaussian": {"num_clients": "3", "batch_size": "32", "lr": "0.005", "target_epsilon": "8.0", "sigma_factor_gaussian": "0.15", "clipping_bound": "2.5", "local_epoch": "4", "noise_decay": "0.9", "chaotic_factor": "0.0", "sparsity_ratio": "0.0"},
                "dfl": {"num_clients": "3", "batch_size": "32", "lr": "0.005", "target_epsilon": "8.0", "sigma_factor_gaussian": "0.15", "sigma_factor_dfl": "0.15", "clipping_bound": "2.5", "local_epoch": "4", "noise_decay": "1.0", "use_chaotic": "1", "chaotic_factor": "1.0", "dfl_mu": "3.99", "dfl_alpha": "0.82", "dfl_burn_in": "2048", "dfl_decimation": "12", "sparsity_ratio": "0.0"},
            },
            "MNIST": {
                "none": {"num_clients": "3", "batch_size": "64", "lr": "0.01", "target_epsilon": "8.0", "clipping_bound": "1.2", "local_epoch": "3", "noise_decay": "0.9", "chaotic_factor": "0.0", "sparsity_ratio": "0.0"},
                "gaussian": {"num_clients": "3", "batch_size": "64", "lr": "0.01", "target_epsilon": "8.0", "sigma_factor_gaussian": "0.0035", "clipping_bound": "1.2", "local_epoch": "3", "noise_decay": "0.9", "chaotic_factor": "0.0", "sparsity_ratio": "0.0"},
                "dfl": {"num_clients": "3", "batch_size": "64", "lr": "0.01", "target_epsilon": "8.0", "sigma_factor_gaussian": "0.0035", "sigma_factor_dfl": "0.0035", "clipping_bound": "1.2", "local_epoch": "3", "noise_decay": "0.9", "chaotic_factor": "0.5", "dfl_mu": "3.99", "dfl_alpha": "0.98", "sparsity_ratio": "0.0"},
            },
            "SVHN": {
                "none": {"num_clients": "3", "batch_size": "32", "lr": "0.01", "target_epsilon": "8.0", "clipping_bound": "1.5", "local_epoch": "4", "noise_decay": "0.9", "chaotic_factor": "0.0", "sparsity_ratio": "0.0"},
                "gaussian": {"num_clients": "3", "batch_size": "32", "lr": "0.01", "target_epsilon": "8.0", "sigma_factor_gaussian": "0.0045", "clipping_bound": "1.5", "local_epoch": "4", "noise_decay": "0.9", "chaotic_factor": "0.0", "sparsity_ratio": "0.0"},
                "dfl": {"num_clients": "3", "batch_size": "32", "lr": "0.01", "target_epsilon": "8.0", "sigma_factor_gaussian": "0.0045", "sigma_factor_dfl": "0.0045", "clipping_bound": "1.5", "local_epoch": "4", "noise_decay": "0.9", "chaotic_factor": "0.5", "dfl_mu": "3.99", "dfl_alpha": "0.98", "sparsity_ratio": "0.0"},
            },
        }
        self._validate_privacy_order()

    def _validate_privacy_order(self):
        """
        Enforce fair comparison settings:
        1) Shared training hyper-parameters are identical across methods.
        2) Gaussian and SCM use the same target epsilon.
        3) Gaussian and SCM use the same sigma (pure mechanism control).
        """
        for dataset, method_params in self.dataset_params.items():
            required = {"none", "gaussian", "scm"}
            if not required.issubset(method_params.keys()):
                continue

            none_cfg = method_params["none"]
            gaussian_cfg = method_params["gaussian"]
            scm_cfg = method_params["scm"]

            shared_keys = [
                "num_clients", "batch_size", "lr",
                "target_epsilon", "clipping_bound", "local_epoch",
                "noise_decay", "sparsity_ratio",
            ]
            for key in shared_keys:
                n_val = str(none_cfg.get(key))
                g_val = str(gaussian_cfg.get(key))
                s_val = str(scm_cfg.get(key))
                if not (n_val == g_val == s_val):
                    raise ValueError(
                        f"Unfair setting in {dataset}/{key}: none={n_val}, gaussian={g_val}, scm={s_val}"
                    )

            gaussian_eps = float(gaussian_cfg["target_epsilon"])
            scm_eps = float(scm_cfg["target_epsilon"])
            if abs(gaussian_eps - scm_eps) > 1e-12:
                raise ValueError(
                    f"Unfair epsilon in {dataset}: gaussian({gaussian_eps}) must equal scm({scm_eps})."
                )

            gaussian_sigma = float(gaussian_cfg.get("sigma_factor_gaussian", "0.0"))
            dfl_sigma = float(scm_cfg.get("sigma_factor_dfl", str(gaussian_sigma)))
            if gaussian_sigma <= 0:
                raise ValueError(
                    f"Gaussian sigma must be positive in {dataset}, got {gaussian_sigma}."
                )
            if gaussian_sigma > 0 and dfl_sigma > 0 and dfl_sigma > gaussian_sigma:
                raise ValueError(
                    f"Unfair sigma in {dataset}: dfl({dfl_sigma}) must not exceed gaussian({gaussian_sigma})."
                )

    def build_command(self, dataset: str, dp_method: str, run_id: int):
        params = self.dataset_params.get(dataset, self.dataset_params["CIFAR10"])[dp_method]

        cmd = [
            sys.executable,
            "-u",
            "ours.py",
            "--dataset",
            dataset,
            "--global_epoch",
            str(self.global_epochs[dataset]),
            "--num_clients",
            params["num_clients"],
            "--local_epoch",
            params["local_epoch"],
            "--batch_size",
            params["batch_size"],
            "--lr",
            params["lr"],
            "--target_epsilon",
            params["target_epsilon"],
            "--target_delta",
            "1e-5",
            "--clipping_bound",
            params["clipping_bound"],
            "--dir_alpha",
            "100",
            "--device",
            "0",
            "--user_sample_rate",
            "1.0",
            "--seed",
            str(20260312 + run_id),
            "--noise_decay",
            params["noise_decay"],
            "--noise_decay_type",
            "exponential",
            "--sparsity_ratio",
            params.get("sparsity_ratio", "0.0"),
            "--dp_method",
            dp_method,
        ]

        for k, v in self.gir_common.items():
            cmd.extend([f"--{k}", v])

        # Keep method-specific sigma factors consistent with each DP design.
        if dp_method in ("gaussian", "dfl"):
            sigma_gaussian = params.get("sigma_factor_gaussian", "0.002")
            sigma_dfl = params.get("sigma_factor_dfl", sigma_gaussian)
            cmd.extend(
                [
                    "--sigma_factor_gaussian",
                    sigma_gaussian,
                    "--sigma_factor_dfl",
                    sigma_dfl,
                ]
            )

        if dp_method == "none":
            cmd.extend(["--no_noise"])
        elif dp_method == "gaussian":
            cmd.extend(
                [
                    "--use_chaotic",
                    "--chaotic_factor",
                    "0.0",
                ]
            )
        else:
            cmd.extend(
                [
                    "--use_chaotic",
                    "--chaotic_factor",
                    params.get("chaotic_factor", "0.3"),
                    "--dfl_mu",
                    params.get("dfl_mu", "3.99"),
                    "--dfl_alpha",
                    params.get("dfl_alpha", "0.98"),
                    "--dfl_burn_in",
                    params.get("dfl_burn_in", "512"),
                    "--dfl_max_direct_uniform",
                    "4096",
                    "--dfl_decimation",
                    "2",
                ]
            )

        return cmd

    def run_experiment(self, dataset: str, dp_method: str, run_id: int):
        print(f"\nRunning {dataset} - {dp_method} - run {run_id + 1}")
        cmd = self.build_command(dataset, dp_method, run_id)

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0")

        start_time = time.time()
        print(f"Start time: {datetime.fromtimestamp(start_time).strftime('%H:%M:%S')}")

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=".",
            shell=False,
            env=env,
            bufsize=1,
        )

        output_lines = []
        last_log_time = time.time()
        output_file = os.path.join(self.results_dir, f"{dataset}_{dp_method}_run{run_id + 1}_output.txt")
        live_log_file = os.path.join(self.results_dir, f"{dataset}_{dp_method}_run{run_id + 1}_live.log")

        # Create live log file immediately
        with open(live_log_file, "w", encoding="utf-8") as log_f:
            log_f.write("命令: " + " ".join(cmd) + "\n")
            log_f.write(f"开始时间: {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_f.write("=" * 80 + "\n")

        with open(output_file, "w", encoding="utf-8") as f:
            f.write("命令: " + " ".join(cmd) + "\n")

        while True:
            line = process.stdout.readline() if process.stdout is not None else ""
            if line:
                print(line.rstrip("\n"))
                sys.stdout.flush()
                output_lines.append(line)
                # Write to live log file immediately
                with open(live_log_file, "a", encoding="utf-8") as log_f:
                    log_f.write(line)
                    log_f.flush()
                last_log_time = time.time()
            elif process.poll() is not None:
                break
            else:
                if time.time() - last_log_time > 30:
                    elapsed_now = int(time.time() - start_time)
                    status_msg = f"[Runner] still running... {elapsed_now}s elapsed\n"
                    print(status_msg.rstrip("\n"))
                    with open(live_log_file, "a", encoding="utf-8") as log_f:
                        log_f.write(status_msg)
                    last_log_time = time.time()
                time.sleep(0.2)

        return_code = process.wait()
        stdout_text = "".join(output_lines)

        elapsed = time.time() - start_time
        print(f"Elapsed: {elapsed:.1f}s (~{elapsed / 60:.1f}min), return code={return_code}")

        with open(output_file, "a", encoding="utf-8") as f:
            f.write(f"返回码: {return_code}\n")
            f.write(f"运行时间: {elapsed:.1f}秒\n")
            f.write("=== 标准输出 ===\n")
            f.write(stdout_text)
            f.write(f"返回码: {return_code}\n")
            f.write(f"运行时间: {elapsed:.1f}秒\n")
            f.write("=== 标准输出 ===\n")
            f.write(stdout_text)

        if return_code != 0:
            tail = "\n".join(stdout_text.strip().split("\n")[-20:])
            raise RuntimeError(f"Experiment failed (code={return_code})\nLast output:\n{tail}")

        metrics = self.parse_results(stdout_text, elapsed, dataset)
        if not metrics["accuracies"]:
            print("Warning: no accuracy series parsed from output.")
        return metrics

    def parse_results(self, output: str, elapsed_time: float, dataset: str):
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
        }

        if not output:
            return metrics

        # 优先尝试提取 "平均准确率历程:" 后面的列表（含百分号）
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
            if len(vals) >= 2:
                metrics["accuracies"] = vals
                # 继续解析风险等其他指标，不在此处 return，因为还要提取风险等

        # 如果上面没找到，尝试匹配包含百分号的列表（兼容旧输出）
        if not metrics["accuracies"]:
            list_candidates = re.findall(r"\[([0-9\.,\s%\-]+)\]", output)
            # 从后往前找，优先找包含百分号的列表
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
                    if len(vals) >= 2:
                        metrics["accuracies"] = vals
                        break

        # 如果仍然没有，回退到任意长度≥3的列表（但排除明显是数据量的列表）
        if not metrics["accuracies"]:
            list_candidates = re.findall(r"\[([0-9\.,\s\-]+)\]", output)
            for candidate in reversed(list_candidates):
                vals = []
                for item in candidate.split(","):
                    cleaned = re.sub(r"[^0-9\.\-]", "", item.strip())
                    if cleaned:
                        try:
                            num = float(cleaned)
                            # 排除过大数值（如数据量>1000），因为准确率通常在0-100之间
                            if num > 1000:
                                continue
                            vals.append(num)
                        except ValueError:
                            pass
                if len(vals) >= 3:
                    metrics["accuracies"] = vals
                    break

        # 优先提取“抗梯度反演能力”（越高越好）
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

        # 兼容旧日志：只有“风险”时，自动转成能力=1-risk
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

        # 如果 accuracies 不为空，计算其他指标
        accs = metrics["accuracies"]
        if accs:
            metrics["final_accuracy"] = accs[-1]
            metrics["peak_accuracy"] = max(accs)
            metrics["accuracy_std"] = float(np.std(accs))
            metrics["improvement"] = accs[-1] - accs[0] if len(accs) > 1 else 0.0
            metrics["convergence_epoch"] = self.calculate_convergence_epoch(accs)

        return metrics

    def calculate_convergence_epoch(self, accuracies):
        if not accuracies:
            return 0
        final_acc = accuracies[-1]
        threshold = final_acc * 0.95
        for i, acc in enumerate(accuracies):
            if acc >= threshold:
                return i + 1
        return len(accuracies)

    def run_comparison_experiments(self):
        all_results = {d: {} for d in self.datasets}
        all_abilities = {d: {} for d in self.datasets}
        all_times = {d: {} for d in self.datasets}

        for dataset in self.datasets:
            for dp_method in self.dp_methods:
                run_metrics = []
                for run_id in range(self.num_runs):
                    try:
                        m = self.run_experiment(dataset, dp_method, run_id)
                        run_metrics.append(m)
                    except Exception as e:
                        print(f"Run failed: {dataset}/{dp_method}/run{run_id + 1}: {e}")

                if run_metrics:
                    # choose longest parsed series and average by aligned prefix
                    series = [m["accuracies"] for m in run_metrics if m["accuracies"]]
                    if series:
                        min_len = min(len(s) for s in series)
                        avg_series = np.mean([s[:min_len] for s in series], axis=0).tolist()
                    else:
                        avg_series = []

                    all_results[dataset][dp_method] = avg_series
                    all_abilities[dataset][dp_method] = float(np.mean([m["gradient_inversion_ability"] for m in run_metrics]))
                    all_times[dataset][dp_method] = float(np.mean([m["training_time"] for m in run_metrics]))

        self.save_results(all_results, all_abilities, all_times)
        return all_results

    def save_results(self, all_results, all_abilities, all_times):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        json_file = os.path.join(self.results_dir, f"comparison_results_{timestamp}.json")
        with open(json_file, "w", encoding="utf-8") as f:
            payload = {}
            for dataset, methods in all_results.items():
                payload[dataset] = {}
                for method, accs in methods.items():
                    method_cfg = self.dataset_params.get(dataset, {}).get(method, {})
                    ability = all_abilities.get(dataset, {}).get(method, 0.0)
                    payload[dataset][method] = {
                        "accuracies": accs,
                        "anti_inversion_ability": ability,
                        "risk": max(0.0, min(1.0, 1.0 - ability)),
                        "training_time": all_times.get(dataset, {}).get(method, 0.0),
                        "final_accuracy": accs[-1] if accs else 0.0,
                        "convergence_epoch": self.calculate_convergence_epoch(accs),
                        "target_epsilon": float(method_cfg.get("target_epsilon", 0.0)),
                        "sigma_factor_gaussian": float(method_cfg.get("sigma_factor_gaussian", 0.002)),
                        "sigma_factor_dfl": float(method_cfg.get("sigma_factor_dfl", method_cfg.get("sigma_factor_gaussian", 0.002))),
                    }
            json.dump(payload, f, indent=2, ensure_ascii=False)

        csv_file = os.path.join(self.results_dir, f"comparison_summary_{timestamp}.csv")
        with open(csv_file, "w", encoding="utf-8") as f:
            f.write("Dataset,DP_Method,Initial_Accuracy,Final_Accuracy,Improvement,Training_Time,Convergence_Epoch,Anti_Inversion_Ability,Leakage_Risk,Peak_Accuracy\n")
            for dataset in self.datasets:
                for method in self.dp_methods:
                    accs = all_results.get(dataset, {}).get(method, [])
                    if not accs:
                        continue
                    init_acc = accs[0]
                    final_acc = accs[-1]
                    peak_acc = max(accs)
                    conv = self.calculate_convergence_epoch(accs)
                    ability = all_abilities.get(dataset, {}).get(method, 0.0)
                    risk = max(0.0, min(1.0, 1.0 - ability))
                    t = all_times.get(dataset, {}).get(method, 0.0)
                    improvement = final_acc - init_acc
                    f.write(f"{dataset},{method},{init_acc:.2f}%,{final_acc:.2f}%,{improvement:+.2f}%,{t:.1f}s,{conv:.0f},{ability:.4f},{risk:.4f},{peak_acc:.2f}%\n")

        print(f"Saved JSON: {json_file}")
        print(f"Saved CSV : {csv_file}")

    def run_all_experiments(self):
        print("=" * 80)
        print("DP comparison experiments")
        print("=" * 80)
        print(f"Datasets: {self.datasets}")
        print(f"Methods : {self.dp_methods}")
        print(f"Runs/method: {self.num_runs}")
        print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        results = self.run_comparison_experiments()

        print("\n" + "=" * 80)
        print("Done")
        for dataset in self.datasets:
            if dataset in results:
                print(f"\n{dataset}:")
                for method in self.dp_methods:
                    accs = results[dataset].get(method, [])
                    if accs:
                        print(f"  {method.upper():9s} {accs[0]:.2f}% -> {accs[-1]:.2f}%")
                    else:
                        print(f"  {method.upper():9s} no parsed accuracy")
        print("=" * 80)
        return results


def main():
    parser = argparse.ArgumentParser(description="Run DP method comparison experiments")
    parser.add_argument("--num_runs", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--datasets", nargs="+", default=["CIFAR10", "MNIST", "SVHN"])
    parser.add_argument("--dp_methods", nargs="+", default=["none", "gaussian", "dfl"])
    args = parser.parse_args()

    runner = ExperimentRunner()
    runner.num_runs = args.num_runs
    runner.datasets = args.datasets
    runner.dp_methods = args.dp_methods
    if args.epochs is not None:
        for d in runner.datasets:
            runner.global_epochs[d] = args.epochs

    runner.run_all_experiments()


if __name__ == "__main__":
    main()
