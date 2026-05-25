import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime

import numpy as np

from experiment_config import (
    DEFAULT_ABLATION_SWEEPS,
    DEFAULT_DATASETS,
    DEFAULT_DP_METHODS,
    DEFAULT_GIR_COMMON,
    DEFAULT_GLOBAL_EPOCHS,
    DEFAULT_LOCAL_EPOCHS,
    build_ablation_specs,
    build_default_specs,
    clone_dataset_params,
    display_method_name,
    get_base_params,
    sync_feedback_sigma_with_gaussian_if_needed,
)
from experiment_results_io import (
    aggregate_metric_dicts,
    calculate_convergence_epoch,
    parse_experiment_output,
    save_experiment_results,
)


class ExperimentRunner:
    def __init__(self):
        self.results_dir = "experiment_results"
        os.makedirs(self.results_dir, exist_ok=True)

        self.python_executable = self._select_python_executable()

        self.dp_methods = list(DEFAULT_DP_METHODS)
        self.datasets = list(DEFAULT_DATASETS)
        self.num_runs = 3
        self.experiment_mode = "baseline"
        self.ablation_target = "quantization"
        self.ablation_dataset = "CIFAR10"
        self.global_epochs = dict(DEFAULT_GLOBAL_EPOCHS)
        self.local_epochs = dict(DEFAULT_LOCAL_EPOCHS)
        self.gir_common = dict(DEFAULT_GIR_COMMON)
        self.dataset_params = clone_dataset_params()
        self.ablation_sweeps = dict(DEFAULT_ABLATION_SWEEPS)

    def _sync_feedback_sigma_with_gaussian_if_needed(self):
        sync_feedback_sigma_with_gaussian_if_needed(self.dataset_params, self.dp_methods)

    def _clone_params(self, params):
        return {k: str(v) for k, v in params.items()}

    def _get_base_params(self, dataset: str, dp_method: str):
        return get_base_params(self.dataset_params, dataset, dp_method)

    @staticmethod
    def _display_method_name(method: str) -> str:
        return display_method_name(method)

    def _build_default_specs(self, dataset: str):
        return build_default_specs(self.dataset_params, dataset)

    def _build_ablation_specs(self, dataset: str, target: str):
        return build_ablation_specs(self.dataset_params, self.ablation_sweeps, dataset, target)

    def _python_has_torch(self, python_executable: str) -> bool:
        try:
            result = subprocess.run(
                [python_executable, "-c", "import torch; print(torch.__version__)"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=20,
                check=False,
            )
            return result.returncode == 0
        except Exception:
            return False

    def _select_python_executable(self) -> str:
        """
        Prefer a Python interpreter that already has PyTorch installed.
        This avoids failing when the shell's default Python is a base env without torch.
        """
        candidates = []

        env_python = os.environ.get("PYTHON_EXECUTABLE")
        if env_python:
            candidates.append(env_python)

        conda_prefix = os.environ.get("CONDA_PREFIX")
        if conda_prefix:
            candidates.append(os.path.join(conda_prefix, "python.exe"))

        candidates.extend(
            [
                sys.executable,
                r"F:\CondaEnvs\dynamic_env\python.exe",
                r"F:\CondaEnvs\drone_simulation\python.exe",
                r"F:\Anaconda\python.exe",
            ]
        )

        seen = set()
        for candidate in candidates:
            if not candidate:
                continue
            candidate = os.path.abspath(candidate)
            if candidate in seen:
                continue
            seen.add(candidate)
            if os.path.exists(candidate) and self._python_has_torch(candidate):
                print(f"Using Python with torch: {candidate}")
                return candidate

        raise RuntimeError(
            "No Python interpreter with torch was found. "
            "Set PYTHON_EXECUTABLE to an environment that has torch installed "
            "(for example F:\\CondaEnvs\\dynamic_env\\python.exe)."
        )

    def _validate_privacy_order(self):
        """
        Enforce fair comparison settings:
        1) Shared training hyper-parameters are identical across methods.
        2) Gaussian and DFL use the same target epsilon.
        3) Gaussian and DFL use the same sigma (pure mechanism control).
        """
        selected_methods = set(self.dp_methods or [])
        compare_gaussian_and_feedback = "gaussian" in selected_methods and "delayed_feedback" in selected_methods

        for dataset, method_params in self.dataset_params.items():
            required = {"none", "gaussian", "delayed_feedback"}
            if not required.issubset(method_params.keys()):
                continue

            none_cfg = method_params["none"]
            gaussian_cfg = method_params["gaussian"]
            feedback_cfg = method_params["delayed_feedback"]

            shared_keys = [
                "num_clients", "batch_size", "lr",
                "target_epsilon", "clipping_bound", "local_epoch",
                "noise_decay", "sparsity_ratio",
            ]
            for key in shared_keys:
                n_val = str(none_cfg.get(key))
                g_val = str(gaussian_cfg.get(key))
                s_val = str(feedback_cfg.get(key))
                if not (n_val == g_val == s_val):
                    raise ValueError(
                        f"Unfair setting in {dataset}/{key}: none={n_val}, gaussian={g_val}, delayed_feedback={s_val}"
                    )

            if compare_gaussian_and_feedback:
                gaussian_eps = float(gaussian_cfg["target_epsilon"])
                feedback_eps = float(feedback_cfg["target_epsilon"])
                if abs(gaussian_eps - feedback_eps) > 1e-12:
                    raise ValueError(
                        f"Unfair epsilon in {dataset}: gaussian({gaussian_eps}) must equal delayed_feedback({feedback_eps})."
                    )

                gaussian_sigma = float(gaussian_cfg.get("sigma_factor_gaussian", "0.005"))
                feedback_sigma = float(feedback_cfg.get("sigma_factor_delayed_feedback", str(gaussian_sigma)))
                if gaussian_sigma <= 0:
                    raise ValueError(
                        f"Gaussian sigma must be positive in {dataset}, got {gaussian_sigma}."
                    )
                if abs(feedback_sigma - gaussian_sigma) > 1e-12:
                    raise ValueError(
                        f"Unfair sigma in {dataset}: gaussian({gaussian_sigma}) must equal delayed_feedback({feedback_sigma})."
                    )

    def build_command(self, dataset: str, dp_method: str, run_id: int, params_override=None):
        params = self._get_base_params(dataset, dp_method)
        if params_override:
            params.update(self._clone_params(params_override))

        cmd = [
            self.python_executable,
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
        if dp_method in ("gaussian", "delayed_feedback"):
            sigma_gaussian = params.get("sigma_factor_gaussian", "0.005")
            sigma_feedback = params.get("sigma_factor_delayed_feedback", sigma_gaussian)
            cmd.extend(
                [
                    "--sigma_factor_gaussian",
                    sigma_gaussian,
                    "--sigma_factor_delayed_feedback",
                    sigma_feedback,
                ]
            )

        if dp_method == "none":
            cmd.extend(["--no_noise"])
        elif dp_method == "gaussian":
            pass
        else:
            cmd.extend(
                [
                    "--a",
                    params.get("a", "4.0"),
                    "--b",
                    params.get("b", "501.0"),
                    "--k",
                    params.get("k", "7"),
                    "--burn_in",
                    params.get("burn_in", "2048"),
                    "--decimation",
                    params.get("decimation", "12"),
                    "--q_bits",
                    params.get("q_bits", "32"),
                    "--q_mode",
                    params.get("q_mode", "round"),
                    "--map_type",
                    params.get("map_type", "logistic"),
                ]
            )

        return cmd

    def run_experiment(self, dataset: str, dp_method: str, run_id: int, params_override=None, label=None):
        run_label = label or dp_method
        print(f"\nRunning {dataset} - {run_label} - run {run_id + 1}")
        cmd = self.build_command(dataset, dp_method, run_id, params_override=params_override)

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

        while True:
            line = process.stdout.readline() if process.stdout is not None else ""
            if line:
                print(line.rstrip("\n"))
                output_lines.append(line)
                last_log_time = time.time()
            elif process.poll() is not None:
                break
            else:
                if time.time() - last_log_time > 30:
                    elapsed_now = int(time.time() - start_time)
                    print(f"[Runner] still running... {elapsed_now}s elapsed")
                    last_log_time = time.time()
                time.sleep(0.2)

        return_code = process.wait()
        stdout_text = "".join(output_lines)

        elapsed = time.time() - start_time
        print(f"Elapsed: {elapsed:.1f}s (~{elapsed / 60:.1f}min), return code={return_code}")

        output_file = os.path.join(self.results_dir, f"{dataset}_{run_label}_run{run_id + 1}_output.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("命令: " + " ".join(cmd) + "\n")
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
        return parse_experiment_output(output, elapsed_time)

    def calculate_convergence_epoch(self, accuracies):
        return calculate_convergence_epoch(accuracies)

    def run_comparison_experiments(self, method_specs_by_dataset=None, result_tag="baseline"):
        if method_specs_by_dataset is None:
            method_specs_by_dataset = {
                dataset: [
                    spec for spec in self._build_default_specs(dataset)
                    if spec["dp_method"] in self.dp_methods
                ]
                for dataset in self.datasets
            }

        all_results = {d: {} for d in self.datasets}
        all_abilities = {d: {} for d in self.datasets}
        all_times = {d: {} for d in self.datasets}
        all_meta = {d: {} for d in self.datasets}

        for dataset in self.datasets:
            for spec in method_specs_by_dataset.get(dataset, []):
                run_metrics = []
                for run_id in range(self.num_runs):
                    try:
                        m = self.run_experiment(
                            dataset,
                            spec["dp_method"],
                            run_id,
                            params_override=spec.get("params"),
                            label=spec["label"],
                        )
                        run_metrics.append(m)
                    except Exception as e:
                        print(f"Run failed: {dataset}/{spec['label']}/run{run_id + 1}: {e}")

                if run_metrics:
                    # choose longest parsed series and average by aligned prefix
                    series = [m["accuracies"] for m in run_metrics if m["accuracies"]]
                    if series:
                        min_len = min(len(s) for s in series)
                        avg_series = np.mean([s[:min_len] for s in series], axis=0).tolist()
                    else:
                        avg_series = []

                    all_results[dataset][spec["label"]] = avg_series
                    all_abilities[dataset][spec["label"]] = float(np.mean([m["gradient_inversion_ability"] for m in run_metrics]))
                    all_times[dataset][spec["label"]] = float(np.mean([m["training_time"] for m in run_metrics]))
                    all_meta[dataset][spec["label"]] = {
                        "paper_metrics": aggregate_metric_dicts([m.get("paper_metrics", {}) for m in run_metrics]),
                        "client_metrics": aggregate_metric_dicts([m.get("client_metrics", {}) for m in run_metrics]),
                        "attack_metrics": aggregate_metric_dicts([m.get("attack_metrics", {}) for m in run_metrics]),
                    }

        self.save_results(all_results, all_abilities, all_times, all_meta, tag=result_tag)
        return all_results

    def save_results(self, all_results, all_abilities, all_times, all_meta=None, tag="baseline"):
        json_file, csv_file = save_experiment_results(
            results_dir=self.results_dir,
            datasets=self.datasets,
            dataset_params=self.dataset_params,
            display_method_name=self._display_method_name,
            all_results=all_results,
            all_abilities=all_abilities,
            all_times=all_times,
            all_meta=all_meta,
            tag=tag,
        )
        print(f"Saved JSON: {json_file}")
        print(f"Saved summary table: {csv_file}")

    def run_all_experiments(self):
        if self.experiment_mode == "ablation":
            return self.run_ablation_experiments()

        print("=" * 80)
        print("DP comparison experiments")
        print("=" * 80)
        print(f"Datasets: {self.datasets}")
        print(f"Methods : {self.dp_methods}")
        print(f"Runs/method: {self.num_runs}")
        print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        results = self.run_comparison_experiments(result_tag="baseline")

        print("\n" + "=" * 80)
        print("Done")
        for dataset in self.datasets:
            if dataset in results:
                print(f"\n{dataset}:")
                for method in self.dp_methods:
                    accs = results[dataset].get(method, [])
                    if accs:
                        print(f"  {self._display_method_name(method):18s} {accs[0]:.2f}% -> {accs[-1]:.2f}%")
                    else:
                        print(f"  {self._display_method_name(method):18s} no parsed accuracy")
        print("=" * 80)
        return results

    def run_ablation_experiments(self):
        print("=" * 80)
        print("DP ablation experiments")
        print("=" * 80)
        print(f"Datasets: {self.datasets}")
        print(f"Ablation target: {self.ablation_target}")
        print(f"Runs/method: {self.num_runs}")
        print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        targets = ["decimation"] if self.ablation_target == "all" else [self.ablation_target]
        all_results = {}

        for target in targets:
            print("\n" + "=" * 80)
            print(f"Running ablation sweep: {target}")
            print("=" * 80)
            method_specs_by_dataset = {
                dataset: self._build_ablation_specs(dataset, target) for dataset in self.datasets
            }
            all_results[target] = self.run_comparison_experiments(
                method_specs_by_dataset=method_specs_by_dataset,
                result_tag=f"ablation_{target}",
            )

        print("\n" + "=" * 80)
        print("Ablation done")
        print("=" * 80)
        return all_results


def main():
    parser = argparse.ArgumentParser(description="Run DP method comparison experiments")
    parser.add_argument("--num_runs", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--datasets", nargs="+", default=["CIFAR10", "MNIST", "SVHN"])
    parser.add_argument("--dp_methods", nargs="+", default=["none", "gaussian", "delayed_feedback"])
    parser.add_argument("--mode", type=str, default="baseline", choices=["baseline", "ablation"])
    parser.add_argument("--ablation_target", type=str, default="decimation", choices=["decimation", "all"])
    parser.add_argument("--ablation_dataset", type=str, default="CIFAR10")
    parser.add_argument("--map_type", type=str, default=None, choices=["logistic", "tent", "sine"],
                        help="Override delayed-feedback seed map f(x) for all datasets")
    args = parser.parse_args()

    runner = ExperimentRunner()
    runner.num_runs = args.num_runs
    runner.experiment_mode = args.mode
    runner.ablation_target = args.ablation_target
    runner.ablation_dataset = args.ablation_dataset
    runner.datasets = [args.ablation_dataset] if args.mode == "ablation" else args.datasets
    runner.dp_methods = args.dp_methods
    if args.map_type is not None:
        for method_params in runner.dataset_params.values():
            if "delayed_feedback" in method_params:
                method_params["delayed_feedback"]["map_type"] = args.map_type
    runner._sync_feedback_sigma_with_gaussian_if_needed()
    runner._validate_privacy_order()
    if args.epochs is not None:
        for d in runner.datasets:
            runner.global_epochs[d] = args.epochs

    runner.run_all_experiments()


if __name__ == "__main__":
    main()
