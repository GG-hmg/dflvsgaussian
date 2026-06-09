import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime

import numpy as np


VALID_NOISE_KINDS = [
    "gaussian",
    "dfl_uniform",
    "dfl_gaussian",
    "mix_dgauss_gauss",
    "mix_dgauss_dchaos",
]


class ExperimentRunner:
    """
    Batch driver for ours.py — runs the same training config for several
    noise_kind variants and writes timestamped artifacts under
    experiment_results/.

    File naming (one session = one timestamp, shared across every artifact):
        {noise_kind}_{dataset}_{ts}.log    real-time training log
        {noise_kind}_{dataset}_{ts}.txt    final summary
        comparison_results_{ts}.json       JSON aggregate
        comparison_summary_{ts}.csv        CSV aggregate
    """

    def __init__(self):
        self.results_dir = "experiment_results"
        os.makedirs(self.results_dir, exist_ok=True)
        self.session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.noise_kinds = ["gaussian", "dfl_gaussian"]
        self.datasets = ["CIFAR10", "MNIST", "SVHN"]
        self.global_epochs = {"CIFAR10": 30, "MNIST": 30, "SVHN": 30}
        self.mix_alpha = 0.5

        self.gir_common = {
            "gir_attack_steps": "12",
            # trials=5: with trials=1 the single-attack noise dominated the
            # per-epoch anti-inversion signal (~0.46–0.74 swings observed).
            # Averaging over 5 trials shrinks that to roughly ±0.05.
            "gir_attack_trials": "5",
            "gir_attack_batch_size": "1",
            "gir_attack_lr": "0.1",
            "gir_eval_interval": "0",
            "gir_max_evals_per_client_update": "1",
        }

        # Per-dataset training hyper-parameters. Shared by every noise_kind:
        # noise injection only changes which RNG fills the noise tensor — the
        # rest of training is identical so the comparison stays fair.
        #
        # target_epsilon=20.0: ε=8 collapsed every noise_kind to ~10–30% on
        # CIFAR10, leaving no room to differentiate mechanisms. ε=20 lifts the
        # accuracy ceiling enough for treatments to spread, at the cost of a
        # weaker (still meaningful) DP guarantee for the gaussian baseline.
        self.dataset_params = {
            "CIFAR10":      {"num_clients": "3", "batch_size": "32", "lr": "0.004", "local_epoch": "4",
                             "target_epsilon": "20.0", "clipping_bound": "2.0", "sparsity_ratio": "0.0",
                             "dfl_decimation": "11"},
            "MNIST":        {"num_clients": "3", "batch_size": "64", "lr": "0.002", "local_epoch": "3",
                             "target_epsilon": "20.0", "clipping_bound": "1.5", "sparsity_ratio": "0.4",
                             "dfl_decimation": "8"},
            "SVHN":         {"num_clients": "3", "batch_size": "32", "lr": "0.004", "local_epoch": "4",
                             "target_epsilon": "20.0", "clipping_bound": "1.5", "sparsity_ratio": "0.4",
                             "dfl_decimation": "8"},
            "FashionMNIST": {"num_clients": "3", "batch_size": "64", "lr": "0.003", "local_epoch": "3",
                             "target_epsilon": "20.0", "clipping_bound": "1.5", "sparsity_ratio": "0.4",
                             "dfl_decimation": "8"},
        }

    def build_command(self, dataset: str, noise_kind: str):
        if noise_kind not in VALID_NOISE_KINDS:
            raise ValueError(f"Unknown noise_kind {noise_kind!r}; choose from {VALID_NOISE_KINDS}")

        params = self.dataset_params.get(dataset, self.dataset_params["CIFAR10"])
        cmd = [
            sys.executable, "-u", "ours.py",
            "--dataset", dataset,
            "--global_epoch", str(self.global_epochs[dataset]),
            "--num_clients", params["num_clients"],
            "--local_epoch", params["local_epoch"],
            "--batch_size", params["batch_size"],
            "--lr", params["lr"],
            "--target_epsilon", params["target_epsilon"],
            "--target_delta", "1e-5",
            "--clipping_bound", params["clipping_bound"],
            "--dir_alpha", "100",
            "--device", "0",
            "--user_sample_rate", "1.0",
            "--seed", "20260313",
            "--sparsity_ratio", params.get("sparsity_ratio", "0.0"),
            "--noise_kind", noise_kind,
            "--mix_alpha", str(self.mix_alpha),
        ]
        for k, v in self.gir_common.items():
            cmd.extend([f"--{k}", v])

        # DFL parameters are needed by every non-gaussian noise_kind. Passing
        # them unconditionally is harmless for gaussian (just ignored).
        cmd.extend([
            "--dfl_a", "4.0",
            "--dfl_b", "501.0",
            "--dfl_k", "3",
            "--dfl_burn_in", "2048",
            "--dfl_decimation", params.get("dfl_decimation", "11"),
        ])
        return cmd

    def run_experiment(self, dataset: str, noise_kind: str):
        print(f"\nRunning {dataset} - {noise_kind}")
        cmd = self.build_command(dataset, noise_kind)

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0")

        start_time = time.time()
        print(f"Start time: {datetime.fromtimestamp(start_time).strftime('%H:%M:%S')}")

        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
            cwd=".", shell=False, env=env, bufsize=1,
        )

        ts = self.session_timestamp
        output_file = os.path.join(self.results_dir, f"{noise_kind}_{dataset}_{ts}.txt")
        live_log_file = os.path.join(self.results_dir, f"{noise_kind}_{dataset}_{ts}.log")

        with open(live_log_file, "w", encoding="utf-8") as log_f:
            log_f.write("命令: " + " ".join(cmd) + "\n")
            log_f.write(f"开始时间: {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_f.write("=" * 80 + "\n")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("命令: " + " ".join(cmd) + "\n")

        output_lines = []
        last_log_time = time.time()
        while True:
            line = process.stdout.readline() if process.stdout is not None else ""
            if line:
                print(line.rstrip("\n"))
                sys.stdout.flush()
                output_lines.append(line)
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

        if return_code != 0:
            tail = "\n".join(stdout_text.strip().split("\n")[-20:])
            raise RuntimeError(f"Experiment failed (code={return_code})\nLast output:\n{tail}")

        metrics = self.parse_results(stdout_text, elapsed, dataset)
        if not metrics["accuracies"]:
            print("Warning: no accuracy series parsed from output.")
        return metrics

    def parse_results(self, output: str, elapsed_time: float, dataset: str):
        metrics = {
            "accuracies": [], "training_time": elapsed_time,
            "final_accuracy": 0, "convergence_epoch": 0,
            "peak_accuracy": 0, "accuracy_std": 0, "improvement": 0,
            "gradient_inversion_ability": 0, "gradient_inversion_risk": 0,
        }
        if not output:
            return metrics

        m = re.search(r"平均准确率历程:\s*\[([^\]]+)\]", output)
        if m:
            vals = []
            for item in m.group(1).split(","):
                cleaned = re.sub(r"[^0-9\.\-]", "", item.strip())
                if cleaned:
                    try:
                        vals.append(float(cleaned))
                    except ValueError:
                        pass
            if len(vals) >= 2:
                metrics["accuracies"] = vals

        ability_match = re.search(r"最终抗梯度反演能力:\s*(\d+\.\d+)", output)
        if not ability_match:
            ability_match = re.search(r"平均抗梯度反演能力:\s*(\d+\.\d+)", output)
        if ability_match:
            metrics["gradient_inversion_ability"] = float(ability_match.group(1))
            metrics["gradient_inversion_risk"] = max(0.0, min(1.0, 1.0 - metrics["gradient_inversion_ability"]))

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
        threshold = accuracies[-1] * 0.95
        for i, acc in enumerate(accuracies):
            if acc >= threshold:
                return i + 1
        return len(accuracies)

    def run_comparison_experiments(self):
        all_results = {d: {} for d in self.datasets}
        all_abilities = {d: {} for d in self.datasets}
        all_times = {d: {} for d in self.datasets}

        for dataset in self.datasets:
            for noise_kind in self.noise_kinds:
                try:
                    m = self.run_experiment(dataset, noise_kind)
                except Exception as e:
                    print(f"Run failed: {dataset}/{noise_kind}: {e}")
                    continue
                all_results[dataset][noise_kind] = m["accuracies"]
                all_abilities[dataset][noise_kind] = float(m["gradient_inversion_ability"])
                all_times[dataset][noise_kind] = float(m["training_time"])

        self.save_results(all_results, all_abilities, all_times)
        return all_results

    def save_results(self, all_results, all_abilities, all_times):
        ts = self.session_timestamp
        json_file = os.path.join(self.results_dir, f"comparison_results_{ts}.json")
        with open(json_file, "w", encoding="utf-8") as f:
            payload = {}
            for dataset, kinds in all_results.items():
                payload[dataset] = {}
                for kind, accs in kinds.items():
                    ds_cfg = self.dataset_params.get(dataset, {})
                    ability = all_abilities.get(dataset, {}).get(kind, 0.0)
                    payload[dataset][kind] = {
                        "accuracies": accs,
                        "anti_inversion_ability": ability,
                        "risk": max(0.0, min(1.0, 1.0 - ability)),
                        "training_time": all_times.get(dataset, {}).get(kind, 0.0),
                        "final_accuracy": accs[-1] if accs else 0.0,
                        "convergence_epoch": self.calculate_convergence_epoch(accs),
                        "target_epsilon": float(ds_cfg.get("target_epsilon", 0.0)),
                        "clipping_bound": float(ds_cfg.get("clipping_bound", 0.0)),
                        "mix_alpha": float(self.mix_alpha) if kind.startswith("mix_") else None,
                    }
            json.dump(payload, f, indent=2, ensure_ascii=False)

        csv_file = os.path.join(self.results_dir, f"comparison_summary_{ts}.csv")
        with open(csv_file, "w", encoding="utf-8") as f:
            f.write("Dataset,Noise_Kind,Initial_Accuracy,Final_Accuracy,Improvement,Training_Time,Convergence_Epoch,Anti_Inversion_Ability,Leakage_Risk,Peak_Accuracy\n")
            for dataset in self.datasets:
                for kind in self.noise_kinds:
                    accs = all_results.get(dataset, {}).get(kind, [])
                    if not accs:
                        continue
                    init_acc, final_acc = accs[0], accs[-1]
                    ability = all_abilities.get(dataset, {}).get(kind, 0.0)
                    f.write(
                        f"{dataset},{kind},{init_acc:.2f}%,{final_acc:.2f}%,"
                        f"{final_acc - init_acc:+.2f}%,"
                        f"{all_times.get(dataset, {}).get(kind, 0.0):.1f}s,"
                        f"{self.calculate_convergence_epoch(accs):.0f},"
                        f"{ability:.4f},{max(0.0, min(1.0, 1.0 - ability)):.4f},"
                        f"{max(accs):.2f}%\n"
                    )

        print(f"Saved JSON: {json_file}")
        print(f"Saved CSV : {csv_file}")

    def run_all_experiments(self):
        print("=" * 80)
        print("Noise-mechanism comparison experiments")
        print("=" * 80)
        print(f"Datasets   : {self.datasets}")
        print(f"Noise kinds: {self.noise_kinds}")
        if any(k.startswith("mix_") for k in self.noise_kinds):
            print(f"Mix alpha  : {self.mix_alpha}")
        print(f"Start      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        results = self.run_comparison_experiments()

        print("\n" + "=" * 80)
        print("Done")
        for dataset in self.datasets:
            if dataset in results:
                print(f"\n{dataset}:")
                for kind in self.noise_kinds:
                    accs = results[dataset].get(kind, [])
                    if accs:
                        print(f"  {kind:20s} {accs[0]:.2f}% -> {accs[-1]:.2f}%")
                    else:
                        print(f"  {kind:20s} no parsed accuracy")
        print("=" * 80)
        return results


def main():
    parser = argparse.ArgumentParser(description="Run noise-mechanism comparison experiments")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--datasets", nargs="+", default=["CIFAR10", "MNIST", "SVHN"])
    parser.add_argument("--noise_kinds", nargs="+", default=["gaussian", "dfl_gaussian"],
                        choices=VALID_NOISE_KINDS,
                        help="Which noise mechanisms to compare in this session")
    parser.add_argument("--mix_alpha", type=float, default=0.5,
                        help="Mix ratio α ∈ [0,1] for mix_* noise kinds (default: 0.5)")
    args = parser.parse_args()

    runner = ExperimentRunner()
    runner.datasets = args.datasets
    runner.noise_kinds = args.noise_kinds
    runner.mix_alpha = float(args.mix_alpha)
    if args.epochs is not None:
        for d in runner.datasets:
            runner.global_epochs[d] = args.epochs
    runner.run_all_experiments()


if __name__ == "__main__":
    main()
