"""Sweep dfl_decimation (Gap) across CIFAR10, sequential."""
import subprocess, time, os
from datetime import datetime

PYTHON = r"D:\anaconda\envs\pp\python.exe"
DATASETS = ["CIFAR10"]
GAPS = [12, 24]
EPOCHS = 60

os.makedirs("experiment_results", exist_ok=True)
log_path = "experiment_results/gap_sweep.log"

def log(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")

total = len(DATASETS) * len(GAPS)
current = 0
log(f"Starting CIFAR10 Gap sweep: {total} experiments")

for ds in DATASETS:
    for gap in GAPS:
        current += 1
        out_file = f"experiment_results/{ds}_dfl_run{gap}_output.txt"
        live_file = f"experiment_results/{ds}_dfl_run{gap}_live.log"

        cmd = [
            PYTHON, "-u", "ours.py",
            "--dataset", ds,
            "--global_epoch", str(EPOCHS),
            "--num_clients", "3",
            "--batch_size", "32",
            "--local_epoch", "4",
            "--lr", "0.004",
            "--target_epsilon", "8.0",
            "--target_delta", "1e-5",
            "--clipping_bound", "2.0",
            "--dir_alpha", "100",
            "--device", "0",
            "--user_sample_rate", "1.0",
            "--seed", str(20260312),
            "--noise_decay", "1.0",
            "--noise_decay_type", "exponential",
            "--sparsity_ratio", "0.0",
            "--dp_method", "dfl",
            "--sigma_factor_gaussian", "0.10",
            "--sigma_factor_dfl", "0.10",
            "--use_chaotic",
            "--chaotic_factor", "1.0",
            "--dfl_mu", "3.99",
            "--dfl_alpha", "0.85",
            "--dfl_burn_in", "2048",
            "--dfl_decimation", str(gap),
            "--dfl_max_direct_uniform", "4096",
            "--gir_attack_steps", "12",
            "--gir_attack_trials", "1",
            "--gir_attack_batch_size", "1",
            "--gir_attack_lr", "0.1",
            "--gir_eval_interval", "0",
            "--gir_max_evals_per_client_update", "1",
        ]

        log(f"[{current}/{total}] {ds} gap={gap}")

        with open(out_file, "w", encoding="utf-8") as of, open(live_file, "w", encoding="utf-8") as lf:
            of.write(f"Command: {' '.join(cmd)}\nStart: {datetime.now()}\n{'='*80}\n")
            lf.write(f"Command: {' '.join(cmd)}\nStart: {datetime.now()}\n{'='*80}\n")

            t0 = time.time()
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
            all_output = []
            for line in proc.stdout:
                all_output.append(line)
                lf.write(line)
                lf.flush()
            proc.wait()
            elapsed = time.time() - t0

            full = "".join(all_output)
            of.write(full)
            of.write(f"\nReturn code: {proc.returncode}\nElapsed: {elapsed:.1f}s\n")

        if proc.returncode == 0:
            log(f"[{current}/{total}] {ds} gap={gap} DONE ({elapsed/60:.1f} min)")
        else:
            log(f"[{current}/{total}] {ds} gap={gap} FAILED (rc={proc.returncode})")

log("Gap sweep complete")
