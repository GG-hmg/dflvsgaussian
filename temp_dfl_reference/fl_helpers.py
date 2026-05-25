import builtins
import copy
import datetime
import math
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from data import *
from gradient_inversion_risk_simulator import (
    DefenseSimulationConfig,
    GradientInversionRiskConfig,
    simulate_gradient_inversion_risk,
)
from net import *
from options import parse_args
from runtime_support import stable_u64_seed
from utils import (
    adaptive_privacy_budget,
    compute_fisher_diag,
    generate_delayed_feedback_noise,
    generate_random_gaussian_noise_like,
    sparsify_gradients,
)


args = parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _configure_console_output() -> None:
    """Avoid UnicodeEncodeError on Windows terminals."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except Exception:
            pass


def _fix_mojibake_text(text):
    """Best-effort mojibake recovery for printed Chinese text."""
    if not isinstance(text, str) or not text:
        return text
    if text.isascii():
        return text
    suspicious = ("é", "ç", "î", "å", "ï", "æ", "â", "", "", "")
    if not any(token in text for token in suspicious):
        return text
    try:
        return text.encode("gbk", errors="strict").decode("utf-8", errors="strict")
    except Exception:
        return text


_original_print = builtins.print


def _safe_print(*args, **kwargs):
    fixed = tuple(_fix_mojibake_text(a) if isinstance(a, str) else a for a in args)
    _original_print(*fixed, **kwargs)


builtins.print = _safe_print
_configure_console_output()


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def _stable_u64_seed(*parts) -> int:
    return stable_u64_seed(*parts)


def _clip_tensors_by_global_norm(tensors, clipping_bound):
    """Clip a tensor list by its combined L2 norm."""
    valid_tensors = [tensor for tensor in tensors if tensor is not None]
    if not valid_tensors or clipping_bound <= 0:
        return tensors

    norm_sq = torch.zeros((), device=valid_tensors[0].device, dtype=valid_tensors[0].dtype)
    for tensor in valid_tensors:
        norm_sq = norm_sq + tensor.detach().pow(2).sum()
    norm = torch.sqrt(norm_sq)

    if norm <= clipping_bound:
        return tensors

    scale = clipping_bound / (norm + 1e-12)
    return [tensor * scale if tensor is not None else None for tensor in tensors]


def _clip_local_gradients_for_stability(model, clipping_bound):
    """Optional local gradient clipping for training stability, not DP accounting."""
    if clipping_bound <= 0:
        return

    grad_params = [param for param in model.parameters() if param.grad is not None]
    if not grad_params:
        return

    clipped_grads = _clip_tensors_by_global_norm([param.grad for param in grad_params], clipping_bound)
    for param, clipped_grad in zip(grad_params, clipped_grads):
        param.grad.data.copy_(clipped_grad)


def _add_client_update_noise(model_update, sigma, dp_method, client_id, current_epoch):
    """Add DP noise once to the uploaded client model update."""
    if sigma <= 0:
        return model_update

    if dp_method == "gaussian":
        return [
            update + generate_random_gaussian_noise_like(update) * sigma
            if update is not None else None
            for update in model_update
        ]

    if dp_method == "delayed_feedback":
        total_numel = sum(update.numel() for update in model_update if update is not None)
        if total_numel == 0:
            return model_update

        first_update = next(update for update in model_update if update is not None)
        flat_feedback_noise = generate_delayed_feedback_batch_noise(
            torch.Size([total_numel]),
            client_id,
            current_epoch,
            -1,
            dp_method,
        ).to(first_update.device)
        flat_noise = flat_feedback_noise * sigma
        offset = 0
        noisy_update = []
        for update in model_update:
            if update is None:
                noisy_update.append(None)
                continue
            numel = update.numel()
            update_noise = flat_noise[offset:offset + numel].view_as(update)
            noisy_update.append(update + update_noise)
            offset += numel
        return noisy_update

    return model_update


def generate_delayed_feedback_batch_noise(shape, client_id, epoch, batch_idx, dp_method):
    """
    Improved delayed-feedback noise generation using the paper formula.
    """
    a = getattr(args, "a", 4.0)
    b = getattr(args, "b", 501.0)
    k = getattr(args, "k", 4)
    burn_in = getattr(args, "burn_in", 2048)
    decimation = getattr(args, "decimation", 12)
    q_bits = getattr(args, "q_bits", 32)
    q_mode = getattr(args, "q_mode", "round")
    map_type = getattr(args, "map_type", "logistic")

    seed_value = _stable_u64_seed(
        "delayed_feedback_batch_noise",
        client_id,
        epoch,
        batch_idx,
        int(torch.Size(shape).numel()),
        a,
        b,
        k,
        burn_in,
        decimation,
        q_bits,
        q_mode,
        map_type,
        dp_method,
    )
    x0 = ((seed_value >> 11) + 0.5) / float(1 << 53)

    feedback_noise = generate_delayed_feedback_noise(
        shape,
        a=a,
        b=b,
        x0=x0,
        decimation=decimation,
        burn_in=burn_in,
        k=k,
        q_bits=q_bits,
        q_mode=q_mode,
        map_type=map_type,
    ).to(device)
    return feedback_noise


def adaptive_noise_scale(client_epsilon, delta, clipping_bound, dp_method, current_epoch, total_epochs):
    """
    Compute the adaptive sigma scale used by DP training.
    """
    if client_epsilon <= 0:
        return 0.0

    base_sigma = clipping_bound * np.sqrt(2 * np.log(1.25 / delta)) / client_epsilon
    noise_decay = getattr(args, "noise_decay", 1.0)

    if dp_method == "none":
        sigma_factor = 0.0
    elif dp_method == "gaussian":
        sigma_factor = args.sigma_factor_gaussian
    elif dp_method == "delayed_feedback":
        sigma_factor = args.sigma_factor_delayed_feedback
    else:
        sigma_factor = 0.005

    decay_factor = noise_decay ** max(0, current_epoch)

    sigma = base_sigma * sigma_factor * decay_factor
    sigma = max(0.001, sigma)

    sigma_cap = 0.030 if args.dataset in ("CIFAR10", "SVHN") else 0.020
    if dp_method != "none":
        sigma = min(sigma, sigma_cap)
    return sigma


def local_update_with_dp(
    model,
    dataloader,
    global_model,
    client_data_size,
    total_data_size,
    client_epsilon,
    current_epoch=0,
    dp_method=None,
    client_id=0,
):
    """
    Local update with clipping, DP noise, sparsification, and risk evaluation.
    """
    if dp_method is None:
        dp_method = args.dp_method

    model = model.to(device)
    global_model = global_model.to(device)

    global_params = [param.clone().detach() for param in global_model.parameters()]

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)
    criterion = nn.CrossEntropyLoss()

    model.train()
    epoch_losses = []
    risk_scores_per_batch = []
    risk_details_per_batch = []
    risk_eval_count = 0
    max_risk_evals = max(1, int(getattr(args, "gir_max_evals_per_client_update", 1)))

    risk_cfg = GradientInversionRiskConfig(
        attack_steps=int(getattr(args, "gir_attack_steps", 30)),
        attack_trials=int(getattr(args, "gir_attack_trials", 2)),
        attack_lr=float(getattr(args, "gir_attack_lr", 0.1)),
        attack_batch_size=int(getattr(args, "gir_attack_batch_size", 1)),
        tv_weight=float(getattr(args, "gir_tv_weight", 1e-4)),
        l2_weight=float(getattr(args, "gir_l2_weight", 1e-4)),
        dataset=str(args.dataset),
    )

    try:
        dataloader_len = len(dataloader)
    except Exception:
        dataloader_len = 0

    total_local_batches = int(dataloader_len) * int(args.local_epoch) if dataloader_len > 0 else 0
    explicit_eval_interval = int(getattr(args, "gir_eval_interval", 0))
    if explicit_eval_interval > 0:
        risk_eval_interval = max(1, explicit_eval_interval)
        eval_targets = None
    elif total_local_batches > 0:
        target_count = max(1, min(max_risk_evals, total_local_batches))
        eval_targets = sorted(
            set(
                int(round(idx))
                for idx in np.linspace(0, total_local_batches - 1, num=target_count)
            )
        )
        risk_eval_interval = None
    else:
        eval_targets = [0]
        risk_eval_interval = None

    global_batch_counter = 0

    for local_epoch_idx in range(args.local_epoch):
        batch_losses = []
        for batch_idx, (data, labels) in enumerate(dataloader):
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            _clip_local_gradients_for_stability(
                model,
                float(getattr(args, "train_grad_clip", 0.0)),
            )

            batch_losses.append(loss.item())
            sigma = 0.0
            pending_risk_eval = False

            with torch.no_grad():
                if dp_method != "none":
                    sigma = adaptive_noise_scale(
                        client_epsilon,
                        args.target_delta,
                        args.clipping_bound,
                        dp_method,
                        current_epoch,
                        args.global_epoch,
                    )
                    # Client-level DP is applied once to the uploaded model update below.
                    # Batch-gradient clipping is only available via --train_grad_clip for stability.

                if risk_eval_count < max_risk_evals:
                    if risk_eval_interval is not None:
                        if (global_batch_counter + 1) % risk_eval_interval == 0:
                            pending_risk_eval = True
                    else:
                        if global_batch_counter in eval_targets or global_batch_counter == total_local_batches - 1:
                            pending_risk_eval = True

            if pending_risk_eval and risk_eval_count < max_risk_evals:
                defense_cfg = DefenseSimulationConfig(
                    dp_method=str(dp_method),
                    sigma=float(sigma),
                    clipping_bound=float(args.clipping_bound),
                    apply_clipping=bool(dp_method != "none" and (not args.no_clip)),
                    apply_noise=bool(dp_method != "none" and (not args.no_noise) and sigma > 0),
                    sparsity_ratio=float(args.sparsity_ratio),
                    seed_context=int(
                        _stable_u64_seed(
                            "defense_simulation",
                            args.seed,
                            current_epoch,
                            client_id,
                            local_epoch_idx,
                            batch_idx,
                            dp_method,
                        )
                    ),
                    a=float(getattr(args, "a", 4.0)),
                    b=float(getattr(args, "b", 501.0)),
                    k=int(getattr(args, "k", 4)),
                    burn_in=int(getattr(args, "burn_in", 2048)),
                    decimation=int(getattr(args, "decimation", 12)),
                    q_bits=int(getattr(args, "q_bits", 32)),
                    q_mode=str(getattr(args, "q_mode", "round")),
                    map_type=str(getattr(args, "map_type", "logistic")),
                )
                risk_result = simulate_gradient_inversion_risk(
                    model=model,
                    batch_data=data.detach(),
                    batch_labels=labels.detach(),
                    risk_cfg=risk_cfg,
                    defense_cfg=defense_cfg,
                )
                risk_scores_per_batch.append(
                    float(
                        risk_result.get(
                            "defense_score",
                            1.0 - float(risk_result.get("risk_score", 0.5)),
                        )
                    )
                )
                risk_details_per_batch.append(
                    {
                        "risk_score": float(risk_result.get("risk_score", 0.5)),
                        "defense_score": float(risk_result.get("defense_score", 0.5)),
                        "avg_psnr": float(risk_result.get("avg_psnr", 0.0)),
                        "best_psnr": float(risk_result.get("best_psnr", 0.0)),
                        "avg_mse": float(risk_result.get("avg_mse", 0.0)),
                        "best_mse": float(risk_result.get("best_mse", 0.0)),
                        "avg_ssim": float(risk_result.get("avg_ssim", 0.0)),
                        "best_ssim": float(risk_result.get("best_ssim", 0.0)),
                        "avg_lpips": float(risk_result.get("avg_lpips", 0.0)),
                        "best_lpips": float(risk_result.get("best_lpips", 0.0)),
                        "attack_success_rate": float(risk_result.get("attack_success_rate", 0.0)),
                        "noise_gaussianity": float(risk_result.get("noise_gaussianity", 0.0)),
                        "noise_structure": float(risk_result.get("noise_structure", 0.0)),
                    }
                )
                risk_eval_count += 1

            optimizer.step()
            global_batch_counter += 1

        scheduler.step()
        if batch_losses:
            epoch_losses.append(np.mean(batch_losses))

    if risk_scores_per_batch:
        mean_ability = float(np.mean(risk_scores_per_batch))
        worst_case_ability = float(np.min(risk_scores_per_batch))
        avg_risk = 0.3 * mean_ability + 0.7 * worst_case_ability
    else:
        avg_risk = 0.0 if dp_method == "none" else 0.5

    with torch.no_grad():
        final_params = [param.clone().detach() for param in model.parameters()]
        model_update = [final_param - global_param for final_param, global_param in zip(final_params, global_params)]
        if dp_method != "none":
            update_params = [u for u in model_update if u is not None]
            if update_params and not args.no_clip:
                model_update = _clip_tensors_by_global_norm(model_update, args.clipping_bound)

            if update_params and not args.no_noise:
                update_sigma = adaptive_noise_scale(
                    client_epsilon,
                    args.target_delta,
                    args.clipping_bound,
                    dp_method,
                    current_epoch,
                    args.global_epoch,
                )
                if update_sigma > 0:
                    model_update = _add_client_update_noise(
                        model_update,
                        update_sigma,
                        dp_method,
                        client_id,
                        current_epoch,
                    )

        if args.sparsity_ratio > 0 and dp_method != "none":
            model_update = sparsify_gradients(model_update, sparsity=args.sparsity_ratio)

    model = model.to("cpu")
    avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
    if risk_details_per_batch:
        aggregate_keys = list(risk_details_per_batch[0].keys())
        aggregated_risk_details = {
            key: float(np.mean([item.get(key, 0.0) for item in risk_details_per_batch]))
            for key in aggregate_keys
        }
    else:
        aggregated_risk_details = {}
    local_update_with_dp.last_risk_details = aggregated_risk_details
    return model_update, avg_risk, avg_loss


def test(client_model, client_testloader):
    """Evaluate a client model on its test loader."""
    client_model.eval()
    client_model = client_model.to(device)

    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in client_testloader:
            data, labels = data.to(device), labels.to(device)
            outputs = client_model(data)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100.0 * correct / total if total > 0 else 0.0
    client_model = client_model.to("cpu")
    return accuracy


def check_dataset_exists(data_dir, dataset_name):
    """Check whether the dataset is already present locally."""
    dataset_paths = {
        "MNIST": os.path.join(data_dir, "MNIST"),
        "CIFAR10": os.path.join(data_dir, "CIFAR10"),
        "SVHN": os.path.join(data_dir, "SVHN"),
    }
    if dataset_name in dataset_paths:
        path = dataset_paths[dataset_name]
        return os.path.exists(path) and len(os.listdir(path)) > 0
    return False


def compute_client_importance_weights(client_data_sizes, client_accuracies):
    total_size = sum(client_data_sizes)
    size_weights = [size / total_size for size in client_data_sizes]

    if sum(client_accuracies) == 0:
        return size_weights

    accuracy_weights = [acc / sum(client_accuracies) for acc in client_accuracies]
    balanced_weights = []
    for size_w, acc_w in zip(size_weights, accuracy_weights):
        balanced_w = 0.6 * size_w + 0.4 * acc_w
        balanced_weights.append(balanced_w)

    total_balanced = sum(balanced_weights)
    return [w / total_balanced for w in balanced_weights]


def personalized_local_update(
    model,
    dataloader,
    global_model,
    client_data_size,
    total_data_size,
    client_epsilon,
    client_id,
):
    dp_method = getattr(args, "dp_method", "delayed_feedback")
    model.train()
    model = model.to(device)
    global_model = global_model.to(device)

    global_params = [param.clone().detach() for param in global_model.parameters()]
    fisher_diag = compute_fisher_diag(model, dataloader)

    important_params_mask = []
    personal_params_mask = []
    for fisher_value in fisher_diag:
        important_mask = (fisher_value > args.fisher_threshold).float()
        personal_mask = (fisher_value <= args.fisher_threshold).float()
        important_params_mask.append(important_mask)
        personal_params_mask.append(personal_mask)

    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    adaptive_lr = args.lr * (client_data_size / total_data_size) ** 0.5
    for param_group in optimizer.param_groups:
        param_group["lr"] = adaptive_lr

    num_batches = len(dataloader)
    sampling_rate = args.batch_size / client_data_size
    steps = args.local_epoch * num_batches

    adaptive_epsilon = client_epsilon * (total_data_size / client_data_size) ** 0.2
    adaptive_epsilon = min(adaptive_epsilon, client_epsilon * 2)

    sigma = args.clipping_bound * math.sqrt(2 * math.log(1.25 / args.target_delta)) / adaptive_epsilon

    print(f"client {client_id}: data={client_data_size}, epsilon={adaptive_epsilon:.4f}, lr={adaptive_lr:.6f}")

    for epoch in range(args.local_epoch):
        for batch_idx, (data, labels) in enumerate(dataloader):
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = loss_fn(outputs, labels)
            loss.backward()
            _clip_local_gradients_for_stability(
                model,
                float(getattr(args, "train_grad_clip", 0.0)),
            )

            optimizer.step()

    with torch.no_grad():
        final_params = [param.clone().detach() for param in model.parameters()]
        model_update = [final_param - global_param for final_param, global_param in zip(final_params, global_params)]
        if dp_method != "none":
            if not args.no_clip:
                model_update = _clip_tensors_by_global_norm(model_update, args.clipping_bound)
            if not args.no_noise:
                model_update = _add_client_update_noise(
                    model_update,
                    sigma,
                    dp_method,
                    client_id,
                    current_epoch=0,
                )
        sparse_update = sparsify_gradients(model_update, sparsity=args.sparsity_ratio)

    model = model.to("cpu")
    return sparse_update, fisher_diag


def adaptive_model_aggregation(global_model, clients_updates, clients_weights, clients_fisher_info):
    with torch.no_grad():
        base_aggregated = []
        for param_idx in range(len(clients_updates[0])):
            weighted_updates = []
            for update, weight in zip(clients_updates, clients_weights):
                if update[param_idx] is not None:
                    weighted_updates.append(update[param_idx] * weight)

            if weighted_updates:
                aggregated_param = torch.sum(torch.stack(weighted_updates), dim=0)
                base_aggregated.append(aggregated_param)
            else:
                base_aggregated.append(torch.zeros_like(clients_updates[0][param_idx]))

        if clients_fisher_info and len(clients_fisher_info) == len(clients_updates):
            avg_fisher = []
            for param_idx in range(len(clients_fisher_info[0])):
                fisher_values = [fisher[param_idx] for fisher in clients_fisher_info]
                avg_fisher_param = torch.mean(torch.stack(fisher_values), dim=0)
                avg_fisher.append(avg_fisher_param)

            fisher_aggregated = []
            for param_idx, (base_param, fisher_mask) in enumerate(zip(base_aggregated, avg_fisher)):
                important_weight = 1.0
                personal_weight = 0.8
                adaptive_weight = important_weight * fisher_mask + personal_weight * (1 - fisher_mask)
                adjusted_param = base_param * adaptive_weight
                fisher_aggregated.append(adjusted_param)

            return fisher_aggregated

        return base_aggregated


def analyze_data_distribution(clients_train_loaders, num_classes=10):
    print("\n=== Data Distribution ===")
    client_distributions = []

    for i, train_loader in enumerate(clients_train_loaders):
        if i >= 5:
            break

        class_counts = [0] * num_classes
        total_samples = 0

        for data, labels in train_loader:
            for label in labels:
                class_counts[label.item()] += 1
                total_samples += 1

        if total_samples > 0:
            class_ratios = [count / total_samples for count in class_counts]
            client_distributions.append(class_ratios)
            print(f"client {i}: {[f'{ratio:.2f}' for ratio in class_ratios]}")

    return client_distributions


def test_global_model(global_model, test_loaders):
    global_model.eval()
    global_model = global_model.to(device)

    client_accuracies = []
    for test_loader in test_loaders:
        correct = 0
        total = 0
        with torch.no_grad():
            for data, labels in test_loader:
                data, labels = data.to(device), labels.to(device)
                outputs = global_model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100.0 * correct / total if total > 0 else 0.0
        client_accuracies.append(accuracy)

    global_model = global_model.to("cpu")
    return client_accuracies
