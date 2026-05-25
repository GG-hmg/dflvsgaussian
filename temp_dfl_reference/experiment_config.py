import copy


DEFAULT_DP_METHODS = ["none", "gaussian", "delayed_feedback"]
DEFAULT_DATASETS = ["CIFAR10", "MNIST", "SVHN"]
DEFAULT_GLOBAL_EPOCHS = {"CIFAR10": 40, "MNIST": 30, "SVHN": 40}
DEFAULT_LOCAL_EPOCHS = {"CIFAR10": 4, "MNIST": 3, "SVHN": 4}
DEFAULT_GIR_COMMON = {
    "gir_attack_steps": "24",
    "gir_attack_trials": "2",
    "gir_attack_batch_size": "1",
    "gir_attack_lr": "0.1",
    "gir_eval_interval": "0",
    "gir_max_evals_per_client_update": "2",
}


DEFAULT_DATASET_PARAMS = {
    "CIFAR10": {
        "none": {"num_clients": "3", "batch_size": "32", "lr": "0.01", "target_epsilon": "8.0", "clipping_bound": "4.0", "local_epoch": "4", "noise_decay": "1.0", "sparsity_ratio": "0.0"},
        "gaussian": {"num_clients": "3", "batch_size": "32", "lr": "0.01", "target_epsilon": "8.0", "sigma_factor_gaussian": "0.005", "clipping_bound": "4.0", "local_epoch": "4", "noise_decay": "1.0", "sparsity_ratio": "0.0"},
        "delayed_feedback": {"num_clients": "3", "batch_size": "32", "lr": "0.01", "target_epsilon": "8.0", "sigma_factor_gaussian": "0.005", "sigma_factor_delayed_feedback": "0.005", "clipping_bound": "4.0", "local_epoch": "4", "noise_decay": "1.0", "a": "4.0", "b": "501.0", "k": "4", "burn_in": "2048", "decimation": "12", "q_bits": "32", "q_mode": "round", "map_type": "logistic", "sparsity_ratio": "0.0"},
    },
    "MNIST": {
        "none": {"num_clients": "3", "batch_size": "64", "lr": "0.01", "target_epsilon": "8.0", "clipping_bound": "4.0", "local_epoch": "3", "noise_decay": "1.0", "sparsity_ratio": "0.0"},
        "gaussian": {"num_clients": "3", "batch_size": "64", "lr": "0.01", "target_epsilon": "8.0", "sigma_factor_gaussian": "0.005", "clipping_bound": "4.0", "local_epoch": "3", "noise_decay": "1.0", "sparsity_ratio": "0.0"},
        "delayed_feedback": {"num_clients": "3", "batch_size": "64", "lr": "0.01", "target_epsilon": "8.0", "sigma_factor_gaussian": "0.005", "sigma_factor_delayed_feedback": "0.005", "clipping_bound": "4.0", "local_epoch": "3", "noise_decay": "1.0", "a": "4.0", "b": "501.0", "k": "4", "burn_in": "2048", "decimation": "12", "q_bits": "32", "q_mode": "round", "map_type": "logistic", "sparsity_ratio": "0.0"},
    },
    "SVHN": {
        "none": {"num_clients": "3", "batch_size": "32", "lr": "0.01", "target_epsilon": "8.0", "clipping_bound": "4.0", "local_epoch": "4", "noise_decay": "1.0", "sparsity_ratio": "0.0"},
        "gaussian": {"num_clients": "3", "batch_size": "32", "lr": "0.01", "target_epsilon": "8.0", "sigma_factor_gaussian": "0.005", "clipping_bound": "4.0", "local_epoch": "4", "noise_decay": "1.0", "sparsity_ratio": "0.0"},
        "delayed_feedback": {"num_clients": "3", "batch_size": "32", "lr": "0.01", "target_epsilon": "8.0", "sigma_factor_gaussian": "0.005", "sigma_factor_delayed_feedback": "0.005", "clipping_bound": "4.0", "local_epoch": "4", "noise_decay": "1.0", "a": "4.0", "b": "501.0", "k": "4", "burn_in": "2048", "decimation": "12", "q_bits": "32", "q_mode": "round", "map_type": "logistic", "sparsity_ratio": "0.0"},
    },
}

DEFAULT_ABLATION_SWEEPS = {
    "decimation": [1, 2, 4, 8],
}


def clone_dataset_params():
    return copy.deepcopy(DEFAULT_DATASET_PARAMS)


def clone_params(params):
    return {k: str(v) for k, v in params.items()}


def get_base_params(dataset_params, dataset, dp_method):
    fallback = dataset_params.get("CIFAR10", {})
    return clone_params(dataset_params.get(dataset, fallback)[dp_method])


def format_value_for_label(value):
    if isinstance(value, float):
        text = f"{value:.4f}".rstrip("0").rstrip(".")
    else:
        text = str(value)
    return text.replace(".", "p")


def display_method_name(method: str) -> str:
    if method == "gaussian":
        return "Pure Gaussian"
    if method == "delayed_feedback":
        return "Delayed Feedback"
    if method == "none":
        return "No Noise"
    if method.startswith("delayed_feedback_decimation_"):
        return f"Delayed Feedback, {method.replace('delayed_feedback_decimation_', 'decimation=')}"
    return method


def build_default_specs(dataset_params, dataset: str):
    return [
        {"label": "none", "dp_method": "none", "params": get_base_params(dataset_params, dataset, "none")},
        {"label": "gaussian", "dp_method": "gaussian", "params": get_base_params(dataset_params, dataset, "gaussian")},
        {"label": "delayed_feedback", "dp_method": "delayed_feedback", "params": get_base_params(dataset_params, dataset, "delayed_feedback")},
    ]


def build_ablation_specs(dataset_params, ablation_sweeps, dataset: str, target: str):
    base = get_base_params(dataset_params, dataset, "delayed_feedback")
    specs = [
        {"label": "none", "dp_method": "none", "params": get_base_params(dataset_params, dataset, "none")},
        {"label": "gaussian", "dp_method": "gaussian", "params": get_base_params(dataset_params, dataset, "gaussian")},
    ]

    targets = ["decimation"] if target == "all" else [target]

    for sweep_name in targets:
        for value in ablation_sweeps[sweep_name]:
            params = dict(base)
            if sweep_name == "decimation":
                params["decimation"] = str(value)
                label = f"delayed_feedback_decimation_{value}"
            else:
                label = f"delayed_feedback_{sweep_name}_{format_value_for_label(value)}"
                params[sweep_name] = str(value)

            specs.append(
                {
                    "label": label,
                    "dp_method": "delayed_feedback",
                    "params": params,
                }
            )

    return specs


def sync_feedback_sigma_with_gaussian_if_needed(dataset_params, dp_methods):
    methods = set(dp_methods or [])
    if "gaussian" not in methods or "delayed_feedback" not in methods:
        return

    for dataset, method_params in dataset_params.items():
        if "gaussian" not in method_params or "delayed_feedback" not in method_params:
            continue
        gaussian_sigma = method_params["gaussian"].get("sigma_factor_gaussian")
        if gaussian_sigma is not None:
            method_params["delayed_feedback"]["sigma_factor_delayed_feedback"] = str(gaussian_sigma)
