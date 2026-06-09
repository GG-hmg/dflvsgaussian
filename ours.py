import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from options import parse_args
from data import get_mnist_datasets, get_clients_datasets, get_CIFAR10, get_SVHN, get_FashionMNIST
from net import cifar10Net, SVHNNet, fashionmnistNet, mnistNet
from utils import (
    adaptive_privacy_budget,
    generate_dfl_gaussian_noise,
    generate_dfl_uniform_noise,
    generate_random_gaussian_noise,
    generate_random_gaussian_noise_like,
)
from tqdm.auto import trange
from gradient_inversion_risk_simulator import (
    GradientInversionRiskConfig,
    DefenseSimulationConfig,
    simulate_gradient_inversion_risk,
)
import random
import datetime
import torch.nn.functional as F
import numpy as np
import warnings
import math

warnings.filterwarnings('ignore')

# Make Windows GBK console safe for UTF-8 print() output.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

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


set_global_seed(int(args.seed))

num_clients = args.num_clients
local_epoch = args.local_epoch
global_epoch = args.global_epoch
batch_size = args.batch_size
target_epsilon = args.target_epsilon
target_delta = args.target_delta
clipping_bound = args.clipping_bound
dataset = args.dataset
user_sample_rate = args.user_sample_rate
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_losses = {}      # keyed by noise_kind, filled at end of run
train_accuracies = {}

print(f"Noise kind: {args.noise_kind}" + (f" (mix_alpha={args.mix_alpha})" if args.noise_kind.startswith("mix_") else ""))
print(f"Random seed: {args.seed}")
print(f"数据集: {dataset}, 学习率: {args.lr}, 隐私预算ε: {target_epsilon}, 裁剪边界: {clipping_bound}")


def adaptive_noise_scale(client_epsilon, delta, clipping_bound):
    """
    Gaussian-mechanism noise scale: sigma = C * sqrt(2 ln(1.25/delta)) / epsilon.

    Note: only `noise_kind=gaussian` actually has the formal (ε, δ)-DP
    guarantee that this sigma corresponds to. Non-Gaussian / chaotic noise
    kinds reuse the same sigma magnitude for fair comparison but carry no
    formal DP claim.
    """
    if client_epsilon <= 0:
        return 0.0
    return clipping_bound * np.sqrt(2.0 * np.log(1.25 / delta)) / client_epsilon


def _clip_gradients_inplace(model, clipping_bound):
    """Global L2 clip of all parameter grads in-place. Returns the pre-clip norm."""
    grads = [p.grad.view(-1) for p in model.parameters() if p.grad is not None]
    if not grads:
        return 0.0
    grad_norm = torch.norm(torch.cat(grads))
    if grad_norm > clipping_bound:
        scale = clipping_bound / grad_norm
        for param in model.parameters():
            if param.grad is not None:
                param.grad.data.mul_(scale)
    return float(grad_norm.item())


def _x0_from_seed(client_id, current_epoch, batch_seed, salt=0):
    """Map (client_id, epoch, batch_seed) → x0 ∈ [0, 1) via Knuth multiplicative hash.

    `salt` lets us produce two independent x0's from the same triple (used for
    the second DFL source in `mix_dgauss_dchaos`).
    """
    seed_val = client_id * 1000000 + current_epoch * 10000 + batch_seed + salt * 7919
    mult = 2654435761 if salt == 0 else 2246822519  # both Knuth/Marsaglia magic constants
    return ((seed_val * mult) & 0xFFFFFFFF) / 4294967296.0


def _build_unit_noise(numel, noise_kind, mix_alpha,
                      client_id, current_epoch, batch_seed):
    """
    Build a unit-variance flat noise tensor of length `numel`.

    Dispatches on `noise_kind` (see options.py for the 5 choices). The output
    is mean ≈ 0, var ≈ 1; the *shape* of the marginal distribution and the
    sample correlations depend on the kind:
        gaussian          → iid N(0,1)
        dfl_uniform       → uniform-shape on [-√3, √3), chaotic correlation
        dfl_gaussian      → N(0,1)-shape, chaotic correlation
        mix_dgauss_gauss  → √α·dfl_gaussian + √(1-α)·gaussian   (variance-preserving)
        mix_dgauss_dchaos → √α·dfl_gaussian + √(1-α)·dfl_uniform (independent x0)
    """
    shape = torch.Size([numel])

    if noise_kind == 'gaussian':
        return generate_random_gaussian_noise(shape, device=device)

    dfl_kw = dict(
        a=float(args.dfl_a), b=float(args.dfl_b), k=int(args.dfl_k),
        burn_in=int(args.dfl_burn_in), decimation=int(args.dfl_decimation),
        device=device,
    )
    x0_a = _x0_from_seed(client_id, current_epoch, batch_seed, salt=0)

    if noise_kind == 'dfl_uniform':
        return generate_dfl_uniform_noise(shape, x0=x0_a, **dfl_kw)

    if noise_kind == 'dfl_gaussian':
        return generate_dfl_gaussian_noise(shape, x0=x0_a, **dfl_kw)

    alpha = max(0.0, min(1.0, float(mix_alpha)))
    a_term = generate_dfl_gaussian_noise(shape, x0=x0_a, **dfl_kw)

    if noise_kind == 'mix_dgauss_gauss':
        b_term = generate_random_gaussian_noise(shape, device=device)
    elif noise_kind == 'mix_dgauss_dchaos':
        x0_b = _x0_from_seed(client_id, current_epoch, batch_seed, salt=1)
        b_term = generate_dfl_uniform_noise(shape, x0=x0_b, **dfl_kw)
    else:
        raise ValueError(f"Unknown noise_kind: {noise_kind!r}")

    return math.sqrt(alpha) * a_term + math.sqrt(1.0 - alpha) * b_term


def _add_dp_noise_inplace(model, sigma, noise_kind, mix_alpha,
                          client_id, current_epoch, batch_seed):
    """Inject `noise_kind`-shaped unit-variance noise into param.grad in-place, scaled by sigma."""
    grad_params = [p for p in model.parameters() if p.grad is not None]
    if not grad_params or sigma <= 0:
        return

    total_numel = sum(p.grad.numel() for p in grad_params)
    flat_unit = _build_unit_noise(total_numel, noise_kind, mix_alpha,
                                  client_id, current_epoch, batch_seed)
    flat_noise = flat_unit * sigma

    offset = 0
    for param in grad_params:
        n = param.grad.numel()
        param.grad.data.add_(flat_noise[offset:offset + n].view_as(param.grad))
        offset += n


def _apply_sparsity_inplace(model, sparsity_ratio):
    """Zero out (1 - sparsity_ratio) of smallest-magnitude grad entries per parameter."""
    if sparsity_ratio <= 0:
        return
    for param in model.parameters():
        if param.grad is None:
            continue
        grad_flat = param.grad.view(-1)
        k = int((1 - sparsity_ratio) * grad_flat.numel())
        if 0 < k < grad_flat.numel():
            _, indices = torch.topk(grad_flat.abs(), k)
            mask = torch.zeros_like(grad_flat)
            mask[indices] = 1
            param.grad.data.mul_(mask.view(param.grad.shape))


def _snapshot_grads(model):
    """Return a detached clone of every parameter's grad (in parameter order)."""
    return [p.grad.detach().clone() for p in model.parameters() if p.grad is not None]


def _run_gir_evaluation(model, data, labels, real_noisy_grads,
                        sigma, risk_cfg, current_epoch, client_id,
                        local_epoch_idx, batch_idx):
    """
    Evaluate gradient-inversion defense quality using the ACTUAL clipped+noised
    gradients (real_noisy_grads) that training transmitted, not a freshly
    resampled noise stream inside the simulator. The simulator is now
    noise-kind agnostic — `real_noisy_grads` carries everything the attacker
    sees, so DefenseSimulationConfig only needs sigma / clipping / sparsity.
    """
    defense_cfg = DefenseSimulationConfig(
        sigma=float(sigma),
        clipping_bound=float(clipping_bound),
        apply_clipping=bool(not args.no_clip),
        apply_noise=bool((not args.no_noise) and sigma > 0),
        sparsity_ratio=float(args.sparsity_ratio),
        seed_context=int(
            args.seed
            + current_epoch * 100000
            + client_id * 1000
            + local_epoch_idx * 100
            + batch_idx
        ),
    )
    risk_result = simulate_gradient_inversion_risk(
        model=model,
        batch_data=data.detach(),
        batch_labels=labels.detach(),
        risk_cfg=risk_cfg,
        defense_cfg=defense_cfg,
        real_noisy_grads=real_noisy_grads,
    )
    return float(
        risk_result.get(
            "defense_score",
            1.0 - float(risk_result.get("risk_score", 0.5))
        )
    )


def local_update_with_dp(model, dataloader, global_model, client_data_size,
                         total_data_size, client_epsilon,
                         current_epoch=0, client_id=0):
    """
    Client-side training with DP. Each batch:
        forward → loss → backward → (clip → noise) → snapshot → GIR eval
        → sparsity → optimizer.step
    Returns (model_update, avg_defense_score, avg_loss).
    Noise kind is taken from `args.noise_kind`; mix ratio from `args.mix_alpha`.
    """
    noise_kind = args.noise_kind
    mix_alpha = float(args.mix_alpha)

    model = model.to(device)
    global_model = global_model.to(device)

    global_params = [param.clone().detach() for param in global_model.parameters()]
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001)
    criterion = nn.CrossEntropyLoss()
    model.train()

    risk_cfg = GradientInversionRiskConfig(
        attack_steps=int(getattr(args, 'gir_attack_steps', 30)),
        attack_trials=int(getattr(args, 'gir_attack_trials', 2)),
        attack_lr=float(getattr(args, 'gir_attack_lr', 0.1)),
        attack_batch_size=int(getattr(args, 'gir_attack_batch_size', 1)),
        tv_weight=float(getattr(args, 'gir_tv_weight', 1e-4)),
        l2_weight=float(getattr(args, 'gir_l2_weight', 1e-4)),
        dataset=str(dataset),
    )

    try:
        dataloader_len = len(dataloader)
    except Exception:
        dataloader_len = 0

    if int(getattr(args, 'gir_eval_interval', 0)) > 0:
        risk_eval_interval = max(1, int(args.gir_eval_interval))
    else:
        risk_eval_interval = max(1, dataloader_len) if dataloader_len > 0 else 1

    max_risk_evals = max(1, int(getattr(args, 'gir_max_evals_per_client_update', 1)))
    epoch_losses = []
    risk_scores_per_batch = []
    risk_eval_count = 0

    for local_epoch_idx in range(local_epoch):
        batch_losses = []
        for batch_idx, (data, labels) in enumerate(dataloader):
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            batch_losses.append(loss.item())

            sigma = 0.0
            with torch.no_grad():
                sigma = adaptive_noise_scale(client_epsilon, target_delta, clipping_bound)
                if not args.no_clip:
                    _clip_gradients_inplace(model, clipping_bound)
                if not args.no_noise:
                    batch_seed = batch_idx + local_epoch_idx * max(1, dataloader_len)
                    _add_dp_noise_inplace(model, sigma, noise_kind, mix_alpha,
                                          client_id, current_epoch, batch_seed)

            # Snapshot the clipped+noised gradients BEFORE sparsity / optimizer step.
            # This is what GIR should attack — the actual gradient the attacker observes.
            should_eval = (
                risk_eval_count < max_risk_evals
                and ((batch_idx + 1) % risk_eval_interval == 0 or batch_idx == len(dataloader) - 1)
            )
            if should_eval:
                real_noisy_grads = _snapshot_grads(model)
                score = _run_gir_evaluation(
                    model, data, labels, real_noisy_grads,
                    sigma, risk_cfg,
                    current_epoch, client_id, local_epoch_idx, batch_idx,
                )
                risk_scores_per_batch.append(score)
                risk_eval_count += 1

            _apply_sparsity_inplace(model, args.sparsity_ratio)
            optimizer.step()

        if batch_losses:
            epoch_losses.append(np.mean(batch_losses))

    avg_risk = float(np.mean(risk_scores_per_batch)) if risk_scores_per_batch else 0.5

    with torch.no_grad():
        final_params = [param.clone().detach() for param in model.parameters()]
        model_update = [fp - gp for fp, gp in zip(final_params, global_params)]

    model = model.to('cpu')
    avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
    return model_update, avg_risk, avg_loss


def test(client_model, client_testloader):
    """在客户端测试集上评估模型"""
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
    client_model = client_model.to('cpu')
    return accuracy


def check_dataset_exists(data_dir, dataset_name):
    """检查数据集文件是否已存在于本地"""
    dataset_paths = {
        'MNIST': os.path.join(data_dir, 'MNIST'),
        'CIFAR10': os.path.join(data_dir, 'CIFAR10'),
        'SVHN': os.path.join(data_dir, 'SVHN')
    }
    if dataset_name in dataset_paths:
        path = dataset_paths[dataset_name]
        return os.path.exists(path) and len(os.listdir(path)) > 0
    return False


def main():
    current_num_clients = num_clients
    mean_acc_s = []
    acc_matrix = []
    risk_scores = []
    epoch_losses = []

    print(f"正在加载 {dataset} 数据集...")
    data_dir = './data'
    os.makedirs(data_dir, exist_ok=True)

    if check_dataset_exists(data_dir, dataset):
        print(f"检测到本地{dataset}数据集，直接使用...")
        download_flag = False
    else:
        print(f"数据集 {dataset} 未在本地找到，正在下载...")
        download_flag = True

    try:
        if dataset == 'MNIST':
            print("使用MNIST数据集...")
            train_dataset, test_dataset = get_mnist_datasets()
            clients_train_set = get_clients_datasets(train_dataset, current_num_clients)
            client_data_sizes = [len(client_dataset) for client_dataset in clients_train_set]
            clients_train_loaders = [DataLoader(client_dataset, batch_size=batch_size, shuffle=True)
                                     for client_dataset in clients_train_set]
            clients_test_loaders = [DataLoader(test_dataset, batch_size=256) for i in range(current_num_clients)]
            clients_models = [mnistNet() for _ in range(current_num_clients)]
            global_model = mnistNet()
            print(f"使用原始数据加载方法成功: {len(clients_train_loaders)}个客户端")

        elif dataset == 'CIFAR10':
            print("使用CIFAR10数据集...")
            clients_train_loaders, clients_test_loaders, client_data_sizes = get_CIFAR10(
                args.dir_alpha, current_num_clients, batch_size=args.batch_size)
            clients_models = [cifar10Net() for _ in range(current_num_clients)]
            global_model = cifar10Net()
            print(f"使用原始数据加载方法成功: {len(clients_train_loaders)}个客户端")

        elif dataset == 'SVHN':
            print("使用SVHN数据集...")
            clients_train_loaders, clients_test_loaders, client_data_sizes = get_SVHN(
                args.dir_alpha, current_num_clients, batch_size=args.batch_size)
            clients_models = [SVHNNet() for _ in range(current_num_clients)]
            global_model = SVHNNet()
            print(f"使用原始数据加载方法成功: {len(clients_train_loaders)}个客户端")

        elif dataset == 'FashionMNIST':
            print("使用FashionMNIST数据集...")
            clients_train_loaders, clients_test_loaders, client_data_sizes = get_FashionMNIST(
                args.dir_alpha, current_num_clients, batch_size=args.batch_size)
            clients_models = [fashionmnistNet() for _ in range(current_num_clients)]
            global_model = fashionmnistNet()
            print(f"使用原始数据加载方法成功: {len(clients_train_loaders)}个客户端")

        else:
            print('不支持的数据集名称。')
            return

        print(f"数据加载成功: {len(clients_train_loaders)}个客户端，总数据量: {sum(client_data_sizes)}")

    except Exception as e:
        print(f"数据加载失败: {str(e)}")
        return

    if 'clients_train_loaders' not in locals() or len(clients_train_loaders) == 0:
        print("错误: 未加载任何数据")
        return

    actual_num_clients = len(clients_train_loaders)
    if actual_num_clients < current_num_clients:
        print(f"警告: 实际客户端数量({actual_num_clients})小于请求数量({current_num_clients})，调整客户端数量")
        current_num_clients = actual_num_clients

    if 'clients_test_loaders' not in locals():
        clients_test_loaders = [clients_train_loaders[i] for i in range(current_num_clients)]

    clients_models = clients_models[:current_num_clients]
    clients_test_loaders = clients_test_loaders[:current_num_clients]
    client_data_sizes = client_data_sizes[:current_num_clients]

    total_data_size = sum(client_data_sizes)
    client_privacy_budgets = adaptive_privacy_budget(client_data_sizes, target_epsilon)

    print(f"客户端隐私预算分配: {[f'{eps:.2f}' for eps in client_privacy_budgets]}")
    print(f"客户端数据量: {client_data_sizes}")

    for client_model in clients_models:
        client_model.load_state_dict(global_model.state_dict())

    start_time = datetime.datetime.now()

    for epoch in trange(global_epoch, desc=f"全局轮次 {args.noise_kind}"):
        num_total_clients = len(clients_train_loaders)
        client_num_per_round = max(1, min(current_num_clients, int(user_sample_rate * num_total_clients)))
        available_indices = list(range(num_total_clients))
        sampled_client_indices = random.sample(available_indices, client_num_per_round)

        sampled_clients_models = [clients_models[i] for i in sampled_client_indices]
        sampled_clients_train_loaders = [clients_train_loaders[i] for i in sampled_client_indices]
        sampled_clients_test_loaders = [clients_test_loaders[i] for i in sampled_client_indices]
        sampled_client_data_sizes = [client_data_sizes[i] for i in sampled_client_indices]
        sampled_client_privacy_budgets = [client_privacy_budgets[i] for i in sampled_client_indices]

        clients_updates = []
        clients_accuracies = []
        clients_risks = []
        clients_losses = []

        for idx, (client_model, client_trainloader, client_testloader, client_data_size, client_epsilon) in enumerate(
                zip(sampled_clients_models, sampled_clients_train_loaders, sampled_clients_test_loaders,
                    sampled_client_data_sizes, sampled_client_privacy_budgets)):

            update, risk_score, avg_loss = local_update_with_dp(
                client_model,
                client_trainloader,
                global_model,
                client_data_size,
                total_data_size,
                client_epsilon,
                current_epoch=epoch,
                client_id=idx
            )

            clients_updates.append(update)
            clients_risks.append(risk_score)
            clients_losses.append(avg_loss)

            accuracy = test(client_model, client_testloader)
            clients_accuracies.append(accuracy)

        print(f"轮次 {epoch + 1} 客户端准确率: {[f'{acc:.2f}%' for acc in clients_accuracies]}")
        mean_acc_s.append(sum(clients_accuracies) / len(clients_accuracies))
        acc_matrix.append(clients_accuracies)

        if clients_losses:
            avg_loss = sum(clients_losses) / len(clients_losses)
            epoch_losses.append(avg_loss)
            print(f"轮次 {epoch + 1} 平均训练损失: {avg_loss:.4f}")

        if clients_risks:
            avg_risk = sum(clients_risks) / len(clients_risks)
            risk_scores.append(avg_risk)
            print(f"轮次 {epoch + 1} 平均抗梯度反演能力: {avg_risk:.4f}")

        sampled_client_weights = [
            client_data_size / sum(sampled_client_data_sizes)
            for client_data_size in sampled_client_data_sizes
        ]

        aggregated_update = []
        if len(clients_updates) > 0 and len(clients_updates[0]) > 0:
            for param_index in range(len(clients_updates[0])):
                param_updates = []
                valid_weights = []

                for idx, update in enumerate(clients_updates):
                    if update[param_index] is not None:
                        weighted_update = update[param_index] * sampled_client_weights[idx]
                        param_updates.append(weighted_update)
                        valid_weights.append(sampled_client_weights[idx])

                if param_updates:
                    aggregated_param = torch.sum(torch.stack(param_updates), dim=0)
                    if sum(valid_weights) > 0:
                        aggregated_param = aggregated_param / sum(valid_weights)
                    aggregated_update.append(aggregated_param)
                else:
                    aggregated_update.append(torch.zeros_like(clients_updates[0][param_index]))
        else:
            for param in global_model.parameters():
                aggregated_update.append(torch.zeros_like(param))

        with torch.no_grad():
            for global_param, update in zip(global_model.parameters(), aggregated_update):
                global_param.add_(update)

        for client_model in clients_models:
            client_model.load_state_dict(global_model.state_dict())

    end_time = datetime.datetime.now()
    training_time = (end_time - start_time).total_seconds()

    train_losses[args.noise_kind] = epoch_losses
    train_accuracies[args.noise_kind] = mean_acc_s

    print("\n" + "=" * 80)
    print(f"{args.noise_kind} 噪声机制 - 实验结果汇总")
    print("=" * 80)
    print(f"数据集: {dataset}")
    print(f"全局训练轮次: {global_epoch}")
    print(f"训练时间: {training_time:.1f}秒")

    if mean_acc_s:
        acc_str = [f"{acc:.2f}%" for acc in mean_acc_s]
        print(f"平均准确率历程: {acc_str}")
    if epoch_losses:
        loss_str = [f"{loss:.4f}" for loss in epoch_losses]
        print(f"平均损失历程: {loss_str}")
    if risk_scores:
        risk_str = [f"{risk:.4f}" for risk in risk_scores]
        print(f"抗梯度反演能力历程: {risk_str}")

    if mean_acc_s:
        print("\n准确率变化:")
        for i, acc in enumerate(mean_acc_s, 1):
            print(f"轮次 {i:2d}: {acc:.2f}%")

    if mean_acc_s:
        initial_acc = mean_acc_s[0]
        final_acc = mean_acc_s[-1]
        improvement = final_acc - initial_acc
        rel_improvement = (improvement / initial_acc * 100) if initial_acc > 0 else 0
        print("\n总结:")
        print(f"初始准确率: {initial_acc:.2f}%")
        print(f"最终准确率: {final_acc:.2f}%")
        print(f"性能提升: {improvement:+.2f}% ({rel_improvement:+.1f}%)")

    if epoch_losses:
        initial_loss = epoch_losses[0]
        final_loss = epoch_losses[-1]
        loss_improvement = ((initial_loss - final_loss) / initial_loss * 100) if initial_loss > 0 else 0
        print(f"\n损失变化:")
        print(f"初始损失: {initial_loss:.4f}")
        print(f"最终损失: {final_loss:.4f}")
        print(f"损失改善: {loss_improvement:.1f}%")

    if training_time > 0:
        avg_epoch_time = training_time / global_epoch
        print(f"\n效率分析:")
        print(f"总训练时间: {training_time:.1f}秒")
        print(f"平均每轮时间: {avg_epoch_time:.1f}秒")

    if risk_scores:
        final_ability = risk_scores[-1]
        # 隐私保护带来的准确率损失（相对于无DP基线，这里简单设为0，实际可使用外部传入的基线）
        # 为了格式统一，保留该行，数值留空或设为0
        privacy_loss = 0.0  # 实际应用中可从文件读取none的结果
        privacy_loss_percent = 0.0
        print(f"隐私保护带来的准确率损失: {privacy_loss:+.2f}% ({privacy_loss_percent:+.1f}%)")
        # 隐私-性能权衡比（简单计算）
        tradeoff = (1 - final_acc/100) * (1.0 - final_ability) if final_acc > 0 else 0
        print(f"隐私-性能权衡比: {tradeoff:.4f} (越低越好)")

        print(f"\n隐私保护评估:")
        print(f"目标隐私预算ε: {target_epsilon}")
        print(f"最终抗梯度反演能力: {final_ability:.4f}")
        if final_ability >= 0.7:
            protection = "优秀 (强抗梯度反演能力)"
        elif final_ability >= 0.4:
            protection = "良好 (中等抗梯度反演能力)"
        elif final_ability > 0.0:
            protection = "一般 (较弱抗梯度反演能力)"
        else:
            protection = "较差 (几乎无防护)"
        print(f"隐私保护效果: {protection}")

    print("=" * 80)


if __name__ == '__main__':
    main()

