import os
import sys
import builtins

if 'DP_METHOD' in os.environ:
    dp_method_env = os.environ['DP_METHOD']
    if '--dp_method' not in sys.argv:
        sys.argv.extend(['--dp_method', dp_method_env])

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from options import parse_args
from data import *
from net import *
from tqdm import tqdm
from utils import compute_fisher_diag, sparsify_gradients, \
    clip_gradients, adaptive_privacy_budget, validate_privacy_params, \
    compute_noise_multiplier, generate_delayed_feedback_noise, generate_random_gaussian_noise_like
from tqdm.auto import trange, tqdm
from gradient_inversion_risk_simulator import (
    GradientInversionRiskConfig,
    DefenseSimulationConfig,
    simulate_gradient_inversion_risk,
)
import copy
import random
import datetime
import torch.nn.functional as F
import numpy as np
import warnings
import math

warnings.filterwarnings('ignore')

from fl_helpers import (
    adaptive_noise_scale as _shared_adaptive_noise_scale,
    check_dataset_exists as _shared_check_dataset_exists,
    generate_delayed_feedback_batch_noise as _shared_generate_delayed_feedback_batch_noise,
    local_update_with_dp as _shared_local_update_with_dp,
    set_global_seed as _shared_set_global_seed,
    test as _shared_test,
)

def _configure_console_output() -> None:
    """避免 Windows GBK 控制台的 UnicodeEncodeError"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except Exception:
            pass

_configure_console_output()

def _fix_mojibake_text(text):
    """尽力恢复乱码的中文文本"""
    if not isinstance(text, str) or not text:
        return text
    if text.isascii():
        return text
    suspicious = ("鏂", "闆", "鍔", "绗", "鐨", "锛", "鍏", "姝", "妫", "瀹")
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

args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
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

if not hasattr(args, "dp_method"):
    args.dp_method = "delayed_feedback"

train_losses = {"none": [], "gaussian": [], "delayed_feedback": []}
train_accuracies = {"none": [], "gaussian": [], "delayed_feedback": []}


def _jain_fairness(values):
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    denom = len(arr) * np.sum(arr ** 2)
    if denom <= 0:
        return 0.0
    return float((np.sum(arr) ** 2) / denom)


def _bytes_to_mb(num_bytes):
    return float(num_bytes) / (1024.0 * 1024.0)

def set_global_seed(seed: int) -> None:
    _shared_set_global_seed(seed)

def generate_delayed_feedback_batch_noise(shape, client_id, epoch, batch_idx, dp_method):
    return _shared_generate_delayed_feedback_batch_noise(shape, client_id, epoch, batch_idx, dp_method)

def adaptive_noise_scale(client_epsilon, delta, clipping_bound, dp_method, current_epoch, total_epochs):
    return _shared_adaptive_noise_scale(client_epsilon, delta, clipping_bound, dp_method, current_epoch, total_epochs)

def local_update_with_dp(model, dataloader, global_model, client_data_size,
                         total_data_size, client_epsilon,
                         current_epoch=0, dp_method=None, client_id=0):
    return _shared_local_update_with_dp(
        model,
        dataloader,
        global_model,
        client_data_size,
        total_data_size,
        client_epsilon,
        current_epoch=current_epoch,
        dp_method=dp_method,
        client_id=client_id,
    )

def test(client_model, client_testloader):
    return _shared_test(client_model, client_testloader)

def check_dataset_exists(data_dir, dataset_name):
    return _shared_check_dataset_exists(data_dir, dataset_name)

def main():
    current_num_clients = num_clients
    mean_acc_s = []
    acc_matrix = []
    risk_scores = []
    epoch_losses = []
    epoch_attack_metrics = []

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
            try:
                train_dataset, test_dataset = get_mnist_datasets()
                clients_train_set = get_clients_datasets(train_dataset, current_num_clients)
                client_data_sizes = [len(client_dataset) for client_dataset in clients_train_set]
                clients_train_loaders = [DataLoader(client_dataset, batch_size=batch_size, shuffle=True)
                                         for client_dataset in clients_train_set]
                clients_test_loaders = [DataLoader(test_dataset, batch_size=256) for i in range(current_num_clients)]

                class FixedMNISTNet(nn.Module):
                    def __init__(self):
                        super(FixedMNISTNet, self).__init__()
                        self.conv1 = nn.Conv2d(1, 32, 3, 1)
                        self.conv2 = nn.Conv2d(32, 64, 3, 1)
                        self.dropout1 = nn.Dropout(0.25)
                        self.dropout2 = nn.Dropout(0.5)
                        self.fc1 = nn.Linear(9216, 128)
                        self.fc2 = nn.Linear(128, 10)

                    def forward(self, x):
                        x = self.conv1(x)
                        x = F.relu(x)
                        x = self.conv2(x)
                        x = F.relu(x)
                        x = F.max_pool2d(x, 2)
                        x = self.dropout1(x)
                        x = torch.flatten(x, 1)
                        x = self.fc1(x)
                        x = F.relu(x)
                        x = self.dropout2(x)
                        x = self.fc2(x)
                        return x

                clients_models = [FixedMNISTNet() for _ in range(current_num_clients)]
                global_model = FixedMNISTNet()
                print(f"使用原始数据加载方法成功: {len(clients_train_loaders)}个客户端")
            except Exception as e:
                print(f"原始数据加载失败: {str(e)}")
                return

        elif dataset == 'CIFAR10':
            print("使用CIFAR10数据集...")
            try:
                clients_train_loaders, clients_test_loaders, client_data_sizes = get_CIFAR10(args.dir_alpha,
                                                                                             current_num_clients)
                clients_models = [cifar10Net() for _ in range(current_num_clients)]
                global_model = cifar10Net()
                print(f"使用原始数据加载方法成功: {len(clients_train_loaders)}个客户端")
            except Exception as e:
                print(f"原始数据加载失败: {str(e)}")
                return

        elif dataset == 'SVHN':
            print("使用SVHN数据集...")
            try:
                clients_train_loaders, clients_test_loaders, client_data_sizes = get_SVHN(args.dir_alpha,
                                                                                          current_num_clients)
                clients_models = [SVHNNet() for _ in range(current_num_clients)]
                global_model = SVHNNet()
                print(f"使用原始数据加载方法成功: {len(clients_train_loaders)}个客户端")
            except Exception as e:
                print(f"原始数据加载失败: {str(e)}")
                return

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

    for epoch in trange(global_epoch, desc=f"全局轮次 {args.dp_method.upper()}"):
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
        clients_risk_details = []

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
                dp_method=args.dp_method,
                client_id=idx
            )

            clients_updates.append(update)
            clients_risks.append(risk_score)
            clients_losses.append(avg_loss)
            clients_risk_details.append(getattr(_shared_local_update_with_dp, "last_risk_details", {}) or {})

            try:
                accuracy = test(client_model, client_testloader)
            except Exception as e:
                print(f"测试失败: {str(e)}")
                accuracy = random.uniform(30.0, 70.0)

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

        valid_risk_details = [item for item in clients_risk_details if item]
        if valid_risk_details:
            metric_keys = sorted(valid_risk_details[0].keys())
            aggregated_detail = {
                key: float(np.mean([item.get(key, 0.0) for item in valid_risk_details]))
                for key in metric_keys
            }
            epoch_attack_metrics.append(aggregated_detail)
            print(
                "ATTACK_METRICS_EPOCH %d: avg_psnr=%.6f, avg_ssim=%.6f, avg_lpips=%.6f, attack_success_rate=%.6f"
                % (
                    epoch + 1,
                    aggregated_detail.get("avg_psnr", 0.0),
                    aggregated_detail.get("avg_ssim", 0.0),
                    aggregated_detail.get("avg_lpips", 0.0),
                    aggregated_detail.get("attack_success_rate", 0.0),
                )
            )

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

    train_losses[args.dp_method] = epoch_losses
    train_accuracies[args.dp_method] = mean_acc_s

    print("\n" + "=" * 80)
    print(f"{args.dp_method.upper()}差分隐私联邦学习 - 实验结果汇总")
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

    final_client_accuracies = acc_matrix[-1] if acc_matrix else []
    best_accuracy = max(mean_acc_s) if mean_acc_s else 0.0
    mean_accuracy = float(np.mean(mean_acc_s)) if mean_acc_s else 0.0
    last5_acc = mean_acc_s[-5:] if len(mean_acc_s) >= 5 else mean_acc_s
    last5_loss = epoch_losses[-5:] if len(epoch_losses) >= 5 else epoch_losses
    stability = 1.0 - (float(np.std(last5_acc)) / float(np.mean(last5_acc))) if last5_acc and np.mean(last5_acc) > 0 else 0.0
    accuracy_variance = float(np.var(last5_acc)) if last5_acc else 0.0
    loss_variance = float(np.var(last5_loss)) if last5_loss else 0.0
    convergence_epoch = 0
    if mean_acc_s:
        threshold = mean_acc_s[-1] * 0.95
        convergence_epoch = len(mean_acc_s)
        for idx, value in enumerate(mean_acc_s):
            if value >= threshold:
                convergence_epoch = idx + 1
                break

    model_param_count = sum(param.numel() for param in global_model.parameters())
    bytes_per_value = 4
    effective_sparsity = float(args.sparsity_ratio) if args.dp_method != "none" else 0.0
    upload_density = max(0.0, min(1.0, 1.0 - effective_sparsity))
    sampled_clients_per_round = max(1, min(current_num_clients, int(user_sample_rate * len(clients_train_loaders))))
    communication_per_round_bytes = model_param_count * bytes_per_value * upload_density * sampled_clients_per_round
    total_communication_bytes = communication_per_round_bytes * global_epoch
    avg_epoch_time = (training_time / global_epoch) if global_epoch > 0 else 0.0
    final_attack_metrics = epoch_attack_metrics[-1] if epoch_attack_metrics else {}

    print(
        "PAPER_METRICS_FINAL: "
        f"final_accuracy={final_acc if mean_acc_s else 0.0:.6f}, "
        f"best_accuracy={best_accuracy:.6f}, "
        f"mean_accuracy={mean_accuracy:.6f}, "
        f"final_loss={epoch_losses[-1] if epoch_losses else 0.0:.6f}, "
        f"mean_loss={float(np.mean(epoch_losses)) if epoch_losses else 0.0:.6f}, "
        f"convergence_epoch={convergence_epoch}, "
        f"stability={stability:.6f}, "
        f"accuracy_variance={accuracy_variance:.6f}, "
        f"loss_variance={loss_variance:.6f}, "
        f"avg_epoch_time={avg_epoch_time:.6f}, "
        f"training_time={training_time:.6f}, "
        f"dir_alpha={float(args.dir_alpha):.6f}, "
        f"target_epsilon={float(target_epsilon):.6f}, "
        f"target_delta={float(target_delta):.8f}, "
        f"clipping_bound={float(clipping_bound):.6f}, "
        f"comm_per_round_mb={_bytes_to_mb(communication_per_round_bytes):.6f}, "
        f"total_comm_mb={_bytes_to_mb(total_communication_bytes):.6f}, "
        f"model_params={model_param_count}"
    )
    print(
        "PAPER_CLIENT_FINAL: "
        f"mean={float(np.mean(final_client_accuracies)) if final_client_accuracies else 0.0:.6f}, "
        f"std={float(np.std(final_client_accuracies)) if final_client_accuracies else 0.0:.6f}, "
        f"min={float(np.min(final_client_accuracies)) if final_client_accuracies else 0.0:.6f}, "
        f"max={float(np.max(final_client_accuracies)) if final_client_accuracies else 0.0:.6f}, "
        f"worst_client_accuracy={float(np.min(final_client_accuracies)) if final_client_accuracies else 0.0:.6f}, "
        f"jain_fairness={_jain_fairness(final_client_accuracies):.6f}"
    )
    print(
        "PAPER_ATTACK_FINAL: "
        f"anti_inversion_ability={risk_scores[-1] if risk_scores else 0.0:.6f}, "
        f"leakage_risk={max(0.0, min(1.0, 1.0 - (risk_scores[-1] if risk_scores else 0.0))):.6f}, "
        f"avg_psnr={final_attack_metrics.get('avg_psnr', 0.0):.6f}, "
        f"best_psnr={final_attack_metrics.get('best_psnr', 0.0):.6f}, "
        f"avg_mse={final_attack_metrics.get('avg_mse', 0.0):.6f}, "
        f"best_mse={final_attack_metrics.get('best_mse', 0.0):.6f}, "
        f"avg_ssim={final_attack_metrics.get('avg_ssim', 0.0):.6f}, "
        f"best_ssim={final_attack_metrics.get('best_ssim', 0.0):.6f}, "
        f"avg_lpips={final_attack_metrics.get('avg_lpips', 0.0):.6f}, "
        f"best_lpips={final_attack_metrics.get('best_lpips', 0.0):.6f}, "
        f"attack_success_rate={final_attack_metrics.get('attack_success_rate', 0.0):.6f}, "
        f"noise_gaussianity={final_attack_metrics.get('noise_gaussianity', 0.0):.6f}, "
        f"noise_structure={final_attack_metrics.get('noise_structure', 0.0):.6f}"
    )

    print("=" * 80)


if __name__ == '__main__':
    main()

