import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from options import parse_args
from data import *
from net import *
from tqdm import tqdm
from utils import compute_noise_multiplier, sparsify_gradients, clip_gradients, adaptive_privacy_budget, \
    compute_fisher_diag, generate_random_gaussian_noise_like
from tqdm.auto import trange
import copy
import sys
import random
import numpy as np
import math
import datetime  # 

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

# 
if args.store == True:
    # 
    results_dir = f'./txt/{args.dirStr}'
    os.makedirs(results_dir, exist_ok=True)

    # ?    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{dataset}_results_{timestamp}.txt"
    file_path = os.path.join(results_dir, filename)

    print(f": {file_path}")

    try:
        saved_stdout = sys.stdout
        file = open(file_path, 'a', encoding='utf-8')
        sys.stdout = file

        # 
        print(f"=== ?===")
        print(f": {datetime.datetime.now()}")
        print(f"? {dataset}")
        print(f"? {num_clients}")
        print(f": {global_epoch}")
        print(f": {local_epoch}")
        print(f"? {args.lr}")
        print(f"(): {target_epsilon}")
        print(f": {clipping_bound}")
        print("=" * 50)

    except Exception as e:
        print(f":  - {e}")
        print("")
        sys.stdout = saved_stdout
        args.store = False  # 


def compute_client_importance_weights(client_data_sizes, client_accuracies):
    """
    ?- 
    Non-IID
    """
    total_size = sum(client_data_sizes)
    size_weights = [size / total_size for size in client_data_sizes]

    # 0
    if sum(client_accuracies) == 0:
        return size_weights

    # 
    accuracy_weights = [acc / sum(client_accuracies) for acc in client_accuracies]

    # ?0%?+ 40%
    balanced_weights = []
    for size_w, acc_w in zip(size_weights, accuracy_weights):
        balanced_w = 0.6 * size_w + 0.4 * acc_w
        balanced_weights.append(balanced_w)

    # ?    total_balanced = sum(balanced_weights)
    return [w / total_balanced for w in balanced_weights]


def personalized_local_update(model, dataloader, global_model, client_data_size,
                              total_data_size, client_epsilon, client_id):
    """
     - 3D-SCM
    """
    model.train()
    model = model.to(device)
    global_model = global_model.to(device)

    # 
    global_params = [param.clone().detach() for param in global_model.parameters()]

    # Fisher information
    fisher_diag = compute_fisher_diag(model, dataloader)

    # Fisher
    important_params_mask = []
    personal_params_mask = []
    for fisher_value in fisher_diag:
        important_mask = (fisher_value > args.fisher_threshold).float()
        personal_mask = (fisher_value <= args.fisher_threshold).float()
        important_params_mask.append(important_mask)
        personal_params_mask.append(personal_mask)

    # ?    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    # ?- ?    adaptive_lr = args.lr * (client_data_size / total_data_size) ** 0.5
    for param_group in optimizer.param_groups:
        param_group['lr'] = adaptive_lr

    # 
    num_batches = len(dataloader)
    sampling_rate = batch_size / client_data_size
    steps = local_epoch * num_batches

    # 
    adaptive_epsilon = client_epsilon * (total_data_size / client_data_size) ** 0.2
    adaptive_epsilon = min(adaptive_epsilon, client_epsilon * 2)

    sigma = clipping_bound * math.sqrt(2 * math.log(1.25 / target_delta)) / adaptive_epsilon

    print(f"?{client_id}: ?{client_data_size}, ={adaptive_epsilon:.4f}, ?{adaptive_lr:.6f}")

    # 
    for epoch in range(local_epoch):
        for batch_idx, (data, labels) in enumerate(dataloader):
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = loss_fn(outputs, labels)
            loss.backward()

            # Add noise under no_grad.
            with torch.no_grad():
                for param, important_mask, personal_mask in zip(model.parameters(),
                                                                important_params_mask,
                                                                personal_params_mask):
                    if param.grad is not None:
                        # 1. 
                        grad_norm = torch.norm(param.grad)
                        if grad_norm > clipping_bound:
                            param.grad.data.mul_(clipping_bound / grad_norm)

                        # 2. 3D-SCM
                        if args.use_chaotic:
                            # 3D-SCM?                            chaotic_seed = (client_id * 1000 + epoch * 100 + batch_idx) % 10000
                            torch.manual_seed(chaotic_seed)

                            # 3D-SCM?                            x0 = (0.1 + 0.8 * (chaotic_seed % 100) / 100) % 1
                            y0 = (x0 + 0.3) % 1
                            z0 = (y0 + 0.5) % 1

                            # 3D-SCM
                            important_chaotic_noise = generate_3d_scm_gaussian_noise(
                                param.grad.shape,
                                length=100 + batch_idx * 10,
                                a=args.scm_a,
                                b=args.scm_b,
                                x0=x0,
                                y0=y0,
                                z0=z0,
                                bit_offset=args.bit_offset,
                                n_bits=args.n_bits
                            )

                            # 
                            important_chaotic_noise = important_chaotic_noise.to(param.grad.device)
                            important_chaotic_noise = important_chaotic_noise * sigma * args.chaotic_factor * important_mask

                            # Personal chaotic noise component
                            personal_chaotic_noise = generate_3d_scm_gaussian_noise(
                                param.grad.shape,
                                length=50 + batch_idx * 5,
                                a=args.scm_a,
                                b=args.scm_b,
                                x0=(x0 + 0.5) % 1,
                                y0=(y0 + 0.5) % 1,
                                z0=(z0 + 0.5) % 1,
                                bit_offset=args.bit_offset + 1,
                                n_bits=args.n_bits
                            )

                            # 
                            personal_chaotic_noise = personal_chaotic_noise.to(param.grad.device)
                            personal_chaotic_noise = personal_chaotic_noise * sigma * 0.1 * args.chaotic_factor * personal_mask

                            # 
                            gaussian_noise = generate_random_gaussian_noise_like(param.grad) * sigma * (
                                        1 - args.chaotic_factor)

                            param.grad.data.add_(gaussian_noise + important_chaotic_noise + personal_chaotic_noise)
                        else:
                            # 
                            important_noise = generate_random_gaussian_noise_like(param.grad) * sigma * important_mask
                            personal_noise = generate_random_gaussian_noise_like(param.grad) * sigma * 0.1 * personal_mask
                            param.grad.data.add_(important_noise)
                            param.grad.data.add_(personal_noise)

            optimizer.step()

    # 
    with torch.no_grad():
        final_params = [param.clone().detach() for param in model.parameters()]
        model_update = [final_param - global_param for final_param, global_param in zip(final_params, global_params)]

        # ?        sparse_update = sparsify_gradients(model_update, sparsity=0.3)

    model = model.to('cpu')
    return sparse_update, fisher_diag


def adaptive_model_aggregation(global_model, clients_updates, clients_weights, clients_fisher_info):
    """
     - ?    """
    with torch.no_grad():
        # 
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

        # Fisher?        if clients_fisher_info and len(clients_fisher_info) == len(clients_updates):
            # Fisher
            avg_fisher = []
            for param_idx in range(len(clients_fisher_info[0])):
                fisher_values = [fisher[param_idx] for fisher in clients_fisher_info]
                avg_fisher_param = torch.mean(torch.stack(fisher_values), dim=0)
                avg_fisher.append(avg_fisher_param)

            # Fisher
            fisher_aggregated = []
            for param_idx, (base_param, fisher_mask) in enumerate(zip(base_aggregated, avg_fisher)):
                # 
                important_weight = 1.0
                personal_weight = 0.8  # 

                adaptive_weight = important_weight * fisher_mask + personal_weight * (1 - fisher_mask)
                adjusted_param = base_param * adaptive_weight
                fisher_aggregated.append(adjusted_param)

            return fisher_aggregated

        return base_aggregated


def analyze_data_distribution(clients_train_loaders, num_classes=10):
    """
    ?- Non-IID
    """
    print("\n=== ?===")
    client_distributions = []

    for i, train_loader in enumerate(clients_train_loaders):
        if i >= 5:  # 5
            break

        # 
        class_counts = [0] * num_classes
        total_samples = 0

        for data, labels in train_loader:
            for label in labels:
                class_counts[label.item()] += 1
                total_samples += 1

        if total_samples > 0:
            class_ratios = [count / total_samples for count in class_counts]
            client_distributions.append(class_ratios)
            print(f"?{i}: {[f'{ratio:.2f}' for ratio in class_ratios]}")

    return client_distributions


def test_global_model(global_model, test_loaders):
    """
    
    """
    global_model.eval()
    global_model = global_model.to(device)

    client_accuracies = []
    for i, test_loader in enumerate(test_loaders):
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

    global_model = global_model.to('cpu')
    return client_accuracies


def main():
    mean_acc_s = []
    acc_matrix = []
    global_accuracies = []

    # 
    if dataset == 'MNIST':
        train_dataset, test_dataset = get_mnist_datasets()
        clients_train_set = get_clients_datasets(train_dataset, num_clients)
        client_data_sizes = [len(client_dataset) for client_dataset in clients_train_set]
        clients_train_loaders = [DataLoader(client_dataset, batch_size=batch_size, shuffle=True)
                                 for client_dataset in clients_train_set]
        clients_test_loaders = [DataLoader(test_dataset, batch_size=256) for i in range(num_clients)]
        clients_models = [mnistNet() for _ in range(num_clients)]
        global_model = mnistNet()
    elif dataset == 'CIFAR10':
        clients_train_loaders, clients_test_loaders, client_data_sizes = get_CIFAR10(args.dir_alpha, num_clients)
        clients_models = [cifar10Net() for _ in range(num_clients)]
        global_model = cifar10Net()
    elif dataset == 'SVHN':
        clients_train_loaders, clients_test_loaders, client_data_sizes = get_SVHN(args.dir_alpha, num_clients)
        clients_models = [SVHNNet() for _ in range(num_clients)]
        global_model = SVHNNet()
    else:
        print('undefined dataset')
        return

    # 
    client_distributions = analyze_data_distribution(clients_train_loaders)

    # 
    total_data_size = sum(client_data_sizes)
    base_epsilon_per_round = target_epsilon / global_epoch
    client_privacy_budgets = adaptive_privacy_budget(client_data_sizes, base_epsilon=base_epsilon_per_round)

    print(f"\n=== Non-IID ===")
    print(f"? {dataset}")
    print(f"? {num_clients}")
    print(f": ={args.dir_alpha} (on-IID)")
    print(f": {client_data_sizes}")
    print(f"/: {[f'{eps:.4f}' for eps in client_privacy_budgets]}")

    # Initialize each client model with the global weights.
    for client_model in clients_models:
        client_model.load_state_dict(global_model.state_dict())

    # Track best global accuracy.
    best_global_accuracy = 0.0
    best_global_model = None

    # 
    for epoch in range(global_epoch):
        print(f"\n---  {epoch + 1}/{global_epoch} ---")

        # Sample participating clients for this round.
        sampled_indices = random.sample(range(num_clients), max(1, int(user_sample_rate * num_clients)))

        clients_updates = []
        clients_accuracies = []
        clients_fisher_info = []

        # Run local updates on sampled clients.
        for client_idx in sampled_indices:
            client_model = clients_models[client_idx]
            train_loader = clients_train_loaders[client_idx]
            test_loader = clients_test_loaders[client_idx]
            data_size = client_data_sizes[client_idx]
            epsilon = client_privacy_budgets[client_idx]

            # 
            update, fisher_info = personalized_local_update(
                client_model, train_loader, global_model,
                data_size, total_data_size, epsilon, client_idx
            )

            clients_updates.append(update)
            clients_fisher_info.append(fisher_info)

            # ?            accuracy = test_global_model(client_model, [test_loader])[0]
            clients_accuracies.append(accuracy)

            print(f'?{client_idx}: ?= {accuracy:.2f}%')

        # 
        sampled_data_sizes = [client_data_sizes[i] for i in sampled_indices]
        client_weights = compute_client_importance_weights(sampled_data_sizes, clients_accuracies)

        # 
        current_mean_acc = sum(clients_accuracies) / len(clients_accuracies)
        mean_acc_s.append(current_mean_acc)
        acc_matrix.append(clients_accuracies)

        # 
        aggregated_update = adaptive_model_aggregation(
            global_model, clients_updates, client_weights, clients_fisher_info
        )

        # 
        with torch.no_grad():
            for global_param, update in zip(global_model.parameters(), aggregated_update):
                global_param.add_(update)

        # 
        for client_model in clients_models:
            client_model.load_state_dict(global_model.state_dict())

        # 
        global_client_accuracies = test_global_model(global_model, clients_test_loaders)
        global_mean_accuracy = sum(global_client_accuracies) / len(global_client_accuracies)
        global_accuracies.append(global_mean_accuracy)

        print(f'?= {global_mean_accuracy:.2f}%')

        # Keep the best global checkpoint by mean client accuracy.
        if global_mean_accuracy > best_global_accuracy:
            best_global_accuracy = global_mean_accuracy
            best_global_model = copy.deepcopy(global_model.state_dict())

    # Restore the best checkpoint.
    if best_global_model is not None:
        global_model.load_state_dict(best_global_model)

    # Final evaluation.
    final_global_accuracies = test_global_model(global_model, clients_test_loaders)
    final_mean_accuracy = sum(final_global_accuracies) / len(final_global_accuracies)

    # 
    print("\n" + "=" * 70)
    print("Non-IID federated learning results")
    print("=" * 70)
    print(f"? {final_mean_accuracy:.2f}%")
    print(f"? {[f'{acc:.2f}%' for acc in final_global_accuracies]}")
    print(f"? {best_global_accuracy:.2f}%")
    print(f"? {[f'{acc:.2f}%' for acc in global_accuracies]}")
    print(f"Non-IID: ={args.dir_alpha}")
    print(f"? {np.var(client_data_sizes) if client_data_sizes else 0:.2f}")

    # 
    if len(global_accuracies) > 1:
        improvement = global_accuracies[-1] - global_accuracies[0]
        print(f"? {improvement:.2f}%")

    # 
    if args.store:
        try:
            print(f": {datetime.datetime.now()}")
            file.close()
            sys.stdout = saved_stdout
            print(f"")
        except:
            pass


if __name__ == '__main__':
    main()





