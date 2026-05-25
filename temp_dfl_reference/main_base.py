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
from fl_helpers import (
    adaptive_model_aggregation as _shared_adaptive_model_aggregation,
    analyze_data_distribution as _shared_analyze_data_distribution,
    compute_client_importance_weights as _shared_compute_client_importance_weights,
    personalized_local_update as _shared_personalized_local_update,
    set_global_seed as _shared_set_global_seed,
    test_global_model as _shared_test_global_model,
)

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
    return _shared_compute_client_importance_weights(client_data_sizes, client_accuracies)

def personalized_local_update(model, dataloader, global_model, client_data_size,
                              total_data_size, client_epsilon, client_id):
    return _shared_personalized_local_update(
        model,
        dataloader,
        global_model,
        client_data_size,
        total_data_size,
        client_epsilon,
        client_id,
    )

def adaptive_model_aggregation(global_model, clients_updates, clients_weights, clients_fisher_info):
    return _shared_adaptive_model_aggregation(
        global_model,
        clients_updates,
        clients_weights,
        clients_fisher_info,
    )

def analyze_data_distribution(clients_train_loaders, num_classes=10):
    return _shared_analyze_data_distribution(clients_train_loaders, num_classes=num_classes)

def test_global_model(global_model, test_loaders):
    return _shared_test_global_model(global_model, test_loaders)

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





