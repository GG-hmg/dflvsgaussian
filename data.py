import torch
import os
import pickle
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset, SubsetRandomSampler
import numpy as np
from typing import Tuple, List
from options import parse_args
import random
from collections import Counter

args = parse_args()

# 添加条件导入检查fedlab
try:
    from fedlab.utils.dataset.functional import hetero_dir_partition

    FEDLAB_AVAILABLE = True
except ImportError:
    FEDLAB_AVAILABLE = False
    print("警告：fedlab未安装，非IID数据分区功能受限")

# 如果fedlab不可用，提供一个简单的替代函数
if not FEDLAB_AVAILABLE:
    def hetero_dir_partition(labels, num_clients, num_classes, alpha):
        """
        简化的狄利克雷分布分区函数
        当fedlab不可用时使用
        """
        print("使用简化的数据分区方法（fedlab不可用）")
        # 简单的随机分区
        n = len(labels)
        indices = list(range(n))
        random.shuffle(indices)

        partition = []
        split_size = n // num_clients
        for i in range(num_clients):
            if i == num_clients - 1:
                client_indices = indices[i * split_size:]
            else:
                client_indices = indices[i * split_size:(i + 1) * split_size]
            partition.append(client_indices)

        return partition


# SVHN------------------------------------------------------------
def get_SVHN(alpha: float, num_clients: int) -> Tuple[List[DataLoader], List[DataLoader], List[int]]:
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # 检查数据集是否已存在
    data_path = './data/SVHN'
    download_flag = not os.path.exists(data_path) or len(os.listdir(data_path)) == 0

    train_dataset = datasets.SVHN(root=data_path, split='train', download=download_flag, transform=transform)
    test_dataset = datasets.SVHN(root=data_path, split='test', download=download_flag, transform=transform)

    num_classes = len(np.unique(train_dataset.labels))

    # 使用hetero_dir_partition函数分区数据
    train_partition = hetero_dir_partition(train_dataset.labels, num_clients, num_classes, alpha)

    train_loaders = []
    test_loaders = []
    client_data_sizes = []

    # Create a shared test_loader for all clients
    shared_test_loader = DataLoader(test_dataset, batch_size=256, shuffle=True)

    for i in range(num_clients):
        train_sampler = SubsetRandomSampler(train_partition[i])

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, drop_last=True)

        train_loaders.append(train_loader)
        test_loaders.append(shared_test_loader)
        client_data_sizes.append(len(train_partition[i]))

        # Calculate and print label percentages for each client
        label_counts = np.zeros(num_classes)
        for idx in train_partition[i]:
            label_counts[train_dataset.labels[idx]] += 1
        label_percentages = label_counts / len(train_partition[i]) * 100

        # Uncomment below lines to print label percentages for each client
        # print(f"Client {i}: Label Percentages:")
        # for label, percentage in enumerate(label_percentages):
        #     print(f"Label {label}: {percentage:.2f}%")

    return train_loaders, test_loaders, client_data_sizes


# MNIST-------------------------------------------------------------------------------------------------------
def get_mnist_datasets():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    # 检查数据集是否已存在
    data_path = './data/MNIST'
    download_flag = not os.path.exists(data_path) or len(os.listdir(data_path)) == 0

    train_dataset = datasets.MNIST(root=data_path, train=True, download=download_flag, transform=transform)
    test_dataset = datasets.MNIST(root=data_path, train=False, download=download_flag, transform=transform)

    return train_dataset, test_dataset


def get_clients_datasets(train_dataset, num_clients):
    n = len(train_dataset)
    indices = list(range(n))
    split_size = n // num_clients

    clients_datasets = []
    for i in range(num_clients):
        client_indices = indices[i * split_size:(i + 1) * split_size]
        client_dataset = Subset(train_dataset, client_indices)
        clients_datasets.append(client_dataset)

    return clients_datasets


def get_CIFAR10(alpha: float, num_clients: int) -> Tuple[List[DataLoader], List[DataLoader], List[int]]:
    if not FEDLAB_AVAILABLE:
        print("警告：fedlab不可用，使用IID CIFAR10数据分区")
        return get_iid_cifar10(num_clients)

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # 检查数据集是否已存在
    data_path = './data/CIFAR10'
    download_flag = not os.path.exists(data_path) or len(os.listdir(data_path)) == 0

    train_dataset = datasets.CIFAR10(root=data_path, train=True, download=download_flag, transform=transform)
    test_dataset = datasets.CIFAR10(root=data_path, train=False, download=download_flag, transform=transform)

    num_classes = len(np.unique(train_dataset.targets))

    train_partition = hetero_dir_partition(train_dataset.targets, num_clients, num_classes, alpha)

    train_loaders = []
    test_loaders = []
    client_data_sizes = []

    # Create a shared test_loader for all clients
    shared_test_loader = DataLoader(test_dataset, batch_size=256, shuffle=True)

    for i in range(num_clients):
        train_sampler = torch.utils.data.SubsetRandomSampler(train_partition[i])

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, drop_last=True)

        train_loaders.append(train_loader)
        test_loaders.append(shared_test_loader)
        client_data_sizes.append(len(train_partition[i]))

        # Calculate and print label percentages for each client
        label_counts = np.zeros(num_classes)
        for idx in train_partition[i]:
            label_counts[train_dataset.targets[idx]] += 1
        label_percentages = label_counts / len(train_partition[i]) * 100

        # print(f"Client {i}: Label Percentages:")
        # for label, percentage in enumerate(label_percentages):
        #     print(f"Label {label}: {percentage:.2f}%")

    return train_loaders, test_loaders, client_data_sizes


from typing import List, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_iid_cifar10(num_clients: int) -> Tuple[List[DataLoader], List[DataLoader], List[int]]:
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # 检查数据集是否已存在
    data_path = './data/CIFAR10'
    download_flag = not os.path.exists(data_path) or len(os.listdir(data_path)) == 0

    train_dataset = datasets.CIFAR10(root=data_path, train=True, download=download_flag, transform=transform)
    test_dataset = datasets.CIFAR10(root=data_path, train=False, download=download_flag, transform=transform)

    num_classes = len(np.unique(train_dataset.targets))

    train_loaders = []
    test_loaders = []
    client_data_sizes = []

    # Organize indices by class
    train_indices_by_class = [[] for _ in range(num_classes)]
    test_indices_by_class = [[] for _ in range(num_classes)]

    for idx, label in enumerate(train_dataset.targets):
        train_indices_by_class[label].append(idx)

    for idx, label in enumerate(test_dataset.targets):
        test_indices_by_class[label].append(idx)

    # Shuffle indices within each class
    for class_indices in train_indices_by_class:
        np.random.shuffle(class_indices)

    for class_indices in test_indices_by_class:
        np.random.shuffle(class_indices)

    # Split indices into num_clients partitions, ensuring equal class distribution
    train_partitions = [[] for _ in range(num_clients)]
    test_partitions = [[] for _ in range(num_clients)]

    for i in range(num_clients):
        for class_indices in train_indices_by_class:
            num_samples_per_client = len(class_indices) // num_clients
            start_idx = i * num_samples_per_client
            end_idx = (i + 1) * num_samples_per_client
            train_partitions[i].extend(class_indices[start_idx:end_idx])

        for class_indices in test_indices_by_class:
            num_samples_per_client = len(class_indices) // num_clients
            start_idx = i * num_samples_per_client
            end_idx = (i + 1) * num_samples_per_client
            test_partitions[i].extend(class_indices[start_idx:end_idx])

        np.random.shuffle(train_partitions[i])
        np.random.shuffle(test_partitions[i])

    for i in range(num_clients):
        # Create sub-datasets for each client using the selected indices.
        train_subset = torch.utils.data.Subset(train_dataset, train_partitions[i])
        test_subset = torch.utils.data.Subset(test_dataset, test_partitions[i])

        # Create data loaders using the sub-datasets and shuffle=True.
        train_loader = DataLoader(train_subset, batch_size=50, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=256, shuffle=True)

        train_loaders.append(train_loader)
        test_loaders.append(test_loader)
        client_data_sizes.append(len(train_partitions[i]))

    return train_loaders, test_loaders, client_data_sizes


# FEMNIST-------------------------------------------------------------------------------------------
# 注释掉TensorFlow相关的FEMNIST代码，因为存在兼容性问题
# 如果需要FEMNIST数据集，请安装兼容的TensorFlow版本

class TFDatasetToTorch(Dataset):
    def __init__(self, data, transform=None):
        self.transform = transform
        self.data = []
        # 这是一个空的实现，因为TensorFlow不可用

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 返回空数据
        return torch.tensor([]), torch.tensor(0)


def get_FEMNIST(numOfClients):
    print("错误：FEMNIST数据集需要TensorFlow和tensorflow-federated，但这些包已被移除以避免兼容性问题")
    print("请使用其他数据集：MNIST、CIFAR10或SVHN")
    return [], [], []


# EMNIST-----------------------------------------------------------------------

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

from collections import Counter


def get_EMNIST(num_clients):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 检查数据集是否已存在
    data_path = './data'
    download_flag = not os.path.exists(os.path.join(data_path, 'EMNIST')) or len(
        os.listdir(os.path.join(data_path, 'EMNIST'))) == 0

    emnist_train = datasets.EMNIST(data_path, split='byclass', train=True, download=download_flag, transform=transform)
    emnist_test = datasets.EMNIST(data_path, split='byclass', train=False, download=download_flag, transform=transform)

    client_data_sizes = []
    clients_train_loaders = []
    clients_test_loaders = []

    train_data_indices_by_client = [[] for _ in range(num_clients)]
    test_data_indices_by_client = [[] for _ in range(num_clients)]

    unique_labels = np.unique(emnist_train.targets.numpy())

    for label in unique_labels:
        label_train_indices = np.where(emnist_train.targets.numpy() == label)[0]
        label_test_indices = np.where(emnist_test.targets.numpy() == label)[0]

        samples_per_client_train = int(0.2 * len(label_train_indices)) // num_clients
        samples_per_client_test = int(0.2 * len(label_test_indices)) // num_clients

        for i in range(num_clients):
            train_indices = np.random.choice(label_train_indices, samples_per_client_train, replace=False)
            test_indices = np.random.choice(label_test_indices, samples_per_client_test, replace=False)

            train_data_indices_by_client[i].extend(train_indices)
            test_data_indices_by_client[i].extend(test_indices)

            label_train_indices = np.setdiff1d(label_train_indices, train_indices)
            label_test_indices = np.setdiff1d(label_test_indices, test_indices)

    for i in range(num_clients):
        client_train_dataset = torch.utils.data.Subset(emnist_train, train_data_indices_by_client[i])
        client_test_dataset = torch.utils.data.Subset(emnist_test, test_data_indices_by_client[i])

        clients_train_loaders.append(DataLoader(client_train_dataset, batch_size=32, shuffle=True))
        clients_test_loaders.append(DataLoader(client_test_dataset, batch_size=32, shuffle=False))

        client_data_sizes.append(len(train_data_indices_by_client[i]))

        print("Client ", i + 1, " size: ", client_data_sizes[i])
        client_train_labels = [int(emnist_train.targets[train_data_indices_by_client[i][j]]) for j in
                               range(len(train_data_indices_by_client[i]))]
        counter = Counter(client_train_labels)
        print("Client ", i + 1, " labels distribution: ", counter)

    return clients_train_loaders, clients_test_loaders, client_data_sizes


# FashionMNIST-------------------------------------------------------------------------------------------------
def get_FashionMNIST(alpha: float, num_clients: int) -> Tuple[List[DataLoader], List[DataLoader], List[int]]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # FashionMNIST 标准化
    ])

    data_path = './data/FashionMNIST'
    download_flag = not os.path.exists(data_path) or len(os.listdir(data_path)) == 0

    train_dataset = datasets.FashionMNIST(root=data_path, train=True, download=download_flag, transform=transform)
    test_dataset = datasets.FashionMNIST(root=data_path, train=False, download=download_flag, transform=transform)

    num_classes = len(np.unique(train_dataset.targets))

    # 使用你现有的 hetero_dir_partition 进行联邦分区
    train_partition = hetero_dir_partition(train_dataset.targets, num_clients, num_classes, alpha)

    train_loaders = []
    test_loaders = []
    client_data_sizes = []

    shared_test_loader = DataLoader(test_dataset, batch_size=256, shuffle=True)

    for i in range(num_clients):
        train_sampler = torch.utils.data.SubsetRandomSampler(train_partition[i])
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, drop_last=True)

        train_loaders.append(train_loader)
        test_loaders.append(shared_test_loader)
        client_data_sizes.append(len(train_partition[i]))

    return train_loaders, test_loaders, client_data_sizes