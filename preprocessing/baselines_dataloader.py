# 导入必要的库
import torch  # PyTorch深度学习框架
import torchvision  # PyTorch视觉库
import torchvision.transforms as transforms  # 用于数据预处理和增强
from tqdm import tqdm  # 进度条显示
from torch.utils.data import Subset, Dataset, ConcatDataset  # 数据集处理工具
import os  # 操作系统接口
import PIL  # 图像处理库
import numpy as np  # 用于生成狄利克雷分布
from collections import defaultdict
from PIL import Image
import random


def load_data(name, root='./data', download=True, save_pre_data=True):
    """
    加载不同数据集的函数
    参数:
    - name: 数据集名称
    - root: 数据存储路径
    - download: 是否下载数据集
    """
    data_dict = ['MNIST', 'EMNIST', 'FashionMNIST', 'CelebA', 'CIFAR10', 'QMNIST', 'SVHN', "IMAGENET", 'CIFAR100']
    assert name in data_dict, "The dataset is not present"

    if not os.path.exists(root):
        os.makedirs(root, exist_ok=True)

    if name == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        trainset = torchvision.datasets.MNIST(root=root, train=True,  download=download, transform=transform)
        testset = torchvision.datasets.MNIST(root=root, train=False,  download=download, transform=transform)
        
    elif name == 'EMNIST':
        # byclass, bymerge, balanced, letters, digits, mnist
        transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
        trainset = torchvision.datasets.EMNIST(root=root, train=True, split= 'letters', download=download, transform=transform)
        testset = torchvision.datasets.EMNIST(root=root, train=False, split= 'letters', download=download, transform=transform)

    elif name == 'FashionMNIST':
        transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])
        trainset = torchvision.datasets.FashionMNIST(root=root, train=True, download=download, transform=transform)
        testset = torchvision.datasets.FashionMNIST(root=root, train=False, download=download, transform=transform)

    elif name == 'CelebA':
        # Could not loaded possibly for google drive break downs, try again at week days
        target_transform = transforms.Compose([transforms.ToTensor()])
        transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = torchvision.datasets.CelebA(root=root, split='train', target_type=list, download=download, transform=transform, target_transform=target_transform)
        testset = torchvision.datasets.CelebA(root=root, split='test', target_type=list, download=download, transform=transform, target_transform=target_transform)


    elif name == 'CIFAR10':
        # CIFAR10的训练集数据增强
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),  # 随机裁剪
            transforms.RandomHorizontalFlip(),      # 随机水平翻转
            transforms.ColorJitter(
                brightness=0.2,  # 亮度
                contrast=0.2,    # 对比度
                saturation=0.2,  # 饱和度
                hue=0.1         # 色相
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],  # CIFAR10的RGB通道均值
                std=[0.2023, 0.1994, 0.2010]    # CIFAR10的RGB通道标准差
            ),
        ])
        
        # 测试集只需要标准化处理
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010]
            ),
        ])
        
        trainset = torchvision.datasets.CIFAR10(
            root=root, 
            train=True, 
            download=download, 
            transform=train_transform
        )
        testset = torchvision.datasets.CIFAR10(
            root=root, 
            train=False, 
            download=download, 
            transform=test_transform
        )
        
        # 将标签转换为张量
        trainset.targets = torch.tensor(trainset.targets, dtype=torch.long)
        testset.targets = torch.tensor(testset.targets, dtype=torch.long)

    elif name == 'CIFAR100':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        trainset = torchvision.datasets.CIFAR100(root=root, train=True, transform=transform, download=True)
        testset = torchvision.datasets.CIFAR100(root=root, train=False, transform=transform, download=True)
        trainset.targets = torch.Tensor(trainset.targets)
        testset.targets = torch.Tensor(testset.targets)

    elif name == 'QMNIST':
        transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
        trainset = torchvision.datasets.QMNIST(root=root, what='train', compat=True, download=download, transform=transform)
        testset = torchvision.datasets.QMNIST(root=root, what='test', compat=True, download=download, transform=transform)

    elif name == 'SVHN':
        transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
        trainset = torchvision.datasets.SVHN(root=root, split='train', download=download, transform=transform)
        testset = torchvision.datasets.SVHN(root=root, split='test', download=download, transform=transform)
        trainset.targets = torch.Tensor(trainset.labels)
        testset.targets = torch.Tensor(testset.labels)

    elif name == 'IMAGENET':
        train_val_transform = transforms.Compose([
            transforms.ColorJitter(hue=.05, saturation=.05),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20, resample=PIL.Image.BILINEAR),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.ColorJitter(hue=.05, saturation=.05),
            transforms.ToTensor(),
        ])
        # transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                          std=[0.229, 0.224, 0.225])])
        trainset = torchvision.datasets.ImageFolder(root='./data/tiny-imagenet-200/train', transform=train_val_transform)
        testset = torchvision.datasets.ImageFolder(root='./data/tiny-imagenet-200/val', transform=test_transform)
        trainset.targets = torch.Tensor(trainset.targets)
        testset.targets = torch.Tensor(testset.targets)

    len_classes_dict = {
        'MNIST': 10,
        'EMNIST': 26, # ByClass: 62. ByMerge: 814,255 47.Digits: 280,000 10.Letters: 145,600 26.MNIST: 70,000 10.
        'FashionMNIST': 10,
        'CelebA': 0,
        'CIFAR10': 10,
        'QMNIST': 10,
        'SVHN': 10,
        'IMAGENET': 200,
        'CIFAR100': 100
    }

    len_classes = len_classes_dict[name]
    
    return trainset, testset, len_classes

def divide_data(num_client=1, num_local_class=10, dataset_name='emnist', i_seed=0):
    """
    将数据集分割给不同客户端的函数
    参数:
    - num_client: 客户端数量
    - num_local_class: 每个客户端拥有的类别数
    - dataset_name: 数据集名称
    - i_seed: 随机种子
    """

    torch.manual_seed(i_seed)

    trainset, testset, len_classes = load_data(dataset_name, download=True, save_pre_data=False)

    num_classes = len_classes
    if num_local_class == -1:
        num_local_class = num_classes
    assert 0 < num_local_class <= num_classes, "number of local class should smaller than global number of class"

    trainset_config = {'users': [],
                       'user_data': {},
                       'num_samples': []}
    config_division = {}  # Count of the classes for division记录每个客户端分配到的类别
    config_class = {}  # Configuration of class distribution in clients记录每个类别被分配的次数
    config_data = {}  # Configuration of data indexes for each class : Config_data[cls] = [0, []] | pointer and indexes存储每个类别的数据索引

    # 将config_data分成十份，每份num_client*num_local_class/10个类别
    # config_class {'user_id': [cls1, cls2], 'user_id': [cls1, cls2], ...}
    # user_id 数量为 num_client 的数值
    # config_division {cls1: num_client*num_local_class/10, cls2: num_client*num_local_class/10, ...}
    # config_data {cls1: [0, [data1, data2, ..., data{num_client*num_local_class/10}]], cls2: [0, [data1, data2, ..., data{num_client*num_local_class/10}]], ...}
    for i in range(num_client):
        config_class['f_{0:05d}'.format(i)] = []
        for j in range(num_local_class):
            cls = (i+j) % num_classes
            if cls not in config_division:
                config_division[cls] = 1
                config_data[cls] = [0, []]

            else:
                config_division[cls] += 1
            config_class['f_{0:05d}'.format(i)].append(cls)

    # print(config_class)
    # print(config_division)
    # print(config_data)

    # 遍历每个类别，为每个类别的数据进行分区
    for cls in config_division.keys():
        # 找出当前类别的所有数据索引
        indexes = torch.nonzero(trainset.targets == cls)
        # 获取当前类别的总数据点数量
        num_datapoint = indexes.shape[0]
        # 随机打乱当前类别的数据索引
        indexes = indexes[torch.randperm(num_datapoint)]
        # 计算每个分区应该包含的数据点数量
        num_partition = num_datapoint // config_division[cls]
        
        # 遍历每个分区，将数据均匀分配
        for i_partition in range(config_division[cls]):
            # 如果是最后一个分区，将剩余所有数据都放入
            if i_partition == config_division[cls] - 1:
                config_data[cls][1].append(indexes[i_partition * num_partition:])
            # 否则，按照计算好的分区大小进行分配
            else:
                config_data[cls][1].append(indexes[i_partition * num_partition: (i_partition + 1) * num_partition])



    # 遍历每个用户，为其分配数据
    for user in tqdm(config_class.keys()):
        # 初始化当前用户的数据索引tensor
        user_data_indexes = torch.tensor([])
        
        # 遍历分配给该用户的所有类别
        for cls in config_class[user]:
            # 获取当前类别的下一个可用分区的数据索引
            user_data_index = config_data[cls][1][config_data[cls][0]]
            # 将新的数据索引添加到用户的数据索引集合中
            user_data_indexes = torch.cat((user_data_indexes, user_data_index))
            # 更新该类别的分区指针，指向下一个可用分区
            config_data[cls][0] += 1
        
        # 处理数据索引格式：压缩维度，转换为整数列表
        user_data_indexes = user_data_indexes.squeeze().int().tolist()
        # 使用索引创建用户的数据子集
        user_data = Subset(trainset, user_data_indexes)
        
        # 将用户信息添加到配置中
        trainset_config['users'].append(user)  # 添加用户ID
        trainset_config['user_data'][user] = user_data  # 存储用户的数据子集
        trainset_config['num_samples'] = len(user_data)  # 记录数据样本数量

    return trainset_config, testset

def divide_noniid_data(num_client=1, imbalance_factor=0.6, dataset_name='emnist', i_seed=0):
    """
    将数据集以非独立同分布的方式分割给不同客户端
    
    参数:
    - num_client: 客户端数量
    - imbalance_factor: 不平衡因子(0-1之间)，越大分布越不均衡
    - dataset_name: 数据集名称
    - i_seed: 随机种子
    
    返回:
    - trainset_config: 训练集配置
    - testset: 测试集
    """
    torch.manual_seed(i_seed)
    np.random.seed(i_seed)
    
    # 加载数据集
    trainset, testset, num_classes = load_data(dataset_name, download=True, save_pre_data=False)
    
    # 初始化配置字典
    trainset_config = {
        'users': [],
        'user_data': {},
        'num_samples': []
    }
    
    # 为每个类别创建数据索引列表
    class_indices = {}
    for cls in range(num_classes):
        indices = torch.nonzero(trainset.targets == cls).squeeze()
        class_indices[cls] = indices[torch.randperm(len(indices))]
    
    # 计算每个客户端的目标类别分布
    def generate_dirichlet_distribution():
        """生成狄利克雷分布的类别权重"""
        alpha = [1.0 - imbalance_factor] * num_classes  # alpha越小，分布越不均衡
        return np.random.dirichlet(alpha)
    
    # 为每个客户端分配数据
    client_distributions = []
    for i in range(num_client):
        client_distributions.append(generate_dirichlet_distribution())
    
    # 归一化客户端分布
    client_distributions = np.array(client_distributions)
    client_distributions = client_distributions / client_distributions.sum(axis=0, keepdims=True)
    
    # 为每个客户端分配数据
    for i in range(num_client):
        user_id = f'f_{i:05d}'
        user_data_indices = []
        
        # 根据分布为每个类别分配数据
        for cls in range(num_classes):
            # 计算该客户端在该类别上应获得的数据量
            class_size = len(class_indices[cls])
            num_samples = int(class_size * client_distributions[i][cls])
            
            # 确保每个类别至少有一些数据
            num_samples = max(1, min(num_samples, len(class_indices[cls])))
            
            # 获取数据索引
            if len(class_indices[cls]) > 0:
                selected_indices = class_indices[cls][:num_samples]
                user_data_indices.extend(selected_indices.tolist())
                class_indices[cls] = class_indices[cls][num_samples:]
        
        # 创建用户数据子集
        if len(user_data_indices) > 0:
            user_data = Subset(trainset, user_data_indices)
            
            # 更新配置
            trainset_config['users'].append(user_id)
            trainset_config['user_data'][user_id] = user_data
            trainset_config['num_samples'].append(len(user_data_indices))
    
    # 打印分布情况统计
    print("\n数据分布统计:")
    print("-" * 50)
    for i, user_id in enumerate(trainset_config['users']):
        user_data = trainset_config['user_data'][user_id]
        if isinstance(user_data, Subset):
            targets = trainset.targets[user_data.indices]
            unique, counts = torch.unique(targets, return_counts=True)
            class_dist = dict(zip(unique.tolist(), counts.tolist()))
            print(f"客户端 {user_id}:")
            print(f"总样本数: {len(user_data)}")
            print(f"类别分布: {class_dist}")
            print("-" * 30)
    
    return trainset_config, testset

# 首先定义CustomCIFAR10Subset类
class CustomCIFAR10Subset(Dataset):
    """自定义CIFAR10子集，支持数据增强"""
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
        
    def __getitem__(self, idx):
        img = self.dataset.data[self.indices[idx]]
        label = self.dataset.targets[self.indices[idx]]
        img = Image.fromarray(img)
        
        if self.transform:
            img = self.transform(img)
            
        return img, label
    
    def __len__(self):
        return len(self.indices)

class MemoryCIFAR10Subset(Dataset):
    """将CIFAR10数据预加载到内存的数据集"""
    def __init__(self, dataset, indices, transform=None, is_augment=False):
        # 预加载数据到内存
        self.data = [dataset.data[i] for i in indices]
        self.targets = [dataset.targets[i] for i in indices]
        self.transform = transform
        self.is_augment = is_augment
        
        # 基础转换
        self.base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010]
            )
        ])
        
        # 额外的数据增强
        self.extra_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))
        ])
    
    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.targets[idx]
        img = Image.fromarray(img)
        
        # 应用基础转换
        if not self.is_augment:
            img = self.base_transform(img)
        else:
            # 应用额外的数据增强
            img = self.extra_transform(img)
            img = self.base_transform(img)
            
        return img, label
    
    def __len__(self):
        return len(self.data)

def divide_data_cifar(num_client=10, alpha=0.5, seed=42, min_size=50):
    """
    改进的CIFAR10数据划分方法，支持数据重复分配
    
    Args:
        num_client: 客户端数量
        alpha: Dirichlet分布的alpha参数
        seed: 随机种子
        min_size: 每个类别的最小样本数
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    
    # 加载数据集
    trainset, testset, num_classes = load_data('CIFAR10')
    labels = np.array(trainset.targets)
    
    # 使用Dirichlet分布进行初始数据划分
    client_proportions = np.random.dirichlet(alpha=[alpha] * num_client, size=num_classes)
    
    # 存储每个客户端的数据索引
    client_data_indices = [[] for _ in range(num_client)]
    
    # 按类别划分数据
    for cls_idx in range(num_classes):
        # 获取当前类别的所有索引
        cls_indices = np.where(labels == cls_idx)[0]
        
        # 计算每个客户端应获得的样本数
        proportions = client_proportions[cls_idx]
        proportions = proportions / proportions.sum()
        required_samples = np.maximum(
            min_size,  # 至少需要min_size个样本
            (proportions * len(cls_indices)).astype(int)
        )
        
        # 如果需要的总样本数超过现有样本数，则需要重复采样
        total_required = sum(required_samples)
        if total_required > len(cls_indices):
            # 计算需要重复的次数
            repeat_times = int(np.ceil(total_required / len(cls_indices)))
            # 重复数据
            cls_indices = np.tile(cls_indices, repeat_times)
            # 打乱顺序
            np.random.shuffle(cls_indices)
        
        # 分配数据给各个客户端
        start_idx = 0
        for client_idx in range(num_client):
            end_idx = start_idx + required_samples[client_idx]
            client_data_indices[client_idx].extend(cls_indices[start_idx:end_idx])
            start_idx = end_idx
    
    # 创建数据集配置
    trainset_config = {
        'users': [],
        'user_data': {},
        'num_samples': []
    }
    
    # 为每个客户端创建数据集
    for i in range(num_client):
        user_id = f'f_{i:05d}'
        trainset_config['users'].append(user_id)
        
        # 打乱当前客户端的数据索引
        client_indices = client_data_indices[i]
        np.random.shuffle(client_indices)
        
        # 创建基础数据集
        base_dataset = MemoryCIFAR10Subset(
            trainset, 
            client_indices,
            is_augment=False
        )
        
        # 创建增强数据集
        augmented_dataset = MemoryCIFAR10Subset(
            trainset, 
            client_indices,
            is_augment=True
        )
        
        # 合并数据集
        combined_dataset = ConcatDataset([base_dataset, augmented_dataset])
        
        trainset_config['user_data'][user_id] = combined_dataset
        trainset_config['num_samples'].append(len(combined_dataset))
        
        # 打印每个客户端的类别分布
        client_labels = [trainset.targets[idx] for idx in client_indices]
        unique_labels, counts = np.unique(client_labels, return_counts=True)
        print(f"\nClient {user_id} class distribution:")
        total_samples = sum(counts)
        for label, count in zip(unique_labels, counts):
            print(f"Class {label}: {count} samples ({count/total_samples*100:.2f}%)")
        print(f"Total samples: {total_samples}")
    
    return trainset_config, testset

def create_mixed_batches(dataset, batch_size=32):
    """创建混合批次，确保每个批次包含多个类别的样本"""
    labels = np.array(dataset.targets)
    n_classes = len(np.unique(labels))
    
    # 按类别组织数据索引
    class_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        class_indices[label].append(idx)
    
    # 计算每个批次中每个类别的样本数
    samples_per_class = batch_size // n_classes
    
    batches = []
    while True:
        batch = []
        for class_label in range(n_classes):
            if len(class_indices[class_label]) >= samples_per_class:
                # 随机选择样本
                selected = random.sample(class_indices[class_label], samples_per_class)
                batch.extend(selected)
                # 移除已选择的样本
                class_indices[class_label] = [i for i in class_indices[class_label] if i not in selected]
            else:
                # 如果某个类别的样本不足，停止创建批次
                return batches
        
        if len(batch) == batch_size:
            batches.append(batch)
        
        # 检查是否所有类别都没有足够的样本
        if all(len(indices) < samples_per_class for indices in class_indices.values()):
            break
    
    return batches

if __name__ == "__main__":
    # 'MNIST', 'EMNIST', 'FashionMNIST', 'CelebA', 'CIFAR10', 'QMNIST', 'SVHN'
    # data_dict = ['MNIST', 'EMNIST', 'FashionMNIST', 'CIFAR10', 'QMNIST', 'SVHN']

    # for name in data_dict:
    #     print(name)
    #     divide_data(num_client=20, num_local_class=2, dataset_name=name, i_seed=0)

    # divide_data(num_client=50, num_local_class=2, dataset_name='MNIST', i_seed=0)

    # 测试不同的不平衡因子
    for imbalance in [0.3, 0.6, 0.9]:
        print(f"\n测试不平衡因子: {imbalance}")
        print("=" * 60)
        trainset_config, testset = divide_noniid_data(
            num_client=5,
            imbalance_factor=imbalance,
            dataset_name='MNIST',
            i_seed=42
        )