#!/usr/bin/env python
import os
import random
import json
import pickle
import argparse
import yaml
from json import JSONEncoder

from matplotlib import pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from tqdm import tqdm

import numpy as np
import torch
import time

from fed_baselines.client_base import FedClient
from fed_baselines.server_base import FedServer

from postprocessing.recorder import Recorder
from preprocessing.baselines_dataloader import divide_data, divide_noniid_data
from utils.models import LeNet

# 用于将python对象序列化为json对象
json_types = (list, dict, str, int, float, bool, type(None))
class PythonObjectEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, json_types):
            return super().default(self, obj)
        return {'_python_object': pickle.dumps(obj).decode('latin-1')}

def fed_args():
    """
    Arguments for running federated learning baselines
    :return: Arguments for federated learning baselines
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, required=True, help='Yaml file for configuration')

    args = parser.parse_args()
    return args

def fed_run():
    args = fed_args()
    with open(args.config, "r") as yaml_file:
        try:
            config = yaml.safe_load(yaml_file)
        except yaml.YAMLError as exc:
            print(exc)

    algo_list = ["FedAvg"]
    assert config["client"]["fed_algo"] in algo_list, "The federated learning algorithm is not supported"

    dataset_list = ["MNIST", "CIFAR10", "CIFAR100", "FashionMNIST", "SVHN"]
    assert config["system"]["dataset"] in dataset_list, "The dataset is not supported"

    model_list = ["LeNet", "AlexCifarNet", "CNN", "ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152"]
    assert config["system"]["model"] in model_list, "The model is not supported"

    # 初始化随机数
    np.random.seed(config["system"]["i_seed"])
    torch.manual_seed(config["system"]["i_seed"])
    random.seed(config["system"]["i_seed"])

    client_dict = {}
    recorder = Recorder()

    # trainset_config {'users': ['user_id1', ...], 'user_data': {'user_id1': train_data, ...}, 'num_samples': number}
    trainset_config, testset = divide_data(num_client=config["system"]["num_client"], 
                                        num_local_class=config["system"]["num_local_class"], 
                                        dataset_name=config["system"]["dataset"],
                                        i_seed=config["system"]["i_seed"])
    
    # trainset_config, testset = divide_noniid_data(num_client=config["system"]["num_client"], 
    #                                     imbalance_factor=0.3, 
    #                                     dataset_name=config["system"]["dataset"],
    #                                     i_seed=config["system"]["i_seed"])
    
    max_acc = 0

    # client_id: f00000-f00xxx
    # 初始化
    for client_id in trainset_config['users']:
        client_dict[client_id] = FedClient(client_id, 
                                       dataset_id=config["system"]["dataset"], 
                                       epoch=config["client"]["num_local_epoch"], 
                                       model_name=config["system"]["model"])
        client_dict[client_id].load_trainset(trainset_config['user_data'][client_id])
    
    fed_server = FedServer(trainset_config['users'], 
                           dataset_id=config["system"]["dataset"], 
                           model_name=config["system"]["model"])
    fed_server.load_testset(testset)
    # 初始化全局模型
    global_state_dict = fed_server.state_dict()

    start_time = time.time()
    print("开始初始训练和聚类分析...")

    # 1. 初始训练，收集所有客户端的模型
    client_models = {}
    for client_id in trainset_config['users']:
        # 更新使用全局模型
        client_dict[client_id].update(global_state_dict)
        # 本地训练得到模型参数
        state_dict, n_data, loss = client_dict[client_id].train()
        client_models[client_id] = state_dict

    # 2. 计算模型间的JS散度
    n_clients = len(trainset_config['users'])
    similarity_matrix = np.zeros((n_clients, n_clients))
    client_ids = list(trainset_config['users'])

    print("计算模型相似度矩阵...")
    for i in range(n_clients):
        for j in range(i, n_clients):
            # 创建临时模型来加载状态字典
            model_i = LeNet(num_classes=10, in_channels=1)
            model_j = LeNet(num_classes=10, in_channels=1)
            
            # 加载状态字典
            model_i.load_state_dict(client_models[client_ids[i]])
            model_j.load_state_dict(client_models[client_ids[j]])
            
            # 计算增强的相似度和JS散度
            similarity_score, js_div = calculate_model_similarity_enhanced(model_i, model_j)
            
            # 将相似度存入矩阵
            similarity_matrix[i, j] = similarity_score
            similarity_matrix[j, i] = similarity_score

    # 3. 处理相似度矩阵
    def process_similarity_matrix(matrix):
        """处理相似度矩阵，提高数值稳定性"""
        # 确保对称性
        matrix = (matrix + matrix.T) / 2
        
        # 数值稳定性处理
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=1.0, neginf=0.0)
        
        # 计算特征的范围用于自适应sigma
        feature_ranges = np.ptp(matrix, axis=0)
        adaptive_sigma = np.mean(feature_ranges) * 0.5
        
        # 应用高斯核
        matrix = np.exp(-((1 - matrix) ** 2) / (2 * adaptive_sigma ** 2))
        
        return matrix

    # 处理相似度矩阵
    similarity_matrix = process_similarity_matrix(similarity_matrix)

    # 4. 执行谱聚类（使用改进的参数）
    n_clusters = 3  # 可以根据需要调整
    spectral = SpectralClustering(
        n_clusters=n_clusters,
        affinity='precomputed',
        random_state=42,
        assign_labels='discretize',  # 使用离散化标签分配
        n_init=20  # 增加初始化次数提高稳定性
    )
    cluster_labels = spectral.fit_predict(similarity_matrix)

    cluster_time = time.time() - start_time
    print(f"谱聚类 运算时间: {cluster_time:.4f} 秒")
    # 4. 输出聚类结果
    print("\n客户端聚类结果:")
    print("-" * 50)
    clusters = {}
    for i, label in enumerate(cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(client_ids[i])

    for cluster_id, members in clusters.items():
        print(f"\n簇 {cluster_id}:")
        print(f"成员数量: {len(members)}")
        print("客户端列表:", members)

    # 5. 可视化聚类结果
    plt.figure(figsize=(15, 5))

    # 5.1 聚类结果散点图（使用PCA降维到2D）
    pca = PCA(n_components=2)
    similarity_2d = pca.fit_transform(similarity_matrix)

    plt.subplot(121)
    scatter = plt.scatter(similarity_2d[:, 0], similarity_2d[:, 1], 
                        c=cluster_labels, cmap='tab10')
    plt.colorbar(scatter)
    plt.title("Client clustering results")
    plt.xlabel("1th principal component")
    plt.ylabel("2th principal component")

    # 5.2 每个簇的相似度分布箱线图
    plt.subplot(122)
    cluster_similarities = []
    cluster_labels_list = []
    for label in range(n_clusters):
        mask = cluster_labels == label
        cluster_sim = similarity_matrix[mask][:, mask].flatten()
        cluster_similarities.extend(cluster_sim)
        cluster_labels_list.extend([f'Cluster {label}'] * len(cluster_sim))

    plt.boxplot([similarity_matrix[cluster_labels == i].flatten() 
                for i in range(n_clusters)],
                labels=[f'Cluster {i}' for i in range(n_clusters)])
    plt.title("Similarity distribution within a cluster")
    plt.ylabel("Similarity score")

    plt.tight_layout()
    plt.show()

    # 保存聚类结果
    clustering_results = {
        'similarity_matrix': similarity_matrix.tolist(),
        'cluster_labels': cluster_labels.tolist(),
        'client_ids': client_ids,
        'clusters': {str(k): v for k, v in clusters.items()}
    }

    # 将结果保存到JSON文件
    clustering_file = os.path.join(
        config["system"]["res_root"],
        f'clustering_results_{config["system"]["dataset"]}_{config["system"]["num_client"]}.json'
    )
    with open(clustering_file, 'w') as f:
        json.dump(clustering_results, f, indent=2)

    print(f"\n聚类结果已保存到: {clustering_file}")

    # 继续原有的联邦学习训练循环
    print("\n开始联邦学习训练...")

    pbar = tqdm(range(config["system"]["num_round"]))
    for global_round in pbar:
        # 每个节点进行本地训练
        for client_id in trainset_config['users']:
            # 更新使用全局模型
            client_dict[client_id].update(global_state_dict)
            # 本地训练得到模型参数、数据量和loss
            state_dict, n_data, loss = client_dict[client_id].train()
            fed_server.rec(client_dict[client_id].name, state_dict, n_data, loss)

        # 服务器聚合模型
        fed_server.select_clients()
        gglobal_state_dict, avg_loss, _ = fed_server.agg()

        # 测试&flush
        accuracy = fed_server.test()
        fed_server.flush()

        # 记录结果
        recorder.res['server']['iid_accuracy'].append(accuracy)
        recorder.res['server']['train_loss'].append(avg_loss)
        
        # 更新最大准确率
        if max_acc < accuracy:
            max_acc = accuracy
        pbar.set_description(
            'Global Round: %d' % global_round +
            '| Train loss: %.4f ' % avg_loss +
            '| Accuracy: %.4f' % accuracy +
            '| Max Acc: %.4f' % max_acc)
        
        # 保存结果到设置中编写的文件夹下
        if not os.path.exists(config["system"]["res_root"]):
            os.makedirs(config["system"]["res_root"])

        with open(os.path.join(config["system"]["res_root"], '[\'%s\',' % config["client"]["fed_algo"] +
                                '\'%s\',' % config["system"]["model"] +
                                str(config["client"]["num_local_epoch"]) + ',' +
                                str(config["system"]["num_local_class"]) + ',' +
                                str(config["system"]["i_seed"])) + ']', "w") as jsfile:
            json.dump(recorder.res, jsfile, cls=PythonObjectEncoder)

def preprocess_model_weights(model):
    """
    预处理模型权重，提取特征向量
    参数:
    - model: 待处理的模型
    返回:
    - 处理后的特征向量
    """
    weights_dict = {}
    for name, param in model.named_parameters():
        if 'weight' in name:  # 只处理权重参数
            weights = param.detach().cpu().numpy().flatten()
            # 使用Sturges规则计算最优bin数量
            num_bins = int(np.ceil(np.log2(len(weights)) + 1))
            
            # 计算权重范围并添加边界保护
            w_min, w_max = np.min(weights), np.max(weights)
            margin = 0.1 * (w_max - w_min)
            w_min -= margin
            w_max += margin
            
            # 计算直方图并归一化
            hist, _ = np.histogram(weights, 
                                 bins=num_bins,
                                 range=(w_min, w_max),
                                 density=True)
            hist = hist / np.sum(hist)
            weights_dict[name] = hist
    
    return weights_dict

def js_divergence_stable(p, q):
    """
    计算稳定的JS散度
    参数:
    - p, q: 两个概率分布
    返回:
    - js_div: JS散度值
    """
    epsilon = 1e-10
    
    # 确保非负和归一化
    p = np.maximum(p, epsilon)
    q = np.maximum(q, epsilon)
    
    p = p / np.sum(p)
    q = q / np.sum(q)
    
    # 计算中点分布
    m = 0.5 * (p + q)
    
    # 使用稳定的KL散度计算
    kl_pm = np.sum(p * (np.log(p + epsilon) - np.log(m + epsilon)))
    kl_qm = np.sum(q * (np.log(q + epsilon) - np.log(m + epsilon)))
    
    # 返回JS散度
    return 0.5 * (kl_pm + kl_qm)

def calculate_model_similarity_enhanced(model1, model2):
    """
    计算两个模型间的增强相似度
    参数:
    - model1, model2: 待比较的两个模型
    返回:
    - similarity_score: 相似度分数
    - js_div: JS散度值
    """
    # 提取两个模型的权重分布
    weights1 = preprocess_model_weights(model1)
    weights2 = preprocess_model_weights(model2)
    
    # 为不同层设置权重
    layer_weights = {
        'conv1.weight': 0,
        'conv2.weight': 0,
        'fc1.weight': 0,
        'fc2.weight': 0,
        'fc3.weight': 1
    }
    
    # 计算加权JS散度
    total_js_div = 0
    layer_js = js_divergence_stable(weights1['fc3.weight'], weights2['fc3.weight'])
    total_js_div += layer_js
    # for layer_name in weights1.keys():
    #     if layer_name in layer_weights:
    #         layer_js = js_divergence_stable(weights1[layer_name], weights2[layer_name])
    #         total_js_div += layer_weights[layer_name] * layer_js
    
    # 计算相似度分数
    similarity_score = 1 / (1 + total_js_div)
    
    return similarity_score, total_js_div

if __name__ == "__main__":
    fed_run()
