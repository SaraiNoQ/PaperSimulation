#!/usr/bin/env python
import os
import random
import json
import pickle
import argparse
import yaml
from json import JSONEncoder
from abc import ABC, abstractmethod
from scipy.stats import pearsonr
from scipy.stats import wasserstein_distance

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
from utils.similarity_cal import WassersteinSimilarity, PearsonSimilarity, JSDistanceSimilarity, perform_spectral_clustering, visualize_clustering_results

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
    if (config["system"]["noniid"]):
        trainset_config, testset = divide_noniid_data(num_client=config["system"]["num_client"], 
                                        imbalance_factor=config["system"]["imbalance_factor"], 
                                        dataset_name=config["system"]["dataset"],
                                        i_seed=config["system"]["i_seed"])
    else:
        trainset_config, testset = divide_data(num_client=config["system"]["num_client"], 
                                            num_local_class=config["system"]["num_local_class"], 
                                            dataset_name=config["system"]["dataset"],
                                            i_seed=config["system"]["i_seed"])
    
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

    # 2. 使用JS散度相似度计算器进行聚类
    n_clients = len(trainset_config['users'])
    client_ids = list(trainset_config['users'])
    
    print("计算模型相似度矩阵...")

    if (config["system"]["similarity_method"] == "wasserstein"):
        wasserstein_calculator = WassersteinSimilarity(n_clients, client_models, client_ids)
        similarity_matrix = wasserstein_calculator.compute_similarity_matrix()
    elif (config["system"]["similarity_method"] == "pearson"):
        pearson_calculator = PearsonSimilarity(n_clients, client_models, client_ids)
        similarity_matrix = pearson_calculator.compute_similarity_matrix()
    elif (config["system"]["similarity_method"] == "js"):
        js_calculator = JSDistanceSimilarity(n_clients, client_models, client_ids)
        similarity_matrix = js_calculator.compute_similarity_matrix()

    # 3. 执行谱聚类
    n_clusters = 3  # 可以根据需要调整
    cluster_labels = perform_spectral_clustering(similarity_matrix, n_clusters)

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
    visualize_clustering_results(similarity_matrix, cluster_labels, n_clusters, config["system"]["similarity_method"])

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
        f'clustering_results_{config["system"]["dataset"]}_{config["system"]["num_client"]}_{config["system"]["similarity_method"]}_{config["system"]["imbalance_factor"]}_{config["system"]["i_seed"]}.json'
    )
    with open(clustering_file, 'w') as f:
        json.dump(clustering_results, f, indent=2)

    print(f"\n聚类结果已保存到: {clustering_file}")

    # 继续原有的联邦学习训练循环
    print("\n开始分层联邦学习训练...")

    # 为每个簇创建一个独立的服务器和记录器
    cluster_servers = {}
    cluster_recorders = {}
    cluster_max_acc = {}
    
    # 创建顶层服务器用于簇间聚合
    # top_level_server = FedServer(
    #     list(clusters.keys()),  # 使用簇ID作为客户端ID
    #     dataset_id=config["system"]["dataset"],
    #     model_name=config["system"]["model"]
    # )
    # top_level_server.load_testset(testset)
    # top_level_recorder = Recorder()
    # top_level_max_acc = 0
    
    for cluster_id in clusters.keys():
        # 为每个簇创建服务器
        cluster_servers[cluster_id] = FedServer(
            clusters[cluster_id],  # 该簇的客户端列表
            dataset_id=config["system"]["dataset"],
            model_name=config["system"]["model"]
        )
        # 加载测试集
        cluster_servers[cluster_id].load_testset(testset)
        # 创建记录器
        cluster_recorders[cluster_id] = Recorder()
        cluster_max_acc[cluster_id] = 0

    # 并行训练每个簇
    pbar = tqdm(range(config["system"]["num_round"]))
    for global_round in pbar:
        cluster_metrics = {}
        
        # 1. 簇内训练和聚合
        for cluster_id, members in clusters.items():
            # 获取该簇的服务器和记录器
            cluster_server = cluster_servers[cluster_id]
            cluster_recorder = cluster_recorders[cluster_id]
            
            # 簇内每个节点进行本地训练
            for client_id in members:
                # 获取簇内服务器的全局模型
                cluster_global_state = cluster_server.state_dict()
                # 更新使用簇内全局模型
                client_dict[client_id].update(cluster_global_state)
                # 本地训练得到模型参数、数据量和loss
                state_dict, n_data, loss = client_dict[client_id].train()
                cluster_server.rec(client_id, state_dict, n_data, loss)

            # 簇内服务器聚合模型
            cluster_server.select_clients()
            cluster_global_state, avg_loss, _ = cluster_server.agg()

            # 簇内测试和记录
            accuracy = cluster_server.test()
            cluster_server.flush()

            # 记录簇内结果
            cluster_recorder.res['server']['iid_accuracy'].append(accuracy)
            cluster_recorder.res['server']['train_loss'].append(avg_loss)
            
            # 更新簇内最大准确率
            if cluster_max_acc[cluster_id] < accuracy:
                cluster_max_acc[cluster_id] = accuracy
            
            # 保存簇的指标用于显示
            cluster_metrics[cluster_id] = {
                'loss': avg_loss,
                'accuracy': accuracy,
                'max_acc': cluster_max_acc[cluster_id]
            }
            
            # 将簇的全局模型提交给顶层服务器
            # 使用簇内客户端数量作为权重
            # top_level_server.rec(cluster_id, cluster_global_state, len(members), avg_loss)
        
        # 2. 簇间聚合
        # top_level_server.select_clients()
        # global_state_dict, global_avg_loss, _ = top_level_server.agg()
        
        # 测试全局模型性能
        # global_accuracy = top_level_server.test()
        # top_level_server.flush()
        
        # 记录全局结果
        # top_level_recorder.res['server']['iid_accuracy'].append(global_accuracy)
        # top_level_recorder.res['server']['train_loss'].append(global_avg_loss)
        
        # # 更新全局最大准确率
        # if top_level_max_acc < global_accuracy:
        #     top_level_max_acc = global_accuracy
        
        # 更新进度条显示所有指标
        display_str = f'Global Round: {global_round}'
        for cluster_id, metrics in cluster_metrics.items():
            display_str += f' | Cluster {cluster_id} - '
            display_str += f'Loss: {metrics["loss"]:.4f} '
            display_str += f'Acc: {metrics["accuracy"]:.4f} '
            display_str += f'Max: {metrics["max_acc"]:.4f}'
        # display_str += f' | Global - Loss: {global_avg_loss:.4f} Acc: {global_accuracy:.4f} Max: {top_level_max_acc:.4f}'
        pbar.set_description(display_str)
        
        # 保存结果
        if not os.path.exists(config["system"]["res_root"]):
            os.makedirs(config["system"]["res_root"])

        # 保存簇内结果
        for cluster_id, recorder in cluster_recorders.items():
            result_filename = os.path.join(
                config["system"]["res_root"], 
                f'[\'cluster_{cluster_id}_{config["client"]["fed_algo"]}\','
                f'\'cluster_{cluster_id}_{config["system"]["model"]}\',' 
                f'{config["client"]["num_local_epoch"]},'
                f'{config["system"]["num_local_class"]},'
                f'{config["system"]["i_seed"]}]'
            )
            with open(result_filename, "w") as jsfile:
                json.dump(recorder.res, jsfile, cls=PythonObjectEncoder)
        
        # 保存全局结果
        # global_result_filename = os.path.join(
        #     config["system"]["res_root"], 
        #     f'[\'global_{config["client"]["fed_algo"]}\','
        #     f'\'global_{config["system"]["model"]}\',' 
        #     f'{config["client"]["num_local_epoch"]},'
        #     f'{config["system"]["num_local_class"]},'
        #     f'{config["system"]["i_seed"]}]'
        # )
        # with open(global_result_filename, "w") as jsfile:
        #     json.dump(top_level_recorder.res, jsfile, cls=PythonObjectEncoder)

# class SimilarityCalculator(ABC):
#     """相似度计算的抽象基类"""
    
#     def __init__(self, n_clients, client_models, client_ids):
#         self.n_clients = n_clients
#         self.client_models = client_models
#         self.client_ids = client_ids
#         self.similarity_matrix = np.zeros((n_clients, n_clients))
    
#     @abstractmethod
#     def calculate_similarity(self, model1, model2):
#         """计算两个模型之间的相似度"""
#         pass
    
#     def compute_similarity_matrix(self):
#         """计算相似度矩阵"""
#         print(f"使用 {self.__class__.__name__} 计算相似度矩阵...")
#         for i in range(self.n_clients):
#             for j in range(i, self.n_clients):
#                 # 创建临时模型
#                 model_i = LeNet(num_classes=10, in_channels=1)
#                 model_j = LeNet(num_classes=10, in_channels=1)
                
#                 # 加载状态字典
#                 model_i.load_state_dict(self.client_models[self.client_ids[i]])
#                 model_j.load_state_dict(self.client_models[self.client_ids[j]])
                
#                 # 计算相似度
#                 similarity_score = self.calculate_similarity(model_i, model_j)
                
#                 # 填充相似度矩阵
#                 self.similarity_matrix[i, j] = similarity_score
#                 self.similarity_matrix[j, i] = similarity_score
        
#         return self.process_similarity_matrix()
    
#     def process_similarity_matrix(self):
#         """处理相似度矩阵，提高数值稳定性"""
#         # 确保对称性
#         matrix = (self.similarity_matrix + self.similarity_matrix.T) / 2
        
#         # 数值稳定性处理
#         matrix = np.nan_to_num(matrix, nan=0.0, posinf=1.0, neginf=0.0)
        
#         # 归一化到[0,1]区间
#         if matrix.max() != matrix.min():
#             matrix = (matrix - matrix.min()) / (matrix.max() - matrix.min())
        
#         # 应用高斯核
#         feature_ranges = np.ptp(matrix, axis=0)
#         adaptive_sigma = np.mean(feature_ranges) * 0.5
#         matrix = np.exp(-((1 - matrix) ** 2) / (2 * adaptive_sigma ** 2))
        
#         return matrix

# class JSDistanceSimilarity(SimilarityCalculator):
#     """基于JS散度的相似度计算"""
    
#     def calculate_similarity(self, model1, model2):
#         """
#         计算两个模型间的增强相似度
#         参数:
#         - model1, model2: 待比较的两个模型
#         返回:
#         - similarity_score: 相似度分数
#         - js_div: JS散度值
#         """
#         # 提取两个模型的权重分布
#         weights1 = self._preprocess_model_weights(model1)
#         weights2 = self._preprocess_model_weights(model2)

#         # 为不同层设置权重
#         layer_weights = {
#             'conv1.weight': 0,
#             'conv2.weight': 0,
#             'fc1.weight': 0,
#             'fc2.weight': 0,
#             'fc3.weight': 1
#         }
        
#         # 只计算fc3层的JS散度
#         total_js_div = 0
#         layer_js = self._js_divergence_stable(weights1['fc3.weight'], weights2['fc3.weight'])
#         total_js_div += layer_js
#         # for layer_name in weights1.keys():
#         #     if layer_name in layer_weights:
#         #         layer_js = _js_divergence_stable(weights1[layer_name], weights2[layer_name])
#         #         total_js_div += layer_weights[layer_name] * layer_js
#         similarity_score = 1 / (1 + total_js_div)
        
#         return similarity_score
    
       
    
#     def _preprocess_model_weights(self, model):
#         """
#         预处理模型权重，提取特征向量
#         参数:
#         - model: 待处理的模型
#         返回:
#         - 处理后的特征向量
#         """
#         weights_dict = {}
#         for name, param in model.named_parameters():
#             if 'weight' in name: # 只处理权重参数
#                 weights = param.detach().cpu().numpy().flatten()
#                 # 使用Sturges规则计算最优bin数量
#                 num_bins = int(np.ceil(np.log2(len(weights)) + 1))
                
#                 # 计算权重范围并添加边界保护
#                 w_min, w_max = np.min(weights), np.max(weights)
#                 margin = 0.1 * (w_max - w_min)
#                 w_min -= margin
#                 w_max += margin
                
#                 hist, _ = np.histogram(weights, bins=num_bins,
#                                      range=(w_min, w_max), density=True)
#                 hist = hist / np.sum(hist)
#                 weights_dict[name] = hist
        
#         return weights_dict
    
#     def _js_divergence_stable(self, p, q):
#         """
#         计算稳定的JS散度
#         参数:
#         - p, q: 两个概率分布
#         返回:
#         - js_div: JS散度值
#         """
#         epsilon = 1e-10

#         # 确保非负和归一化
#         p = np.maximum(p, epsilon)
#         q = np.maximum(q, epsilon)
        
#         p = p / np.sum(p)
#         q = q / np.sum(q)
        
#         # 计算中点分布
#         m = 0.5 * (p + q)

#         # 使用稳定的KL散度计算
#         kl_pm = np.sum(p * (np.log(p + epsilon) - np.log(m + epsilon)))
#         kl_qm = np.sum(q * (np.log(q + epsilon) - np.log(m + epsilon)))
        
#         return 0.5 * (kl_pm + kl_qm)

# class PearsonSimilarity(SimilarityCalculator):
#     """基于Pearson相关系数的相似度计算"""
    
#     def calculate_similarity(self, model1, model2):
#         # 提取fc3层权重
#         weights1 = model1.fc3.weight.detach().cpu().numpy().flatten()
#         weights2 = model2.fc3.weight.detach().cpu().numpy().flatten()
        
#         # 计算Pearson相关系数
#         correlation, _ = pearsonr(weights1, weights2)
#         # 将相关系数转换到[0,1]区间
#         similarity_score = (correlation + 1) / 2
        
#         return similarity_score

# class WassersteinSimilarity(SimilarityCalculator):
#     """基于Wasserstein距离的相似度计算"""
    
#     def calculate_similarity(self, model1, model2):
#         # 提取fc3层权重
#         weights1 = model1.fc3.weight.detach().cpu().numpy().flatten()
#         weights2 = model2.fc3.weight.detach().cpu().numpy().flatten()
        
#         # 计算Wasserstein距离
#         distance = wasserstein_distance(weights1, weights2)
#         # 将距离转换为相似度分数
#         similarity_score = 1 / (1 + distance)
        
#         return similarity_score

# def perform_spectral_clustering(similarity_matrix, n_clusters=3):
#     """执行谱聚类"""
#     spectral = SpectralClustering(
#         n_clusters=n_clusters,
#         affinity='precomputed',
#         random_state=42,
#         assign_labels='discretize',
#         n_init=20
#     )
#     return spectral.fit_predict(similarity_matrix)

# def visualize_clustering_results(similarity_matrix, cluster_labels, n_clusters, method_name):
#     """可视化聚类结果"""
#     plt.figure(figsize=(15, 5))
    
#     # PCA降维可视化
#     pca = PCA(n_components=2)
#     similarity_2d = pca.fit_transform(similarity_matrix)
    
#     plt.subplot(121)
#     scatter = plt.scatter(similarity_2d[:, 0], similarity_2d[:, 1], 
#                          c=cluster_labels, cmap='tab10')
#     plt.colorbar(scatter)
#     plt.title(f"Client clustering results ({method_name})")
#     plt.xlabel("1st principal component")
#     plt.ylabel("2nd principal component")
    
#     # 相似度分布箱线图
#     plt.subplot(122)
#     plt.boxplot([similarity_matrix[cluster_labels == i].flatten() 
#                  for i in range(n_clusters)],
#                 labels=[f'Cluster {i}' for i in range(n_clusters)])
#     plt.title(f"Similarity distribution within clusters ({method_name})")
#     plt.ylabel("Similarity score")
    
#     plt.tight_layout()
#     plt.show()

# # 在主程序中使用这些类
# def compare_clustering_methods(client_models, client_ids, n_clusters=3):
#     """比较不同的聚类方法"""
#     n_clients = len(client_ids)
#     methods = {
#         'JS Distance': JSDistanceSimilarity,
#         'Pearson Correlation': PearsonSimilarity,
#         'Wasserstein Distance': WassersteinSimilarity
#     }
    
#     results = {}
#     for method_name, calculator_class in methods.items():
#         print(f"\n使用 {method_name} 进行聚类...")
        
#         # 计算相似度矩阵
#         calculator = calculator_class(n_clients, client_models, client_ids)
#         similarity_matrix = calculator.compute_similarity_matrix()
        
#         # 执行谱聚类
#         cluster_labels = perform_spectral_clustering(similarity_matrix, n_clusters)
        
#         # 可视化结果
#         visualize_clustering_results(similarity_matrix, cluster_labels, n_clusters, method_name)
        
#         # 保存结果
#         results[method_name] = {
#             'similarity_matrix': similarity_matrix,
#             'cluster_labels': cluster_labels
#         }
    
#     return results

if __name__ == "__main__":
    fed_run()
