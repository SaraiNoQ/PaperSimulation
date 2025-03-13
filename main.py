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
    top_level_server = FedServer(
        list(clusters.keys()),  # 使用簇ID作为客户端ID
        dataset_id=config["system"]["dataset"],
        model_name=config["system"]["model"]
    )
    top_level_server.load_testset(testset)
    top_level_recorder = Recorder()
    top_level_max_acc = 0
    
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
            top_level_server.rec(cluster_id, cluster_global_state, len(members), avg_loss)
        
        # 2. 簇间聚合
        top_level_server.select_clients()
        global_state_dict, global_avg_loss, _ = top_level_server.agg()
        
        # 测试全局模型性能
        global_accuracy = top_level_server.test()
        top_level_server.flush()
        
        # 记录全局结果
        top_level_recorder.res['server']['iid_accuracy'].append(global_accuracy)
        top_level_recorder.res['server']['train_loss'].append(global_avg_loss)
        
        # # 更新全局最大准确率
        if top_level_max_acc < global_accuracy:
            top_level_max_acc = global_accuracy
        
        # 更新进度条显示所有指标
        display_str = f'Global Round: {global_round}'
        for cluster_id, metrics in cluster_metrics.items():
            display_str += f' | Cluster {cluster_id} - '
            display_str += f'Loss: {metrics["loss"]:.4f} '
            display_str += f'Acc: {metrics["accuracy"]:.4f} '
            display_str += f'Max: {metrics["max_acc"]:.4f}'
        display_str += f' | Global - Loss: {global_avg_loss:.4f} Acc: {global_accuracy:.4f} Max: {top_level_max_acc:.4f}'
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
        global_result_filename = os.path.join(
            config["system"]["res_root"], 
            f'[\'global_{config["client"]["fed_algo"]}\','
            f'\'global_{config["system"]["model"]}\',' 
            f'{config["client"]["num_local_epoch"]},'
            f'{config["system"]["num_local_class"]},'
            f'{config["system"]["i_seed"]}]'
        )
        with open(global_result_filename, "w") as jsfile:
            json.dump(top_level_recorder.res, jsfile, cls=PythonObjectEncoder)

if __name__ == "__main__":
    fed_run()
