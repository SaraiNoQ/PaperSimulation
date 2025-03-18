import numpy as np
import torch
from utils.models import LeNet, AlexCifarNet
from abc import ABC, abstractmethod
from scipy.stats import pearsonr
from scipy.stats import wasserstein_distance
from matplotlib import pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from typing import Dict, List
from blockchain.consensus import Transaction


class SimilarityCalculator(ABC):
    """相似度计算的抽象基类"""
    
    def __init__(self, n_clients, client_models, client_ids):
        self.n_clients = n_clients
        self.client_models = client_models
        self.client_ids = client_ids
        self.similarity_matrix = np.zeros((n_clients, n_clients))
    
    @abstractmethod
    def calculate_similarity(self, model1, model2):
        """计算两个模型之间的相似度"""
        pass
    
    def compute_similarity_matrix(self):
        """计算相似度矩阵"""
        print(f"使用 {self.__class__.__name__} 计算相似度矩阵...")
        for i in range(self.n_clients):
            for j in range(i, self.n_clients):
                # 创建临时模型
                model_i = LeNet(num_classes=10, in_channels=1)
                model_j = LeNet(num_classes=10, in_channels=1)
                
                # 加载状态字典
                model_i.load_state_dict(self.client_models[self.client_ids[i]])
                model_j.load_state_dict(self.client_models[self.client_ids[j]])
                
                # 计算相似度
                similarity_score = self.calculate_similarity(model_i, model_j)
                
                # 填充相似度矩阵
                self.similarity_matrix[i, j] = similarity_score
                self.similarity_matrix[j, i] = similarity_score
        
        return self.process_similarity_matrix()
    
    def process_similarity_matrix(self):
        """处理相似度矩阵，提高数值稳定性"""
        # 确保对称性
        matrix = (self.similarity_matrix + self.similarity_matrix.T) / 2
        
        # 数值稳定性处理
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=1.0, neginf=0.0)
        
        # 归一化到[0,1]区间
        if matrix.max() != matrix.min():
            matrix = (matrix - matrix.min()) / (matrix.max() - matrix.min())
        
        # 应用高斯核
        feature_ranges = np.ptp(matrix, axis=0)
        adaptive_sigma = np.mean(feature_ranges) * 0.5
        matrix = np.exp(-((1 - matrix) ** 2) / (2 * adaptive_sigma ** 2))
        
        return matrix

class JSDistanceSimilarity(SimilarityCalculator):
    """基于JS散度的相似度计算"""
    
    def calculate_similarity(self, model1, model2):
        """
        计算两个模型间的增强相似度
        参数:
        - model1, model2: 待比较的两个模型
        返回:
        - similarity_score: 相似度分数
        - js_div: JS散度值
        """
        # 提取两个模型的权重分布
        weights1 = self._preprocess_model_weights(model1)
        weights2 = self._preprocess_model_weights(model2)

        # 为不同层设置权重
        layer_weights = {
            'conv1.weight': 0,
            'conv2.weight': 0,
            'fc1.weight': 0,
            'fc2.weight': 0,
            'fc3.weight': 1
        }
        
        # 只计算fc3层的JS散度
        total_js_div = 0
        layer_js = self._js_divergence_stable(weights1['fc3.weight'], weights2['fc3.weight'])
        total_js_div += layer_js
        # for layer_name in weights1.keys():
        #     if layer_name in layer_weights:
        #         layer_js = _js_divergence_stable(weights1[layer_name], weights2[layer_name])
        #         total_js_div += layer_weights[layer_name] * layer_js
        similarity_score = 1 / (1 + total_js_div)
        
        return similarity_score
    
       
    
    def _preprocess_model_weights(self, model):
        """
        预处理模型权重，提取特征向量
        参数:
        - model: 待处理的模型
        返回:
        - 处理后的特征向量
        """
        weights_dict = {}
        for name, param in model.named_parameters():
            if 'weight' in name: # 只处理权重参数
                weights = param.detach().cpu().numpy().flatten()
                # 使用Sturges规则计算最优bin数量
                num_bins = int(np.ceil(np.log2(len(weights)) + 1))
                
                # 计算权重范围并添加边界保护
                w_min, w_max = np.min(weights), np.max(weights)
                margin = 0.1 * (w_max - w_min)
                w_min -= margin
                w_max += margin
                
                hist, _ = np.histogram(weights, bins=num_bins,
                                     range=(w_min, w_max), density=True)
                hist = hist / np.sum(hist)
                weights_dict[name] = hist
        
        return weights_dict
    
    def _js_divergence_stable(self, p, q):
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
        
        return 0.5 * (kl_pm + kl_qm)

class PearsonSimilarity(SimilarityCalculator):
    """基于Pearson相关系数的相似度计算"""
    
    def calculate_similarity(self, model1, model2):
        # 提取fc3层权重
        weights1 = model1.fc3.weight.detach().cpu().numpy().flatten()
        weights2 = model2.fc3.weight.detach().cpu().numpy().flatten()
        
        # 计算Pearson相关系数
        correlation, _ = pearsonr(weights1, weights2)
        # 将相关系数转换到[0,1]区间
        similarity_score = (correlation + 1) / 2
        
        return similarity_score

class WassersteinSimilarity(SimilarityCalculator):
    """基于Wasserstein距离的相似度计算"""
    
    def calculate_similarity(self, model1, model2):
        # 提取fc3层权重
        weights1 = model1.fc3.weight.detach().cpu().numpy().flatten()
        weights2 = model2.fc3.weight.detach().cpu().numpy().flatten()
        
        # 计算Wasserstein距离
        distance = wasserstein_distance(weights1, weights2)
        # 将距离转换为相似度分数
        similarity_score = 1 / (1 + distance)
        
        return similarity_score

class AlexCifarNetSimilarity(SimilarityCalculator):
    """基于 AlexCifarNet 全连接层的相似度计算"""
    
    def __init__(self, n_clients, client_models, client_ids, target_layer='all'):
        """
        初始化相似度计算器
        
        Args:
            n_clients: 客户端数量
            client_models: 客户端模型字典
            client_ids: 客户端ID列表
            target_layer: 用于计算相似度的目标层
                         可选值: 'fc1'(4096->384), 'fc2'(384->192), 
                                'fc3'(192->10), 'all'(所有层加权)
        """
        super().__init__(n_clients, client_models, client_ids)
        self.target_layer = target_layer
        # 为不同层设置权重
        self.layer_weights = {
            'fc1': 0.2,  # 第一层权重较小，因为维度太大，可能包含较多噪声
            'fc2': 0.3,  # 中间层权重适中
            'fc3': 0.5   # 最后一层权重最大，因为更接近最终决策
        }
    
    def calculate_similarity(self, model1, model2):
        """计算两个模型间的相似度"""
        total_similarity = 0.0
        
        # 提取模型的全连接层权重
        weights1 = {
            'fc1': model1.classifier[0].weight.detach().cpu().numpy().flatten(),
            'fc2': model1.classifier[2].weight.detach().cpu().numpy().flatten(),
            'fc3': model1.classifier[4].weight.detach().cpu().numpy().flatten()
        }
        
        weights2 = {
            'fc1': model2.classifier[0].weight.detach().cpu().numpy().flatten(),
            'fc2': model2.classifier[2].weight.detach().cpu().numpy().flatten(),
            'fc3': model2.classifier[4].weight.detach().cpu().numpy().flatten()
        }
        
        if self.target_layer == 'all':
            # 使用加权平均计算总体相似度
            for layer in ['fc1', 'fc2', 'fc3']:
                # 计算 Wasserstein 距离
                distance = wasserstein_distance(weights1[layer], weights2[layer])
                # 将距离转换为相似度并加权
                similarity = 1 / (1 + distance)
                total_similarity += self.layer_weights[layer] * similarity
        else:
            # 只计算指定层的相似度
            distance = wasserstein_distance(
                weights1[self.target_layer], 
                weights2[self.target_layer]
            )
            total_similarity = 1 / (1 + distance)
        
        return total_similarity

    def compute_similarity_matrix(self):
        """计算相似度矩阵"""
        print(f"AlexCifarNet使用 {self.__class__.__name__} 计算相似度矩阵...")
        for i in range(self.n_clients):
            for j in range(i, self.n_clients):
                # 创建临时 AlexCifarNet 模型
                model_i = AlexCifarNet()
                model_j = AlexCifarNet()
                
                # 加载状态字典
                model_i.load_state_dict(self.client_models[self.client_ids[i]])
                model_j.load_state_dict(self.client_models[self.client_ids[j]])
                
                # 计算相似度
                similarity_score = self.calculate_similarity(model_i, model_j)
                
                # 填充相似度矩阵
                self.similarity_matrix[i, j] = similarity_score
                self.similarity_matrix[j, i] = similarity_score
                
                # 清理内存
                del model_i
                del model_j
                torch.cuda.empty_cache()
        
        return self.process_similarity_matrix()


def perform_spectral_clustering(similarity_matrix, n_clusters=3):
    """执行谱聚类"""
    spectral = SpectralClustering(
        n_clusters=n_clusters,
        affinity='precomputed',
        random_state=42,
        assign_labels='discretize',
        n_init=20
    )
    return spectral.fit_predict(similarity_matrix)

def visualize_clustering_results(similarity_matrix, cluster_labels, n_clusters, method_name):
    """可视化聚类结果"""
    plt.figure(figsize=(15, 5))
    
    # PCA降维可视化
    pca = PCA(n_components=2)
    similarity_2d = pca.fit_transform(similarity_matrix)
    
    plt.subplot(121)
    scatter = plt.scatter(similarity_2d[:, 0], similarity_2d[:, 1], 
                         c=cluster_labels, cmap='tab10')
    plt.colorbar(scatter)
    plt.title(f"Client clustering results ({method_name})")
    plt.xlabel("1st principal component")
    plt.ylabel("2nd principal component")
    
    # 相似度分布箱线图
    plt.subplot(122)
    plt.boxplot([similarity_matrix[cluster_labels == i].flatten() 
                 for i in range(n_clusters)],
                labels=[f'Cluster {i}' for i in range(n_clusters)])
    plt.title(f"Similarity distribution within clusters ({method_name})")
    plt.ylabel("Similarity score")
    
    plt.tight_layout()
    plt.show()

# 在主程序中使用这些类
def compare_clustering_methods(client_models, client_ids, n_clusters=3):
    """比较不同的聚类方法"""
    n_clients = len(client_ids)
    methods = {
        'JS Distance': JSDistanceSimilarity,
        'Pearson Correlation': PearsonSimilarity,
        'Wasserstein Distance': WassersteinSimilarity
    }
    
    results = {}
    for method_name, calculator_class in methods.items():
        print(f"\n使用 {method_name} 进行聚类...")
        
        # 计算相似度矩阵
        calculator = calculator_class(n_clients, client_models, client_ids)
        similarity_matrix = calculator.compute_similarity_matrix()
        
        # 执行谱聚类
        cluster_labels = perform_spectral_clustering(similarity_matrix, n_clusters)
        
        # 可视化结果
        visualize_clustering_results(similarity_matrix, cluster_labels, n_clusters, method_name)
        
        # 保存结果
        results[method_name] = {
            'similarity_matrix': similarity_matrix,
            'cluster_labels': cluster_labels
        }
    
    return results

def calculate_reputation_by_similarity(client_id: str, global_model_params: Dict[str, torch.Tensor], 
                       transactions: List[Transaction], old_reputation: float = 50.0) -> float:
    """使用 Wasserstein 距离计算客户端信誉值
    
    Args:
        client_id: 客户端ID
        global_model_params: 簇内全局模型参数
        transactions: 交易列表，用于获取客户端的模型参数
        old_reputation: 历史信誉值，默认为50.0
        
    Returns:
        float: 更新后的信誉值
    """
    # 从交易列表中找到对应客户端的交易
    client_tx = next((tx for tx in transactions if tx.client_id == client_id), None)
    if client_tx is None:
        return old_reputation
    
    client_model_params = client_tx.model_params
    
    # 确保所有张量在同一设备上
    device = next(iter(global_model_params.values())).device
    
    # 计算每一层的 Wasserstein 距离
    total_distance = 0
    num_layers = 0
    
    for key in global_model_params:
        if key in client_model_params:
            # 确保两个张量在同一设备上
            global_param = global_model_params[key].to(device)
            client_param = client_model_params[key].to(device)
            
            # 将参数展平并转换为numpy数组
            global_flat = global_param.cpu().detach().numpy().flatten()
            client_flat = client_param.cpu().detach().numpy().flatten()
            
            # 计算 Wasserstein 距离
            try:
                distance = wasserstein_distance(global_flat, client_flat)
                total_distance += distance
                num_layers += 1
            except Exception as e:
                print(f"警告: 计算层 {key} 的 Wasserstein 距离时出错: {str(e)}")
                continue
    
    # 计算平均距离
    avg_distance = total_distance / num_layers if num_layers > 0 else float('inf')
    
    # 将距离转换为得分（距离越小，得分越高）
    k = 2.0  # 衰减系数，可以根据需要调整
    score = 100 * np.exp(-k * avg_distance)
    
    # 更新信誉值
    alpha = 0.7  # 平滑因子，可以根据需要调整
    reputation = alpha * old_reputation + (1 - alpha) * score
    
    return float(reputation)