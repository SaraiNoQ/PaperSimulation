import numpy as np
from utils.models import LeNet
from abc import ABC, abstractmethod
from scipy.stats import pearsonr
from scipy.stats import wasserstein_distance
from matplotlib import pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA


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