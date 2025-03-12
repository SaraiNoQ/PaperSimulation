import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import entropy, wasserstein_distance # 导入 JS 散度计算相关的库 (虽然此代码示例中没有直接计算JS散度，但后续计算会用到)
import random


class LeNet(nn.Module):
    supported_dims = {28}

    def __init__(self, num_classes=10, in_channels=1):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        out = F.relu(self.conv1(x), inplace=True)  # 6 x 28 x 28  
        out = F.max_pool2d(out, 2)  # 6 x 14 x 14  
        out = F.relu(self.conv2(out), inplace=True)  # 16 x 7 x 7  
        out = F.max_pool2d(out, 2)  # 16 x 5 x 5
        out = out.view(out.size(0), -1)  # 16 x 5 x 5  
        out = F.relu(self.fc1(out), inplace=True)
        out = F.relu(self.fc2(out), inplace=True)
        out = self.fc3(out)

        return out

def smooth_distribution(p, epsilon=1e-10):
    """
    平滑概率分布，避免出现零概率
    参数:
    - p: 输入的概率分布
    - epsilon: 平滑因子
    """
    p = np.array(p)
    p = p + epsilon
    return p / np.sum(p)

def calculate_js_divergence_improved(p, q):
    """
    改进的JS散度计算方法
    
    参数:
    - p, q: 两个概率分布
    返回:
    - js_div: JS散度值
    """
    # 1. 数值稳定性处理
    p = smooth_distribution(p)
    q = smooth_distribution(q)
    
    # 2. 计算中间分布M
    m = 0.5 * (p + q)
    
    # 3. 使用改进的KL散度计算
    kl_p_m = np.sum(p * np.log2(p / m))
    kl_q_m = np.sum(q * np.log2(q / m))
    
    # 4. 计算JS散度
    js_div = 0.5 * (kl_p_m + kl_q_m)
    
    return js_div

def preprocess_fc3_with_improved_histogram(model, num_bins=100):
    """
    改进的fc3层权重预处理方法
    
    参数:
    - model: 模型
    - num_bins: 直方图bin数量
    返回:
    - 处理后的权重分布
    """
    # 1. 提取fc3层权重
    weights = model.fc3.weight.detach().cpu().numpy().flatten()
    
    # 2. 使用Sturges规则计算最优bin数量
    if num_bins is None:
        num_bins = int(np.ceil(np.log2(len(weights)) + 1))
    
    # 3. 计算权重的范围
    w_min, w_max = np.min(weights), np.max(weights)
    
    # 4. 添加边界保护
    margin = 0.1 * (w_max - w_min)
    w_min -= margin
    w_max += margin
    
    # 5. 计算改进的直方图
    hist, _ = np.histogram(weights, 
                          bins=num_bins, 
                          range=(w_min, w_max), 
                          density=True)
    
    # 6. 归一化处理
    hist = hist / np.sum(hist)
    
    return hist

def calculate_model_similarity(model1, model2, num_bins=100):
    """
    计算两个模型的相似度
    
    参数:
    - model1, model2: 待比较的两个模型
    - num_bins: 直方图bin数量
    返回:
    - similarity_score: 相似度分数
    """
    # 1. 获取改进的权重分布
    dist1 = preprocess_fc3_with_improved_histogram(model1, num_bins)
    dist2 = preprocess_fc3_with_improved_histogram(model2, num_bins)
    
    # 2. 计算改进的JS散度
    js_div = calculate_js_divergence_improved(dist1, dist2)
    
    # 3. 将JS散度转换为相似度分数（0-1之间）
    similarity_score = 1 / (1 + js_div)
    
    return similarity_score, js_div

def initialize_model_same():
    """案例一：创建两个 LeNet 模型，使用相同的初始化参数 (设置随机种子)"""
    seed = random.randint(1, 10000) # 设置一个固定的随机种子
    torch.manual_seed(seed)
    model1_same = LeNet(num_classes=10, in_channels=1)
    torch.manual_seed(seed) # 再次设置相同的随机种子，确保 model2 使用相同的初始化
    model2_same = LeNet(num_classes=10, in_channels=1)
    return model1_same, model2_same

def initialize_model_different():
    """案例二：创建两个 LeNet 模型，使用不同的随机初始化参数 (不设置或使用不同的随机种子)"""
    model1_diff = LeNet(num_classes=10, in_channels=1) # 默认随机初始化
    torch.manual_seed(random.randint(1, 10000)) # 设置一个不同的随机种子给 model2，确保参数不同
    model2_diff = LeNet(num_classes=10, in_channels=1)
    return model1_diff, model2_diff

# 示例用法
if __name__ == '__main__':
    # 创建测试模型
    model1_same, model2_same = initialize_model_same()
    model1_diff, model2_diff = initialize_model_different()
    
    print("改进后的模型相似度计算结果：")
    print("-" * 50)
    
    # 测试相同初始化的模型
    sim_score_same, js_div_same = calculate_model_similarity(model1_same, model2_same)
    print("相同初始化模型：")
    print(f"相似度分数: {sim_score_same:.6f}")
    print(f"JS散度: {js_div_same:.6f}")
    print("-" * 30)
    
    # 测试不同初始化的模型
    sim_score_diff, js_div_diff = calculate_model_similarity(model1_diff, model2_diff)
    print("不同初始化模型：")
    print(f"相似度分数: {sim_score_diff:.6f}")
    print(f"JS散度: {js_div_diff:.6f}")

