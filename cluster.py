import numpy as np
import matplotlib.pyplot as plt
import yaml
from sklearn.cluster import SpectralClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
import argparse
from sklearn.cluster import KMeans
import time

def get_config():
    """获取配置参数"""
    parser = argparse.ArgumentParser(description='聚类分析程序')
    parser.add_argument('--config', type=str, required=True, 
                       help='配置文件路径 (yaml格式)')
    args = parser.parse_args()

    # 读取yaml配置文件
    with open(args.config, 'r', encoding='utf-8') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(f"错误：配置文件读取失败 - {e}")
            exit(1)
    
    # 验证配置参数
    if 'n_clients' not in config:
        print("错误：配置文件中缺少 'n_clients' 参数")
        exit(1)
    
    return config

def main(config):
    n_clients = config['n_clients']
    print(f"config setting: n_clients = {n_clients}")
    
    # ======================
    # 1. 生成模拟数据（增强鲁棒性）
    # ======================
    np.random.seed(config['clustering']['random_seed'])

    # 生成模型特征（添加微小噪声防止零方差）
    n_features = 5
    
    # 根据客户端数量调整每个簇的大小
    cluster_sizes = [
        int(n_clients * 0.3),  # 30% 的客户端
        int(n_clients * 0.4),  # 40% 的客户端
        n_clients - int(n_clients * 0.3) - int(n_clients * 0.4)  # 剩余客户端
    ]
    
    model_features, _ = make_blobs(n_samples=n_clients,
                                   n_features=n_features,
                                   centers=3,
                                   cluster_std=1.5)

    # 添加高斯噪声（防止特征完全一致）
    model_features += np.random.normal(0, 0.01, model_features.shape)

    # 生成网络延迟（确保无无效值）
    network_delay = np.concatenate([
        np.abs(np.random.normal(loc=50, scale=10, size=cluster_sizes[0])),  # 确保正值
        np.abs(np.random.normal(loc=150, scale=30, size=cluster_sizes[1])),
        np.abs(np.random.normal(loc=300, scale=50, size=cluster_sizes[2])),
    ])

    # ======================
    # 2. 谱聚类（使用JS散度）
    # ======================
    def preprocess_features(features):
         """预处理特征向量"""
         # 标准化到非负值
         features_min = features.min(axis=1, keepdims=True)
         features = features - features_min
         
         # L1归一化，确保和为1
         features_sum = features.sum(axis=1, keepdims=True)
         features_sum[features_sum == 0] = 1  # 防止除零
         features = features / features_sum
         
         return features
    
    def js_divergence_stable(p, q):
         """计算稳定的JS散度"""
         # 数值稳定性参数
         epsilon = 1e-10
         
         # 确保非负和归一化
         p = np.maximum(p, epsilon)
         q = np.maximum(q, epsilon)
         
         # 计算中点分布
         m = 0.5 * (p + q)
         
         # 使用稳定的KL散度计算
         kl_pm = np.sum(p * (np.log(p + epsilon) - np.log(m + epsilon)))
         kl_qm = np.sum(q * (np.log(q + epsilon) - np.log(m + epsilon)))
         
         # 返回JS散度
         return 0.5 * (kl_pm + kl_qm)
    
    start_time = time.time()

    # 预处理特征矩阵
    processed_features = preprocess_features(model_features)

    # 计算相似度矩阵
    n = len(model_features)
    similarity_matrix = np.zeros((n, n))

    # 计算特征的范围用于自适应sigma
    feature_ranges = np.ptp(processed_features, axis=0)
    adaptive_sigma = np.mean(feature_ranges) * 0.5  # 自适应核宽度

    # 对每对样本计算JS散度
    for i in range(n):
        for j in range(i, n):
            # 获取两个样本的特征
            p = processed_features[i]
            q = processed_features[j]

            # 计算JS散度
            js_dist = js_divergence_stable(p, q)

            # 将JS散度转换为相似度（使用高斯核）
            similarity = np.exp(-js_dist / (2 * adaptive_sigma ** 2))

            # 由于相似度矩阵是对称的
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity

     # 归一化相似度矩阵
    row_sums = similarity_matrix.sum(axis=1, keepdims=True)
    similarity_matrix = similarity_matrix / row_sums

    # 确保相似度矩阵的数值稳定性
    similarity_matrix = np.nan_to_num(similarity_matrix, nan=0.0, posinf=1.0, neginf=0.0)

    # 执行谱聚类（增加鲁棒性参数）
    spectral = SpectralClustering(n_clusters=config['clustering']['n_coarse_clusters'],
                                 affinity='precomputed',
                                 random_state=config['clustering']['random_seed'],
                                 assign_labels='discretize')  # 避免浮点精度问题
    coarse_labels = spectral.fit_predict(similarity_matrix)

    cluster_time = time.time() - start_time
    print(f"谱聚类 运算时间: {cluster_time:.4f} 秒")
    
    # ======================
    # 3. DBSCAN细分（优化参数）
    # ======================
    final_labels = np.zeros_like(coarse_labels)
    current_max_label = 0

    for cluster_id in np.unique(coarse_labels):
        mask = (coarse_labels == cluster_id)
        delay_subset = network_delay[mask].reshape(-1, 1)
        subset_size = len(delay_subset)
        
        print(f"\n处理粗分簇 {cluster_id}:")
        print(f"用户数量: {subset_size}")
        
        # 根据用户数量决定是否进行细分
        if subset_size < 100:
            print("用户数量少于100，不进行细分")
            final_labels[mask] = current_max_label
            current_max_label += 1
            continue
            
        # 计算需要细分的组数
        n_subclusters = max(2, subset_size // 100)
        print(f"用户数量为{subset_size}，将细分为{n_subclusters}组")
        
        # 使用StandardScaler进行数据标准化
        scaler = StandardScaler()
        delay_scaled = scaler.fit_transform(delay_subset)
        
        # 增加特征维度：添加原始延迟值的差分作为第二维度
        delay_diff = np.diff(delay_subset, axis=0)
        delay_diff = np.vstack([delay_diff, delay_diff[-1]])
        delay_diff_scaled = scaler.fit_transform(delay_diff)
        
        # 组合特征
        combined_features = np.hstack([delay_scaled, delay_diff_scaled])
        
        # 使用K-means进行预聚类，获取初始中心点

        kmeans = KMeans(n_clusters=n_subclusters, random_state=config['clustering']['random_seed'])
        kmeans_labels = kmeans.fit_predict(delay_scaled)
        
        # 计算每个类的平均距离作为eps
        eps_list = []
        for i in range(n_subclusters):
            cluster_points = combined_features[kmeans_labels == i]
            if len(cluster_points) > 1:
                # 计算类内平均距离
                from scipy.spatial.distance import pdist
                distances = pdist(cluster_points)
                eps_list.append(np.mean(distances))
        
        # 使用平均eps
        eps = np.mean(eps_list) if eps_list else 0.5
        
        # 设置DBSCAN参数
        min_samples = max(3, min(20, int(np.log2(subset_size) * 2)))
        
        print(f"DBSCAN参数: eps={eps:.4f}, min_samples={min_samples}")
        
        # 执行DBSCAN聚类
        db = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
        sub_labels = db.fit_predict(combined_features)
        
        # 处理聚类结果
        unique_labels = np.unique(sub_labels)
        n_clusters_found = len(unique_labels[unique_labels != -1])
        n_noise = np.sum(sub_labels == -1)
        
        print(f"实际获得的细分簇数量: {n_clusters_found}")
        print(f"噪声点数量: {n_noise}")
        
        # 如果没有找到足够的簇，使用K-means结果
        if n_clusters_found < n_subclusters:
            print(f"DBSCAN未能产生足够的簇，使用K-means结果")
            sub_labels = kmeans_labels
        
        # 更新标签
        sub_labels[sub_labels != -1] += current_max_label
        final_labels[mask] = sub_labels
        current_max_label = np.max(final_labels) + 1

    # ======================
    # 4. 可视化（添加异常值提示）
    # ======================
    plt.figure(figsize=(15, 6))

    # （原有可视化代码保持不变，此处省略...）

    # 原始数据分布
    plt.subplot(131)
    plt.scatter(model_features[:, 0], model_features[:, 1],
                c=network_delay, cmap='viridis', alpha=0.6)
    plt.title("Original Data Distribution\n(Color=Network Delay)")
    plt.xlabel("Feature Dimension 1")
    plt.ylabel("Feature Dimension 2")
    plt.colorbar()

    # 谱聚类结果
    plt.subplot(132)
    for cl in np.unique(coarse_labels):
        mask = (coarse_labels == cl)
        plt.scatter(model_features[mask, 0], model_features[mask, 1],
                    label=f'Coarse cluster {cl}', alpha=0.6)
    plt.title("After Spectral Clustering\n(3 Coarse Clusters)")
    plt.xlabel("Feature Dimension 1")
    plt.ylabel("Feature Dimension 2")
    plt.legend()

    # DBSCAN细分结果
    plt.subplot(133)
    unique_labels = np.unique(final_labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask = (final_labels == label)
        plt.scatter(model_features[mask, 0], model_features[mask, 1],
                    color=colors[i],
                    label=f'Sub-Cluster {i}' if label != -1 else 'Noise',
                    alpha=0.6)

    plt.title("After DBSCAN Refinement\n(Color=Sub-Clusters)")
    plt.xlabel("Feature Dimension 1")
    plt.ylabel("Feature Dimension 2")
    plt.legend()

    plt.tight_layout()
    plt.show()

    # ======================
    # 附加分析：网络延迟分布可视化优化
    # ======================
    plt.figure(figsize=(15, 10))

    # 1. 箱线图比较
    plt.subplot(221)
    box_data = [network_delay[coarse_labels == cl] for cl in np.unique(coarse_labels)]
    plt.boxplot(box_data, labels=[f'Coarse Cluster {cl}' for cl in np.unique(coarse_labels)])
    plt.title("Network delay distribution of each coarse cluster (box plot)")
    plt.ylabel("Network delay (ms)")
    plt.grid(True, alpha=0.3)

    # 2. 核密度估计
    plt.subplot(222)
    for cl in np.unique(coarse_labels):
        mask = coarse_labels == cl
        plt.hist(network_delay[mask], bins=30, density=True, alpha=0.3,
                 label=f'Coarse cluster {cl}')
        # 添加核密度估计曲线
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(network_delay[mask])
        x_range = np.linspace(network_delay.min(), network_delay.max(), 200)
        plt.plot(x_range, kde(x_range), label=f'KDE cluster{cl}')
    plt.title("Network delay distribution (kernel density estimation)")
    plt.xlabel("Network delay (ms)")
    plt.ylabel("density")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 3. 细分簇的箱线图
    plt.subplot(223)
    # 获取所有标签，包括噪声点（-1）
    all_labels = np.unique(final_labels)
    box_data_refined = [network_delay[final_labels == label] for label in all_labels]
    labels = [f'Noise' if label == -1 else f'sub {label}' for label in all_labels]
    
    # 绘制箱线图
    bp = plt.boxplot(box_data_refined, labels=labels)
    
    # 为噪声点的箱线图添加特殊颜色
    if -1 in all_labels:
        noise_idx = np.where(all_labels == -1)[0][0]
        plt.setp(bp['boxes'][noise_idx], color='red', alpha=0.5)
        plt.setp(bp['medians'][noise_idx], color='red')
    
    plt.title("Network delay distribution of each sub cluster (box plot)")
    plt.ylabel("Network delay (ms)")
    plt.xticks(rotation=45)  # 旋转标签以防重叠
    plt.grid(True, alpha=0.3)

    # 4. 散点图展示
    plt.subplot(224)
    scatter = plt.scatter(range(len(network_delay)), network_delay, 
                        c=final_labels, cmap='tab20', 
                        alpha=0.6)
    plt.colorbar(scatter, label='sub cluster label')
    plt.title("Network delay scatter distribution")
    plt.xlabel("Sample Index")
    plt.ylabel("Network delay (ms)")
    plt.grid(True, alpha=0.3)

    # 调整子图布局，确保标签不重叠
    plt.tight_layout(pad=3.0)
    plt.show()

    # 添加统计信息输出
    print("\n=== 聚类统计信息 ===")
    print("\n粗分簇统计：")
    for cl in np.unique(coarse_labels):
        delay_stats = network_delay[coarse_labels == cl]
        print(f"\n簇 {cl}:")
        print(f"样本数量: {len(delay_stats)}")
        print(f"平均延迟: {delay_stats.mean():.2f} ms")
        print(f"标准差: {delay_stats.std():.2f} ms")
        print(f"延迟范围: [{delay_stats.min():.2f}, {delay_stats.max():.2f}] ms")

    print("\n细分簇统计：")
    for label in np.unique(final_labels[final_labels != -1]):
        delay_stats = network_delay[final_labels == label]
        print(f"\n簇 {label}:")
        print(f"样本数量: {len(delay_stats)}")
        print(f"平均延迟: {delay_stats.mean():.2f} ms")
        print(f"标准差: {delay_stats.std():.2f} ms")
        print(f"延迟范围: [{delay_stats.min():.2f}, {delay_stats.max():.2f}] ms")

if __name__ == "__main__":
    config = get_config()
    main(config)