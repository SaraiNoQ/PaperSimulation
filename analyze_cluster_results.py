import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns

def load_cluster_results(res_root):
    """加载所有簇的训练结果"""
    # 获取所有簇的结果文件
    cluster_files = glob.glob(os.path.join(res_root, "[\'cluster_*"))
    cluster_results = {}
    
    for file_path in cluster_files:
        # 从文件名中提取簇ID
        filename = os.path.basename(file_path)
        cluster_id = filename.split('_')[1]
        
        # 加载结果
        with open(file_path, 'r') as f:
            results = json.load(f)
            cluster_results[cluster_id] = results
    
    return cluster_results

def plot_accuracy_curves(cluster_results, save_path=None):
    """绘制每个簇的准确率变化曲线"""
    plt.figure(figsize=(12, 6))
    
    # 设置颜色方案
    colors = sns.color_palette("husl", len(cluster_results))
    
    # 绘制每个簇的准确率曲线
    for (cluster_id, results), color in zip(cluster_results.items(), colors):
        accuracy_history = results['server']['iid_accuracy']
        rounds = range(1, len(accuracy_history) + 1)
        
        # 绘制准确率曲线
        plt.plot(rounds, accuracy_history, 
                label=f'簇 {cluster_id}',
                color=color,
                marker='o',
                markersize=3,
                linewidth=2)
        
        # 标注最终准确率
        final_acc = accuracy_history[-1]
        plt.annotate(f'{final_acc:.4f}', 
                    xy=(len(accuracy_history), final_acc),
                    xytext=(5, 5),
                    textcoords='offset points')
    
    # 设置图表属性
    plt.title('各簇联邦学习准确率变化曲线', fontsize=14)
    plt.xlabel('训练轮次', fontsize=12)
    plt.ylabel('准确率', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower right')
    
    # X轴使用整数刻度
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # 添加网格
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # 保存图表
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存至: {save_path}")
    
    plt.show()

def plot_loss_curves(cluster_results, save_path=None):
    """绘制每个簇的损失值变化曲线"""
    plt.figure(figsize=(12, 6))
    
    # 设置颜色方案
    colors = sns.color_palette("husl", len(cluster_results))
    
    # 绘制每个簇的损失曲线
    for (cluster_id, results), color in zip(cluster_results.items(), colors):
        loss_history = results['server']['train_loss']
        rounds = range(1, len(loss_history) + 1)
        
        # 绘制损失曲线
        plt.plot(rounds, loss_history, 
                label=f'簇 {cluster_id}',
                color=color,
                marker='o',
                markersize=3,
                linewidth=2)
        
        # 标注最终损失值
        final_loss = loss_history[-1]
        plt.annotate(f'{final_loss:.4f}', 
                    xy=(len(loss_history), final_loss),
                    xytext=(5, 5),
                    textcoords='offset points')
    
    # 设置图表属性
    plt.title('各簇联邦学习损失值变化曲线', fontsize=14)
    plt.xlabel('训练轮次', fontsize=12)
    plt.ylabel('损失值', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper right')
    
    # X轴使用整数刻度
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # 添加网格
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # 保存图表
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存至: {save_path}")
    
    plt.show()

def analyze_cluster_performance(cluster_results):
    """分析每个簇的性能指标"""
    performance_stats = {}
    
    for cluster_id, results in cluster_results.items():
        accuracy_history = results['server']['iid_accuracy']
        loss_history = results['server']['train_loss']
        
        stats = {
            '最终准确率': accuracy_history[-1],
            '最高准确率': max(accuracy_history),
            '平均准确率': np.mean(accuracy_history),
            '最终损失值': loss_history[-1],
            '最低损失值': min(loss_history),
            '平均损失值': np.mean(loss_history),
            '收敛轮次': len(accuracy_history)
        }
        
        performance_stats[cluster_id] = stats
    
    # 打印性能统计
    print("\n各簇性能统计:")
    print("-" * 50)
    for cluster_id, stats in performance_stats.items():
        print(f"\n簇 {cluster_id}:")
        for metric, value in stats.items():
            print(f"{metric}: {value:.4f}")
    
    return performance_stats

def main():
    # 设置结果文件夹路径
    res_root = "results"  # 根据实际情况修改路径
    
    # 加载结果
    cluster_results = load_cluster_results(res_root)
    
    if not cluster_results:
        print("未找到簇训练结果文件！")
        return
    
    # 创建保存图表的文件夹
    plots_dir = os.path.join(res_root, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # 绘制准确率曲线
    acc_plot_path = os.path.join(plots_dir, "cluster_accuracy_curves.png")
    plot_accuracy_curves(cluster_results, acc_plot_path)
    
    # 绘制损失值曲线
    loss_plot_path = os.path.join(plots_dir, "cluster_loss_curves.png")
    plot_loss_curves(cluster_results, loss_plot_path)
    
    # 分析性能
    analyze_cluster_performance(cluster_results)

if __name__ == "__main__":
    main() 