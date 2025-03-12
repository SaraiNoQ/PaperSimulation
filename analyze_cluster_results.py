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
        try:
            # 从文件名中提取簇ID
            filename = os.path.basename(file_path)
            # 解析文件名格式 ['cluster_X_FedAvg','cluster_X_LeNet',...]
            parts = filename.strip('[]\'').split(',')
            if len(parts) >= 1:
                cluster_part = parts[0].strip('\'')
                cluster_id = cluster_part.split('_')[1]  # 提取数字部分
                
                # 加载结果
                with open(file_path, 'r') as f:
                    results = json.load(f)
                    # 确保数据格式正确
                    if isinstance(results, dict) and 'server' in results:
                        cluster_results[cluster_id] = results
                    else:
                        print(f"Warning: File {filename} has incorrect data format, skipped")
        except Exception as e:
            print(f"Warning: Error loading file {filename}: {str(e)}")
    
    return cluster_results

def plot_accuracy_curves(cluster_results, dataset, num_clients, similarity_method, imbalance_factor, seed, save_dir=None):
    """Plot accuracy curves for all clusters"""
    plt.figure(figsize=(12, 6))
    
    # Set color scheme
    colors = sns.color_palette("husl", len(cluster_results))
    
    # Create legend elements list
    legend_elements = []
    
    # Plot accuracy curves for each cluster
    for (cluster_id, results), color in zip(cluster_results.items(), colors):
        try:
            accuracy_history = results['server']['iid_accuracy']
            # Ensure accuracy_history is a numeric list
            accuracy_history = [float(acc) for acc in accuracy_history]
            rounds = list(range(1, len(accuracy_history) + 1))
            
            # Plot accuracy curve
            line = plt.plot(rounds, accuracy_history, 
                    color=color,
                    marker='o',
                    markersize=3,
                    linewidth=2)[0]
            
            # Annotate final accuracy
            final_acc = accuracy_history[-1]
            plt.annotate(f'{final_acc:.4f}', 
                        xy=(len(accuracy_history), final_acc),
                        xytext=(5, 5),
                        textcoords='offset points')
            
            # Add to legend
            legend_elements.append((line, f'Cluster {cluster_id}'))
        except Exception as e:
            print(f"Warning: Error plotting accuracy curve for cluster {cluster_id}: {str(e)}")
    
    # Set plot properties
    plt.title('Federated Learning Accuracy Curves by Cluster', fontsize=14)
    plt.xlabel('Training Rounds', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Create custom legend
    if legend_elements:
        legend = plt.legend(
            [item[0] for item in legend_elements],
            [item[1] for item in legend_elements],
            loc='lower left',
            bbox_to_anchor=(0.02, 0.02),
            title='Cluster Index',
            title_fontsize=12,
            frameon=True,
            facecolor='white',
            edgecolor='gray',
            framealpha=0.8
        )
    
    # Use integer ticks for X axis
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Save plot
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(
            save_dir,
            f'accuracy_curves_{dataset}_{num_clients}_{similarity_method}_{imbalance_factor}_{seed}.png'
        )
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()

def plot_loss_curves(cluster_results, dataset, num_clients, similarity_method, imbalance_factor, seed, save_dir=None):
    """Plot loss curves for all clusters"""
    plt.figure(figsize=(12, 6))
    
    # Set color scheme
    colors = sns.color_palette("husl", len(cluster_results))
    
    # Create legend elements list
    legend_elements = []
    
    # Plot loss curves for each cluster
    for (cluster_id, results), color in zip(cluster_results.items(), colors):
        try:
            loss_history = results['server']['train_loss']
            # Ensure loss_history is a numeric list
            loss_history = [float(loss) for loss in loss_history]
            rounds = list(range(1, len(loss_history) + 1))
            
            # Plot loss curve
            line = plt.plot(rounds, loss_history, 
                    color=color,
                    marker='o',
                    markersize=3,
                    linewidth=2)[0]
            
            # Annotate final loss
            final_loss = loss_history[-1]
            plt.annotate(f'{final_loss:.4f}', 
                        xy=(len(loss_history), final_loss),
                        xytext=(5, 5),
                        textcoords='offset points')
            
            # Add to legend
            legend_elements.append((line, f'Cluster {cluster_id}'))
        except Exception as e:
            print(f"Warning: Error plotting loss curve for cluster {cluster_id}: {str(e)}")
    
    # Set plot properties
    plt.title('Federated Learning Loss Curves by Cluster', fontsize=14)
    plt.xlabel('Training Rounds', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Create custom legend
    if legend_elements:
        legend = plt.legend(
            [item[0] for item in legend_elements],
            [item[1] for item in legend_elements],
            loc='upper left',
            bbox_to_anchor=(0.02, 0.98),
            title='Cluster Index',
            title_fontsize=12,
            frameon=True,
            facecolor='white',
            edgecolor='gray',
            framealpha=0.8
        )
    
    # Use integer ticks for X axis
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Save plot
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(
            save_dir,
            f'loss_curves_{dataset}_{num_clients}_{similarity_method}_{imbalance_factor}_{seed}.png'
        )
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()

def analyze_cluster_performance(cluster_results):
    """Analyze performance metrics for each cluster"""
    performance_stats = {}
    
    for cluster_id, results in cluster_results.items():
        try:
            accuracy_history = [float(acc) for acc in results['server']['iid_accuracy']]
            loss_history = [float(loss) for loss in results['server']['train_loss']]
            
            stats = {
                'Final Accuracy': accuracy_history[-1],
                'Best Accuracy': max(accuracy_history),
                'Average Accuracy': np.mean(accuracy_history),
                'Final Loss': loss_history[-1],
                'Minimum Loss': min(loss_history),
                'Average Loss': np.mean(loss_history),
                'Convergence Rounds': len(accuracy_history)
            }
            
            performance_stats[cluster_id] = stats
        except Exception as e:
            print(f"Warning: Error analyzing metrics for cluster {cluster_id}: {str(e)}")
    
    # Print performance statistics
    if performance_stats:
        print("\nCluster Performance Statistics:")
        print("-" * 50)
        for cluster_id, stats in performance_stats.items():
            print(f"\nCluster {cluster_id}:")
            for metric, value in stats.items():
                print(f"{metric}: {value:.4f}")
    else:
        print("\nWarning: No performance statistics available")
    
    return performance_stats

def main():
    # Set results directory path
    res_root = "results"  # Modify according to actual path
    
    # Load clustering results to get metadata
    clustering_results_file = None
    for file in os.listdir(res_root):
        if file.startswith("clustering_results_"):
            clustering_results_file = os.path.join(res_root, file)
            break
    
    if not clustering_results_file:
        print("Error: Could not find clustering results file!")
        return
        
    # Extract metadata from filename
    filename = os.path.basename(clustering_results_file)
    # 从clustering_results文件名中提取参数
    parts = filename.replace("clustering_results_", "").replace(".json", "").split("_")
    try:
        dataset = parts[0]  # 数据集名称
        num_clients = parts[1]  # 客户端数量
        similarity_method = parts[2]  # 相似度方法
        imbalance_factor = parts[3]  # 不平衡因子
        seed = parts[4]  # 随机种子
    except IndexError:
        print("Warning: Could not parse all metadata from filename. Using default values.")
        dataset = "Unknown"
        num_clients = "Unknown"
        similarity_method = "Unknown"
        imbalance_factor = "Unknown"
        seed = "Unknown"
    
    # Load cluster results
    cluster_results = load_cluster_results(res_root)
    
    if not cluster_results:
        print("No cluster training result files found!")
        return
    
    try:
        # Plot accuracy curves
        plot_accuracy_curves(
            cluster_results,
            dataset,
            num_clients,
            similarity_method,
            imbalance_factor,
            seed,
            save_dir=os.path.join(res_root, "plots")
        )
        
        # Plot loss curves
        plot_loss_curves(
            cluster_results,
            dataset,
            num_clients,
            similarity_method,
            imbalance_factor,
            seed,
            save_dir=os.path.join(res_root, "plots")
        )
        
        # Analyze performance
        analyze_cluster_performance(cluster_results)
    except Exception as e:
        print(f"Error: Exception occurred during execution: {str(e)}")

if __name__ == "__main__":
    main() 