import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from postprocessing.recorder import Recorder

def load_and_plot_results(res_root):
    """使用Recorder加载并绘制所有结果"""
    # 创建记录器实例
    recorder = Recorder()
    
    # 获取所有结果文件
    cluster_files = glob.glob(os.path.join(res_root, "[\'cluster_*"))
    global_files = glob.glob(os.path.join(res_root, "[\'global_*"))
    
    # 加载簇结果
    for file_path in cluster_files:
        try:
            # 从文件名中提取簇ID和模型信息
            filename = os.path.basename(file_path)
            parts = filename.strip('[]\'').split(',')
            if len(parts) >= 1:
                cluster_part = parts[0].strip('\'')
                # 使用完整的cluster_part作为标签
                recorder.load(file_path, cluster_part)
        except Exception as e:
            print(f"Warning: Error loading cluster file {filename}: {str(e)}")
    
    # 加载全局结果
    for file_path in global_files:
        try:
            filename = os.path.basename(file_path)
            parts = filename.strip('[]\'').split(',')
            if len(parts) >= 1:
                global_part = parts[0].strip('\'')
                recorder.load(file_path, global_part)
        except Exception as e:
            print(f"Warning: Error loading global file {filename}: {str(e)}")
    
    # 绘制结果
    recorder.plot()
    
    # 保存图表
    save_dir = os.path.join(res_root, "plots")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {os.path.join(save_dir, 'training_curves.png')}")
    
    return recorder

def main():
    # 设置结果目录路径
    res_root = "results"  # 根据实际路径修改
    
    try:
        # 加载并绘制结果
        recorder = load_and_plot_results(res_root)
        plt.show()
    except Exception as e:
        print(f"Error: Exception occurred during execution: {str(e)}")

if __name__ == "__main__":
    main() 