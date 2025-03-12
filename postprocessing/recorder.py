# 导入必要的库
import numpy as np  # 用于数值计算
import matplotlib.pyplot as plt  # 用于绘图
import json  # 用于JSON文件处理
from json import JSONEncoder  # 导入JSON编码器
import pickle  # 用于Python对象序列化

# 定义可以直接被JSON序列化的Python类型
json_types = (list, dict, str, int, float, bool, type(None))

class PythonObjectEncoder(JSONEncoder):
    """
    自定义JSON编码器，用于处理非标准JSON类型的Python对象
    通过pickle序列化来处理复杂的Python对象
    """
    def default(self, obj):
        if isinstance(obj, json_types):
            return super().default(self, obj)
        return {'_python_object': pickle.dumps(obj).decode('latin-1')}
    
def as_python_object(dct):
    """
    JSON解码器的辅助函数
    将之前序列化的Python对象重新转换回来
    """
    if '_python_object' in dct:
        return pickle.loads(dct['_python_object'].encode('latin-1'))
    return dct

class Recorder(object):
    """
    记录器类，用于记录和可视化训练结果
    """
    def __init__(self):
        """
        初始化记录器
        创建存储服务器和客户端训练结果的数据结构
        """
        self.res_list = []  # 存储所有结果的列表
        self.res = {
            'server': {'iid_accuracy': [], 'train_loss': []},  # 服务器端的准确率和损失
            'clients': {'iid_accuracy': [], 'train_loss': []}  # 客户端的准确率和损失
        }

    def load(self, filename, label):
        """
        加载结果文件的方法
        参数:
        filename: 结果文件的名称
        label: 为该结果指定的标签
        """
        with open(filename) as json_file:
            res = json.load(json_file, object_hook=as_python_object)
        self.res_list.append((res, label))

    def plot(self):
        """
        绘制测试准确率和训练损失的方法
        创建两个子图：上面显示准确率，下面显示损失
        """
        # 创建包含两个子图的图表
        fig, axes = plt.subplots(2)
        
        # 遍历所有结果并绘制
        for i, (res, label) in enumerate(self.res_list):
            # 绘制准确率
            axes[0].plot(np.array(res['server']['iid_accuracy']), 
                        label=label, alpha=1, linewidth=2)
            # 绘制损失
            axes[1].plot(np.array(res['server']['train_loss']), 
                        label=label, alpha=1, linewidth=2)

        # 设置每个子图的属性
        for i, ax in enumerate(axes):
            ax.set_xlabel('# of Epochs', size=12)  # 设置x轴标签
            if i == 0:
                ax.set_ylabel('Testing Accuracy', size=12)  # 设置准确率y轴标签
            if i == 1:
                ax.set_ylabel('Training Loss', size=12)  # 设置损失y轴标签
            ax.legend(prop={'size': 12})  # 添加图例
            ax.tick_params(axis='both', labelsize=12)  # 设置刻度标签大小
            ax.grid()  # 添加网格