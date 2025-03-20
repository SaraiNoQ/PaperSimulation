from utils.models import *
from utils.fed_utils import assign_dataset, init_model
from torch.utils.data import DataLoader
import torch
from utils.js_cal import calculate_model_similarity
from tqdm import tqdm
import sys
import torch.nn.functional as F

class FedClient(object):
    def __init__(self, name, epoch, dataset_id, model_name):
        """
        Initialize the client k for federated learning.
        :param name: Name of the client k
        :param epoch: Number of local training epochs in the client k
        :param dataset_id: Local dataset in the client k
        :param model_name: Local model in the client k
        """
        # Initialize the metadata in the local client
        self.target_ip = '127.0.0.3'
        self.port = 9999
        self.name = name

        # Initialize the parameters in the local client
        self._epoch = epoch
        self._batch_size = 50
        self._lr = 0.01
        self._momentum = 0.9
        self.num_workers = 2
        self.loss_rec = []
        self.n_data = 0

        # Initialize the local training and testing dataset
        self.trainset = None
        self.test_data = None

        # Initialize the local model
        self._num_class, self._image_dim, self._image_channel = assign_dataset(dataset_id)
        self.model_name = model_name
        self.model = init_model(model_name=self.model_name, num_class=self._num_class, image_channel=self._image_channel)
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        self.param_len = sum([np.prod(p.size()) for p in model_parameters])

        # Training on GPU
        gpu = 0
        self._device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() and gpu != -1 else "cpu")

    def load_trainset(self, trainset):
        """
        Client loads the training dataset.
        :param trainset: Dataset for training.
        """
        self.trainset = trainset
        self.n_data = len(trainset)

    def update(self, model_state_dict):
        """
        Client updates the model from the server.
        :param model_state_dict: Global model.
        """
        self.model = init_model(model_name=self.model_name, num_class=self._num_class, image_channel=self._image_channel)
        self.model.load_state_dict(model_state_dict)

    # def train(self):
    #     """
    #     Client trains the model on local dataset
    #     :return: Local updated model, number of local data points, training loss
    #     """
    #     # 1. 数据加载
    #     train_loader = DataLoader(self.trainset, batch_size=self._batch_size, shuffle=True)

    #     # 2. 模型与优化器设置
    #     self.model.to(self._device)
        
    #     # 设置初始学习率和优化器
    #     initial_lr = 0.01
    #     optimizer = torch.optim.AdamW(
    #         self.model.parameters(),
    #         lr=initial_lr,
    #         betas=(0.9, 0.999),
    #         eps=1e-8,
    #         weight_decay=0.02
    #     )
        
    #     # 设置学习率调度器
    #     scheduler = torch.optim.lr_scheduler.StepLR(
    #         optimizer,
    #         step_size=5,
    #         gamma=0.9
    #     )
        
    #     # 设置损失函数
    #     loss_func = nn.CrossEntropyLoss()

    #     # 3. 训练过程
    #     total_loss = 0
    #     for epoch in range(self._epoch):
    #         epoch_loss = 0
    #         for step, (x, y) in enumerate(train_loader):
    #             # 数据搬运到设备
    #             b_x = x.to(self._device)
    #             b_y = y.to(self._device)

    #             # 前向传播 + 反向传播
    #             self.model.train()
    #             optimizer.zero_grad()
                
    #             output = self.model(b_x)
    #             loss = loss_func(output, b_y.long())
                
    #             loss.backward()
    #             torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # 梯度裁剪
                
    #             optimizer.step()
    #             epoch_loss += loss.item()
            
    #         # 更新学习率
    #         scheduler.step()
            
    #         # 计算平均损失
    #         avg_epoch_loss = epoch_loss / len(train_loader)
    #         total_loss = avg_epoch_loss  # 保存最后一个epoch的损失
            
    #         # 打印训练信息（可选）
    #         if (epoch + 1) % 3 == 0:  # 每3个epoch打印一次
    #             current_lr = optimizer.param_groups[0]['lr']
    #             print(f"Client {self.name} - Epoch [{epoch+1}/{self._epoch}], "
    #                   f"Loss: {avg_epoch_loss:.4f}, "
    #                   f"Learning Rate: {current_lr:.6f}")

    #     # 返回模型参数和训练信息
    #     return self.model.state_dict(), self.n_data, total_loss
    
    def train(self):
        """
        Client trains the model on local dataset
        :return: Local updated model, number of local data points, training loss
        """

        # 1. 数据加载
        train_loader = DataLoader(self.trainset, batch_size=self._batch_size, shuffle=True)

        # 2. 模型与优化器设置
        self.model.to(self._device)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self._lr, momentum=self._momentum)
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=self._lr, weight_decay=1e-4)

        loss_func = nn.CrossEntropyLoss()

        # Training process
        for epoch in range(self._epoch):
            for step, (x, y) in enumerate(train_loader):
                with torch.no_grad():
                    # 数据搬运到设备
                    b_x = x.to(self._device)  # Tensor on GPU
                    b_y = y.to(self._device)  # Tensor on GPU

                with torch.enable_grad():
                    # 前向传播 + 反向传播
                    self.model.train()
                    output = self.model(b_x)
                    loss = loss_func(output, b_y.long())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

        # 返回模型的权重参数（Weights），该字典包含模型所有可学习参数（如全连接层权重、卷积核参数等）的当前值
        return self.model.state_dict(), self.n_data, loss.data.cpu().numpy()