from utils.models import *
from utils.fed_utils import assign_dataset, init_model
from torch.utils.data import DataLoader
import torch
from utils.js_cal import calculate_model_similarity
from tqdm import tqdm

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
        针对 AlexCifarNet 模型优化的训练方法，专门用于 CIFAR10 数据集
        """
        # 1. 数据加载优化
        train_loader = DataLoader(
            self.trainset, 
            batch_size=256,          # 增大批次大小到256
            shuffle=True,
            num_workers=4,           # 增加工作进程数
            pin_memory=True,         # 使用锁页内存
            prefetch_factor=2,       # 预加载因子
            persistent_workers=True   # 保持工作进程存活
        )

        # 2. 模型设置
        self.model = self.model.to(self._device)
        scaler = torch.cuda.amp.GradScaler()  # 添加自动混合精度训练
        
        # 3. 优化器设置 - 使用 SGD with momentum，这对 AlexNet 系列模型更有效
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=0.1,                # 提高初始学习率
            momentum=0.9,
            weight_decay=5e-4,
            nesterov=True
        )
        
        # 4. 学习率调度器
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=0.1,
            epochs=self._epoch,
            steps_per_epoch=len(train_loader),
            pct_start=0.2,         # 前20%时间提升学习率
            div_factor=10,         # 初始学习率为max_lr/10
            final_div_factor=100   # 最终学习率为初始学习率/100
        )
        
        # 5. 损失函数
        criterion = nn.CrossEntropyLoss().to(self._device)
        
        # 6. 训练过程
        total_loss = 0
        for epoch in range(self._epoch):
            epoch_loss = 0
            correct = 0
            total = 0
            
            # 设置进度条
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self._epoch}')
            
            self.model.train()
            for inputs, targets in pbar:
                # 数据移动到GPU
                inputs = inputs.to(self._device, non_blocking=True)
                targets = targets.to(self._device, non_blocking=True)
                
                # 使用自动混合精度训练
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)
                
                # 优化器步骤
                optimizer.zero_grad(set_to_none=True)  # 更高效的梯度清零
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                # 更新学习率
                scheduler.step()
                
                # 统计
                epoch_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # 更新进度条
                pbar.set_postfix({
                    'Loss': f'{loss.item():.3f}',
                    'Acc': f'{100.*correct/total:.2f}%',
                    'LR': f'{scheduler.get_last_lr()[0]:.6f}'
                })
            
            # 计算平均损失
            avg_epoch_loss = epoch_loss / len(train_loader)
            total_loss = avg_epoch_loss
            
            # 打印epoch结果
            print(f'\nEpoch {epoch+1} Summary:')
            print(f'Loss: {avg_epoch_loss:.3f}')
            print(f'Accuracy: {100.*correct/total:.2f}%')
            print(f'Learning Rate: {scheduler.get_last_lr()[0]:.6f}')
            
            # 同步GPU
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        
        return self.model.state_dict(), self.n_data, total_loss