import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import numpy as np
from torchvision import transforms
from typing import Dict, Optional
import platform

class DistillationDataset(Dataset):
    """用于知识蒸馏的数据集包装器"""
    def __init__(self, original_dataset, teacher_model, device, temperature=2.0):
        self.dataset = original_dataset
        self.teacher_model = teacher_model
        self.device = device
        self.temperature = temperature
        self.teacher_model.eval()
        
    def __getitem__(self, index):
        data, target = self.dataset[index]
        with torch.no_grad():
            # 获取教师模型的软标签
            data_tensor = data.unsqueeze(0).to(self.device)
            teacher_logits = self.teacher_model(data_tensor).squeeze(0)
            soft_target = F.softmax(teacher_logits / self.temperature, dim=0)
        return data, target, soft_target.cpu()
    
    def __len__(self):
        return len(self.dataset)

class AdaptiveBatchSizer:
    """自适应批次大小管理器"""
    def __init__(self, init_batch_size: int, min_batch_size: int, max_batch_size: int,
                 growth_factor: float = 1.1, shrink_factor: float = 0.8):
        self.current_batch_size = init_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.growth_factor = growth_factor
        self.shrink_factor = shrink_factor
        self.loss_history = []
        self.window_size = 5
        
    def update(self, loss: float) -> int:
        """根据loss更新批次大小"""
        self.loss_history.append(loss)
        if len(self.loss_history) < self.window_size:
            return self.current_batch_size
            
        # 计算loss趋势
        recent_losses = self.loss_history[-self.window_size:]
        loss_trend = np.mean(np.diff(recent_losses))
        
        if loss_trend > 0:  # loss在增加，减小批次大小
            self.current_batch_size = max(
                self.min_batch_size,
                int(self.current_batch_size * self.shrink_factor)
            )
        else:  # loss在减小，增加批次大小
            self.current_batch_size = min(
                self.max_batch_size,
                int(self.current_batch_size * self.growth_factor)
            )
            
        # 保持最近的loss记录
        self.loss_history = self.loss_history[-self.window_size:]
        return self.current_batch_size

def get_dataloader_args():
    """获取根据操作系统优化的DataLoader参数"""
    if platform.system() == 'Windows':
        return {
            'num_workers': 2,
            'pin_memory': True,
            'multiprocessing_context': 'spawn'
        }
    else:  # Linux或MacOS
        return {
            'num_workers': 2,
            'pin_memory': True
        }

def train_cifar(model, trainset, device, epoch=3, batch_size=64, lr=0.03, momentum=0.9,
               teacher_model: Optional[nn.Module] = None):
    """
    优化的CIFAR10训练方法，适用于联邦学习场景
    改进：
    1. 自适应批次大小动态调整
    2. 知识蒸馏改善模型聚合
    3. 优化的损失函数组合
    4. 动态调整学习率和动量
    
    Args:
        model: 模型实例
        trainset: 训练数据集
        device: 训练设备
        epoch: 训练轮数
        batch_size: 初始批次大小
        lr: 初始学习率
        momentum: 动量参数
        teacher_model: 教师模型（上一轮的全局模型）
    """
    # 初始化自适应批次大小管理器
    batch_sizer = AdaptiveBatchSizer(
        init_batch_size=batch_size,
        min_batch_size=32,
        max_batch_size=128
    )
    
    # 如果有教师模型，使用知识蒸馏数据集
    if teacher_model is not None:
        teacher_model.to(device)
        trainset = DistillationDataset(trainset, teacher_model, device, temperature=2.0)
    
    # 优化器配置
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=5e-4,
        nesterov=True
    )
    
    # 使用CosineAnnealingWarmRestarts调度器
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=len(trainset) // batch_size,  # 第一次重启的周期
        T_mult=2,  # 每次重启后周期翻倍
        eta_min=lr * 0.01  # 最小学习率
    )
    
    # 损失函数
    ce_criterion = LabelSmoothingLoss(smoothing=0.1)
    kd_criterion = nn.KLDivLoss(reduction='batchmean')
    
    model.to(device)
    best_acc = 0
    best_state_dict = None
    total_loss = 0
    n_data = len(trainset)
    
    for epoch_idx in range(epoch):
        model.train()
        batch_loss = 0
        correct = 0
        total = 0
        current_batch_size = batch_sizer.current_batch_size
        
        # 使用get_dataloader_args获取适合当前操作系统的参数
        dataloader_args = get_dataloader_args()
        train_loader = DataLoader(
            trainset,
            batch_size=current_batch_size,
            shuffle=True,
            drop_last=True,
            **dataloader_args
        )
        
        for batch_idx, data in enumerate(train_loader):
            if teacher_model is not None:
                inputs, targets, teacher_outputs = data
                inputs, targets = inputs.to(device), targets.long().to(device)
                teacher_outputs = teacher_outputs.to(device)
            else:
                inputs, targets = data
                inputs, targets = inputs.to(device), targets.long().to(device)
            
            # 前向传播
            outputs = model(inputs)
            
            # 计算损失
            if teacher_model is not None:
                # 知识蒸馏损失
                T = 1.8  # 温度参数
                soft_targets = F.softmax(teacher_outputs / T, dim=1)
                soft_outputs = F.log_softmax(outputs / T, dim=1)
                
                # 结合硬标签和软标签的损失
                hard_loss = ce_criterion(outputs, targets)
                soft_loss = kd_criterion(soft_outputs, soft_targets) * (T * T)
                loss = 0.75 * hard_loss + 0.25 * soft_loss
            else:
                loss = ce_criterion(outputs, targets)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 参数更新
            optimizer.step()
            scheduler.step()
            
            # 更新批次大小
            current_batch_size = batch_sizer.update(loss.item())
            
            # 计算准确率
            batch_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # 打印训练信息
            if (batch_idx + 1) % 50 == 0:
                print(f'Epoch: {epoch_idx+1}/{epoch} | Batch: {batch_idx+1}/{len(train_loader)} | '
                      f'Loss: {loss.item():.4f} | Acc: {100.*correct/total:.2f}% | '
                      f'Batch Size: {current_batch_size} | LR: {scheduler.get_last_lr()[0]:.6f}')
        
        # 计算epoch的统计信息
        epoch_loss = batch_loss / len(train_loader)
        epoch_acc = correct / total
        total_loss += epoch_loss
        
        # 保存最佳模型
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_state_dict = {
                key: value.cpu() for key, value in model.state_dict().items()
            }
        
        print(f'Epoch: {epoch_idx+1}/{epoch} | Loss: {epoch_loss:.4f} | '
              f'Acc: {100.*epoch_acc:.2f}% | Best Acc: {100.*best_acc:.2f}% | '
              f'Final Batch Size: {current_batch_size}')
    
    # 使用最佳模型状态
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    
    avg_loss = total_loss / epoch
    return model.state_dict(), n_data, avg_loss

class LabelSmoothingLoss(nn.Module):
    """标签平滑损失函数"""
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        
    def forward(self, pred, target):
        pred = F.log_softmax(pred, dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (pred.size(-1) - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))

def evaluate_cifar(model, testset, device, batch_size=256):
    """
    评估模型在CIFAR10测试集上的性能
    
    Args:
        model: AlexCifarNet模型实例
        testset: 测试数据集
        device: 评估设备
        batch_size: 批次大小
    Returns:
        accuracy: 测试集上的准确率
        test_loss: 测试集上的平均损失
    """
    model.eval()
    dataloader_args = get_dataloader_args()
    test_loader = DataLoader(
        testset, 
        batch_size=batch_size, 
        shuffle=False,
        **dataloader_args
    )
    criterion = nn.CrossEntropyLoss()
    
    correct = 0
    total = 0
    test_loss = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.long().to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    accuracy = correct / total
    test_loss /= len(test_loader)
    
    return accuracy, test_loss 