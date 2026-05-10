"""
train.py - 模型训练脚本
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

from model.action_model import DanceRecognitionModel, save_model
from utils.config import (
    EPOCHS, BATCH_SIZE, LEARNING_RATE, MODEL_DIR,
    KEYPOINTS_DIR,  # 添加这一行
    ACTION_LABELS, STYLE_LABELS
)
class Trainer:
    """训练器"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        
        # 双任务损失函数
        self.action_criterion = nn.CrossEntropyLoss()
        self.style_criterion = nn.CrossEntropyLoss()
        
        # 优化器
        self.optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
    
    def train_epoch(self, train_loader):
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0
        correct_action = 0
        correct_style = 0
        total = 0
        
        for X, y_action, y_style in tqdm(train_loader, desc="训练"):
            X = X.to(self.device)
            y_action = y_action.to(self.device)
            y_style = y_style.to(self.device)
            
            # 前向传播
            action_logits, style_logits = self.model(X)
            
            # 计算损失
            action_loss = self.action_criterion(action_logits, y_action)
            style_loss = self.style_criterion(style_logits, y_style)
            loss = action_loss + style_loss
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            _, action_pred = torch.max(action_logits, 1)
            _, style_pred = torch.max(style_logits, 1)
            total += y_action.size(0)
            correct_action += (action_pred == y_action).sum().item()
            correct_style += (style_pred == y_style).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        action_acc = correct_action / total
        style_acc = correct_style / total
        
        return avg_loss, action_acc, style_acc
    
    def validate(self, val_loader):
        """验证"""
        self.model.eval()
        total_loss = 0
        correct_action = 0
        correct_style = 0
        total = 0
        
        with torch.no_grad():
            for X, y_action, y_style in tqdm(val_loader, desc="验证"):
                X = X.to(self.device)
                y_action = y_action.to(self.device)
                y_style = y_style.to(self.device)
                
                action_logits, style_logits = self.model(X)
                
                action_loss = self.action_criterion(action_logits, y_action)
                style_loss = self.style_criterion(style_logits, y_style)
                loss = action_loss + style_loss
                
                total_loss += loss.item()
                _, action_pred = torch.max(action_logits, 1)
                _, style_pred = torch.max(style_logits, 1)
                total += y_action.size(0)
                correct_action += (action_pred == y_action).sum().item()
                correct_style += (style_pred == y_style).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        action_acc = correct_action / total
        style_acc = correct_style / total
        
        return avg_loss, action_acc, style_acc
    
    def train(self, train_loader, val_loader, epochs=EPOCHS):
        """完整训练流程"""
        best_val_acc = 0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # 训练
            train_loss, train_action_acc, train_style_acc = self.train_epoch(train_loader)
            
            # 验证
            val_loss, val_action_acc, val_style_acc = self.validate(val_loader)
            
            # 记录
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append((train_action_acc + train_style_acc) / 2)
            self.val_accs.append((val_action_acc + val_style_acc) / 2)
            
            # 打印
            print(f"训练 - Loss: {train_loss:.4f}, 动作准确率: {train_action_acc:.4f}, 风格准确率: {train_style_acc:.4f}")
            print(f"验证 - Loss: {val_loss:.4f}, 动作准确率: {val_action_acc:.4f}, 风格准确率: {val_style_acc:.4f}")
            
            # 学习率调度
            self.scheduler.step(val_loss)
            
            # 保存最佳模型
            avg_val_acc = (val_action_acc + val_style_acc) / 2
            if avg_val_acc > best_val_acc:
                best_val_acc = avg_val_acc
                save_model(self.model, os.path.join(MODEL_DIR, 'best_model.pth'))
                print(f"保存最佳模型，准确率: {best_val_acc:.4f}")
        
        # 绘制训练曲线
        self.plot_curves()
        
        return best_val_acc
    
    def plot_curves(self):
        """绘制训练曲线"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # 损失曲线
        axes[0].plot(self.train_losses, label='训练损失')
        axes[0].plot(self.val_losses, label='验证损失')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('训练曲线')
        axes[0].legend()
        axes[0].grid(True)
        
        # 准确率曲线
        axes[1].plot(self.train_accs, label='训练准确率')
        axes[1].plot(self.val_accs, label='验证准确率')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('准确率曲线')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(MODEL_DIR, 'training_curves.png'))
        plt.show()


def load_data(npz_path):
    """加载预处理的数据集"""
    data = np.load(npz_path)
    X = data['X']
    y_action = data['y_action']
    y_style = data['y_style']
    
    # 转换为 PyTorch Tensor
    X = torch.FloatTensor(X)
    y_action = torch.LongTensor(y_action)
    y_style = torch.LongTensor(y_style)
    
    return X, y_action, y_style


def main():
    """主训练函数"""
    # 加载数据
    dataset_path = os.path.join(KEYPOINTS_DIR, 'dataset.npz')
    if not os.path.exists(dataset_path):
        print(f"[ERROR] 数据集不存在: {dataset_path}")
        print("请先运行 DatasetBuilder 构建数据集")
        return
    
    X, y_action, y_style = load_data(dataset_path)
    
    # 划分训练集和验证集
    num_samples = len(X)
    num_train = int(0.8 * num_samples)
    indices = np.random.permutation(num_samples)
    train_idx = indices[:num_train]
    val_idx = indices[num_train:]
    
    X_train, y_action_train, y_style_train = X[train_idx], y_action[train_idx], y_style[train_idx]
    X_val, y_action_val, y_style_val = X[val_idx], y_action[val_idx], y_style[val_idx]
    
    # 创建 DataLoader
    train_dataset = TensorDataset(X_train, y_action_train, y_style_train)
    val_dataset = TensorDataset(X_val, y_action_val, y_style_val)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 创建模型
    from utils.config import ACTION_LABELS, STYLE_LABELS
    model = DanceRecognitionModel(
        num_actions=len(ACTION_LABELS),
        num_styles=len(STYLE_LABELS)
    )
    
    # 训练
    trainer = Trainer(model)
    best_acc = trainer.train(train_loader, val_loader)
    
    print(f"\n训练完成！最佳验证准确率: {best_acc:.4f}")


if __name__ == "__main__":
    main()