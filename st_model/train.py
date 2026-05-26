"""
train.py – 训练长视频风格识别模型
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# 添加项目根目录
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from st_model.config import (
    BATCH_SIZE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    LR_SCHEDULER_STEP, LR_SCHEDULER_GAMMA,
    PATIENCE, MODEL_SAVE_PATH, KEYPOINTS_DIR,
    NUM_STYLES
)
from st_model.dataset import build_dataloaders, get_style_labels
from st_model.stgcn_style import StyleSTGCN
from st_model.utils import save_training_curves, save_confusion_matrix


def train_one_epoch(model, loader, optimizer, criterion, device, epoch, total_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [训练]")
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = torch.argmax(logits, dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        
        pbar.set_postfix({'loss': f'{loss.item():.3f}', 'acc': f'{correct/total:.2f}'})
    
    return total_loss / len(loader), correct / total


def evaluate(model, loader, criterion, device, desc="验证"):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        pbar = tqdm(loader, desc=desc)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item()
            pred = torch.argmax(logits, dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return total_loss / len(loader), correct / total


def train_main():
    print("=" * 70)
    print("        长视频风格识别模型训练 (10秒片段)")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    npz_path = os.path.join(KEYPOINTS_DIR, "dataset_long.npz")
    if not os.path.exists(npz_path):
        print(f"错误: 未找到 {npz_path}")
        print("请先运行 st_model/extract_features.py 生成10秒片段骨架数据")
        return
    
    # 加载数据统计
    data = np.load(npz_path)
    print(f"数据集信息: {len(data['X'])} 样本, 序列长度: {data['X'].shape[1]} 帧")
    
    # 数据加载
    train_loader, val_loader, test_loader = build_dataloaders(npz_path, BATCH_SIZE)
    print(f"训练集: {len(train_loader.dataset)} 样本, {len(train_loader)} batches")
    print(f"验证集: {len(val_loader.dataset)} 样本")
    print(f"测试集: {len(test_loader.dataset)} 样本")
    
    model = StyleSTGCN().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = StepLR(optimizer, step_size=LR_SCHEDULER_STEP, gamma=LR_SCHEDULER_GAMMA)
    
    best_val_acc = 0.0
    patience_counter = 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, EPOCHS)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, desc=f"Epoch {epoch} 验证")
        scheduler.step()
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f"\nEpoch {epoch}/{EPOCHS}:")
        print(f"  训练 Loss: {train_loss:.4f}, Acc: {train_acc:.2%}")
        print(f"  验证 Loss: {val_loss:.4f}, Acc: {val_acc:.2%}")
        print(f"  学习率: {optimizer.param_groups[0]['lr']:.6f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  ✅ 保存最佳模型 (验证 Acc: {val_acc:.2%})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"早停触发！最佳验证准确率: {best_val_acc:.2%}")
                break
    
    # 测试
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device, desc="测试集评估")
    print(f"\n测试集结果: Loss={test_loss:.4f}, 风格准确率={test_acc:.2%}")
    
    # 保存曲线和混淆矩阵
    style_names = get_style_labels()
    save_training_curves(train_losses, val_losses, train_accs, val_accs, KEYPOINTS_DIR)
    save_confusion_matrix(model, test_loader, device, style_names, KEYPOINTS_DIR)
    
    print(f"\n训练完成！模型保存至: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train_main()