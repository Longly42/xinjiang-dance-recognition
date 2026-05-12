"""
train.py – 训练双任务 ST-GCN，支持早停、学习率调度、模型保存和评估
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

from .config import (
    BATCH_SIZE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    LR_SCHEDULER_STEP, LR_SCHEDULER_GAMMA,
    PATIENCE, USE_CLASS_WEIGHTS, MODEL_SAVE_PATH,
    NUM_ACTIONS, NUM_STYLES, KEYPOINTS_DIR
)
from .dataset import build_dataloaders
from .stgcn import DualTaskSTGCN
from .utils import plot_confusion_matrix, plot_training_curves


def train_one_epoch(model, loader, optimizer, criterion_action, criterion_style, device):
    model.train()
    total_loss = 0
    total_action_loss = 0
    total_style_loss = 0
    action_acc = 0
    style_acc = 0
    count = 0

    for x, y_action, y_style in tqdm(loader, desc="Training", leave=False):
        x, y_action, y_style = x.to(device), y_action.to(device), y_style.to(device)
        optimizer.zero_grad()

        action_logits, style_logits = model(x)

        loss_a = criterion_action(action_logits, y_action)
        loss_s = criterion_style(style_logits, y_style)
        loss = loss_a + loss_s

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_action_loss += loss_a.item()
        total_style_loss += loss_s.item()

        pred_a = torch.argmax(action_logits, dim=1)
        pred_s = torch.argmax(style_logits, dim=1)
        action_acc += (pred_a == y_action).sum().item()
        style_acc += (pred_s == y_style).sum().item()
        count += x.size(0)

    return (total_loss / len(loader),
            total_action_loss / len(loader),
            total_style_loss / len(loader),
            action_acc / count,
            style_acc / count)


def evaluate(model, loader, criterion_action, criterion_style, device):
    model.eval()
    total_loss = 0
    action_acc = 0
    style_acc = 0
    count = 0

    with torch.no_grad():
        for x, y_action, y_style in tqdm(loader, desc="Evaluating", leave=False):
            x, y_action, y_style = x.to(device), y_action.to(device), y_style.to(device)
            action_logits, style_logits = model(x)
            loss = criterion_action(action_logits, y_action) + criterion_style(style_logits, y_style)
            total_loss += loss.item()

            pred_a = torch.argmax(action_logits, dim=1)
            pred_s = torch.argmax(style_logits, dim=1)
            action_acc += (pred_a == y_action).sum().item()
            style_acc += (pred_s == y_style).sum().item()
            count += x.size(0)

    return total_loss / len(loader), action_acc / count, style_acc / count


def get_class_weights(npz_path):
    """从数据集分布计算类别权重，用于损失函数"""
    data = np.load(npz_path)
    y_action = data['y_action']
    y_style = data['y_style']

    action_counts = np.bincount(y_action, minlength=NUM_ACTIONS)
    style_counts = np.bincount(y_style, minlength=NUM_STYLES)

    action_weights = 1.0 / (action_counts + 1e-6)
    action_weights = action_weights / action_weights.sum() * NUM_ACTIONS

    style_weights = 1.0 / (style_counts + 1e-6)
    style_weights = style_weights / style_weights.sum() * NUM_STYLES

    return torch.tensor(action_weights, dtype=torch.float), torch.tensor(style_weights, dtype=torch.float)


def train_main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    npz_path = os.path.join(KEYPOINTS_DIR, "dataset.npz")
    if not os.path.exists(npz_path):
        print(f"错误: 未找到数据集文件 {npz_path}")
        print("请先运行 scripts/extract_keypoints.py 生成骨架数据")
        return

    # 类别权重
    if USE_CLASS_WEIGHTS:
        action_weights, style_weights = get_class_weights(npz_path)
        action_weights = action_weights.to(device)
        style_weights = style_weights.to(device)
        criterion_action = nn.CrossEntropyLoss(weight=action_weights)
        criterion_style = nn.CrossEntropyLoss(weight=style_weights)
    else:
        criterion_action = nn.CrossEntropyLoss()
        criterion_style = nn.CrossEntropyLoss()

    # 数据加载
    train_loader, val_loader, test_loader = build_dataloaders(npz_path, BATCH_SIZE)

    model = DualTaskSTGCN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = StepLR(optimizer, step_size=LR_SCHEDULER_STEP, gamma=LR_SCHEDULER_GAMMA)

    best_val_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []
    train_acc_a, val_acc_a = [], []
    train_acc_s, val_acc_s = [], []

    for epoch in range(1, EPOCHS+1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        train_loss, train_loss_a, train_loss_s, train_acc_a_ep, train_acc_s_ep = train_one_epoch(
            model, train_loader, optimizer, criterion_action, criterion_style, device)
        val_loss, val_acc_a_ep, val_acc_s_ep = evaluate(model, val_loader, criterion_action, criterion_style, device)
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_acc_a.append(train_acc_a_ep)
        val_acc_a.append(val_acc_a_ep)
        train_acc_s.append(train_acc_s_ep)
        val_acc_s.append(val_acc_s_ep)

        print(f"Train Loss: {train_loss:.4f} (A:{train_loss_a:.4f} S:{train_loss_s:.4f}) | Acc A: {train_acc_a_ep:.4f} S: {train_acc_s_ep:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Acc A: {val_acc_a_ep:.4f} S: {val_acc_s_ep:.4f}")

        # 早停 & 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  -> 保存最佳模型到 {MODEL_SAVE_PATH}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"早停触发，停止训练")
                break

    # 加载最佳模型进行测试
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    test_loss, test_acc_a, test_acc_s = evaluate(model, test_loader, criterion_action, criterion_style, device)
    print(f"\n测试集结果: Loss={test_loss:.4f}, Action Acc={test_acc_a:.4f}, Style Acc={test_acc_s:.4f}")

    # 绘制混淆矩阵
    from .utils import save_confusion_matrices
    save_confusion_matrices(model, test_loader, device, save_dir=KEYPOINTS_DIR)

    # 绘制训练曲线
    plot_training_curves(train_losses, val_losses, train_acc_a, val_acc_a, train_acc_s, val_acc_s, save_dir=KEYPOINTS_DIR)

    print("训练完成！")


if __name__ == "__main__":
    train_main()