"""
utils.py – 混淆矩阵、训练曲线绘图等工具
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os


def plot_confusion_matrix(y_true, y_pred, labels, title, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_confusion_matrices(model, dataloader, device, save_dir):
    model.eval()
    all_action_true = []
    all_action_pred = []
    all_style_true = []
    all_style_pred = []

    with torch.no_grad():
        for x, y_action, y_style in dataloader:
            x = x.to(device)
            action_logits, style_logits = model(x)
            action_pred = torch.argmax(action_logits, dim=1)
            style_pred = torch.argmax(style_logits, dim=1)
            all_action_true.extend(y_action.cpu().numpy())
            all_action_pred.extend(action_pred.cpu().numpy())
            all_style_true.extend(y_style.cpu().numpy())
            all_style_pred.extend(style_pred.cpu().numpy())

    # 加载标签名（需要从全局获取，这里简单使用数字标签）
    from .config import NUM_ACTIONS, NUM_STYLES
    action_labels = [f"动作{i}" for i in range(NUM_ACTIONS)]
    style_labels = [f"风格{i}" for i in range(NUM_STYLES)]

    plot_confusion_matrix(all_action_true, all_action_pred, action_labels,
                          "动作分类混淆矩阵", os.path.join(save_dir, "confusion_matrix_action.png"))
    plot_confusion_matrix(all_style_true, all_style_pred, style_labels,
                          "风格分类混淆矩阵", os.path.join(save_dir, "confusion_matrix_style.png"))


def plot_training_curves(train_loss, val_loss, train_acc_a, val_acc_a, train_acc_s, val_acc_s, save_dir):
    epochs = range(1, len(train_loss)+1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, 'b-', label='训练 Loss')
    plt.plot(epochs, val_loss, 'r-', label='验证 Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss 曲线')

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc_a, 'b-', label='训练动作准确率')
    plt.plot(epochs, val_acc_a, 'r-', label='验证动作准确率')
    plt.plot(epochs, train_acc_s, 'g--', label='训练风格准确率')
    plt.plot(epochs, val_acc_s, 'm--', label='验证风格准确率')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('准确率曲线')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curves.png"))
    plt.close()