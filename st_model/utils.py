"""
utils.py – 训练曲线和混淆矩阵绘制（中文支持）
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
from sklearn.metrics import confusion_matrix

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def save_training_curves(train_losses, val_losses, train_accs, val_accs, save_dir):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('损失曲线')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='训练准确率')
    plt.plot(val_accs, label='验证准确率')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('准确率曲线')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'style_training_curves.png'), dpi=150)
    plt.close()
    print(f"训练曲线已保存: {save_dir}/style_training_curves.png")


def save_confusion_matrix(model, dataloader, device, class_names, save_dir):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('风格分类混淆矩阵')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'style_confusion_matrix.png'), dpi=150)
    plt.close()
    print(f"混淆矩阵已保存: {save_dir}/style_confusion_matrix.png")