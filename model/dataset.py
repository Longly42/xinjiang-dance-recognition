"""
dataset.py – 加载 .npz 格式的骨架数据集，并实现数据增强
"""

import numpy as np
import json
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, Dict

from .config import (
    KEYPOINTS_DIR, SEQUENCE_LENGTH, NUM_KEYPOINTS, KEYPOINT_FEATURE_DIM,
    USE_AUGMENTATION, AUG_NOISE_STD, AUG_DROP_FRAME_RATIO, AUG_TIME_MASK
)


class SkeletonDataset(Dataset):
    """骨架序列数据集，支持动作+风格双标签"""

    def __init__(self, npz_path: str, transform=None):
        """
        npz_path: dataset.npz 文件路径，包含 X, y_action, y_style
        transform: 数据增强函数（可选）
        """
        data = np.load(npz_path)
        self.X = data['X']          # (N, T, V, C)
        self.y_action = data['y_action']
        self.y_style = data['y_style']

        # 验证形状
        assert self.X.shape[1] == SEQUENCE_LENGTH, \
            f"序列长度 {self.X.shape[1]} 不等于配置的 {SEQUENCE_LENGTH}"
        assert self.X.shape[2] == NUM_KEYPOINTS, \
            f"关键点数量 {self.X.shape[2]} 不等于 {NUM_KEYPOINTS}"
        assert self.X.shape[3] == KEYPOINT_FEATURE_DIM, \
            f"特征维度 {self.X.shape[3]} 不等于 {KEYPOINT_FEATURE_DIM}"

        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].copy()          # (T, V, C)
        if self.transform:
            x = self.transform(x)
        # 转换为 PyTorch 张量，形状 (C, T, V) 供 ST-GCN 使用
        x = torch.from_numpy(x).permute(2, 0, 1).float()  # (C, T, V)
        return x, self.y_action[idx], self.y_style[idx]


def get_data_augmentation():
    """返回数据增强函数（可应用于单个样本）"""
    def augment(sequence: np.ndarray) -> np.ndarray:
        # sequence: (T, V, C)
        # 1. 高斯噪声
        if AUG_NOISE_STD > 0:
            noise = np.random.normal(0, AUG_NOISE_STD, sequence.shape)
            sequence = sequence + noise

        # 2. 随机丢弃帧（用线性插值替换）
        if AUG_DROP_FRAME_RATIO > 0 and np.random.rand() < 0.5:
            T = sequence.shape[0]
            drop_idx = np.random.choice(T, size=int(T * AUG_DROP_FRAME_RATIO), replace=False)
            for idx in drop_idx:
                # 用前后帧平均替换
                if idx == 0:
                    sequence[idx] = sequence[1]
                elif idx == T-1:
                    sequence[idx] = sequence[-2]
                else:
                    sequence[idx] = (sequence[idx-1] + sequence[idx+1]) / 2

        # 3. 时间掩码（随机将连续若干帧置零）
        if AUG_TIME_MASK and np.random.rand() < 0.3:
            mask_len = np.random.randint(1, max(2, SEQUENCE_LENGTH // 8))
            start = np.random.randint(0, SEQUENCE_LENGTH - mask_len)
            sequence[start:start+mask_len] = 0

        return sequence

    return augment


def build_dataloaders(npz_path: str, batch_size: int, val_ratio=0.2, test_ratio=0.1):
    """划分训练/验证/测试集，返回 DataLoader"""
    from sklearn.model_selection import train_test_split

    dataset = SkeletonDataset(npz_path, transform=get_data_augmentation())

    # 简单随机划分（实际可按视频分层划分，这里简化）
    indices = list(range(len(dataset)))
    train_val_idx, test_idx = train_test_split(indices, test_size=test_ratio, random_state=42)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=val_ratio/(1-test_ratio), random_state=42)

    train_set = torch.utils.data.Subset(dataset, train_idx)
    val_set = torch.utils.data.Subset(dataset, val_idx)
    test_set = torch.utils.data.Subset(dataset, test_idx)

    # 注意：验证/测试集不应使用数据增强
    val_set.dataset.transform = None
    test_set.dataset.transform = None

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader