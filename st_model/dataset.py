"""
dataset.py – 加载长视频骨架数据集（每个样本是一个10秒片段）
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import json
from sklearn.model_selection import train_test_split

from .config import (
    SEQUENCE_LENGTH, NUM_KEYPOINTS, KEYPOINT_FEATURE_DIM,
    USE_AUGMENTATION, AUG_NOISE_STD, AUG_DROP_FRAME_RATIO, AUG_TIME_MASK,
    KEYPOINTS_DIR, LABEL_MAP_PATH
)


class LongVideoDataset(Dataset):
    """
    长视频风格数据集
    每个样本形状: (T, V, C)，其中 T = SEQUENCE_LENGTH（固定采样）
    """
    def __init__(self, npz_path, transform=None):
        data = np.load(npz_path)
        self.X = data['X']          # (N, T, V, C)
        self.y_style = data['y_style']
        
        # 验证形状
        assert self.X.shape[1] == SEQUENCE_LENGTH, \
            f"序列长度 {self.X.shape[1]} != {SEQUENCE_LENGTH}"
        
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].copy()
        if self.transform:
            x = self.transform(x)
        # 转换为 (C, T, V) 供 ST-GCN 使用
        x = torch.from_numpy(x).permute(2, 0, 1).float()
        y = torch.tensor(self.y_style[idx], dtype=torch.long)
        return x, y


def get_data_augmentation():
    """数据增强（与原来类似，但针对长序列）"""
    def augment(sequence):
        # sequence: (T, V, C)
        if AUG_NOISE_STD > 0:
            noise = np.random.normal(0, AUG_NOISE_STD, sequence.shape)
            sequence = sequence + noise
        
        if AUG_DROP_FRAME_RATIO > 0 and np.random.rand() < 0.5:
            T = sequence.shape[0]
            drop_idx = np.random.choice(T, size=int(T * AUG_DROP_FRAME_RATIO), replace=False)
            for idx in drop_idx:
                if idx == 0:
                    sequence[idx] = sequence[1]
                elif idx == T-1:
                    sequence[idx] = sequence[-2]
                else:
                    sequence[idx] = (sequence[idx-1] + sequence[idx+1]) / 2
        
        if AUG_TIME_MASK and np.random.rand() < 0.3:
            mask_len = np.random.randint(1, max(2, SEQUENCE_LENGTH // 8))
            start = np.random.randint(0, SEQUENCE_LENGTH - mask_len)
            sequence[start:start+mask_len] = 0
        
        return sequence
    return augment


def build_dataloaders(npz_path, batch_size, val_ratio=0.2, test_ratio=0.1):
    """划分训练/验证/测试集，返回 DataLoader"""
    dataset = LongVideoDataset(npz_path, transform=get_data_augmentation())
    
    indices = list(range(len(dataset)))
    train_val_idx, test_idx = train_test_split(indices, test_size=test_ratio, random_state=42)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=val_ratio/(1-test_ratio), random_state=42)
    
    train_set = torch.utils.data.Subset(dataset, train_idx)
    val_set = torch.utils.data.Subset(dataset, val_idx)
    test_set = torch.utils.data.Subset(dataset, test_idx)
    
    # 验证/测试集不用增强
    val_set.dataset.transform = None
    test_set.dataset.transform = None
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader, test_loader


def get_style_labels():
    """从 label_map.json 加载风格名称列表"""
    if os.path.exists(LABEL_MAP_PATH):
        with open(LABEL_MAP_PATH, 'r', encoding='utf-8') as f:
            label_map = json.load(f)
        style_names = [label_map["style"][str(i)] for i in range(len(label_map["style"]))]
        return style_names
    else:
        return ["赛乃姆", "刀郎舞", "盘子舞", "萨玛舞", "手鼓舞", "纳孜尔库姆"]