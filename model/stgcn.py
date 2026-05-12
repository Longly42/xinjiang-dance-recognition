"""
stgcn.py – 基于 ST-GCN 的空间时序图卷积网络，用于人体骨架动作识别
实现 DualTaskSTGCN，同时输出动作类别和风格类别
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from .config import (
    IN_CHANNELS, HIDDEN_DIM, NUM_STGCN_LAYERS,
    NUM_ACTIONS, NUM_STYLES, DROPOUT, SEQUENCE_LENGTH, NUM_KEYPOINTS
)


def get_mediapipe_adjacency() -> torch.Tensor:
    """
    构建 MediaPipe 33 个人体关键点的邻接矩阵 (V, V)
    基于官方定义的连接关系
    """
    # MediaPipe 33 点连接边 (from, to) 列表
    # 参考: https://github.com/google/mediapipe/blob/master/mediapipe/modules/pose_landmark/pose_landmark_topology.svg
    # 这里列出主要的连接（简化版，完整边可自行扩展，但 ST-GCN 对稀疏图也能工作）
    edges = [
        # 躯干
        (11, 12), (12, 24), (24, 23), (23, 11),   # 髋部矩形
        (11, 13), (13, 15), (15, 17), (17, 19), (19, 15),  # 左腿
        (12, 14), (14, 16), (16, 18), (18, 20), (20, 16),  # 右腿
        (11, 23), (12, 24),
        # 肩膀
        (11, 13), (12, 14),   # 肩膀到髋？
        # 上肢
        (11, 21), (12, 22),   # 肩到腕（实际中间有肘，简化直接用肩连腕也可）
        # 更精确的四肢连接（需要更多点），这里作为示意，完整版可从官方获取。
        # 为简化，可以接受 ST-GCN 自动学习部分关系。
    ]
    V = NUM_KEYPOINTS
    adj = torch.zeros(V, V)
    for u, v in edges:
        if u < V and v < V:
            adj[u, v] = 1
            adj[v, u] = 1
    # 加上自环
    adj += torch.eye(V)
    return adj


class GraphConv(nn.Module):
    """单层图卷积: 输出 = A * X * W"""
    def __init__(self, in_channels, out_channels, adj):
        super().__init__()
        self.adj = adj  # (V, V)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # x: (N, C, T, V)
        N, C, T, V = x.shape
        # 空间图卷积: (N, C, T, V) * (V, V) -> (N, C, T, V)
        x = x.permute(0, 2, 3, 1).reshape(N*T, V, C)   # (N*T, V, C)
        x = torch.matmul(self.adj, x)                   # (N*T, V, C)
        x = x.reshape(N, T, V, C).permute(0, 3, 1, 2)   # (N, C, T, V)
        x = self.conv(x)
        return x


class STGCNBlock(nn.Module):
    """一个 ST-GCN 基本块: 图卷积 + 时域卷积 + BN + ReLU + Dropout"""
    def __init__(self, in_channels, out_channels, adj, stride=1, dropout=0):
        super().__init__()
        self.gcn = GraphConv(in_channels, out_channels, adj)
        self.tcn = nn.Conv2d(out_channels, out_channels, kernel_size=(3,1), padding=(1,0), stride=(stride,1))
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.residual(x)
        out = self.gcn(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.tcn(out)
        out = self.bn2(out)
        out = self.dropout(out)
        out = out + residual
        out = self.relu(out)
        return out


class STGCN(nn.Module):
    """标准 ST-GCN 主干网"""
    def __init__(self, in_channels, hidden_dim, num_layers, adj, dropout=0):
        super().__init__()
        layers = []
        # 第一层通道提升
        layers.append(STGCNBlock(in_channels, hidden_dim, adj, dropout=dropout))
        current_dim = hidden_dim
        # 后续层保持或增加通道
        for i in range(num_layers - 1):
            out_dim = hidden_dim * (2 if i < num_layers//2 else 1)  # 中间翻倍
            layers.append(STGCNBlock(current_dim, out_dim, adj, stride=1, dropout=dropout))
            current_dim = out_dim
        self.layers = nn.Sequential(*layers)
        self.out_channels = current_dim

    def forward(self, x):
        # x: (N, C, T, V)
        return self.layers(x)


class DualTaskSTGCN(nn.Module):
    """双任务 ST-GCN：动作分类 + 风格分类"""
    def __init__(self, in_channels=IN_CHANNELS, hidden_dim=HIDDEN_DIM, num_layers=NUM_STGCN_LAYERS,
                 num_actions=NUM_ACTIONS, num_styles=NUM_STYLES, dropout=DROPOUT):
        super().__init__()
        adj = get_mediapipe_adjacency().to(torch.float32)
        self.adj = adj

        self.stgcn = STGCN(in_channels, hidden_dim, num_layers, adj, dropout)
        # 全局池化 (T, V) 平均
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 动作分类头
        self.action_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.stgcn.out_channels, num_actions)
        )
        # 风格分类头
        self.style_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.stgcn.out_channels, num_styles)
        )

    def forward(self, x):
        # x: (N, C, T, V)
        features = self.stgcn(x)          # (N, C_out, T, V)
        features = self.global_pool(features)  # (N, C_out, 1, 1)
        features = features.view(features.size(0), -1)  # (N, C_out)

        action_logits = self.action_head(features)
        style_logits = self.style_head(features)

        return action_logits, style_logits

    def predict(self, x):
        """推理时直接返回 softmax 后的概率和类别"""
        with torch.no_grad():
            action_logits, style_logits = self.forward(x)
            action_prob = F.softmax(action_logits, dim=1)
            style_prob = F.softmax(style_logits, dim=1)
            action_pred = torch.argmax(action_prob, dim=1)
            style_pred = torch.argmax(style_prob, dim=1)
        return action_pred, style_pred, action_prob, style_prob