"""
stgcn_style.py – 仅输出风格的 ST-GCN（去掉动作分支）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import (
    IN_CHANNELS, HIDDEN_DIM, NUM_STGCN_LAYERS,
    NUM_STYLES, DROPOUT, NUM_KEYPOINTS
)


def get_mediapipe_adjacency():
    """构建 MediaPipe 33 点邻接矩阵（与原model相同）"""
    edges = [
        (11, 12), (12, 24), (24, 23), (23, 11),
        (11, 13), (13, 15), (15, 17), (17, 19), (19, 15),
        (12, 14), (14, 16), (16, 18), (18, 20), (20, 16),
        (11, 23), (12, 24),
        (11, 13), (12, 14),
        (11, 21), (12, 22),
    ]
    V = NUM_KEYPOINTS
    adj = torch.zeros(V, V)
    for u, v in edges:
        if u < V and v < V:
            adj[u, v] = 1
            adj[v, u] = 1
    adj += torch.eye(V)
    return adj


class GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, adj):
        super().__init__()
        self.adj = adj
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        N, C, T, V = x.shape
        x = x.permute(0, 2, 3, 1).reshape(N*T, V, C)
        x = torch.matmul(self.adj, x)
        x = x.reshape(N, T, V, C).permute(0, 3, 1, 2)
        x = self.conv(x)
        return x


class STGCNBlock(nn.Module):
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
    def __init__(self, in_channels, hidden_dim, num_layers, adj, dropout=0):
        super().__init__()
        layers = []
        layers.append(STGCNBlock(in_channels, hidden_dim, adj, dropout=dropout))
        current_dim = hidden_dim
        for i in range(num_layers - 1):
            out_dim = hidden_dim * (2 if i < num_layers//2 else 1)
            layers.append(STGCNBlock(current_dim, out_dim, adj, stride=1, dropout=dropout))
            current_dim = out_dim
        self.layers = nn.Sequential(*layers)
        self.out_channels = current_dim

    def forward(self, x):
        return self.layers(x)


class StyleSTGCN(nn.Module):
    """单任务 ST-GCN：只输出风格"""
    def __init__(self, in_channels=IN_CHANNELS, hidden_dim=HIDDEN_DIM, num_layers=NUM_STGCN_LAYERS,
                 num_styles=NUM_STYLES, dropout=DROPOUT):
        super().__init__()
        adj = get_mediapipe_adjacency().to(torch.float32)
        self.stgcn = STGCN(in_channels, hidden_dim, num_layers, adj, dropout)
        # 全局时空池化（时间+空间）
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        # 风格分类头
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.stgcn.out_channels, num_styles)
        )

    def forward(self, x):
        # x: (N, C, T, V)
        features = self.stgcn(x)          # (N, C_out, T, V)
        features = self.global_pool(features)  # (N, C_out, 1, 1)
        features = features.view(features.size(0), -1)  # (N, C_out)
        style_logits = self.classifier(features)
        return style_logits

    def predict(self, x):
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1)
        return pred, probs