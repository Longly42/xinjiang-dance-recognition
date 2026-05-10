"""
action_model.py - 双任务模型（动作识别 + 风格识别）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DanceRecognitionModel(nn.Module):
    """新疆舞双任务识别模型"""
    
    def __init__(self, num_actions=5, num_styles=4, num_keypoints=33, keypoint_dim=3, hidden_dim=256):
        """
        初始化模型
        :param num_actions: 动作类别数
        :param num_styles: 风格类别数
        :param num_keypoints: 关键点数量
        :param keypoint_dim: 每个关键点的特征维度
        :param hidden_dim: LSTM 隐藏层维度
        """
        super(DanceRecognitionModel, self).__init__()
        
        # 输入特征维度 = 33 * 3 = 99
        input_dim = num_keypoints * keypoint_dim
        
        # 空间特征提取（对每帧的关键点进行编码）
        self.spatial_encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 时序建模（LSTM）
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        
        # 动作分类头
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),  # *2 因为双向
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_actions)
        )
        
        # 风格分类头
        self.style_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_styles)
        )
    
    def forward(self, x):
        """
        前向传播
        :param x: (batch, seq_len, num_keypoints, keypoint_dim)
        :return: (action_logits, style_logits)
        """
        batch_size, seq_len, num_kpts, kpt_dim = x.shape
        
        # 展平关键点维度
        x = x.view(batch_size, seq_len, num_kpts * kpt_dim)
        
        # 空间编码（对每个时间步独立处理）
        # 需要 reshape 成 (batch * seq_len, input_dim)
        x = x.view(batch_size * seq_len, -1)
        x = self.spatial_encoder(x)
        x = x.view(batch_size, seq_len, -1)
        
        # LSTM 时序建模
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # 取最后一个时间步的输出
        # 对于双向 LSTM，需要拼接前后向的 hidden
        # hidden 形状：(2*2, batch, hidden_dim) -> (batch, hidden_dim*2)
        hidden_forward = hidden[-2, :, :]  # 最后一层前向
        hidden_backward = hidden[-1, :, :] # 最后一层后向
        features = torch.cat([hidden_forward, hidden_backward], dim=1)
        
        # 分类
        action_logits = self.action_head(features)
        style_logits = self.style_head(features)
        
        return action_logits, style_logits
    
    def predict(self, x):
        """
        推理接口
        :param x: (seq_len, num_keypoints, keypoint_dim)
        :return: (action_label, style_label, action_conf, style_conf)
        """
        self.eval()
        with torch.no_grad():
            # 添加 batch 维度
            if len(x.shape) == 3:
                x = x.unsqueeze(0)
            
            action_logits, style_logits = self.forward(x)
            
            action_probs = F.softmax(action_logits, dim=1)
            style_probs = F.softmax(style_logits, dim=1)
            
            action_label = torch.argmax(action_probs, dim=1).item()
            style_label = torch.argmax(style_probs, dim=1).item()
            action_conf = torch.max(action_probs, dim=1)[0].item()
            style_conf = torch.max(style_probs, dim=1)[0].item()
        
        return action_label, style_label, action_conf, style_conf


def create_model(num_actions=5, num_styles=4):
    """创建模型的工厂函数"""
    return DanceRecognitionModel(num_actions=num_actions, num_styles=num_styles)


def save_model(model, path):
    """保存模型"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_actions': model.action_head[-1].out_features,
        'num_styles': model.style_head[-1].out_features,
    }, path)
    print(f"[INFO] 模型已保存到 {path}")


def load_model(path, device='cpu'):
    """加载模型"""
    checkpoint = torch.load(path, map_location=device)
    model = DanceRecognitionModel(
        num_actions=checkpoint['num_actions'],
        num_styles=checkpoint['num_styles']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model