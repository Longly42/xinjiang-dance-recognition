"""
predict.py – 风格识别推理接口 + 规则动作识别（无需训练）
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
import json

from .config import MODEL_SAVE_PATH, LABEL_MAP_PATH, SEQUENCE_LENGTH
from .stgcn_style import StyleSTGCN
from .action_rules import RuleBasedActionRecognizer


class StyleRecognizer:
    """
    风格识别器 + 动作识别器（基于规则）
    组合：深度学习模型（风格）+ 启发式规则（动作）
    """
    
    def __init__(self, model_path=MODEL_SAVE_PATH, label_map_path=LABEL_MAP_PATH):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = StyleSTGCN().to(self.device)
        
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"[StyleRecognizer] 加载模型: {model_path}")
        else:
            print(f"[StyleRecognizer] 未找到模型，使用随机权重")
        self.model.eval()
        
        # 加载风格标签
        self.style_names = ["风格0", "风格1", "风格2", "风格3", "风格4", "风格5"]
        if os.path.exists(label_map_path):
            with open(label_map_path, 'r', encoding='utf-8') as f:
                label_map = json.load(f)
            self.style_names = [label_map["style"][str(i)] for i in range(len(label_map["style"]))]
            print(f"[StyleRecognizer] 加载风格标签: {self.style_names}")
        
        # 初始化规则动作识别器（无需训练）
        self.action_recognizer = RuleBasedActionRecognizer(history_frames=15)
        print("[StyleRecognizer] 初始化规则动作识别器")
    
    def update_action_with_pose(self, pose):
        """
        用最新一帧骨架更新动作识别器历史，并返回当前识别的动作
        pose: (33, 3) 单帧骨架
        return: 动作名称字符串
        """
        if pose is not None:
            self.action_recognizer.update(pose)
        return self.action_recognizer.get_action()
    
    def preprocess(self, poses):
        """
        poses: (T, V, C) 骨架序列，T 可能大于或小于 SEQUENCE_LENGTH
        返回: (1, C, T_new, V) 张量
        """
        T, V, C = poses.shape
        # 均匀采样到 SEQUENCE_LENGTH
        if T > SEQUENCE_LENGTH:
            indices = np.linspace(0, T-1, SEQUENCE_LENGTH, dtype=int)
            poses = poses[indices]
        elif T < SEQUENCE_LENGTH:
            # 重复最后一帧填充
            pad = [poses[-1]] * (SEQUENCE_LENGTH - T)
            poses = np.concatenate([poses, pad], axis=0)
        tensor = torch.from_numpy(poses).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
        return tensor
    
    def predict_style(self, poses):
        """
        输入 (T, V, C) 骨架序列，返回 (风格名称, 置信度)
        """
        with torch.no_grad():
            x = self.preprocess(poses)
            logits = self.model(x)
            probs = F.softmax(logits, dim=1)[0]
            pred_id = torch.argmax(probs).item()
            confidence = probs[pred_id].item()
        return self.style_names[pred_id], confidence
    
    def predict(self, poses):
        """
        输入 (T, V, C) 骨架序列，返回 (动作名称, 风格名称, 置信度)
        兼容 thread_tools 的调用约定
        """
        # 1. 动作识别：基于最后一帧骨架更新规则识别器
        action = "无动作"
        if poses is not None and len(poses) > 0:
            action = self.update_action_with_pose(poses[-1])
        
        # 2. 风格识别：使用深度学习模型
        style_name, confidence = self.predict_style(poses)
        
        return action, style_name, confidence
    
    def predict_from_poses(self, poses):
        """
        别名，兼容不同调用方式
        """
        return self.predict(poses)


def get_recognizer():
    """获取识别器实例"""
    return StyleRecognizer()