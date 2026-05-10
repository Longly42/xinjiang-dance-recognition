"""
predict.py - 推理接口封装
供 A 组调用的统一接口
"""

import torch
import numpy as np
from typing import Tuple, List, Optional

from model.action_model import load_model
from model.pose_extract import get_pose_extractor
from utils.config import (
    ACTION_LABELS, STYLE_LABELS, 
    SEQUENCE_LENGTH, INPUT_SIZE
)
import cv2


class DanceRecognizer:
    """舞蹈识别器（单例）"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.model = None
        self.pose_extractor = get_pose_extractor()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._load_model()
        self.frame_buffer = []  # 帧缓存
        self._initialized = True
    
    def _load_model(self):
        """加载模型"""
        import os
        from utils.config import MODEL_DIR
        
        model_path = os.path.join(MODEL_DIR, 'best_model.pth')
        
        if os.path.exists(model_path):
            try:
                self.model = load_model(model_path, self.device)
                print(f"[INFO] 模型加载成功: {model_path}")
            except Exception as e:
                print(f"[ERROR] 模型加载失败: {e}")
                self.model = None
        else:
            print(f"[WARN] 模型文件不存在，使用 Mock 模式: {model_path}")
            self.model = None
    
    def predict_from_frames(self, frames: List[np.ndarray]) -> Tuple[str, str, float]:
        """
        从帧列表进行推理（A 组调用此接口）
        :param frames: RGB 图像列表，长度至少为 SEQUENCE_LENGTH
        :return: (动作标签, 风格标签, 置信度)
        """
        if len(frames) < SEQUENCE_LENGTH:
            return "等待更多帧", "等待更多帧", 0.0
        
        # 提取骨架
        poses = self._extract_poses_from_frames(frames[-SEQUENCE_LENGTH:])
        
        if poses is None:
            return "未检测到人体", "未知", 0.0
        
        # 推理
        if self.model is not None:
            action_label, style_label, action_conf, style_conf = self._predict_with_model(poses)
            # 使用动作和风格置信度的平均值作为整体置信度
            confidence = (action_conf + style_conf) / 2
        else:
            # Mock 推理
            action_label, style_label, confidence = self._predict_mock(poses)
        
        # 转换为标签名称
        action_name = ACTION_LABELS.get(action_label, f"动作{action_label}")
        style_name = STYLE_LABELS.get(style_label, f"风格{style_label}")
        
        return action_name, style_name, confidence
    
    def _extract_poses_from_frames(self, frames: List[np.ndarray]) -> Optional[np.ndarray]:
        """从帧列表提取骨架序列"""
        poses = self.pose_extractor.extract_sequence(frames)
        
        if poses is None or len(poses) < SEQUENCE_LENGTH:
            return None
        
        # 确保序列长度一致
        if len(poses) > SEQUENCE_LENGTH:
            indices = np.linspace(0, len(poses)-1, SEQUENCE_LENGTH, dtype=int)
            poses = poses[indices]
        
        return poses
    
    def _predict_with_model(self, poses: np.ndarray) -> Tuple[int, int, float, float]:
        """使用模型推理"""
        # 转换为 Tensor
        poses_tensor = torch.FloatTensor(poses).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_logits, style_logits = self.model(poses_tensor)
            action_probs = torch.softmax(action_logits, dim=1)
            style_probs = torch.softmax(style_logits, dim=1)
            
            action_label = torch.argmax(action_probs, dim=1).item()
            style_label = torch.argmax(style_probs, dim=1).item()
            action_conf = torch.max(action_probs, dim=1)[0].item()
            style_conf = torch.max(style_probs, dim=1)[0].item()
        
        return action_label, style_label, action_conf, style_conf
    
    def _predict_mock(self, poses: np.ndarray) -> Tuple[int, int, float]:
        """Mock 推理（用于测试）"""
        import random
        action_label = random.randint(0, len(ACTION_LABELS) - 1)
        style_label = random.randint(0, len(STYLE_LABELS) - 1)
        confidence = 0.5 + random.random() * 0.4
        return action_label, style_label, confidence
    
    def reset(self):
        """重置状态"""
        self.frame_buffer.clear()


# 全局单例
_recognizer = None

def get_recognizer() -> DanceRecognizer:
    """获取全局识别器实例"""
    global _recognizer
    if _recognizer is None:
        _recognizer = DanceRecognizer()
    return _recognizer