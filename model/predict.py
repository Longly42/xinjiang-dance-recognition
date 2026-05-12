"""
predict.py – 封装模型加载和实时推理，供 GUI 的 CaptureThread 调用
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import os
from typing import Tuple

from .config import (
    MODEL_SAVE_PATH, LABEL_MAP_PATH,
    SEQUENCE_LENGTH, NUM_KEYPOINTS, KEYPOINT_FEATURE_DIM,
    NUM_ACTIONS, NUM_STYLES, INFERENCE_CONFIDENCE_THRESHOLD
)
from .stgcn import DualTaskSTGCN


class Recognizer:
    """推理识别器，负责加载模型、预处理输入帧、执行推理"""
    def __init__(self, model_path: str = MODEL_SAVE_PATH, label_map_path: str = LABEL_MAP_PATH):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DualTaskSTGCN().to(self.device)

        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"[Recognizer] 加载模型成功: {model_path}")
        else:
            print(f"[Recognizer] 未找到模型文件 {model_path}，将使用随机权重（演示模式）")
        self.model.eval()

        # 加载标签映射
        self.action_names = [f"动作{i}" for i in range(NUM_ACTIONS)]
        self.style_names = [f"风格{i}" for i in range(NUM_STYLES)]
        if os.path.exists(label_map_path):
            with open(label_map_path, 'r', encoding='utf-8') as f:
                label_map = json.load(f)
                # 假设格式: {"action": {"0":"旋转", ...}, "style": {"0":"赛乃姆",...}}
                self.action_names = [label_map["action"][str(i)] for i in range(NUM_ACTIONS)]
                self.style_names = [label_map["style"][str(i)] for i in range(NUM_STYLES)]

    def _preprocess_frames(self, frames: np.ndarray) -> torch.Tensor:
        """
        frames: 形状 (T, H, W, 3) 的 RGB 图像列表（T=SEQUENCE_LENGTH）
        需先经过 PoseExtractor 提取骨架，但此处为了简化，直接要求输入骨架数据。
        实际 GUI 流程：
        CaptureThread 缓存 RGB 帧 -> 调用 PoseExtractor 提取骨架 -> 调用此预测方法。
        本方法直接处理骨架序列。
        """
        # 假定 frames 已经是 (T, V, C) 的骨架数组
        if frames.ndim == 3:
            frames = frames[np.newaxis, ...]  # (1, T, V, C)
        # 转换为 (C, T, V) 格式
        tensor = torch.from_numpy(frames).float().permute(0, 3, 1, 2).to(self.device)  # (N, C, T, V)
        return tensor

    def predict_from_poses(self, poses: np.ndarray) -> Tuple[str, str, float]:
        """
        输入：(T, V, C) 骨架序列
        输出：(动作名称, 风格名称, 置信度（最大 softmax 概率）)
        """
        if poses.shape[0] < SEQUENCE_LENGTH:
            # 帧数不足，返回默认值
            return "检测中", "检测中", 0.0

        # 取最后 SEQUENCE_LENGTH 帧
        poses = poses[-SEQUENCE_LENGTH:]
        tensor = self._preprocess_frames(poses)

        with torch.no_grad():
            action_logits, style_logits = self.model(tensor)
            action_prob = F.softmax(action_logits, dim=1)[0]
            style_prob = F.softmax(style_logits, dim=1)[0]

        action_id = torch.argmax(action_prob).item()
        style_id = torch.argmax(style_prob).item()
        confidence = max(action_prob.max().item(), style_prob.max().item())

        action_name = self.action_names[action_id]
        style_name = self.style_names[style_id]

        return action_name, style_name, confidence

    def predict_from_frames(self, frames_rgb_list: list) -> Tuple[str, str, float]:
        """
        直接接收 RGB 帧列表（需先提取骨架）
        实际中，GUI 会调用 PoseExtractor 提取后调用 predict_from_poses。
        此接口仅用于兼容原有 thread_tools 调用方式。
        """
        # 这里期望 frames_rgb_list 已经是提取好的骨架（为了兼容 Mock 方式）
        # 如果传入的是图像，需要先提取骨架，但会导致循环依赖。建议保持统一：
        # GUI 线程维护 FrameProcessor 和 PoseExtractor，推理时传递骨架数据。
        raise NotImplementedError("请先通过 PoseExtractor 提取骨架，再调用 predict_from_poses")


def get_recognizer() -> Recognizer:
    """工厂函数，供 GUI 调用"""
    return Recognizer()