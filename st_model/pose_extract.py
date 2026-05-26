"""
pose_extract.py - 骨架提取器（MediaPipe）
从RGB帧序列提取33个关键点
"""

import numpy as np
import mediapipe as mp
from collections import deque
from typing import List, Optional


class PoseExtractor:
    """骨架提取器（MediaPipe）- 从RGB帧序列提取33个关键点"""
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,               # 中等复杂度，平衡速度和精度
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        # 用于帧间平滑的缓存
        self.prev_poses = deque(maxlen=5)
    
    def extract_sequence(self, frames_rgb_list: List[np.ndarray]) -> Optional[np.ndarray]:
        """
        从RGB帧列表提取骨架序列
        frames_rgb_list: list of (H, W, 3) RGB图像
        返回: (T, 33, 3) 数组，T = len(frames_rgb_list)
        """
        poses = []
        for frame in frames_rgb_list:
            results = self.pose.process(frame)
            if results.pose_landmarks:
                frame_pose = []
                for lm in results.pose_landmarks.landmark:
                    frame_pose.append([lm.x, lm.y, lm.visibility])
                pose_array = np.array(frame_pose, dtype=np.float32)
                poses.append(pose_array)
                self.prev_poses.append(pose_array)
            else:
                # 提取失败，用前一帧填充或零向量
                if self.prev_poses:
                    poses.append(self.prev_poses[-1].copy())
                else:
                    poses.append(np.zeros((33, 3), dtype=np.float32))
        
        if not poses:
            return None
        return np.array(poses)
    
    def extract_from_frame(self, frame_rgb: np.ndarray) -> Optional[np.ndarray]:
        """
        从单帧RGB图像提取骨架
        返回: (33, 3) 数组或None
        """
        results = self.pose.process(frame_rgb)
        if results.pose_landmarks:
            frame_pose = []
            for lm in results.pose_landmarks.landmark:
                frame_pose.append([lm.x, lm.y, lm.visibility])
            return np.array(frame_pose, dtype=np.float32)
        return None
    
    def release(self):
        """释放MediaPipe资源"""
        self.pose.close()


def get_pose_extractor():
    """获取骨架提取器实例"""
    return PoseExtractor()