"""
pose_extract.py - MediaPipe 姿态提取（非单例，线程安全）
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple

try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False
    print("[WARN] mediapipe 未安装，姿态提取将不可用")


class PoseExtractor:
    """MediaPipe 姿态提取器（非单例，每个线程独立实例）"""

    def __init__(self):
        if not HAS_MEDIAPIPE:
            raise ImportError("mediapipe 未安装，请运行: pip install mediapipe")
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def extract_single_frame(self, frame_rgb: np.ndarray) -> Optional[np.ndarray]:
        """从单帧 RGB 图像提取关键点，返回 (33, 3) 数组"""
        results = self.pose.process(frame_rgb)
        if not results.pose_landmarks:
            return None
        landmarks = []
        for lm in results.pose_landmarks.landmark:
            landmarks.append([lm.x, lm.y, lm.visibility])
        return np.array(landmarks, dtype=np.float32)

    def extract_sequence(self, frames_rgb: List[np.ndarray]) -> Optional[np.ndarray]:
        """从多帧提取骨架序列，返回 (T, 33, 3) 数组"""
        poses = []
        for frame in frames_rgb:
            pose = self.extract_single_frame(frame)
            if pose is None:
                return None
            poses.append(pose)
        return np.array(poses, dtype=np.float32)

    def release(self):
        """释放 MediaPipe 资源"""
        self.pose.close()


def get_pose_extractor() -> PoseExtractor:
    """每次调用返回新实例"""
    return PoseExtractor()