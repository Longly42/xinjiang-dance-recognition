"""
pose_extract.py - MediaPipe 骨架提取
功能：从帧序列中提取人体关键点
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import List, Optional


class PoseExtractor:
    """MediaPipe 姿态提取器"""
    
    def __init__(self, static_image_mode=False, model_complexity=1):
        """
        初始化 MediaPipe Pose
        :param static_image_mode: 静态图像模式（True 用于单帧，False 用于视频）
        :param model_complexity: 模型复杂度 (0,1,2)
        """
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
    
    def extract_single_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        从单帧提取关键点
        :param frame: RGB 图像 (H, W, 3)
        :return: (33, 3) 关键点数组，失败返回 None
        """
        # 转换颜色空间（如果输入是 BGR）
        if frame.shape[2] == 3 and frame[0,0,0] < frame[0,0,2]:  # 简单判断是否为 BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = self.pose.process(frame)
        
        if not results.pose_landmarks:
            return None
        
        # 提取关键点坐标和可见性
        keypoints = []
        for lm in results.pose_landmarks.landmark:
            # (x, y, visibility) 归一化坐标
            keypoints.append([lm.x, lm.y, lm.visibility])
        
        return np.array(keypoints, dtype=np.float32)
    
    def extract_sequence(self, frames: List[np.ndarray]) -> Optional[np.ndarray]:
        """
        从帧序列提取关键点序列
        :param frames: RGB 图像列表
        :return: (T, 33, 3) 关键点序列
        """
        pose_sequence = []
        
        for frame in frames:
            pose = self.extract_single_frame(frame)
            if pose is not None:
                pose_sequence.append(pose)
            else:
                # 如果某帧检测失败，使用前一帧填充
                if pose_sequence:
                    pose_sequence.append(pose_sequence[-1].copy())
                else:
                    # 第一帧就失败，返回 None
                    return None
        
        return np.array(pose_sequence, dtype=np.float32)
    
    def draw_pose(self, frame: np.ndarray, pose: np.ndarray) -> np.ndarray:
        """
        在图像上绘制骨架
        :param frame: 原始图像
        :param pose: (33, 3) 关键点
        :return: 绘制后的图像
        """
        h, w = frame.shape[:2]
        
        # 将归一化坐标转换为像素坐标
        landmarks = []
        for x, y, vis in pose:
            landmarks.append(type('Landmark', (), {
                'x': x, 'y': y, 'z': 0, 'visibility': vis
            })())
        
        # 创建临时结果对象
        class Results:
            pass
        results = Results()
        results.pose_landmarks = type('Landmarks', (), {'landmark': landmarks})()
        
        # 绘制
        annotated_frame = frame.copy()
        self.mp_drawing.draw_landmarks(
            annotated_frame,
            results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS
        )
        
        return annotated_frame


# 全局单例
_pose_extractor = None

def get_pose_extractor():
    """获取全局骨架提取器单例"""
    global _pose_extractor
    if _pose_extractor is None:
        _pose_extractor = PoseExtractor()
    return _pose_extractor