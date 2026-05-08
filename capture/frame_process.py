"""
frame_process.py - 帧预处理模块
"""
import cv2
import numpy as np
from typing import Tuple


class FrameProcessor:
    """帧预处理器"""
    
    def __init__(self, target_size: Tuple[int, int] = (640, 480)):
        """
        target_size: (width, height) 目标尺寸
        """
        self.target_size = target_size
    
    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        单帧预处理
        1. 调整尺寸
        2. 确保 RGB 格式
        3. 归一化（可选）
        """
        if frame is None:
            return None
        
        # 调整尺寸
        h, w = self.target_size[1], self.target_size[0]
        frame = cv2.resize(frame, (w, h))
        
        # 确保是 RGB
        if len(frame.shape) == 2:  # 灰度图
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 4:  # BGRA
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        elif frame.shape[2] == 3:
            # OpenCV 默认 BGR，转 RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        return frame
    
    def batch_process(self, frames: list) -> np.ndarray:
        """批量处理帧序列"""
        processed = [self.process(f) for f in frames if f is not None]
        if not processed:
            return None
        return np.array(processed)