"""
frame_process.py - 帧预处理模块（性能优化版）
"""

import cv2
import numpy as np
from typing import Tuple


class FrameProcessor:
    """帧预处理器（优化版）"""
    
    def __init__(self, target_size: Tuple[int, int] = (480, 360)):
        """
        target_size: (width, height) 目标尺寸
        """
        self.target_size = target_size
        # 预分配缓存（可选）
        self.last_frame = None
        self.last_processed = None
    
    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        单帧预处理（优化版）
        1. 快速调整尺寸
        2. 确保 RGB 格式
        """
        if frame is None:
            return None
        
        # 快速尺寸调整（使用 INTER_NEAREST 最快，精度损失极小）
        h, w = self.target_size[1], self.target_size[0]
        
        # 如果尺寸已经匹配，跳过 resize
        if frame.shape[1] != w or frame.shape[0] != h:
            # 使用 INTER_LINEAR 保持一定精度，INTER_NEAREST 更快但稍模糊
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # 快速颜色转换（使用查表或直接判断）
        if len(frame.shape) == 2:  # 灰度图
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 4:  # BGRA
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        elif frame.shape[2] == 3:
            # OpenCV 默认 BGR，转 RGB（这一步不可避免）
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        return frame
    
    def batch_process(self, frames: list) -> np.ndarray:
        """批量处理帧序列"""
        processed = [self.process(f) for f in frames if f is not None]
        if not processed:
            return None
        return np.array(processed)