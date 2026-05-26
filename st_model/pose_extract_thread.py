"""
pose_extract_thread.py - 独立高频骨架提取线程
持续从共享帧队列获取最新帧，实时提取骨架，更新历史队列
"""

import threading
import time
import numpy as np
from collections import deque

from .pose_extract import PoseExtractor


class PoseExtractThread(threading.Thread):
    """
    骨架提取线程：独立于采集和推理，高频运行（与视频帧率同步）
    """
    def __init__(self, frame_queue, skeleton_history, skeleton_lock, max_history_len=64):
        """
        frame_queue: 线程安全的队列（存放最新帧，由CaptureThread放入）
        skeleton_history: deque 用于存储历史骨架 (T,33,3)
        skeleton_lock: 保护 skeleton_history 的锁
        max_history_len: 历史骨架最大长度（通常等于 SEQUENCE_LENGTH）
        """
        super().__init__(daemon=True)
        self.frame_queue = frame_queue
        self.skeleton_history = skeleton_history
        self.skeleton_lock = skeleton_lock
        self.max_history_len = max_history_len
        self.running = True
        self.pose_extractor = PoseExtractor()   # 单例，避免重复加载
        self.last_pose = None

    def run(self):
        print("[PoseExtractThread] 启动，等待帧数据...")
        while self.running:
            try:
                # 从队列获取一帧（阻塞，超时0.1秒避免僵死）
                frame = self.frame_queue.get(timeout=0.1)
                if frame is None:
                    continue
                # 提取骨架 (33,3)
                pose = self.pose_extractor.extract_from_frame(frame)
                if pose is not None:
                    self.last_pose = pose
                elif self.last_pose is not None:
                    pose = self.last_pose.copy()
                else:
                    pose = np.zeros((33, 3), dtype=np.float32)

                # 更新历史队列（线程安全）
                with self.skeleton_lock:
                    self.skeleton_history.append(pose)
                    # 保持最大长度
                    while len(self.skeleton_history) > self.max_history_len:
                        self.skeleton_history.popleft()
            except Exception as e:
                # 超时或其他异常，继续循环
                continue
        print("[PoseExtractThread] 退出")

    def stop(self):
        self.running = False
        self.join(timeout=1.0)
        self.pose_extractor.release()