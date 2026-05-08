"""
thread_tools.py - 多线程工具类
"""
import time
import numpy as np
from collections import deque
from PyQt6.QtCore import QThread, pyqtSignal, QMutex, QWaitCondition

from capture.frame_capture import ScreenCapture, VideoCapture
from capture.frame_process import FrameProcessor
from utils.config import SEQUENCE_LENGTH, INPUT_SIZE


class MockInference:
    """Mock 推理接口（在 B 组模型完成前使用）"""
    
    def __init__(self):
        # 模拟的动作和风格标签
        self.actions = ["旋转", "翻腕", "移颈", "跺步", "蹲起"]
        self.styles = ["刀郎", "木卡姆", "赛乃姆", "萨玛舞"]
        self.frame_count = 0
    
    def predict(self, frames: np.ndarray):
        """
        模拟推理
        frames: (N, H, W, C) 帧序列
        返回: (action_label, style_label, confidence)
        """
        self.frame_count += 1
        
        # 简单的周期性变化，方便测试
        import random
        action = random.choice(self.actions)
        style = random.choice(self.styles)
        confidence = 0.6 + random.random() * 0.35
        
        return action, style, confidence


class CaptureThread(QThread):
    """采集与推理线程"""
    
    # 信号定义
    frame_ready = pyqtSignal(np.ndarray)  # 发送预处理后的帧用于预览
    result_ready = pyqtSignal(str, str, float)  # 发送推理结果 (action, style, confidence)
    status_updated = pyqtSignal(str)  # 发送状态信息
    error_occurred = pyqtSignal(str)  # 发送错误信息
    
    def __init__(self, source_type="screen", source_param=None):
        """
        初始化采集线程
        source_type: "screen" 或 "video"
        source_param: 屏幕选区(tuple) 或 视频路径(str)
        """
        super().__init__()
        self.source_type = source_type
        self.source_param = source_param
        self.running = False
        self.paused = False
        
        # 帧处理
        self.frame_processor = FrameProcessor(target_size=INPUT_SIZE)
        
        # 帧缓存（用于推理）
        self.frame_buffer = deque(maxlen=SEQUENCE_LENGTH)
        
        # 推理接口（先用 Mock，后续替换为 B 组的真实接口）
        self.inference = MockInference()
        
        # 状态
        self.fps = 0
        self.frame_count = 0
        self.last_time = time.time()
        
        # 线程同步
        self.mutex = QMutex()
        self.cond = QWaitCondition()
        
        # 采集器
        self.capturer = None
        self._init_capturer()
    
    def _init_capturer(self):
        """初始化采集器"""
        if self.source_type == "screen":
            if self.source_param:
                self.capturer = ScreenCapture(area=self.source_param)
            else:
                self.capturer = ScreenCapture()
        elif self.source_type == "video":
            self.capturer = VideoCapture(self.source_param)
    
    def update_source(self, source_type, source_param):
        """动态更新采集源"""
        self.mutex.lock()
        
        # 释放旧的采集器
        if self.capturer and hasattr(self.capturer, 'release'):
            self.capturer.release()
        
        self.source_type = source_type
        self.source_param = source_param
        self._init_capturer()
        
        # 清空帧缓存
        self.frame_buffer.clear()
        
        self.mutex.unlock()
    
    def run(self):
        """线程主循环"""
        self.running = True
        self.last_time = time.time()
        
        # FPS 统计
        frame_times = []
        
        while self.running:
            # 处理暂停
            if self.paused:
                self.msleep(10)
                continue
            
            # 采集帧
            frame = self._capture_frame()
            
            if frame is None:
                # 视频结束或采集失败
                if self.source_type == "video":
                    # 视频播放完毕，循环播放
                    if hasattr(self.capturer, '_open'):
                        self.capturer._open()
                        continue
                self.msleep(1)
                continue
            
            # 预处理
            processed_frame = self.frame_processor.process(frame)
            
            if processed_frame is not None:
                # 发送预览帧
                self.frame_ready.emit(processed_frame)
                
                # 加入缓存
                self.frame_buffer.append(processed_frame)
                
                # 当缓存满时进行推理
                if len(self.frame_buffer) == SEQUENCE_LENGTH:
                    self._run_inference()
            
            # FPS 统计
            self.frame_count += 1
            now = time.time()
            if now - self.last_time >= 1.0:
                self.fps = self.frame_count / (now - self.last_time)
                self.status_updated.emit(f"采集帧率: {self.fps:.1f} fps")
                self.frame_count = 0
                self.last_time = now
            
            # 控制帧率（约 30fps）
            self.msleep(33)
        
        # 清理
        if self.capturer and hasattr(self.capturer, 'release'):
            self.capturer.release()
    
    def _capture_frame(self):
        """采集一帧"""
        try:
            if self.source_type == "screen":
                return self.capturer.capture()
            elif self.source_type == "video":
                return self.capturer.read()
        except Exception as e:
            self.error_occurred.emit(f"采集失败: {e}")
            return None
    
    def _run_inference(self):
        """执行推理"""
        try:
            # 将缓存帧转换为推理输入格式
            frames_list = list(self.frame_buffer)
            frames_array = np.array(frames_list)
            
            # 调用推理接口
            action, style, confidence = self.inference.predict(frames_array)
            
            # 发送结果
            self.result_ready.emit(action, style, confidence)
            
        except Exception as e:
            self.error_occurred.emit(f"推理失败: {e}")
    
    def stop(self):
        """停止线程"""
        self.running = False
        self.wait()
    
    def pause(self):
        """暂停"""
        self.paused = True
    
    def resume(self):
        """恢复"""
        self.paused = False