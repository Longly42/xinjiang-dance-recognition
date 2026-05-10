"""
thread_tools.py - 多线程工具类（高性能版）
"""

import time
import os
import cv2
import numpy as np
from collections import deque
from datetime import datetime
from PyQt6.QtCore import QThread, pyqtSignal, QMutex, QWaitCondition
import threading

from capture.frame_capture import ScreenCapture, VideoCapture
from capture.frame_process import FrameProcessor
from utils.config import (
    SEQUENCE_LENGTH, INPUT_SIZE, 
    RECORDING_DIR, SLICED_DIR,
    ACTION_LABELS, STYLE_LABELS,
    FPS_TARGET, FRAME_SKIP, ASYNC_INFERENCE
)
from utils.video_slicer import VideoSlicer


class MockInference:
    """Mock 推理接口"""
    
    def __init__(self):
        self.actions = list(ACTION_LABELS.values()) if ACTION_LABELS else ["旋转", "翻腕", "移颈", "跺步", "蹲起"]
        self.styles = list(STYLE_LABELS.values()) if STYLE_LABELS else ["刀郎", "木卡姆", "赛乃姆", "萨玛舞"]
        self.frame_count = 0
    
    def predict(self, frames: np.ndarray):
        import random
        action = random.choice(self.actions)
        style = random.choice(self.styles)
        confidence = 0.6 + random.random() * 0.35
        return action, style, confidence


class CaptureThread(QThread):
    """采集与推理线程（高性能优化版）"""
    
    frame_ready = pyqtSignal(np.ndarray)
    result_ready = pyqtSignal(str, str, float)
    status_updated = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, source_type="screen", source_param=None):
        super().__init__()
        self.source_type = source_type
        self.source_param = source_param
        self.running = False
        self.paused = False
        
        # 帧处理器
        self.frame_processor = FrameProcessor(target_size=INPUT_SIZE)
        
        # 帧缓存（双缓存：采集缓存 + 推理缓存）
        self.frame_buffer = deque(maxlen=SEQUENCE_LENGTH)
        self.inference_buffer = []
        
        # 跳帧计数器
        self.skip_counter = 0
        self.frame_skip = FRAME_SKIP
        
        # 推理接口
        try:
            from model.predict import get_recognizer
            self.recognizer = get_recognizer()
            self.use_mock = False
            print("[INFO] 使用真实推理模型")
        except ImportError as e:
            self.inference = MockInference()
            self.use_mock = True
            print(f"[INFO] 使用 Mock 推理: {e}")
        
        # 异步推理相关
        self.async_inference = ASYNC_INFERENCE
        self.inference_lock = threading.Lock()
        self.last_inference_time = 0
        self.inference_interval = 0.5  # 每0.5秒推理一次（约2次/秒）
        
        # 性能统计
        self.fps = 0
        self.frame_count = 0
        self.last_time = time.time()
        self.processing_times = deque(maxlen=30)  # 记录最近30帧处理时间
        
        # 采集器
        self.capturer = None
        self._init_capturer()
    
    def _init_capturer(self):
        if self.source_type == "screen":
            if self.source_param:
                self.capturer = ScreenCapture(area=self.source_param)
            else:
                self.capturer = ScreenCapture()
        elif self.source_type == "video":
            self.capturer = VideoCapture(self.source_param)
    
    def run(self):
        self.running = True
        self.last_time = time.time()
        
        # 动态帧率控制
        target_fps = FPS_TARGET
        frame_interval = 1.0 / target_fps
        adaptive_interval = frame_interval
        
        while self.running:
            if self.paused:
                self.msleep(10)
                continue
            
            loop_start = time.time()
            
            # 采集帧
            frame = self._capture_frame()
            
            if frame is None:
                if self.source_type == "video":
                    if hasattr(self.capturer, '_open'):
                        self.capturer._open()
                        continue
                self.msleep(1)
                continue
            
            # 跳帧处理
            self.skip_counter += 1
            if self.skip_counter % self.frame_skip != 0:
                # 跳过处理，只控制帧率
                elapsed = time.time() - loop_start
                sleep_time = adaptive_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                continue
            
            # 处理帧
            process_start = time.time()
            processed_frame = self.frame_processor.process(frame)
            process_time = time.time() - process_start
            
            # 记录处理时间
            self.processing_times.append(process_time)
            
            if processed_frame is not None:
                # 发送预览帧
                self.frame_ready.emit(processed_frame)
                
                # 加入缓存
                self.frame_buffer.append(processed_frame)
                
                # 推理（异步或同步）
                if self.async_inference:
                    self._try_async_inference()
                else:
                    if len(self.frame_buffer) == SEQUENCE_LENGTH:
                        self._run_inference()
            
            # FPS 统计
            self.frame_count += 1
            now = time.time()
            if now - self.last_time >= 1.0:
                self.fps = self.frame_count / (now - self.last_time)
                avg_process_time = np.mean(self.processing_times) if self.processing_times else 0
                self.status_updated.emit(f"帧率: {self.fps:.1f} fps | 处理延迟: {avg_process_time*1000:.1f}ms")
                self.frame_count = 0
                self.last_time = now
                
                # 动态调整目标帧率
                if avg_process_time > 0:
                    max_possible_fps = 1.0 / avg_process_time
                    if max_possible_fps < target_fps * 0.8:
                        adaptive_interval = avg_process_time * 1.2
                    else:
                        adaptive_interval = frame_interval
            
            # 精确帧率控制
            elapsed = time.time() - loop_start
            sleep_time = adaptive_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        # 清理
        if self.capturer and hasattr(self.capturer, 'release'):
            self.capturer.release()
    
    def _capture_frame(self):
        try:
            if self.source_type == "screen":
                return self.capturer.capture()
            elif self.source_type == "video":
                return self.capturer.read()
        except Exception as e:
            self.error_occurred.emit(f"采集失败: {e}")
            return None
    
    def _try_async_inference(self):
        """异步推理：不阻塞采集线程"""
        current_time = time.time()
        
        # 限制推理频率（每0.5秒最多一次）
        if current_time - self.last_inference_time < self.inference_interval:
            return
        
        # 至少收集了 SEQUENCE_LENGTH 帧
        if len(self.frame_buffer) >= SEQUENCE_LENGTH:
            self.last_inference_time = current_time
            # 复制当前缓存用于推理（避免在推理时被修改）
            frames_to_infer = list(self.frame_buffer)[-SEQUENCE_LENGTH:]
            
            # 在新线程中执行推理
            inference_thread = threading.Thread(
                target=self._run_inference_async,
                args=(frames_to_infer,)
            )
            inference_thread.daemon = True
            inference_thread.start()
    
    def _run_inference_async(self, frames_list):
        """异步推理执行"""
        try:
            frames_array = np.array(frames_list)
            
            if self.use_mock:
                action, style, confidence = self.inference.predict(frames_array)
            else:
                action, style, confidence = self.recognizer.predict_from_frames(frames_array)
            
            # 发送结果到主线程
            self.result_ready.emit(action, style, confidence)
        except Exception as e:
            self.error_occurred.emit(f"推理失败: {e}")
    
    def _run_inference(self):
        """同步推理"""
        try:
            frames_list = list(self.frame_buffer)
            frames_array = np.array(frames_list)
            
            if self.use_mock:
                action, style, confidence = self.inference.predict(frames_array)
            else:
                action, style, confidence = self.recognizer.predict_from_frames(frames_array)
            
            self.result_ready.emit(action, style, confidence)
        except Exception as e:
            self.error_occurred.emit(f"推理失败: {e}")
    
    def stop(self):
        self.running = False
        self.wait()
    
    def pause(self):
        self.paused = True
    
    def resume(self):
        self.paused = False


class RecordingThread(QThread):
    """录屏线程（优化版）"""
    
    status_updated = pyqtSignal(str)
    finished = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    progress_updated = pyqtSignal(int)
    frame_ready = pyqtSignal(np.ndarray)
    
    def __init__(self, area, output_dir=RECORDING_DIR, fps=20, max_duration_sec=None):
        super().__init__()
        self.area = area
        self.output_dir = output_dir
        self.target_fps = fps
        self.max_duration_sec = max_duration_sec
        self.running = False
        self.start_time = None
        self.frame_processor = FrameProcessor(target_size=INPUT_SIZE)
    
    def run(self):
        os.makedirs(self.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = os.path.join(self.output_dir, f"recording_{timestamp}.mp4")
        
        capturer = ScreenCapture(area=self.area)
        first_frame = capturer.capture()
        
        if first_frame is None:
            self.error_occurred.emit("无法捕获屏幕区域")
            return
        
        h, w = first_frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(video_path, fourcc, self.target_fps, (w, h))
        
        self.running = True
        self.start_time = time.time()
        frame_interval = 1.0 / self.target_fps
        frame_count = 0
        
        self.status_updated.emit(f"开始录屏，保存至 {video_path}")
        
        while self.running:
            if self.max_duration_sec and (time.time() - self.start_time) > self.max_duration_sec:
                break
            
            loop_start = time.time()
            frame = capturer.capture()
            
            if frame is not None:
                writer.write(frame)
                frame_count += 1
                
                # 预览（降低预览频率，不降低录制质量）
                if frame_count % 2 == 0:  # 每2帧发送一次预览
                    processed_frame = self.frame_processor.process(frame)
                    if processed_frame is not None:
                        self.frame_ready.emit(processed_frame)
                
                if self.max_duration_sec:
                    progress = int((time.time() - self.start_time) / self.max_duration_sec * 100)
                    self.progress_updated.emit(min(progress, 100))
            
            elapsed = time.time() - loop_start
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
        
        writer.release()
        capturer.release()
        
        self.status_updated.emit(f"录制结束，共 {frame_count} 帧")
        self.finished.emit(video_path)
    
    def stop(self):
        self.running = False
        self.wait()


class DatasetBuildThread(QThread):
    """数据集构建线程"""
    
    status_updated = pyqtSignal(str)
    finished = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    progress_updated = pyqtSignal(int)
    
    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path
    
    def run(self):
        try:
            os.makedirs(SLICED_DIR, exist_ok=True)
            
            self.status_updated.emit("正在切割视频...")
            slicer = VideoSlicer(window_sec=2, step_sec=1)
            clips = slicer.slice_video(self.video_path, SLICED_DIR)
            
            if not clips:
                self.error_occurred.emit("视频切割失败")
                return
            
            self.status_updated.emit(f"切割完成，生成 {len(clips)} 个片段")
            self._extract_poses_for_clips(clips)
            
            self.finished.emit(f"数据集构建完成！共处理 {len(clips)} 个片段")
            
        except Exception as e:
            self.error_occurred.emit(f"数据集构建失败: {e}")
    
    def _extract_poses_for_clips(self, clips):
        try:
            from model.pose_extract import get_pose_extractor
            from utils.config import KEYPOINTS_DIR, SEQUENCE_LENGTH, INPUT_SIZE
            
            pose_extractor = get_pose_extractor()
            os.makedirs(KEYPOINTS_DIR, exist_ok=True)
            
            all_poses = []
            all_action_labels = []
            all_style_labels = []
            
            for i, clip_path in enumerate(clips):
                cap = cv2.VideoCapture(clip_path)
                frames = []
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.resize(frame, INPUT_SIZE)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
                cap.release()
                
                if len(frames) >= SEQUENCE_LENGTH:
                    poses = pose_extractor.extract_sequence(frames[:SEQUENCE_LENGTH])
                    if poses is not None:
                        all_poses.append(poses)
                        # TODO: 这里需要根据文件名或其他方式获取标签
                        all_action_labels.append(0)
                        all_style_labels.append(0)
                
                self.progress_updated.emit(int((i + 1) / len(clips) * 100))
            
            # 保存为 npz 文件
            if all_poses:
                np.savez(
                    os.path.join(KEYPOINTS_DIR, 'dataset.npz'),
                    X=np.array(all_poses),
                    y_action=np.array(all_action_labels),
                    y_style=np.array(all_style_labels)
                )
                self.status_updated.emit(f"已保存 {len(all_poses)} 个骨架样本到 {KEYPOINTS_DIR}")
                
        except ImportError:
            self.status_updated.emit("骨架提取模块未就绪，跳过")