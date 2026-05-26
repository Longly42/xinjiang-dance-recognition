"""
thread_tools.py - 多线程工具类（独立骨架提取线程版）
包含：CaptureThread（采集+骨架提取线程）、RecordingThread（录屏）、DatasetBuildThread（数据集构建）
"""

import time
import os
import cv2
import numpy as np
from collections import deque
from datetime import datetime
from PyQt6.QtCore import QThread, pyqtSignal
import threading
import json
import random
import queue

from capture.frame_capture import ScreenCapture, VideoCapture
from capture.frame_process import FrameProcessor
from utils.config import (
    SEQUENCE_LENGTH, INPUT_SIZE,
    RECORDING_DIR, SLICED_DIR,
    FPS_TARGET, FRAME_SKIP, ASYNC_INFERENCE,
    LABEL_MAP_PATH
)
from utils.video_slicer import VideoSlicer

# 导入 st_model 模块
from st_model.predict import get_recognizer
from st_model.pose_extract_thread import PoseExtractThread


class MockInference:
    """Mock 推理接口（模型不可用时使用）"""
    
    def __init__(self):
        self.styles = ["赛乃姆", "刀郎舞", "盘子舞", "萨玛舞", "手鼓舞", "纳孜尔库姆"]
        print("[MockInference] 使用模拟推理")
    
    def predict_from_poses(self, poses):
        style = random.choice(self.styles)
        confidence = 0.6 + random.random() * 0.35
        return "", style, confidence


class CaptureThread(QThread):
    """采集与推理线程（独立高频骨架提取版）"""

    frame_ready = pyqtSignal(np.ndarray)      # 已绘制骨架的帧
    result_ready = pyqtSignal(str, str, float)  # 动作, 风格, 置信度
    status_updated = pyqtSignal(str)          # 状态更新（FPS等）
    error_occurred = pyqtSignal(str)          # 错误信息
    skeleton_ready = pyqtSignal(bool, int)    # 骨架检测状态 (是否检测到, 关键点数量)

    def __init__(self, source_type="screen", source_param=None, show_skeleton=True):
        super().__init__()
        self.source_type = source_type
        self.source_param = source_param
        self.show_skeleton = show_skeleton
        self.running = False
        self.paused = False

        # 帧处理器（尺寸调整、颜色转换）
        self.frame_processor = FrameProcessor(target_size=INPUT_SIZE)

        # 帧队列（供骨架提取线程使用）
        self.frame_queue = queue.Queue(maxsize=10)

        # 骨架历史队列（线程安全）
        self.skeleton_history = deque(maxlen=SEQUENCE_LENGTH)
        self.skeleton_lock = threading.Lock()

        # 启动独立的骨架提取线程
        self.pose_extract_thread = PoseExtractThread(
            frame_queue=self.frame_queue,
            skeleton_history=self.skeleton_history,
            skeleton_lock=self.skeleton_lock,
            max_history_len=SEQUENCE_LENGTH
        )
        self.pose_extract_thread.start()

        # 识别器（风格+动作规则）
        self.recognizer = None
        self._recognizer_lock = threading.Lock()
        self._init_recognizer()

        # 跳帧计数器
        self.skip_counter = 0
        self.frame_skip = FRAME_SKIP

        # 异步推理相关
        self.async_inference = ASYNC_INFERENCE
        self.last_inference_time = 0
        self.inference_interval = 0.5  # 每0.5秒推理一次

        # 最新骨架（用于绘制）
        self._latest_skeleton = None
        self._latest_skeleton_lock = threading.Lock()

        # 性能统计
        self.fps = 0
        self.frame_count = 0
        self.last_time = time.time()
        self.processing_times = deque(maxlen=30)

        # 采集器
        self.capturer = None
        self._init_capturer()

        print("[CaptureThread] 创建完成，骨架提取线程已启动")

    def _init_recognizer(self):
        """延迟加载识别器（避免阻塞启动）"""
        def load():
            with self._recognizer_lock:
                if self.recognizer is None:
                    try:
                        self.recognizer = get_recognizer()
                        print("[CaptureThread] 识别器加载完成")
                    except Exception as e:
                        print(f"[CaptureThread] 识别器加载失败: {e}")
                        self.recognizer = MockInference()
        threading.Thread(target=load, daemon=True).start()

    def _init_capturer(self):
        if self.source_type == "screen":
            if self.source_param:
                self.capturer = ScreenCapture(area=self.source_param)
            else:
                self.capturer = ScreenCapture()
        elif self.source_type == "video":
            self.capturer = VideoCapture(self.source_param)

    def get_current_poses(self):
        """获取最新骨架（供GUI绘制）"""
        with self._latest_skeleton_lock:
            return self._latest_skeleton.copy() if self._latest_skeleton is not None else None

    def _update_latest_skeleton(self):
        """从共享历史中取最新的骨架用于绘制"""
        with self.skeleton_lock:
            if self.skeleton_history:
                latest = self.skeleton_history[-1]
                if latest is not None:
                    with self._latest_skeleton_lock:
                        self._latest_skeleton = latest.copy()

    def _draw_skeleton_on_frame(self, frame, skeleton):
        """在帧上绘制骨架（优化版）"""
        h, w = frame.shape[:2]
        
        # MediaPipe 33个关键点的连接关系
        connections = [
            # 面部
            (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
            # 躯干
            (9, 10), (11, 12), (11, 23), (12, 24), (23, 24),
            # 左臂
            (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
            # 右臂
            (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
            # 左腿
            (23, 25), (25, 27), (27, 29), (27, 31),
            # 右腿
            (24, 26), (26, 28), (28, 30), (28, 32)
        ]
        
        # 转换为像素坐标
        points = []
        for i in range(min(len(skeleton), 33)):
            x = int(skeleton[i][0] * w)
            y = int(skeleton[i][1] * h)
            vis = skeleton[i][2] if len(skeleton[i]) > 2 else 1.0
            if vis > 0.5:
                points.append((x, y))
            else:
                points.append(None)
        
        # 绘制骨架连接（线条）
        for a, b in connections:
            if a < len(points) and b < len(points):
                pa, pb = points[a], points[b]
                if pa is not None and pb is not None:
                    cv2.line(frame, pa, pb, (0, 255, 0), 1, cv2.LINE_AA)
        
        # 绘制关键点（圆圈）
        for i, pt in enumerate(points):
            if pt is not None:
                # 不同部位不同颜色
                if i in [0, 1, 2, 3, 4, 5, 6, 7, 8]:  # 头部
                    color = (0, 0, 255)  # 红色
                    radius = 3
                elif i in [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]:  # 四肢
                    color = (255, 255, 0)  # 青色
                    radius = 2
                else:  # 躯干
                    color = (0, 255, 255)  # 黄色
                    radius = 2
                cv2.circle(frame, pt, radius, color, -1, cv2.LINE_AA)
        
        return frame

    def _get_skeleton_sequence(self):
        """获取当前最新的完整骨架序列（用于推理）"""
        with self.skeleton_lock:
            if len(self.skeleton_history) >= SEQUENCE_LENGTH:
                return np.array(list(self.skeleton_history)[-SEQUENCE_LENGTH:])
            else:
                return None

    def _try_async_inference(self):
        """尝试异步推理"""
        current_time = time.time()
        if current_time - self.last_inference_time < self.inference_interval:
            return
        
        skeleton_seq = self._get_skeleton_sequence()
        if skeleton_seq is not None:
            self.last_inference_time = current_time
            inference_thread = threading.Thread(
                target=self._run_inference_async,
                args=(skeleton_seq,)
            )
            inference_thread.daemon = True
            inference_thread.start()

    def _run_inference_async(self, skeleton_seq):
        """异步推理执行"""
        try:
            # 等待识别器加载完成
            timeout = 5
            start = time.time()
            while self.recognizer is None and time.time() - start < timeout:
                time.sleep(0.1)
            
            if self.recognizer is None:
                self.error_occurred.emit("识别器加载超时")
                return

            # 执行推理
            action, style, confidence = self.recognizer.predict_from_poses(skeleton_seq)
            
            # 更新骨架检测状态（基于最后一帧可见点数）
            if len(skeleton_seq) > 0:
                last_pose = skeleton_seq[-1]
                visible_count = np.sum(last_pose[:, 2] > 0.5) if last_pose is not None else 0
                self.skeleton_ready.emit(visible_count > 10, visible_count)
            else:
                self.skeleton_ready.emit(False, 0)
            
            self.result_ready.emit(action, style, confidence)
            
        except Exception as e:
            self.error_occurred.emit(f"推理失败: {e}")

    def _run_inference_sync(self):
        """同步推理（兼容模式）"""
        skeleton_seq = self._get_skeleton_sequence()
        if skeleton_seq is None:
            return
        
        try:
            if self.recognizer is None:
                return
            
            action, style, confidence = self.recognizer.predict_from_poses(skeleton_seq)
            
            if len(skeleton_seq) > 0:
                last_pose = skeleton_seq[-1]
                visible_count = np.sum(last_pose[:, 2] > 0.5) if last_pose is not None else 0
                self.skeleton_ready.emit(visible_count > 10, visible_count)
            else:
                self.skeleton_ready.emit(False, 0)
            
            self.result_ready.emit(action, style, confidence)
            
        except Exception as e:
            self.error_occurred.emit(f"推理失败: {e}")

    def run(self):
        """主循环"""
        self.running = True
        self.last_time = time.time()

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
                elapsed = time.time() - loop_start
                sleep_time = adaptive_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                continue

            # 处理帧
            process_start = time.time()
            processed_frame = self.frame_processor.process(frame)
            process_time = time.time() - process_start
            self.processing_times.append(process_time)

            if processed_frame is not None:
                # 发送帧到骨架提取线程的队列
                if not self.frame_queue.full():
                    self.frame_queue.put(processed_frame.copy())

                # 更新最新骨架（用于绘制）
                self._update_latest_skeleton()

                # 如果需要显示骨架，在帧上绘制
                if self.show_skeleton:
                    skeleton = self.get_current_poses()
                    if skeleton is not None:
                        processed_frame = self._draw_skeleton_on_frame(processed_frame, skeleton)

                # 发送帧到GUI显示
                self.frame_ready.emit(processed_frame)

                # 触发推理（异步或同步）
                if self.async_inference:
                    self._try_async_inference()
                else:
                    if len(self.skeleton_history) >= SEQUENCE_LENGTH:
                        self._run_inference_sync()

            # FPS 统计
            self.frame_count += 1
            now = time.time()
            if now - self.last_time >= 1.0:
                self.fps = self.frame_count / (now - self.last_time)
                avg_process_time = np.mean(self.processing_times) if self.processing_times else 0
                self.status_updated.emit(f"帧率: {self.fps:.1f} fps | 延迟: {avg_process_time*1000:.1f}ms")
                self.frame_count = 0
                self.last_time = now

                if avg_process_time > 0:
                    max_possible_fps = 1.0 / avg_process_time
                    if max_possible_fps < target_fps * 0.8:
                        adaptive_interval = avg_process_time * 1.2
                    else:
                        adaptive_interval = frame_interval

            # 控制帧率
            elapsed = time.time() - loop_start
            sleep_time = adaptive_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        # 清理资源
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

    def set_show_skeleton(self, show):
        """设置是否显示骨架"""
        self.show_skeleton = show

    def stop(self):
        """停止线程"""
        self.running = False
        self.wait()
        if hasattr(self, 'pose_extract_thread'):
            self.pose_extract_thread.stop()

    def pause(self):
        """暂停"""
        self.paused = True

    def resume(self):
        """恢复"""
        self.paused = False


class RecordingThread(QThread):
    """录屏线程"""

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

                if frame_count % 2 == 0:
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
    """数据集构建线程（骨架提取专用，用于从录制视频构建数据集）"""

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
            from st_model.pose_extract import get_pose_extractor
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
                        all_action_labels.append(0)
                        all_style_labels.append(0)

                self.progress_updated.emit(int((i + 1) / len(clips) * 100))

            if all_poses:
                np.savez(
                    os.path.join(KEYPOINTS_DIR, 'dataset_untagged.npz'),
                    X=np.array(all_poses),
                    y_action=np.array(all_action_labels),
                    y_style=np.array(all_style_labels)
                )
                self.status_updated.emit(f"已保存 {len(all_poses)} 个骨架样本到 {KEYPOINTS_DIR}")

        except ImportError:
            self.status_updated.emit("骨架提取模块未就绪，跳过")
        except Exception as e:
            self.status_updated.emit(f"骨架提取失败: {e}")