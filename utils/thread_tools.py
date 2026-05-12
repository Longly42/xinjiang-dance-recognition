"""
thread_tools.py - 多线程工具类（高性能版 + 骨架提取与 ST-GCN 推理）
"""

import time
import os
import cv2
import numpy as np
from collections import deque
from datetime import datetime
from PyQt6.QtCore import QThread, pyqtSignal, QMutex, QWaitCondition
import threading
import json
import random

from capture.frame_capture import ScreenCapture, VideoCapture
from capture.frame_process import FrameProcessor
from utils.config import (
    SEQUENCE_LENGTH, INPUT_SIZE,
    RECORDING_DIR, SLICED_DIR,
    FPS_TARGET, FRAME_SKIP, ASYNC_INFERENCE,
    LABEL_MAP_PATH          # 新增导入标签映射路径
)
from utils.video_slicer import VideoSlicer

# 导入骨架提取与模型推理
try:
    from model.pose_extract import get_pose_extractor
    from model.predict import get_recognizer
    MODEL_AVAILABLE = True
except ImportError as e:
    MODEL_AVAILABLE = False
    print(f"[WARN] 模型模块未就绪: {e}，将使用 Mock 推理")


class MockInference:
    """Mock 推理接口（模型不可用时使用），标签从 label_map.json 动态加载"""
    
    def __init__(self):
        self.actions = []
        self.styles = []
        self._load_labels()
    
    def _load_labels(self):
        """从 label_map.json 加载动作和风格列表"""
        try:
            if os.path.exists(LABEL_MAP_PATH):
                with open(LABEL_MAP_PATH, 'r', encoding='utf-8') as f:
                    label_map = json.load(f)
                # label_map 格式: {"action": {"0":"扭颈动头", ...}, "style": {"0":"赛乃姆", ...}}
                action_dict = label_map.get("action", {})
                style_dict = label_map.get("style", {})
                # 按索引顺序提取名称
                self.actions = [action_dict[str(i)] for i in sorted(map(int, action_dict.keys()))]
                self.styles = [style_dict[str(i)] for i in sorted(map(int, style_dict.keys()))]
                print(f"[MockInference] 从 {LABEL_MAP_PATH} 加载标签: 动作 {len(self.actions)} 类, 风格 {len(self.styles)} 类")
            else:
                raise FileNotFoundError(f"标签文件不存在: {LABEL_MAP_PATH}")
        except Exception as e:
            print(f"[MockInference] 加载标签失败: {e}，使用默认标签")
            # 回退默认标签（与任务书一致）
            self.actions = ["扭颈动头", "翻腕绕腕", "摆肩抖肩", "垫步进退", "叉腰摆胯", "击掌拍手", "旋转", "行礼", "抬臂托帽"]
            self.styles = ["赛乃姆", "刀郎舞", "盘子舞", "萨玛舞", "夏地亚纳", "纳孜尔库姆"]
    
    def predict(self, frames: np.ndarray):
        """模拟推理，随机返回一个动作和风格"""
        action = random.choice(self.actions) if self.actions else "未知动作"
        style = random.choice(self.styles) if self.styles else "未知风格"
        confidence = 0.6 + random.random() * 0.35
        return action, style, confidence


class CaptureThread(QThread):
    """采集与推理线程（高性能优化版，集成骨架提取 + ST-GCN）"""

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

        # 帧处理器（尺寸调整、颜色转换）
        self.frame_processor = FrameProcessor(target_size=INPUT_SIZE)

        # 帧缓存
        self.frame_buffer = deque(maxlen=SEQUENCE_LENGTH)

        # 跳帧计数器
        self.skip_counter = 0
        self.frame_skip = FRAME_SKIP

        # 推理接口（优先真实模型，降级 Mock）
        self.use_mock = False
        self.pose_extractor = None
        self.recognizer = None

        if MODEL_AVAILABLE:
            try:
                self.pose_extractor = get_pose_extractor()
                self.recognizer = get_recognizer()
                print("[INFO] 使用真实模型（ST-GCN + MediaPipe）")
            except Exception as e:
                print(f"[WARN] 加载真实模型失败: {e}，使用 Mock 推理")
                self.use_mock = True
                self.inference = MockInference()
        else:
            self.use_mock = True
            self.inference = MockInference()

        # 异步推理相关
        self.async_inference = ASYNC_INFERENCE
        self.inference_lock = threading.Lock()
        self.last_inference_time = 0
        self.inference_interval = 0.5  # 每0.5秒推理一次

        # 性能统计
        self.fps = 0
        self.frame_count = 0
        self.last_time = time.time()
        self.processing_times = deque(maxlen=30)

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

        target_fps = FPS_TARGET
        frame_interval = 1.0 / target_fps
        adaptive_interval = frame_interval

        while self.running:
            if self.paused:
                self.msleep(10)
                continue

            loop_start = time.time()

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

            process_start = time.time()
            processed_frame = self.frame_processor.process(frame)
            process_time = time.time() - process_start
            self.processing_times.append(process_time)

            if processed_frame is not None:
                self.frame_ready.emit(processed_frame)
                self.frame_buffer.append(processed_frame)

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

                if avg_process_time > 0:
                    max_possible_fps = 1.0 / avg_process_time
                    if max_possible_fps < target_fps * 0.8:
                        adaptive_interval = avg_process_time * 1.2
                    else:
                        adaptive_interval = frame_interval

            elapsed = time.time() - loop_start
            sleep_time = adaptive_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

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
        """异步推理：不阻塞采集线程，先提取骨架再推理"""
        current_time = time.time()
        if current_time - self.last_inference_time < self.inference_interval:
            return

        if len(self.frame_buffer) >= SEQUENCE_LENGTH:
            self.last_inference_time = current_time
            frames_to_infer = list(self.frame_buffer)[-SEQUENCE_LENGTH:]

            inference_thread = threading.Thread(
                target=self._run_inference_async,
                args=(frames_to_infer,)
            )
            inference_thread.daemon = True
            inference_thread.start()

    def _run_inference_async(self, frames_list):
        """异步推理：先提取骨架，再调用模型"""
        try:
            if self.use_mock:
                frames_array = np.array(frames_list)
                action, style, confidence = self.inference.predict(frames_array)
            else:
                # 真实模型：先提取骨架
                if self.pose_extractor is None or self.recognizer is None:
                    return
                poses = self.pose_extractor.extract_sequence(frames_list)
                if poses is None:
                    # 未检测到人体，跳过本次推理
                    return
                action, style, confidence = self.recognizer.predict_from_poses(poses)

            self.result_ready.emit(action, style, confidence)
        except Exception as e:
            self.error_occurred.emit(f"推理失败: {e}")

    def _run_inference(self):
        """同步推理（保留兼容，但实际使用异步）"""
        try:
            frames_list = list(self.frame_buffer)
            if self.use_mock:
                frames_array = np.array(frames_list)
                action, style, confidence = self.inference.predict(frames_array)
            else:
                if self.pose_extractor is None or self.recognizer is None:
                    return
                poses = self.pose_extractor.extract_sequence(frames_list)
                if poses is None:
                    return
                action, style, confidence = self.recognizer.predict_from_poses(poses)
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
        
    def stop(self):
        self.running = False
        self.wait()
        # 释放 MediaPipe 资源
        if self.pose_extractor:
            self.pose_extractor.release()


# ---------- RecordingThread 和 DatasetBuildThread 保持不变 ----------
# （以下代码与之前相同，为完整起见也列出，但实际未改动）

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
                        # 无标签，暂时填0
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