"""
video_slicer.py - 视频滑动窗口切割器
功能：将长视频按固定窗口和步长切割成短视频片段
"""

import cv2
import os
import math
from typing import List, Tuple
from datetime import datetime


class VideoSlicer:
    """视频滑动窗口切割器"""
    
    def __init__(self, window_sec: float = 2.0, step_sec: float = 1.0):
        """
        初始化切割器
        :param window_sec: 窗口长度（秒）
        :param step_sec: 步长（秒）
        """
        self.window_sec = window_sec
        self.step_sec = step_sec
    
    def slice_video(self, src_path: str, dst_folder: str, min_frames: int = 10) -> List[str]:
        """
        将视频切成多个小片段
        :param src_path: 源视频路径
        :param dst_folder: 输出文件夹
        :param min_frames: 最小帧数阈值（小于则丢弃）
        :return: 生成的片段路径列表
        """
        if not os.path.exists(src_path):
            print(f"[ERROR] 视频文件不存在: {src_path}")
            return []
        
        cap = cv2.VideoCapture(src_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if fps <= 0:
            print(f"[ERROR] 无效帧率: {src_path}")
            cap.release()
            return []
        
        # 计算窗口和步长对应的帧数
        window_frames = int(round(self.window_sec * fps))
        step_frames = int(round(self.step_sec * fps))
        
        if window_frames < min_frames:
            print(f"[WARN] 窗口帧数 {window_frames} 小于最小值 {min_frames}，跳过")
            cap.release()
            return []
        
        os.makedirs(dst_folder, exist_ok=True)
        basename = os.path.splitext(os.path.basename(src_path))[0]
        out_paths = []
        
        for start_frame in range(0, total_frames - window_frames + 1, step_frames):
            end_frame = start_frame + window_frames
            
            # 生成唯一文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            out_name = f"{basename}_{timestamp}_{start_frame}_{end_frame}.mp4"
            out_path = os.path.join(dst_folder, out_name)
            
            # 执行切割
            success = self._write_clip(cap, start_frame, end_frame, out_path, fps)
            if success:
                out_paths.append(out_path)
        
        cap.release()
        print(f"[INFO] 切割完成，生成 {len(out_paths)} 个片段")
        return out_paths
    
    def _write_clip(self, cap, start_frame: int, end_frame: int, out_path: str, fps: float) -> bool:
        """从 cap 中读取指定帧区间并保存为视频"""
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # 获取原始视频尺寸
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        
        frames_written = 0
        for _ in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            frames_written += 1
        
        out.release()
        return frames_written == (end_frame - start_frame)
    
    def slice_batch(self, video_paths: List[str], dst_folder: str) -> List[str]:
        """批量切割多个视频"""
        all_clips = []
        for video_path in video_paths:
            clips = self.slice_video(video_path, dst_folder)
            all_clips.extend(clips)
        return all_clips