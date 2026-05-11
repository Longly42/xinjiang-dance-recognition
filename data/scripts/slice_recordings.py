# scripts/slice_recordings.py
"""
视频自动切片程序
功能：将 recordings 文件夹中的所有视频，按固定时长切成短视频片段
输入：data/recordings/*.mp4
输出：data/sliced_clips/*.mp4 (无标签，临时存储)
"""

import os
import cv2
import sys
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ==================== 配置参数 ====================
RECORDINGS_DIR = "data/recordings"      # 原始录屏文件夹
SLICED_DIR = "data/sliced_clips"        # 切片输出文件夹

# 切片参数
WINDOW_SEC = 2.0        # 窗口长度（秒），2-3秒推荐
STEP_SEC = 2.0          # 步长（秒），等于窗口长度表示不重叠，小于表示有重叠
MIN_FRAMES = 30         # 最小帧数阈值（小于则丢弃）

# ==================== 核心函数 ====================

def get_video_info(video_path):
    """获取视频的基本信息"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None, None, None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    cap.release()
    return fps, total_frames, width, height


def slice_single_video(video_path, output_dir, window_sec=2.0, step_sec=2.0, min_frames=30):
    """
    将单个视频切成多个片段
    返回：生成的片段路径列表
    """
    # 获取视频信息
    fps, total_frames, width, height = get_video_info(video_path)
    
    if fps is None:
        print(f"  [ERROR] 无法读取视频: {video_path}")
        return []
    
    if fps <= 0:
        print(f"  [ERROR] 无效帧率: {fps}")
        return []
    
    # 计算窗口和步长对应的帧数
    window_frames = int(round(window_sec * fps))
    step_frames = int(round(step_sec * fps))
    
    # 检查最小帧数要求
    if window_frames < min_frames:
        print(f"  [WARN] 窗口帧数 {window_frames} < {min_frames}，跳过")
        return []
    
    # 如果视频太短，跳过
    if total_frames < window_frames:
        print(f"  [WARN] 视频太短 ({total_frames}帧 < {window_frames}帧)，跳过")
        return []
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    
    # 生成输出文件名前缀（使用原视频名+时间戳）
    video_name = Path(video_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    out_paths = []
    slice_idx = 0
    
    # 滑动窗口切割
    for start_frame in range(0, total_frames - window_frames + 1, step_frames):
        end_frame = start_frame + window_frames
        
        # 生成输出文件名
        out_name = f"{video_name}_slice{slice_idx:04d}_{start_frame}_{end_frame}.mp4"
        out_path = os.path.join(output_dir, out_name)
        
        # 写入切片
        success = write_video_clip(cap, start_frame, end_frame, out_path, fps, (width, height))
        
        if success:
            out_paths.append(out_path)
            slice_idx += 1
    
    cap.release()
    return out_paths


def write_video_clip(cap, start_frame, end_frame, out_path, fps, size):
    """
    从视频捕获对象中读取指定帧区间并保存为视频文件
    """
    # 设置读取位置
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    width, height = size
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
    
    # 验证写入的帧数是否正确
    return frames_written == (end_frame - start_frame)


def process_all_recordings():
    """
    处理 recordings 文件夹中的所有视频
    """
    # 创建输出目录
    os.makedirs(SLICED_DIR, exist_ok=True)
    
    # 获取所有视频文件
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(Path(RECORDINGS_DIR).glob(f"*{ext}"))
        video_files.extend(Path(RECORDINGS_DIR).glob(f"*{ext.upper()}"))
    
    # 去重
    video_files = list(set(video_files))
    
    if not video_files:
        print(f"[ERROR] 在 {RECORDINGS_DIR} 中没有找到视频文件")
        print("请先将录制的视频放到 data/recordings/ 文件夹中")
        return
    
    print(f"[INFO] 找到 {len(video_files)} 个视频文件")
    print(f"[INFO] 切片参数: 窗口={WINDOW_SEC}秒, 步长={STEP_SEC}秒, 最小帧数={MIN_FRAMES}")
    print("-" * 50)
    
    total_clips = 0
    
    for video_path in video_files:
        print(f"\n[处理] {video_path.name}")
        
        clips = slice_single_video(
            str(video_path), 
            SLICED_DIR,
            window_sec=WINDOW_SEC,
            step_sec=STEP_SEC,
            min_frames=MIN_FRAMES
        )
        
        print(f"  => 生成 {len(clips)} 个切片")
        total_clips += len(clips)
    
    print("\n" + "=" * 50)
    print(f"[完成] 总共生成 {total_clips} 个视频切片")
    print(f"[位置] {os.path.abspath(SLICED_DIR)}")
    print("\n下一步: 人工筛选这些切片，按风格/动作分类放入 raw_videos/")
    print("  raw_videos/[风格]/[动作]/xxx.mp4")


# ==================== 命令行入口 ====================
if __name__ == "__main__":
    print("=" * 50)
    print("    视频自动切片工具")
    print("=" * 50)
    print(f"输入目录: {RECORDINGS_DIR}")
    print(f"输出目录: {SLICED_DIR}")
    print("=" * 50)
    
    process_all_recordings()