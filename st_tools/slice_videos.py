"""
slice_videos.py - 智能10秒片段剪辑工具（适配已分类长视频）
功能：
1. 从 data/Used_video_clips/长视频/[风格]/ 读取长视频
2. 文件名格式：风格_序号.mp4（如：赛乃姆_001.mp4）
3. 切出10秒片段，输出到 st_data/raw_videos/[风格]/
4. 最后不足10秒的片段，从前面补帧凑够10秒（时间拉伸）
5. 自动适配原始视频帧率，保存时保持原始帧率
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# 添加项目根目录
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ==================== 配置 ====================
INPUT_RAW_VIDEOS = "./data/Used_video_clips/长视频"   # 原始长视频目录（已按风格分类）
OUTPUT_RAW_VIDEOS = "./st_data/raw_videos"           # 输出目录（10秒片段）

CLIP_DURATION = 10.0        # 目标片段时长（秒）
MIN_CLIP_DURATION = 8.0     # 最小可接受时长（低于此值才补帧）
# 不再固定 FPS_TARGET，而是保持原始帧率


# ==================== 核心函数 ====================

def get_video_info(video_path):
    """获取视频信息"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None, None, None
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, total_frames, width, height


def write_video_clip(frames, output_path, fps):
    """将帧列表写入视频文件（使用原始帧率）"""
    if not frames:
        return False
    
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    for frame in frames:
        out.write(frame)
    
    out.release()
    return True


def extract_clip(video_path, start_frame, end_frame):
    """提取视频中的一段连续帧"""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frames = []
    for _ in range(end_frame - start_frame):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    return frames


def time_stretch_frames(frames, target_frame_count):
    """
    时间拉伸：将帧列表通过插值凑够 target_frame_count 帧
    用于处理最后的不足片段
    """
    if len(frames) >= target_frame_count:
        return frames[:target_frame_count]
    
    original_count = len(frames)
    target_count = target_frame_count
    
    # 计算每帧的索引映射
    indices = np.linspace(0, original_count - 1, target_count)
    
    stretched_frames = []
    for idx in indices:
        src_idx = int(round(idx))
        src_idx = min(src_idx, original_count - 1)
        stretched_frames.append(frames[src_idx].copy())
    
    return stretched_frames


def smart_slice_video(video_path, output_dir, clip_duration=10.0, min_duration=8.0):
    """
    智能切片视频
    1. 优先切完整的 clip_duration 秒片段
    2. 最后不足 min_duration 秒的片段，从前面补帧凑够 clip_duration 秒
    """
    fps, total_frames, w, h = get_video_info(video_path)
    
    if fps is None:
        print(f"  [错误] 无法读取视频: {video_path}")
        return []
    
    total_duration = total_frames / fps
    clip_frames = int(clip_duration * fps)
    min_frames = int(min_duration * fps)
    
    print(f"    视频时长: {total_duration:.1f}s, 帧率: {fps:.1f}, 总帧数: {total_frames}")
    print(f"    目标片段: {clip_duration}s = {clip_frames}帧, 最小: {min_frames}帧")
    
    output_paths = []
    start_frame = 0
    clip_index = 0
    remaining_frames = total_frames
    
    # 获取文件名前缀（去掉序号）
    video_name = Path(video_path).stem
    base_name = video_name.split('_')[0] if '_' in video_name else video_name
    
    # ========== 第一阶段：切完整的10秒片段 ==========
    while remaining_frames >= clip_frames:
        end_frame = start_frame + clip_frames
        
        frames = extract_clip(video_path, start_frame, end_frame)
        
        if len(frames) == clip_frames:
            output_path = os.path.join(output_dir, f"{base_name}_clip{clip_index:03d}.mp4")
            if write_video_clip(frames, output_path, fps):
                output_paths.append(output_path)
                actual_duration = len(frames) / fps
                print(f"    ✅ 片段 {clip_index:03d}: {len(frames)}帧 ({actual_duration:.1f}s)")
            else:
                print(f"    ❌ 片段 {clip_index:03d}: 写入失败")
        
        start_frame = end_frame
        remaining_frames = total_frames - start_frame
        clip_index += 1
    
    # ========== 第二阶段：处理剩余不足的片段 ==========
    if remaining_frames > 0:
        remaining_duration = remaining_frames / fps
        print(f"    剩余 {remaining_frames} 帧 ({remaining_duration:.1f}s)，不足10秒，正在处理...")
        
        remaining_frames_list = extract_clip(video_path, start_frame, total_frames)
        
        if len(remaining_frames_list) >= min_frames:
            # 补帧到 clip_frames
            stretched_frames = time_stretch_frames(remaining_frames_list, clip_frames)
            output_path = os.path.join(output_dir, f"{base_name}_clip{clip_index:03d}_stretched.mp4")
            if write_video_clip(stretched_frames, output_path, fps):
                output_paths.append(output_path)
                actual_duration = len(stretched_frames) / fps
                print(f"    ✅ 补帧片段 {clip_index:03d}: 原{len(remaining_frames_list)}帧 → {len(stretched_frames)}帧 ({actual_duration:.1f}s)")
            else:
                print(f"    ❌ 补帧片段 {clip_index:03d}: 写入失败")
        else:
            # 剩余片段太短（< min_duration），丢弃并向前一个片段补帧
            print(f"    ⚠️ 剩余片段太短 ({len(remaining_frames_list)}帧，{len(remaining_frames_list)/fps:.1f}s)，丢弃")
            
            # 如果最后一个完整片段存在，给它补帧到10秒
            if output_paths:
                last_clip_path = output_paths[-1]
                print(f"    尝试向前一个片段补帧...")
                
                # 重新提取前一个片段 + 剩余片段
                last_start = (clip_index - 1) * clip_frames
                combined_frames = extract_clip(video_path, last_start, total_frames)
                
                if len(combined_frames) >= clip_frames:
                    # 已经够10秒或更多，直接截取
                    combined_frames = combined_frames[:clip_frames]
                    output_path = os.path.join(output_dir, f"{base_name}_clip{clip_index-1:03d}_extended.mp4")
                    if write_video_clip(combined_frames, output_path, fps):
                        os.remove(last_clip_path)
                        output_paths[-1] = output_path
                        actual_duration = len(combined_frames) / fps
                        print(f"    ✅ 已扩展前一个片段到 {len(combined_frames)}帧 ({actual_duration:.1f}s)")
                else:
                    # 仍然不足，进行时间拉伸
                    stretched_frames = time_stretch_frames(combined_frames, clip_frames)
                    output_path = os.path.join(output_dir, f"{base_name}_clip{clip_index-1:03d}_stretched.mp4")
                    if write_video_clip(stretched_frames, output_path, fps):
                        os.remove(last_clip_path)
                        output_paths[-1] = output_path
                        actual_duration = len(stretched_frames) / fps
                        print(f"    ✅ 已拉伸前一个片段: {len(combined_frames)}帧 → {len(stretched_frames)}帧 ({actual_duration:.1f}s)")
    
    return output_paths


def batch_slice_all_videos():
    """批量处理所有视频"""
    
    # 创建输出目录
    os.makedirs(OUTPUT_RAW_VIDEOS, exist_ok=True)
    
    # 统计
    total_videos = 0
    total_clips = 0
    style_clip_counts = {}
    
    print("=" * 70)
    print("              智能10秒片段剪辑工具")
    print("=" * 70)
    print(f"输入目录: {INPUT_RAW_VIDEOS}")
    print(f"输出目录: {OUTPUT_RAW_VIDEOS}")
    print(f"目标时长: {CLIP_DURATION} 秒")
    print(f"最小时长: {MIN_CLIP_DURATION} 秒")
    print("=" * 70)
    
    # 检查输入目录
    if not os.path.exists(INPUT_RAW_VIDEOS):
        print(f"\n❌ 错误: 输入目录不存在: {INPUT_RAW_VIDEOS}")
        print("   请确保目录结构为: data/Used_video_clips/长视频/[风格]/[视频文件]")
        return
    
    # 遍历风格目录
    for style_name in os.listdir(INPUT_RAW_VIDEOS):
        style_path = os.path.join(INPUT_RAW_VIDEOS, style_name)
        if not os.path.isdir(style_path):
            continue
        
        print(f"\n📁 风格: {style_name}")
        
        # 创建输出目录
        output_dir = os.path.join(OUTPUT_RAW_VIDEOS, style_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取该风格下所有视频文件
        video_files = []
        for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.MP4']:
            video_files.extend(Path(style_path).glob(ext))
        
        if not video_files:
            print(f"    没有找到视频文件")
            continue
        
        print(f"    找到 {len(video_files)} 个视频")
        
        style_clips = 0
        for video_path in tqdm(video_files, desc=f"    处理中"):
            total_videos += 1
            clips = smart_slice_video(str(video_path), output_dir, CLIP_DURATION, MIN_CLIP_DURATION)
            style_clips += len(clips)
            total_clips += len(clips)
        
        style_clip_counts[style_name] = style_clips
        print(f"    本风格共生成 {style_clips} 个片段")
    
    # 输出统计
    print("\n" + "=" * 70)
    print("剪辑完成！")
    print("=" * 70)
    print(f"处理视频数: {total_videos}")
    print(f"生成片段数: {total_clips}")
    print(f"输出目录: {OUTPUT_RAW_VIDEOS}")
    
    # 按风格统计
    print("\n各风格片段统计:")
    for style_name, count in style_clip_counts.items():
        print(f"  {style_name}: {count} 个片段")
    
    print("\n下一步: python -m st_model.extract_features")


def check_video_lengths(video_path):
    """检查视频时长"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames / fps if fps > 0 else 0


def analyze_input_videos():
    """分析输入视频的时长分布"""
    print("\n" + "=" * 70)
    print("输入视频时长分析")
    print("=" * 70)
    
    if not os.path.exists(INPUT_RAW_VIDEOS):
        print(f"❌ 目录不存在: {INPUT_RAW_VIDEOS}")
        return
    
    durations = []
    style_stats = {}
    
    for style_name in os.listdir(INPUT_RAW_VIDEOS):
        style_path = os.path.join(INPUT_RAW_VIDEOS, style_name)
        if not os.path.isdir(style_path):
            continue
        
        style_durations = []
        for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
            for video_path in Path(style_path).glob(ext):
                duration = check_video_lengths(str(video_path))
                if duration > 0:
                    durations.append(duration)
                    style_durations.append(duration)
        
        if style_durations:
            style_stats[style_name] = {
                'count': len(style_durations),
                'avg': np.mean(style_durations),
                'min': np.min(style_durations),
                'max': np.max(style_durations)
            }
    
    if durations:
        print(f"总视频数: {len(durations)}")
        print(f"平均时长: {np.mean(durations):.1f} 秒")
        print(f"最短时长: {np.min(durations):.1f} 秒")
        print(f"最长时长: {np.max(durations):.1f} 秒")
        
        print("\n各风格统计:")
        for style, stats in style_stats.items():
            print(f"  {style}: {stats['count']}个视频, "
                  f"时长 {stats['min']:.1f}-{stats['max']:.1f}秒, "
                  f"平均 {stats['avg']:.1f}秒")
    else:
        print("没有找到视频文件")


# ==================== 命令行入口 ====================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='智能10秒片段剪辑工具（适配已分类长视频）')
    parser.add_argument('--analyze', action='store_true', help='分析输入视频时长')
    parser.add_argument('--duration', type=float, default=10.0, help='目标片段时长（秒）')
    parser.add_argument('--min_duration', type=float, default=8.0, help='最小可接受时长（秒）')
    args = parser.parse_args()
    
    if args.duration != 10.0:
        CLIP_DURATION = args.duration
    if args.min_duration != 8.0:
        MIN_CLIP_DURATION = args.min_duration
    
    if args.analyze:
        analyze_input_videos()
    else:
        batch_slice_all_videos()