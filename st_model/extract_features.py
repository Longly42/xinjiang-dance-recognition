"""
extract_features.py – 从 st_data/raw_videos 读取10秒长片段，提取骨架
输入目录: st_data/raw_videos/[风格]/[视频文件]
输出: st_data/keypoints/dataset_long.npz
"""

import os
import sys
import cv2
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm

# 添加项目根目录
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ==================== 配置 ====================
ST_DATA_ROOT = "./st_data"                    # 新数据根目录
RAW_VIDEOS_DIR = os.path.join(ST_DATA_ROOT, "raw_videos")   # 输入视频
KEYPOINTS_DIR = os.path.join(ST_DATA_ROOT, "keypoints")     # 输出骨架
SEQUENCE_LENGTH = 64                          # 采样帧数
INPUT_SIZE = (480, 360)                       # 输入尺寸

# 风格映射（与 label_map.json 一致）
STYLE_MAP = {
    "赛乃姆": 0,
    "刀郎舞": 1,
    "盘子舞": 2,
    "萨玛舞": 3,
    "手鼓舞": 4,
    "纳孜尔库姆": 5,
}

# ==================== MediaPipe 初始化 ====================
def get_pose_extractor():
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=0,               # 快速模式
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    )
    return pose, mp_pose


# ==================== 单视频骨架提取 ====================
def extract_skeleton_from_video(video_path, pose):
    """
    从单个视频文件提取骨架序列（假设视频已经是10秒左右）
    返回: (T, 33, 3) 数组，T = SEQUENCE_LENGTH
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, INPUT_SIZE)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    
    if len(frames) < 30:  # 至少1秒以上，否则跳过
        return None
    
    # 采样到固定长度 SEQUENCE_LENGTH
    if len(frames) > SEQUENCE_LENGTH:
        indices = np.linspace(0, len(frames) - 1, SEQUENCE_LENGTH, dtype=int)
        frames = [frames[i] for i in indices]
    elif len(frames) < SEQUENCE_LENGTH:
        # 帧数不足，重复最后一帧补齐
        while len(frames) < SEQUENCE_LENGTH:
            frames.append(frames[-1])
    
    # 提取骨架（允许部分帧失败，用前一帧填充）
    poses = []
    last_valid = None
    failed = 0
    
    for frame in frames:
        results = pose.process(frame)
        if results.pose_landmarks:
            frame_pose = []
            for lm in results.pose_landmarks.landmark:
                frame_pose.append([lm.x, lm.y, lm.visibility])
            pose_array = np.array(frame_pose, dtype=np.float32)
            poses.append(pose_array)
            last_valid = pose_array
        else:
            failed += 1
            if last_valid is not None:
                poses.append(last_valid.copy())
            else:
                poses.append(np.zeros((33, 3), dtype=np.float32))
    
    # 失败帧太多则丢弃
    if failed > SEQUENCE_LENGTH * 0.3:
        return None
    
    if len(poses) != SEQUENCE_LENGTH:
        return None
    
    return np.array(poses, dtype=np.float32)


# ==================== 主处理函数 ====================
def extract_all_videos():
    """遍历 st_data/raw_videos 下所有视频，提取骨架"""
    
    # 创建输出目录
    os.makedirs(KEYPOINTS_DIR, exist_ok=True)
    
    # 初始化 MediaPipe
    print("正在初始化 MediaPipe...")
    pose, _ = get_pose_extractor()
    print("初始化完成")
    
    # 检查输入目录
    if not os.path.exists(RAW_VIDEOS_DIR):
        print(f"错误: 目录不存在 {RAW_VIDEOS_DIR}")
        print("请创建以下目录结构:")
        print("  st_data/raw_videos/赛乃姆/*.mp4")
        print("  st_data/raw_videos/刀郎舞/*.mp4")
        print("  st_data/raw_videos/盘子舞/*.mp4")
        print("  st_data/raw_videos/萨玛舞/*.mp4")
        print("  st_data/raw_videos/手鼓舞/*.mp4")
        print("  st_data/raw_videos/纳孜尔库姆/*.mp4")
        return
    
    # 统计
    all_poses = []
    all_styles = []
    total_videos = 0
    success_count = 0
    
    print("\n" + "=" * 60)
    print("开始提取10秒长片段骨架")
    print("=" * 60)
    
    # 遍历每个风格
    for style_name in STYLE_MAP.keys():
        style_path = os.path.join(RAW_VIDEOS_DIR, style_name)
        if not os.path.exists(style_path):
            print(f"\n⚠️ 风格目录不存在: {style_name}，跳过")
            continue
        
        style_id = STYLE_MAP[style_name]
        print(f"\n📁 风格: {style_name}")
        
        # 获取该风格下所有视频
        video_files = []
        for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
            video_files.extend(Path(style_path).glob(ext))
        
        if not video_files:
            print(f"  没有找到视频文件")
            continue
        
        print(f"  找到 {len(video_files)} 个视频")
        
        for video_path in tqdm(video_files, desc=f"  处理中"):
            total_videos += 1
            poses = extract_skeleton_from_video(str(video_path), pose)
            
            if poses is not None:
                all_poses.append(poses)
                all_styles.append(style_id)
                success_count += 1
    
    # 释放资源
    pose.close()
    
    # 保存结果
    print("\n" + "=" * 60)
    print("保存数据集")
    print("=" * 60)
    
    if all_poses:
        X = np.array(all_poses)
        y_style = np.array(all_styles)
        
        np.savez(os.path.join(KEYPOINTS_DIR, 'dataset_long.npz'), 
                 X=X, y_style=y_style)
        
        print(f"✅ 保存成功!")
        print(f"  输入视频数: {total_videos}")
        print(f"  成功提取: {success_count}")
        print(f"  样本形状: {X.shape}")
        print(f"\n  风格分布:")
        for i, name in enumerate(STYLE_MAP.keys()):
            count = np.sum(y_style == i)
            print(f"    {name}: {count} 个样本")
        
        # 同时保存标签映射
        label_map = {
            "style": {str(v): k for k, v in STYLE_MAP.items()}
        }
        with open(os.path.join(KEYPOINTS_DIR, 'label_map.json'), 'w', encoding='utf-8') as f:
            json.dump(label_map, f, ensure_ascii=False, indent=2)
        
        print(f"\n输出目录: {KEYPOINTS_DIR}")
        print(f"  - dataset_long.npz (骨架数据)")
        print(f"  - label_map.json (标签映射)")
    else:
        print("❌ 没有成功提取任何骨架数据")
    
    print("\n下一步: python -m st_model.train")


# ==================== 辅助函数 ====================
def check_dataset():
    """检查已生成的数据集"""
    npz_path = os.path.join(KEYPOINTS_DIR, 'dataset_long.npz')
    if not os.path.exists(npz_path):
        print("数据集不存在，请先运行 extract_features.py")
        return
    
    data = np.load(npz_path)
    X = data['X']
    y_style = data['y_style']
    
    print("\n" + "=" * 60)
    print("数据集统计")
    print("=" * 60)
    print(f"样本数: {len(X)}")
    print(f"形状: {X.shape}")
    print(f"\n风格分布:")
    for i, name in enumerate(STYLE_MAP.keys()):
        count = np.sum(y_style == i)
        bar = "█" * int(count / max(1, np.max(np.bincount(y_style))) * 30)
        print(f"  {name}: {count} 个样本 {bar}")


def verify_folder_structure():
    """验证文件夹结构"""
    print("\n" + "=" * 60)
    print("验证文件夹结构")
    print("=" * 60)
    
    if not os.path.exists(RAW_VIDEOS_DIR):
        print(f"❌ 目录不存在: {RAW_VIDEOS_DIR}")
        print("\n请创建以下结构:")
        print("  st_data/")
        print("  └── raw_videos/")
        print("      ├── 赛乃姆/")
        print("      │   ├── clip1.mp4")
        print("      │   └── ...")
        print("      ├── 刀郎舞/")
        print("      └── ...")
        return
    
    print(f"✅ 根目录存在: {RAW_VIDEOS_DIR}")
    
    for style_name in STYLE_MAP.keys():
        style_path = os.path.join(RAW_VIDEOS_DIR, style_name)
        if os.path.exists(style_path):
            video_count = len(list(Path(style_path).glob("*.mp4")))
            print(f"  ✅ {style_name}: {video_count} 个视频")
        else:
            print(f"  ❌ {style_name}: 目录不存在")


# ==================== 命令行入口 ====================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='10秒长片段骨架提取')
    parser.add_argument('--check', action='store_true', help='查看数据集统计')
    parser.add_argument('--verify', action='store_true', help='验证文件夹结构')
    args = parser.parse_args()
    
    if args.verify:
        verify_folder_structure()
    elif args.check:
        check_dataset()
    else:
        extract_all_videos()