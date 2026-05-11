# scripts/extract_keypoints.py
"""
骨架特征批量提取程序
适配你的文件夹结构：
    raw_videos/[风格]/[动作]/*.mp4
输出：
    keypoints/dataset.npz (汇总数据集)
    keypoints/label_map.json (标签映射)
"""

import os
import sys
import cv2
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ==================== 配置参数 ====================
RAW_VIDEOS_DIR = "data/raw_videos"      # 已标注视频文件夹
KEYPOINTS_DIR = "data/keypoints"        # 骨架特征输出文件夹

# 处理参数
SEQUENCE_LENGTH = 16        # 每个样本的帧数（从40帧中均匀采样16帧）
INPUT_SIZE = (480, 360)     # 输入尺寸
MIN_FRAMES = 10             # 最小帧数阈值（少于则跳过）

# ==================== 标签映射（从你的 JSON 同步）====================
STYLE_MAP = {
    "赛乃姆": 0,
    "刀郎舞": 1,
    "盘子舞": 2,
    "萨玛舞": 3,
    "夏地亚纳": 4,
    "纳孜尔库姆": 5,
}

ACTION_MAP = {
    "扭颈动头": 0,
    "翻腕绕腕": 1,
    "摆肩抖肩": 2,
    "垫步进退": 3,
    "叉腰摆胯": 4,
    "击掌拍手": 5,
    "旋转": 6,
    "行礼": 7,
    "抬臂托帽": 8,
}


# ==================== 核心函数 ====================

def get_pose_extractor():
    """获取 MediaPipe 姿态提取器（延迟加载）"""
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    return pose, mp_pose


def extract_poses_from_video(video_path, pose, mp_pose, sequence_length=16):
    """
    从单个视频提取骨架序列
    返回：(poses, success) 成功返回 (T,33,3) 数组和 True
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, False
    
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 缩放
        frame = cv2.resize(frame, INPUT_SIZE)
        # BGR 转 RGB（MediaPipe 需要 RGB）
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    
    cap.release()
    
    # 检查帧数
    if len(frames) < MIN_FRAMES:
        return None, False
    
    # 如果视频帧数多于 sequence_length，均匀采样
    if len(frames) > sequence_length:
        indices = np.linspace(0, len(frames) - 1, sequence_length, dtype=int)
        frames = [frames[i] for i in indices]
    elif len(frames) < sequence_length:
        # 帧数不足，重复最后一帧补齐
        while len(frames) < sequence_length:
            frames.append(frames[-1])
    
    # 提取骨架
    poses = []
    for frame in frames:
        results = pose.process(frame)
        
        if results.pose_landmarks:
            # 提取关键点 (x, y, visibility)
            frame_pose = []
            for lm in results.pose_landmarks.landmark:
                frame_pose.append([lm.x, lm.y, lm.visibility])
            poses.append(np.array(frame_pose, dtype=np.float32))
        else:
            # 未检测到人体，返回失败
            return None, False
    
    if len(poses) != sequence_length:
        return None, False
    
    return np.array(poses, dtype=np.float32), True


def process_all_videos():
    """
    遍历 raw_videos 下所有视频，提取骨架特征
    """
    # 初始化
    os.makedirs(KEYPOINTS_DIR, exist_ok=True)
    pose, mp_pose = get_pose_extractor()
    
    # 统计信息
    total_videos = 0
    success_count = 0
    fail_count = 0
    fail_reasons = []
    
    # 数据集存储
    all_poses = []
    all_action_labels = []
    all_style_labels = []
    
    print("=" * 60)
    print("    骨架特征批量提取工具")
    print("=" * 60)
    print(f"输入目录: {RAW_VIDEOS_DIR}")
    print(f"输出目录: {KEYPOINTS_DIR}")
    print(f"序列长度: {SEQUENCE_LENGTH} 帧")
    print(f"输入尺寸: {INPUT_SIZE}")
    print("=" * 60)
    
    # 遍历每个风格
    for style_name in os.listdir(RAW_VIDEOS_DIR):
        style_path = os.path.join(RAW_VIDEOS_DIR, style_name)
        if not os.path.isdir(style_path):
            continue
        
        # 检查风格是否在映射中
        if style_name not in STYLE_MAP:
            print(f"\n[WARN] 未知风格: {style_name}，跳过（请在 STYLE_MAP 中添加）")
            continue
        
        style_id = STYLE_MAP[style_name]
        print(f"\n[风格] {style_name} (ID: {style_id})")
        
        # 遍历每个动作
        for action_name in os.listdir(style_path):
            action_path = os.path.join(style_path, action_name)
            if not os.path.isdir(action_path):
                continue
            
            # 检查动作是否在映射中
            if action_name not in ACTION_MAP:
                print(f"  [WARN] 未知动作: {action_name}，跳过（请在 ACTION_MAP 中添加）")
                continue
            
            action_id = ACTION_MAP[action_name]
            
            # 获取该动作下的所有视频
            video_files = []
            for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.MP4']:
                video_files.extend(Path(action_path).glob(ext))
            
            if not video_files:
                print(f"  [动作] {action_name} - 无视频文件")
                continue
            
            print(f"  [动作] {action_name} (ID: {action_id}, {len(video_files)} 个视频)")
            
            # 处理每个视频
            for video_path in tqdm(video_files, desc=f"    处理中", leave=False):
                total_videos += 1
                
                # 提取骨架
                poses, success = extract_poses_from_video(
                    str(video_path), 
                    pose, 
                    mp_pose,
                    sequence_length=SEQUENCE_LENGTH
                )
                
                if success and poses is not None:
                    success_count += 1
                    
                    # 添加到数据集
                    all_poses.append(poses)
                    all_action_labels.append(action_id)
                    all_style_labels.append(style_id)
                else:
                    fail_count += 1
                    if fail_count <= 5:  # 只记录前5个失败原因
                        fail_reasons.append(f"    {video_path.name}: 未检测到人体或帧数不足")
    
    # ==================== 生成汇总数据集 ====================
    print("\n" + "=" * 60)
    print("生成汇总数据集...")
    
    if all_poses:
        # 转换为 numpy 数组
        X = np.array(all_poses)
        y_action = np.array(all_action_labels)
        y_style = np.array(all_style_labels)
        
        # 保存为 npz 文件（供训练使用）
        npz_path = os.path.join(KEYPOINTS_DIR, 'dataset.npz')
        np.savez(npz_path, X=X, y_action=y_action, y_style=y_style)
        
        # 保存标签映射（与你的 label_map.json 格式一致）
        label_map = {
            "style": {str(v): k for k, v in STYLE_MAP.items()},
            "action": {str(v): k for k, v in ACTION_MAP.items()}
        }
        with open(os.path.join(KEYPOINTS_DIR, 'label_map.json'), 'w', encoding='utf-8') as f:
            json.dump(label_map, f, ensure_ascii=False, indent=2)
        
        print(f"\n{'='*60}")
        print(f"[SUCCESS] 数据集生成完成!")
        print(f"{'='*60}")
        print(f"  扫描视频数: {total_videos}")
        print(f"  成功提取: {success_count}")
        print(f"  失败跳过: {fail_count}")
        print(f"  有效样本: {len(all_poses)}")
        print(f"  骨架形状: {X.shape}")
        print(f"\n  动作分布:")
        action_counts = np.bincount(y_action)
        for action_id, count in enumerate(action_counts):
            if count > 0:
                action_name = [k for k, v in ACTION_MAP.items() if v == action_id][0]
                print(f"    {action_name}: {count} 个样本 ({count/len(y_action)*100:.1f}%)")
        
        print(f"\n  风格分布:")
        style_counts = np.bincount(y_style)
        for style_id, count in enumerate(style_counts):
            if count > 0:
                style_name = [k for k, v in STYLE_MAP.items() if v == style_id][0]
                print(f"    {style_name}: {count} 个样本 ({count/len(y_style)*100:.1f}%)")
        
        print(f"\n  保存位置:")
        print(f"    - 汇总数据: {npz_path}")
        print(f"    - 标签映射: {KEYPOINTS_DIR}/label_map.json")
        
        # 如果有失败，显示原因
        if fail_reasons:
            print(f"\n  失败示例（前5个）:")
            for reason in fail_reasons:
                print(reason)
    else:
        print("\n[ERROR] 没有成功提取任何骨架数据！")
        print("请检查:")
        print("  1. raw_videos 文件夹是否有视频？")
        print(f"     当前路径: {os.path.abspath(RAW_VIDEOS_DIR)}")
        print("  2. 视频中是否有人体？")
        print("  3. 文件夹名称是否与 STYLE_MAP/ACTION_MAP 完全一致？（中文必须精确匹配）")
    
    print(f"\n下一步: 运行 python model/train.py 开始训练")
    
    # 释放资源
    pose.close()


def check_dataset_stats():
    """检查已提取的数据集统计信息"""
    npz_path = os.path.join(KEYPOINTS_DIR, 'dataset.npz')
    
    if not os.path.exists(npz_path):
        print("数据集不存在，请先运行提取程序")
        return
    
    data = np.load(npz_path)
    X = data['X']
    y_action = data['y_action']
    y_style = data['y_style']
    
    print("\n" + "=" * 60)
    print("数据集统计")
    print("=" * 60)
    print(f"总样本数: {len(X)}")
    print(f"骨架形状: {X.shape}")
    
    # 动作分布
    print("\n动作分布:")
    action_counts = np.bincount(y_action)
    for action_id in range(len(ACTION_MAP)):
        count = action_counts[action_id] if action_id < len(action_counts) else 0
        action_name = [k for k, v in ACTION_MAP.items() if v == action_id][0]
        bar = "█" * int(count / max(action_counts) * 30) if max(action_counts) > 0 else ""
        print(f"  {action_name:10s}: {count:4d} 个样本 {bar}")
    
    # 风格分布
    print("\n风格分布:")
    style_counts = np.bincount(y_style)
    for style_id in range(len(STYLE_MAP)):
        count = style_counts[style_id] if style_id < len(style_counts) else 0
        style_name = [k for k, v in STYLE_MAP.items() if v == style_id][0]
        bar = "█" * int(count / max(style_counts) * 30) if max(style_counts) > 0 else ""
        print(f"  {style_name:10s}: {count:4d} 个样本 {bar}")


def verify_folder_structure():
    """验证文件夹结构与标签映射是否匹配"""
    print("\n" + "=" * 60)
    print("验证文件夹结构")
    print("=" * 60)
    
    found_styles = []
    found_actions = set()
    missing_styles = []
    missing_actions = set()
    
    if not os.path.exists(RAW_VIDEOS_DIR):
        print(f"[ERROR] raw_videos 目录不存在: {RAW_VIDEOS_DIR}")
        return
    
    for style_name in os.listdir(RAW_VIDEOS_DIR):
        style_path = os.path.join(RAW_VIDEOS_DIR, style_name)
        if not os.path.isdir(style_path):
            continue
        
        if style_name in STYLE_MAP:
            found_styles.append(style_name)
        else:
            missing_styles.append(style_name)
        
        for action_name in os.listdir(style_path):
            action_path = os.path.join(style_path, action_name)
            if not os.path.isdir(action_path):
                continue
            
            if action_name in ACTION_MAP:
                found_actions.add(action_name)
            else:
                missing_actions.add(action_name)
    
    print(f"\n已匹配的风格 ({len(found_styles)}/{len(STYLE_MAP)}):")
    for s in found_styles:
        print(f"  ✓ {s}")
    
    if missing_styles:
        print(f"\n未匹配的风格（需要添加到 STYLE_MAP）:")
        for s in missing_styles:
            print(f"  ✗ {s}")
    
    print(f"\n已匹配的动作 ({len(found_actions)}/{len(ACTION_MAP)}):")
    for a in sorted(found_actions):
        print(f"  ✓ {a}")
    
    if missing_actions:
        print(f"\n未匹配的动作（需要添加到 ACTION_MAP）:")
        for a in sorted(missing_actions):
            print(f"  ✗ {a}")


# ==================== 命令行入口 ====================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='骨架特征批量提取')
    parser.add_argument('--check', action='store_true', help='仅查看数据集统计')
    parser.add_argument('--verify', action='store_true', help='验证文件夹结构')
    parser.add_argument('--sequence', type=int, default=16, help='序列长度（帧数）')
    args = parser.parse_args()
    
    if args.verify:
        verify_folder_structure()
    elif args.check:
        check_dataset_stats()
    else:
        if args.sequence != 16:
            SEQUENCE_LENGTH = args.sequence
        process_all_videos()