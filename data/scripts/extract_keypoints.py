# scripts/extract_keypoints.py
"""
骨架特征批量提取程序（鲁棒版）
功能：遍历 data/raw_videos/[风格]/[动作]/*.mp4，提取 MediaPipe 骨架序列
输出：data/keypoints/dataset.npz, data/keypoints/label_map.json
"""

import os
import sys
import cv2
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm

# ==================== 环境优化（避免 MediaPipe 卡死）====================
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
# ====================================================================

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ==================== 配置参数 ====================
RAW_VIDEOS_DIR = "data/raw_videos"      # 已标注视频文件夹
KEYPOINTS_DIR = "data/keypoints"        # 骨架特征输出文件夹

# 处理参数
SEQUENCE_LENGTH = 16        # 每个样本的帧数
INPUT_SIZE = (480, 360)     # 输入尺寸（宽,高）
MIN_FRAMES = 10             # 最小帧数阈值（少于则跳过）

# ==================== 标签映射（与 label_map.json 严格一致）====================
STYLE_MAP = {
    "赛乃姆": 0,
    "刀郎舞": 1,
    "盘子舞": 2,
    "萨玛舞": 3,
    "手鼓舞": 4,      # 原“夏地亚纳”已改为“手鼓舞”
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

# ==================== MediaPipe 初始化（鲁棒参数）====================
def get_pose_extractor():
    """获取 MediaPipe 姿态提取器（低阈值 + 快速模式）"""
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,          # 视频模式
        model_complexity=0,               # 0=最快, 1=平衡, 2=最准（推荐0提升速度）
        smooth_landmarks=True,
        min_detection_confidence=0.3,     # 降低检测阈值（原0.5过高）
        min_tracking_confidence=0.3
    )
    return pose, mp_pose


# ==================== 单视频骨架提取（鲁棒版：允许部分帧失败，用前一帧填充）====================
def extract_poses_from_video(video_path, pose, sequence_length=16):
    """
    从单个视频提取骨架序列
    返回：(poses, success)  成功返回 (T,33,3) 数组和 True
    失败返回 (None, False)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, False

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 缩放 + 颜色转换（BGR→RGB）
        frame = cv2.resize(frame, INPUT_SIZE)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()

    # 帧数不足检查
    if len(frames) < MIN_FRAMES:
        return None, False

    # 采样到固定长度 SEQUENCE_LENGTH
    if len(frames) > sequence_length:
        indices = np.linspace(0, len(frames) - 1, sequence_length, dtype=int)
        frames = [frames[i] for i in indices]
    elif len(frames) < sequence_length:
        # 帧数不足：重复最后一帧补齐
        while len(frames) < sequence_length:
            frames.append(frames[-1])

    # 提取骨架（允许部分帧失败）
    poses = []
    last_valid_pose = None
    failed_count = 0

    for frame in frames:
        results = pose.process(frame)
        if results.pose_landmarks:
            # 提取 33 个关键点的 (x, y, visibility)
            frame_pose = []
            for lm in results.pose_landmarks.landmark:
                frame_pose.append([lm.x, lm.y, lm.visibility])
            pose_array = np.array(frame_pose, dtype=np.float32)
            poses.append(pose_array)
            last_valid_pose = pose_array
        else:
            failed_count += 1
            # 未检测到：用上一帧有效骨架填充（如果没有则用零填充）
            if last_valid_pose is not None:
                poses.append(last_valid_pose.copy())
            else:
                poses.append(np.zeros((33, 3), dtype=np.float32))

    # 如果失败帧太多（超过 30%），认为视频不可用
    if failed_count > sequence_length * 0.3:
        return None, False

    if len(poses) != sequence_length:
        return None, False

    return np.array(poses, dtype=np.float32), True


# ==================== 主处理流程 ====================
def process_all_videos():
    """遍历 raw_videos，提取所有视频的骨架特征"""
    # 创建输出目录
    os.makedirs(KEYPOINTS_DIR, exist_ok=True)

    # 初始化 MediaPipe
    print("正在初始化 MediaPipe 姿态检测器...")
    pose, mp_pose = get_pose_extractor()
    print("初始化完成。")

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
    print("    骨架特征批量提取工具（鲁棒版）")
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
        if style_name not in STYLE_MAP:
            print(f"\n[警告] 未知风格: {style_name}，跳过（请在 STYLE_MAP 中添加）")
            continue

        style_id = STYLE_MAP[style_name]
        print(f"\n[风格] {style_name} (ID: {style_id})")

        # 遍历该风格下的动作
        for action_name in os.listdir(style_path):
            action_path = os.path.join(style_path, action_name)
            if not os.path.isdir(action_path):
                continue
            if action_name not in ACTION_MAP:
                print(f"  [警告] 未知动作: {action_name}，跳过（请在 ACTION_MAP 中添加）")
                continue

            action_id = ACTION_MAP[action_name]

            # 获取该动作下的所有视频文件
            video_files = []
            for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.MP4']:
                video_files.extend(Path(action_path).glob(ext))

            if not video_files:
                print(f"  [动作] {action_name} - 无视频文件")
                continue

            print(f"  [动作] {action_name} (ID: {action_id}, {len(video_files)} 个视频)")

            # 逐个处理视频
            for video_path in tqdm(video_files, desc=f"    {action_name}", leave=False):
                total_videos += 1
                poses, success = extract_poses_from_video(
                    str(video_path),
                    pose,
                    sequence_length=SEQUENCE_LENGTH
                )

                if success and poses is not None:
                    success_count += 1
                    all_poses.append(poses)
                    all_action_labels.append(action_id)
                    all_style_labels.append(style_id)
                else:
                    fail_count += 1
                    if fail_count <= 10:  # 只记录前10个失败原因
                        fail_reasons.append(f"    {video_path.name}: 骨架提取失败（检测率过低或帧数不足）")

    # 释放 MediaPipe 资源
    pose.close()

    # ==================== 生成汇总数据集 ====================
    print("\n" + "=" * 60)
    print("生成汇总数据集...")

    if all_poses:
        X = np.array(all_poses)
        y_action = np.array(all_action_labels)
        y_style = np.array(all_style_labels)

        # 保存为 npz 文件
        npz_path = os.path.join(KEYPOINTS_DIR, 'dataset.npz')
        np.savez(npz_path, X=X, y_action=y_action, y_style=y_style)

        # 保存标签映射（与 label_map.json 格式一致）
        label_map = {
            "style": {str(v): k for k, v in STYLE_MAP.items()},
            "action": {str(v): k for k, v in ACTION_MAP.items()}
        }
        label_map_path = os.path.join(KEYPOINTS_DIR, 'label_map.json')
        with open(label_map_path, 'w', encoding='utf-8') as f:
            json.dump(label_map, f, ensure_ascii=False, indent=2)

        print(f"\n{'='*60}")
        print("[成功] 数据集生成完成!")
        print(f"{'='*60}")
        print(f"  扫描视频数: {total_videos}")
        print(f"  成功提取: {success_count}")
        print(f"  失败跳过: {fail_count}")
        print(f"  有效样本: {len(all_poses)}")
        print(f"  骨架形状: {X.shape}")

        # 动作分布
        print(f"\n  动作分布:")
        action_counts = np.bincount(y_action, minlength=len(ACTION_MAP))
        for action_id, count in enumerate(action_counts):
            if count > 0:
                action_name = [k for k, v in ACTION_MAP.items() if v == action_id][0]
                print(f"    {action_name}: {count} 个样本 ({count/len(y_action)*100:.1f}%)")

        # 风格分布
        print(f"\n  风格分布:")
        style_counts = np.bincount(y_style, minlength=len(STYLE_MAP))
        for style_id, count in enumerate(style_counts):
            if count > 0:
                style_name = [k for k, v in STYLE_MAP.items() if v == style_id][0]
                print(f"    {style_name}: {count} 个样本 ({count/len(y_style)*100:.1f}%)")

        print(f"\n  保存位置:")
        print(f"    - 汇总数据: {npz_path}")
        print(f"    - 标签映射: {label_map_path}")

        # 显示失败示例
        if fail_reasons:
            print(f"\n  失败示例（前10个）:")
            for reason in fail_reasons:
                print(reason)

        print("\n下一步: 运行 python -m model.train 开始训练")
    else:
        print("\n[错误] 没有成功提取任何骨架数据！")
        print("请检查:")
        print("  1. raw_videos 文件夹是否有视频？")
        print(f"     当前路径: {os.path.abspath(RAW_VIDEOS_DIR)}")
        print("  2. 视频中是否有人体？尝试用测试脚本查看单帧检测效果")
        print("  3. 文件夹名称是否与 STYLE_MAP/ACTION_MAP 完全一致（中文必须精确匹配）")


# ==================== 辅助功能：数据集统计 ====================
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

    print("\n动作分布:")
    action_counts = np.bincount(y_action, minlength=len(ACTION_MAP))
    max_count = max(action_counts) if len(action_counts) > 0 else 1
    for action_id, count in enumerate(action_counts):
        if count > 0:
            action_name = [k for k, v in ACTION_MAP.items() if v == action_id][0]
            bar = "█" * int(count / max_count * 30)
            print(f"  {action_name:10s}: {count:4d} 个样本 {bar}")

    print("\n风格分布:")
    style_counts = np.bincount(y_style, minlength=len(STYLE_MAP))
    max_count = max(style_counts) if len(style_counts) > 0 else 1
    for style_id, count in enumerate(style_counts):
        if count > 0:
            style_name = [k for k, v in STYLE_MAP.items() if v == style_id][0]
            bar = "█" * int(count / max_count * 30)
            print(f"  {style_name:10s}: {count:4d} 个样本 {bar}")


def verify_folder_structure():
    """验证文件夹结构与标签映射是否匹配"""
    print("\n" + "=" * 60)
    print("验证文件夹结构")
    print("=" * 60)

    if not os.path.exists(RAW_VIDEOS_DIR):
        print(f"[错误] raw_videos 目录不存在: {RAW_VIDEOS_DIR}")
        return

    found_styles = []
    missing_styles = []
    for style_name in os.listdir(RAW_VIDEOS_DIR):
        style_path = os.path.join(RAW_VIDEOS_DIR, style_name)
        if not os.path.isdir(style_path):
            continue
        if style_name in STYLE_MAP:
            found_styles.append(style_name)
        else:
            missing_styles.append(style_name)

    found_actions = set()
    missing_actions = set()
    for style_name in found_styles:
        style_path = os.path.join(RAW_VIDEOS_DIR, style_name)
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
        print(f"\n未匹配的风格（需要添加到 STYLE_MAP 或修改文件夹名）:")
        for s in missing_styles:
            print(f"  ✗ {s}")

    print(f"\n已匹配的动作 ({len(found_actions)}/{len(ACTION_MAP)}):")
    for a in sorted(found_actions):
        print(f"  ✓ {a}")
    if missing_actions:
        print(f"\n未匹配的动作（需要添加到 ACTION_MAP 或修改文件夹名）:")
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

    if args.sequence != 16:
        SEQUENCE_LENGTH = args.sequence

    if args.verify:
        verify_folder_structure()
    elif args.check:
        check_dataset_stats()
    else:
        process_all_videos()