"""
extract_for_stmodel.py - 智能多人选择骨架提取工具
功能：
1. 自动检测视频中的多个人体
2. 智能选择最完整的人体（关键点最多、置信度最高）
3. 提取该人的骨架序列用于训练
4. 输出质量报告和统计数据
"""

import os
import cv2
import numpy as np
import json
from collections import deque

os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

print("导入 mediapipe...")
import mediapipe as mp
print("导入完成")

# ==================== 配置 ====================
RAW_VIDEOS_DIR = "st_data/raw_videos_augmented"
KEYPOINTS_DIR = "st_data/keypoints"
INPUT_SIZE = (480, 360)
SEQUENCE_LENGTH = 64

# 风格映射（6类）
STYLE_MAP = {
    "赛乃姆": 0,
    "刀郎舞": 1,
    "盘子舞": 2,
    "萨玛舞": 3,
    "手鼓舞": 4,
    "纳孜尔库姆": 5,
}

# 提取质量阈值
MIN_VISIBLE_KEYPOINTS = 15      # 最少可见关键点数量
MIN_FRAME_SUCCESS_RATE = 0.7    # 最少成功帧比例
QUALITY_SCORE_THRESHOLD = 0.3   # 最低质量得分


class MultiPersonPoseExtractor:
    """多人姿态提取器，自动选择最完整的人体"""
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,               # 提高检测精度
            smooth_landmarks=True,            # 平滑关键点
            min_detection_confidence=0.5,     # 检测置信度阈值
            min_tracking_confidence=0.5,      # 跟踪置信度阈值
            enable_segmentation=False         # 禁用分割提高速度
        )
        
        # 缓存上一帧的关键点（用于帧间平滑）
        self.prev_poses = deque(maxlen=5)
        
    def select_best_person(self, pose_landmarks_list, pose_detections_list=None):
        """
        从多人中选择最完整的人体
        
        选择标准：
        1. 可见关键点数量 (visibility > 0.5)
        2. 关键点平均可见度
        3. 检测置信度（如果有）
        
        Returns:
            (landmarks_array, person_index, quality_score)
        """
        if not pose_landmarks_list:
            return None, -1, 0.0
        
        best_idx = 0
        best_score = -1.0
        best_landmarks = None
        
        for idx, landmarks in enumerate(pose_landmarks_list):
            # 统计可见关键点
            visible_count = 0
            total_visibility = 0.0
            
            for lm in landmarks.landmark:
                if lm.visibility > 0.5:
                    visible_count += 1
                total_visibility += lm.visibility
            
            avg_visibility = total_visibility / 33.0
            
            # 获取检测置信度
            detection_score = 0.5
            if pose_detections_list and idx < len(pose_detections_list):
                detection_score = pose_detections_list[idx].score[0] if pose_detections_list[idx].score else 0.5
            
            # 综合得分（权重可调）
            # 可见关键点数量最重要（权重2.0），平均可见度次之（权重1.5），检测置信度最低（权重1.0）
            score = (visible_count * 2.0) + (avg_visibility * 15.0) + (detection_score * 5.0)
            
            # 帧间一致性奖励（如果与上一帧相似，提高分数）
            if len(self.prev_poses) > 0:
                prev_landmarks = self.prev_poses[-1]
                if prev_landmarks is not None:
                    # 计算与上一帧的平均距离（简化版）
                    prev_array = np.array([[lm.x, lm.y] for lm in prev_landmarks.landmark])
                    curr_array = np.array([[lm.x, lm.y] for lm in landmarks.landmark])
                    movement = np.mean(np.linalg.norm(curr_array - prev_array, axis=1))
                    if movement < 0.1:  # 移动较小，认为是同一个人
                        score += 10.0
            
            if score > best_score:
                best_score = score
                best_idx = idx
                best_landmarks = landmarks
        
        # 提取关键点坐标
        if best_landmarks:
            frame_pose = []
            for lm in best_landmarks.landmark:
                frame_pose.append([lm.x, lm.y, lm.visibility])
            
            # 缓存当前帧的 landmarks 用于下一帧
            self.prev_poses.append(best_landmarks)
            
            return np.array(frame_pose, dtype=np.float32), best_idx, best_score
        
        return None, -1, 0.0
    
    def extract_from_frame(self, frame_rgb):
        """
        从单帧RGB图像提取骨架（自动选择最佳人体）
        
        Returns:
            (pose_array, person_count, quality_score, visible_keypoints)
            pose_array: (33, 3) 或 None
            person_count: 检测到的人数
            quality_score: 质量得分
            visible_keypoints: 可见关键点数量
        """
        results = self.pose.process(frame_rgb)
        
        # 统计检测到的人数
        person_count = 0
        pose_landmarks_list = []
        pose_detections_list = []
        
        # MediaPipe Pose 返回单个结果，但可以通过 pose_detections 获取多人
        if results.pose_landmarks:
            pose_landmarks_list.append(results.pose_landmarks)
            person_count += 1
        
        # 尝试获取多个检测结果
        if hasattr(results, 'pose_detections') and results.pose_detections:
            pose_detections_list = results.pose_detections
            # 如果有多个检测但只有一个 landmarks，需要扩展
            if len(results.pose_detections) > person_count:
                person_count = len(results.pose_detections)
        
        # 选择最佳人体
        pose_array, person_idx, quality_score = self.select_best_person(
            pose_landmarks_list, pose_detections_list
        )
        
        if pose_array is not None:
            visible_keypoints = np.sum(pose_array[:, 2] > 0.5)
        else:
            visible_keypoints = 0
        
        return pose_array, person_count, quality_score, visible_keypoints
    
    def extract_sequence(self, frames_rgb_list, show_progress=False):
        """
        从帧序列提取骨架序列
        
        Returns:
            (poses_sequence, metadata)
            poses_sequence: (T, 33, 3) 数组
            metadata: 包含每帧的统计信息
        """
        poses = []
        metadata = {
            'person_counts': [],
            'quality_scores': [],
            'visible_keypoints': [],
            'failed_frames': []
        }
        
        total_frames = len(frames_rgb_list)
        
        for idx, frame in enumerate(frames_rgb_list):
            pose_array, person_count, quality_score, visible_kp = self.extract_from_frame(frame)
            
            if pose_array is not None:
                poses.append(pose_array)
                metadata['person_counts'].append(person_count)
                metadata['quality_scores'].append(quality_score)
                metadata['visible_keypoints'].append(visible_kp)
            else:
                # 提取失败，用前一帧填充
                metadata['failed_frames'].append(idx)
                if poses:
                    poses.append(poses[-1].copy())
                    metadata['person_counts'].append(metadata['person_counts'][-1])
                    metadata['quality_scores'].append(0.0)
                    metadata['visible_keypoints'].append(0)
                else:
                    poses.append(np.zeros((33, 3), dtype=np.float32))
                    metadata['person_counts'].append(0)
                    metadata['quality_scores'].append(0.0)
                    metadata['visible_keypoints'].append(0)
            
            # 进度显示
            if show_progress and (idx + 1) % 50 == 0:
                print(f"    处理帧: {idx+1}/{total_frames}")
        
        return np.array(poses, dtype=np.float32), metadata
    
    def release(self):
        self.pose.close()


def extract_skeleton_from_video(video_path, extractor, target_frames=SEQUENCE_LENGTH):
    """
    从单个视频提取骨架序列
    
    Returns:
        (poses, metadata, success)
    """
    # 读取视频
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 调整尺寸并转换颜色
        frame = cv2.resize(frame, INPUT_SIZE)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    
    if len(frames) < 30:  # 至少1秒
        return None, None, False
    
    # 采样到目标帧数
    if len(frames) > target_frames:
        indices = np.linspace(0, len(frames)-1, target_frames, dtype=int)
        frames = [frames[i] for i in indices]
    elif len(frames) < target_frames:
        # 重复最后一帧补齐
        while len(frames) < target_frames:
            frames.append(frames[-1])
    
    # 提取骨架
    poses, metadata = extractor.extract_sequence(frames)
    
    # 质量评估
    success_rate = 1.0 - len(metadata['failed_frames']) / target_frames
    avg_quality = np.mean(metadata['quality_scores']) if metadata['quality_scores'] else 0
    avg_visible = np.mean(metadata['visible_keypoints']) if metadata['visible_keypoints'] else 0
    
    # 判断是否成功
    success = (success_rate >= MIN_FRAME_SUCCESS_RATE and 
               avg_visible >= MIN_VISIBLE_KEYPOINTS and
               avg_quality >= QUALITY_SCORE_THRESHOLD)
    
    return poses, metadata, success


# ==================== 主处理流程 ====================
def main():
    print("\n" + "=" * 70)
    print("       智能多人选择骨架提取工具 (MediaPipe)")
    print("=" * 70)
    print(f"输入目录: {RAW_VIDEOS_DIR}")
    print(f"输出目录: {KEYPOINTS_DIR}")
    print(f"目标帧数: {SEQUENCE_LENGTH}")
    print(f"质量阈值: 可见关键点>{MIN_VISIBLE_KEYPOINTS}, 成功率>{MIN_FRAME_SUCCESS_RATE*100:.0f}%")
    print("=" * 70)
    
    # 检查输入目录
    if not os.path.exists(RAW_VIDEOS_DIR):
        print(f"\n❌ 错误: 输入目录不存在: {RAW_VIDEOS_DIR}")
        return
    
    # 初始化提取器
    print("\n初始化 MediaPipe 多人姿态提取器...")
    extractor = MultiPersonPoseExtractor()
    print("初始化完成")
    
    os.makedirs(KEYPOINTS_DIR, exist_ok=True)
    
    # 数据收集
    all_poses = []
    all_styles = []
    all_metadata = []
    
    total_videos = 0
    success_count = 0
    multi_person_videos = 0
    quality_stats = {
        'avg_quality': [],
        'avg_person_count': [],
        'avg_visible_kp': []
    }
    
    # 遍历每个风格
    for style_name, style_id in STYLE_MAP.items():
        style_path = os.path.join(RAW_VIDEOS_DIR, style_name)
        
        if not os.path.exists(style_path):
            print(f"\n⚠️ 目录不存在: {style_name}，跳过")
            continue
        
        # 获取视频文件
        video_files = []
        for ext in ['.mp4', '.avi', '.mov', '.mkv']:
            video_files.extend([f for f in os.listdir(style_path) if f.endswith(ext)])
        
        if not video_files:
            print(f"\n📁 {style_name}: 没有视频文件")
            continue
        
        print(f"\n📁 {style_name}: {len(video_files)} 个视频")
        
        style_success = 0
        style_multi = 0
        
        for idx, video_file in enumerate(video_files):
            total_videos += 1
            video_path = os.path.join(style_path, video_file)
            
            print(f"  [{idx+1}/{len(video_files)}] {video_file}...", end=" ")
            
            # 提取骨架
            poses, metadata, success = extract_skeleton_from_video(video_path, extractor)
            
            if success and poses is not None:
                all_poses.append(poses)
                all_styles.append(style_id)
                
                # 记录元数据
                avg_person = np.mean(metadata['person_counts']) if metadata['person_counts'] else 0
                avg_quality = np.mean(metadata['quality_scores']) if metadata['quality_scores'] else 0
                avg_visible = np.mean(metadata['visible_keypoints']) if metadata['visible_keypoints'] else 0
                
                all_metadata.append({
                    'video': video_file,
                    'style': style_name,
                    'avg_person_count': avg_person,
                    'avg_quality_score': avg_quality,
                    'avg_visible_keypoints': avg_visible,
                    'failed_frames': len(metadata['failed_frames']),
                    'total_frames': SEQUENCE_LENGTH
                })
                
                quality_stats['avg_quality'].append(avg_quality)
                quality_stats['avg_person_count'].append(avg_person)
                quality_stats['avg_visible_kp'].append(avg_visible)
                
                success_count += 1
                style_success += 1
                
                if avg_person > 1.1:
                    style_multi += 1
                    multi_person_videos += 1
                    print(f"✅ (多人: {avg_person:.1f}人, 质量: {avg_quality:.2f})")
                else:
                    print(f"✅ (单人, 质量: {avg_quality:.2f})")
            else:
                print(f"❌ 失败")
        
        print(f"  📊 {style_name} 完成: {style_success}/{len(video_files)} 成功, 其中多人: {style_multi}")
    
    # ==================== 保存结果 ====================
    print("\n" + "=" * 70)
    print("保存数据集")
    print("=" * 70)
    
    if all_poses:
        X = np.array(all_poses)
        y_style = np.array(all_styles)
        
        # 保存 npz 文件
        np.savez(os.path.join(KEYPOINTS_DIR, 'dataset_long.npz'),
                 X=X, y_style=y_style)
        
        # 保存标签映射
        label_map = {
            "style": {str(v): k for k, v in STYLE_MAP.items()}
        }
        with open(os.path.join(KEYPOINTS_DIR, 'label_map.json'), 'w', encoding='utf-8') as f:
            json.dump(label_map, f, ensure_ascii=False, indent=2)
        
        # 保存提取报告
        report = {
            'total_videos': total_videos,
            'success_count': success_count,
            'multi_person_videos': multi_person_videos,
            'sample_shape': list(X.shape),
            'style_distribution': {style_name: int(np.sum(y_style == style_id)) 
                                   for style_name, style_id in STYLE_MAP.items()},
            'quality_stats': {
                'avg_quality_score': float(np.mean(quality_stats['avg_quality'])),
                'avg_person_count': float(np.mean(quality_stats['avg_person_count'])),
                'avg_visible_keypoints': float(np.mean(quality_stats['avg_visible_kp']))
            },
            'metadata': all_metadata
        }
        
        with open(os.path.join(KEYPOINTS_DIR, 'extraction_report.json'), 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ 保存成功!")
        print(f"  总视频数: {total_videos}")
        print(f"  成功提取: {success_count}")
        print(f"  多人场景: {multi_person_videos} 个视频")
        print(f"  样本形状: {X.shape}")
        
        print(f"\n各风格统计:")
        for i, name in enumerate(STYLE_MAP.keys()):
            count = np.sum(y_style == i)
            bar_length = int(count / max(1, np.max(np.bincount(y_style))) * 30)
            bar = "█" * bar_length
            print(f"  {name:8}: {count:3} 个样本 {bar}")
        
        print(f"\n📊 提取质量报告:")
        print(f"  平均质量得分: {np.mean(quality_stats['avg_quality']):.2f}")
        print(f"  平均人数: {np.mean(quality_stats['avg_person_count']):.2f}")
        print(f"  平均可见关键点: {np.mean(quality_stats['avg_visible_kp']):.1f}")
        
        print(f"\n输出目录: {KEYPOINTS_DIR}")
        print(f"  - dataset_long.npz (骨架数据)")
        print(f"  - label_map.json (标签映射)")
        print(f"  - extraction_report.json (提取报告)")
        
    else:
        print("❌ 没有成功提取任何骨架数据")
    
    # 释放资源
    extractor.release()
    
    print("\n" + "=" * 70)
    print("完成！")
    print("下一步:")
    print("  1. 查看报告: cat st_data/keypoints/extraction_report.json")
    print("  2. 训练模型: python -m st_model.train")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️ 用户中断")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        input("\n按 Enter 退出...")