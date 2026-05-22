"""最简单的骨架提取版本"""

import os
import cv2
import numpy as np
import json

os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

print("导入 mediapipe...")
import mediapipe as mp
print("导入完成")

# 配置
RAW_VIDEOS_DIR = "data/raw_videos"
KEYPOINTS_DIR = "data/keypoints"
INPUT_SIZE = (480, 360)
SEQUENCE_LENGTH = 16

STYLE_MAP = {
    "赛乃姆": 0, "刀郎舞": 1, "盘子舞": 2,
    "萨玛舞": 3, "手鼓舞": 4, "纳孜尔库姆": 5,
}

ACTION_MAP = {
    "扭颈动头": 0, "翻腕绕腕": 1, "摆肩抖肩": 2,
    "垫步进退": 3, "叉腰摆胯": 4, "击掌拍手": 5,
    "旋转": 6, "行礼": 7, "抬臂托帽": 8,
}

print("初始化 MediaPipe...")
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=0,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)
print("初始化完成")

os.makedirs(KEYPOINTS_DIR, exist_ok=True)

all_poses = []
all_action_labels = []
all_style_labels = []

total = 0
success = 0

# 遍历
for style_name in STYLE_MAP.keys():
    style_path = os.path.join(RAW_VIDEOS_DIR, style_name)
    if not os.path.exists(style_path):
        print(f"跳过: {style_path} 不存在")
        continue
    
    style_id = STYLE_MAP[style_name]
    print(f"\n处理: {style_name}")
    
    for action_name in ACTION_MAP.keys():
        action_path = os.path.join(style_path, action_name)
        if not os.path.exists(action_path):
            continue
        
        action_id = ACTION_MAP[action_name]
        videos = [f for f in os.listdir(action_path) if f.endswith('.mp4')]
        
        for video_file in videos:
            total += 1
            video_path = os.path.join(action_path, video_file)
            
            # 读取视频
            cap = cv2.VideoCapture(video_path)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, INPUT_SIZE)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            cap.release()
            
            if len(frames) < 10:
                print(f"  {video_file}: 帧数不足")
                continue
            
            # 采样
            if len(frames) > SEQUENCE_LENGTH:
                indices = np.linspace(0, len(frames)-1, SEQUENCE_LENGTH, dtype=int)
                frames = [frames[i] for i in indices]
            
            # 提取骨架
            poses = []
            for frame in frames:
                result = pose.process(frame)
                if result.pose_landmarks:
                    frame_pose = [[lm.x, lm.y, lm.visibility] for lm in result.pose_landmarks.landmark]
                    poses.append(np.array(frame_pose, dtype=np.float32))
                else:
                    if poses:
                        poses.append(poses[-1].copy())
                    else:
                        poses.append(np.zeros((33, 3), dtype=np.float32))
            
            if len(poses) == SEQUENCE_LENGTH:
                all_poses.append(np.array(poses))
                all_action_labels.append(action_id)
                all_style_labels.append(style_id)
                success += 1
                print(f"  ✅ {video_file}")
            else:
                print(f"  ❌ {video_file}: 失败")

print(f"\n成功: {success}/{total}")

if all_poses:
    X = np.array(all_poses)
    np.savez(os.path.join(KEYPOINTS_DIR, 'dataset.npz'),
             X=X, y_action=np.array(all_action_labels), y_style=np.array(all_style_labels))
    
    label_map = {
        "style": {str(v): k for k, v in STYLE_MAP.items()},
        "action": {str(v): k for k, v in ACTION_MAP.items()}
    }
    with open(os.path.join(KEYPOINTS_DIR, 'label_map.json'), 'w', encoding='utf-8') as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    
    print(f"数据集保存: {X.shape}")

pose.close()
print("完成！")
input("按 Enter 退出...")