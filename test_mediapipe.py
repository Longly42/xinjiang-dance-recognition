"""调试版骨架提取 - 定位崩溃位置"""

import os
import sys
import traceback

# 设置环境变量
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"

print("=" * 60)
print("调试版骨架提取工具")
print("=" * 60)

# 1. 测试导入
print("\n[1/6] 测试导入...")
try:
    import cv2
    print("  ✅ opencv-python 导入成功")
except Exception as e:
    print(f"  ❌ 失败: {e}")
    sys.exit(1)

try:
    import numpy as np
    print("  ✅ numpy 导入成功")
except Exception as e:
    print(f"  ❌ 失败: {e}")
    sys.exit(1)

try:
    import mediapipe as mp
    print(f"  ✅ mediapipe 导入成功, 版本: {mp.__version__}")
except Exception as e:
    print(f"  ❌ 失败: {e}")
    sys.exit(1)

try:
    from tqdm import tqdm
    print("  ✅ tqdm 导入成功")
except Exception as e:
    print(f"  ❌ 失败: {e}")
    sys.exit(1)

# 2. 测试 MediaPipe 初始化
print("\n[2/6] 测试 MediaPipe 初始化...")
try:
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    print("  ✅ Pose 初始化成功")
    pose.close()
except Exception as e:
    print(f"  ❌ 失败: {e}")
    traceback.print_exc()
    sys.exit(1)

# 3. 检查数据目录
print("\n[3/6] 检查数据目录...")
RAW_VIDEOS_DIR = "data/raw_videos"

if not os.path.exists(RAW_VIDEOS_DIR):
    print(f"  ❌ 目录不存在: {RAW_VIDEOS_DIR}")
    sys.exit(1)

# 列出所有风格
styles = [d for d in os.listdir(RAW_VIDEOS_DIR) if os.path.isdir(os.path.join(RAW_VIDEOS_DIR, d))]
print(f"  ✅ 找到 {len(styles)} 个风格目录: {styles}")

# 4. 测试单个视频
print("\n[4/6] 测试单个视频处理...")

# 找到第一个视频文件
test_video = None
for style in styles:
    style_path = os.path.join(RAW_VIDEOS_DIR, style)
    for action in os.listdir(style_path):
        action_path = os.path.join(style_path, action)
        if os.path.isdir(action_path):
            for f in os.listdir(action_path):
                if f.endswith(('.mp4', '.avi', '.mov')):
                    test_video = os.path.join(action_path, f)
                    break
            if test_video:
                break
    if test_video:
        break

if not test_video:
    print("  ❌ 没有找到任何视频文件")
    sys.exit(1)

print(f"  测试视频: {test_video}")

# 配置参数
INPUT_SIZE = (480, 360)
SEQUENCE_LENGTH = 16
MIN_FRAMES = 10

try:
    print("  读取视频...")
    cap = cv2.VideoCapture(test_video)
    frames = []
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, INPUT_SIZE)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        frame_count += 1
        if frame_count % 50 == 0:
            print(f"    已读取 {frame_count} 帧")
    cap.release()
    print(f"  ✅ 读取完成: {len(frames)} 帧")

    # 采样
    if len(frames) > SEQUENCE_LENGTH:
        indices = np.linspace(0, len(frames) - 1, SEQUENCE_LENGTH, dtype=int)
        frames = [frames[i] for i in indices]
        print(f"  采样到 {len(frames)} 帧")

    # 提取骨架
    print("  提取骨架（可能会慢）...")
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    poses = []
    for i, frame in enumerate(frames):
        result = pose.process(frame)
        if result.pose_landmarks:
            frame_pose = []
            for lm in result.pose_landmarks.landmark:
                frame_pose.append([lm.x, lm.y, lm.visibility])
            poses.append(np.array(frame_pose, dtype=np.float32))
            if (i + 1) % 5 == 0:
                print(f"    已处理 {i+1}/{len(frames)} 帧")
        else:
            print(f"    ❌ 第{i}帧未检测到人体")
            break
    
    pose.close()
    
    if len(poses) == SEQUENCE_LENGTH:
        print(f"  ✅ 成功提取 {len(poses)} 帧骨架")
    else:
        print(f"  ❌ 只提取到 {len(poses)} 帧")

except Exception as e:
    print(f"  ❌ 处理失败: {e}")
    traceback.print_exc()
    sys.exit(1)

# 5. 测试批量处理
print("\n[5/6] 测试批量处理配置...")

STYLE_MAP = {
    "赛乃姆": 0,
    "刀郎舞": 1,
    "盘子舞": 2,
    "萨玛舞": 3,
    "手鼓舞": 4,
    "纳孜尔库姆": 5,
}

ACTION_MAP = {
    "扭颈动头": 0, "翻腕绕腕": 1, "摆肩抖肩": 2,
    "垫步进退": 3, "叉腰摆胯": 4, "击掌拍手": 5,
    "旋转": 6, "行礼": 7, "抬臂托帽": 8,
}

print(f"  ✅ 风格映射: {len(STYLE_MAP)} 种")
print(f"  ✅ 动作映射: {len(ACTION_MAP)} 种")

# 6. 检查输出目录
print("\n[6/6] 检查输出目录...")
KEYPOINTS_DIR = "data/keypoints"
os.makedirs(KEYPOINTS_DIR, exist_ok=True)
print(f"  ✅ 输出目录: {KEYPOINTS_DIR}")

print("\n" + "=" * 60)
print("✅ 所有检查通过！")
print("=" * 60)
print("\n现在可以运行正式提取了。")
print("如果正式提取仍然退出，请运行：")
print("  python -c \"import mediapipe; print(mediapipe.__file__)\"")