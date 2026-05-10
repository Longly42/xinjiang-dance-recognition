"""
配置文件 - 性能优化版
"""

import os

# ==================== 输入输出约定 ====================
# 优化：降低输入尺寸，提升帧率
INPUT_SIZE = (480, 360)          # 原 (640, 480)，计算量减少约 44%
SEQUENCE_LENGTH = 16             # 时序长度（不变）
FPS_TARGET = 20                  # 目标采集帧率（原30）

# 性能优化参数
USE_FAST_RESIZE = True           # 使用快速缩放
FRAME_SKIP = 1                   # 跳帧数（1=不跳帧，2=每2帧处理1帧）
ASYNC_INFERENCE = True           # 异步推理（不阻塞采集）
CACHE_COLOR_CONVERSION = True    # 缓存颜色转换结果

# ==================== 骨架关键点 ====================
NUM_KEYPOINTS = 33
KEYPOINT_FEATURE_DIM = 3

# ==================== 标签映射 ====================
ACTION_LABELS = {
    0: "旋转",
    1: "翻腕", 
    2: "移颈",
    3: "跺步",
    4: "蹲起",
}

STYLE_LABELS = {
    0: "刀郎",
    1: "木卡姆",
    2: "赛乃姆",
    3: "萨玛舞",
}

# ==================== 数据路径 ====================
DATA_ROOT = "./data"
RECORDING_DIR = os.path.join(DATA_ROOT, "recordings")
SLICED_DIR = os.path.join(DATA_ROOT, "sliced_clips")
KEYPOINTS_DIR = os.path.join(DATA_ROOT, "keypoints")
MODEL_DIR = os.path.join(DATA_ROOT, "models")

# 切片参数
SLICE_WINDOW_SEC = 2
SLICE_STEP_SEC = 1

# 训练参数
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

# 创建目录
for dir_path in [RECORDING_DIR, SLICED_DIR, KEYPOINTS_DIR, MODEL_DIR]:
    os.makedirs(dir_path, exist_ok=True)