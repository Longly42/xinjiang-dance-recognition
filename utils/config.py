"""
配置文件 - 统一使用 st_data 目录，时序长度 64 帧
"""

import os

# ==================== 输入输出约定 ====================
# 统一为64帧（与st_model保持一致）
INPUT_SIZE = (480, 360)          # 输入尺寸
SEQUENCE_LENGTH = 64             # 时序长度（64帧）
FPS_TARGET = 15                  # 目标采集帧率（64帧约4.3秒缓存）

# 性能优化参数
USE_FAST_RESIZE = True           # 使用快速缩放
FRAME_SKIP = 1                   # 跳帧数（1=不跳帧，2=每2帧处理1帧）
ASYNC_INFERENCE = True           # 异步推理（不阻塞采集）
CACHE_COLOR_CONVERSION = True    # 缓存颜色转换结果

# ==================== 骨架关键点 ====================
NUM_KEYPOINTS = 33
KEYPOINT_FEATURE_DIM = 3

# ==================== 数据路径（统一使用 st_data）====================
DATA_ROOT = "./st_data"                    # 数据根目录改为 st_data
RECORDING_DIR = os.path.join(DATA_ROOT, "recordings")
SLICED_DIR = os.path.join(DATA_ROOT, "sliced_clips")
KEYPOINTS_DIR = os.path.join(DATA_ROOT, "keypoints")
MODEL_DIR = os.path.join(DATA_ROOT, "models")

# 标签映射文件路径
LABEL_MAP_PATH = os.path.join(KEYPOINTS_DIR, "label_map.json")

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