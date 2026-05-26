"""
st_model 配置文件 - 10秒长片段风格识别
"""

import os

# ==================== 数据路径（新目录）====================
ST_DATA_ROOT = "./st_data"
RAW_VIDEOS_DIR = os.path.join(ST_DATA_ROOT, "raw_videos")
KEYPOINTS_DIR = os.path.join(ST_DATA_ROOT, "keypoints")
MODEL_SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "best_style_model.pth")

# 标签映射文件（从提取时生成）
LABEL_MAP_PATH = os.path.join(KEYPOINTS_DIR, "label_map.json")

# 原 model 的标签映射（用于获取名称，可选）
ORIGINAL_LABEL_MAP = "./data/keypoints/label_map.json"

# ==================== 数据参数 ====================
SEQUENCE_LENGTH = 64          # 每个样本采样帧数
NUM_KEYPOINTS = 33
KEYPOINT_FEATURE_DIM = 3

# ==================== 模型结构 ====================
IN_CHANNELS = KEYPOINT_FEATURE_DIM
HIDDEN_DIM = 64
NUM_STGCN_LAYERS = 4
DROPOUT = 0.3
NUM_STYLES = 6

# ==================== 训练参数 ====================
BATCH_SIZE = 16
EPOCHS = 80
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 1e-4
LR_SCHEDULER_STEP = 30
LR_SCHEDULER_GAMMA = 0.1
PATIENCE = 15

# 数据增强
USE_AUGMENTATION = True
AUG_NOISE_STD = 0.01
AUG_DROP_FRAME_RATIO = 0.1
AUG_TIME_MASK = True