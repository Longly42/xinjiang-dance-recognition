"""
模型专用配置 – 适配 ST-GCN
"""

import os

# ==================== 数据参数 ====================
SEQUENCE_LENGTH = 16          # 输入时序长度 (帧)
NUM_KEYPOINTS = 33            # MediaPipe 33 个关键点
KEYPOINT_FEATURE_DIM = 3      # 每个关键点特征维度: (x, y, visibility) 或 (x, y, z)
# 使用 x, y, visibility 即可（z 坐标对平面动作不太可靠）

# ==================== 模型结构 ====================
# ST-GCN 参数
IN_CHANNELS = KEYPOINT_FEATURE_DIM   # 输入特征维度
HIDDEN_DIM = 64                      # 第一层输出通道数
NUM_STGCN_LAYERS = 4                 # ST-GCN 块数量
DROPOUT = 0.3

# 分类输出
NUM_ACTIONS = 9      # 动作类别数（根据任务书）
NUM_STYLES = 6       # 风格类别数（根据任务书）

# ST-GCN 图拓扑（基于 MediaPipe 33 点的自然连接）
# 我们将在 stgcn.py 中定义人体骨架图的邻接矩阵

# ==================== 训练参数 ====================
BATCH_SIZE = 32
EPOCHS = 120
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
LR_SCHEDULER_STEP = 30      # 每 30 个 epoch 学习率乘以 0.1
LR_SCHEDULER_GAMMA = 0.1

# 早停
PATIENCE = 20               # 验证 loss 不下降多少轮后停止

# 类别不平衡处理
USE_CLASS_WEIGHTS = True

# 数据增强
USE_AUGMENTATION = True
AUG_NOISE_STD = 0.01        # 添加高斯噪声
AUG_DROP_FRAME_RATIO = 0.1  # 随机丢弃部分帧（用插值补齐）
AUG_TIME_MASK = True        # 时间掩码

# ==================== 路径 ====================
DATA_ROOT = "./data"
KEYPOINTS_DIR = os.path.join(DATA_ROOT, "keypoints")
MODEL_SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "best_model.pth")

# 标签映射文件
LABEL_MAP_PATH = os.path.join(KEYPOINTS_DIR, "label_map.json")

# ==================== 推理参数 ====================
INFERENCE_CONFIDENCE_THRESHOLD = 0.5   # 低于此阈值的动作/风格不更新