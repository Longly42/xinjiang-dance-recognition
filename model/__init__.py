"""
model 模块入口
"""
from .pose_extract import PoseExtractor, get_pose_extractor
from .stgcn import STGCN, DualTaskSTGCN
from .predict import Recognizer, get_recognizer
from .config import (
    NUM_ACTIONS, NUM_STYLES, SEQUENCE_LENGTH,
    NUM_KEYPOINTS, KEYPOINT_FEATURE_DIM
)

__all__ = [
    'PoseExtractor', 'get_pose_extractor',
    'STGCN', 'DualTaskSTGCN',
    'Recognizer', 'get_recognizer',
    'NUM_ACTIONS', 'NUM_STYLES', 'SEQUENCE_LENGTH'
]