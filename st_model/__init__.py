"""
st_model - 新疆舞风格识别模型
"""

from .predict import get_recognizer
from .pose_extract import get_pose_extractor
from .stgcn_style import StyleSTGCN

__all__ = ['get_recognizer', 'get_pose_extractor', 'StyleSTGCN']