"""
ui_settings.py - GUI样式与常量配置
深色专业风，参考 OBS Studio / Premiere Pro
"""
# 必须先导入 QFont，否则 NameError
from PyQt6.QtGui import QFont

# ==================== 窗口尺寸 ====================
MAIN_WINDOW_SIZE = (1200, 800)      # 主窗口宽度×高度
PREVIEW_SIZE = (640, 480)           # 预览画面尺寸
RESULT_PANEL_WIDTH = 280            # 右侧结果面板宽度

# ==================== 颜色主题 ====================
COLORS = {
    # 背景色系
    "bg_main": "#1e1e1e",           # 主窗口背景（深黑灰）
    "bg_panel": "#2d2d2d",          # 面板背景（深灰）
    "bg_preview": "#000000",        # 预览区背景（纯黑）
    "bg_button": "#3c3c3c",         # 按钮背景
    "bg_button_hover": "#505050",   # 按钮悬停
    "bg_button_pressed": "#2a2a2a", # 按钮按下
    
    # 文字色系
    "text_primary": "#e0e0e0",      # 主要文字（浅灰白）
    "text_secondary": "#a0a0a0",    # 次要文字（中灰）
    "text_disabled": "#606060",     # 禁用文字（深灰）
    
    # 功能色系
    "accent": "#0078d4",            # 强调色/主题蓝
    "accent_hover": "#1a88d9",      # 强调色悬停
    "success": "#6b9c3e",           # 成功/高置信度（冷绿）
    "warning": "#d4872e",           # 警告/中置信度（橙）
    "error": "#d32f2f",             # 错误/低置信度（红）
    
    # 边框与分隔
    "border": "#3c3c3c",            # 边框颜色
    "separator": "#2a2a2a",         # 分隔线颜色
}

# ==================== 字体 ====================
# 标准化字体配置：(字体名, 字号pt, 样式)
FONTS = {
    "title": ("微软雅黑", 18, QFont.Weight.Bold),           # 标题（软件名）
    "result_label": ("微软雅黑", 12, QFont.Weight.Normal),  # 结果标签（"动作："）
    "result_value": ("微软雅黑", 24, QFont.Weight.Bold),    # 结果数值（"旋转"）
    "confidence": ("微软雅黑", 32, QFont.Weight.Bold),      # 置信度大数字
    "button": ("微软雅黑", 10, QFont.Weight.Normal),        # 按钮文字
    "status": ("微软雅黑", 9, QFont.Weight.Normal),         # 状态栏
}

# ==================== 按钮文字 ====================
BTN_TEXT = {
    "select_area": "🎯 选择区域",
    "start": "▶ 开始识别",
    "stop": "⏹ 停止",
    "load_video": "📁 加载视频",
}

# ==================== PyQt6 全局样式表 ====================
GLOBAL_STYLESHEET = f"""
/* 主窗口 */
QMainWindow {{
    background-color: {COLORS["bg_main"]};
}}

/* 中央面板 */
QWidget {{
    background-color: {COLORS["bg_panel"]};
    color: {COLORS["text_primary"]};
    font-family: "微软雅黑";
    font-size: 12pt;
}}

/* 按钮通用样式 */
QPushButton {{
    background-color: {COLORS["bg_button"]};
    color: {COLORS["text_primary"]};
    border: none;
    border-radius: 4px;
    padding: 8px 16px;
    font-size: 10pt;
    font-weight: normal;
}}
QPushButton:hover {{
    background-color: {COLORS["bg_button_hover"]};
}}
QPushButton:pressed {{
    background-color: {COLORS["bg_button_pressed"]};
}}
QPushButton:disabled {{
    color: {COLORS["text_disabled"]};
    background-color: {COLORS["bg_main"]};
}}

/* 主要按钮（强调色） */
QPushButton[primary="true"] {{
    background-color: {COLORS["accent"]};
}}
QPushButton[primary="true"]:hover {{
    background-color: {COLORS["accent_hover"]};
}}

/* 预览区域 QLabel */
QLabel#preview_label {{
    background-color: {COLORS["bg_preview"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 4px;
}}

/* 结果面板分组框 */
QGroupBox {{
    font-weight: bold;
    border: 1px solid {COLORS["border"]};
    border-radius: 6px;
    margin-top: 10px;
    padding-top: 10px;
    font-size: 12pt;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px 0 5px;
    font-size: 12pt;
}}

/* 状态栏 */
QStatusBar {{
    background-color: {COLORS["bg_panel"]};
    color: {COLORS["text_secondary"]};
    border-top: 1px solid {COLORS["separator"]};
    font-size: 9pt;
}}

/* 滚动条 */
QScrollBar:vertical {{
    background: {COLORS["bg_main"]};
    width: 8px;
    border-radius: 4px;
}}
QScrollBar::handle:vertical {{
    background: {COLORS["border"]};
    border-radius: 4px;
    min-height: 20px;
}}
QScrollBar::handle:vertical:hover {{
    background: {COLORS["accent"]};
}}
"""

# ==================== 置信度颜色获取函数 ====================
def get_confidence_color(confidence: float) -> str:
    """根据置信度返回对应颜色"""
    if confidence >= 0.8:
        return COLORS["success"]
    elif confidence >= 0.6:
        return COLORS["warning"]
    else:
        return COLORS["error"]

def get_confidence_style(confidence: float) -> str:
    """生成置信度数值的样式表（带颜色）"""
    color = get_confidence_color(confidence)
    return f"""
        color: {color}; 
        font-family: "微软雅黑";
        font-size: 32pt; 
        font-weight: bold;
    """