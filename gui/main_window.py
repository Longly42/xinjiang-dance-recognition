"""
main_window.py - 主窗口
包含：控制栏、预览区、结果区、状态栏
"""
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QGroupBox, QStatusBar
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from gui.ui_settings import (
    MAIN_WINDOW_SIZE, PREVIEW_SIZE,
    COLORS, FONTS, BTN_TEXT, get_confidence_style
)


class MainWindow(QMainWindow):
    """新疆舞动作识别主窗口"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("新疆舞多风格细粒度动作识别")
        self.resize(*MAIN_WINDOW_SIZE)
        
        # 初始化变量
        self.is_running = False      # 识别是否进行中
        self.selected_area = None    # 存储选中的屏幕区域
        self.loaded_video_path = None  # 存储加载的视频路径
        
        # 设置UI布局
        self.setup_ui()
        
        # 设置状态栏
        self.setup_statusbar()
    
    # ==================== UI 初始化 ====================
    
    def setup_ui(self):
        """创建所有UI组件并布局"""
        
        # 创建中央容器
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局（水平布局：左侧预览区 + 右侧结果区）
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(15)
        
        # ===== 左侧区域（预览区 + 控制栏） =====
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(10)
        
        # 1. 控制栏（按钮行）
        control_bar = self.create_control_bar()
        left_layout.addWidget(control_bar)
        
        # 2. 预览区域
        self.preview_label = self.create_preview_label()
        left_layout.addWidget(self.preview_label, alignment=Qt.AlignmentFlag.AlignCenter)
        
        # 将左侧区域添加到主布局
        main_layout.addWidget(left_widget, stretch=3)
        
        # ===== 右侧区域（结果显示） =====
        right_widget = self.create_result_panel()
        main_layout.addWidget(right_widget, stretch=1)
    
    def create_control_bar(self):
        """创建顶部控制栏（4个按钮）"""
        
        # 水平布局容器
        bar_widget = QWidget()
        bar_layout = QHBoxLayout(bar_widget)
        bar_layout.setContentsMargins(0, 0, 0, 0)
        bar_layout.setSpacing(10)
        
        # 按钮1：选择屏幕区域
        self.btn_select_area = QPushButton(BTN_TEXT["select_area"])
        self.btn_select_area.clicked.connect(self.on_select_area)
        self.btn_select_area.setFont(QFont(*FONTS["button"]))
        bar_layout.addWidget(self.btn_select_area)
        
        # 按钮2：开始识别（主要按钮，蓝色强调）
        self.btn_start = QPushButton(BTN_TEXT["start"])
        self.btn_start.setProperty("primary", True)
        self.btn_start.clicked.connect(self.on_start)
        self.btn_start.setFont(QFont(*FONTS["button"]))
        bar_layout.addWidget(self.btn_start)
        
        # 按钮3：停止识别
        self.btn_stop = QPushButton(BTN_TEXT["stop"])
        self.btn_stop.clicked.connect(self.on_stop)
        self.btn_stop.setEnabled(False)
        self.btn_stop.setFont(QFont(*FONTS["button"]))
        bar_layout.addWidget(self.btn_stop)
        
        # 按钮4：加载本地视频
        self.btn_load_video = QPushButton(BTN_TEXT["load_video"])
        self.btn_load_video.clicked.connect(self.on_load_video)
        self.btn_load_video.setFont(QFont(*FONTS["button"]))
        bar_layout.addWidget(self.btn_load_video)
        
        return bar_widget
    
    def create_preview_label(self):
        """创建预览画面标签"""
        label = QLabel()
        label.setObjectName("preview_label")
        label.setFixedSize(*PREVIEW_SIZE)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setText("等待开始识别...")
        label.setStyleSheet(f"""
            color: {COLORS['text_secondary']};
            font-family: "微软雅黑";
            font-size: 12pt;
        """)
        return label
    
    def create_result_panel(self):
        """创建右侧结果显示面板"""
        panel = QWidget()
        panel_layout = QVBoxLayout(panel)
        panel_layout.setSpacing(15)
        
        # ===== 动作识别结果 =====
        action_group = QGroupBox("🏃 动作识别")
        action_group.setFont(QFont("微软雅黑", 12, QFont.Weight.Bold))
        action_layout = QVBoxLayout(action_group)
        
        self.action_label = QLabel("未识别")
        self.action_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # 安全创建字体对象，避免解包错误
        self.action_label.setFont(QFont(*FONTS["result_value"]))
        self.action_label.setStyleSheet(f"color: {COLORS['accent']};")
        action_layout.addWidget(self.action_label)
        
        panel_layout.addWidget(action_group)
        
        # ===== 风格识别结果 =====
        style_group = QGroupBox("🎭 风格识别")
        style_group.setFont(QFont("微软雅黑", 12, QFont.Weight.Bold))
        style_layout = QVBoxLayout(style_group)
        
        self.style_label = QLabel("未识别")
        self.style_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.style_label.setFont(QFont(*FONTS["result_value"]))
        self.style_label.setStyleSheet(f"color: {COLORS['accent']};")
        style_layout.addWidget(self.style_label)
        
        panel_layout.addWidget(style_group)
        
        # ===== 置信度 =====
        confidence_group = QGroupBox("📊 置信度")
        confidence_group.setFont(QFont("微软雅黑", 12, QFont.Weight.Bold))
        confidence_layout = QVBoxLayout(confidence_group)
        
        self.confidence_label = QLabel("0%")
        self.confidence_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.confidence_label.setFont(QFont(*FONTS["confidence"]))
        self.confidence_label.setStyleSheet(get_confidence_style(0.0))
        confidence_layout.addWidget(self.confidence_label)
        
        panel_layout.addWidget(confidence_group)

        panel_layout.addStretch()
        
        return panel
    
    def setup_statusbar(self):
        """创建状态栏"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.setFont(QFont(*FONTS["status"]))
        self.update_status("就绪 | 请先选择屏幕区域或加载本地视频")
    
    # ==================== 按钮槽函数 ====================
    
    def on_select_area(self):
        """选择屏幕区域"""
        if self.is_running:
            self.update_status("识别中，无法选择区域！")
            return
            
        self.update_status("请按住鼠标左键拖拽选择需要识别的屏幕区域...")
        # 模拟选区结果
        self.selected_area = (100, 100, 740, 580)
        self.update_status(f"已选择区域：{self.selected_area} | 可点击开始识别")
        print(f"[DEBUG] 选择区域：{self.selected_area}")
    
    def on_start(self):
        """开始识别"""
        if not self.selected_area and not self.loaded_video_path:
            self.update_status("请先选择屏幕区域或加载本地视频！")
            return
            
        self.is_running = True
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_select_area.setEnabled(False)
        self.btn_load_video.setEnabled(False)
        
        self.update_status("正在识别新疆舞动作...")
        print("[DEBUG] 开始识别按钮被点击")
        
        # 模拟识别结果
        self.update_result_display("旋转", "刀郎", 0.87)
    
    def on_stop(self):
        """停止识别"""
        self.is_running = False
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_select_area.setEnabled(True)
        self.btn_load_video.setEnabled(True)
        
        self.update_status("已停止识别 | 可重新选择区域/加载视频")
        print("[DEBUG] 停止识别按钮被点击")
    
    def on_load_video(self):
        """加载本地视频文件"""
        if self.is_running:
            self.update_status("识别中，无法加载视频！")
            return
            
        self.update_status("正在打开文件选择对话框...")
        # 模拟加载结果
        self.loaded_video_path = "D:/test_video.mp4"
        self.selected_area = None
        self.update_status(f"已加载视频：{self.loaded_video_path} | 可点击开始识别")
        print(f"[DEBUG] 加载视频：{self.loaded_video_path}")
    
    # ==================== 结果更新 ====================
    
    def update_result_display(self, action: str, style: str, confidence: float):
        """更新界面显示的结果"""
        self.action_label.setText(action)
        self.style_label.setText(style)
        
        confidence_percent = int(confidence * 100)
        self.confidence_label.setText(f"{confidence_percent}%")
        self.confidence_label.setStyleSheet(get_confidence_style(confidence))
        
        self.update_status(f"识别中 | 动作：{action} | 风格：{style} | 置信度：{confidence_percent}%")
    
    def update_status(self, message: str):
        """更新状态栏消息"""
        self.status_bar.showMessage(message)