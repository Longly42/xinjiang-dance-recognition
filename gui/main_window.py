"""
main_window.py - 主窗口（完整多线程版）
包含：控制栏、预览区、结果区、状态栏、多线程采集与推理
"""
import sys
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QGroupBox, QStatusBar,
    QFileDialog
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QImage, QPixmap

from gui.ui_settings import (
    MAIN_WINDOW_SIZE, PREVIEW_SIZE,
    COLORS, FONTS, BTN_TEXT, get_confidence_style
)
from utils.thread_tools import CaptureThread


class MainWindow(QMainWindow):
    """新疆舞动作识别主窗口（多线程版）"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("新疆舞多风格细粒度动作识别")
        self.resize(*MAIN_WINDOW_SIZE)
        
        # 初始化变量
        self.is_running = False
        self.selected_area = None
        self.loaded_video_path = None
        self.capture_thread = None
        
        # 设置UI布局
        self.setup_ui()
        
        # 设置状态栏
        self.setup_statusbar()
        
        # 设置窗口关闭事件处理
        self.setup_close_handler()
    
    def setup_close_handler(self):
        """设置窗口关闭时的清理"""
        def on_close(event):
            self._stop_thread()
            event.accept()
        self.closeEvent = on_close
    
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
        
        # ===== 性能信息（帧率） =====
        fps_group = QGroupBox("⚡ 性能")
        fps_group.setFont(QFont("微软雅黑", 12, QFont.Weight.Bold))
        fps_layout = QVBoxLayout(fps_group)
        
        self.fps_label = QLabel("未开始")
        self.fps_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.fps_label.setFont(QFont("微软雅黑", 10))
        self.fps_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        fps_layout.addWidget(self.fps_label)
        
        panel_layout.addWidget(fps_group)
        
        # 添加弹性空间，将内容向上对齐
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
        
        self.update_status("请在屏幕上按住鼠标左键拖拽选择需要识别的区域...")
        
        # 临时隐藏主窗口，避免遮挡选区界面
        self.hide()
        
        # 等待一小段时间让窗口隐藏完成
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(200, self._open_selector)
    
    def _open_selector(self):
        """打开选区窗口"""
        try:
            from capture.screen_selector import select_screen_area
            
            area = select_screen_area()
        except ImportError as e:
            self.show()
            self.update_status(f"选区模块加载失败: {e}")
            print(f"[ERROR] 选区模块加载失败: {e}")
            return
        
        # 重新显示主窗口
        self.show()
        self.raise_()
        
        if area:
            x, y, w, h = area
            # 确保选区尺寸不小于最小要求
            min_width, min_height = 320, 240
            if w < min_width or h < min_height:
                self.update_status(f"选区太小（{w}×{h}），至少需要 {min_width}×{min_height}")
                self.selected_area = None
            else:
                self.selected_area = area
                self.loaded_video_path = None  # 清除视频选择
                self.update_status(f"已选择区域：({x}, {y}) {w}×{h} | 可点击开始识别")
                print(f"[DEBUG] 选择区域：{self.selected_area}")
        else:
            self.update_status("未选择区域或已取消")
            self.selected_area = None
    
    def on_start(self):
        """开始识别"""
        # 检查是否已选择采集源
        if not self.selected_area and not self.loaded_video_path:
            self.update_status("请先选择屏幕区域或加载本地视频！")
            return
        
        # 停止之前的线程（如果有）
        self._stop_thread()
        
        # 确定采集源类型和参数
        if self.loaded_video_path:
            source_type = "video"
            source_param = self.loaded_video_path
            print(f"[DEBUG] 使用视频源: {source_param}")
        else:
            source_type = "screen"
            source_param = self.selected_area
            print(f"[DEBUG] 使用屏幕源: {source_param}")
        
        # 创建并启动采集线程
        try:
            self.capture_thread = CaptureThread(source_type, source_param)
        except Exception as e:
            self.update_status(f"创建采集线程失败: {e}")
            print(f"[ERROR] 创建采集线程失败: {e}")
            return
        
        # 连接信号槽
        self.capture_thread.frame_ready.connect(self.update_preview)
        self.capture_thread.result_ready.connect(self.update_result_display)
        self.capture_thread.status_updated.connect(self.update_fps_display)
        self.capture_thread.error_occurred.connect(self.on_thread_error)
        
        # 启动线程
        self.capture_thread.start()
        
        # 更新UI状态
        self.is_running = True
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_select_area.setEnabled(False)
        self.btn_load_video.setEnabled(False)
        
        # 清空之前的结果
        self.action_label.setText("识别中...")
        self.style_label.setText("识别中...")
        self.confidence_label.setText("--%")
        
        self.update_status("正在识别新疆舞动作...")
        print("[DEBUG] 开始识别")
    
    def on_stop(self):
        """停止识别"""
        # 停止线程
        self._stop_thread()
        
        # 更新UI状态
        self.is_running = False
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_select_area.setEnabled(True)
        self.btn_load_video.setEnabled(True)
        
        # 清空预览
        self.preview_label.setText("等待开始识别...")
        self.preview_label.setPixmap(QPixmap())
        
        # 清空结果
        self.action_label.setText("未识别")
        self.style_label.setText("未识别")
        self.confidence_label.setText("0%")
        self.fps_label.setText("未开始")
        
        self.update_status("已停止识别 | 可重新选择区域/加载视频")
        print("[DEBUG] 停止识别")
    
    def on_load_video(self):
        """加载本地视频文件"""
        if self.is_running:
            self.update_status("识别中，无法加载视频！")
            return
        
        # 打开文件选择对话框
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择视频文件",
            "",
            "视频文件 (*.mp4 *.avi *.mov *.mkv *.flv);;所有文件 (*.*)"
        )
        
        if file_path:
            self.loaded_video_path = file_path
            self.selected_area = None  # 清除屏幕选区
            # 显示文件名（处理长路径）
            import os
            file_name = os.path.basename(file_path)
            if len(file_name) > 50:
                file_name = file_name[:47] + "..."
            self.update_status(f"已加载视频：{file_name} | 可点击开始识别")
            print(f"[DEBUG] 加载视频：{self.loaded_video_path}")
        else:
            self.update_status("未选择视频文件")
    
    def _stop_thread(self):
        """停止并清理采集线程"""
        if self.capture_thread and self.capture_thread.isRunning():
            print("[DEBUG] 正在停止采集线程...")
            self.capture_thread.stop()
            self.capture_thread = None
            print("[DEBUG] 采集线程已停止")
    
    # ==================== 信号槽函数 ====================
    
    def update_preview(self, frame):
        """
        更新预览画面
        frame: RGB 格式的 numpy 数组 (H, W, C)
        """
        try:
            # 获取图像尺寸
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            
            # 创建 QImage
            qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            
            # 缩放以适应预览区域并保持宽高比
            scaled_pixmap = pixmap.scaled(
                PREVIEW_SIZE[0],
                PREVIEW_SIZE[1],
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.preview_label.setPixmap(scaled_pixmap)
            
        except Exception as e:
            # 避免频繁打印错误
            if not hasattr(self, '_last_preview_error'):
                self._last_preview_error = 0
            import time
            if time.time() - self._last_preview_error > 1.0:
                print(f"[ERROR] 更新预览失败: {e}")
                self._last_preview_error = time.time()
    
    def update_result_display(self, action: str, style: str, confidence: float):
        """
        更新界面显示的结果
        action: 动作标签
        style: 风格标签
        confidence: 置信度 (0-1)
        """
        # 更新动作标签
        self.action_label.setText(action)
        
        # 更新风格标签
        self.style_label.setText(style)
        
        # 更新置信度显示
        confidence_percent = int(confidence * 100)
        self.confidence_label.setText(f"{confidence_percent}%")
        self.confidence_label.setStyleSheet(get_confidence_style(confidence))
        
        # 更新状态栏（但不频繁刷新，避免干扰）
        self.update_status(f"识别中 | 动作：{action} | 风格：{style} | 置信度：{confidence_percent}%")
        
        # 调试输出
        print(f"[RESULT] 动作: {action}, 风格: {style}, 置信度: {confidence:.2%}")
    
    def update_fps_display(self, message: str):
        """
        更新帧率显示
        message: 状态消息（如 "采集帧率: 29.5 fps"）
        """
        self.fps_label.setText(message)
        print(f"[INFO] {message}")
    
    def on_thread_error(self, error: str):
        """
        处理线程错误
        error: 错误信息
        """
        self.update_status(f"错误：{error}")
        print(f"[ERROR] 线程错误: {error}")
        
        # 自动停止识别
        self.on_stop()
    
    def update_status(self, message: str):
        """
        更新状态栏消息
        message: 状态文本
        """
        self.status_bar.showMessage(message)
    
    # ==================== 可选：键盘快捷键支持 ====================
    
    def keyPressEvent(self, event):
        """键盘快捷键处理"""
        if event.key() == Qt.Key.Key_Space:
            # 空格键开始/停止
            if self.is_running:
                self.on_stop()
            else:
                if self.selected_area or self.loaded_video_path:
                    self.on_start()
        elif event.key() == Qt.Key.Key_Escape:
            # ESC 键停止
            if self.is_running:
                self.on_stop()
        else:
            super().keyPressEvent(event)