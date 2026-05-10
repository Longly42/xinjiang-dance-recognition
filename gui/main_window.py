"""
main_window.py - 主窗口（完整版包含录屏模式）
"""

import sys
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QGroupBox, QStatusBar,
    QFileDialog, QComboBox, QProgressBar, QMessageBox
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QImage, QPixmap

from gui.ui_settings import (
    MAIN_WINDOW_SIZE, PREVIEW_SIZE,
    COLORS, FONTS, BTN_TEXT, get_confidence_style
)
from utils.thread_tools import CaptureThread, RecordingThread, DatasetBuildThread


class MainWindow(QMainWindow):
    """新疆舞动作识别主窗口（多线程版 + 录屏模式）"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("新疆舞多风格细粒度动作识别")
        self.resize(*MAIN_WINDOW_SIZE)
        
        # 初始化变量
        self.is_running = False
        self.selected_area = None
        self.loaded_video_path = None
        self.capture_thread = None
        self.recording_thread = None
        self.dataset_thread = None
        
        # 设置UI布局
        self.setup_ui()
        
        # 设置状态栏
        self.setup_statusbar()
        
        # 设置窗口关闭事件处理
        self.setup_close_handler()
    
    def setup_close_handler(self):
        """设置窗口关闭时的清理"""
        def on_close(event):
            self._stop_all_threads()
            event.accept()
        self.closeEvent = on_close
    
    # ==================== UI 初始化 ====================
    
    def setup_ui(self):
        """创建所有UI组件并布局"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(15)
        
        # ===== 左侧区域 =====
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(10)
        
        # 控制栏
        control_bar = self.create_control_bar()
        left_layout.addWidget(control_bar)
        
        # 预览区域
        self.preview_label = self.create_preview_label()
        left_layout.addWidget(self.preview_label, alignment=Qt.AlignmentFlag.AlignCenter)
        
        # 进度条（录屏时显示）
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        left_layout.addWidget(self.progress_bar)
        
        main_layout.addWidget(left_widget, stretch=3)
        
        # ===== 右侧区域 =====
        right_widget = self.create_result_panel()
        main_layout.addWidget(right_widget, stretch=1)
    
    def create_control_bar(self):
        """创建顶部控制栏"""
        bar_widget = QWidget()
        bar_layout = QHBoxLayout(bar_widget)
        bar_layout.setContentsMargins(0, 0, 0, 0)
        bar_layout.setSpacing(10)
        
        # 模式选择
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["🎯 识别模式", "📹 录屏模式"])
        self.mode_combo.setFont(QFont(*FONTS["button"]))
        self.mode_combo.setFixedWidth(120)
        bar_layout.addWidget(self.mode_combo)
        
        # 按钮1：选择屏幕区域
        self.btn_select_area = QPushButton(BTN_TEXT["select_area"])
        self.btn_select_area.clicked.connect(self.on_select_area)
        self.btn_select_area.setFont(QFont(*FONTS["button"]))
        bar_layout.addWidget(self.btn_select_area)
        
        # 按钮2：开始
        self.btn_start = QPushButton(BTN_TEXT["start"])
        self.btn_start.setProperty("primary", True)
        self.btn_start.clicked.connect(self.on_start)
        self.btn_start.setFont(QFont(*FONTS["button"]))
        bar_layout.addWidget(self.btn_start)
        
        # 按钮3：停止
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
        label = QLabel()
        label.setObjectName("preview_label")
        label.setFixedSize(*PREVIEW_SIZE)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setText("等待开始...")
        label.setStyleSheet(f"""
            color: {COLORS['text_secondary']};
            font-family: "微软雅黑";
            font-size: 12pt;
        """)
        return label
    
    def create_result_panel(self):
        panel = QWidget()
        panel_layout = QVBoxLayout(panel)
        panel_layout.setSpacing(15)
        
        # 动作识别结果
        action_group = QGroupBox("🏃 动作识别")
        action_group.setFont(QFont("微软雅黑", 12, QFont.Weight.Bold))
        action_layout = QVBoxLayout(action_group)
        self.action_label = QLabel("未识别")
        self.action_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.action_label.setFont(QFont(*FONTS["result_value"]))
        self.action_label.setStyleSheet(f"color: {COLORS['accent']};")
        action_layout.addWidget(self.action_label)
        panel_layout.addWidget(action_group)
        
        # 风格识别结果
        style_group = QGroupBox("🎭 风格识别")
        style_group.setFont(QFont("微软雅黑", 12, QFont.Weight.Bold))
        style_layout = QVBoxLayout(style_group)
        self.style_label = QLabel("未识别")
        self.style_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.style_label.setFont(QFont(*FONTS["result_value"]))
        self.style_label.setStyleSheet(f"color: {COLORS['accent']};")
        style_layout.addWidget(self.style_label)
        panel_layout.addWidget(style_group)
        
        # 置信度
        confidence_group = QGroupBox("📊 置信度")
        confidence_layout = QVBoxLayout(confidence_group)
        self.confidence_label = QLabel("0%")
        self.confidence_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.confidence_label.setFont(QFont(*FONTS["confidence"]))
        self.confidence_label.setStyleSheet(get_confidence_style(0.0))
        confidence_layout.addWidget(self.confidence_label)
        panel_layout.addWidget(confidence_group)
        
        # 性能信息
        fps_group = QGroupBox("⚡ 性能")
        fps_layout = QVBoxLayout(fps_group)
        self.fps_label = QLabel("未开始")
        self.fps_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.fps_label.setFont(QFont("微软雅黑", 10))
        self.fps_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        fps_layout.addWidget(self.fps_label)
        panel_layout.addWidget(fps_group)
        
        panel_layout.addStretch()
        return panel
    
    def setup_statusbar(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.setFont(QFont(*FONTS["status"]))
        self.update_status("就绪 | 请选择模式并框选区域")
    
    # ==================== 按钮槽函数 ====================
    
    def on_select_area(self):
        if self.is_running:
            self.update_status("任务进行中，无法选择区域！")
            return
        
        self.update_status("请在屏幕上拖拽选择区域...")
        self.hide()
        QTimer.singleShot(200, self._open_selector)
    
    def _open_selector(self):
        try:
            from capture.screen_selector import select_screen_area
            area = select_screen_area()
        except ImportError as e:
            self.show()
            self.update_status(f"选区模块加载失败: {e}")
            return
        
        self.show()
        self.raise_()
        
        if area:
            x, y, w, h = area
            min_width, min_height = 320, 240
            if w < min_width or h < min_height:
                self.update_status(f"选区太小（{w}×{h}），至少需要 {min_width}×{min_height}")
                self.selected_area = None
            else:
                self.selected_area = area
                self.loaded_video_path = None
                mode = self.mode_combo.currentText()
                self.update_status(f"已选择区域：{w}×{h} | 当前模式: {mode}")
        else:
            self.update_status("未选择区域或已取消")
            self.selected_area = None
    
    def on_start(self):
        mode = self.mode_combo.currentText()
        
        if mode == "📹 录屏模式":
            self.start_recording()
        else:  # 识别模式
            self.start_recognition()
    
    def start_recognition(self):
        if not self.selected_area and not self.loaded_video_path:
            self.update_status("请先选择屏幕区域或加载本地视频！")
            return
        
        self._stop_all_threads()
        
        if self.loaded_video_path:
            source_type = "video"
            source_param = self.loaded_video_path
        else:
            source_type = "screen"
            source_param = self.selected_area
        
        try:
            self.capture_thread = CaptureThread(source_type, source_param)
        except Exception as e:
            self.update_status(f"创建采集线程失败: {e}")
            return
        
        self.capture_thread.frame_ready.connect(self.update_preview)
        self.capture_thread.result_ready.connect(self.update_result_display)
        self.capture_thread.status_updated.connect(self.update_fps_display)
        self.capture_thread.error_occurred.connect(self.on_thread_error)
        self.capture_thread.start()
        
        self._set_running_state(True)
        self.update_status("正在识别新疆舞动作...")
    
    def start_recording(self):
        if not self.selected_area:
            self.update_status("请先选择屏幕区域！")
            return
        
        if self.recording_thread and self.recording_thread.isRunning():
            return
        
        self.recording_thread = RecordingThread(self.selected_area, max_duration_sec=60)
        self.recording_thread.status_updated.connect(self.update_status)
        self.recording_thread.finished.connect(self.on_recording_finished)
        self.recording_thread.error_occurred.connect(self.on_thread_error)
        self.recording_thread.progress_updated.connect(self.on_recording_progress)
        self.recording_thread.frame_ready.connect(self.update_preview)  # 添加这一行：连接预览信号
        
        self.recording_thread.start()
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self._set_running_state(True, is_recording=True)
        self.update_status("正在录屏... 点击停止结束录制")
    
    def on_recording_finished(self, video_path):
        self.update_status(f"录制完成: {video_path}")
        
        # 询问是否自动构建数据集
        reply = QMessageBox.question(
            self, "数据集构建",
            f"录制完成！是否自动切分视频并提取骨架特征？\n\n视频路径: {video_path}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.build_dataset_from_video(video_path)
        
        self.progress_bar.setVisible(False)
        self._set_running_state(False)
    
    def on_recording_progress(self, percent):
        self.progress_bar.setValue(percent)
    
    def build_dataset_from_video(self, video_path):
        """从录制的视频构建数据集"""
        self.dataset_thread = DatasetBuildThread(video_path)
        self.dataset_thread.status_updated.connect(self.update_status)
        self.dataset_thread.finished.connect(self.on_dataset_built)
        self.dataset_thread.error_occurred.connect(self.on_thread_error)
        self.dataset_thread.progress_updated.connect(self.on_dataset_progress)
        self.dataset_thread.start()
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.update_status("正在构建数据集...")
    
    def on_dataset_built(self, message):
        self.update_status(message)
        self.progress_bar.setVisible(False)
        QMessageBox.information(self, "数据集构建完成", message)
    
    def on_dataset_progress(self, percent):
        self.progress_bar.setValue(percent)
    
    def on_load_video(self):
        if self.is_running:
            self.update_status("任务进行中，无法加载视频！")
            return
        
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "",
            "视频文件 (*.mp4 *.avi *.mov *.mkv *.flv);;所有文件 (*.*)"
        )
        
        if file_path:
            self.loaded_video_path = file_path
            self.selected_area = None
            import os
            file_name = os.path.basename(file_path)
            if len(file_name) > 50:
                file_name = file_name[:47] + "..."
            self.update_status(f"已加载视频：{file_name}")
    
    def on_stop(self):
        self._stop_all_threads()
        self._set_running_state(False)
        self.progress_bar.setVisible(False)
        self.preview_label.setText("等待开始...")
        self.preview_label.setPixmap(QPixmap())
        self.update_status("已停止")
    
    def _stop_all_threads(self):
        if self.capture_thread and self.capture_thread.isRunning():
            self.capture_thread.stop()
            self.capture_thread = None
        
        if self.recording_thread and self.recording_thread.isRunning():
            self.recording_thread.stop()
            self.recording_thread = None
        
        if self.dataset_thread and self.dataset_thread.isRunning():
            self.dataset_thread.quit()
            self.dataset_thread.wait()
            self.dataset_thread = None
    
    def _set_running_state(self, running, is_recording=False):
        self.is_running = running
        self.btn_start.setEnabled(not running)
        self.btn_stop.setEnabled(running)
        self.btn_select_area.setEnabled(not running)
        self.btn_load_video.setEnabled(not running)
        self.mode_combo.setEnabled(not running)
        
        if not running:
            self.action_label.setText("未识别")
            self.style_label.setText("未识别")
            self.confidence_label.setText("0%")
            self.fps_label.setText("未开始")
    
    # ==================== 信号槽函数 ====================
    
    def update_preview(self, frame):
        try:
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            scaled_pixmap = pixmap.scaled(
                PREVIEW_SIZE[0], PREVIEW_SIZE[1],
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.preview_label.setPixmap(scaled_pixmap)
        except Exception as e:
            pass
    
    def update_result_display(self, action: str, style: str, confidence: float):
        self.action_label.setText(action)
        self.style_label.setText(style)
        confidence_percent = int(confidence * 100)
        self.confidence_label.setText(f"{confidence_percent}%")
        self.confidence_label.setStyleSheet(get_confidence_style(confidence))
        self.update_status(f"动作：{action} | 风格：{style} | 置信度：{confidence_percent}%")
    
    def update_fps_display(self, message: str):
        self.fps_label.setText(message)
    
    def on_thread_error(self, error: str):
        self.update_status(f"错误：{error}")
        self.on_stop()
    
    def update_status(self, message: str):
        self.status_bar.showMessage(message)
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space:
            if self.is_running:
                self.on_stop()
            else:
                if self.selected_area or self.loaded_video_path:
                    self.on_start()
        elif event.key() == Qt.Key.Key_Escape:
            if self.is_running:
                self.on_stop()
        else:
            super().keyPressEvent(event)