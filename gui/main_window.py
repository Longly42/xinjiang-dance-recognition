"""
main_window.py - 主窗口（美化版 + 性能仪表盘 + 识别历史报告）
"""

import sys
import cv2
import os
import numpy as np
from datetime import datetime
from collections import deque
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QGroupBox, QStatusBar,
    QFileDialog, QComboBox, QProgressBar, QMessageBox, QCheckBox,
    QFrame, QSplitter, QScrollArea, QDialog, QGridLayout, QTableWidget,
    QTableWidgetItem, QHeaderView, QTabWidget, QTextEdit
)
from PyQt6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, pyqtSignal
from PyQt6.QtGui import QFont, QImage, QPixmap, QColor, QPalette, QBrush, QLinearGradient


from PyQt6.QtGui import (
    QFont, QImage, QPixmap, QColor, QPalette, QBrush, QLinearGradient,
    QPainter,  
    QPen       
)
from gui.ui_settings import (
    MAIN_WINDOW_SIZE, PREVIEW_SIZE,
    COLORS, FONTS, BTN_TEXT, get_confidence_style
)
from gui.performance_dashboard import PerformanceDashboard
from utils.thread_tools import CaptureThread, RecordingThread, DatasetBuildThread


class CircularProgressBar(QWidget):
    """环形置信度进度条"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.confidence = 0
        self.setFixedSize(100, 100)
    
    def set_confidence(self, confidence):
        self.confidence = confidence
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        w, h = self.width(), self.height()
        pen_width = 8
        radius = min(w, h) // 2 - pen_width
        
        center = w // 2, h // 2
        
        # 背景圆环
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor(60, 60, 60), pen_width))
        painter.drawEllipse(center[0] - radius, center[1] - radius, 2 * radius, 2 * radius)
        
        # 置信度圆环
        angle = int(360 * self.confidence)
        
        if self.confidence < 0.6:
            color = QColor(211, 47, 47)  # 红色
        elif self.confidence < 0.8:
            color = QColor(212, 135, 46)  # 橙色
        else:
            color = QColor(107, 156, 62)  # 绿色
        
        painter.setPen(QPen(color, pen_width))
        painter.drawArc(center[0] - radius, center[1] - radius, 2 * radius, 2 * radius, 90 * 16, -angle * 16)
        
        # 中心文字
        painter.setFont(QFont("微软雅黑", 16, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(224, 224, 224), 1))
        painter.drawText(center[0] - 25, center[1] + 8, f"{int(self.confidence * 100)}%")
        
        painter.end()


class StyleBadge(QLabel):
    """风格徽章组件"""
    
    style_colors = {
        "赛乃姆": "#e74c3c",
        "刀郎舞": "#e67e22",
        "盘子舞": "#2ecc71",
        "萨玛舞": "#3498db",
        "手鼓舞": "#9b59b6",
        "纳孜尔库姆": "#1abc9c",
    }
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFont(QFont("微软雅黑", 18, QFont.Weight.Bold))
        self.setMinimumHeight(60)
        self.setStyleSheet("""
            border-radius: 10px;
            padding: 10px;
        """)
    
    def set_style(self, style_name, confidence):
        color = self.style_colors.get(style_name, "#0078d4")
        alpha = int(80 + confidence * 100)
        self.setStyleSheet(f"""
            background-color: rgba{tuple(int(color[i:i+2], 16) for i in (1,3,5)) + (alpha,)};
            border-radius: 10px;
            padding: 10px;
        """)
        self.setText(f"{style_name}\n{int(confidence*100)}%")
        
        if confidence > 0.7:
            self.setFont(QFont("微软雅黑", 18, QFont.Weight.Bold))
        else:
            self.setFont(QFont("微软雅黑", 14))


class RecognitionReportDialog(QDialog):
    """识别历史报告弹窗"""
    
    def __init__(self, history, parent=None):
        super().__init__(parent)
        self.setWindowTitle("识别历史报告")
        self.setMinimumSize(600, 500)
        self.setStyleSheet("background-color: #2d2d2d; color: #e0e0e0;")
        
        layout = QVBoxLayout(self)
        
        # 标题
        title = QLabel("📊 本次识别统计报告")
        title.setFont(QFont("微软雅黑", 16, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("padding: 10px;")
        layout.addWidget(title)
        
        # 创建选项卡
        tabs = QTabWidget()
        tabs.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #3c3c3c; border-radius: 4px; }
            QTabBar::tab { padding: 8px 16px; background-color: #3c3c3c; color: #e0e0e0; }
            QTabBar::tab:selected { background-color: #0078d4; }
        """)
        
        # 统计信息
        stats_widget = self.create_stats_tab(history)
        tabs.addTab(stats_widget, "📈 统计概览")
        
        # 时间线
        timeline_widget = self.create_timeline_tab(history)
        tabs.addTab(timeline_widget, "⏱️ 识别时间线")
        
        # 详细记录
        details_widget = self.create_details_tab(history)
        tabs.addTab(details_widget, "📋 详细记录")
        
        layout.addWidget(tabs)
        
        # 关闭按钮
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.accept)
        close_btn.setFixedWidth(100)
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #0078d4;
                border-radius: 4px;
                padding: 8px 16px;
                font-size: 12px;
            }
            QPushButton:hover { background-color: #1a88d9; }
        """)
        
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.addWidget(close_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
    
    def create_stats_tab(self, history):
        widget = QWidget()
        layout = QGridLayout(widget)
        
        if not history:
            label = QLabel("暂无识别记录")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(label)
            return widget
        
        # 统计各风格出现次数
        style_counts = {}
        action_counts = {}
        confidences = []
        
        for record in history:
            style = record.get('style', '未知')
            action = record.get('action', '未知')
            conf = record.get('confidence', 0)
            
            style_counts[style] = style_counts.get(style, 0) + 1
            action_counts[action] = action_counts.get(action, 0) + 1
            confidences.append(conf)
        
        # 显示统计
        row = 0
        label = QLabel("🎭 风格分布")
        label.setFont(QFont("微软雅黑", 12, QFont.Weight.Bold))
        layout.addWidget(label, row, 0, 1, 2)
        row += 1
        
        for style, count in sorted(style_counts.items(), key=lambda x: -x[1]):
            layout.addWidget(QLabel(f"  {style}:"), row, 0)
            layout.addWidget(QLabel(f"{count} 次 ({count/len(history)*100:.1f}%)"), row, 1)
            row += 1
        
        row += 1
        label = QLabel("🏃 动作分布")
        label.setFont(QFont("微软雅黑", 12, QFont.Weight.Bold))
        layout.addWidget(label, row, 0, 1, 2)
        row += 1
        
        for action, count in sorted(action_counts.items(), key=lambda x: -x[1]):
            layout.addWidget(QLabel(f"  {action}:"), row, 0)
            layout.addWidget(QLabel(f"{count} 次 ({count/len(history)*100:.1f}%)"), row, 1)
            row += 1
        
        # 平均置信度
        avg_conf = np.mean(confidences) if confidences else 0
        row += 1
        layout.addWidget(QLabel("📊 平均置信度:"), row, 0)
        layout.addWidget(QLabel(f"{avg_conf*100:.1f}%"), row, 1)
        
        row += 1
        layout.addWidget(QLabel("📝 总识别次数:"), row, 0)
        layout.addWidget(QLabel(f"{len(history)} 次"), row, 1)
        
        layout.setColumnStretch(1, 1)
        layout.setRowStretch(row + 1, 1)
        
        return widget
    
    def create_timeline_tab(self, history):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("border: none;")
        
        timeline_widget = QWidget()
        timeline_layout = QVBoxLayout(timeline_widget)
        
        for i, record in enumerate(history):
            time_str = record.get('time', '')
            style = record.get('style', '')
            action = record.get('action', '')
            conf = record.get('confidence', 0)
            
            frame = QFrame()
            frame.setStyleSheet("""
                QFrame {
                    background-color: #3c3c3c;
                    border-radius: 6px;
                    margin: 2px;
                }
            """)
            frame_layout = QHBoxLayout(frame)
            
            # 序号和时间
            idx_label = QLabel(f"#{i+1}")
            idx_label.setFixedWidth(40)
            idx_label.setStyleSheet("color: #0078d4; font-weight: bold;")
            frame_layout.addWidget(idx_label)
            
            time_label = QLabel(time_str)
            time_label.setFixedWidth(100)
            time_label.setStyleSheet("color: #a0a0a0;")
            frame_layout.addWidget(time_label)
            
            # 风格徽章
            style_label = QLabel(style)
            style_label.setFixedWidth(100)
            style_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            style_label.setStyleSheet(f"""
                background-color: rgba(0,120,212,0.3);
                border-radius: 4px;
                padding: 2px 8px;
            """)
            frame_layout.addWidget(style_label)
            
            # 动作
            action_label = QLabel(action)
            action_label.setFixedWidth(100)
            frame_layout.addWidget(action_label)
            
            # 置信度条
            bar = QProgressBar()
            bar.setFixedWidth(150)
            bar.setRange(0, 100)
            bar.setValue(int(conf * 100))
            bar.setStyleSheet("""
                QProgressBar::chunk {
                    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 #d32f2f, stop:0.6 #d4872e, stop:1 #6b9c3e);
                }
            """)
            frame_layout.addWidget(bar)
            
            timeline_layout.addWidget(frame)
        
        timeline_layout.addStretch()
        scroll.setWidget(timeline_widget)
        layout.addWidget(scroll)
        
        return widget
    
    def create_details_tab(self, history):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        table = QTableWidget()
        table.setColumnCount(5)
        table.setHorizontalHeaderLabels(["时间", "风格", "动作", "置信度", "耗时(ms)"])
        table.setRowCount(len(history))
        table.setStyleSheet("""
            QTableWidget {
                gridline-color: #3c3c3c;
                selection-background-color: #0078d4;
            }
            QHeaderView::section {
                background-color: #3c3c3c;
                padding: 4px;
                border: none;
            }
        """)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        
        for i, record in enumerate(history):
            table.setItem(i, 0, QTableWidgetItem(record.get('time', '')))
            table.setItem(i, 1, QTableWidgetItem(record.get('style', '')))
            table.setItem(i, 2, QTableWidgetItem(record.get('action', '')))
            table.setItem(i, 3, QTableWidgetItem(f"{record.get('confidence', 0)*100:.1f}%"))
            table.setItem(i, 4, QTableWidgetItem(str(record.get('inference_time', 0))))
        
        layout.addWidget(table)
        return widget


class MainWindow(QMainWindow):
    """新疆舞风格识别主窗口（美化版）"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("新疆舞风格识别系统 - 专业版")
        self.resize(*MAIN_WINDOW_SIZE)
        
        # 初始化变量
        self.is_running = False
        self.selected_area = None
        self.loaded_video_path = None
        self.capture_thread = None
        self.recording_thread = None
        self.dataset_thread = None
        self.show_skeleton = True
        
        # 识别历史记录
        self.recognition_history = deque(maxlen=200)
        
        # 性能数据
        self.fps_history = deque(maxlen=60)
        self.delay_history = deque(maxlen=60)
        
        # 设置UI
        self.setup_ui()
        self.setup_statusbar()
        self.setup_close_handler()
        
        # 应用样式
        self.apply_styles()
    
    def apply_styles(self):
        """应用全局样式"""
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {COLORS['bg_main']};
            }}
            QGroupBox {{
                font-weight: bold;
                border: 1px solid {COLORS['border']};
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                font-size: 12pt;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 8px 0 8px;
            }}
            QPushButton {{
                background-color: {COLORS['bg_button']};
                color: {COLORS['text_primary']};
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 10pt;
            }}
            QPushButton:hover {{
                background-color: {COLORS['bg_button_hover']};
            }}
            QPushButton:pressed {{
                background-color: {COLORS['bg_button_pressed']};
            }}
            QPushButton[primary="true"] {{
                background-color: {COLORS['accent']};
            }}
            QPushButton[primary="true"]:hover {{
                background-color: {COLORS['accent_hover']};
            }}
            QComboBox {{
                background-color: {COLORS['bg_button']};
                border-radius: 6px;
                padding: 6px 12px;
            }}
            QCheckBox {{
                spacing: 8px;
            }}
        """)
    
    def setup_close_handler(self):
        def on_close(event):
            self.show_report_dialog() if self.recognition_history else None
            self._stop_all_threads()
            event.accept()
        self.closeEvent = on_close
    
    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        
        # 主分割器
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # ===== 左侧区域 =====
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(10)
        left_layout.setContentsMargins(10, 10, 10, 10)
        
        # 控制栏
        control_bar = self.create_control_bar()
        left_layout.addWidget(control_bar)
        
        # 预览区
        self.preview_label = QLabel()
        self.preview_label.setFixedSize(*PREVIEW_SIZE)
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setStyleSheet(f"""
            background-color: {COLORS['bg_preview']};
            border-radius: 12px;
            border: 1px solid {COLORS['border']};
        """)
        self.preview_label.setText("等待开始...")
        left_layout.addWidget(self.preview_label, alignment=Qt.AlignmentFlag.AlignCenter)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border-radius: 4px;
                text-align: center;
                background-color: #3c3c3c;
            }
            QProgressBar::chunk {
                background-color: #0078d4;
                border-radius: 4px;
            }
        """)
        left_layout.addWidget(self.progress_bar)
        
        # ===== 右侧面板 =====
        right_widget = self.create_result_panel()
        
        main_splitter.addWidget(left_widget)
        main_splitter.addWidget(right_widget)
        main_splitter.setSizes([800, 400])
        
        layout = QHBoxLayout(central)
        layout.addWidget(main_splitter)
    
    def create_control_bar(self):
        bar = QWidget()
        layout = QHBoxLayout(bar)
        layout.setSpacing(12)
        
        # 模式选择
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["🎯 识别模式", "📹 录屏模式"])
        self.mode_combo.setFont(QFont(*FONTS["button"]))
        self.mode_combo.setFixedWidth(130)
        layout.addWidget(self.mode_combo)
        
        # 骨架开关
        self.skeleton_checkbox = QCheckBox("🦴 显示骨架")
        self.skeleton_checkbox.setChecked(True)
        self.skeleton_checkbox.stateChanged.connect(self.on_skeleton_toggle)
        layout.addWidget(self.skeleton_checkbox)
        
        # 按钮
        self.btn_select = QPushButton(BTN_TEXT["select_area"])
        self.btn_select.clicked.connect(self.on_select_area)
        layout.addWidget(self.btn_select)
        
        self.btn_start = QPushButton(BTN_TEXT["start"])
        self.btn_start.setProperty("primary", True)
        self.btn_start.clicked.connect(self.on_start)
        layout.addWidget(self.btn_start)
        
        self.btn_stop = QPushButton(BTN_TEXT["stop"])
        self.btn_stop.clicked.connect(self.on_stop)
        self.btn_stop.setEnabled(False)
        layout.addWidget(self.btn_stop)
        
        self.btn_load = QPushButton(BTN_TEXT["load_video"])
        self.btn_load.clicked.connect(self.on_load_video)
        layout.addWidget(self.btn_load)
        
        # 报告按钮
        self.btn_report = QPushButton("📊 识别报告")
        self.btn_report.clicked.connect(self.show_report_dialog)
        self.btn_report.setEnabled(True)
        layout.addWidget(self.btn_report)
        
        layout.addStretch()
        return bar
    
    def create_result_panel(self):
        panel = QWidget()
        panel.setMaximumWidth(380)
        layout = QVBoxLayout(panel)
        layout.setSpacing(12)
        
        # ===== 风格识别区 =====
        style_group = QGroupBox("🎭 风格识别")
        style_layout = QVBoxLayout(style_group)
        
        self.style_badge = StyleBadge()
        style_layout.addWidget(self.style_badge)
        
        # ===== 动作识别区 =====
        action_group = QGroupBox("🏃 动作识别")
        action_layout = QVBoxLayout(action_group)
        
        self.action_label = QLabel("等待识别...")
        self.action_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.action_label.setFont(QFont("微软雅黑", 20, QFont.Weight.Bold))
        self.action_label.setStyleSheet(f"color: {COLORS['text_primary']}; padding: 10px;")
        action_layout.addWidget(self.action_label)
        
        # ===== 置信度环形图 =====
        confidence_group = QGroupBox("📊 置信度")
        conf_layout = QHBoxLayout(confidence_group)
        conf_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.circular_progress = CircularProgressBar()
        conf_layout.addWidget(self.circular_progress)
        
        # ===== 性能仪表盘 =====
        self.performance_dashboard = PerformanceDashboard()
        
        # ===== 骨架状态 =====
        skeleton_group = QGroupBox("🦴 骨架检测")
        skeleton_layout = QVBoxLayout(skeleton_group)
        self.skeleton_status = QLabel("等待检测...")
        self.skeleton_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.skeleton_status.setFont(QFont("微软雅黑", 10))
        skeleton_layout.addWidget(self.skeleton_status)
        
        # 添加到主布局
        layout.addWidget(style_group)
        layout.addWidget(action_group)
        layout.addWidget(confidence_group)
        layout.addWidget(self.performance_dashboard)
        layout.addWidget(skeleton_group)
        layout.addStretch()
        
        return panel
    
    def setup_statusbar(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.setFont(QFont(*FONTS["status"]))
        self.status_bar.setStyleSheet(f"""
            QStatusBar {{
                background-color: {COLORS['bg_panel']};
                color: {COLORS['text_secondary']};
                border-top: 1px solid {COLORS['separator']};
            }}
        """)
        self.update_status("就绪 | 请选择屏幕区域或加载视频")
    
    # ==================== 槽函数 ====================
    
    def on_select_area(self):
        if self.is_running:
            self.update_status("任务进行中，无法选择区域！")
            return
        
        self.update_status("请在屏幕上拖拽选择区域...")
        self.hide()
        
        from PyQt6.QtCore import QTimer
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
            if w < 320 or h < 240:
                self.update_status(f"选区太小（{w}×{h}），至少需要 320×240")
                self.selected_area = None
            else:
                self.selected_area = area
                self.loaded_video_path = None
                self.update_status(f"已选择区域：{w}×{h}")
        else:
            self.update_status("未选择区域或已取消")
    
    def on_start(self):
        mode = self.mode_combo.currentText()
        if "录屏" in mode:
            self.start_recording()
        else:
            self.start_recognition()
    
    def start_recognition(self):
        if not self.selected_area and not self.loaded_video_path:
            self.update_status("请先选择屏幕区域或加载视频！")
            return
        
        self._stop_all_threads()
        
        source_type = "video" if self.loaded_video_path else "screen"
        source_param = self.loaded_video_path if self.loaded_video_path else self.selected_area
        
        try:
            self.capture_thread = CaptureThread(source_type, source_param, self.show_skeleton)
        except Exception as e:
            self.update_status(f"创建线程失败: {e}")
            return
        
        # 连接信号
        self.capture_thread.frame_ready.connect(self.update_preview)
        self.capture_thread.result_ready.connect(self.update_result_display)
        self.capture_thread.status_updated.connect(self.update_performance)
        self.capture_thread.error_occurred.connect(self.on_thread_error)
        self.capture_thread.skeleton_ready.connect(self.update_skeleton_status)
        self.capture_thread.start()
        
        self._set_running_state(True)
        self.update_status("正在识别新疆舞风格...")
    
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
        self.recording_thread.frame_ready.connect(self.update_preview)
        self.recording_thread.start()
        
        self.progress_bar.setVisible(True)
        self._set_running_state(True, is_recording=True)
        self.update_status("正在录屏...")
    
    def on_recording_finished(self, video_path):
        self.update_status(f"录制完成: {video_path}")
        
        reply = QMessageBox.question(
            self, "数据集构建",
            f"是否自动构建数据集？\n{video_path}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.build_dataset(video_path)
        
        self.progress_bar.setVisible(False)
        self._set_running_state(False)
    
    def on_recording_progress(self, percent):
        self.progress_bar.setValue(percent)
    
    def build_dataset(self, video_path):
        self.dataset_thread = DatasetBuildThread(video_path)
        self.dataset_thread.status_updated.connect(self.update_status)
        self.dataset_thread.finished.connect(lambda msg: QMessageBox.information(self, "完成", msg))
        self.dataset_thread.error_occurred.connect(self.on_thread_error)
        self.dataset_thread.progress_updated.connect(self.progress_bar.setValue)
        self.dataset_thread.start()
        
        self.progress_bar.setVisible(True)
        self.update_status("正在构建数据集...")
    
    def on_load_video(self):
        if self.is_running:
            self.update_status("任务进行中，无法加载视频！")
            return
        
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频", "", "视频文件 (*.mp4 *.avi *.mov *.mkv)"
        )
        
        if file_path:
            self.loaded_video_path = file_path
            self.selected_area = None
            self.update_status(f"已加载: {os.path.basename(file_path)}")
    
    def on_stop(self):
        self._stop_all_threads()
        self._set_running_state(False)
        self.progress_bar.setVisible(False)
        self.preview_label.setText("等待开始...")
        self.preview_label.setPixmap(QPixmap())
        self.update_status("已停止")
    
    def on_skeleton_toggle(self, state):
        self.show_skeleton = (state == Qt.CheckState.Checked.value)
        if self.capture_thread and self.capture_thread.isRunning():
            self.capture_thread.set_show_skeleton(self.show_skeleton)
        self.update_status(f"骨架可视化: {'开启' if self.show_skeleton else '关闭'}")
    
    def show_report_dialog(self):
        if not self.recognition_history:
            QMessageBox.information(self, "提示", "暂无识别记录")
            return
        
        dialog = RecognitionReportDialog(list(self.recognition_history), self)
        dialog.exec()
    
    def _stop_all_threads(self):
        if self.capture_thread:
            self.capture_thread.stop()
            self.capture_thread = None
        if self.recording_thread:
            self.recording_thread.stop()
            self.recording_thread = None
        if self.dataset_thread:
            self.dataset_thread.quit()
            self.dataset_thread.wait()
            self.dataset_thread = None
    
    def _set_running_state(self, running, is_recording=False):
        self.is_running = running
        self.btn_start.setEnabled(not running)
        self.btn_stop.setEnabled(running)
        self.btn_select.setEnabled(not running)
        self.btn_load.setEnabled(not running)
        self.mode_combo.setEnabled(not running)
        
        if not running:
            self.action_label.setText("等待识别...")
            self.style_badge.set_style("未识别", 0)
            self.circular_progress.set_confidence(0)
    
    # ==================== 信号处理 ====================
    
    def update_preview(self, frame):
        try:
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qt_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_img)
            scaled = pixmap.scaled(
                PREVIEW_SIZE[0], PREVIEW_SIZE[1],
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.preview_label.setPixmap(scaled)
        except Exception:
            pass
    
    def update_result_display(self, action, style, confidence):
        # 更新UI
        self.action_label.setText(action if action else "待扩展")
        self.style_badge.set_style(style, confidence)
        self.circular_progress.set_confidence(confidence)
        
        # 记录历史
        self.recognition_history.append({
            'time': datetime.now().strftime("%H:%M:%S"),
            'style': style,
            'action': action if action else "无动作",
            'confidence': confidence,
            'inference_time': 0  # 实际可从线程获取
        })
        
        self.update_status(f"风格: {style} | 动作: {action} | 置信度: {confidence*100:.1f}%")
    
    def update_performance(self, message):
        """更新性能数据"""
        # 解析FPS和延迟
        if "帧率:" in message and "延迟:" in message:
            try:
                fps_part = message.split("帧率:")[1].split("fps")[0].strip()
                delay_part = message.split("延迟:")[1].split("ms")[0].strip()
                fps = float(fps_part)
                delay = float(delay_part)
                
                self.performance_dashboard.update_fps(fps)
                self.performance_dashboard.update_delay(delay)
                
                self.fps_history.append(fps)
                self.delay_history.append(delay)
            except:
                pass
        
        self.status_bar.showMessage(message)
    
    def update_skeleton_status(self, has_skeleton, keypoints_count):
        self.performance_dashboard.update_skeleton_status(has_skeleton, keypoints_count)
        
        if has_skeleton:
            self.skeleton_status.setText(f"✅ 检测到人体 ({keypoints_count} 关键点)")
            self.skeleton_status.setStyleSheet(f"color: {COLORS['success']};")
        else:
            self.skeleton_status.setText("⚠️ 未检测到人体")
            self.skeleton_status.setStyleSheet(f"color: {COLORS['warning']};")
    
    def on_thread_error(self, error):
        self.update_status(f"错误: {error}")
        self.on_stop()
    
    def update_status(self, message):
        self.status_bar.showMessage(message)
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space:
            if self.is_running:
                self.on_stop()
            elif self.selected_area or self.loaded_video_path:
                self.on_start()
        elif event.key() == Qt.Key.Key_S:
            self.skeleton_checkbox.setChecked(not self.skeleton_checkbox.isChecked())
        elif event.key() == Qt.Key.Key_Escape:
            if self.is_running:
                self.on_stop()
        elif event.key() == Qt.Key.Key_R:
            self.show_report_dialog()
        else:
            super().keyPressEvent(event)