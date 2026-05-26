"""
performance_dashboard.py - 性能仪表盘组件（FPS折线图 + 推理延迟条形图）
"""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QPainter, QPen, QColor, QBrush, QFont, QLinearGradient

import numpy as np


class FPSLineChart(QWidget):
    """FPS实时折线图"""
    
    def __init__(self, parent=None, history_len=60):
        super().__init__(parent)
        self.history_len = history_len
        self.fps_history = [0] * history_len
        self.max_fps = 30  # 目标帧率
        self.setMinimumHeight(100)
        self.setMinimumWidth(200)
        self.setStyleSheet("background-color: #2d2d2d; border-radius: 4px;")
    
    def update_fps(self, fps):
        """更新FPS数据"""
        self.fps_history.pop(0)
        self.fps_history.append(min(fps, self.max_fps))
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        w, h = self.width(), self.height()
        margin = 20
        chart_w = w - 2 * margin
        chart_h = h - 2 * margin
        
        # 绘制背景网格
        painter.setPen(QPen(QColor(60, 60, 60), 1))
        for i in range(5):
            y = margin + i * chart_h / 4
            painter.drawLine(margin, y, w - margin, y)
        
        # 绘制FPS折线
        if len(self.fps_history) > 1:
            points = []
            for i, fps in enumerate(self.fps_history):
                x = margin + i * chart_w / (self.history_len - 1)
                y = margin + chart_h - (fps / self.max_fps) * chart_h
                points.append((x, y))
            
            # 绘制折线
            for i in range(len(points) - 1):
                painter.setPen(QPen(QColor(0, 120, 212), 2))
                painter.drawLine(int(points[i][0]), int(points[i][1]), 
                               int(points[i+1][0]), int(points[i+1][1]))
        
        # 绘制参考线标签
        painter.setPen(QPen(QColor(150, 150, 150), 1))
        painter.setFont(QFont("微软雅黑", 8))
        for i in range(5):
            fps_val = self.max_fps - i * self.max_fps / 4
            y = margin + i * chart_h / 4
            painter.drawText(2, int(y + 3), f"{int(fps_val)}")
        
        # 绘制当前FPS数值
        current_fps = self.fps_history[-1] if self.fps_history else 0
        painter.setFont(QFont("微软雅黑", 12, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(0, 200, 100), 1))
        painter.drawText(w - 60, margin + 15, f"{current_fps:.1f} fps")
        
        painter.end()


class DelayBarChart(QWidget):
    """推理延迟条形图"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.delay_ms = 0
        self.max_delay = 200  # 最大延迟200ms
        self.setMinimumHeight(60)
        self.setMinimumWidth(200)
    
    def update_delay(self, delay_ms):
        self.delay_ms = min(delay_ms, self.max_delay)
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        w, h = self.width(), self.height()
        margin = 40
        bar_width = w - margin - 10
        bar_height = h - 20
        
        # 背景条
        painter.setBrush(QBrush(QColor(60, 60, 60)))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(margin, 10, bar_width, bar_height, 4, 4)
        
        # 进度条（根据延迟变色）
        ratio = self.delay_ms / self.max_delay
        fill_width = bar_width * ratio
        
        if ratio < 0.3:
            color = QColor(0, 200, 100)  # 绿色
        elif ratio < 0.7:
            color = QColor(255, 200, 0)  # 黄色
        else:
            color = QColor(255, 50, 50)  # 红色
        
        painter.setBrush(QBrush(color))
        painter.drawRoundedRect(margin, 10, fill_width, bar_height, 4, 4)
        
        # 文字
        painter.setPen(QPen(QColor(220, 220, 220), 1))
        painter.setFont(QFont("微软雅黑", 10))
        painter.drawText(5, 25, "延迟")
        painter.drawText(margin + 5, 28, f"{self.delay_ms:.0f} ms")
        
        painter.end()


class PerformanceDashboard(QWidget):
    """性能仪表盘组合组件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setSpacing(5)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 标题
        title = QLabel("⚡ 实时性能")
        title.setFont(QFont("微软雅黑", 10, QFont.Weight.Bold))
        title.setStyleSheet("color: #e0e0e0; padding: 5px;")
        layout.addWidget(title)
        
        # FPS折线图
        self.fps_chart = FPSLineChart()
        layout.addWidget(self.fps_chart)
        
        # 延迟条形图
        self.delay_bar = DelayBarChart()
        layout.addWidget(self.delay_bar)
        
        # 骨架提取频率指示
        self.skel_status = QLabel("🦴 骨架检测: 等待中")
        self.skel_status.setFont(QFont("微软雅黑", 9))
        self.skel_status.setStyleSheet("color: #a0a0a0; padding: 2px;")
        layout.addWidget(self.skel_status)
    
    def update_fps(self, fps):
        self.fps_chart.update_fps(fps)
    
    def update_delay(self, delay_ms):
        self.delay_bar.update_delay(delay_ms)
    
    def update_skeleton_status(self, has_skeleton, keypoints_count):
        if has_skeleton:
            self.skel_status.setText(f"🦴 骨架检测: ✅ {keypoints_count} 关键点")
            self.skel_status.setStyleSheet("color: #6b9c3e; padding: 2px;")
        else:
            self.skel_status.setText("🦴 骨架检测: ⚠️ 未检测到人体")
            self.skel_status.setStyleSheet("color: #d4872e; padding: 2px;")