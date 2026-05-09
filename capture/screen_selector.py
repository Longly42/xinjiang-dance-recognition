"""
screen_selector.py - 屏幕区域选择器（使用 QDialog 模态，可靠拖拽）
"""

import sys
from PyQt6.QtWidgets import (
    QDialog, QRubberBand, QApplication, QLabel
)
from PyQt6.QtCore import Qt, QRect, QPoint, QTimer
from PyQt6.QtGui import QPainter, QColor, QBrush


class ScreenSelector(QDialog):
    """全屏区域选择器（模态对话框）"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        # 无边框、置顶
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        # 获取所有屏幕的总体几何区域
        self.screen_geometry = self.get_full_screen_geometry()
        self.setGeometry(self.screen_geometry)
        
        # 橡胶框（选区拖拽框）
        self.rubber_band = QRubberBand(QRubberBand.Shape.Rectangle, self)
        self.origin = QPoint()
        self.is_dragging = False      # 是否真正拖拽（区别于单击）
        self.selected_rect = None     # 最终选择的矩形
        
        # 顶部提示标签
        self.setup_hint_label()
    
    def get_full_screen_geometry(self):
        """获取所有屏幕的总区域（支持多显示器）"""
        screens = QApplication.screens()
        if not screens:
            return QRect(0, 0, 1920, 1080)
        min_x = min(s.geometry().x() for s in screens)
        min_y = min(s.geometry().y() for s in screens)
        max_x = max(s.geometry().x() + s.geometry().width() for s in screens)
        max_y = max(s.geometry().y() + s.geometry().height() for s in screens)
        return QRect(min_x, min_y, max_x - min_x, max_y - min_y)
    
    def setup_hint_label(self):
        """设置顶部提示标签"""
        self.hint_label = QLabel("按住鼠标左键拖拽选择区域，松开即确认（按ESC取消）", self)
        self.hint_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.hint_label.setStyleSheet("""
            QLabel {
                background-color: rgba(0, 0, 0, 180);
                color: white;
                font-size: 16px;
                font-family: "微软雅黑";
                padding: 12px 24px;
                border-radius: 8px;
            }
        """)
        self.hint_label.adjustSize()
        self.hint_label.move(
            (self.width() - self.hint_label.width()) // 2,
            30
        )
    
    def paintEvent(self, event):
        """绘制半透明遮罩 + 选中区域清除效果"""
        painter = QPainter(self)
        # 全屏半透明黑色遮罩
        painter.setBrush(QBrush(QColor(0, 0, 0, 100)))
        painter.drawRect(self.rect())
        # 如果有选中区域，将其“挖空”（完全透明）
        if self.selected_rect:
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
            painter.drawRect(self.selected_rect)
    
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.origin = event.pos()
            self.is_dragging = False
            self.selected_rect = None          # 清除之前选区
            self.rubber_band.setGeometry(QRect(self.origin, self.origin))
            self.rubber_band.show()
            self.update()                       # 重绘清除之前的选区效果
    
    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.LeftButton:
            current_rect = QRect(self.origin, event.pos()).normalized()
            # 判断是否移动超过5像素，视为拖拽开始
            if not self.is_dragging and (current_rect.width() > 5 or current_rect.height() > 5):
                self.is_dragging = True
            self.rubber_band.setGeometry(current_rect)
    
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.rubber_band.isVisible():
            final_rect = self.rubber_band.geometry().normalized()
            self.rubber_band.hide()
            
            # 情况1：没有拖拽（只是单击）-> 忽略，不关闭窗口
            if not self.is_dragging:
                return
            
            # 情况2：有拖拽但矩形太小 -> 无效，提示用户
            min_size = 10
            if final_rect.width() < min_size or final_rect.height() < min_size:
                self.hint_label.setText("选区太小，请重新拖拽（至少10×10像素）")
                QTimer.singleShot(1500, self.restore_hint)
                self.is_dragging = False   # 重置拖拽标志，等待下一次
                return
            
            # 情况3：有效选区
            self.selected_rect = final_rect
            self.update()   # 立即重绘，显示挖空效果
            # 延迟关闭，让用户看清选区
            QTimer.singleShot(500, self.accept)
    
    def restore_hint(self):
        """恢复提示文字"""
        self.hint_label.setText("按住鼠标左键拖拽选择区域，松开即确认（按ESC取消）")
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.selected_rect = None
            self.reject()   # 取消关闭
    
    def get_selected_area(self):
        """返回 (x, y, width, height) 或 None"""
        if self.selected_rect:
            return (
                self.selected_rect.x(),
                self.selected_rect.y(),
                self.selected_rect.width(),
                self.selected_rect.height()
            )
        return None


def select_screen_area():
    """
    静态函数：打开全屏选区窗口，返回选区坐标或 None
    使用模态对话框方式，不会引起事件循环嵌套问题
    """
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)
    
    selector = ScreenSelector()
    result = selector.exec()   # 模态运行
    if result == QDialog.DialogCode.Accepted:
        return selector.get_selected_area()
    else:
        return None