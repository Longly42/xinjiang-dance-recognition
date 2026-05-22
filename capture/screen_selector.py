"""
screen_selector.py - 屏幕区域选择器（修复高DPI坐标问题）
使用 QScreen 获取物理像素坐标，确保选区与截屏一致
"""

import sys
from PyQt6.QtWidgets import (
    QDialog, QRubberBand, QApplication, QLabel
)
from PyQt6.QtCore import Qt, QRect, QPoint, QTimer
from PyQt6.QtGui import QPainter, QColor, QBrush, QGuiApplication


class ScreenSelector(QDialog):
    """全屏区域选择器 - 修复高DPI坐标问题"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 设置透明背景
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        # 无边框、置顶
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint
        )
        
        # 获取所有屏幕的物理几何区域
        self.screen_geometry = self.get_physical_screen_geometry()
        self.setGeometry(self.screen_geometry)
        
        # 橡胶框（选区拖拽框）
        self.rubber_band = QRubberBand(QRubberBand.Shape.Rectangle, self)
        self.origin = QPoint()
        self.is_dragging = False
        self.selected_rect = None
        
        # 顶部提示标签
        self.setup_hint_label()
        
        # 显示窗口
        self.showFullScreen()
    
    def get_physical_screen_geometry(self) -> QRect:
        """
        获取所有屏幕的物理像素几何区域
        关键：使用 QGuiApplication.screens() 获取真实物理坐标
        """
        screens = QGuiApplication.screens()
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
            self.selected_rect = None
            self.rubber_band.setGeometry(QRect(self.origin, self.origin))
            self.rubber_band.show()
            self.update()
    
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
                self.is_dragging = False
                return
            
            # 情况3：有效选区
            self.selected_rect = final_rect
            self.update()
            # 延迟关闭，让用户看清选区
            QTimer.singleShot(500, self.accept)
    
    def restore_hint(self):
        """恢复提示文字"""
        self.hint_label.setText("按住鼠标左键拖拽选择区域，松开即确认（按ESC取消）")
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.selected_rect = None
            self.reject()
    
    # ========== 新增：逻辑坐标转物理坐标 ==========
    def logical_to_physical_rect(self, rect: QRect) -> tuple:
        """
        将逻辑坐标矩形转换为物理像素坐标。
        解决高DPI缩放下选区偏移/缩放问题。
        """
        # 获取主屏幕（作为 fallback）
        screen = QGuiApplication.primaryScreen()
        if not screen:
            # 没有屏幕信息时，直接返回原始坐标（按逻辑坐标处理）
            return (rect.x(), rect.y(), rect.width(), rect.height())
        
        # 获取选区中心点所在的屏幕（支持多显示器不同缩放）
        center = rect.center()
        target_screen = QGuiApplication.screenAt(center)
        if target_screen is None:
            target_screen = screen
        
        dpr = target_screen.devicePixelRatio()
        x = int(rect.x() * dpr)
        y = int(rect.y() * dpr)
        w = int(rect.width() * dpr)
        h = int(rect.height() * dpr)
        
        print(f"[ScreenSelector] 坐标转换: 逻辑({rect.x()},{rect.y()},{rect.width()},{rect.height()}) -> "
              f"物理({x},{y},{w},{h}) (dpr={dpr})")
        return (x, y, w, h)
    
    def get_selected_area(self):
        """
        返回选区的**物理像素坐标** (x, y, width, height)
        此坐标可直接用于 mss 截图。
        """
        if self.selected_rect:
            return self.logical_to_physical_rect(self.selected_rect)
        return None


def select_screen_area():
    """
    静态函数：打开全屏选区窗口，返回选区物理坐标或 None
    """
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)
    
    selector = ScreenSelector()
    result = selector.exec()
    
    if result == QDialog.DialogCode.Accepted:
        return selector.get_selected_area()
    else:
        return None


# 测试入口
if __name__ == "__main__":
    app = QApplication(sys.argv)
    area = select_screen_area()
    print(f"选择的物理区域: {area}")
    sys.exit(0)