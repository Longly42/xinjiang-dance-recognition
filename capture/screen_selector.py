"""
screen_selector.py - 屏幕区域选择器
使用 QRuberBand 实现全屏拖拽选区
"""
import sys
from PyQt6.QtWidgets import (
    QWidget, QRubberBand, QApplication, 
    QVBoxLayout, QLabel
)
from PyQt6.QtCore import Qt, QRect, QPoint
from PyQt6.QtGui import QScreen, QGuiApplication


class ScreenSelector(QWidget):
    """全屏区域选择器"""
    
    def __init__(self):
        super().__init__()
        # 设置窗口样式：无边框、置顶、工具窗口（不显示在任务栏）
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        # 设置窗口背景透明
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        # 获取所有屏幕的总区域并设置窗口大小（铺满全屏）
        self.setGeometry(self.get_full_screen_geometry())
        
        # 初始化选区矩形框（拖拽时显示的方框）
        self.rubber_band = QRubberBand(QRubberBand.Shape.Rectangle, self)
        # 记录鼠标按下的起点坐标
        self.origin = QPoint()
        # 标记选区是否完成
        self.selection_finished = False
        # 存储最终选中的矩形区域
        self.selected_rect = None
        
        # 初始化顶部提示文字
        self.setup_hint_label()
    
    def get_full_screen_geometry(self):
        """获取所有屏幕的组合区域（支持多显示器）"""
        # 获取系统所有屏幕
        screens = QApplication.screens()
        # 无屏幕时默认返回 1080P 区域
        if not screens:
            return QRect(0, 0, 1920, 1080)
        
        # 计算所有屏幕的最小/最大坐标，得到整体矩形
        min_x = min(s.geometry().x() for s in screens)
        min_y = min(s.geometry().y() for s in screens)
        max_x = max(s.geometry().x() + s.geometry().width() for s in screens)
        max_y = max(s.geometry().y() + s.geometry().height() for s in screens)
        
        # 返回总屏幕矩形
        return QRect(min_x, min_y, max_x - min_x, max_y - min_y)
    
    def setup_hint_label(self):
        """设置顶部操作提示标签"""
        self.hint_label = QLabel("按住鼠标左键拖拽选择区域，松开即确认", self)
        # 文字居中
        self.hint_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # 设置标签样式：半透明黑底、白字、圆角
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
        # 自动适应文字大小
        self.hint_label.adjustSize()
        # 将提示标签放在窗口顶部居中位置
        self.hint_label.move(
            (self.width() - self.hint_label.width()) // 2,
            30
        )
    
    def paintEvent(self, event):
        """绘制半透明黑色遮罩，选中区域会变透明"""
        from PyQt6.QtGui import QPainter, QColor, QBrush
        
        painter = QPainter(self)
        # 绘制全屏半透明黑色遮罩
        painter.setBrush(QBrush(QColor(0, 0, 0, 100)))
        painter.drawRect(self.rect())
        
        # 如果选区已完成，擦除选中区域（变成完全透明）
        if self.selected_rect and self.selection_finished:
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
            painter.drawRect(self.selected_rect)
    
    def mousePressEvent(self, event):
        """鼠标左键按下：记录起点，显示选区框"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.origin = event.pos()  # 记录起始坐标
            # 初始化选区框大小（起点到起点，0大小）
            self.rubber_band.setGeometry(QRect(self.origin, self.origin))
            self.rubber_band.show()       # 显示选区框
            self.selection_finished = False
            self.selected_rect = None
    
    def mouseMoveEvent(self, event):
        """鼠标拖拽：实时更新选区框大小"""
        if self.rubber_band.isVisible():
            # 生成标准化矩形（防止宽/高为负数）
            rect = QRect(self.origin, event.pos()).normalized()
            self.rubber_band.setGeometry(rect)
    
    def mouseReleaseEvent(self, event):
        """鼠标左键松开：完成选区，隐藏选区框"""
        if event.button() == Qt.MouseButton.LeftButton and self.rubber_band.isVisible():
            # 获取最终选中的矩形
            self.selected_rect = self.rubber_band.geometry().normalized()
            self.rubber_band.hide()        # 隐藏拖拽框
            self.selection_finished = True
            self.update()  # 触发重绘，显示透明选区效果
            
            # 延迟 500ms 再关闭，让用户看清选区
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(500, self.confirm_selection)
    
    def confirm_selection(self):
        """确认选区有效（宽高>10）则关闭窗口，否则重置"""
        if self.selected_rect and self.selected_rect.width() > 10 and self.selected_rect.height() > 10:
            self.close()  # 有效选区，关闭窗口
        else:
            # 无效选区，重置状态
            self.selection_finished = False
            self.selected_rect = None
            self.update()
    
    def keyPressEvent(self, event):
        """按 ESC 键：取消选择并关闭窗口"""
        if event.key() == Qt.Key.Key_Escape:
            self.selected_rect = None
            self.close()
    
    def get_selected_area(self):
        """返回选中区域坐标：(x, y, width, height)，未选择返回 None"""
        if self.selected_rect:
            return (
                self.selected_rect.x(),
                self.selected_rect.y(),
                self.selected_rect.width(),
                self.selected_rect.height()
            )
        return None


def select_screen_area() -> tuple:
    """
    静态方法：打开全屏选区窗口，返回用户选择的区域
    返回: (x, y, width, height) 或 None（用户取消）
    """
    # 获取当前应用实例，没有则新建
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)
    
    # 创建选择器并显示
    selector = ScreenSelector()
    selector.show()
    
    # 启动 Qt 事件循环，直到窗口关闭
    app.exec()
    
    # 返回最终选中的坐标
    return selector.get_selected_area()