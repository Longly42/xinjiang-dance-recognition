"""
main.py - 程序入口
"""

import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt6.QtWidgets import QApplication
from gui.ui_settings import GLOBAL_STYLESHEET
from gui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    
    # 设置全局样式表
    app.setStyleSheet(GLOBAL_STYLESHEET)
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()