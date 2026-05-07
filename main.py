import sys
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
