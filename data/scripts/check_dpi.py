"""
检查系统 DPI 设置 - 用于诊断坐标问题
"""

import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QGuiApplication


def check_dpi():
    app = QApplication(sys.argv)
    
    print("=" * 60)
    print("DPI 诊断信息")
    print("=" * 60)
    
    screens = QGuiApplication.screens()
    
    for i, screen in enumerate(screens):
        geometry = screen.geometry()
        available_geometry = screen.availableGeometry()
        logical_dpi = screen.logicalDotsPerInch()
        physical_dpi = screen.physicalDotsPerInch()
        scale = logical_dpi / 96.0
        
        print(f"\n屏幕 {i}:")
        print(f"  名称: {screen.name()}")
        print(f"  逻辑分辨率: {geometry.width()} x {geometry.height()}")
        print(f"  可用区域: {available_geometry.width()} x {available_geometry.height()}")
        print(f"  逻辑DPI: {logical_dpi:.1f}")
        print(f"  物理DPI: {physical_dpi:.1f}")
        print(f"  缩放比例: {scale:.2f} (100% = 96 DPI)")
        
        if scale > 1.05:
            print(f"  ⚠️ 检测到缩放 {scale*100:.0f}%，坐标可能需要转换")
        else:
            print(f"  ✅ 无缩放或缩放比例正常")
    
    print("\n" + "=" * 60)
    print("修复建议:")
    print("  如果选区与实际截屏不符，说明坐标需要转换")
    print("  当前代码已启用 DPI 感知，应该可以正常工作")
    print("=" * 60)
    
    sys.exit(0)


if __name__ == "__main__":
    check_dpi()