"""
frame_capture.py - 屏幕/视频帧采集模块（修复高DPI问题）
"""

import cv2
import numpy as np
import mss
import ctypes
import sys
from typing import Optional, Tuple


class ScreenCapture:
    """屏幕采集器 - 修复高DPI问题"""
    
    def __init__(self, area: Tuple[int, int, int, int] = None):
        """
        初始化屏幕采集器
        :param area: 屏幕采集区域 (x, y, width, height)
        """
        # 关键修复：设置 Windows DPI 感知
        self._set_dpi_aware()
        
        self.area = area
        self.sct = mss.mss()
        self._monitor = None
        self._update_monitor()
    
    def _set_dpi_aware(self):
        """设置 Windows DPI 感知，确保坐标正确"""
        if sys.platform == 'win32':
            try:
                # Windows 10 1607+ 推荐使用
                ctypes.windll.user32.SetProcessDPIAwareness(2)
                print("[ScreenCapture] 已设置 Per-Monitor DPI Aware")
            except AttributeError:
                try:
                    # 旧版 Windows
                    ctypes.windll.user32.SetProcessDPIAware()
                    print("[ScreenCapture] 已设置 System DPI Aware")
                except:
                    print("[ScreenCapture] 无法设置 DPI 感知，坐标可能偏移")
            except Exception as e:
                print(f"[ScreenCapture] DPI 设置失败: {e}")
    
    def _update_monitor(self):
        """更新采集区域配置"""
        if self.area:
            x, y, w, h = self.area
            # mss 直接使用物理坐标
            self._monitor = {
                "left": x,
                "top": y,
                "width": w,
                "height": h
            }
            print(f"[ScreenCapture] 采集区域: left={x}, top={y}, width={w}, height={h}")
        else:
            self._monitor = self.sct.monitors[1]
            print(f"[ScreenCapture] 使用全屏: {self._monitor}")
    
    def set_area(self, area: Tuple[int, int, int, int]):
        """动态设置新的采集区域"""
        self.area = area
        self._update_monitor()
    
    def capture(self) -> Optional[np.ndarray]:
        """执行一次屏幕采集"""
        if not self._monitor:
            return None
        
        try:
            img = self.sct.grab(self._monitor)
            frame = np.array(img)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            return frame
        except Exception as e:
            print(f"[ScreenCapture] 截屏失败: {e}")
            return None
    
    def release(self):
        """释放资源"""
        pass


class VideoCapture:
    """本地视频采集器"""
    
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = None
        self._open()
    
    def _open(self):
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"无法打开视频文件: {self.video_path}")
    
    def read(self) -> Optional[np.ndarray]:
        if self.cap is None:
            return None
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame
    
    def release(self):
        if self.cap:
            self.cap.release()
            self.cap = None
    
    def is_opened(self) -> bool:
        return self.cap is not None and self.cap.isOpened()