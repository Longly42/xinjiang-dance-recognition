"""
frame_capture.py - 屏幕/视频帧采集模块
功能：提供屏幕区域采集 + 本地视频文件采集两种帧获取方式
"""
import cv2
import numpy as np
import mss  # 高速屏幕截图库
from typing import Optional, Tuple  # 类型注解，方便代码提示


class ScreenCapture:
    """屏幕采集器：使用 mss 高速截取指定屏幕区域的画面"""
    
    def __init__(self, area: Tuple[int, int, int, int] = None):
        """
        初始化屏幕采集器
        :param area: 屏幕采集区域 (x, y, width, height)，不传则默认全屏
        """
        # 存储要采集的屏幕坐标区域
        self.area = area
        # 初始化 mss 截图核心对象
        self.sct = mss.mss()
        # 存储最终传给 mss 的监控区域参数
        self._monitor = None
        # 第一次初始化时，更新采集区域配置
        self._update_monitor()
    
    def _update_monitor(self):
        """内部方法：根据传入的 area，更新 mss 需要的采集区域格式"""
        if self.area:
            # 如果用户指定了区域，解析 x/y/宽/高
            x, y, w, h = self.area
            # 构造 mss 要求的区域字典
            self._monitor = {
                "left": x,      # 左上角 x 坐标
                "top": y,       # 左上角 y 坐标
                "width": w,     # 宽度
                "height": h     # 高度
            }
        else:
            # 没有指定区域 → 默认采集第一个屏幕（全屏）
            self._monitor = self.sct.monitors[1]
    
    def set_area(self, area: Tuple[int, int, int, int]):
        """
        动态设置新的采集区域
        :param area: (x, y, width, height)
        """
        self.area = area
        self._update_monitor()  # 重新更新区域
    
    def capture(self) -> Optional[np.ndarray]:
        """
        执行一次屏幕采集
        :return: OpenCV 标准格式的帧（BGR 通道），失败返回 None
        """
        if not self._monitor:
            return None
        
        try:
            # 1. 使用 mss 抓取指定区域的屏幕画面
            img = self.sct.grab(self._monitor)
            # 2. 转为 numpy 数组（mss 默认输出 BGRA 格式：带透明通道）
            frame = np.array(img)
            # 3. 转换成 OpenCV 标准的 BGR 格式（去掉透明通道）
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            return frame
        except Exception as e:
            print(f"[ERROR] 截屏失败: {e}")
            return None


class VideoCapture:
    """本地视频采集器：从视频文件中逐帧读取画面"""
    
    def __init__(self, video_path: str):
        """
        初始化视频采集器
        :param video_path: 视频文件路径（mp4/avi/mov 等）
        """
        self.video_path = video_path
        # OpenCV 视频捕获对象
        self.cap = None
        # 打开视频文件
        self._open()
    
    def _open(self):
        """内部方法：打开视频文件，失败直接抛异常"""
        self.cap = cv2.VideoCapture(self.video_path)
        # 检查是否成功打开
        if not self.cap.isOpened():
            raise ValueError(f"无法打开视频文件: {self.video_path}")
    
    def read(self) -> Optional[np.ndarray]:
        """
        读取视频的下一帧
        :return: 视频帧图像（BGR），读取完毕/失败返回 None
        """
        if self.cap is None:
            return None
        # ret：是否读取成功；frame：读取到的图像
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame
    
    def release(self):
        """释放视频资源：必须调用，否则视频文件会被占用"""
        if self.cap:
            self.cap.release()
            self.cap = None
    
    def is_opened(self) -> bool:
        """判断视频是否处于打开状态"""
        return self.cap is not None and self.cap.isOpened()