"""
MSS 屏幕捕获实现
"""

import mss
import numpy as np
from typing import Tuple, Optional
from loguru import logger

from wot_ai.data_collection.core.screen_capture import ScreenCapture


class MssScreenCapture(ScreenCapture):
    """使用 mss 库的屏幕捕获实现"""
    
    def __init__(self):
        """初始化 MSS 捕获器"""
        self.sct_ = None
        self.monitor_ = None
        self.width_ = 0
        self.height_ = 0
    
    def Initialize(self, width: int, height: int, monitor: Optional[dict] = None) -> bool:
        """
        初始化屏幕捕获
        
        Args:
            width: 屏幕宽度
            height: 屏幕高度
            monitor: 显示器配置（如果为 None，自动检测主显示器）
            
        Returns:
            是否初始化成功
        """
        try:
            self.sct_ = mss.mss()
            
            if monitor is None:
                # 自动获取主显示器
                monitor_info = self.sct_.monitors[1]  # 主显示器
                actual_width = monitor_info['width']
                actual_height = monitor_info['height']
            else:
                actual_width = monitor.get('width', width)
                actual_height = monitor.get('height', height)
            
            self.width_ = actual_width
            self.height_ = actual_height
            
            self.monitor_ = {
                "top": monitor.get('top', 0) if monitor else 0,
                "left": monitor.get('left', 0) if monitor else 0,
                "width": actual_width,
                "height": actual_height
            }
            
            logger.info(f"MSS 屏幕捕获初始化成功: {actual_width}x{actual_height}")
            return True
            
        except Exception as e:
            logger.error(f"MSS 初始化失败: {e}")
            return False
    
    def Capture(self) -> Optional[np.ndarray]:
        """
        捕获一帧屏幕
        
        Returns:
            BGR 格式的 numpy 数组，失败返回 None
        """
        if self.sct_ is None:
            return None
        
        try:
            screenshot = self.sct_.grab(self.monitor_)
            # mss 返回 BGRA 格式，去除 alpha 通道得到 BGR
            # 使用视图而非复制以提高性能
            frame = np.array(screenshot, dtype=np.uint8)
            # 如果只有3个通道，直接返回；否则去除alpha通道
            if frame.shape[2] == 4:
                frame = frame[:, :, :3]  # 取前3个通道（BGR）
            return frame
        except Exception as e:
            logger.debug(f"屏幕捕获失败: {e}")
            return None
    
    def GetResolution(self) -> Tuple[int, int]:
        """获取分辨率"""
        return (self.width_, self.height_)
    
    def Cleanup(self):
        """清理资源"""
        if self.sct_:
            # mss 的上下文管理器会自动清理
            self.sct_ = None
    
    def GetCaptureMode(self) -> str:
        """获取捕获模式名称"""
        return "mss"

