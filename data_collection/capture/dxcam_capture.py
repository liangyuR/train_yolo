"""
DXCam 屏幕捕获实现（高性能，Windows 专用）
"""

import numpy as np
from typing import Tuple, Optional
from loguru import logger

try:
    import dxcam
    DXCAM_AVAILABLE = True
except ImportError:
    DXCAM_AVAILABLE = False
    logger.warning("dxcam 不可用，请安装: pip install dxcam")

from wot_ai.data_collection.core.screen_capture import ScreenCapture


class DxcamScreenCapture(ScreenCapture):
    """使用 dxcam 库的屏幕捕获实现（GPU 加速，比 mss 快 2-3 倍）"""
    
    def __init__(self):
        """初始化 DXCam 捕获器"""
        self.camera_ = None
        self.width_ = 0
        self.height_ = 0
    
    def Initialize(self, width: int, height: int, monitor: Optional[dict] = None) -> bool:
        """
        初始化屏幕捕获
        
        Args:
            width: 屏幕宽度
            height: 屏幕高度
            monitor: 显示器配置（可选，dxcam 会自动检测）
            
        Returns:
            是否初始化成功
        """
        if not DXCAM_AVAILABLE:
            logger.error("dxcam 库未安装")
            return False
        
        try:
            self.camera_ = dxcam.create(region=(0, 0, width, height), output_color="BGR")
            
            # 获取实际分辨率
            if monitor:
                self.width_ = monitor.get('width', width)
                self.height_ = monitor.get('height', height)
            else:
                self.width_ = width
                self.height_ = height
            
            logger.info(f"DXCam 屏幕捕获初始化成功: {self.width_}x{self.height_}")
            return True
            
        except Exception as e:
            logger.error(f"DXCam 初始化失败: {e}")
            return False
    
    def Capture(self) -> Optional[np.ndarray]:
        """
        捕获一帧屏幕
        
        Returns:
            BGR 格式的 numpy 数组，失败返回 None
        """
        if self.camera_ is None:
            return None
        
        try:
            # dxcam 直接返回 BGR numpy 数组，无需转换
            frame = self.camera_.grab()
            return frame
        except Exception as e:
            logger.debug(f"屏幕捕获失败: {e}")
            return None
    
    def GetResolution(self) -> Tuple[int, int]:
        """获取分辨率"""
        return (self.width_, self.height_)
    
    def Cleanup(self):
        """清理资源"""
        if self.camera_:
            try:
                self.camera_.release()
            except:
                pass
            self.camera_ = None
    
    def GetCaptureMode(self) -> str:
        """获取捕获模式名称"""
        return "dxcam"

