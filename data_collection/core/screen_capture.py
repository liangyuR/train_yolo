"""
屏幕捕获抽象接口
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional
import numpy as np
from loguru import logger


class ScreenCapture(ABC):
    """屏幕捕获抽象基类"""
    
    @abstractmethod
    def Initialize(self, width: int, height: int, monitor: Optional[dict] = None) -> bool:
        """
        初始化屏幕捕获
        
        Args:
            width: 屏幕宽度
            height: 屏幕高度
            monitor: 显示器配置（可选）
            
        Returns:
            是否初始化成功
        """
        pass
    
    @abstractmethod
    def Capture(self) -> Optional[np.ndarray]:
        """
        捕获一帧屏幕
        
        Returns:
            BGR 格式的 numpy 数组，shape=(height, width, 3)，失败返回 None
        """
        pass
    
    @abstractmethod
    def GetResolution(self) -> Tuple[int, int]:
        """
        获取分辨率
        
        Returns:
            (width, height)
        """
        pass
    
    @abstractmethod
    def Cleanup(self):
        """清理资源"""
        pass
    
    @abstractmethod
    def GetCaptureMode(self) -> str:
        """
        获取捕获模式名称
        
        Returns:
            模式名称，如 'mss', 'dxcam', 'window'
        """
        pass

