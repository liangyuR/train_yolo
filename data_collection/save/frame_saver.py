"""
帧保存器抽象接口
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
import numpy as np
from loguru import logger


class FrameSaver(ABC):
    """帧保存器抽象基类"""
    
    @abstractmethod
    def SaveFrame(self, frame: np.ndarray, frame_number: int, output_path: Path) -> bool:
        """
        保存帧
        
        Args:
            frame: BGR 格式的 numpy 数组
            frame_number: 帧号
            output_path: 输出文件路径
            
        Returns:
            是否保存成功
        """
        pass
    
    @abstractmethod
    def GetStats(self) -> dict:
        """
        获取统计信息
        
        Returns:
            统计信息字典
        """
        pass

