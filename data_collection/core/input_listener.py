"""
输入监听抽象接口
"""

from abc import ABC, abstractmethod
from typing import Set, Callable, Tuple
from loguru import logger


class InputListener(ABC):
    """输入监听抽象基类"""
    
    @abstractmethod
    def Start(self):
        """启动监听"""
        pass
    
    @abstractmethod
    def Stop(self):
        """停止监听"""
        pass
    
    @abstractmethod
    def GetPressedKeys(self) -> Set[str]:
        """
        获取当前按下的键
        
        Returns:
            按键名称集合
        """
        pass
    
    @abstractmethod
    def GetMousePosition(self) -> Tuple[int, int]:
        """
        获取鼠标位置
        
        Returns:
            (x, y)
        """
        pass
    
    @abstractmethod
    def SetHotkeyCallback(self, key_name: str, callback: Callable[[], None]):
        """
        设置热键回调
        
        Args:
            key_name: 按键名称（如 'f9', 'f10'）
            callback: 回调函数
        """
        pass


class InputAction:
    """输入动作数据类"""
    
    def __init__(self):
        """初始化输入动作"""
        self.keys_ = set()
        self.mouse_pos_ = (0, 0)
    
    def GetKeys(self) -> Set[str]:
        """获取按键集合"""
        return self.keys_.copy()
    
    def GetMousePos(self) -> Tuple[int, int]:
        """获取鼠标位置"""
        return self.mouse_pos_
    
    def ToDict(self) -> dict:
        """转换为字典"""
        return {
            "keys": list(self.keys_),
            "mouse_pos": self.mouse_pos_
        }

