"""
Pynput 输入监听实现
"""

from typing import Set, Callable, Tuple
from pynput import keyboard, mouse
from loguru import logger

from wot_ai.data_collection.core.input_listener import InputListener, InputAction


class PynputInputListener(InputListener):
    """使用 pynput 的输入监听实现"""
    
    def __init__(self):
        """初始化输入监听器"""
        self.keyboard_listener_ = None
        self.mouse_listener_ = None
        self.current_keys_ = set()
        self.mouse_position_ = (0, 0)
        self.hotkey_callbacks_ = {}  # key_name -> callback
        self.is_running_ = False
    
    def Start(self):
        """启动监听"""
        if self.is_running_:
            return
        
        self.is_running_ = True
        
        # 键盘监听
        self.keyboard_listener_ = keyboard.Listener(
            on_press=self._on_key_press,
            on_release=self._on_key_release
        )
        self.keyboard_listener_.start()
        
        # 鼠标监听
        self.mouse_listener_ = mouse.Listener(
            on_move=self._on_mouse_move,
            on_click=self._on_mouse_click
        )
        self.mouse_listener_.start()
        
        logger.info("Pynput 输入监听器已启动")
    
    def Stop(self):
        """停止监听"""
        if not self.is_running_:
            return
        
        self.is_running_ = False
        
        if self.keyboard_listener_:
            self.keyboard_listener_.stop()
        if self.mouse_listener_:
            self.mouse_listener_.stop()
        
        logger.info("Pynput 输入监听器已停止")
    
    def GetPressedKeys(self) -> Set[str]:
        """获取当前按下的键"""
        return self.current_keys_.copy()
    
    def GetMousePosition(self) -> Tuple[int, int]:
        """获取鼠标位置"""
        return self.mouse_position_
    
    def SetHotkeyCallback(self, key_name: str, callback: Callable[[], None]):
        """设置热键回调"""
        self.hotkey_callbacks_[key_name.lower()] = callback
    
    def _on_key_press(self, key):
        """按键按下处理"""
        try:
            key_str = key.char
            self.current_keys_.add(key_str)
        except AttributeError:
            # 特殊键
            key_str = str(key).replace('Key.', '').lower()
            self.current_keys_.add(key_str)
            
            # 检查热键
            if key_str in self.hotkey_callbacks_:
                try:
                    self.hotkey_callbacks_[key_str]()
                except Exception as e:
                    logger.error(f"热键回调执行失败: {e}")
    
    def _on_key_release(self, key):
        """按键释放处理"""
        try:
            key_str = key.char
            self.current_keys_.discard(key_str)
        except AttributeError:
            key_str = str(key).replace('Key.', '').lower()
            self.current_keys_.discard(key_str)
    
    def _on_mouse_move(self, x, y):
        """鼠标移动处理"""
        self.mouse_position_ = (x, y)
    
    def _on_mouse_click(self, x, y, button, pressed):
        """鼠标点击处理"""
        if pressed:
            self.current_keys_.add(f"mouse_{button.name}")
        else:
            self.current_keys_.discard(f"mouse_{button.name}")

