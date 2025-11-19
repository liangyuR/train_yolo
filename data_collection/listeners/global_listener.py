"""
全局键盘监听实现（Windows Hook API）
"""

from typing import Set, Callable, Tuple
from loguru import logger

try:
    from data_collection.global_hotkey import GlobalKeyboardListener
    GLOBAL_HOOK_AVAILABLE = True
except ImportError:
    try:
        from global_hotkey import GlobalKeyboardListener
        GLOBAL_HOOK_AVAILABLE = True
    except ImportError:
        GLOBAL_HOOK_AVAILABLE = False
        logger.warning("全局键盘钩子不可用")

from wot_ai.data_collection.core.input_listener import InputListener, InputAction


class GlobalInputListener(InputListener):
    """使用全局键盘钩子的输入监听实现（可捕获所有窗口的输入）"""
    
    def __init__(self):
        """初始化输入监听器"""
        self.global_keyboard_ = None
        self.mouse_listener_ = None
        self.current_keys_ = set()
        self.mouse_position_ = (0, 0)
        self.hotkey_callbacks_ = {}
        self.is_running_ = False
        
        # 鼠标监听仍使用 pynput（全局钩子主要针对键盘）
        try:
            from pynput import mouse
            self.mouse_listener_ = mouse.Listener(
                on_move=self._on_mouse_move,
                on_click=self._on_mouse_click
            )
        except ImportError:
            logger.warning("pynput 不可用，鼠标监听将不可用")
    
    def Start(self):
        """启动监听"""
        if self.is_running_:
            return
        
        if not GLOBAL_HOOK_AVAILABLE:
            logger.error("全局键盘钩子不可用")
            return
        
        self.is_running_ = True
        
        # 全局键盘监听
        def on_key_press(key_name: str):
            self.current_keys_.add(key_name)
            # 检查热键
            if key_name.lower() in self.hotkey_callbacks_:
                try:
                    self.hotkey_callbacks_[key_name.lower()]()
                except Exception as e:
                    logger.error(f"热键回调执行失败: {e}")
        
        def on_key_release(key_name: str):
            self.current_keys_.discard(key_name)
        
        self.global_keyboard_ = GlobalKeyboardListener(
            on_press=on_key_press,
            on_release=on_key_release
        )
        self.global_keyboard_.Start()
        
        # 鼠标监听
        if self.mouse_listener_:
            self.mouse_listener_.start()
        
        logger.info("全局输入监听器已启动")
    
    def Stop(self):
        """停止监听"""
        if not self.is_running_:
            return
        
        self.is_running_ = False
        
        if self.global_keyboard_:
            self.global_keyboard_.Stop()
        if self.mouse_listener_:
            self.mouse_listener_.stop()
        
        logger.info("全局输入监听器已停止")
    
    def GetPressedKeys(self) -> Set[str]:
        """获取当前按下的键"""
        return self.current_keys_.copy()
    
    def GetMousePosition(self) -> Tuple[int, int]:
        """获取鼠标位置"""
        return self.mouse_position_
    
    def SetHotkeyCallback(self, key_name: str, callback: Callable[[], None]):
        """设置热键回调"""
        self.hotkey_callbacks_[key_name.lower()] = callback
    
    def _on_mouse_move(self, x, y):
        """鼠标移动处理"""
        self.mouse_position_ = (x, y)
    
    def _on_mouse_click(self, x, y, button, pressed):
        """鼠标点击处理"""
        if pressed:
            self.current_keys_.add(f"mouse_{button.name}")
        else:
            self.current_keys_.discard(f"mouse_{button.name}")

