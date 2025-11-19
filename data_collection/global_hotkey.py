"""
全局键盘监听器 - 使用 Windows Hook API
Global keyboard listener using Windows Hook API

pynput 在游戏等应用中可能失效，此模块使用底层 Windows API
确保能够捕获所有键盘输入，无论焦点在哪个窗口
"""

import ctypes
from ctypes import wintypes
import threading
from typing import Callable, Set, Dict

# 使用标准 logging，更兼容
try:
    from loguru import logger
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger(__name__)

# Windows 常量
WH_KEYBOARD_LL = 13
WM_KEYDOWN = 0x0100
WM_KEYUP = 0x0101
WM_SYSKEYDOWN = 0x0104
WM_SYSKEYUP = 0x0105

# Virtual Key Codes
VK_ESCAPE = 0x1B
VK_SHIFT = 0x10
VK_CONTROL = 0x11
VK_MENU = 0x12  # ALT key

# 虚拟键码到字符的映射
VK_TO_CHAR = {
    # 字母键
    0x41: 'a', 0x42: 'b', 0x43: 'c', 0x44: 'd', 0x45: 'e', 0x46: 'f',
    0x47: 'g', 0x48: 'h', 0x49: 'i', 0x4A: 'j', 0x4B: 'k', 0x4C: 'l',
    0x4D: 'm', 0x4E: 'n', 0x4F: 'o', 0x50: 'p', 0x51: 'q', 0x52: 'r',
    0x53: 's', 0x54: 't', 0x55: 'u', 0x56: 'v', 0x57: 'w', 0x58: 'x',
    0x59: 'y', 0x5A: 'z',
    
    # 数字键
    0x30: '0', 0x31: '1', 0x32: '2', 0x33: '3', 0x34: '4',
    0x35: '5', 0x36: '6', 0x37: '7', 0x38: '8', 0x39: '9',
    
    # 功能键
    0x70: 'f1', 0x71: 'f2', 0x72: 'f3', 0x73: 'f4', 0x74: 'f5', 0x75: 'f6',
    0x76: 'f7', 0x77: 'f8', 0x78: 'f9', 0x79: 'f10', 0x7A: 'f11', 0x7B: 'f12',
    
    # 特殊键
    0x20: 'space',
    0x0D: 'enter',
    0x09: 'tab',
    0x08: 'backspace',
    0x1B: 'esc',
    0x10: 'shift',
    0x11: 'ctrl',
    0x12: 'alt',
    0x14: 'caps_lock',
    
    # 方向键
    0x25: 'left',
    0x26: 'up',
    0x27: 'right',
    0x28: 'down',
    
    # 其他
    0x2D: 'insert',
    0x2E: 'delete',
    0x21: 'page_up',
    0x22: 'page_down',
    0x24: 'home',
    0x23: 'end',
}


class KBDLLHOOKSTRUCT(ctypes.Structure):
    """键盘钩子结构"""
    _fields_ = [
        ("vkCode", wintypes.DWORD),
        ("scanCode", wintypes.DWORD),
        ("flags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(wintypes.ULONG)),
    ]


class GlobalKeyboardListener:
    """
    全局键盘监听器
    
    使用 Windows Low-Level Keyboard Hook (WH_KEYBOARD_LL)
    可以捕获系统中所有键盘事件，不受窗口焦点影响
    """
    
    def __init__(self, on_press: Callable = None, on_release: Callable = None):
        """
        初始化全局键盘监听器
        
        Args:
            on_press: 按键按下回调函数 callback(key_name: str)
            on_release: 按键释放回调函数 callback(key_name: str)
        """
        self.on_press_ = on_press
        self.on_release_ = on_release
        
        # 当前按下的键
        self.pressed_keys_: Set[str] = set()
        
        # Hook 相关
        self.hook_id_ = None
        self.hook_thread_ = None
        self.running_ = False
        
        # Windows API
        self.user32_ = ctypes.windll.user32
        self.kernel32_ = ctypes.windll.kernel32
        
        # 创建回调函数（必须保持引用，否则会被垃圾回收）
        self.callback_func_ = ctypes.WINFUNCTYPE(
            ctypes.c_int,
            ctypes.c_int,
            wintypes.WPARAM,
            wintypes.LPARAM
        )(self._hookCallback)
        
    def _hookCallback(self, nCode: int, wParam: wintypes.WPARAM, lParam: wintypes.LPARAM) -> int:
        """
        Hook 回调函数
        
        这个函数会在每次键盘事件时被 Windows 调用
        """
        if nCode >= 0:
            # 解析键盘事件
            kb_struct = ctypes.cast(lParam, ctypes.POINTER(KBDLLHOOKSTRUCT)).contents
            vk_code = kb_struct.vkCode
            
            # 转换虚拟键码为字符
            key_name = VK_TO_CHAR.get(vk_code, f'vk_{vk_code}')
            
            # 按键按下事件
            if wParam in (WM_KEYDOWN, WM_SYSKEYDOWN):
                if key_name not in self.pressed_keys_:
                    self.pressed_keys_.add(key_name)
                    if self.on_press_:
                        try:
                            self.on_press_(key_name)
                        except Exception as e:
                            logger.error(f"on_press callback error: {e}")
            
            # 按键释放事件
            elif wParam in (WM_KEYUP, WM_SYSKEYUP):
                if key_name in self.pressed_keys_:
                    self.pressed_keys_.discard(key_name)
                    if self.on_release_:
                        try:
                            self.on_release_(key_name)
                        except Exception as e:
                            logger.error(f"on_release callback error: {e}")
        
        # 调用下一个钩子
        return self.user32_.CallNextHookEx(None, nCode, wParam, lParam)
    
    def _messageLoop(self):
        """消息循环（必须在独立线程中运行）"""
        try:
            # 保存线程 ID 用于发送退出消息
            self.thread_id_ = self.kernel32_.GetCurrentThreadId()
            
            # 设置键盘钩子
            h_mod = self.kernel32_.GetModuleHandleW(None)
            self.hook_id_ = self.user32_.SetWindowsHookExW(
                WH_KEYBOARD_LL,
                self.callback_func_,
                h_mod,
                0
            )
            
            if not self.hook_id_:
                error_code = self.kernel32_.GetLastError()
                logger.error(f"Failed to install keyboard hook, error code: {error_code}")
                return
            
            logger.info(f"✓ 全局键盘钩子已安装 (Hook ID: {self.hook_id_})")
            
            # 进入消息循环 - 使用阻塞式 GetMessage
            msg = wintypes.MSG()
            while self.running_:
                # GetMessage 会阻塞等待消息，返回 -1 表示错误
                bRet = self.user32_.GetMessageW(
                    ctypes.byref(msg),
                    None,
                    0,
                    0
                )
                
                if bRet == 0:  # WM_QUIT
                    break
                elif bRet == -1:  # Error
                    logger.error("GetMessage failed")
                    break
                
                self.user32_.TranslateMessage(ctypes.byref(msg))
                self.user32_.DispatchMessageW(ctypes.byref(msg))
                
        except Exception as e:
            logger.error(f"Message loop error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # 卸载钩子
            if self.hook_id_:
                self.user32_.UnhookWindowsHookEx(self.hook_id_)
                logger.info("键盘钩子已卸载")
                self.hook_id_ = None
    
    def Start(self):
        """启动监听器"""
        if self.running_:
            logger.warning("Listener already running")
            return
        
        self.running_ = True
        self.hook_thread_ = threading.Thread(target=self._messageLoop, daemon=True)
        self.hook_thread_.start()
        
        # 等待钩子安装完成
        import time
        time.sleep(0.1)
        
    def Stop(self):
        """停止监听器"""
        if not self.running_:
            return
        
        self.running_ = False
        
        if self.hook_thread_ and self.hook_thread_.is_alive():
            # 发送退出消息到消息循环线程
            if hasattr(self, 'thread_id_') and self.thread_id_:
                try:
                    self.user32_.PostThreadMessageW(
                        self.thread_id_,
                        0x0012,  # WM_QUIT
                        0,
                        0
                    )
                except:
                    pass
            
            self.hook_thread_.join(timeout=2)
            if self.hook_thread_.is_alive():
                logger.warning("Hook thread did not stop cleanly")
            self.hook_thread_ = None
    
    def GetPressedKeys(self) -> Set[str]:
        """获取当前按下的键"""
        return self.pressed_keys_.copy()


def test():
    """测试全局键盘监听器"""
    logger.info("=" * 80)
    logger.info("全局键盘监听器测试")
    logger.info("=" * 80)
    
    # 检查管理员权限
    try:
        is_admin = ctypes.windll.shell32.IsUserAnAdmin()
        if is_admin:
            logger.info("✓ 以管理员权限运行")
        else:
            logger.warning("⚠️  未以管理员权限运行（可能影响某些游戏）")
    except:
        pass
    
    logger.info("")
    logger.info("测试说明：")
    logger.info("  1. 此监听器使用底层 Windows Hook API")
    logger.info("  2. 可以捕获所有应用程序的键盘输入")
    logger.info("  3. 切换到其他窗口（如游戏）按键仍会被记录")
    logger.info("  4. 按 F10 键退出测试")
    logger.info("=" * 80)
    logger.info("")
    
    press_count = 0
    stop_flag = threading.Event()
    
    def on_press(key_name: str):
        nonlocal press_count
        press_count += 1
        logger.info(f"✓ 按下: '{key_name}' | 总计: {press_count}")
        
        if key_name == 'f10':
            logger.info("")
            logger.info("检测到 F10，准备退出...")
            stop_flag.set()
    
    def on_release(key_name: str):
        logger.info(f"  释放: '{key_name}'")
    
    # 创建监听器
    listener = GlobalKeyboardListener(on_press=on_press, on_release=on_release)
    
    try:
        listener.Start()
        logger.info("监听器已启动，现在可以切换到其他窗口测试...")
        logger.info("按任意键测试，按 F10 退出")
        logger.info("")
        
        # 等待退出信号
        stop_flag.wait()
        
    finally:
        listener.Stop()
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("测试结束")
        logger.info(f"总按键次数: {press_count}")
        if press_count == 0:
            logger.error("❌ 没有捕获到任何按键！")
        else:
            logger.info("✓ 全局键盘监听功能正常！")
        logger.info("=" * 80)


if __name__ == "__main__":
    test()

