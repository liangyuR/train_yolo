"""
全局键盘监听器 - 使用 Windows Hook API
Global keyboard listener using Windows Hook API
"""

import ctypes
from ctypes import wintypes
import threading
from typing import Callable, Set

try:
    from loguru import logger
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger(__name__)

WH_KEYBOARD_LL = 13
WM_KEYDOWN = 0x0100
WM_KEYUP = 0x0101
WM_SYSKEYDOWN = 0x0104
WM_SYSKEYUP = 0x0105

# Virtual Key Codes Mappings
VK_TO_CHAR = {
    0x41: 'a', 0x42: 'b', 0x43: 'c', 0x44: 'd', 0x45: 'e', 0x46: 'f',
    0x47: 'g', 0x48: 'h', 0x49: 'i', 0x4A: 'j', 0x4B: 'k', 0x4C: 'l',
    0x4D: 'm', 0x4E: 'n', 0x4F: 'o', 0x50: 'p', 0x51: 'q', 0x52: 'r',
    0x53: 's', 0x54: 't', 0x55: 'u', 0x56: 'v', 0x57: 'w', 0x58: 'x',
    0x59: 'y', 0x5A: 'z',
    0x30: '0', 0x31: '1', 0x32: '2', 0x33: '3', 0x34: '4',
    0x35: '5', 0x36: '6', 0x37: '7', 0x38: '8', 0x39: '9',
    0x70: 'f1', 0x71: 'f2', 0x72: 'f3', 0x73: 'f4', 0x74: 'f5', 0x75: 'f6',
    0x76: 'f7', 0x77: 'f8', 0x78: 'f9', 0x79: 'f10', 0x7A: 'f11', 0x7B: 'f12',
    0x20: 'space', 0x0D: 'enter', 0x09: 'tab', 0x08: 'backspace',
    0x1B: 'esc', 0x10: 'shift', 0x11: 'ctrl', 0x12: 'alt',
    0x25: 'left', 0x26: 'up', 0x27: 'right', 0x28: 'down',
}

class KBDLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [
        ("vkCode", wintypes.DWORD),
        ("scanCode", wintypes.DWORD),
        ("flags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(wintypes.ULONG)),
    ]

HOOKPROC = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)

class GlobalKeyboardListener:
    def __init__(self, on_press: Callable = None, on_release: Callable = None):
        self.on_press_ = on_press
        self.on_release_ = on_release
        self.pressed_keys_: Set[str] = set()
        self.hook_id_ = None
        self.hook_thread_ = None
        self.running_ = False
        
        self.user32_ = ctypes.windll.user32
        self.kernel32_ = ctypes.windll.kernel32
        
        self.user32_.SetWindowsHookExW.argtypes = [ctypes.c_int, HOOKPROC, wintypes.HINSTANCE, wintypes.DWORD]
        self.user32_.SetWindowsHookExW.restype = wintypes.HHOOK
        self.user32_.CallNextHookEx.argtypes = [wintypes.HHOOK, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM]
        self.user32_.CallNextHookEx.restype = ctypes.c_long
        self.callback_func_ = HOOKPROC(self._hookCallback)

    def _hookCallback(self, nCode: int, wParam: wintypes.WPARAM, lParam: wintypes.LPARAM) -> int:
        if nCode >= 0:
            kb_struct = ctypes.cast(lParam, ctypes.POINTER(KBDLLHOOKSTRUCT)).contents
            vk_code = kb_struct.vkCode
            key_name = VK_TO_CHAR.get(vk_code, f'vk_{vk_code}')
            
            if wParam in (WM_KEYDOWN, WM_SYSKEYDOWN):
                if key_name not in self.pressed_keys_:
                    self.pressed_keys_.add(key_name)
                    if self.on_press_:
                        try: self.on_press_(key_name)
                        except: pass
            elif wParam in (WM_KEYUP, WM_SYSKEYUP):
                if key_name in self.pressed_keys_:
                    self.pressed_keys_.discard(key_name)
                    if self.on_release_:
                        try: self.on_release_(key_name)
                        except: pass
                        
        return self.user32_.CallNextHookEx(None, nCode, wParam, lParam)
    
    def _messageLoop(self):
        try:
            # Error 126: The specified module could not be found.
            # This usually happens when passing a module handle for a hook proc that is not in a DLL.
            # For WH_KEYBOARD_LL, we must set hMod to NULL (or the module handle of the current process if Python behaves like a DLL host, but NULL usually works best for low-level hooks in scripts) 
            # AND dwThreadId to 0.
            
            # However, some Python ctypes implementations require GetModuleHandle(None) to work correctly on some Windows versions.
            # If GetModuleHandle(None) fails (Error 126 might indicate it's looking for a DLL logic), let's try explicit NULL (0).
            
            self.hook_id_ = self.user32_.SetWindowsHookExW(
                WH_KEYBOARD_LL,
                self.callback_func_,
                0, # Try passing 0 (NULL) for hInstance
                0
            )
            
            if not self.hook_id_:
                # If 0 fails, try GetModuleHandleW(None)
                h_mod = self.kernel32_.GetModuleHandleW(None)
                self.hook_id_ = self.user32_.SetWindowsHookExW(
                    WH_KEYBOARD_LL,
                    self.callback_func_,
                    h_mod,
                    0
                )
            
            if not self.hook_id_:
                logger.error(f"Failed to install keyboard hook, error code: {self.kernel32_.GetLastError()}")
                return
            
            logger.info(f"✓ 全局键盘钩子已安装 (Hook ID: {self.hook_id_})")
            
            msg = wintypes.MSG()
            while self.running_:
                bRet = self.user32_.GetMessageW(ctypes.byref(msg), None, 0, 0)
                if bRet <= 0: break
                self.user32_.TranslateMessage(ctypes.byref(msg))
                self.user32_.DispatchMessageW(ctypes.byref(msg))
                
        except Exception as e:
            logger.error(f"Loop Error: {e}")
        finally:
            if self.hook_id_:
                self.user32_.UnhookWindowsHookEx(self.hook_id_)
                self.hook_id_ = None

    def Start(self):
        if self.running_: return
        self.running_ = True
        self.hook_thread_ = threading.Thread(target=self._messageLoop, daemon=True)
        self.hook_thread_.start()
        import time; time.sleep(0.1)
        
    def Stop(self):
        if not self.running_: return
        self.running_ = False
        self.hook_thread_ = None
    
    def GetPressedKeys(self) -> Set[str]:
        return self.pressed_keys_.copy()

if __name__ == "__main__":
    l = GlobalKeyboardListener(lambda k: print(f"Press {k}"), lambda k: print(f"Release {k}"))
    l.Start()
    import time
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        l.Stop()
