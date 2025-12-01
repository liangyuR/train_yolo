"""
Gameplay Recorder Module - 游戏录制核心逻辑

简化后的录制管线：
1. 初始化：配置 -> 屏幕捕获(MSS) -> 异步保存器 -> 输入监听
2. 循环：
   - Wait for Start (F9)
   - Recording Loop:
     - Capture Screen
     - Capture Input
     - Async Save Frame
     - Check Stop (F10)
   - Stop & Save Metadata
"""

import time
import json
import threading
import sys
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import cv2
from loguru import logger

# 确保 src 可以被导入
# 这种问题通常是因为 record_gameplay.py 被导入时，sys.path 还没有正确设置，或者导入路径与 sys.path 不匹配
# 在 main.py 中已经设置了 PROJECT_ROOT (data_collection 目录) 到 sys.path
# 所以 import src.core.mss_capture 应该是正确的，前提是 data_collection 目录下有 src 目录

# 为了保险起见，我们也在这里尝试修复路径
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.core.mss_capture import MssScreenCapture
    from src.core.async_frame_saver import AsyncFrameSaver
    from src.core.global_listener import GlobalInputListener
    from src.utils.global_config import GetConfig
except ImportError:
    # 如果直接从 data_collection/src 运行，可能需要 data_collection.src.core...
    # 或者如果 sys.path 是 data_collection/src，那么应该是 core.mss_capture
    # 让我们尝试相对导入或调整 sys.path
    logger.warning("尝试标准导入失败，尝试备用导入路径...")
    try:
        # 假设 sys.path 包含了 data_collection/src
        from core.mss_capture import MssScreenCapture
        from core.async_frame_saver import AsyncFrameSaver
        from core.global_listener import GlobalInputListener
        from utils.global_config import GetConfig
    except ImportError:
        # 假设我们在 data_collection 外部运行
        from data_collection.src.core.mss_capture import MssScreenCapture
        from data_collection.src.core.async_frame_saver import AsyncFrameSaver
        from data_collection.src.core.global_listener import GlobalInputListener
        from data_collection.src.utils.global_config import GetConfig


class RecorderController:
    def __init__(self):
        self.config = GetConfig()
        self.is_running = False  # 整体运行状态
        self.is_recording = False  # 录制状态
        self.should_exit = False   # 退出标志

        # Modules
        self.capture = MssScreenCapture()
        self.saver: Optional[AsyncFrameSaver] = None
        self.listener = GlobalInputListener()
        
        # State
        self.session_dir: Optional[Path] = None
        self.frame_count = 0
        self.start_time = 0.0
        self.actions = []
        self.frame_map = []  # frame_id -> timestamp

        # Threads
        self.record_thread: Optional[threading.Thread] = None

    def initialize(self) -> bool:
        """初始化所有子模块"""
        try:
            # 1. Screen Capture
            c_conf = self.config['capture']['fullscreen']
            if not self.capture.Initialize(c_conf['width'], c_conf['height']):
                logger.error("屏幕捕获初始化失败")
                return False

            # 2. Input Listener
            self.listener.Start()
            
            # 注册热键回调
            hotkeys = self.config.get('hotkeys', {'start': 'f9', 'stop': 'f10'})
            self.listener.SetHotkeyCallback(hotkeys['start'], self.start_recording_trigger)
            self.listener.SetHotkeyCallback(hotkeys['stop'], self.stop_recording_trigger)

            logger.info(f"录制控制器就绪. 按 {hotkeys['start']} 开始, {hotkeys['stop']} 停止")
            return True
        except Exception as e:
            logger.error(f"初始化异常: {e}")
            return False

    def start_recording_trigger(self):
        """热键触发开始"""
        if not self.is_recording:
            logger.info(">>> 触发开始录制")
            self.is_recording = True

    def stop_recording_trigger(self):
        """热键触发停止"""
        if self.is_recording:
            logger.info("<<< 触发停止录制")
            self.is_recording = False

    def prepare_session(self):
        """准备新会话目录与保存器"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = Path(self.config['capture']['output_dir'])
        self.session_dir = out_dir / f"session_{timestamp}"
        frames_dir = self.session_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        self.saver = AsyncFrameSaver(
            frames_dir=frames_dir,
            png_compression=3,
            queue_size=120
        )
        self.saver.Start()

        self.frame_count = 0
        self.actions = []
        self.frame_map = []
        self.start_time = time.time()
        
        logger.info(f"新会话: {self.session_dir}")

    def run_loop(self):
        """主运行循环 (Blocking)"""
        self.is_running = True
        fps = self.config['capture']['fps']
        interval = 1.0 / fps
        
        logger.info("进入主循环，等待录制...")

        while not self.should_exit:
            # Idle wait
            if not self.is_recording:
                time.sleep(0.1)
                continue

            # Start Recording
            self.prepare_session()
            
            # Recording Loop
            while self.is_recording and not self.should_exit:
                loop_start = time.time()
                
                # 1. Capture
                frame = self.capture.Capture()
                if frame is None:
                    logger.warning("捕获失败，跳过帧")
                    continue

                # 2. Input
                keys = self.listener.GetPressedKeys()
                mouse_pos = self.listener.GetMousePosition()
                
                current_time = time.time() - self.start_time

                # 3. Save Frame (Async)
                # Resize if needed (Target Resolution)
                tgt_w = self.config['target_resolution']['width']
                tgt_h = self.config['target_resolution']['height']
                if frame.shape[1] != tgt_w or frame.shape[0] != tgt_h:
                    frame = cv2.resize(frame, (tgt_w, tgt_h), interpolation=cv2.INTER_AREA)

                if self.saver:
                    self.saver.SaveFrame(frame, self.frame_count)

                # 4. Record Action
                self.actions.append({
                    "frame": self.frame_count,
                    "time": current_time,
                    "keys": list(keys),
                    "mouse": mouse_pos
                })
                
                self.frame_count += 1

                # 5. Sleep to maintain FPS
                elapsed = time.time() - loop_start
                sleep_time = max(0, interval - elapsed)
                time.sleep(sleep_time)

                # Log periodically
                if self.frame_count % int(fps * 5) == 0:
                    q_size = self.saver.GetQueueSize() if self.saver else 0
                    logger.info(f"录制中... 帧: {self.frame_count}, 队列: {q_size}")

            # Stop Recording & Save Metadata
            self.finish_session()

    def finish_session(self):
        """结束会话并保存元数据"""
        if self.saver:
            self.saver.Stop()
            self.saver = None

        if not self.session_dir:
            return

        duration = time.time() - self.start_time
        meta = {
            "fps": self.config['capture']['fps'],
            "total_frames": self.frame_count,
            "duration": duration,
            "resolution": self.config['target_resolution'],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        with open(self.session_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        
        with open(self.session_dir / "actions.json", "w", encoding="utf-8") as f:
            json.dump(self.actions, f, indent=2)

        logger.info(f"会话结束. 保存 {self.frame_count} 帧, 时长 {duration:.1f}s")

    def cleanup(self):
        """清理资源"""
        self.should_exit = True
        self.is_recording = False
        if self.saver:
            self.saver.Stop()
        self.listener.Stop()
        self.capture.Cleanup()
        logger.info("资源已清理")

def run_recorder():
    """Entry point for background thread"""
    controller = RecorderController()
    if controller.initialize():
        try:
            controller.run_loop()
        except KeyboardInterrupt:
            pass
        finally:
            controller.cleanup()
