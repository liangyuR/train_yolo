"""
Record gameplay data for imitation learning
"""

import argparse
import cv2
import numpy as np
import time
import json
import os
from pathlib import Path
from pynput import keyboard, mouse
from loguru import logger
import sys
import yaml

sys.path.append(str(Path(__file__).parent.parent))


def get_base_path():
    """获取程序基础路径（兼容 PyInstaller 打包）"""
    if getattr(sys, 'frozen', False):
        # 如果是打包后的 exe
        return Path(sys.executable).parent
    else:
        # 如果是 Python 脚本
        return Path(__file__).parent.parent

# 屏幕捕获：使用 Python mss 库
# 注：C++ 模块已移除，仅用于未来的高性能实时 bot
import mss
CPP_AVAILABLE = False  # 数据采集不需要 C++ 高性能模块

# 窗口捕获
try:
    from utils.window_capture import WindowCapture
    WINDOW_CAPTURE_AVAILABLE = True
except ImportError:
    logger.warning("Window capture not available")
    WINDOW_CAPTURE_AVAILABLE = False

# 异步帧保存器
try:
    from data_collection.async_frame_saver import AsyncFrameSaver
    ASYNC_SAVER_AVAILABLE = True
except ImportError:
    try:
        from async_frame_saver import AsyncFrameSaver
        ASYNC_SAVER_AVAILABLE = True
    except ImportError:
        logger.warning("异步帧保存器不可用，将使用同步保存（可能导致卡顿）")
        ASYNC_SAVER_AVAILABLE = False

# 游戏状态检测器（OCR）
try:
    from data_collection.detection.game_state_detector import GameStateDetector
    OCR_DETECTOR_AVAILABLE = True
except ImportError:
    try:
        from detection.game_state_detector import GameStateDetector
        OCR_DETECTOR_AVAILABLE = True
    except ImportError:
        try:
            from game_state_detector import GameStateDetector
            OCR_DETECTOR_AVAILABLE = True
        except ImportError:
            OCR_DETECTOR_AVAILABLE = False

# 游戏状态检测器（YOLO）
try:
    from data_collection.detection.yolo_state_detector import YoloGameStateDetector
    YOLO_DETECTOR_AVAILABLE = True
except ImportError:
    try:
        from detection.yolo_state_detector import YoloGameStateDetector
        YOLO_DETECTOR_AVAILABLE = True
    except ImportError:
        YOLO_DETECTOR_AVAILABLE = False

# 综合检测器可用性
STATE_DETECTOR_AVAILABLE = OCR_DETECTOR_AVAILABLE or YOLO_DETECTOR_AVAILABLE

# 录制状态覆盖层
try:
    from data_collection.recording_overlay import RecordingOverlay
    OVERLAY_AVAILABLE = True
except ImportError:
    try:
        from recording_overlay import RecordingOverlay
        OVERLAY_AVAILABLE = True
    except ImportError:
        logger.warning("录制状态覆盖层不可用")
        OVERLAY_AVAILABLE = False

# 录制事件系统
try:
    from data_collection.recording_events import EventDispatcher, RecordingEvent, RecordingEventType, RecordingStats
    EVENTS_AVAILABLE = True
except ImportError:
    try:
        from recording_events import EventDispatcher, RecordingEvent, RecordingEventType, RecordingStats
        EVENTS_AVAILABLE = True
    except ImportError:
        logger.warning("录制事件系统不可用")
        EVENTS_AVAILABLE = False
        EventDispatcher = None
        RecordingEvent = None
        RecordingEventType = None
        RecordingStats = None

# 全局键盘监听器
try:
    from data_collection.global_hotkey import GlobalKeyboardListener
    GLOBAL_HOOK_AVAILABLE = True
except ImportError:
    try:
        from global_hotkey import GlobalKeyboardListener
        GLOBAL_HOOK_AVAILABLE = True
    except ImportError:
        logger.warning("全局键盘钩子不可用，将使用 pynput（仅限焦点窗口）")
        GLOBAL_HOOK_AVAILABLE = False


class GameplayRecorder:
    """Record gameplay footage and input data"""
    
    def __init__(
        self,
        output_dir: str = "data/sessions",
        fps: int = 30,
        capture_mode: str = "fullscreen",
        process_name: str = None,
        window_title: str = None,
        save_format: str = "frames",
        frame_step: int = 1,
        screen_width: int = 1920,
        screen_height: int = 1080,
        target_width: int = 960,
        target_height: int = 540,
        use_global_hook: bool = True,
        auto_mode: bool = False,
        detector_type: str = 'auto',
        yolo_model_path: str = None
    ):
        """
        Initialize recorder
        
        Args:
            output_dir: Output directory for sessions
            fps: Recording FPS
            capture_mode: 'window' or 'fullscreen'
            process_name: Process name for window capture (e.g., 'WorldOfTanks.exe')
            window_title: Window title for window capture (e.g., 'World of Tanks')
            save_format: 'frames' (save PNG frames), 'video' (encode video), 'both'
            frame_step: Save every N frames (1 = save all, 5 = save every 5th frame)
        """
        self.output_dir_ = Path(output_dir)
        self.output_dir_.mkdir(parents=True, exist_ok=True)

        self.fps_ = fps
        self.frame_interval_ = 1.0 / fps
        self.capture_mode_ = capture_mode
        self.save_format_ = save_format
        self.frame_step_ = frame_step
        self.screen_width_ = screen_width
        self.screen_height_ = screen_height
        self.target_width_ = target_width
        self.target_height_ = target_height
        self.use_global_hook_ = use_global_hook and GLOBAL_HOOK_AVAILABLE
        self.auto_mode_ = auto_mode
        self.detector_type_ = detector_type
        
        # 初始化事件系统
        self.event_dispatcher_ = EventDispatcher() if EVENTS_AVAILABLE else None
        self.stats_ = RecordingStats() if EVENTS_AVAILABLE else None
        
        # 初始化游戏状态检测器（如果启用自动模式）
        self.state_detector_ = None
        if self.auto_mode_ and STATE_DETECTOR_AVAILABLE:
            try:
                # 优先使用YOLO检测器（如果可用），否则使用OCR
                use_yolo = (self.detector_type_ == 'yolo' or 
                          (self.detector_type_ == 'auto' and YOLO_DETECTOR_AVAILABLE))
                
                if use_yolo and YOLO_DETECTOR_AVAILABLE:
                    # 使用YOLO检测器（支持UI特征检测，可选YOLO模型）
                    self.state_detector_ = YoloGameStateDetector(
                        screen_width=screen_width,
                        screen_height=screen_height,
                        yolo_model_path=yolo_model_path,  # 如果提供则加载YOLO模型
                        use_yolo=(yolo_model_path is not None),  # 如果提供模型则使用
                        use_ui_features=True,
                        confidence_threshold=0.5
                    )
                    if yolo_model_path:
                        logger.info("✓ YOLO游戏状态检测器初始化成功（YOLO模型 + UI特征检测）")
                    else:
                        logger.info("✓ YOLO游戏状态检测器初始化成功（UI特征检测模式）")
                elif OCR_DETECTOR_AVAILABLE:
                    # 使用OCR检测器（向后兼容）
                    self.state_detector_ = GameStateDetector(
                        screen_width=screen_width,
                        screen_height=screen_height
                    )
                    logger.info("✓ OCR游戏状态检测器初始化成功")
                else:
                    raise RuntimeError("没有可用的检测器")
                    
            except Exception as e:
                logger.warning(f"游戏状态检测器初始化失败: {e}")
                logger.warning("将退回到手动模式（F9/F10）")
                self.auto_mode_ = False
        elif self.auto_mode_ and not STATE_DETECTOR_AVAILABLE:
            logger.warning("游戏状态检测器不可用，将退回到手动模式（F9/F10）")
            self.auto_mode_ = False
        
        # Recording state
        self.recording_ = False
        self.saved_ = False  # 是否已保存
        self.session_dir_ = None  # 当前会话目录
        self.frames_dir_ = None  # 帧保存目录
        self.frames_ = []  # 仅当 save_format 包含 'video' 时使用
        self.actions_ = []
        self.frame_numbers_ = []  # 记录每个 action 对应的帧号
        
        # Input tracking
        self.current_keys_ = set()
        self.mouse_position_ = (0, 0)
        
        # Input listeners
        self.keyboard_listener_ = None
        self.mouse_listener_ = None
        self.global_keyboard_ = None  # 全局键盘钩子
        
        # Capture objects
        self.window_capture_ = None
        self.screen_capture_ = None
        self.sct_ = None
        self.async_saver_ = None  # 异步帧保存器
        
        # 录制状态覆盖层（通过事件系统自动更新）
        self.overlay_ = None
        if OVERLAY_AVAILABLE:
            try:
                self.overlay_ = RecordingOverlay(
                    screen_width=screen_width,
                    screen_height=screen_height,
                    event_dispatcher=self.event_dispatcher_  # 传递事件分发器，自动订阅
                )
                logger.info("✓ 录制状态覆盖层已初始化")
            except Exception as e:
                logger.warning(f"录制状态覆盖层初始化失败: {e}")
                self.overlay_ = None
        
        # Initialize screen capture
        logger.info("初始化屏幕捕获模块...")
        logger.info(f"  - 捕获模式: {capture_mode}")
        
        if capture_mode == "window":
            # 窗口捕获模式
            if not WINDOW_CAPTURE_AVAILABLE:
                logger.error("✗ 窗口捕获不可用，请安装: pip install pywin32 psutil")
                logger.info("  降级到全屏捕获模式")
                capture_mode = "fullscreen"
            else:
                try:
                    if not process_name and not window_title:
                        logger.warning("未指定进程名或窗口标题，使用默认值")
                        process_name = "WorldOfTanks.exe"
                    
                    if process_name:
                        logger.info(f"  - 通过进程名查找: {process_name}")
                        self.window_capture_ = WindowCapture(
                            process_name=process_name,
                            partial_match=True
                        )
                    else:
                        logger.info(f"  - 通过窗口标题查找: {window_title}")
                        self.window_capture_ = WindowCapture(
                            window_title=window_title,
                            partial_match=True
                        )
                    
                    if self.window_capture_.IsValid():
                        logger.info("✓ 窗口捕获模块初始化成功")
                        self.capture_mode_ = "window"
                    else:
                        logger.error("✗ 未找到目标窗口，降级到全屏模式")
                        self.window_capture_ = None
                        capture_mode = "fullscreen"
                        
                except Exception as e:
                    logger.error(f"✗ 窗口捕获初始化失败: {e}")
                    logger.info("  降级到全屏捕获模式")
                    self.window_capture_ = None
                    capture_mode = "fullscreen"
        
        if capture_mode == "fullscreen":
            # 全屏捕获模式 - 使用 mss
            try:
                self.sct_ = mss.mss()
                # 自动获取主显示器分辨率
                monitor_info = self.sct_.monitors[1]  # 主显示器
                actual_width = monitor_info['width']
                actual_height = monitor_info['height']
                
                # 更新实际分辨率
                self.screen_width_ = actual_width
                self.screen_height_ = actual_height
                
                self.monitor_ = {"top": 0, "left": 0, "width": actual_width, "height": actual_height}
                logger.info(f"✓ 屏幕捕获模块初始化成功 (mss, 自动检测分辨率: {actual_width}x{actual_height})")
                self.capture_mode_ = "fullscreen_mss"
            except Exception as e:
                logger.error(f"✗ mss 初始化失败: {e}")
                logger.error("请安装 mss: pip install mss")
                raise
        
        # Setup input listeners
        logger.info("初始化输入监听器...")
        try:
            self.setupInputListeners()
            logger.info("✓ 输入监听器初始化成功")
        except Exception as e:
            logger.error(f"✗ 输入监听器初始化失败: {e}")
            logger.error("请安装 pynput: pip install pynput")
            raise
        
    def setupInputListeners(self):
        """Setup keyboard and mouse listeners"""
        
        # 键盘监听器：优先使用全局钩子
        if self.use_global_hook_:
            logger.info("  - 键盘监听: 使用全局钩子 (Windows Hook API)")
            
            def on_key_press(key_name: str):
                if key_name not in self.current_keys_:
                    self.current_keys_.add(key_name)
                    logger.debug(f"全局钩子 - 按下: {key_name}")
                
                # F10 停止录制
                if key_name == 'f10':
                    logger.info("检测到 F10 键，准备停止录制...")
                    self.recording_ = False
            
            def on_key_release(key_name: str):
                if key_name in self.current_keys_:
                    self.current_keys_.discard(key_name)
                    logger.debug(f"全局钩子 - 释放: {key_name}")
            
            self.global_keyboard_ = GlobalKeyboardListener(
                on_press=on_key_press,
                on_release=on_key_release
            )
            self.global_keyboard_.Start()
        else:
            logger.info("  - 键盘监听: 使用 pynput (仅焦点窗口)")
            self.keyboard_listener_ = keyboard.Listener(
                on_press=self.onKeyPress,
                on_release=self.onKeyRelease
            )
            self.keyboard_listener_.start()
        
        # 鼠标监听器：使用 pynput
        self.mouse_listener_ = mouse.Listener(
            on_move=self.onMouseMove,
            on_click=self.onMouseClick
        )
        self.mouse_listener_.start()
        
    def onKeyPress(self, key):
        """Keyboard press callback"""
        try:
            key_str = key.char
            self.current_keys_.add(key_str)
            logger.debug(f"按键按下: {key_str}")
        except AttributeError:
            # Special key
            key_str = str(key).replace('Key.', '')  # 简化特殊键名称
            self.current_keys_.add(key_str)
            logger.debug(f"特殊键按下: {key_str}")
            
    def onKeyRelease(self, key):
        """Keyboard release callback"""
        try:
            key_str = key.char
            self.current_keys_.discard(key_str)
            logger.debug(f"按键释放: {key_str}")
        except AttributeError:
            # Special key
            key_str = str(key).replace('Key.', '')  # 简化特殊键名称
            self.current_keys_.discard(key_str)
            logger.debug(f"特殊键释放: {key_str}")
            
        # Stop recording on F10 - 只设置标志，让主循环处理
        if key == keyboard.Key.f10:
            logger.info("检测到 F10 键，准备停止录制...")
            self.recording_ = False
            
    def onMouseMove(self, x, y):
        """Mouse move callback"""
        self.mouse_position_ = (x, y)
        
    def onMouseClick(self, x, y, button, pressed):
        """Mouse click callback"""
        if pressed:
            self.current_keys_.add(f"mouse_{button.name}")
        else:
            self.current_keys_.discard(f"mouse_{button.name}")
            
    def captureScreen(self) -> np.ndarray:
        """Capture current screen"""
        try:
            if self.capture_mode_ == "window":
                # 窗口捕获模式
                logger.debug("Using window capture")
                if not self.window_capture_ or not self.window_capture_.IsValid():
                    # 尝试重新查找窗口
                    if self.window_capture_:
                        self.window_capture_.findWindow()
                    
                    if not self.window_capture_ or not self.window_capture_.IsValid():
                        raise RuntimeError("窗口已关闭或无效")
                
                frame = self.window_capture_.Capture()
                if frame is None:
                    raise RuntimeError("窗口捕获返回 None")
                    
            elif self.capture_mode_ == "fullscreen_mss":
                # Python fallback 全屏捕获
                logger.debug("Using Python fallback (mss)")
                screenshot = self.sct_.grab(self.monitor_)
                # mss 返回 BGRA 格式，去除 alpha 通道得到 BGR（cv2 默认格式）
                frame = np.array(screenshot)[:, :, :3]
            else:
                raise ValueError(f"Unknown capture mode: {self.capture_mode_}")
            
            logger.debug(f"Captured frame: shape={frame.shape}, dtype={frame.dtype}")
            return frame
        except Exception as e:
            logger.error(f"Screen capture failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def ResizeFrame(self, frame: np.ndarray) -> np.ndarray:
        """
        将帧缩放到目标分辨率
        
        Args:
            frame: 原始帧（BGR 格式）
            
        Returns:
            缩放后的帧（BGR 格式）
        """
        if frame is None:
            return None
        
        frame_height, frame_width = frame.shape[:2]
        
        # 如果已经是目标分辨率，直接返回
        if frame_width == self.target_width_ and frame_height == self.target_height_:
            return frame
        
        # 使用 cv2.INTER_AREA 进行降采样（适合缩小分辨率）
        resized_frame = cv2.resize(
            frame,
            (self.target_width_, self.target_height_),
            interpolation=cv2.INTER_AREA
        )
        
        return resized_frame
        
    def getCurrentAction(self) -> dict:
        """Get current input action"""
        return {
            "keys": list(self.current_keys_),
            "mouse_pos": self.mouse_position_
        }
        
    def prepareRecording(self):
        """准备录制（创建目录等）"""
        self.recording_ = True
        self.saved_ = False  # 重置保存标志
        self.frames_ = []  # 仅视频模式使用
        self.actions_ = []
        self.frame_numbers_ = []
        self.recording_start_time_ = time.time()  # 记录录制开始时间
        
        # 更新统计信息
        if self.stats_:
            self.stats_.Start()
            self.stats_.start_time_ = self.recording_start_time_
        
        # 发送录制开始事件
        if self.event_dispatcher_ and RecordingEventType:
            event = RecordingEvent(RecordingEventType.RECORDING_STARTED)
            self.event_dispatcher_.Emit(event)
        
        # 启动覆盖层
        if self.overlay_:
            self.overlay_.Start()
        
        # 创建 session 目录
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.session_dir_ = self.output_dir_ / f"session_{timestamp}"
        self.session_dir_.mkdir(parents=True, exist_ok=True)
        
        # 如果保存帧，创建 frames 目录并启动异步保存器
        if self.save_format_ in ["frames", "both"]:
            self.frames_dir_ = self.session_dir_ / "frames"
            self.frames_dir_.mkdir(exist_ok=True)
            
            # 启动异步帧保存器（如果可用）
            if ASYNC_SAVER_AVAILABLE:
                self.async_saver_ = AsyncFrameSaver(
                    frames_dir=self.frames_dir_,
                    png_compression=3,  # PNG 压缩级别 0-9，默认3（平衡速度和压缩比）
                    queue_size=60  # 队列可容纳 2 秒的帧（30 FPS）
                )
                self.async_saver_.Start()
        
        logger.info("=" * 80)
        logger.info("🎬 录制开始！")
        logger.info("  - 按 F10 键停止录制")
        logger.info(f"  - 目标 FPS: {self.fps_}")
        logger.info(f"  - 捕获模式: {self.capture_mode_}")
        logger.info(f"  - 保存格式: {self.save_format_}")
        logger.info(f"  - 帧采样: 每 {self.frame_step_} 帧保存一次")
        logger.info(f"  - 会话目录: {self.session_dir_}")
        if self.capture_mode_ == "window":
            logger.info(f"  - 窗口大小: {self.window_capture_.GetWidth()}x{self.window_capture_.GetHeight()}")
        logger.info("=" * 80)
    
    def startRecording(self):
        """Start recording"""
        self.prepareRecording()
        
        start_time = time.time()
        frame_count = 0
        last_log_time = start_time
        
        try:
            while self.recording_:
                frame_start = time.time()
                
                try:
                    # Capture frame
                    frame = self.captureScreen()
                    action = self.getCurrentAction()
                    timestamp = time.time() - start_time
                    
                    # 调试：保存第一帧为图片
                    if frame_count == 0:
                        debug_path = self.session_dir_ / "debug_first_frame.png"
                        # mss 捕获的是 BGR 格式，直接保存（cv2.imwrite 期望 BGR）
                        cv2.imwrite(str(debug_path), frame)
                        logger.info(f"[调试] 第一帧已保存到: {debug_path}")
                        logger.info(f"[调试] 帧信息: shape={frame.shape}, mean={frame.mean():.2f}, max={frame.max()}, min={frame.min()}")
                    
                    # 实时保存帧（如果启用）
                    if self.save_format_ in ["frames", "both"]:
                        if frame_count % self.frame_step_ == 0:
                            # 缩放帧到目标分辨率
                            resized_frame = self.ResizeFrame(frame)
                            # 使用异步保存器（如果可用）
                            if self.async_saver_:
                                self.async_saver_.SaveFrame(resized_frame, frame_count)
                            else:
                                # 回退到同步保存
                                if self.frames_dir_ is not None:
                                    if self.frames_dir_ is not None:
                                        frame_path = self.frames_dir_ / f"frame_{frame_count:06d}.png"
                                        # resized_frame 已经是 BGR 格式，直接保存为 PNG
                                        cv2.imwrite(str(frame_path), resized_frame, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                                    else:
                                        logger.warning(f"frames_dir_ 未初始化，无法保存帧 {frame_count}")
                                else:
                                    logger.warning(f"frames_dir_ 未初始化，无法保存帧 {frame_count}")
                    
                    # 仅在视频模式下存储到内存（也需要缩放）
                    if self.save_format_ in ["video", "both"]:
                        resized_frame = self.ResizeFrame(frame)
                        self.frames_.append(resized_frame)
                    
                    # 记录操作和对应的帧号
                    self.actions_.append(action)
                    self.frame_numbers_.append(frame_count)
                    
                    frame_count += 1
                    
                    # 更新统计信息并发送事件（每秒一次）
                    current_time_frame = time.time()
                    if current_time_frame - last_log_time >= 1.0:
                        if self.stats_:
                            self.stats_.UpdateFrameCount(frame_count)
                            self.stats_.UpdateDuration(timestamp)
                        
                        # 发送帧捕获事件（覆盖层会自动更新）
                        if self.event_dispatcher_ and RecordingEventType:
                            event = RecordingEvent(
                                RecordingEventType.FRAME_CAPTURED,
                                {'frame_count': frame_count, 'duration_seconds': timestamp}
                            )
                            self.event_dispatcher_.Emit(event)
                    
                    # Log progress every 5 seconds
                    if current_time_frame - last_log_time >= 5.0:
                        actual_fps = frame_count / timestamp if timestamp > 0 else 0
                        log_msg = f"录制中... 帧数: {frame_count}, 时长: {timestamp:.1f}s, FPS: {actual_fps:.1f}"
                        
                        # 显示异步保存队列状态
                        if self.async_saver_:
                            queue_size = self.async_saver_.GetQueueSize()
                            log_msg += f", 队列: {queue_size}/60"
                        
                        logger.info(log_msg)
                        last_log_time = time.time()
                        
                except Exception as e:
                    logger.error(f"捕获帧失败: {e}")
                    # Continue trying instead of crashing
                    
                # Maintain FPS
                elapsed = time.time() - frame_start
                sleep_time = max(0, self.frame_interval_ - elapsed)
                time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            logger.info("录制被用户中断")
        except Exception as e:
            logger.error(f"录制过程出错: {e}")
            import traceback
            traceback.print_exc()
            
        self.stopRecording()
        
    def stopRecording(self):
        """Stop recording and save data"""
        # 使用 saved_ 标志避免重复保存
        if self.saved_:
            logger.debug("数据已经保存过，跳过")
            return
            
        self.recording_ = False  # 确保停止录制
        self.saved_ = True  # 标记为已保存（即使失败也不重试）
        
        # 更新统计信息
        final_duration = (time.time() - self.recording_start_time_) if hasattr(self, 'recording_start_time_') else 0.0
        if self.stats_:
            self.stats_.Stop()
            self.stats_.UpdateFrameCount(len(self.frame_numbers_))
            self.stats_.UpdateDuration(final_duration)
        
        # 发送录制停止事件（覆盖层会自动更新）
        if self.event_dispatcher_ and RecordingEventType:
            event = RecordingEvent(
                RecordingEventType.RECORDING_STOPPED,
                {'frame_count': len(self.frame_numbers_), 'duration_seconds': final_duration}
            )
            self.event_dispatcher_.Emit(event)
        
        # 停止异步帧保存器（等待队列清空）
        if self.async_saver_:
            logger.info("等待异步保存队列清空...")
            self.async_saver_.Stop()
            stats = self.async_saver_.GetStats()
            logger.info(f"异步保存统计: 已保存={stats['saved']}, 丢弃={stats['dropped']}")
        
        logger.info("Stopping recording...")
        logger.info(f"录制完成，共 {len(self.frame_numbers_)} 帧")
        
        # Check if any data was recorded
        if not self.frame_numbers_ or not self.actions_:
            logger.error("=" * 80)
            logger.error("❌ 录制失败：没有捕获到任何帧！")
            logger.error("=" * 80)
            logger.error("可能的原因：")
            logger.error("  1. 录制时间太短（立即按了 F10）")
            logger.error("  2. 屏幕捕获模块初始化失败")
            logger.error("  3. C++ 绑定未编译或加载失败")
            logger.error("  4. Python fallback (mss) 未安装或无法访问屏幕")
            logger.error("=" * 80)
            logger.error(f"调试信息：")
            logger.error(f"  - 捕获的帧数: {len(self.frame_numbers_)}")
            logger.error(f"  - 操作记录数: {len(self.actions_)}")
            logger.error(f"  - C++ 可用: {CPP_AVAILABLE}")
            logger.error("=" * 80)
            return
        
        try:
            # 保存元数据
            logger.info("保存会话元数据...")
            total_frames = len(self.frame_numbers_)
            duration = (self.frame_numbers_[-1] / self.fps_) if total_frames > 0 else 0
            
            # 获取分辨率信息（使用缩放后的分辨率）
            if self.save_format_ in ["frames", "both"]:
                # 从已保存的第一帧读取分辨率
                first_frame_path = self.frames_dir_ / "frame_000000.png"
                if first_frame_path.exists():
                    first_frame = cv2.imread(str(first_frame_path))
                    resolution = [first_frame.shape[1], first_frame.shape[0]]
                else:
                    resolution = [self.target_width_, self.target_height_]  # 使用目标分辨率
            elif self.save_format_ == "video" and self.frames_:
                resolution = [self.frames_[0].shape[1], self.frames_[0].shape[0]]
            else:
                resolution = [self.target_width_, self.target_height_]  # 使用目标分辨率
            
            meta_data = {
                "session_id": self.session_dir_.name,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "fps": self.fps_,
                "total_frames": total_frames,
                "duration_seconds": duration,
                "resolution": resolution,
                "capture_mode": self.capture_mode_,
                "save_format": self.save_format_,
                "frame_step": self.frame_step_,
                "frames_saved": total_frames // self.frame_step_ if self.save_format_ in ["frames", "both"] else 0
            }
            
            meta_path = self.session_dir_ / "meta.json"
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(meta_data, f, indent=2, ensure_ascii=False)
            logger.info(f"✓ 元数据已保存: {meta_path}")
            
            # 保存操作记录（关联到帧号）
            logger.info("保存操作记录...")
            actions_data = []
            for i, (frame_num, action) in enumerate(zip(self.frame_numbers_, self.actions_)):
                actions_data.append({
                    "frame": frame_num,
                    "timestamp": frame_num / self.fps_,
                    "keys": action["keys"],
                    "mouse_pos": action["mouse_pos"]
                })
            
            actions_path = self.session_dir_ / "actions.json"
            with open(actions_path, 'w', encoding='utf-8') as f:
                json.dump(actions_data, f, indent=2, ensure_ascii=False)
            logger.info(f"✓ 操作记录已保存: {actions_path} ({len(actions_data)} 条)")
            
            # 如果需要保存视频（可选）
            if self.save_format_ in ["video", "both"] and self.frames_:
                logger.info("正在编码视频...")
                self._saveVideo()
            
            # Summary
            logger.info("=" * 80)
            logger.info(f"✓ 录制成功保存到: {self.session_dir_}")
            logger.info(f"  - 总帧数: {total_frames}")
            logger.info(f"  - 时长: {duration:.2f}s")
            logger.info(f"  - 平均 FPS: {total_frames / duration:.2f}" if duration > 0 else "  - 平均 FPS: N/A")
            if self.save_format_ in ["frames", "both"]:
                saved_frames = total_frames // self.frame_step_
                logger.info(f"  - 已保存帧: {saved_frames} 张 PNG")
            if self.save_format_ in ["video", "both"]:
                logger.info(f"  - 视频: gameplay.avi")
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error("=" * 80)
            logger.error(f"❌ 保存录制数据时出错: {e}")
            logger.error("=" * 80)
            import traceback
            traceback.print_exc()
    
    def _saveVideo(self):
        """保存视频文件（仅在 save_format 包含 'video' 时调用）"""
        if not self.frames_:
            logger.warning("没有帧数据，跳过视频保存")
            return
        
        frame_height, frame_width = self.frames_[0].shape[:2]
        video_path = self.session_dir_ / "gameplay.avi"
        
        # 使用 XVID 编解码器（最兼容）
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(
            str(video_path),
            fourcc,
            self.fps_,
            (frame_width, frame_height)
        )
        
        if not out.isOpened():
            logger.error("无法打开视频写入器")
            return
        
        for i, frame in enumerate(self.frames_):
            if i % 100 == 0:
                logger.info(f"编码帧 {i}/{len(self.frames_)}...")
            # frame 已经是 BGR 格式，直接写入
            out.write(frame)
        
        out.release()
        
        if video_path.exists() and video_path.stat().st_size > 1024:
            logger.info(f"✓ 视频已保存: {video_path} ({video_path.stat().st_size:,} bytes)")
        else:
            logger.error("视频文件创建失败或为空")
        
    def runRecordingLoop(self):
        """运行录制循环 - 支持 F9 开始，F10 停止，可多次录制"""
        if self.auto_mode_:
            self.runAutoRecordingLoop()
            return
        
        logger.info("=" * 80)
        logger.info("🎮 录制器已就绪")
        logger.info("=" * 80)
        logger.info("")
        logger.info("⌨️  快捷键:")
        logger.info("  F9  - 开始录制")
        logger.info("  F10 - 停止录制")
        logger.info("  Ctrl+C - 退出程序")
        logger.info("")
        logger.info("等待按 F9 开始录制...")
        logger.info("=" * 80)
        
        # 添加热键状态
        self.waiting_start_ = True
        self.should_exit_ = False
        
        # 注册全局热键处理（F9 开始，F10 停止）
        def on_hotkey_press(key_name: str):
            if key_name == 'f9' and self.waiting_start_:
                logger.info("")
                logger.info("🔴 检测到 F9，准备开始录制...")
                self.waiting_start_ = False
                self.prepareRecording()
            elif key_name == 'f10' and self.recording_:
                logger.info("")
                logger.info("⏹️  检测到 F10，停止录制...")
                self.recording_ = False
        
        # 替换当前的热键处理
        if self.global_keyboard_:
            # 停止旧的监听器
            self.global_keyboard_.Stop()
            # 创建新的监听器
            def on_key_press_wrapper(key_name: str):
                if key_name not in self.current_keys_:
                    self.current_keys_.add(key_name)
                on_hotkey_press(key_name)
            
            def on_key_release_wrapper(key_name: str):
                if key_name in self.current_keys_:
                    self.current_keys_.discard(key_name)
            
            self.global_keyboard_ = GlobalKeyboardListener(
                on_press=on_key_press_wrapper,
                on_release=on_key_release_wrapper
            )
            self.global_keyboard_.Start()
        else:
            # 使用 pynput 的情况
            self.keyboard_listener_.stop()
            
            def on_press(key):
                try:
                    key_str = key.char
                    if key_str not in self.current_keys_:
                        self.current_keys_.add(key_str)
                except AttributeError:
                    key_str = str(key).replace('Key.', '')
                    if key_str not in self.current_keys_:
                        self.current_keys_.add(key_str)
                
                # 处理热键
                if key == keyboard.Key.f9 and self.waiting_start_:
                    logger.info("")
                    logger.info("🔴 检测到 F9，准备开始录制...")
                    self.waiting_start_ = False
                    self.prepareRecording()
                elif key == keyboard.Key.f10 and self.recording_:
                    logger.info("")
                    logger.info("⏹️  检测到 F10，停止录制...")
                    self.recording_ = False
            
            def on_release(key):
                try:
                    key_str = key.char
                    if key_str in self.current_keys_:
                        self.current_keys_.discard(key_str)
                except AttributeError:
                    key_str = str(key).replace('Key.', '')
                    if key_str in self.current_keys_:
                        self.current_keys_.discard(key_str)
            
            self.keyboard_listener_ = keyboard.Listener(
                on_press=on_press,
                on_release=on_release
            )
            self.keyboard_listener_.start()
        
        # 启动覆盖层（即使不在录制状态也显示等待状态）
        if self.overlay_:
            self.overlay_.Start()
            # 初始状态：通过直接调用设置（覆盖层还未完全启动时）
            self.overlay_.UpdateDisplay(is_recording=False, frame_count=0, duration_seconds=0.0)
        
        # 主循环
        session_count = 0
        try:
            while not self.should_exit_:
                # 等待开始录制（覆盖层状态通过事件系统自动更新，无需手动更新）
                while self.waiting_start_ and not self.should_exit_:
                    time.sleep(0.1)
                
                if self.should_exit_:
                    break
                
                session_count += 1
                logger.info(f"\n📹 开始第 {session_count} 次录制...")
                
                # 录制循环
                start_time = time.time()
                frame_count = 0
                last_log_time = start_time
                
                while self.recording_ and not self.should_exit_:
                    frame_start = time.time()
                    
                    try:
                        # Capture frame
                        frame = self.captureScreen()
                        action = self.getCurrentAction()
                        timestamp = time.time() - start_time
                        
                        # 调试：保存第一帧
                        if frame_count == 0:
                            debug_path = self.session_dir_ / "debug_first_frame.png"
                            # frame 已经是 BGR 格式，直接保存
                            cv2.imwrite(str(debug_path), frame)
                            logger.info(f"[调试] 第一帧已保存: {debug_path.name}")
                        
                        # 实时保存帧
                        if self.save_format_ in ["frames", "both"]:
                            if frame_count % self.frame_step_ == 0:
                                # 缩放帧到目标分辨率
                                resized_frame = self.ResizeFrame(frame)
                                if self.async_saver_:
                                    self.async_saver_.SaveFrame(resized_frame, frame_count)
                                else:
                                    if self.frames_dir_ is not None:
                                        frame_path = self.frames_dir_ / f"frame_{frame_count:06d}.png"
                                        # resized_frame 已经是 BGR 格式，直接保存为 PNG
                                        cv2.imwrite(str(frame_path), resized_frame, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                                    else:
                                        logger.warning(f"frames_dir_ 未初始化，无法保存帧 {frame_count}")
                        
                        # 视频模式下存储到内存（也需要缩放）
                        if self.save_format_ in ["video", "both"]:
                            resized_frame = self.ResizeFrame(frame)
                            self.frames_.append(resized_frame)
                        
                        # 记录操作
                        self.actions_.append(action)
                        self.frame_numbers_.append(frame_count)
                        
                        frame_count += 1
                        
                        # 定期日志
                        if time.time() - last_log_time >= 5.0:
                            actual_fps = frame_count / timestamp if timestamp > 0 else 0
                            log_msg = f"录制中... 帧数: {frame_count}, 时长: {timestamp:.1f}s, FPS: {actual_fps:.1f}"
                            
                            if self.async_saver_:
                                queue_size = self.async_saver_.GetQueueSize()
                                log_msg += f", 队列: {queue_size}/60"
                            
                            logger.info(log_msg)
                            last_log_time = time.time()
                            
                    except Exception as e:
                        logger.error(f"捕获帧失败: {e}")
                    
                    # 维持 FPS
                    elapsed = time.time() - frame_start
                    sleep_time = max(0, self.frame_interval_ - elapsed)
                    time.sleep(sleep_time)
                
                # 录制结束，保存数据
                if not self.should_exit_:
                    self.stopRecording()
                    
                    # 准备下一次录制
                    logger.info("")
                    logger.info("=" * 80)
                    logger.info("💡 可以再次按 F9 开始新的录制，或按 Ctrl+C 退出")
                    logger.info("=" * 80)
                    self.waiting_start_ = True
                
        except KeyboardInterrupt:
            logger.info("\n\n⚠️  录制被用户中断")
        except Exception as e:
            logger.error(f"\n\n❌ 录制过程出错: {e}")
            import traceback
            traceback.print_exc()
        finally:
            logger.info(f"\n📊 总共完成 {session_count} 次录制")
    
    def runAutoRecordingLoop(self):
        """运行自动录制循环 - 自动检测战斗开始和结束"""
        logger.info("=" * 80)
        logger.info("🎮 自动录制模式已启用")
        logger.info("=" * 80)
        logger.info("")
        logger.info("📋 自动检测规则:")
        logger.info("  - 检测区域: 屏幕中心靠上1/3区域")
        logger.info("  - 战斗开始时: 自动开始录制")
        logger.info("  - 胜利/被击败/被击毁时: 自动停止录制")
        logger.info("")
        logger.info("⌨️  快捷键:")
        logger.info("  F9  - 手动开始录制（覆盖自动检测）")
        logger.info("  F10 - 手动停止录制（覆盖自动检测）")
        logger.info("  Ctrl+C - 退出程序")
        logger.info("")
        logger.info("正在监控游戏状态...")
        logger.info("=" * 80)
        
        if not self.state_detector_:
            logger.error("❌ 状态检测器未初始化，无法使用自动模式")
            logger.info("退回到手动模式")
            self.auto_mode_ = False
            self.runRecordingLoop()
            return
        
        # 添加热键状态
        self.waiting_start_ = True
        self.should_exit_ = False
        
        # 注册全局热键处理（F9 手动开始，F10 手动停止）
        def on_hotkey_press(key_name: str):
            if key_name == 'f9' and self.waiting_start_:
                logger.info("")
                logger.info("🔴 检测到 F9，手动开始录制...")
                self.waiting_start_ = False
                self.prepareRecording()
            elif key_name == 'f10' and self.recording_:
                logger.info("")
                logger.info("⏹️  检测到 F10，手动停止录制...")
                self.recording_ = False
        
        # 替换当前的热键处理
        if self.global_keyboard_:
            # 停止旧的监听器
            self.global_keyboard_.Stop()
            # 创建新的监听器
            def on_key_press_wrapper(key_name: str):
                if key_name not in self.current_keys_:
                    self.current_keys_.add(key_name)
                on_hotkey_press(key_name)
            
            def on_key_release_wrapper(key_name: str):
                if key_name in self.current_keys_:
                    self.current_keys_.discard(key_name)
            
            self.global_keyboard_ = GlobalKeyboardListener(
                on_press=on_key_press_wrapper,
                on_release=on_key_release_wrapper
            )
            self.global_keyboard_.Start()
        else:
            # 使用 pynput 的情况
            self.keyboard_listener_.stop()
            
            def on_press(key):
                try:
                    key_str = key.char
                    if key_str not in self.current_keys_:
                        self.current_keys_.add(key_str)
                except AttributeError:
                    key_str = str(key).replace('Key.', '')
                    if key_str not in self.current_keys_:
                        self.current_keys_.add(key_str)
                
                # 处理热键
                if key == keyboard.Key.f9 and self.waiting_start_:
                    logger.info("")
                    logger.info("🔴 检测到 F9，手动开始录制...")
                    self.waiting_start_ = False
                    self.prepareRecording()
                elif key == keyboard.Key.f10 and self.recording_:
                    logger.info("")
                    logger.info("⏹️  检测到 F10，手动停止录制...")
                    self.recording_ = False
            
            def on_release(key):
                try:
                    key_str = key.char
                    if key_str in self.current_keys_:
                        self.current_keys_.discard(key_str)
                except AttributeError:
                    key_str = str(key).replace('Key.', '')
                    if key_str in self.current_keys_:
                        self.current_keys_.discard(key_str)
            
            self.keyboard_listener_ = keyboard.Listener(
                on_press=on_press,
                on_release=on_release
            )
            self.keyboard_listener_.start()
        
        # 主循环
        session_count = 0
        state_check_interval = 1.0  # 每1秒检测一次状态（避免频繁检测影响性能）
        last_state_check = 0
        
        try:
            while not self.should_exit_:
                current_time = time.time()
                
                # 状态检测（仅在未录制时检测"战斗开始"，录制时检测结束状态）
                if current_time - last_state_check >= state_check_interval:
                    try:
                        # 快速截屏用于状态检测（不需要保存）
                        frame = self.captureScreen()
                        # 传入 frame_number=None 因为还未开始录制，使用优化检测
                        detected_state = self.state_detector_.DetectState(frame, frame_number=None)
                        
                        if detected_state:
                            if detected_state == 'battle_start' and self.waiting_start_:
                                logger.info("")
                                logger.info("🎯 检测到：战斗开始！自动开始录制...")
                                self.waiting_start_ = False
                                self.prepareRecording()
                                session_count += 1
                                logger.info(f"📹 开始第 {session_count} 次录制...")
                            elif detected_state in ['victory', 'defeat', 'destroyed'] and self.recording_:
                                logger.info("")
                                logger.info(f"🎯 检测到：{detected_state}！自动停止录制...")
                                self.recording_ = False
                    except Exception as e:
                        logger.debug(f"状态检测失败: {e}")
                    
                    last_state_check = current_time
                
                # 等待开始录制
                while self.waiting_start_ and not self.should_exit_:
                    time.sleep(0.1)
                
                if self.should_exit_:
                    break
                
                # 录制循环
                start_time = time.time()
                frame_count = 0
                last_log_time = start_time
                last_state_check_in_recording = start_time
                
                while self.recording_ and not self.should_exit_:
                    frame_start = time.time()
                    
                    try:
                        # Capture frame
                        frame = self.captureScreen()
                        action = self.getCurrentAction()
                        timestamp = time.time() - start_time
                        
                        # 调试：保存第一帧
                        if frame_count == 0:
                            debug_path = self.session_dir_ / "debug_first_frame.png"
                            cv2.imwrite(str(debug_path), frame)
                            logger.info(f"[调试] 第一帧已保存: {debug_path.name}")
                        
                        # 录制过程中定期检测结束状态
                        current_time_in_recording = time.time()
                        if current_time_in_recording - last_state_check_in_recording >= state_check_interval:
                            try:
                                # 传入当前帧号以启用采样检测优化
                                detected_state = self.state_detector_.DetectState(frame, frame_number=frame_count)
                                if detected_state in ['victory', 'defeat', 'destroyed']:
                                    logger.info("")
                                    logger.info(f"🎯 检测到：{detected_state}！自动停止录制...")
                                    self.recording_ = False
                                    break
                            except Exception as e:
                                logger.debug(f"状态检测失败: {e}")
                            last_state_check_in_recording = current_time_in_recording
                        
                        # 实时保存帧
                        if self.save_format_ in ["frames", "both"]:
                            if frame_count % self.frame_step_ == 0:
                                # 缩放帧到目标分辨率
                                resized_frame = self.ResizeFrame(frame)
                                if self.async_saver_:
                                    self.async_saver_.SaveFrame(resized_frame, frame_count)
                                else:
                                    frame_path = self.frames_dir_ / f"frame_{frame_count:06d}.png"
                                    # resized_frame 已经是 BGR 格式，直接保存为 PNG
                                    cv2.imwrite(str(frame_path), resized_frame, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                        
                        # 视频模式下存储到内存（也需要缩放）
                        if self.save_format_ in ["video", "both"]:
                            resized_frame = self.ResizeFrame(frame)
                            self.frames_.append(resized_frame)
                        
                        # 记录操作
                        self.actions_.append(action)
                        self.frame_numbers_.append(frame_count)
                        
                        frame_count += 1
                        
                        # 更新统计信息并发送事件（每秒一次，覆盖层会自动更新）
                        current_time_check = time.time()
                        if current_time_check - last_log_time >= 1.0:
                            if self.stats_:
                                self.stats_.UpdateFrameCount(frame_count)
                                self.stats_.UpdateDuration(timestamp)
                            
                            # 发送帧捕获事件（覆盖层会自动更新）
                            if self.event_dispatcher_ and RecordingEventType:
                                event = RecordingEvent(
                                    RecordingEventType.FRAME_CAPTURED,
                                    {'frame_count': frame_count, 'duration_seconds': timestamp}
                                )
                                self.event_dispatcher_.Emit(event)
                        
                        # 定期日志
                        if current_time_check - last_log_time >= 5.0:
                            actual_fps = frame_count / timestamp if timestamp > 0 else 0
                            log_msg = f"录制中... 帧数: {frame_count}, 时长: {timestamp:.1f}s, FPS: {actual_fps:.1f}"
                            
                            if self.async_saver_:
                                queue_size = self.async_saver_.GetQueueSize()
                                log_msg += f", 队列: {queue_size}/60"
                            
                            logger.info(log_msg)
                            last_log_time = current_time_check
                            
                    except Exception as e:
                        logger.error(f"捕获帧失败: {e}")
                    
                    # 维持 FPS
                    elapsed = time.time() - frame_start
                    sleep_time = max(0, self.frame_interval_ - elapsed)
                    time.sleep(sleep_time)
                
                # 录制结束，保存数据
                if not self.should_exit_:
                    self.stopRecording()
                    
                    # 准备下一次录制
                    logger.info("")
                    logger.info("=" * 80)
                    logger.info("💡 等待下一场战斗开始，或按 F9 手动开始，按 Ctrl+C 退出")
                    logger.info("=" * 80)
                    self.waiting_start_ = True
                
        except KeyboardInterrupt:
            logger.info("\n\n⚠️  录制被用户中断")
        except Exception as e:
            logger.error(f"\n\n❌ 录制过程出错: {e}")
            import traceback
            traceback.print_exc()
        finally:
            logger.info(f"\n📊 总共完成 {session_count} 次录制")
    
    def cleanup(self):
        """Clean up resources"""
        # 关闭覆盖层
        if self.overlay_:
            self.overlay_.Stop()
        
        if self.keyboard_listener_:
            self.keyboard_listener_.stop()
        if self.mouse_listener_:
            self.mouse_listener_.stop()
        if self.global_keyboard_:
            self.global_keyboard_.Stop()


def run_with_config(config_dict=None):
    """使用配置字典运行录制程序"""
    if config_dict is None:
        # 如果没有提供配置，则使用默认的main()函数
        return main()
    
    # 检查管理员权限
    try:
        import ctypes
        is_admin = ctypes.windll.shell32.IsUserAnAdmin()
        if not is_admin:
            logger.warning("=" * 80)
            logger.warning("⚠️  警告：未以管理员权限运行")
            logger.warning("=" * 80)
            logger.warning("在 Windows 上，pynput 需要管理员权限才能监听全局键盘输入")
            logger.warning("这可能导致：")
            logger.warning("  - 键盘按键无法被记录")
            logger.warning("  - actions.json 中所有 keys 字段为空")
            logger.warning("")
            logger.warning("建议：")
            logger.warning("  1. 右键点击 start_recording.bat")
            logger.warning("  2. 选择'以管理员身份运行'")
            logger.warning("")
            logger.warning("或者运行测试脚本验证：")
            logger.warning("  python test_keyboard_listener.py")
            logger.warning("=" * 80)
            logger.warning("")
    except:
        pass
    
    # 从配置字典中提取参数
    capture_config = config_dict.get('capture', {})
    fps = capture_config.get('fps', 30)
    mode = capture_config.get('mode', 'fullscreen')
    fullscreen_config = capture_config.get('fullscreen', {})
    width = fullscreen_config.get('width', 1920)
    height = fullscreen_config.get('height', 1080)
    # 目标分辨率（用于保存帧）
    target_config = config_dict.get('target_resolution', {})
    target_width = target_config.get('width', 960)
    target_height = target_config.get('height', 540)
    auto_mode = config_dict.get('auto_mode', False)  # 自动模式
    
    logger.info("=" * 80)
    logger.info("🎮 World of Tanks - 游戏录制工具")
    logger.info("=" * 80)
    logger.info("")
    logger.info("📋 当前配置:")
    logger.info(f"  录制 FPS: {fps}")
    logger.info(f"  捕获模式: {mode}")
    logger.info(f"  捕获分辨率: {width}x{height}")
    logger.info(f"  保存分辨率: {target_width}x{target_height}")
    logger.info(f"  自动模式: {auto_mode}")
    logger.info("")
    
    # 创建录制器
    try:
        logger.info("\n正在初始化录制器...")
        recorder = GameplayRecorder(
            output_dir="data/recordings",
            fps=fps,
            capture_mode=mode,
            save_format="frames",
            frame_step=2,
            screen_width=width,
            screen_height=height,
            target_width=target_width,
            target_height=target_height,
            use_global_hook=False,  # 暂时禁用全局钩子
            auto_mode=auto_mode
        )
        logger.info("✓ 录制器初始化成功\n")
    except Exception as e:
        logger.error(f"\n✗ 录制器初始化失败: {e}")
        return
    
    try:
        # 使用新的录制循环（支持 F9/F10 热键，可多次录制）
        recorder.runRecordingLoop()
    except Exception as e:
        logger.error(f"录制过程出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        recorder.cleanup()
        logger.info("\n录制器已清理")


def main():
    """Main function"""
    
    # 检查管理员权限
    try:
        import ctypes
        is_admin = ctypes.windll.shell32.IsUserAnAdmin()
        if not is_admin:
            logger.warning("=" * 80)
            logger.warning("⚠️  警告：未以管理员权限运行")
            logger.warning("=" * 80)
            logger.warning("在 Windows 上，pynput 需要管理员权限才能监听全局键盘输入")
            logger.warning("这可能导致：")
            logger.warning("  - 键盘按键无法被记录")
            logger.warning("  - actions.json 中所有 keys 字段为空")
            logger.warning("")
            logger.warning("建议：")
            logger.warning("  1. 右键点击 start_recording.bat")
            logger.warning("  2. 选择'以管理员身份运行'")
            logger.warning("")
            logger.warning("或者运行测试脚本验证：")
            logger.warning("  python test_keyboard_listener.py")
            logger.warning("=" * 80)
            logger.warning("")
    except:
        pass
    
    # 读取配置文件（兼容 PyInstaller 打包）
    base_path = get_base_path()
    config_path = base_path / "configs" / "client_config.yaml"
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"已加载配置文件: {config_path}")
        
        # 调试：显示关键配置值
        logger.debug(f"配置内容 - FPS: {config['capture'].get('fps')}")
        logger.debug(f"配置内容 - 分辨率: {config['capture']['fullscreen'].get('width')}x{config['capture']['fullscreen'].get('height')}")
        logger.debug(f"配置内容 - 模式: {config['capture'].get('mode')}")
    except Exception as e:
        logger.warning(f"无法读取配置文件 {config_path}: {e}, 使用默认值")
        config = {
            'capture': {
                'fps': 30,
                'mode': 'fullscreen',
                'window': {'process_name': 'WorldOfTanks.exe'},
                'fullscreen': {'width': 1920, 'height': 1080},
                'target_resolution': {'width': 960, 'height': 540}
            }
        }
    
    parser = argparse.ArgumentParser(description="Record World of Tanks gameplay")
    parser.add_argument(
        "--output",
        type=str,
        default="data/recordings",
        help="Output directory"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=config['capture'].get('fps', 30),
        help="Recording FPS"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["window", "fullscreen"],
        default=config['capture'].get('mode', 'window'),
        help="Capture mode: 'window' (recommended) or 'fullscreen'"
    )
    parser.add_argument(
        "--process",
        type=str,
        default=config['capture'].get('window', {}).get('process_name', 'WorldOfTanks.exe'),
        help="Process name for window capture (e.g., 'WorldOfTanks.exe')"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=config['capture'].get('fullscreen', {}).get('width', 1920),
        help="Screen width for fullscreen capture"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=config['capture'].get('fullscreen', {}).get('height', 1080),
        help="Screen height for fullscreen capture"
    )
    target_resolution_config = config.get('target_resolution', {})
    parser.add_argument(
        "--target-width",
        type=int,
        default=target_resolution_config.get('width', 960),
        help="Target width for saved frames (default: 960)"
    )
    parser.add_argument(
        "--target-height",
        type=int,
        default=target_resolution_config.get('height', 540),
        help="Target height for saved frames (default: 540)"
    )
    parser.add_argument(
        "--window-title",
        type=str,
        default=None,
        help="Window title for window capture (alternative to --process)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test screen capture without recording"
    )
    parser.add_argument(
        "--save-format",
        type=str,
        choices=["frames", "video", "both"],
        default="frames",
        help="保存格式: 'frames' (仅帧), 'video' (仅视频), 'both' (两者都有)"
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=2,  # 默认每2帧保存一次，配合 fps=5 得到 2.5 FPS
        help="帧采样间隔 (1=保存所有帧, 2=每2帧保存一次, 5=每5帧保存一次)"
    )
    parser.add_argument(
        "--use-global-hook",
        action="store_true",
        default=False,  # 暂时禁用，需要更多测试
        help="使用全局键盘钩子 (实验性功能)"
    )
    parser.add_argument(
        "--no-global-hook",
        dest="use_global_hook",
        action="store_false",
        help="禁用全局键盘钩子，使用 pynput (仅焦点窗口)"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("🎮 World of Tanks - 游戏录制工具")
    logger.info("=" * 80)
    logger.info("")
    logger.info("📋 当前配置:")
    logger.info(f"  输出目录: {args.output}")
    logger.info(f"  录制 FPS: {args.fps} (配置文件: {config['capture'].get('fps', 'N/A')})")
    logger.info(f"  捕获模式: {args.mode} (配置文件: {config['capture'].get('mode', 'N/A')})")
    logger.info(f"  保存格式: {args.save_format}")
    logger.info(f"  帧采样: 每 {args.frame_step} 帧保存一次")
    logger.info(f"  实际保存频率: {args.fps / args.frame_step:.2f} FPS")
    logger.info("")
    if args.mode == "window":
        logger.info("🎯 窗口捕获:")
        if args.process:
            logger.info(f"  目标进程: {args.process}")
        elif args.window_title:
            logger.info(f"  目标窗口: {args.window_title}")
        logger.info(f"  状态: {'可用' if WINDOW_CAPTURE_AVAILABLE else '不可用'}")
    else:
        logger.info("🖥️  全屏捕获:")
        config_width = config['capture'].get('fullscreen', {}).get('width', 'N/A')
        config_height = config['capture'].get('fullscreen', {}).get('height', 'N/A')
        logger.info(f"  分辨率: {args.width}x{args.height}")
        logger.info(f"  配置文件: {config_width}x{config_height}")
        if args.width != config_width or args.height != config_height:
            logger.warning(f"  ⚠️  注意：使用的分辨率与配置文件不同！")
    
    logger.info("")
    logger.info("⚙️  技术栈:")
    logger.info(f"  屏幕捕获: Python mss (数据采集模式)")
    if args.use_global_hook and GLOBAL_HOOK_AVAILABLE:
        logger.info(f"  键盘监听: 全局钩子 (可捕获所有窗口)")
    else:
        logger.info(f"  键盘监听: pynput (需要管理员权限)")
        if args.use_global_hook and not GLOBAL_HOOK_AVAILABLE:
            logger.warning("    ⚠️  全局键盘钩子不可用，已降级到 pynput")
    logger.info("")
    logger.info("使用说明：")
    logger.info("  1. 启动《坦克世界》并进入战斗")
    logger.info("  2. 按 F9 键开始录制")
    logger.info("  3. 正常游戏")
    logger.info("  4. 按 F10 键停止录制并保存")
    logger.info("  5. 可以再次按 F9 开始新的录制")
    logger.info("  6. 按 Ctrl+C 退出程序")
    logger.info("")
    logger.info("注意事项：")
    logger.info("  - 请至少录制 3-5 秒以确保有数据保存")
    logger.info("  - 可以连续录制多场战斗，无需重启程序")
    if args.save_format == "frames":
        logger.info("  - frames 模式内存占用小，推荐长时间录制")
    elif args.save_format == "video":
        logger.info("  - video 模式会占用内存，时间过长可能导致内存不足")
    else:
        logger.info("  - both 模式会同时保存帧和视频，占用较多内存")
    if args.mode == "window":
        logger.info("  - window 模式会自动跟踪游戏窗口（推荐）")
    else:
        logger.info("  - fullscreen 模式捕获整个屏幕")
    logger.info("=" * 80)
    
    if args.test:
        logger.info("\n[测试模式] 测试屏幕捕获功能...")
        try:
            sct = mss.mss()
            logger.info("✓ mss 初始化成功")
            monitor = {"top": 0, "left": 0, "width": args.width, "height": args.height}
            screenshot = sct.grab(monitor)
            frame = np.array(screenshot)[:, :, :3]
            logger.info(f"✓ 捕获成功: shape={frame.shape}, dtype={frame.dtype}")
            logger.info(f"✓ 分辨率: {frame.shape[1]}x{frame.shape[0]}")
            logger.info(f"✓ 数据大小: {frame.nbytes:,} 字节")
            
            # 保存测试帧
            import cv2
            test_path = Path("test_capture_output.png")
            # frame 已经是 BGR 格式，直接保存
            cv2.imwrite(str(test_path), frame)
            logger.info(f"✓ 测试图像已保存: {test_path}")
            
            logger.info("\n✓ 测试通过！屏幕捕获功能正常。")
            logger.info("可以开始正式录制了。")
        except Exception as e:
            logger.error(f"\n✗ 测试失败: {e}")
            import traceback
            traceback.print_exc()
        return
    
    # Create recorder
    try:
        logger.info("\n正在初始化录制器...")
        recorder = GameplayRecorder(
            output_dir=args.output,
            fps=args.fps,
            capture_mode=args.mode,
            process_name=args.process if args.mode == "window" else None,
            window_title=args.window_title if args.mode == "window" else None,
            save_format=args.save_format,
            frame_step=args.frame_step,
            screen_width=args.width,
            screen_height=args.height,
            target_width=args.target_width,
            target_height=args.target_height,
            use_global_hook=args.use_global_hook
        )
        logger.info("✓ 录制器初始化成功\n")
    except Exception as e:
        logger.error(f"\n✗ 录制器初始化失败: {e}")
        logger.error("\n故障排查：")
        logger.error("  1. 检查是否安装了依赖: pip install pywin32 psutil mss pynput opencv-python")
        logger.error("  2. 确保游戏已启动（window 模式）")
        logger.error("  3. 确保有屏幕访问权限")
        if args.mode == "window":
            logger.error(f"  4. 检查进程名是否正确: {args.process or args.window_title}")
            logger.error("  5. 尝试使用 fullscreen 模式: --mode fullscreen")
        return
    
    try:
        # 使用新的录制循环（支持 F9/F10 热键，可多次录制）
        recorder.runRecordingLoop()
    except Exception as e:
        logger.error(f"录制过程出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        recorder.cleanup()
        logger.info("\n录制器已清理")


if __name__ == "__main__":
    main()

