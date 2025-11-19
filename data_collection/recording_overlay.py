"""
录制状态覆盖层
在屏幕左上角显示录制状态信息（适用于无边框全屏游戏）
"""

import tkinter as tk
from tkinter import ttk
from typing import Optional, Dict, Any
import threading
import time
import queue
from loguru import logger

try:
    from data_collection.recording_events import RecordingEvent, RecordingEventType, EventDispatcher
except ImportError:
    try:
        from recording_events import RecordingEvent, RecordingEventType, EventDispatcher
    except ImportError:
        RecordingEvent = None
        RecordingEventType = None
        EventDispatcher = None


class RecordingOverlay:
    """录制状态覆盖层窗口"""
    
    def __init__(self, screen_width: int = 1920, screen_height: int = 1080, event_dispatcher: EventDispatcher = None):
        """
        初始化覆盖层
        
        Args:
            screen_width: 屏幕宽度
            screen_height: 屏幕高度
            event_dispatcher: 事件分发器（可选，如果提供则自动订阅事件）
        """
        self.screen_width_ = screen_width
        self.screen_height_ = screen_height
        self.root_ = None
        self.is_running_ = False
        self.overlay_thread_ = None
        self.event_dispatcher_ = event_dispatcher
        
        # 状态信息
        self.is_recording_ = False
        self.frame_count_ = 0
        self.duration_seconds_ = 0.0
        
        # 线程安全的更新队列
        self.update_queue_ = queue.Queue(maxsize=100)  # 限制队列大小，避免内存问题
        self.window_ready_event_ = threading.Event()  # 窗口初始化完成事件
        
        # 如果提供了事件分发器，自动订阅事件
        if self.event_dispatcher_ and RecordingEventType:
            self.event_dispatcher_.Subscribe(RecordingEventType.FRAME_CAPTURED, self._OnFrameCaptured)
            self.event_dispatcher_.Subscribe(RecordingEventType.RECORDING_STARTED, self._OnRecordingStarted)
            self.event_dispatcher_.Subscribe(RecordingEventType.RECORDING_STOPPED, self._OnRecordingStopped)
        
    def _CreateOverlayWindow(self):
        """
        创建透明覆盖窗口（内部方法，在覆盖层线程中调用）
        """
        try:
            root = tk.Tk()
            root.title("Recording Overlay")
            
            # 设置窗口无边框、置顶、透明
            root.overrideredirect(True)  # 移除标题栏和边框
            root.attributes('-topmost', True)  # 始终置顶
            root.attributes('-alpha', 0.85)  # 半透明（0.0-1.0）
            
            # 设置窗口位置（左上角）
            window_width = 280
            window_height = 160
            x = 10  # 距离左边缘10像素
            y = 10  # 距离上边缘10像素
            
            root.geometry(f"{window_width}x{window_height}+{x}+{y}")
            
            # 设置背景为深色半透明
            root.config(bg='#1a1a1a')
            
            # 创建内容框架
            content_frame = tk.Frame(root, bg='#1a1a1a')
            content_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # 标题
            title_label = tk.Label(
                content_frame,
                text="录制状态",
                font=("微软雅黑", 12, "bold"),
                bg='#1a1a1a',
                fg='#ffffff',
                anchor='w'
            )
            title_label.pack(fill=tk.X, pady=(0, 5))
            
            # 状态标签
            self.status_label_ref_ = tk.Label(
                content_frame,
                text="等待开始...",
                font=("Consolas", 10),
                bg='#1a1a1a',
                fg='#ff6b6b',
                anchor='w'
            )
            self.status_label_ref_.pack(fill=tk.X, pady=(0, 3))
            
            # 帧数标签
            self.frame_label_ref_ = tk.Label(
                content_frame,
                text="帧数: 0",
                font=("Consolas", 9),
                bg='#1a1a1a',
                fg='#95a5a6',
                anchor='w'
            )
            self.frame_label_ref_.pack(fill=tk.X, pady=(0, 3))
            
            # 时长标签
            self.duration_label_ref_ = tk.Label(
                content_frame,
                text="时长: 00:00",
                font=("Consolas", 9),
                bg='#1a1a1a',
                fg='#95a5a6',
                anchor='w'
            )
            self.duration_label_ref_.pack(fill=tk.X)
            
            # 点击窗口可以拖动
            self.drag_start_x_ = 0
            self.drag_start_y_ = 0
            
            def start_drag(event):
                self.drag_start_x_ = event.x
                self.drag_start_y_ = event.y
            
            def do_drag(event):
                x = root.winfo_x() + event.x - self.drag_start_x_
                y = root.winfo_y() + event.y - self.drag_start_y_
                root.geometry(f"+{x}+{y}")
            
            root.bind('<Button-1>', start_drag)
            root.bind('<B1-Motion>', do_drag)
            
            # 右键点击关闭
            def close_overlay(event):
                self.Stop()
            
            root.bind('<Button-3>', close_overlay)
            
            return root
            
        except Exception as e:
            logger.error(f"创建覆盖层窗口失败: {e}")
            return None
    
    def UpdateDisplay(self, is_recording: bool = None, frame_count: int = None, duration_seconds: float = None):
        """
        更新显示内容（线程安全：通过队列传递更新请求）
        
        Args:
            is_recording: 是否正在录制（None则不更新）
            frame_count: 当前帧数（None则不更新）
            duration_seconds: 录制时长（秒）（None则不更新）
        """
        if not self.is_running_:
            return
        
        try:
            # 构造更新数据
            update_data = {}
            if is_recording is not None:
                update_data['is_recording'] = is_recording
            if frame_count is not None:
                update_data['frame_count'] = frame_count
            if duration_seconds is not None:
                update_data['duration_seconds'] = duration_seconds
            
            # 如果没有任何更新，直接返回
            if not update_data:
                return
            
            # 将更新请求放入队列（线程安全）
            try:
                self.update_queue_.put_nowait(update_data)
            except queue.Full:
                # 队列满时，记录警告但继续运行（不会阻塞调用者）
                logger.debug(f"更新队列已满，丢弃更新请求")
                
        except Exception as e:
            logger.debug(f"更新显示失败: {e}")
    
    def _ProcessUpdateQueue(self):
        """
        处理更新队列中的所有更新请求（在覆盖层线程中调用）
        """
        if self.root_ is None or not self.root_.winfo_exists():
            return
        
        try:
            # 处理队列中的所有更新请求（批量处理）
            processed_count = 0
            max_updates_per_cycle = 10  # 每轮最多处理10个更新，避免阻塞
            
            while processed_count < max_updates_per_cycle:
                try:
                    # 非阻塞获取更新请求
                    update_data = self.update_queue_.get_nowait()
                    
                    # 应用更新
                    if 'is_recording' in update_data:
                        self.is_recording_ = update_data['is_recording']
                    if 'frame_count' in update_data:
                        self.frame_count_ = update_data['frame_count']
                    if 'duration_seconds' in update_data:
                        self.duration_seconds_ = update_data['duration_seconds']
                    
                    processed_count += 1
                    
                except queue.Empty:
                    # 队列为空，退出循环
                    break
            
            # 更新显示
            self._RefreshDisplay()
            
        except Exception as e:
            logger.debug(f"处理更新队列失败: {e}")
    
    def _RefreshDisplay(self):
        """
        刷新显示（在覆盖层线程中调用，确保线程安全）
        """
        if self.root_ is None or not self.root_.winfo_exists():
            return
        
        try:
            # 更新状态
            if self.is_recording_:
                status_text = "● 录制中"
                status_color = "#2ecc71"  # 绿色
            else:
                status_text = "○ 等待中"
                status_color = "#ff6b6b"  # 红色
            
            # 格式化时长
            minutes = int(self.duration_seconds_ // 60)
            seconds = int(self.duration_seconds_ % 60)
            duration_text = f"{minutes:02d}:{seconds:02d}"
            
            # 更新标签（现在在覆盖层线程中，可以直接更新）
            self._UpdateLabels(status_text, status_color, self.frame_count_, duration_text)
                
        except Exception as e:
            logger.debug(f"更新覆盖层显示失败: {e}")
    
    def _ScheduleUpdate(self):
        """
        调度下一次更新检查（在覆盖层线程中，通过Tkinter定时器）
        """
        if self.root_ is None or not self.root_.winfo_exists():
            return
        
        try:
            # 处理更新队列
            self._ProcessUpdateQueue()
            
            # 100ms后再次检查（约10 FPS的更新频率）
            self.root_.after(100, self._ScheduleUpdate)
            
        except Exception as e:
            logger.debug(f"调度更新失败: {e}")
    
    def _OnFrameCaptured(self, event: RecordingEvent):
        """帧捕获事件处理"""
        data = event.GetData()
        self.UpdateDisplay(
            frame_count=data.get('frame_count'),
            duration_seconds=data.get('duration_seconds')
        )
    
    def _OnRecordingStarted(self, event: RecordingEvent):
        """录制开始事件处理"""
        self.UpdateDisplay(is_recording=True)
    
    def _OnRecordingStopped(self, event: RecordingEvent):
        """录制停止事件处理"""
        data = event.GetData()
        self.UpdateDisplay(
            is_recording=False,
            frame_count=data.get('frame_count'),
            duration_seconds=data.get('duration_seconds')
        )
    
    def _UpdateLabels(self, status_text: str, status_color: str, frame_count: int, duration_text: str):
        """在主线程中更新标签"""
        try:
            if self.status_label_ref_:
                self.status_label_ref_.config(text=status_text, fg=status_color)
            if self.frame_label_ref_:
                self.frame_label_ref_.config(text=f"帧数: {frame_count:,}")
            if self.duration_label_ref_:
                self.duration_label_ref_.config(text=f"时长: {duration_text}")
        except Exception as e:
            logger.debug(f"更新标签失败: {e}")
    
    def _RunOverlayLoop(self):
        """
        运行覆盖层消息循环（在覆盖层线程中）
        负责：创建窗口、启动定时器、运行主循环
        """
        try:
            # 在覆盖层线程中创建窗口
            self.root_ = self._CreateOverlayWindow()
            
            if not self.root_:
                logger.error("无法创建覆盖层窗口")
                self.window_ready_event_.set()  # 即使失败也设置事件，避免主线程阻塞
                return
            
            # 窗口创建完成，通知主线程
            self.window_ready_event_.set()
            
            # 启动定时器检查更新队列
            self._ScheduleUpdate()
            
            # 运行主循环（阻塞直到窗口关闭）
            self.root_.mainloop()
            
        except Exception as e:
            logger.error(f"覆盖层消息循环错误: {e}")
            self.window_ready_event_.set()  # 确保事件被设置
    
    def Start(self):
        """启动覆盖层"""
        if self.is_running_:
            return
        
        try:
            self.is_running_ = True
            self.window_ready_event_.clear()  # 重置事件
            
            # 在独立线程中运行覆盖层（窗口创建也在该线程中）
            self.overlay_thread_ = threading.Thread(target=self._RunOverlayLoop, daemon=True)
            self.overlay_thread_.start()
            
            # 等待窗口初始化完成（最多等待2秒）
            if self.window_ready_event_.wait(timeout=2.0):
                logger.info("✓ 录制状态覆盖层已启动（屏幕左上角）")
                logger.info("  - 左键拖动：移动窗口")
                logger.info("  - 右键点击：关闭覆盖层")
            else:
                logger.warning("覆盖层窗口初始化超时")
                self.is_running_ = False
            
        except Exception as e:
            logger.error(f"启动覆盖层失败: {e}")
            self.is_running_ = False
            self.window_ready_event_.set()  # 确保事件被设置
    
    def Stop(self):
        """停止覆盖层"""
        if not self.is_running_:
            return
        
        self.is_running_ = False
        
        try:
            # 清空更新队列
            while not self.update_queue_.empty():
                try:
                    self.update_queue_.get_nowait()
                except queue.Empty:
                    break
            
            # 在覆盖层线程中关闭窗口
            if self.root_ and self.root_.winfo_exists():
                # 使用 after 确保在主循环线程中调用
                self.root_.after(0, self._DestroyWindow)
                
                # 等待线程结束（最多1秒）
                if self.overlay_thread_ and self.overlay_thread_.is_alive():
                    self.overlay_thread_.join(timeout=1.0)
            
            self.root_ = None
            logger.info("覆盖层已关闭")
            
        except Exception as e:
            logger.debug(f"关闭覆盖层时出错: {e}")
    
    def _DestroyWindow(self):
        """销毁窗口（在覆盖层线程中调用）"""
        try:
            if self.root_:
                self.root_.destroy()
        except Exception as e:
            logger.debug(f"销毁窗口时出错: {e}")
    
    def IsRunning(self) -> bool:
        """检查覆盖层是否正在运行"""
        return self.is_running_


def test_overlay():
    """测试覆盖层"""
    import mss
    
    logger.info("=" * 80)
    logger.info("录制状态覆盖层测试")
    logger.info("=" * 80)
    
    try:
        sct = mss.mss()
        monitor_info = sct.monitors[1]
        screen_width = monitor_info['width']
        screen_height = monitor_info['height']
        
        overlay = RecordingOverlay(screen_width, screen_height)
        overlay.Start()
        
        logger.info("覆盖层已启动，开始测试更新...")
        logger.info("按 Ctrl+C 退出")
        
        import time
        start_time = time.time()
        
        try:
            while True:
                elapsed = time.time() - start_time
                frame_count = int(elapsed * 30)  # 模拟30 FPS
                
                # 模拟录制状态切换
                is_recording = (int(elapsed) % 10) < 5
                
                overlay.UpdateDisplay(
                    is_recording=is_recording,
                    frame_count=frame_count,
                    duration_seconds=elapsed
                )
                
                time.sleep(0.5)
                
        except KeyboardInterrupt:
            logger.info("\n测试结束")
        finally:
            overlay.Stop()
            
    except Exception as e:
        logger.error(f"测试失败: {e}")


if __name__ == "__main__":
    test_overlay()