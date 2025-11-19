"""
录制事件系统 - 用于解耦组件之间的通信
"""

from typing import Callable, List, Dict, Any
from enum import Enum
from loguru import logger


class RecordingEventType(Enum):
    """录制事件类型"""
    RECORDING_STARTED = "recording_started"
    RECORDING_STOPPED = "recording_stopped"
    FRAME_CAPTURED = "frame_captured"
    STATE_DETECTED = "state_detected"
    ERROR_OCCURRED = "error_occurred"


class RecordingEvent:
    """录制事件"""
    
    def __init__(self, event_type: RecordingEventType, data: Dict[str, Any] = None):
        """
        初始化事件
        
        Args:
            event_type: 事件类型
            data: 事件数据
        """
        self.type_ = event_type
        self.data_ = data or {}
    
    def GetType(self) -> RecordingEventType:
        """获取事件类型"""
        return self.type_
    
    def GetData(self) -> Dict[str, Any]:
        """获取事件数据"""
        return self.data_


class EventDispatcher:
    """事件分发器 - 观察者模式的简化实现"""
    
    def __init__(self):
        """初始化事件分发器"""
        self.listeners_: Dict[RecordingEventType, List[Callable]] = {}
    
    def Subscribe(self, event_type: RecordingEventType, callback: Callable[[RecordingEvent], None]):
        """
        订阅事件
        
        Args:
            event_type: 事件类型
            callback: 回调函数 callback(event: RecordingEvent)
        """
        if event_type not in self.listeners_:
            self.listeners_[event_type] = []
        self.listeners_[event_type].append(callback)
        logger.debug(f"订阅事件: {event_type.value}")
    
    def Unsubscribe(self, event_type: RecordingEventType, callback: Callable):
        """
        取消订阅事件
        
        Args:
            event_type: 事件类型
            callback: 回调函数
        """
        if event_type in self.listeners_:
            try:
                self.listeners_[event_type].remove(callback)
            except ValueError:
                pass
    
    def Emit(self, event: RecordingEvent):
        """
        发送事件
        
        Args:
            event: 录制事件
        """
        event_type = event.GetType()
        if event_type in self.listeners_:
            for callback in self.listeners_[event_type]:
                try:
                    callback(event)
                except Exception as e:
                    logger.error(f"事件回调执行失败: {e}")


class RecordingStats:
    """录制统计信息 - 简化状态管理"""
    
    def __init__(self):
        """初始化统计信息"""
        self.is_recording_ = False
        self.frame_count_ = 0
        self.start_time_ = None
        self.duration_seconds_ = 0.0
    
    def Start(self):
        """开始录制"""
        self.is_recording_ = True
        self.start_time_ = None  # 由外部设置
        self.frame_count_ = 0
        self.duration_seconds_ = 0.0
    
    def Stop(self):
        """停止录制"""
        self.is_recording_ = False
    
    def UpdateFrameCount(self, count: int):
        """更新帧数"""
        self.frame_count_ = count
    
    def UpdateDuration(self, duration: float):
        """更新时长"""
        self.duration_seconds_ = duration
    
    def GetIsRecording(self) -> bool:
        """是否正在录制"""
        return self.is_recording_
    
    def GetFrameCount(self) -> int:
        """获取帧数"""
        return self.frame_count_
    
    def GetDuration(self) -> float:
        """获取时长"""
        return self.duration_seconds_
    
    def GetStatsDict(self) -> Dict[str, Any]:
        """获取统计信息字典"""
        return {
            'is_recording': self.is_recording_,
            'frame_count': self.frame_count_,
            'duration_seconds': self.duration_seconds_
        }

