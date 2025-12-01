"""
异步帧保存器 - 使用独立线程保存帧，避免阻塞主录制循环
"""

import cv2
import queue
import threading
from pathlib import Path
from typing import Optional
import numpy as np
from loguru import logger


class AsyncFrameSaver:
    """异步帧保存器，使用队列和独立线程"""
    
    def __init__(self, frames_dir: Path, png_compression: int = 3, queue_size: int = 60, jpeg_quality: int = None):
        """
        初始化异步帧保存器
        
        Args:
            frames_dir: 帧保存目录
            png_compression: PNG 压缩级别 (0-9, 0=无压缩最快, 9=最高压缩最慢, 默认3)
            queue_size: 队列大小（帧数）
            jpeg_quality: 已弃用，保留用于向后兼容，实际使用 png_compression
        """
        self.frames_dir_ = frames_dir
        # 为了向后兼容，如果提供了 jpeg_quality，转换为 PNG 压缩级别
        if jpeg_quality is not None:
            # JPEG 质量 0-100 映射到 PNG 压缩级别 0-9 (反向映射，高质量=低压缩)
            self.png_compression_ = max(0, min(9, 9 - int(jpeg_quality / 11)))
            logger.warning("jpeg_quality 参数已弃用，请使用 png_compression (0-9)")
        else:
            self.png_compression_ = png_compression
        self.queue_ = queue.Queue(maxsize=queue_size)
        self.running_ = False
        self.thread_ = None
        self.total_saved_ = 0
        self.total_dropped_ = 0
        
    def Start(self):
        """启动保存线程"""
        if self.running_:
            return
        
        self.running_ = True
        self.thread_ = threading.Thread(target=self._saveWorker, daemon=True)
        self.thread_.start()
        logger.info(f"异步帧保存器已启动（队列大小: {self.queue_.maxsize}）")
        
    def Stop(self):
        """停止保存线程并等待队列清空"""
        if not self.running_:
            return
        
        logger.info(f"停止异步帧保存器，队列中还有 {self.queue_.qsize()} 帧...")
        
        # 等待队列清空（最多 10 秒）
        try:
            self.queue_.join()
            logger.info("队列已清空")
        except Exception as e:
            logger.warning(f"等待队列清空时出错: {e}")
        
        # 停止线程
        self.running_ = False
        self.queue_.put(None)  # 发送停止信号
        
        if self.thread_:
            self.thread_.join(timeout=5.0)
        
        logger.info(f"异步帧保存器已停止（已保存: {self.total_saved_}, 丢弃: {self.total_dropped_}）")
        
    def SaveFrame(self, frame: np.ndarray, frame_number: int) -> bool:
        """
        添加帧到保存队列
        
        Args:
            frame: BGR 格式的帧（mss 捕获的格式）
            frame_number: 帧号
            
        Returns:
            是否成功添加到队列
        """
        try:
            # 使用 put_nowait 避免阻塞
            self.queue_.put_nowait((frame.copy(), frame_number))
            return True
        except queue.Full:
            self.total_dropped_ += 1
            if self.total_dropped_ % 10 == 1:
                logger.warning(f"保存队列已满，丢弃帧 {frame_number}（累计丢弃: {self.total_dropped_}）")
            return False
            
    def _saveWorker(self):
        """保存线程工作函数"""
        while self.running_:
            try:
                item = self.queue_.get(timeout=1.0)
                
                if item is None:  # 停止信号
                    self.queue_.task_done()
                    break
                
                frame, frame_number = item
                
                # 检查 frames_dir_ 是否已初始化
                if self.frames_dir_ is None:
                    logger.error(f"frames_dir_ 未初始化，跳过帧 {frame_number}")
                    self.queue_.task_done()
                    continue
                
                # 保存帧为 PNG 格式
                frame_path = self.frames_dir_ / f"frame_{frame_number:06d}.png"
                # frame 已经是 BGR 格式，直接保存（cv2.imwrite 期望 BGR）
                # PNG 压缩级别 0-9，0=无压缩最快，9=最高压缩最慢
                success = cv2.imwrite(
                    str(frame_path),
                    frame,
                    [cv2.IMWRITE_PNG_COMPRESSION, self.png_compression_]
                )
                
                if success:
                    self.total_saved_ += 1
                else:
                    logger.error(f"保存帧 {frame_number} 失败")
                
                self.queue_.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"保存帧时出错: {e}")
                self.queue_.task_done()
                
    def GetQueueSize(self) -> int:
        """获取当前队列大小"""
        return self.queue_.qsize()
        
    def GetStats(self) -> dict:
        """获取统计信息"""
        return {
            "saved": self.total_saved_,
            "dropped": self.total_dropped_,
            "queue_size": self.queue_.qsize(),
            "queue_max": self.queue_.maxsize
        }

