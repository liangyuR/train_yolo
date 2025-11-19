"""
游戏状态检测模块
自动检测坦克世界游戏状态（战斗开始、胜利、被击败、被击毁）
"""

import cv2
import numpy as np
from typing import Optional, Tuple
from loguru import logger
import re

try:
    import pytesseract
    OCR_AVAILABLE = True
    EASYOCR_AVAILABLE = False
except ImportError:
    try:
        import easyocr
        OCR_AVAILABLE = True
        EASYOCR_AVAILABLE = True
    except ImportError:
        OCR_AVAILABLE = False
        EASYOCR_AVAILABLE = False
        logger.warning("OCR不可用，请安装: pip install pytesseract 或 pip install easyocr")


class GameStateDetector:
    """检测游戏状态（战斗开始、胜利、被击败、被击毁）"""
    
    def __init__(self, screen_width: int = 1920, screen_height: int = 1080, use_easyocr: bool = False):
        """
        初始化游戏状态检测器
        
        Args:
            screen_width: 屏幕宽度
            screen_height: 屏幕高度
            use_easyocr: 是否使用EasyOCR（否则使用pytesseract）
        """
        self.screen_width_ = screen_width
        self.screen_height_ = screen_height
        
        # 计算检测区域：屏幕中心靠上1/3
        # 假设检测区域为屏幕中心宽度的一半，高度为屏幕的1/6
        region_width = int(screen_width * 0.5)  # 宽度为屏幕的50%
        region_height = int(screen_height * 0.15)  # 高度为屏幕的15%
        region_left = int((screen_width - region_width) / 2)  # 居中
        region_top = int(screen_height * 0.15)  # 靠上15%的位置
        
        self.detection_region_ = {
            'left': region_left,
            'top': region_top,
            'width': region_width,
            'height': region_height
        }
        
        logger.info(f"检测区域配置: {self.detection_region_}")
        
        # 初始化OCR
        self.use_easyocr_ = use_easyocr and EASYOCR_AVAILABLE if OCR_AVAILABLE else False
        if self.use_easyocr_:
            logger.info("使用 EasyOCR 进行文字识别")
            self.easyocr_reader_ = easyocr.Reader(['en', 'ch_sim'], gpu=False)
        elif OCR_AVAILABLE:
            logger.info("使用 pytesseract 进行文字识别")
            # 尝试设置tesseract路径（如果需要）
            try:
                # Windows默认路径
                pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
            except:
                pass
        
        # 状态关键词
        self.battle_start_keywords_ = ['战斗开始', '战斗', '开始', 'BATTLE', 'FIGHT']
        self.victory_keywords_ = ['胜利', 'VICTORY', 'WIN']
        self.defeat_keywords_ = ['失败', '被击败', 'DEFEAT', 'LOSS']
        self.destroyed_keywords_ = ['被击毁', '摧毁', 'DESTROYED', 'KILLED']
        
        # 状态缓存（避免频繁切换）
        self.last_state_ = None
        self.state_confirmed_frames_ = 0
        self.state_confirmation_threshold_ = 3  # 需要连续3帧确认状态
        
        # 采样检测优化
        self.detection_interval_ = 10  # 每10帧检测一次（可配置）
        self.last_detection_frame_ = -1
        self.frame_counter_ = 0
        
        # 变化检测缓存（避免重复OCR相同画面）
        self.last_region_hash_ = None
        self.hash_check_interval_ = 5  # 每5帧检查一次画面变化
        self.last_hash_check_frame_ = -1
        
    def ExtractDetectionRegion(self, frame: np.ndarray) -> np.ndarray:
        """
        从完整帧中提取检测区域
        
        Args:
            frame: 完整的屏幕截图 (BGR格式)
            
        Returns:
            检测区域的图像 (BGR格式)
        """
        left = self.detection_region_['left']
        top = self.detection_region_['top']
        width = self.detection_region_['width']
        height = self.detection_region_['height']
        
        # 边界检查
        frame_h, frame_w = frame.shape[:2]
        left = max(0, min(left, frame_w - 1))
        top = max(0, min(top, frame_h - 1))
        width = min(width, frame_w - left)
        height = min(height, frame_h - top)
        
        region = frame[top:top+height, left:left+width]
        return region
    
    def PreprocessImage(self, image: np.ndarray) -> np.ndarray:
        """
        预处理图像以提高OCR识别率
        
        Args:
            image: 输入图像 (BGR格式)
            
        Returns:
            预处理后的图像
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 增强对比度
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # 二值化（可选，根据实际效果调整）
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary
    
    def RecognizeText(self, image: np.ndarray) -> str:
        """
        识别图像中的文字
        
        Args:
            image: 输入图像 (灰度或二值图)
            
        Returns:
            识别出的文字（可能为空字符串）
        """
        if not OCR_AVAILABLE:
            return ""
        
        try:
            if self.use_easyocr_:
                # 使用EasyOCR
                results = self.easyocr_reader_.readtext(image)
                text_parts = []
                for (bbox, text, confidence) in results:
                    if confidence > 0.5:  # 置信度阈值
                        text_parts.append(text)
                recognized_text = " ".join(text_parts)
            else:
                # 使用pytesseract
                # 配置OCR参数（中英文混合）
                config = '--psm 6 -l chi_sim+eng'
                recognized_text = pytesseract.image_to_string(image, config=config)
            
            # 清理文本
            recognized_text = recognized_text.strip()
            recognized_text = re.sub(r'\s+', ' ', recognized_text)  # 合并多个空格
            
            return recognized_text
        except Exception as e:
            logger.debug(f"OCR识别失败: {e}")
            return ""
    
    def DetectState(self, frame: np.ndarray, frame_number: int = None) -> Optional[str]:
        """
        检测游戏状态（优化版：采样检测 + 变化缓存）
        
        Args:
            frame: 完整的屏幕截图 (BGR格式)
            frame_number: 帧号（用于采样检测）
            
        Returns:
            检测到的状态: 'battle_start', 'victory', 'defeat', 'destroyed', 或 None
        """
        self.frame_counter_ += 1
        
        # 采样检测：每 N 帧检测一次
        if frame_number is not None:
            if frame_number - self.last_detection_frame_ < self.detection_interval_:
                # 返回上一次的状态（如果有）
                if self.state_confirmed_frames_ >= self.state_confirmation_threshold_:
                    return self.last_state_
                return None
        
        # 提取检测区域
        region = self.ExtractDetectionRegion(frame)
        
        if region.size == 0:
            return None
        
        # 变化检测：仅当画面变化时才进行OCR（节省计算）
        if frame_number is not None:
            should_check_hash = (frame_number - self.last_hash_check_frame_) >= self.hash_check_interval_
            if should_check_hash:
                # 计算区域哈希（使用简化哈希：平均值）
                region_hash = int(region.mean() * 1000)  # 简化哈希
                
                if region_hash == self.last_region_hash_:
                    # 画面未变化，跳过OCR，返回上一次状态
                    if self.state_confirmed_frames_ >= self.state_confirmation_threshold_:
                        return self.last_state_
                    return None
                
                self.last_region_hash_ = region_hash
                self.last_hash_check_frame_ = frame_number
        
        # 更新检测帧号
        if frame_number is not None:
            self.last_detection_frame_ = frame_number
        
        # 预处理图像
        processed = self.PreprocessImage(region)
        
        # OCR识别
        text = self.RecognizeText(processed)
        
        if not text:
            # 如果OCR没有识别到文字，重置状态缓存
            self.state_confirmed_frames_ = 0
            self.last_state_ = None
            return None
        
        logger.debug(f"识别文字: {text}")
        
        # 转换为大写进行匹配（不区分大小写）
        text_upper = text.upper()
        
        # 检测状态
        detected_state = None
        
        # 检测战斗开始
        for keyword in self.battle_start_keywords_:
            if keyword.upper() in text_upper:
                detected_state = 'battle_start'
                break
        
        # 检测胜利
        if not detected_state:
            for keyword in self.victory_keywords_:
                if keyword.upper() in text_upper:
                    detected_state = 'victory'
                    break
        
        # 检测被击败
        if not detected_state:
            for keyword in self.defeat_keywords_:
                if keyword.upper() in text_upper:
                    detected_state = 'defeat'
                    break
        
        # 检测被击毁
        if not detected_state:
            for keyword in self.destroyed_keywords_:
                if keyword.upper() in text_upper:
                    detected_state = 'destroyed'
                    break
        
        # 状态确认（避免误检）
        if detected_state == self.last_state_:
            self.state_confirmed_frames_ += 1
        else:
            self.state_confirmed_frames_ = 1
            self.last_state_ = detected_state
        
        # 只有当状态连续确认多帧后才返回
        if self.state_confirmed_frames_ >= self.state_confirmation_threshold_:
            return detected_state
        
        return None
    
    def SetDetectionInterval(self, interval: int):
        """
        设置检测间隔（采样频率）
        
        Args:
            interval: 每 N 帧检测一次
        """
        self.detection_interval_ = max(1, interval)
        logger.info(f"状态检测间隔设置为: 每 {self.detection_interval_} 帧检测一次")
    
    def GetDetectionRegion(self) -> dict:
        """获取检测区域配置"""
        return self.detection_region_.copy()


def test_detector():
    """测试游戏状态检测器"""
    import mss
    
    logger.info("=" * 80)
    logger.info("游戏状态检测器测试")
    logger.info("=" * 80)
    
    if not OCR_AVAILABLE:
        logger.error("❌ OCR不可用，请安装: pip install pytesseract 或 pip install easyocr")
        return
    
    # 初始化检测器
    try:
        sct = mss.mss()
        monitor_info = sct.monitors[1]
        screen_width = monitor_info['width']
        screen_height = monitor_info['height']
        
        detector = GameStateDetector(screen_width, screen_height)
        logger.info(f"✓ 检测器初始化成功 (屏幕: {screen_width}x{screen_height})")
        logger.info(f"✓ 检测区域: {detector.GetDetectionRegion()}")
    except Exception as e:
        logger.error(f"✗ 初始化失败: {e}")
        return
    
    logger.info("")
    logger.info("测试说明：")
    logger.info("  1. 切换到游戏窗口")
    logger.info("  2. 等待检测结果（按Ctrl+C退出）")
    logger.info("")
    
    try:
        import time
        frame_count = 0
        
        while True:
            # 截取屏幕
            screenshot = sct.grab(monitor_info)
            frame = np.array(screenshot)[:, :, :3]  # BGR格式
            
            # 检测状态
            state = detector.DetectState(frame)
            
            frame_count += 1
            if frame_count % 10 == 0 or state:
                if state:
                    logger.info(f"检测到状态: {state}")
                else:
                    logger.debug(f"未检测到状态 (帧数: {frame_count})")
            
            time.sleep(0.5)  # 每0.5秒检测一次
            
    except KeyboardInterrupt:
        logger.info("\n测试结束")


if __name__ == "__main__":
    test_detector()

