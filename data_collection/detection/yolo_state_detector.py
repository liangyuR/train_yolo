"""
基于YOLO和UI特征的游戏状态检测器
替代OCR方案，提供更快速、稳定的检测
"""

import cv2
import numpy as np
from typing import Optional, Dict, List, Tuple
from pathlib import Path
from loguru import logger
import sys

# YOLO支持
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("YOLO不可用，将仅使用UI特征检测: pip install ultralytics")

# 尝试导入现有的YoloDetector（可选）
try:
    # 尝试从yolo模块导入
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "yolo"))
    from main import YoloDetector as BaseYoloDetector
    BASE_YOLO_AVAILABLE = True
except ImportError:
    BASE_YOLO_AVAILABLE = False
    logger.debug("无法导入基础YoloDetector，将直接使用YOLO")


class YoloGameStateDetector:
    """
    基于YOLO和UI特征的游戏状态检测器
    
    支持：
    1. YOLO模型检测（如果提供模型）
    2. UI特征检测（颜色分析、区域检测）
    3. 混合检测（YOLO + 特征）
    """
    
    def __init__(
        self,
        screen_width: int = 1920,
        screen_height: int = 1080,
        yolo_model_path: Optional[str] = None,
        use_yolo: bool = True,
        use_ui_features: bool = True,
        confidence_threshold: float = 0.5
    ):
        """
        初始化YOLO游戏状态检测器
        
        Args:
            screen_width: 屏幕宽度
            screen_height: 屏幕高度
            yolo_model_path: YOLO模型路径（可选，如果提供则使用YOLO检测）
            use_yolo: 是否使用YOLO检测（需要提供模型）
            use_ui_features: 是否使用UI特征检测
            confidence_threshold: YOLO检测置信度阈值
        """
        self.screen_width_ = screen_width
        self.screen_height_ = screen_height
        self.confidence_threshold_ = confidence_threshold
        
        # 检测区域配置（复用OCR检测器的区域）
        region_width = int(screen_width * 0.5)
        region_height = int(screen_height * 0.15)
        region_left = int((screen_width - region_width) / 2)
        region_top = int(screen_height * 0.15)
        
        self.detection_region_ = {
            'left': region_left,
            'top': region_top,
            'width': region_width,
            'height': region_height
        }
        
        logger.info(f"检测区域配置: {self.detection_region_}")
        
        # YOLO模型初始化
        self.use_yolo_ = use_yolo and YOLO_AVAILABLE
        self.yolo_model_ = None
        self.yolo_model_path_ = yolo_model_path
        
        # YOLO类别到状态的映射（可配置）
        # 这些需要根据实际训练的模型类别ID来配置
        self.yolo_class_mapping_ = {
            # 示例映射（需要根据实际模型调整）
            # 0: 'battle_start',  # 假设类别0是"战斗开始"
            # 1: 'victory',       # 假设类别1是"胜利"
            # 2: 'defeat',        # 假设类别2是"失败"
            # 3: 'destroyed',    # 假设类别3是"被击毁"
        }
        
        if self.use_yolo_ and yolo_model_path:
            try:
                model_path = Path(yolo_model_path)
                if not model_path.is_absolute():
                    # 尝试从项目根目录查找
                    project_root = Path(__file__).parent.parent.parent
                    model_path = project_root / model_path
                
                if model_path.exists():
                    logger.info(f"加载YOLO模型: {model_path}")
                    self.yolo_model_ = YOLO(str(model_path))
                    logger.info("✓ YOLO模型加载成功")
                else:
                    logger.warning(f"YOLO模型文件不存在: {model_path}，将仅使用UI特征检测")
                    self.use_yolo_ = False
            except Exception as e:
                logger.error(f"加载YOLO模型失败: {e}，将仅使用UI特征检测")
                self.use_yolo_ = False
        
        # UI特征检测配置
        self.use_ui_features_ = use_ui_features
        
        # 颜色检测阈值（可配置）
        self.victory_color_threshold_ = 0.05  # 胜利颜色占比阈值
        self.defeat_color_threshold_ = 0.05  # 失败颜色占比阈值
        
        # HSV颜色范围定义（用于UI特征检测）
        # 绿色（胜利）：H=60-150, S>50, V>50
        self.victory_color_range_ = {
            'lower': np.array([50, 50, 50]),   # 绿色下限
            'upper': np.array([150, 255, 255])  # 绿色上限
        }
        
        # 红色（失败/被击毁）：H=0-10 或 170-180
        self.defeat_color_range_ = {
            'lower1': np.array([0, 50, 50]),    # 红色范围1下限
            'upper1': np.array([10, 255, 255]),  # 红色范围1上限
            'lower2': np.array([170, 50, 50]),  # 红色范围2下限
            'upper2': np.array([180, 255, 255])  # 红色范围2上限
        }
        
        # 金色/黄色（胜利，可选）：H=15-45
        self.gold_color_range_ = {
            'lower': np.array([15, 50, 50]),
            'upper': np.array([45, 255, 255])
        }
        
        # 战斗开始检测：历史状态追踪（用于检测从静态到动态的变化）
        self.last_region_brightness_ = None
        self.static_frame_count_ = 0
        self.battle_start_threshold_ = 30  # 连续静态帧数阈值
        
        # 状态缓存（避免频繁切换）
        self.last_state_ = None
        self.state_confirmed_frames_ = 0
        self.state_confirmation_threshold_ = 3  # 需要连续3帧确认状态
        
        # 采样检测优化
        self.detection_interval_ = 10  # 每10帧检测一次
        self.last_detection_frame_ = -1
        self.frame_counter_ = 0
        
        # 变化检测缓存
        self.last_region_hash_ = None
        self.hash_check_interval_ = 5  # 每5帧检查一次画面变化
        self.last_hash_check_frame_ = -1
        
        logger.info(f"YOLO状态检测器初始化完成 (YOLO: {self.use_yolo_}, UI特征: {self.use_ui_features_})")
    
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
    
    def DetectWithYolo(self, frame: np.ndarray) -> Optional[str]:
        """
        使用YOLO模型检测游戏状态
        
        Args:
            frame: 检测区域图像 (BGR格式)
            
        Returns:
            检测到的状态，或None
        """
        if not self.use_yolo_ or self.yolo_model_ is None:
            return None
        
        try:
            # YOLO模型期望RGB格式，但可以直接处理BGR
            results = self.yolo_model_(frame, verbose=False, conf=self.confidence_threshold_)
            
            detected_states = []
            
            for result in results:
                boxes = result.boxes
                if boxes is None or len(boxes) == 0:
                    continue
                
                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    # 检查类别映射
                    if class_id in self.yolo_class_mapping_:
                        state = self.yolo_class_mapping_[class_id]
                        detected_states.append((state, confidence))
            
            # 返回置信度最高的状态
            if detected_states:
                detected_states.sort(key=lambda x: x[1], reverse=True)
                return detected_states[0][0]
            
            return None
            
        except Exception as e:
            logger.debug(f"YOLO检测失败: {e}")
            return None
    
    def DetectWithUIFeatures(self, frame: np.ndarray) -> Optional[str]:
        """
        使用UI特征检测游戏状态
        
        Args:
            frame: 检测区域图像 (BGR格式)
            
        Returns:
            检测到的状态，或None
        """
        if not self.use_ui_features_:
            return None
        
        try:
            # 转换为HSV颜色空间
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # 降采样以提高速度（可选）
            if frame.shape[0] > 480 or frame.shape[1] > 640:
                scale = min(480 / frame.shape[0], 640 / frame.shape[1])
                new_h = int(frame.shape[0] * scale)
                new_w = int(frame.shape[1] * scale)
                hsv_frame = cv2.resize(hsv_frame, (new_w, new_h))
            
            # 检测胜利（绿色/金色）
            green_mask = cv2.inRange(
                hsv_frame,
                self.victory_color_range_['lower'],
                self.victory_color_range_['upper']
            )
            gold_mask = cv2.inRange(
                hsv_frame,
                self.gold_color_range_['lower'],
                self.gold_color_range_['upper']
            )
            
            victory_mask = cv2.bitwise_or(green_mask, gold_mask)
            green_pixel_ratio = np.sum(victory_mask > 0) / (victory_mask.size + 1e-6)
            
            # 检测失败/被击毁（红色）
            red_mask1 = cv2.inRange(
                hsv_frame,
                self.defeat_color_range_['lower1'],
                self.defeat_color_range_['upper1']
            )
            red_mask2 = cv2.inRange(
                hsv_frame,
                self.defeat_color_range_['lower2'],
                self.defeat_color_range_['upper2']
            )
            red_mask = cv2.bitwise_or(red_mask1, red_mask2)
            red_pixel_ratio = np.sum(red_mask > 0) / (red_mask.size + 1e-6)
            
            # 判断状态（基于颜色占比）
            # 使用可配置的阈值（如果有设置，否则使用默认值）
            victory_threshold = getattr(self, 'victory_color_threshold_', 0.05)
            defeat_threshold = getattr(self, 'defeat_color_threshold_', 0.05)
            
            if green_pixel_ratio > victory_threshold:
                # 胜利：绿色/金色通常出现在屏幕顶部中心区域
                # 可以进一步通过位置判断
                return 'victory'
            elif red_pixel_ratio > defeat_threshold:
                # 失败或被击毁：红色通常出现在特定位置
                # 可以根据位置、形状区分，这里简化为失败
                # 未来可以添加更精细的区分逻辑
                return 'defeat'
            
            # 战斗开始检测：检测特定区域的颜色变化或UI元素出现
            # 方法1：检测倒计时区域（通常在屏幕中心上方）
            # 方法2：检测"开始"按钮的颜色/形状
            # 这里可以通过检测特定区域的平均亮度变化来判断
            
            # 简化实现：如果画面从静态变为有变化，且没有检测到胜利/失败，可能是战斗开始
            # 但这个逻辑需要在DetectState中通过历史状态来判断
            # 这里仅返回None，由外部逻辑处理
            
            return None
            
        except Exception as e:
            logger.debug(f"UI特征检测失败: {e}")
            return None
    
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
        
        # 变化检测：仅当画面变化时才进行检测
        if frame_number is not None:
            should_check_hash = (frame_number - self.last_hash_check_frame_) >= self.hash_check_interval_
            if should_check_hash:
                # 计算区域哈希（使用简化哈希：平均值）
                region_hash = int(region.mean() * 1000)
                
                if region_hash == self.last_region_hash_:
                    # 画面未变化，跳过检测，返回上一次状态
                    if self.state_confirmed_frames_ >= self.state_confirmation_threshold_:
                        return self.last_state_
                    return None
                
                self.last_region_hash_ = region_hash
                self.last_hash_check_frame_ = frame_number
        
        # 更新检测帧号
        if frame_number is not None:
            self.last_detection_frame_ = frame_number
        
        # 降采样区域以提高速度（可选）
        if region.shape[0] > 360 or region.shape[1] > 640:
            scale = min(360 / region.shape[0], 640 / region.shape[1])
            new_h = int(region.shape[0] * scale)
            new_w = int(region.shape[1] * scale)
            region = cv2.resize(region, (new_w, new_h))
        
        detected_state = None
        
        # 1. 优先使用YOLO检测（如果可用）
        if self.use_yolo_:
            detected_state = self.DetectWithYolo(region)
        
        # 2. 如果YOLO未检测到，使用UI特征检测
        if detected_state is None and self.use_ui_features_:
            detected_state = self.DetectWithUIFeatures(region)
        
        # 3. 战斗开始检测（基于画面变化）
        if detected_state is None and self.use_ui_features_:
            # 检测画面从静态变为动态，可能是战斗开始
            current_brightness = np.mean(cv2.cvtColor(region, cv2.COLOR_BGR2GRAY))
            
            if self.last_region_brightness_ is not None:
                brightness_diff = abs(current_brightness - self.last_region_brightness_)
                
                # 如果画面亮度变化很小，认为是静态画面
                if brightness_diff < 5:  # 阈值可配置
                    self.static_frame_count_ += 1
                else:
                    # 画面有变化，重置静态计数
                    if self.static_frame_count_ >= self.battle_start_threshold_:
                        # 从长期静态变为动态，可能是战斗开始
                        detected_state = 'battle_start'
                        self.static_frame_count_ = 0
                    else:
                        self.static_frame_count_ = 0
            else:
                self.static_frame_count_ = 0
            
            self.last_region_brightness_ = current_brightness
        
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
    
    def SetYoloClassMapping(self, mapping: Dict[int, str]):
        """
        设置YOLO类别到状态的映射
        
        Args:
            mapping: 字典，键为类别ID，值为状态名称
        """
        self.yolo_class_mapping_ = mapping
        logger.info(f"YOLO类别映射已更新: {mapping}")
    
    def SetColorThresholds(
        self,
        victory_threshold: float = 0.05,
        defeat_threshold: float = 0.05
    ):
        """
        设置颜色检测阈值
        
        Args:
            victory_threshold: 胜利颜色占比阈值
            defeat_threshold: 失败颜色占比阈值
        """
        self.victory_color_threshold_ = victory_threshold
        self.defeat_color_threshold_ = defeat_threshold
        logger.info(f"颜色检测阈值已更新: 胜利={victory_threshold}, 失败={defeat_threshold}")
    
    def LoadYoloModel(self, model_path: str) -> bool:
        """
        加载YOLO模型（动态加载）
        
        Args:
            model_path: YOLO模型路径
            
        Returns:
            是否加载成功
        """
        if not YOLO_AVAILABLE:
            logger.error("YOLO不可用，请安装: pip install ultralytics")
            return False
        
        try:
            model_path_obj = Path(model_path)
            if not model_path_obj.is_absolute():
                # 尝试从项目根目录查找
                project_root = Path(__file__).parent.parent.parent
                model_path_obj = project_root / model_path_obj
            
            if not model_path_obj.exists():
                logger.error(f"YOLO模型文件不存在: {model_path_obj}")
                return False
            
            logger.info(f"加载YOLO模型: {model_path_obj}")
            self.yolo_model_ = YOLO(str(model_path_obj))
            self.yolo_model_path_ = str(model_path_obj)
            self.use_yolo_ = True
            logger.info("✓ YOLO模型加载成功")
            return True
            
        except Exception as e:
            logger.error(f"加载YOLO模型失败: {e}")
            self.use_yolo_ = False
            return False
    
    def GetYoloClassNames(self) -> List[str]:
        """
        获取YOLO模型的类别名称列表
        
        Returns:
            类别名称列表，如果模型未加载则返回空列表
        """
        if self.yolo_model_ is None:
            return []
        
        try:
            # 获取模型的类别名称
            if hasattr(self.yolo_model_, 'names'):
                return list(self.yolo_model_.names.values())
            return []
        except Exception as e:
            logger.debug(f"获取YOLO类别名称失败: {e}")
            return []

