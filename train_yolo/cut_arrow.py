#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 YOLO 模型检测箭头并裁切

功能：
1. 加载 YOLO 模型
2. 对输入图片/目录进行推理
3. 找到置信度最高的箭头检测框
4. 以检测框中心为基准，裁切指定大小的正方形区域
5. 保存裁切后的图片到输出目录
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from loguru import logger
from ultralytics import YOLO


def FindClassId(model, target_class: str) -> Optional[int]:
    """查找类别 ID，支持模糊匹配（不区分大小写）"""
    target_lower = target_class.lower()
    for class_id, class_name in model.names.items():
        if class_name.lower() == target_lower:
            return class_id
    return None


def CalculateCropBox(
    center_x: float,
    center_y: float,
    crop_size: int,
    img_width: int,
    img_height: int
) -> Tuple[int, int, int, int]:
    """
    计算裁剪框坐标，确保不超出图像边界
    
    Args:
        center_x: 中心点 x 坐标
        center_y: 中心点 y 坐标
        crop_size: 裁剪框边长
        img_width: 图像宽度
        img_height: 图像高度
    
    Returns:
        (x1, y1, x2, y2) 裁剪框坐标
    """
    half_size = crop_size // 2
    
    # 计算初始裁剪框
    x1 = int(center_x - half_size)
    y1 = int(center_y - half_size)
    x2 = int(center_x + half_size)
    y2 = int(center_y + half_size)
    
    # 处理边界情况：如果超出边界，则调整到边界
    if x1 < 0:
        x1 = 0
        x2 = min(crop_size, img_width)
    if y1 < 0:
        y1 = 0
        y2 = min(crop_size, img_height)
    if x2 > img_width:
        x2 = img_width
        x1 = max(0, img_width - crop_size)
    if y2 > img_height:
        y2 = img_height
        y1 = max(0, img_height - crop_size)
    
    return (x1, y1, x2, y2)


def ProcessImage(
    model: YOLO,
    image_path: Path,
    output_dir: Path,
    crop_size: int,
    conf_threshold: float,
    target_class_id: int
) -> bool:
    """
    处理单张图片：检测箭头并裁切
    
    Args:
        model: YOLO 模型
        image_path: 输入图片路径
        output_dir: 输出目录
        crop_size: 裁剪框大小
        conf_threshold: 置信度阈值
        target_class_id: 目标类别 ID
    
    Returns:
        是否成功处理
    """
    # 读取图片
    image = cv2.imread(str(image_path))
    if image is None:
        logger.error(f"无法读取图片: {image_path}")
        return False
    
    img_height, img_width = image.shape[:2]
    
    # 推理
    results = model(image, conf=conf_threshold, verbose=False)
    res = results[0]
    
    # 检查是否有检测结果
    if not hasattr(res, "boxes") or res.boxes is None:
        logger.warning(f"未检测到任何目标: {image_path}")
        return False
    
    boxes = res.boxes
    if boxes.shape[0] == 0:
        logger.warning(f"未检测到任何目标: {image_path}")
        return False
    
    # 提取检测框信息
    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    
    # 过滤目标类别
    target_mask = cls == target_class_id
    if not np.any(target_mask):
        logger.warning(f"未检测到目标类别 (ID={target_class_id}): {image_path}")
        return False
    
    # 选择置信度最高的检测框
    target_xyxy = xyxy[target_mask]
    target_conf = conf[target_mask]
    best_idx = np.argmax(target_conf)
    best_box = target_xyxy[best_idx]
    best_conf = target_conf[best_idx]
    
    # 计算检测框中心
    x1, y1, x2, y2 = best_box
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    
    # 计算裁剪框
    crop_x1, crop_y1, crop_x2, crop_y2 = CalculateCropBox(
        center_x, center_y, crop_size, img_width, img_height
    )
    
    # 执行裁剪
    cropped = image[crop_y1:crop_y2, crop_x1:crop_x2]
    
    # 保存结果
    output_path = output_dir / f"{image_path.stem}_cropped{image_path.suffix}"
    cv2.imwrite(str(output_path), cropped)
    
    logger.info(
        f"处理成功: {image_path.name} -> {output_path.name} "
        f"(置信度: {best_conf:.3f}, 中心: ({center_x:.1f}, {center_y:.1f}))"
    )
    return True


def main():
    parser = argparse.ArgumentParser(description="使用 YOLO 模型检测箭头并裁切")
    parser.add_argument("--model", type=str, required=True, help="YOLO 模型路径")
    parser.add_argument("--input", type=str, required=True, help="输入图片路径或目录")
    parser.add_argument("--output", type=str, required=True, help="输出目录")
    parser.add_argument("--size", type=int, default=128, help="裁剪框边长（默认: 128）")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值（默认: 0.25）")
    parser.add_argument(
        "--target-class",
        type=str,
        default="arrow",
        help="目标类别名称（默认: arrow）"
    )
    args = parser.parse_args()
    
    # 路径检查
    model_path = Path(args.model)
    input_path = Path(args.input)
    output_dir = Path(args.output)
    
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    if not input_path.exists():
        raise FileNotFoundError(f"输入路径不存在: {input_path}")
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载模型
    logger.info(f"加载模型: {model_path}")
    model = YOLO(str(model_path))
    logger.info(f"模型设备: {model.device}")
    logger.info(f"类别映射: {model.names}")
    
    # 查找目标类别 ID
    target_class_id = FindClassId(model, args.target_class)
    if target_class_id is None:
        raise ValueError(
            f"未找到类别 '{args.target_class}'。可用类别: {list(model.names.values())}"
        )
    logger.info(f"目标类别: '{args.target_class}' (ID={target_class_id})")
    
    # 收集图片文件
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
    image_paths = []
    
    if input_path.is_file():
        if input_path.suffix.lower() in image_extensions:
            image_paths.append(input_path)
        else:
            raise ValueError(f"输入文件不是支持的图片格式: {input_path}")
    elif input_path.is_dir():
        for ext in image_extensions:
            image_paths.extend(input_path.rglob(f"*{ext}"))
            image_paths.extend(input_path.rglob(f"*{ext.upper()}"))
        image_paths = sorted(set(image_paths))
    else:
        raise ValueError(f"输入路径既不是文件也不是目录: {input_path}")
    
    if not image_paths:
        raise ValueError(f"未找到任何图片文件: {input_path}")
    
    logger.info(f"找到 {len(image_paths)} 张图片")
    
    # 处理每张图片
    success_count = 0
    for i, image_path in enumerate(image_paths, 1):
        logger.info(f"处理进度: {i}/{len(image_paths)} - {image_path.name}")
        if ProcessImage(
            model, image_path, output_dir, args.size, args.conf, target_class_id
        ):
            success_count += 1
    
    logger.info(f"处理完成: 成功 {success_count}/{len(image_paths)} 张")


if __name__ == "__main__":
    main()

