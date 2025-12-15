#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO 模型推理和可视化测试脚本

功能：
1. 加载 YOLO 模型
2. 对图像进行推理
3. 输出关键调试信息：
   - 模型信息（设备、任务、类别）
   - 推理耗时（pre/infer/post）
   - 检测框 / OBB / 关键点的明细
4. （可选）可视化结果
"""

import argparse
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
from ultralytics import YOLO
from loguru import logger


def _get_image_files(image_path: Path):
    """获取图片文件列表（支持单个文件或文件夹）"""
    if image_path.is_file():
        return [image_path]
    elif image_path.is_dir():
        # 支持的图片格式
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
        image_files = []
        for ext in extensions:
            image_files.extend(image_path.glob(f"*{ext}"))
            image_files.extend(image_path.glob(f"*{ext.upper()}"))
        # 去重并排序
        return sorted(set(image_files))
    else:
        raise ValueError(f"输入路径既不是文件也不是目录: {image_path}")


def _process_single_image(model, image_path: Path, conf: float, output_dir: Optional[Path] = None, show: bool = False):
    """处理单张图片"""
    logger.info(f"处理图像: {image_path}")

    # 读取图像（支持中文路径）
    try:
        img_array = np.fromfile(str(image_path), dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except Exception as e:
        logger.error(f"无法读取图像文件 {image_path}: {e}")
        return False

    if image is None:
        logger.error(f"无法解码图像文件: {image_path}")
        return False

    logger.info(f"原始图像尺寸: {image.shape} (H, W, C)")

    # 推理
    results = model(image, conf=conf, verbose=False)
    res = results[0]

    # 通用调试信息
    logger.info(f"结果文件路径: {res.path}")
    logger.info(f"原始尺寸 orig_shape: {res.orig_shape}")

    # 推理耗时
    if hasattr(res, "speed"):
        spd = res.speed
        logger.info(
            "耗时: preprocess={pre:.2f}ms, inference={inf:.2f}ms, postprocess={post:.2f}ms",
            pre=spd.get("preprocess", 0.0),
            inf=spd.get("inference", 0.0),
            post=spd.get("postprocess", 0.0),
        )

    # 检测框 / OBB / 类别统计
    if hasattr(res, "boxes") and res.boxes is not None:
        boxes = res.boxes
        num_boxes = boxes.shape[0]
        logger.info(f"检测到 {num_boxes} 个目标")

        if num_boxes > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            conf_values = boxes.conf.cpu().numpy()
            cls = boxes.cls.cpu().numpy().astype(int)

            logger.info(f"前 5 个检测框 (xyxy): {xyxy[:5]}")
            logger.info(f"前 5 个置信度: {conf_values[:5]}")
            logger.info(f"前 5 个类别: {cls[:5]}")

            # 按类别统计数量
            unique, counts = np.unique(cls, return_counts=True)
            cls_count = {model.names[int(k)]: int(v) for k, v in zip(unique, counts)}
            logger.info(f"按类别统计: {cls_count}")

    # OBB 模型（旋转框）
    if hasattr(res, "obb") and res.obb is not None:
        obb = res.obb
        num_obb = obb.shape[0]
        logger.info(f"检测到 {num_obb} 个 OBB 旋转框")

        if num_obb > 0:
            xywhr = obb.xywhr.cpu().numpy()
            logger.info(f"前 5 个 OBB (cx, cy, w, h, angle_deg): {xywhr[:5]}")

    # Pose 模型（关键点）
    if hasattr(res, "keypoints") and res.keypoints is not None:
        kpts = res.keypoints
        kp_xy = kpts.xy.cpu().numpy()
        kp_conf = kpts.conf.cpu().numpy()

        logger.info(f"关键点数组形状: {kp_xy.shape} (num_instances, num_kpts, 2)")
        logger.info(f"前 1 个实例的前 5 个关键点: {kp_xy[:1, :5, :]}")
        logger.info(f"前 1 个实例的前 5 个关键点置信度: {kp_conf[:1, :5]}")

    # 输出 JSON
    try:
        json_str = res.tojson()
        logger.debug(f"结果 JSON: {json_str[:500]} ...")
    except Exception as e:
        logger.warning(f"tojson() 失败: {e}")

    # 可视化结果
    plotted = res.plot()
    
    # 确定输出路径
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{image_path.stem}_yolo_vis.jpg"
    else:
        out_path = image_path.with_name(image_path.stem + "_yolo_vis.jpg")

    # 保存图片（支持中文路径）
    try:
        success, encoded_img = cv2.imencode('.jpg', plotted)
        if success:
            encoded_img.tofile(str(out_path))
            logger.info(f"可视化结果已保存到: {out_path}")
        else:
            logger.error(f"保存图片失败: {out_path}")
            return False
    except Exception as e:
        logger.error(f"保存图片失败 {out_path}: {e}")
        return False

    if show:
        cv2.imshow(f"YOLO Result - {image_path.name}", plotted)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return True


def main():
    parser = argparse.ArgumentParser(description="YOLO 模型推理和可视化")
    parser.add_argument("--model", type=str, required=True, help="YOLO 模型路径")
    parser.add_argument("--image", type=str, required=True, help="输入图像路径或文件夹路径")
    parser.add_argument("--conf", type=float, default=0.1, help="置信度阈值（默认: 0.1）")
    parser.add_argument("--show", action="store_true", help="是否弹窗显示可视化结果")
    parser.add_argument("--output", type=str, default=None, help="输出目录（可选，默认保存到原图同目录）")
    args = parser.parse_args()

    # 路径检查
    model_path = Path(args.model)
    image_path = Path(args.image)

    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"图像路径不存在: {image_path}")

    # 加载模型
    model = YOLO(str(model_path))

    logger.info(f"加载模型: {model_path}")
    logger.info(f"模型设备: {model.device}")
    logger.info(f"模型任务类型: {getattr(model, 'task', 'unknown')}")
    logger.info(f"类别数: {len(model.names)}")
    logger.info(f"类别映射: {model.names}")

    # 打印网络结构
    model.info(verbose=True)

    # 获取图片文件列表
    image_files = _get_image_files(image_path)
    if not image_files:
        logger.warning(f"未找到图片文件: {image_path}")
        return

    logger.info(f"找到 {len(image_files)} 张图片")

    # 输出目录
    output_dir = Path(args.output) if args.output else None

    # 依次处理每张图片
    success_count = 0
    for img_file in image_files:
        logger.info("=" * 60)
        if _process_single_image(model, img_file, args.conf, output_dir, args.show):
            success_count += 1

    logger.info("=" * 60)
    logger.info(f"处理完成: {success_count}/{len(image_files)} 成功")


if __name__ == "__main__":
    main()


# # 处理单张图片（原有功能）
# python train_yolo/test_yolo.py --model model.pt --image image.jpg

# # 处理文件夹中的所有图片
# python train_yolo/test_yolo.py --model model.pt --image /path/to/images/

# # 指定输出目录
# python train_yolo/test_yolo.py --model model.pt --image /path/to/images/ --output /path/to/output/

# # 显示可视化窗口
# python train_yolo/test_yolo.py --model model.pt --image /path/to/images/ --show