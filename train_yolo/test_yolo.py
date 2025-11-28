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
import cv2
import numpy as np
from ultralytics import YOLO
from loguru import logger


def main():
    parser = argparse.ArgumentParser(description="YOLO 模型推理和可视化")
    parser.add_argument("--model", type=str, required=True, help="YOLO 模型路径")
    parser.add_argument("--image", type=str, required=True, help="输入图像路径")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值（默认: 0.25）")
    parser.add_argument("--show", action="store_true", help="是否弹窗显示可视化结果")
    args = parser.parse_args()

    # ---- 路径检查 ----
    model_path = Path(args.model)
    image_path = Path(args.image)

    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"图像文件不存在: {image_path}")

    # ---- 加载模型 & 基本信息 ----
    model = YOLO(str(model_path))

    logger.info(f"加载模型: {model_path}")
    logger.info(f"模型设备: {model.device}")
    logger.info(f"模型任务类型: {getattr(model, 'task', 'unknown')}")
    logger.info(f"类别数: {len(model.names)}")
    logger.info(f"类别映射: {model.names}")

    # 打印更详细的网络结构（可选，调试时用）
    model.info(verbose=True)

    # ---- 读取图像 ----
    logger.info(f"读取图像: {image_path}")
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"无法读取图像文件: {image_path}")
    logger.info(f"原始图像尺寸: {image.shape} (H, W, C)")

    # ---- 推理 ----
    logger.info("开始推理...")
    # 注意：model(...) 返回的是 list[Results]
    results = model(image, conf=args.conf, verbose=False)
    res = results[0]

    # ---- 通用调试信息 ----
    logger.info(f"结果文件路径: {res.path}")
    logger.info(f"原始尺寸 orig_shape: {res.orig_shape}, resize 后 imgsz: {res.boxes.orig_shape if hasattr(res, 'boxes') else 'N/A'}")

    # 推理耗时（模型自带）
    if hasattr(res, "speed"):
        spd = res.speed
        logger.info(
            "耗时: preprocess={pre:.2f}ms, inference={inf:.2f}ms, postprocess={post:.2f}ms",
            pre=spd.get("preprocess", 0.0),
            inf=spd.get("inference", 0.0),
            post=spd.get("postprocess", 0.0),
        )

    # ---- 检测框 / OBB / 类别统计 ----
    if hasattr(res, "boxes") and res.boxes is not None:
        boxes = res.boxes  # ultralytics.engine.results.Boxes
        num_boxes = boxes.shape[0]
        logger.info(f"检测到 {num_boxes} 个目标")

        if num_boxes > 0:
            # xyxy, 置信度, 类别
            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            cls = boxes.cls.cpu().numpy().astype(int)

            logger.info(f"前 5 个检测框 (xyxy): {xyxy[:5]}")
            logger.info(f"前 5 个置信度: {conf[:5]}")
            logger.info(f"前 5 个类别: {cls[:5]}")

            # 按类别统计数量
            unique, counts = np.unique(cls, return_counts=True)
            cls_count = {model.names[int(k)]: int(v) for k, v in zip(unique, counts)}
            logger.info(f"按类别统计: {cls_count}")

    # ---- 若是 OBB 模型（旋转框） ----
    if hasattr(res, "obb") and res.obb is not None:
        obb = res.obb  # ultralytics.engine.results.OBB
        num_obb = obb.shape[0]
        logger.info(f"检测到 {num_obb} 个 OBB 旋转框")

        if num_obb > 0:
            # 旋转框的 xywhr 格式: (cx, cy, w, h, angle)
            xywhr = obb.xywhr.cpu().numpy()
            logger.info(f"前 5 个 OBB (cx, cy, w, h, angle_deg): {xywhr[:5]}")

    # ---- 若是 Pose 模型（关键点） ----
    if hasattr(res, "keypoints") and res.keypoints is not None:
        kpts = res.keypoints  # ultralytics.engine.results.Keypoints
        kp_xy = kpts.xy.cpu().numpy()          # 绝对坐标
        kp_conf = kpts.conf.cpu().numpy()      # 关键点置信度

        logger.info(f"关键点数组形状: {kp_xy.shape} (num_instances, num_kpts, 2)")
        logger.info(f"前 1 个实例的前 5 个关键点: {kp_xy[:1, :5, :]}")

    # ---- 输出 JSON（便于后续处理或存盘）----
    try:
        json_str = res.tojson()
        logger.debug(f"结果 JSON: {json_str[:500]} ...")  # 截断一下避免太长
    except Exception as e:
        logger.warning(f"tojson() 失败: {e}")

    # ---- 可视化结果（模型自带绘制） ----
    plotted = res.plot()  # 返回绘制好结果的 BGR 图像
    out_path = image_path.with_name(image_path.stem + "_yolo_vis.jpg")
    cv2.imwrite(str(out_path), plotted)
    logger.info(f"可视化结果已保存到: {out_path}")

    if args.show:
        cv2.imshow("YOLO Result", plotted)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
