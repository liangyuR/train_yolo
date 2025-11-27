#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO OBB 模型推理和可视化测试脚本

功能：
1. 加载 YOLO OBB 模型
2. 对图像进行推理
3. 可视化检测结果（旋转框 + 朝向箭头）
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from loguru import logger


def VisualizeObbResults(image: np.ndarray, results, show_arrow: bool = True) -> np.ndarray:
    """可视化 OBB 检测结果。"""
    img = image.copy()

    for result in results:
        if result.obb is None:
            continue

        # 统一搬到 CPU 再处理
        obb = result.obb.cpu()

        # 注意：在多数版本里 .xywhr / .xyxyxyxy 本身就返回 numpy，
        # 如果你这边 print(type(obb.xywhr)) 发现是 BaseTensor，就改成 obb.xywhr.data
        xywhr = obb.xywhr
        xyxyxyxy = obb.xyxyxyxy

        if hasattr(xywhr, "data"):  # 兼容 BaseTensor 新版本
            xywhr = xywhr.data
        if hasattr(xyxyxyxy, "data"):
            xyxyxyxy = xyxyxyxy.data

        xywhr = np.asarray(xywhr)          # (N, 5) -> cx, cy, w, h, r
        boxes = np.asarray(xyxyxyxy)       # (N, 4, 2)
        confidences = np.asarray(obb.conf) # (N,)
        class_ids = np.asarray(obb.cls).astype(int)
        class_names = result.names

        logger.info(f"num dets = {len(xywhr)}")

        for i in range(len(xywhr)):
            pts = boxes[i].astype(int).reshape(-1, 2)

            # 画旋转框
            cv2.drawContours(img, [pts], 0, (0, 255, 0), 2)

            cx, cy, w, h, angle_rad = xywhr[i]
            cx, cy = int(cx), int(cy)
            angle_deg = float(np.degrees(angle_rad))

            class_id = int(class_ids[i])
            conf = float(confidences[i])
            class_name = class_names.get(class_id, f"class_{class_id}")

            logger.info(
                f"[{i}] cls={class_name}, conf={conf:.3f}, "
                f"cx={cx}, cy={cy}, w={w:.1f}, h={h:.1f}, r(rad)={angle_rad:.4f}, r(deg)={angle_deg:.2f}"
            )

            # 箭头长度取长边的 0.4
            arrow_length = float(max(w, h) * 0.4)

            if show_arrow and class_name == "self_arrow":
                # r 是从 x 轴正向逆时针的弧度
                dx_arrow = np.cos(angle_rad) * arrow_length
                dy_arrow = np.sin(angle_rad) * arrow_length

                p_start = (cx, cy)
                p_end = (int(cx + dx_arrow), int(cy + dy_arrow))
                cv2.arrowedLine(img, p_start, p_end, (0, 0, 255), 2, tipLength=0.2)

            # 文本：类别 + 置信度
            label = f"{class_name} {conf:.2f}"
            cv2.putText(
                img,
                label,
                (cx - 40, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            # 文本：角度
            cv2.putText(
                img,
                f"ang: {angle_deg:.1f}°",
                (cx - 40, cy + 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 0),
                1,
                cv2.LINE_AA,
            )

    return img


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="YOLO OBB 模型推理和可视化")
    parser.add_argument("--model", type=str, required=True, help="YOLO OBB 模型路径")
    parser.add_argument("--image", type=str, required=True, help="输入图像路径")
    parser.add_argument("--output", type=str, help="输出图像路径（可选，不指定则显示窗口）")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值（默认: 0.25）")

    args = parser.parse_args()

    # 检查文件是否存在
    model_path = Path(args.model)
    image_path = Path(args.image)

    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    if not image_path.exists():
        raise FileNotFoundError(f"图像文件不存在: {image_path}")

    # 加载模型
    logger.info(f"加载模型: {model_path}")
    model = YOLO(str(model_path), task='obb')

    # 读取图像
    logger.info(f"读取图像: {image_path}")
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"无法读取图像文件: {image_path}")

    # 进行推理
    logger.info("开始推理...")
    results = model.predict(image, conf=args.conf, verbose=True)
    logger.info(results)

    # 可视化结果
    logger.info("可视化结果...")
    vis_image = VisualizeObbResults(image, results, show_arrow=True)

    # 保存或显示结果
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), vis_image)
        logger.info(f"结果已保存到: {output_path}")
    else:
        logger.info("按任意键关闭窗口...")
        cv2.imshow("OBB Detection Results", vis_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

