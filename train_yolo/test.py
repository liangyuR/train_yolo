#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO 模型推理和可视化测试脚本（支持 OBB 和 Pose）

功能：
1. 加载 YOLO 模型（OBB 或 Pose）
2. 对图像进行推理
3. 可视化检测结果：
   - OBB: 旋转框 + 朝向箭头
   - Pose: 关键点 + 边界框 + 连接线
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

def VisualizePoseResults(image: np.ndarray, results, show_arrow: bool = True) -> np.ndarray:
    """可视化 Pose 检测结果。

    根据 Ultralytics 文档：
    - result.keypoints.xy: 关键点坐标（像素坐标），形状 (N, num_keypoints, 2)
    - result.keypoints.xyn: 归一化的关键点坐标，形状 (N, num_keypoints, 2)
    - result.keypoints.conf: 关键点置信度，形状 (N, num_keypoints)
    - result.boxes: 边界框（如果存在）
    """
    img = image.copy()

    for result in results:
        if result.keypoints is None:
            continue

        # 统一搬到 CPU 再处理
        keypoints = result.keypoints.cpu()
        boxes = result.boxes.cpu() if result.boxes is not None else None

        # 获取关键点数据
        kpts_xy = keypoints.xy  # (N, num_keypoints, 2)
        kpts_conf = keypoints.conf if hasattr(keypoints, 'conf') else None  # (N, num_keypoints)

        if hasattr(kpts_xy, "data"):  # 兼容 BaseTensor 新版本
            kpts_xy = kpts_xy.data
        if kpts_conf is not None and hasattr(kpts_conf, "data"):
            kpts_conf = kpts_conf.data

        kpts_xy = np.asarray(kpts_xy)  # (N, num_keypoints, 2)
        if kpts_conf is not None:
            kpts_conf = np.asarray(kpts_conf)  # (N, num_keypoints)

        # 获取边界框信息（如果存在）
        if boxes is not None:
            boxes_xyxy = boxes.xyxy
            if hasattr(boxes_xyxy, "data"):
                boxes_xyxy = boxes_xyxy.data
            boxes_xyxy = np.asarray(boxes_xyxy)  # (N, 4)
            confidences = np.asarray(boxes.conf)  # (N,)
            class_ids = np.asarray(boxes.cls).astype(int)  # (N,)
        else:
            # 如果没有边界框，从关键点计算边界框
            boxes_xyxy = None
            confidences = np.ones(len(kpts_xy))  # 默认置信度为 1.0
            class_ids = np.zeros(len(kpts_xy), dtype=int)  # 默认类别为 0

        class_names = result.names

        logger.info(f"num dets = {len(kpts_xy)}")

        # 定义关键点颜色（BGR格式）
        kpt_colors = [
            (0, 255, 0),    # 绿色 - 第一个关键点
            (255, 0, 0),    # 蓝色 - 第二个关键点
            (0, 0, 255),    # 红色 - 第三个关键点（如果有）
        ]

        for i in range(len(kpts_xy)):
            kpts = kpts_xy[i]  # (num_keypoints, 2)
            num_kpts = len(kpts)

            # 绘制边界框（如果存在）
            if boxes_xyxy is not None:
                x1, y1, x2, y2 = boxes_xyxy[i].astype(int)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
            else:
                # 从关键点计算边界框
                if num_kpts > 0:
                    kpts_int = kpts.astype(int)
                    x1, y1 = kpts_int.min(axis=0)
                    x2, y2 = kpts_int.max(axis=0)
                    # 添加一些边距
                    margin = 10
                    x1 = max(0, x1 - margin)
                    y1 = max(0, y1 - margin)
                    x2 = min(img.shape[1], x2 + margin)
                    y2 = min(img.shape[0], y2 + margin)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                else:
                    continue

            class_id = int(class_ids[i])
            conf = float(confidences[i])
            class_name = class_names.get(class_id, f"class_{class_id}")

            # 绘制关键点
            for kpt_idx, kpt in enumerate(kpts):
                if kpts_conf is not None and kpts_conf[i, kpt_idx] < 0.5:
                    continue  # 跳过低置信度关键点

                x, y = int(kpt[0]), int(kpt[1])
                color = kpt_colors[kpt_idx % len(kpt_colors)]
                
                # 绘制关键点圆圈
                cv2.circle(img, (x, y), 5, color, -1)
                cv2.circle(img, (x, y), 6, (255, 255, 255), 1)

            # 绘制关键点之间的连接（如果有多个关键点）
            if num_kpts >= 2 and show_arrow:
                # 对于 arrow 类别，绘制两个关键点之间的连线
                if class_name == "arrow" and num_kpts >= 2:
                    pt1 = kpts[0].astype(int)
                    pt2 = kpts[1].astype(int)
                    cv2.line(img, tuple(pt1), tuple(pt2), (0, 255, 255), 2)
                    # 绘制箭头指向
                    cv2.arrowedLine(img, tuple(pt1), tuple(pt2), (0, 0, 255), 2, tipLength=0.3)

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

            # 文本：关键点数量
            cv2.putText(
                img,
                f"kpts: {num_kpts}",
                (cx - 40, cy + 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 0),
                1,
                cv2.LINE_AA,
            )

            logger.info(
                f"[{i}] cls={class_name}, conf={conf:.3f}, "
                f"num_kpts={num_kpts}, cx={cx}, cy={cy}"
            )

    return img

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="YOLO 模型推理和可视化")
    parser.add_argument("--model", type=str, required=True, help="YOLO 模型路径")
    parser.add_argument("--task", type=str, required=True, help="任务类型: obb, pose, detect")
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
    logger.info(f"加载模型: {model_path}, 任务类型: {args.task}")
    model = YOLO(str(model_path))

    # 读取图像
    logger.info(f"读取图像: {image_path}")
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"无法读取图像文件: {image_path}")

    # 进行推理
    logger.info("开始推理...")
    results = model(image, conf=0.25, save=True, save_txt=False)
    logger.info(results)

    # 可视化结果
    logger.info("可视化结果...")
    if args.task == 'pose':
        vis_image = VisualizePoseResults(image, results, show_arrow=True)
        window_name = "Pose Detection Results"
    elif args.task == 'obb':
        vis_image = VisualizeObbResults(image, results, show_arrow=True)
        window_name = "OBB Detection Results"
    else:
        logger.warning("检测任务暂不支持自定义可视化，使用默认 plot 方法")
        vis_image = results[0].plot()
        window_name = "Detection Results"

    # 保存或显示结果
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), vis_image)
        logger.info(f"结果已保存到: {output_path}")
    else:
        logger.info("按任意键关闭窗口...")
        cv2.imshow(window_name, vis_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

