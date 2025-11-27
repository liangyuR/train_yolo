#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Polygon -> OBB 转换（使用内角识别箭头尖端）

输入：
    每一行格式类似（你的示例）：
        class x1 y1 x2 y2 x3 y3 ... xN yN
    其中 x、y 为归一化坐标（相对整张图片宽高，范围 [0, 1]）。

输出（OBB 格式）：
    class x1 y1 x2 y2 x3 y3 x4 y4
    - x1, y1, x2, y2, x3, y3, x4, y4: 归一化的四个角点坐标（范围 [0, 1]）
    - 使用内角识别箭头尖端，确定旋转框的朝向

方法：
    - 计算每个顶点的内角，最小内角对应的顶点就是箭头尖端
    - 其他点的平均值作为尾部中心
    - 从尾部中心指向尖端的方向就是箭头方向
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable, Tuple

import cv2
import numpy as np


def _ArrowAngleFromPolygon(points: np.ndarray) -> Tuple[float, float, int]:
    """从多边形点集识别箭头方向。

    Args:
        points: (N, 2) ndarray，图像坐标即可（归一化/像素都行，比例不影响角度）

    Returns:
        angle_rad: float, atan2(dy, dx)，图像坐标系角度（弧度）
        angle_deg: float, 角度制
        tip_idx: int, 识别出的箭头尖端索引
    """
    pts = np.asarray(points, dtype=float)
    n = pts.shape[0]
    if n < 3:
        raise ValueError("需要至少 3 个点")

    # 1. 计算每个顶点的内角
    angles = []
    for i in range(n):
        p_prev = pts[(i - 1) % n]
        p = pts[i]
        p_next = pts[(i + 1) % n]

        v1 = p_prev - p
        v2 = p_next - p
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 == 0 or n2 == 0:
            ang = np.pi  # 退化，先当成钝角
        else:
            cosang = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
            ang = np.arccos(cosang)
        angles.append(ang)

    # 2. 最小内角 = 箭头尖端
    tip_idx = int(np.argmin(angles))
    tip = pts[tip_idx]

    # 3. 其它点的平均 = 尾部中心
    tail = pts[np.arange(n) != tip_idx].mean(axis=0)

    # 4. 尾 -> 尖 的方向角
    dir_vec = tip - tail
    angle_rad = np.arctan2(dir_vec[1], dir_vec[0])
    angle_deg = float(np.degrees(angle_rad))

    return angle_rad, angle_deg, tip_idx


def _parse_yolo_polygon_line(line: str) -> Tuple[int, np.ndarray]:
    """解析一行 YOLO polygon 标注：class x1 y1 x2 y2 ... xN yN

    返回：
        class_id: int
        pts_norm: (N, 2) ndarray，归一化坐标
    """
    parts = line.strip().split()
    if not parts:
        raise ValueError("空行无法解析")

    class_id = int(float(parts[0]))
    coords = list(map(float, parts[1:]))
    if len(coords) % 2 != 0 or len(coords) < 6:
        raise ValueError(f"坐标数量非法，至少需要 3 个点(6 个数)，当前: {len(coords)}")

    pts = np.array(list(zip(coords[0::2], coords[1::2])), dtype=np.float32)  # (N, 2)
    return class_id, pts


def PolygonToObb(
    pts_norm: np.ndarray,
    img_w: int,
    img_h: int,
) -> Tuple[float, float, float, float, float, float, float, float]:
    """使用内角识别箭头尖端的方法计算 OBB 四个角点。

    Ultralytics YOLO OBB 格式：class x1 y1 x2 y2 x3 y3 x4 y4（四个角点坐标）

    步骤：
        1) 所有点还原到像素坐标。
        2) 用 minAreaRect 拿中心 + 宽高（w,h）。
        3) 计算每个顶点的内角，最小内角对应的顶点就是箭头尖端。
        4) 剩余点求均值作为 tail（尾部大致中心）。
        5) 方向向量 tail -> tip，得到 angle。
        6) 根据 (cx, cy, w, h, angle) 计算四个角点坐标。

    Args:
        pts_norm: (N, 2) 归一化坐标点集
        img_w:   原始图像宽度（像素）
        img_h:   原始图像高度（像素）

    Returns:
        x1, y1, x2, y2, x3, y3, x4, y4: 归一化的四个角点坐标
    """
    if pts_norm.ndim != 2 or pts_norm.shape[1] != 2:
        raise ValueError(f"pts_norm 形状非法，期望 (N, 2)，得到 {pts_norm.shape}")

    # 还原到像素坐标
    pts_px = np.empty_like(pts_norm, dtype=np.float32)
    pts_px[:, 0] = pts_norm[:, 0] * img_w
    pts_px[:, 1] = pts_norm[:, 1] * img_h

    # 外接框（中心 + 尺寸）
    rect = cv2.minAreaRect(pts_px)
    (cx_rect, cy_rect), (w, h), angle_deg_rect = rect

    # 使用内角识别箭头尖端，获取方向角度
    _, angle_deg, tip_idx = _ArrowAngleFromPolygon(pts_px)

    # 使用方向角度和 minAreaRect 的尺寸构建旋转矩形
    # 注意：angle_deg 是 tail -> tip 的方向，用于确定旋转框的朝向
    rect_with_angle = ((cx_rect, cy_rect), (w, h), angle_deg)

    # 计算四个角点（像素坐标）
    box_points = cv2.boxPoints(rect_with_angle)  # (4, 2)

    # 归一化角点坐标
    box_points_norm = box_points.copy()
    box_points_norm[:, 0] = box_points[:, 0] / img_w
    box_points_norm[:, 1] = box_points[:, 1] / img_h

    # 返回四个角点坐标（归一化）
    return (
        float(box_points_norm[0, 0]), float(box_points_norm[0, 1]),  # x1, y1
        float(box_points_norm[1, 0]), float(box_points_norm[1, 1]),  # x2, y2
        float(box_points_norm[2, 0]), float(box_points_norm[2, 1]),  # x3, y3
        float(box_points_norm[3, 0]), float(box_points_norm[3, 1]),  # x4, y4
    )


def PolygonLineToObbLine(line: str, img_w: int, img_h: int) -> str:
    """把一行 polygon 标注转换成一行 OBB 标注（字符串）。

    Ultralytics YOLO OBB 格式：class x1 y1 x2 y2 x3 y3 x4 y4

    输入示例：
        "1 0.3396 0.2171 0.3375 0.2293 ..."  # 来自你的 frame_000042.txt

    输出示例：
        "1 0.780811 0.743961 0.782371 0.74686 0.777691 0.752174 0.776131 0.749758"
    """
    class_id, pts_norm = _parse_yolo_polygon_line(line)
    x1, y1, x2, y2, x3, y3, x4, y4 = PolygonToObb(pts_norm, img_w=img_w, img_h=img_h)

    return f"{class_id} {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {x3:.6f} {y3:.6f} {x4:.6f} {y4:.6f}"


def _GetImageSize(image_path: Path) -> Tuple[int, int]:
    """从图像文件读取尺寸。

    Args:
        image_path: 图像文件路径

    Returns:
        img_w, img_h: 图像宽度和高度（像素）
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"图像文件不存在: {image_path}")

    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"无法读取图像文件: {image_path}")

    h, w = img.shape[:2]
    return int(w), int(h)


def _FindImageFile(label_path: Path, images_dir: Path) -> Path:
    """根据标签文件名查找对应的图像文件。

    Args:
        label_path: 标签文件路径
        images_dir: 图像目录路径

    Returns:
        图像文件路径

    Raises:
        FileNotFoundError: 如果找不到对应的图像文件
    """
    label_stem = label_path.stem
    images_dir = Path(images_dir)

    # 支持的图像格式
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']

    for ext in image_extensions:
        image_path = images_dir / f"{label_stem}{ext}"
        if image_path.exists():
            return image_path

    raise FileNotFoundError(
        f"找不到标签文件对应的图像: {label_path.name}\n"
        f"在目录 {images_dir} 中查找了扩展名: {', '.join(image_extensions)}"
    )


def ConvertLabelFile(
    src_path: Path,
    dst_path: Path,
    img_w: int,
    img_h: int,
) -> None:
    """把一个 polygon label txt 转换为 OBB label txt。

    Args:
        src_path: 原始 polygon 标注文件
        dst_path: 输出 OBB 标注文件
        img_w:    对应图片宽度
        img_h:    对应图片高度
    """
    src_path = Path(src_path)
    dst_path = Path(dst_path)

    if not src_path.exists():
        raise FileNotFoundError(f"源文件不存在: {src_path}")

    if not src_path.is_file():
        raise ValueError(f"源路径不是文件: {src_path}")

    lines = src_path.read_text(encoding="utf-8").splitlines()
    obb_lines = []
    for line_num, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obb_line = PolygonLineToObbLine(line, img_w=img_w, img_h=img_h)
            obb_lines.append(obb_line)
        except Exception as e:
            raise ValueError(f"处理第 {line_num} 行时出错: {e}\n行内容: {line}")

    # 确保输出目录存在
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dst_path.write_text("\n".join(obb_lines) + "\n", encoding="utf-8")


def _SearchFiles(input_dir: Path) -> Tuple[dict, dict]:
    """在输入目录中搜索所有 .txt 和 .png 文件（递归搜索所有子目录）。

    Args:
        input_dir: 输入目录路径

    Returns:
        label_dict: {文件名(不含扩展名): 标签文件路径}
        image_dict: {文件名(不含扩展名): 图像文件路径}
    """
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")

    label_dict = {}
    image_dict = {}

    # 搜索 .txt 文件（递归搜索所有子目录）
    print(f"正在搜索 .txt 文件（递归搜索 {input_dir} 及其所有子目录）...")
    txt_files = list(input_dir.rglob("*.txt"))
    print(f"找到 {len(txt_files)} 个 .txt 文件")
    for txt_file in txt_files:
        stem = txt_file.stem
        if stem not in label_dict:
            label_dict[stem] = txt_file
            print(f"  ✓ 标签: {txt_file.relative_to(input_dir)}")
        else:
            print(f"  ⚠ 重复标签（已跳过）: {txt_file.relative_to(input_dir)}")

    # 搜索 .png 文件（递归搜索所有子目录）
    print(f"正在搜索 .png 文件（递归搜索 {input_dir} 及其所有子目录）...")
    png_files = list(input_dir.rglob("*.png"))
    print(f"找到 {len(png_files)} 个 .png 文件")
    for png_file in png_files:
        stem = png_file.stem
        if stem not in image_dict:
            image_dict[stem] = png_file
            print(f"  ✓ 图像: {png_file.relative_to(input_dir)}")
        else:
            print(f"  ⚠ 重复图像（已跳过）: {png_file.relative_to(input_dir)}")

    return label_dict, image_dict


def ConvertLabelDirectory(
    input_dir: Path,
    output_dir: Path,
    test_mode: bool = False,
) -> None:
    """批量转换目录下的所有 polygon 标签文件为 OBB 格式。

    自动搜索 input_dir 下的所有 .txt 和 .png 文件，根据文件名匹配，
    转换后保存到 output_dir/images/ 和 output_dir/labels/。

    Args:
        input_dir: 输入目录（包含 .txt 和 .png 文件）
        output_dir: 输出目录（将创建 images/ 和 labels/ 子目录）
        test_mode: 测试模式，只处理第一个匹配的文件对
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")

    # 搜索所有文件
    print(f"正在搜索 {input_dir} 下的文件...")
    label_dict, image_dict = _SearchFiles(input_dir)

    if not label_dict:
        print(f"警告: 在 {input_dir} 中未找到任何 .txt 文件")
        return

    if not image_dict:
        print(f"警告: 在 {input_dir} 中未找到任何 .png 文件")
        return

    # 找到匹配的文件对
    matched_pairs = []
    for stem in label_dict:
        if stem in image_dict:
            matched_pairs.append((stem, label_dict[stem], image_dict[stem]))

    if not matched_pairs:
        print("警告: 未找到匹配的标签和图像文件对")
        return

    if test_mode:
        matched_pairs = matched_pairs[:1]
        print(f"测试模式: 只处理第一个匹配的文件对")

    print(f"找到 {len(matched_pairs)} 个匹配的文件对，开始转换...")

    # 创建输出目录结构
    output_images_dir = output_dir / "images"
    output_labels_dir = output_dir / "labels"
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    error_count = 0
    errors = []

    for stem, label_file, image_file in matched_pairs:
        try:
            # 读取图像尺寸
            img_w, img_h = _GetImageSize(image_file)

            # 复制图像文件到输出目录
            output_image = output_images_dir / image_file.name
            shutil.copy2(image_file, output_image)

            # 转换标签文件
            output_label = output_labels_dir / label_file.name
            ConvertLabelFile(label_file, output_label, img_w, img_h)

            success_count += 1
            print(f"✓ {stem}")

        except Exception as e:
            error_count += 1
            error_msg = f"✗ {stem}: {e}"
            errors.append(error_msg)
            print(error_msg)

    # 输出总结
    print("\n" + "=" * 60)
    print(f"转换完成: 成功 {success_count} 个, 失败 {error_count} 个")
    print(f"输出目录: {output_dir}")
    print(f"  - 图像: {output_images_dir}")
    print(f"  - 标签: {output_labels_dir}")
    if errors:
        print("\n错误详情:")
        for err in errors:
            print(f"  {err}")
    print("=" * 60)


def DebugVisualize(image_path: str, label_path: Path) -> None:
    """可视化：画出 OBB + tail/tip + 朝向箭头。

    - 读取原始图像（用于显示）
    - 从 label_path 读取 polygon 标注
    - 计算 OBB 并显示 tail（蓝点）、tip（红点）和朝向箭头
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"图像文件不存在: {image_path}")

    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"无法读取图像文件: {image_path}")

    label_path = Path(label_path)
    if not label_path.exists():
        raise FileNotFoundError(f"标注文件不存在: {label_path}")

    img_h, img_w = img.shape[:2]
    lines = label_path.read_text(encoding="utf-8").splitlines()

    for line_num, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue

        try:
            class_id, pts_norm = _parse_yolo_polygon_line(line)

            # 还原到像素坐标
            pts_px = np.empty_like(pts_norm, dtype=np.float32)
            pts_px[:, 0] = pts_norm[:, 0] * img_w
            pts_px[:, 1] = pts_norm[:, 1] * img_h

            # 计算 OBB 四个角点
            x1, y1, x2, y2, x3, y3, x4, y4 = PolygonToObb(pts_norm, img_w, img_h)
            
            # 为了可视化，需要从角点重建旋转矩形
            box_points_px = np.array([
                [x1 * img_w, y1 * img_h],
                [x2 * img_w, y2 * img_h],
                [x3 * img_w, y3 * img_h],
                [x4 * img_w, y4 * img_h]
            ], dtype=np.float32)
            rect = cv2.minAreaRect(box_points_px)
            (cx, cy), (w, h), angle_deg = rect
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(img, [box], 0, (0, 255, 0), 2)

            # 使用内角识别 tail / tip
            _, angle_deg_arrow, tip_idx = _ArrowAngleFromPolygon(pts_px)
            tip = pts_px[tip_idx]

            mask = np.ones(len(pts_px), dtype=bool)
            mask[tip_idx] = False
            tail = pts_px[mask].mean(axis=0)

            cv2.circle(img, (int(tail[0]), int(tail[1])), 4, (255, 0, 0), -1)  # tail 蓝点
            cv2.circle(img, (int(tip[0]), int(tip[1])), 4, (0, 0, 255), -1)   # tip 红点

            # tail -> tip 方向箭头
            cv2.arrowedLine(
                img,
                (int(tail[0]), int(tail[1])),
                (int(tip[0]), int(tip[1])),
                (0, 0, 255),
                2,
                tipLength=0.2,
            )

            cv2.putText(
                img,
                f"cls {class_id}, ang {angle_deg_arrow:.1f}",
                (int(cx + 5), int(cy + 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        except Exception as e:
            raise ValueError(f"处理第 {line_num} 行时出错: {e}\n行内容: {line}")

    cv2.imshow("OBB Debug", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Polygon -> OBB (使用内角识别箭头尖端)")
    parser.add_argument("--input_dir", type=str, help="输入目录（包含 .txt 和 .png 文件）")
    parser.add_argument("--output_dir", type=str, help="输出目录（将创建 images/ 和 labels/ 子目录）")
    parser.add_argument("--test", action="store_true", help="测试模式：只处理第一个匹配的文件对")
    parser.add_argument("--visualize", type=str, nargs=2, metavar=("IMAGE", "LABEL"), help="可视化模式：显示 OBB 和朝向 (图像路径 标签路径)")

    args = parser.parse_args()

    if args.visualize:
        image_path, label_path = args.visualize
        DebugVisualize(image_path, Path(label_path))
    else:
        if not args.input_dir or not args.output_dir:
            raise SystemExit("需要提供 input_dir 和 output_dir")
        ConvertLabelDirectory(Path(args.input_dir), Path(args.output_dir), test_mode=args.test)

