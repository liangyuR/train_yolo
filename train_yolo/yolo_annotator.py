#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO 预标注工具

功能：
1. 加载 YOLO 模型
2. 批量处理图片文件夹
3. 生成 YOLO 格式的标注文件
4. 输出完整的数据集结构（images/、labels/、dataset.yaml）
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml
from loguru import logger
from ultralytics import YOLO


@dataclass
class YOLOAnnotatorConfig:
    """YOLO 预标注配置"""
    model_path: str
    image_dir: str
    output_dir: str
    conf_threshold: float = 0.25


class YOLOAnnotator:
    """YOLO 预标注类

    用于批量处理图片文件夹，使用 YOLO 模型进行预标注，
    输出标准 YOLO 格式的标注数据集。
    """

    IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")

    def __init__(self, config: YOLOAnnotatorConfig):
        """初始化 YOLOAnnotator

        Args:
            config: 预标注配置
        """
        self.config_ = config
        self.model_path_ = Path(config.model_path)
        self.image_dir_ = Path(config.image_dir)
        self.output_dir_ = Path(config.output_dir)
        self.conf_threshold_ = config.conf_threshold

        self.model_: Optional[YOLO] = None
        self.class_names_: Dict[int, str] = {}

        self._loadModel()

    def _loadModel(self) -> None:
        """加载 YOLO 模型并获取类别信息"""
        if not self.model_path_.exists():
            raise FileNotFoundError(f"模型文件不存在: {self.model_path_}")

        logger.info(f"加载模型: {self.model_path_}")
        self.model_ = YOLO(str(self.model_path_))
        self.class_names_ = self.model_.names

        logger.info(f"模型任务类型: {getattr(self.model_, 'task', 'unknown')}")
        logger.info(f"类别数: {len(self.class_names_)}")
        logger.info(f"类别映射: {self.class_names_}")

    def _getImagePaths(self) -> List[Path]:
        """获取图片文件夹中的所有图片路径

        Returns:
            图片路径列表
        """
        if not self.image_dir_.exists():
            raise FileNotFoundError(f"图片文件夹不存在: {self.image_dir_}")

        image_paths = []
        for ext in self.IMAGE_EXTENSIONS:
            image_paths.extend(self.image_dir_.rglob(f"*{ext}"))
            image_paths.extend(self.image_dir_.rglob(f"*{ext.upper()}"))

        return sorted(set(image_paths))

    def _processImage(self, image_path: Path) -> Tuple[np.ndarray, Any]:
        """处理单张图片，返回图片数据和检测结果

        Args:
            image_path: 图片路径

        Returns:
            (图片数据, 检测结果)
        """
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"无法读取图像文件: {image_path}")

        results = self.model_(image, conf=self.conf_threshold_, verbose=False)
        return image, results[0]

    def _convertToYoloFormat(
        self, boxes: Any, image_shape: Tuple[int, int, int]
    ) -> List[str]:
        """将检测结果转换为 YOLO 标注格式

        Args:
            boxes: 检测框对象（ultralytics.engine.results.Boxes）
            image_shape: 图片形状 (H, W, C)

        Returns:
            YOLO 格式的标注行列表
        """
        if boxes is None or boxes.shape[0] == 0:
            return []

        h, w = image_shape[:2]
        annotations = []

        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)

        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i]
            class_id = cls[i]

            # 转换为归一化中心点坐标
            x_center = (x1 + x2) / 2.0 / w
            y_center = (y1 + y2) / 2.0 / h
            width = (x2 - x1) / w
            height = (y2 - y1) / h

            # 确保坐标在 [0, 1] 范围内
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            width = max(0.0, min(1.0, width))
            height = max(0.0, min(1.0, height))

            annotation = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            annotations.append(annotation)

        return annotations

    def _saveAnnotation(self, label_path: Path, annotations: List[str]) -> None:
        """保存标注文件

        Args:
            label_path: 标注文件路径
            annotations: 标注行列表
        """
        content = "\n".join(annotations)
        label_path.write_text(content, encoding="utf-8")

    def _generateTrainTxt(self, image_names: List[str]) -> None:
        """生成 Train.txt 文件

        Args:
            image_names: 图片文件名列表
        """
        train_txt_path = self.output_dir_ / "Train.txt"
        lines = [f"images/Train/{name}" for name in sorted(image_names)]
        train_txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logger.info(f"已生成 Train.txt: {train_txt_path}")

    def _generateDatasetYaml(self) -> None:
        """生成 data.yaml 文件"""
        yaml_path = self.output_dir_ / "data.yaml"

        # 将类别字典转换为字典格式
        names = {i: self.class_names_[i] for i in sorted(self.class_names_.keys())}

        data = {
            "path": ".",
            "Train": "Train.txt",
            "names": names,
        }

        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, allow_unicode=True)

        logger.info(f"已生成 data.yaml: {yaml_path}")

    def _setupOutputDirs(self) -> None:
        """创建输出目录结构"""
        images_dir = self.output_dir_ / "images" / "Train"
        labels_dir = self.output_dir_ / "labels" / "Train"

        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"输出目录: {self.output_dir_}")

    def Annotate(self) -> None:
        """主方法：批量处理所有图片并生成标注"""
        # 获取图片列表
        image_paths = self._getImagePaths()
        if not image_paths:
            raise ValueError(f"图片文件夹为空: {self.image_dir_}")

        logger.info(f"找到 {len(image_paths)} 张图片")

        # 创建输出目录
        self._setupOutputDirs()

        images_dir = self.output_dir_  / "images" / "Train"
        labels_dir = self.output_dir_  / "labels" / "Train"

        success_count = 0
        fail_count = 0
        empty_count = 0
        processed_image_names: List[str] = []

        for i, image_path in enumerate(image_paths):
            try:
                # 处理图片
                image, result = self._processImage(image_path)

                # 转换为 YOLO 格式
                annotations = []
                if hasattr(result, "boxes") and result.boxes is not None:
                    annotations = self._convertToYoloFormat(result.boxes, image.shape)

                # 复制图片
                dest_image_path = images_dir / image_path.name
                shutil.copy(image_path, dest_image_path)
                processed_image_names.append(image_path.name)

                # 保存标注
                label_name = image_path.stem + ".txt"
                label_path = labels_dir / label_name
                self._saveAnnotation(label_path, annotations)

                if annotations:
                    success_count += 1
                else:
                    empty_count += 1

                if (i + 1) % 100 == 0:
                    logger.info(f"进度: {i + 1}/{len(image_paths)}")

            except Exception as e:
                logger.error(f"处理失败 [{image_path.name}]: {e}")
                fail_count += 1

        # 生成 Train.txt 和 data.yaml
        self._generateTrainTxt(processed_image_names)
        self._generateDatasetYaml()

        # 输出统计
        logger.info("=" * 50)
        logger.info("处理完成！")
        logger.info(f"  - 成功（有检测）: {success_count}")
        logger.info(f"  - 成功（无检测）: {empty_count}")
        logger.info(f"  - 失败: {fail_count}")
        logger.info(f"  - 总计: {len(image_paths)}")
        logger.info(f"输出目录: {self.output_dir_}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="YOLO 预标注工具")
    parser.add_argument("--model", type=str, required=True, help="YOLO 模型路径")
    parser.add_argument("--images", type=str, required=True, help="输入图片文件夹")
    parser.add_argument("--output", type=str, required=True, help="输出目录")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值（默认: 0.25）")
    args = parser.parse_args()

    config = YOLOAnnotatorConfig(
        model_path=args.model,
        image_dir=args.images,
        output_dir=args.output,
        conf_threshold=args.conf,
    )

    annotator = YOLOAnnotator(config)
    annotator.Annotate()

