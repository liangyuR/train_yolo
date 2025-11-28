# dataset_preprocessor.py
from dataclasses import dataclass
import json
import random
import shutil
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import yaml


@dataclass
class DatasetPreprocessorConfig:
    input_dir: str
    output_dir: str
    random_seed: int = 42
    train_ratio: float = 0.7
    val_ratio: float = 0.2
    test_ratio: float = 0.1


class DatasetPreprocessor:
    def __init__(self, config: DatasetPreprocessorConfig):
        self.config = config

        self.input_dir = Path(self.config.input_dir)
        self.output_dir = Path(self.config.output_dir)
        self.random_seed = self.config.random_seed

        self.train_ratio = self.config.train_ratio
        self.val_ratio = self.config.val_ratio
        self.test_ratio = self.config.test_ratio

        self.classes = []

        self.train_dir = self.output_dir / "train"
        self.val_dir = self.output_dir / "val"
        self.test_dir = self.output_dir / "test"

        self._load_classes()

    def _load_classes(self):
        """
        仅支持 data.yaml，不再支持 notes.json
        """
        yaml_path = self.input_dir / "data.yaml"
        if not yaml_path.exists():
            raise FileNotFoundError(f"缺少 data.yaml：{yaml_path}")

        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        names = data.get("names")

        if isinstance(names, dict):
            self.classes = [name for _, name in sorted(names.items(), key=lambda kv: int(kv[0]))]
        elif isinstance(names, list):
            self.classes = names
        else:
            raise ValueError("data.yaml 中 names 格式不正确")

        if not self.classes:
            raise ValueError("data.yaml 中 names 为空")

    def _get_image_paths(self):
        images_dir = self.input_dir / "images"
        patterns = ["*.png", "*.jpg", "*.jpeg", "*.bmp"]
        image_paths = []
        for p in patterns:
            image_paths.extend(images_dir.rglob(p))
        return sorted(image_paths)

    def _check_dataset(self, N: int):
        """检查极端情况"""
        if N < 3:
            raise ValueError(f"图片数量过少（{N} 张），无法进行 train/val/test 划分。")

        if self.train_ratio <= 0 or self.val_ratio < 0 or self.test_ratio < 0:
            raise ValueError("train/val/test 比例必须为正数")

        total = self.train_ratio + self.val_ratio + self.test_ratio
        if total <= 0:
            raise ValueError("train/val/test 比例总和必须大于 0")

    def _split_dataset(self, image_paths: List[Path]):
        random.seed(self.random_seed)
        random.shuffle(image_paths)
        N = len(image_paths)

        self._check_dataset(N)

        total = self.train_ratio + self.val_ratio + self.test_ratio
        tr = self.train_ratio / total
        vr = self.val_ratio / total

        n_train = int(N * tr)
        n_val = int(N * vr)

        splits = {
            "train": image_paths[:n_train],
            "val": image_paths[n_train:n_train + n_val],
            "test": image_paths[n_train + n_val:]
        }
        return splits

    def _process_split(self, split_name: str, images: List[Path], start_index: int):
        target = self.output_dir / split_name

        for i, img_path in enumerate(images):
            global_idx = start_index + i  # 保证全局唯一编号
            new_name = f"{split_name}_{global_idx:05d}{img_path.suffix}"

            shutil.copy(img_path, target / "images" / new_name)

            # 标签路径定位
            try:
                rel = img_path.relative_to(self.input_dir / "images")
                label_path = self.input_dir / "labels" / rel.with_suffix(".txt")
            except ValueError:
                label_path = self.input_dir / "labels" / f"{img_path.stem}.txt"

            if label_path.exists():
                new_label_name = f"{Path(new_name).stem}.txt"
                (target / "labels" / new_label_name).write_text(
                    label_path.read_text(encoding="utf-8"),
                    encoding="utf-8"
                )
            else:
                print(f"[WARNING] 标签不存在: {label_path}")

        return start_index + len(images)

    def _generate_yaml(self):
        yaml_path = self.output_dir / "dataset.yaml"
        data = {
            "path": str(self.output_dir).replace("\\", "/"),
            "train": str(self.train_dir / "images").replace("\\", "/"),
            "val": str(self.val_dir / "images").replace("\\", "/"),
            "test": str(self.test_dir / "images").replace("\\", "/"),
            "names": self.classes
        }
        yaml.safe_dump(data, open(yaml_path, "w", encoding="utf-8"), allow_unicode=True)

    def prepare(self):
        for d in [self.train_dir, self.val_dir, self.test_dir]:
            (d / "images").mkdir(parents=True, exist_ok=True)
            (d / "labels").mkdir(parents=True, exist_ok=True)

        images = self._get_image_paths()
        splits = self._split_dataset(images)

        global_index = 0
        for split_name, subset in splits.items():
            global_index = self._process_split(split_name, subset, global_index)

        self._generate_yaml()


if __name__ == "__main__":
    config = DatasetPreprocessorConfig(
        input_dir="C:/Users/11601/project/wot_ai/data/origin_data",
        output_dir="C:/Users/11601/project/wot_ai/data/datasets/processed/minimap"
    )
    processor = DatasetPreprocessor(config)
    processor.prepare()
