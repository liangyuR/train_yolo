import json
import random
import shutil
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import yaml

class DatasetPreprocessor:
    def __init__(self, input_dir: str, output_dir: str, task_type: str = "detect", random_seed: int = 42, train_ratio: float = 0.7, val_ratio: float = 0.2, test_ratio: float = 0.1, kpt_shape: Optional[Sequence[int]] = None):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.task_type = task_type.lower()
        self.random_seed = random_seed
        self.notes_json = self.input_dir / "notes.json"
        self.kpt_shape: Optional[Tuple[int, int]] = tuple(kpt_shape) if kpt_shape else None
        
        # ratios
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        self.classes = []

        # output structure
        self.train_dir = self.output_dir / "train"
        self.val_dir = self.output_dir / "val"
        self.test_dir = self.output_dir / "test"

        self._load_classes()
        if self.task_type == "pose" and not self.kpt_shape:
            raise ValueError("Pose 任务必须提供 kpt_shape (e.g. [17, 3])")

    def _load_classes(self):
        if self.notes_json.exists():
            data = json.loads(self.notes_json.read_text(encoding="utf-8"))
            self.classes = [c["name"] for c in data.get("categories", [])]
            if not self.classes:
                raise ValueError(f"{self.notes_json} 中未找到 categories")
            return

        yaml_path = self.input_dir / "data.yaml"
        if yaml_path.exists():
            data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
            names = data.get("names")
            if isinstance(names, dict):
                # sort by numeric key to保证类别顺序
                self.classes = [name for _, name in sorted(names.items(), key=lambda kv: int(kv[0]))]
            elif isinstance(names, list):
                self.classes = names
            else:
                raise ValueError(f"{yaml_path} 中的 names 字段为空或格式不正确")
            if not self.classes:
                raise ValueError(f"{yaml_path} 中 names 为空")
            return

        raise FileNotFoundError(f"{self.notes_json} not found 且 {yaml_path} 不存在")

    def _get_image_paths(self):
        images_dir = self.input_dir / "images"
        # 递归查找所有图像文件
        image_paths = []
        for pattern in ["*.png", "*.jpg", "*.jpeg", "*.bmp"]:
            image_paths.extend(images_dir.rglob(pattern))
        return sorted(image_paths)

    def _split_dataset(self, image_paths: List[Path]):
        random.seed(self.random_seed)
        random.shuffle(image_paths)
        N = len(image_paths)

        # ensure ratio sum is 1
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

    def _process_split(self, split_name: str, images: List[Path]):
        target = self.output_dir / split_name
        for idx, img_path in enumerate(images):
            new_name = f"img_{idx:05d}{img_path.suffix}"
            shutil.copy(img_path, target / "images" / new_name)

            # 计算相对路径，保持子目录结构
            try:
                rel_path = img_path.relative_to(self.input_dir / "images")
                label_path = self.input_dir / "labels" / rel_path.with_suffix(".txt")
            except ValueError:
                # 如果无法计算相对路径，回退到原来的方法
                label_path = self.input_dir / "labels" / f"{img_path.stem}.txt"
            
            if label_path.exists():
                label_data = label_path.read_text(encoding="utf-8")
                new_label_name = f"{Path(new_name).stem}.txt"
                (target / "labels" / new_label_name).write_text(label_data, encoding="utf-8")
            else:
                print(f"[WARNING] 标签文件不存在: {label_path}")

    def _generate_yaml(self):
        yaml_path = self.output_dir / "dataset.yaml"
        data = {
            "path": str(self.output_dir),
            "train": str(self.train_dir / "images"),
            "val": str(self.val_dir / "images"),
            "test": str(self.test_dir / "images"),
            "names": self.classes
        }
        if self.task_type == "pose":
            data["task"] = "pose"
            data["kpt_shape"] = list(self.kpt_shape or [])
        with open(yaml_path, "w", encoding="utf-8") as f:
            for k, v in data.items():
                if isinstance(v, list):
                    f.write(f"{k}:\n")
                    for item in v:
                        f.write(f"  - {item}\n")
                else:
                    f.write(f"{k}: {v}\n")

    def prepare(self):
        for d in [self.train_dir, self.val_dir, self.test_dir]:
            (d / "images").mkdir(parents=True, exist_ok=True)
            (d / "labels").mkdir(parents=True, exist_ok=True)

        images = self._get_image_paths()
        splits = self._split_dataset(images)

        for name, subset in splits.items():
            self._process_split(name, subset)

        self._generate_yaml()
        print(f"[INFO] Dataset prepared successfully for task={self.task_type}")

if __name__ == "__main__":
    processor = DatasetPreprocessor("C:/Users/11601/project/wot_ai/data/origin_data", "C:/Users/11601/project/wot_ai/data/datasets/processed/minimap")
    processor.prepare()