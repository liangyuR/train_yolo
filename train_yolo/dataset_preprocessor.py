import shutil
import json
from pathlib import Path
from typing import List
import random

class DatasetPreprocessor:
    def __init__(self, input_dir: str, output_dir: str, task_type: str = "detect", random_seed: int = 42, train_ratio: float = 0.7, val_ratio: float = 0.2, test_ratio: float = 0.1):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.task_type = task_type
        self.random_seed = random_seed
        self.notes_json = self.input_dir / "notes.json"
        
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

    def _load_classes(self):
        if not self.notes_json.exists():
            raise FileNotFoundError(f"{self.notes_json} not found")
        data = json.loads(self.notes_json.read_text(encoding="utf-8"))
        self.classes = [c["name"] for c in data.get("categories", [])]

    def _get_image_paths(self):
        images_dir = self.input_dir / "images"
        return sorted(images_dir.glob("*.*"))

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

            label_path = self.input_dir / "labels" / f"{img_path.stem}.txt"
            if label_path.exists():
                label_data = label_path.read_text(encoding="utf-8")
                new_label_name = f"{Path(new_name).stem}.txt"
                (target / "labels" / new_label_name).write_text(label_data, encoding="utf-8")

    def _generate_yaml(self):
        yaml_path = self.output_dir / "dataset.yaml"
        data = {
            "path": str(self.output_dir),
            "train": str(self.train_dir / "images"),
            "val": str(self.val_dir / "images"),
            "test": str(self.test_dir / "images"),
            "names": self.classes
        }
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
        print("[INFO] Dataset prepared successfully")

if __name__ == "__main__":
    processor = DatasetPreprocessor("C:/Users/11601/project/wot_ai/data/origin_data", "C:/Users/11601/project/wot_ai/data/datasets/processed/minimap")
    processor.prepare()