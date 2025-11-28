#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一键训练 YOLO 小地图检测模型
自动检查数据集结构、生成 dataset.yaml、启动训练并输出结果
"""

from pathlib import Path

import yaml
from loguru import logger
from ultralytics import YOLO

from dataset_preprocessor import DatasetPreprocessor, DatasetPreprocessorConfig

def TrainYOLO(config: dict) -> bool:
    """执行 YOLO 模型训练。

    Args:
        config: 训练配置

    Returns:
        是否成功
    """
    # 检查数据集结构
    dataset_cfg = config["dataset"]
    dataset_dir = Path(dataset_cfg["output_dir"])

    # 检查 dataset.yaml
    yaml_path = dataset_dir / "dataset.yaml"
    if not yaml_path.exists():
        logger.error("dataset.yaml 不存在，请先准备数据集")
        return False
    logger.info(f"使用现有的 dataset.yaml: {yaml_path}")

    # 加载模型
    model_path = Path(config["model"]["path"])
    logger.info(f"加载模型: {model_path}")
    model = YOLO(str(model_path))

    # 准备训练参数（核心必选部分）
    training_cfg = config["training"]
    output_cfg = config["output"]
    model_cfg = config["model"]

    train_params: dict = {
        "data": str(yaml_path),
        "imgsz": model_cfg["imgsz"],
        "epochs": training_cfg["epochs"],
        "batch": training_cfg["batch"],
        "workers": training_cfg["workers"],
        "device": model_cfg["device"],
        "project": output_cfg["project"],
        "name": output_cfg["name"],
        "save": output_cfg["save"],
        "save_period": output_cfg["save_period"],
        "val": output_cfg["val"],
        "verbose": True,
    }

    # 可选：优化器和学习率相关参数（如果在配置中显式给出）
    opt_keys = ["optimizer", "lr0", "lrf", "warmup_epochs", "patience", "dropout"]
    for key in opt_keys:
        if key in training_cfg:
            train_params[key] = training_cfg[key]

    # 损失权重配置（如果存在）
    loss_config = training_cfg.get("loss")
    if isinstance(loss_config, dict):
        for key in ("cls", "box", "dfl"):
            if key in loss_config:
                train_params[key] = loss_config[key]
        logger.info(
            "损失权重配置: cls={cls}, box={box}, dfl={dfl}".format(
                cls=loss_config.get("cls", 0.5),
                box=loss_config.get("box", 7.5),
                dfl=loss_config.get("dfl", 1.5),
            )
        )

    # 数据增强参数
    if training_cfg.get("augment", True):
        aug_cfg = config.get("augmentation", {})
        train_params.update(
            {
                "augment": True,
                "hsv_h": aug_cfg.get("hsv_h", 0.015),
                "hsv_s": aug_cfg.get("hsv_s", 0.7),
                "hsv_v": aug_cfg.get("hsv_v", 0.4),
                "degrees": aug_cfg.get("degrees", 5),
                "scale": aug_cfg.get("scale", 0.5),
                "translate": aug_cfg.get("translate", 0.1),
                "fliplr": aug_cfg.get("fliplr", 0.2),
                "mosaic": training_cfg.get("mosaic", 1.0),
                "copy_paste": training_cfg.get("copy_paste", 0.1),
                "mixup": training_cfg.get("mixup", 0.0),
                "close_mosaic": training_cfg.get("close_mosaic", 0),
            }
        )

    # 打印训练参数摘要
    logger.info("开始训练 YOLO 模型...")
    logger.info(
        "训练参数: epochs={epochs}, batch={batch}, imgsz={imgsz}".format(
            epochs=train_params["epochs"],
            batch=train_params["batch"],
            imgsz=train_params["imgsz"],
        )
    )
    if "optimizer" in train_params:
        logger.info(
            "优化器: {opt}, lr0={lr0}, lrf={lrf}".format(
                opt=train_params["optimizer"],
                lr0=train_params.get("lr0", "default"),
                lrf=train_params.get("lrf", "default"),
            )
        )
    if training_cfg.get("augment", True):
        logger.info(
            "数据增强: mosaic={mosaic}, copy_paste={cp}, mixup={mixup}, close_mosaic={cm}".format(
                mosaic=train_params.get("mosaic", 1.0),
                cp=train_params.get("copy_paste", 0.1),
                mixup=train_params.get("mixup", 0.0),
                cm=train_params.get("close_mosaic", 0),
            )
        )

    # 开始训练
    try:
        results = model.train(**train_params)

        logger.info("\n" + "=" * 60)
        logger.info("训练完成！结果摘要：")
        if hasattr(results, "save_dir"):
            logger.info(f"训练日志保存目录: {results.save_dir}")
            best_model = Path(results.save_dir) / "weights" / "best.pt"
            if best_model.exists():
                logger.info(f"最佳模型路径: {best_model}")
        logger.info("=" * 60)

        return True
    except Exception as e:  # noqa: BLE001
        logger.error(f"训练过程出错: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """主函数。"""
    # 加载配置（使用统一路径解析）
    config_path = Path(__file__).parent / "train_config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    dataset_config = DatasetPreprocessorConfig(
        input_dir=config["dataset"]["source_dir"],
        output_dir=config["dataset"]["output_dir"],
        random_seed=config["dataset"]["random_seed"],
        train_ratio=config["dataset"]["train_ratio"],
        val_ratio=config["dataset"]["val_ratio"],
        test_ratio=config["dataset"]["test_ratio"],
    )
    processor = DatasetPreprocessor(dataset_config)
    processor.prepare()

    success = TrainYOLO(config)
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()
