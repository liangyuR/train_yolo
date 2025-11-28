#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一键训练 YOLO 小地图检测模型
自动检查数据集结构、生成 dataset.yaml、启动训练并输出结果
"""

import yaml
from pathlib import Path
from ultralytics import YOLO

# 统一导入机制
from loguru import logger
from dataset_preprocessor import DatasetPreprocessor


def _ListImages(image_dir: Path):
    """列出指定目录下常见格式的图像文件。"""
    patterns = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    images = []
    for pattern in patterns:
        images.extend(image_dir.glob(pattern))
    return images


def ValidateObbLabels(labels_dir: Path) -> bool:
    """验证 OBB 标签文件格式。

    Ultralytics YOLO OBB 格式：class x1 y1 x2 y2 x3 y3 x4 y4（9 个值）
    - class: 类别ID（整数）
    - x1, y1, x2, y2, x3, y3, x4, y4: 归一化的四个角点坐标（范围 [0, 1]）

    Args:
        labels_dir: 标签文件目录

    Returns:
        是否所有标签文件格式正确
    """
    label_files = list(labels_dir.glob("*.txt"))
    if not label_files:
        logger.warning(f"在 {labels_dir} 中未找到任何标签文件")
        return True  # 空目录不算错误

    error_count = 0
    for label_file in label_files:
        try:
            lines = label_file.read_text(encoding="utf-8").splitlines()
            for line_num, line in enumerate(lines, start=1):
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) != 9:
                    logger.error(
                        f"{label_file.name}:{line_num} - OBB 格式错误，需要 9 个值（class + 8个坐标），"
                        f"当前 {len(parts)} 个值: {line}"
                    )
                    error_count += 1
                    continue

                try:
                    class_id = int(float(parts[0]))
                    coords = [float(parts[i]) for i in range(1, 9)]

                    # 验证归一化坐标范围
                    for i, coord in enumerate(coords):
                        if not (0 <= coord <= 1):
                            coord_name = ['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4'][i]
                            logger.error(
                                f"{label_file.name}:{line_num} - {coord_name} 超出范围 [0, 1]: {coord}"
                            )
                            error_count += 1

                except ValueError as e:
                    logger.error(
                        f"{label_file.name}:{line_num} - 数值解析错误: {e}, 行内容: {line}"
                    )
                    error_count += 1

        except Exception as e:
            logger.error(f"读取标签文件 {label_file.name} 时出错: {e}")
            error_count += 1

    if error_count > 0:
        logger.error(f"标签验证失败: 发现 {error_count} 个错误")
        return False

    logger.info(f"标签验证通过: {len(label_files)} 个标签文件")
    return True


def CheckDatasetStructure(dataset_dir: Path, task_type: str = "detect") -> bool:
    """
    检查数据集结构是否正确
    
    新的目录结构:
    - dataset_dir/train/images/
    - dataset_dir/train/labels/
    - dataset_dir/val/images/
    - dataset_dir/val/labels/
    - dataset_dir/test/images/ (可选)
    - dataset_dir/test/labels/ (可选)
    
    Args:
        dataset_dir: 数据集目录
    
    Returns:
        是否结构正确
    """
    required_dirs = [
        dataset_dir / "train" / "images",
        dataset_dir / "train" / "labels",
        dataset_dir / "val" / "images",
        dataset_dir / "val" / "labels"
    ]
    
    for dir_path in required_dirs:
        if not dir_path.exists():
            logger.error(f"数据集目录结构不完整，缺少: {dir_path}")
            return False
    
    # 检查是否有文件
    train_images = _ListImages(dataset_dir / "train" / "images")
    val_images = _ListImages(dataset_dir / "val" / "images")
    test_dir = dataset_dir / "test" / "images"
    test_images = _ListImages(test_dir) if test_dir.exists() else []
    
    if len(train_images) == 0:
        logger.error("训练集图像为空")
        return False
    
    if len(val_images) == 0:
        logger.warning("验证集图像为空，建议添加验证数据")
    
    logger.info(f"数据集结构检查通过: 训练集 {len(train_images)} 张, 验证集 {len(val_images)} 张")
    if len(test_images) > 0:
        logger.info(f"测试集 {len(test_images)} 张")

    # 验证标签格式（如果是 OBB 任务）
    if task_type == "obb":
        logger.info("验证 OBB 标签格式...")
        train_labels_dir = dataset_dir / "train" / "labels"
        val_labels_dir = dataset_dir / "val" / "labels"

        if not ValidateObbLabels(train_labels_dir):
            logger.error("训练集标签验证失败")
            return False

        if not ValidateObbLabels(val_labels_dir):
            logger.error("验证集标签验证失败")
            return False

        test_labels_dir = dataset_dir / "test" / "labels"
        if test_labels_dir.exists():
            if not ValidateObbLabels(test_labels_dir):
                logger.error("测试集标签验证失败")
                return False
    elif task_type == "pose":
        train_labels = list((dataset_dir / "train" / "labels").glob("*.txt"))
        val_labels = list((dataset_dir / "val" / "labels").glob("*.txt"))
        logger.info(f"Pose 标签统计: train={len(train_labels)}, val={len(val_labels)}")

    return True

def TrainYOLO(config: dict, prepare_dataset: bool = False) -> bool:
    """
    执行 YOLO 模型训练
    
    Args:
        config: 训练配置
        prepare_dataset: 是否自动准备数据集
    
    Returns:
        是否成功
    """
    # 检查数据集结构
    dataset_cfg = config['dataset']
    dataset_dir = Path(dataset_cfg['output_dir'])
    task_type = dataset_cfg.get('task_type', 'detect').lower()
    if not CheckDatasetStructure(dataset_dir, task_type=task_type):
        logger.error("数据集结构检查失败")
        return False
    
    # 检查 dataset.yaml
    yaml_path = dataset_dir / "dataset.yaml"
    if not yaml_path.exists():
        logger.error("dataset.yaml 不存在，请先准备数据集")
        return False
    else:
        logger.info(f"使用现有的 dataset.yaml: {yaml_path}")

    # 加载模型
    model_path = Path(config['model']['path'])
    logger.info(f"加载模型: {model_path}, 任务类型: {task_type}")

    # 根据任务类型加载模型
    model = YOLO(str(model_path))
    if task_type in ("obb", "pose"):
        expected_task = task_type
        if hasattr(model, "task") and model.task != expected_task:
            logger.warning(f"模型任务类型为 {getattr(model, 'task', 'unknown')}，强制设置为 {expected_task}")
        model.task = expected_task
    
    # 准备训练参数
    train_params = {
        'data': str(yaml_path),
        'imgsz': config['model']['imgsz'],
        'epochs': config['training']['epochs'],
        'batch': config['training']['batch'],
        'workers': config['training']['workers'],
        'device': config['model']['device'],
        'project': config['output']['project'],
        'name': config['output']['name'],
        'save': config['output']['save'],
        'save_period': config['output']['save_period'],
        'val': config['output']['val'],
        'verbose': True
    }
    
    # 特定任务的额外参数
    if task_type in ("obb", "pose"):
        train_params['task'] = task_type
    
    # 优化器和学习率参数
    if 'optimizer' in config['training']:
        train_params['optimizer'] = config['training']['optimizer']
    if 'lr0' in config['training']:
        train_params['lr0'] = config['training']['lr0']
    if 'lrf' in config['training']:
        train_params['lrf'] = config['training']['lrf']
    if 'warmup_epochs' in config['training']:
        train_params['warmup_epochs'] = config['training']['warmup_epochs']
    if 'patience' in config['training']:
        train_params['patience'] = config['training']['patience']
    if 'dropout' in config['training']:
        train_params['dropout'] = config['training']['dropout']
    
    # 损失权重配置（如果存在）
    if 'loss' in config['training']:
        loss_config = config['training']['loss']
        if isinstance(loss_config, dict):
            # 损失权重直接作为 train_params 的参数传入（Ultralytics YOLO v8 使用 cls, box, dfl）
            if 'cls' in loss_config:
                train_params['cls'] = loss_config['cls']
            if 'box' in loss_config:
                train_params['box'] = loss_config['box']
            if 'dfl' in loss_config:
                train_params['dfl'] = loss_config['dfl']
            
            logger.info(f"损失权重配置: cls={loss_config.get('cls', 0.5)}, "
                       f"box={loss_config.get('box', 7.5)}, "
                       f"dfl={loss_config.get('dfl', 1.5)}")
    
    # 数据增强参数
    if config['training'].get('augment', True):
        train_params.update({
            'augment': True,
            'hsv_h': config['augmentation'].get('hsv_h', 0.015),
            'hsv_s': config['augmentation'].get('hsv_s', 0.7),
            'hsv_v': config['augmentation'].get('hsv_v', 0.4),
            'degrees': config['augmentation'].get('degrees', 5),
            'scale': config['augmentation'].get('scale', 0.5),
            'translate': config['augmentation'].get('translate', 0.1),
            'fliplr': config['augmentation'].get('fliplr', 0.2),
            'mosaic': config['training'].get('mosaic', 1.0),
            'copy_paste': config['training'].get('copy_paste', 0.1),
            'mixup': config['training'].get('mixup', 0.0),
            'close_mosaic': config['training'].get('close_mosaic', 0)
        })
    
    # 开始训练
    logger.info("开始训练 YOLO 模型...")
    logger.info(f"训练参数: epochs={train_params['epochs']}, batch={train_params['batch']}, imgsz={train_params['imgsz']}")
    if 'optimizer' in train_params:
        logger.info(f"优化器: {train_params['optimizer']}, lr0={train_params.get('lr0', 'default')}, "
                   f"lrf={train_params.get('lrf', 'default')}")
    if config['training'].get('augment', True):
        logger.info(f"数据增强: mosaic={train_params.get('mosaic', 1.0)}, "
                   f"copy_paste={train_params.get('copy_paste', 0.1)}, "
                   f"mixup={train_params.get('mixup', 0.0)}, "
                   f"close_mosaic={train_params.get('close_mosaic', 0)}")
    
    try:
        results = model.train(**train_params)
        
        logger.info("\n" + "=" * 60)
        logger.info("训练完成！结果摘要：")
        if hasattr(results, 'save_dir'):
            logger.info(f"训练日志保存目录: {results.save_dir}")
            best_model = Path(results.save_dir) / "weights" / "best.pt"
            if best_model.exists():
                logger.info(f"最佳模型路径: {best_model}")
        logger.info("=" * 60)
        
        return True
    except Exception as e:
        logger.error(f"训练过程出错: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    # 加载配置（使用统一路径解析）
    config_path = Path(__file__).parent / "train_config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))

    # 准备数据集
    dataset_cfg = config['dataset']
    processor = DatasetPreprocessor(
        dataset_cfg['source_dir'],
        dataset_cfg['output_dir'],
        task_type=dataset_cfg.get('task_type', 'detect'),
        kpt_shape=dataset_cfg.get('kpt_shape')
    )
    processor.prepare()

    success = TrainYOLO(config, prepare_dataset=True)
    exit(0 if success else 1)

if __name__ == "__main__":
    main()

