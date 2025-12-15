#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练脚本（优化版）：完整读取 YAML 新增参数 + 预设 + 自检
- 兼容老配置(output.*)与新配置(training 内含 project/name/save 等)
- 支持 augmentation.preset 和 --preset
- 完整映射常用超参到 ultralytics.YOLO.train(...)
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from loguru import logger
from ultralytics import YOLO
import math
import torch

def _gpu_info():
    """Return (enabled, name, total_gb, free_gb). Safe on CPU-only nodes."""
    if not torch.cuda.is_available():
        return (False, "cpu", 0.0, 0.0)
    try:
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        total_gb = float(props.total_memory) / (1024**3)
        free_gb = None
        try:
            free, total = torch.cuda.mem_get_info()
            free_gb = float(free) / (1024**3)
        except Exception:
            # Fallback: estimate free as 85% of total when mem_get_info unavailable
            free_gb = total_gb * 0.85
        return (True, props.name, total_gb, free_gb)
    except Exception:
        return (True, "cuda", 0.0, 0.0)

def _round_to_multiple(x: int, m: int = 32) -> int:
    return int(max(m, math.floor(x / m) * m))

def _suggest_batch_imgsz(params: Dict[str, Any]) -> None:
    """Heuristic suggestions based on VRAM + imgsz + model scale.
    Why: help pick a feasible batch (>=4 for mosaic) without OOM, and propose imgsz trade-offs.
    """
    cuda_ok, gpu_name, total_gb, free_gb = _gpu_info()
    if not cuda_ok:
        logger.info("资源提示：未检测到 CUDA；根据 CPU 训练将自动采用较小 batch。")
        return

    imgsz = int(params.get("imgsz", 640))
    mosaic = float(params.get("mosaic", 0.0))
    amp = bool(params.get("amp", True))

    # crude capacity model (empirical): memory ≈ model_const + per_image_const*(imgsz/640)^2 * batch
    # choose conservative constants for yolo11m-pose
    model_const = 1.2  # GB (weights + optimizer state w/ AdamW + activations baseline)
    per_image_const = 0.28  # GB per image at 640 in FP16; FP32 roughly x1.6
    if not amp:
        per_image_const *= 1.6

    avail_gb = max(0.0, free_gb - 0.3)  # keep 300MB headroom
    if avail_gb <= model_const:
        logger.warning(f"资源提示：GPU({gpu_name}) 可用显存约 {free_gb:.2f}GB，建议降低 imgsz 或改用 accumulate 提升有效批次。")
        return

    scale = (imgsz / 640.0) ** 2
    if scale <= 0:
        return

    max_batch_est = int((avail_gb - model_const) / (per_image_const * scale))
    max_batch_est = max(1, min(256, max_batch_est))

    cur_batch = int(params.get("batch", 16))
    if cur_batch > max_batch_est:
        logger.warning(
            f"资源建议：当前 batch={cur_batch} 可能接近/超过显存容量（估算上限≈{max_batch_est}）→ "
            f"建议将 batch 调整到 {min(cur_batch, max_batch_est)} 或开启 accumulate。"
        )
    elif cur_batch < max_batch_est and cur_batch * 2 <= max_batch_est:
        logger.info(f"资源建议：显存允许更大 batch；可尝试 batch≈{min(max_batch_est, cur_batch*2)} 以提升吞吐。")

    # Mosaic friendliness: aim for batch>=4
    if mosaic > 0 and cur_batch < 4 and max_batch_est >= 4:
        logger.info("资源建议：已开启 mosaic，建议 batch≥4 以提升多样性（估算上限允许）。")

    # If cannot reach batch>=4, propose imgsz reduction to reach target
    target = 4 if mosaic > 0 else cur_batch
    if mosaic > 0 and max_batch_est < 4:
        # compute imgsz' s.t. ((avail-model_const)/(per_image*scale')) >= 4
        needed_scale = (avail_gb - model_const) / (per_image_const * target)
        if needed_scale > 0:
            imgsz_new = int(640 * math.sqrt(needed_scale))
            imgsz_new = _round_to_multiple(imgsz_new, 32)
            if imgsz_new < imgsz and imgsz_new >= 320:
                logger.info(f"资源建议：显存不足以在 imgsz={imgsz} 下满足 batch≥4；可将 imgsz≈{imgsz_new} 以获得更稳定的 mosaic")




# 如果你的工程包含该模块则保留；否则可注释掉两行
from dataset_preprocessor import DatasetPreprocessor, DatasetPreprocessorConfig  # noqa: F401


# ------------------ utils ------------------

def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base) if base else {}
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _apply_preset(config: Dict[str, Any], cli_preset: Optional[str]) -> None:
    """合并 presets[chosen] -> augmentation，记录元信息。"""
    presets = config.get("presets") or {}
    chosen = cli_preset or (config.get("augmentation") or {}).get("preset")
    if not chosen:
        return
    if chosen not in presets:
        logger.error(f"未找到预设: {chosen}；可选: {list(presets.keys())}")
        sys.exit(2)
    before = copy.deepcopy(config.get("augmentation", {}))
    merged = _deep_merge(before, presets[chosen])
    config["augmentation"] = merged
    config.setdefault("_meta", {})["applied_preset"] = chosen
    # 仅打印变化键
    changes = {k: (before.get(k), merged.get(k)) for k in sorted(set(before) | set(merged)) if before.get(k) != merged.get(k)}
    if changes:
        logger.info("已应用预设: %s", chosen)
        logger.info("预设变更: " + ", ".join(f"{k}:{o}->{n}" for k,(o,n) in changes.items()))


def _pull(cfg_a: Dict[str, Any], cfg_b: Dict[str, Any], key: str, default: Any = None) -> Any:
    """从两个命名空间中择一取配置，先取 cfg_a[key]，否则 cfg_b[key]，否则 default。"""
    if cfg_a and key in cfg_a:
        return cfg_a[key]
    if cfg_b and key in cfg_b:
        return cfg_b[key]
    return default


def _merge_augmentation(training_cfg: Dict[str, Any], root_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """构建 YOLO 所需增强参数集。"""
    aug_cfg: Dict[str, Any] = dict(root_cfg.get("augmentation", {}) or {})
    enable = training_cfg.get("augment", True) and aug_cfg.get("enable", True)
    if not enable:
        return {"augment": False}

    defaults = {
        "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4,
        "degrees": 0.0, "translate": 0.1, "scale": 0.5,
        "shear": 0.0, "perspective": 0.0,
        "fliplr": 0.0, "flipud": 0.0,
        "mosaic": 0.0, "close_mosaic": 0,
        "copy_paste": 0.0, "mixup": 0.0,
    }
    # 过滤掉 preset 和 enable 键
    merged = {**defaults, **{k: v for k, v in aug_cfg.items() if k not in ("preset", "enable")}}
    # 边界修正
    def _clamp(x, lo, hi):
        try:
            return max(lo, min(hi, float(x)))
        except Exception:
            return lo
    merged["fliplr"] = _clamp(merged.get("fliplr", 0.0), 0.0, 1.0)
    merged["flipud"] = _clamp(merged.get("flipud", 0.0), 0.0, 1.0)
    merged["mosaic"] = _clamp(merged.get("mosaic", 0.0), 0.0, 1.0)
    merged["mixup"]  = _clamp(merged.get("mixup", 0.0), 0.0, 1.0)
    merged["degrees"] = float(merged.get("degrees", 0.0))
    merged["translate"] = _clamp(merged.get("translate", 0.1), 0.0, 1.0)
    merged["scale"] = _clamp(merged.get("scale", 0.5), 0.0, 1.0)
    merged["shear"] = float(merged.get("shear", 0.0))
    merged["perspective"] = float(merged.get("perspective", 0.0))
    merged["close_mosaic"] = int(max(0, merged.get("close_mosaic", 0)))
    merged["augment"] = True
    return merged


def _collect_train_params(config: Dict[str, Any], yaml_path: Path) -> Dict[str, Any]:
    t = config.get("training", {}) or {}
    m = config.get("model", {}) or {}
    o = config.get("output", {}) or {}  # 兼容老结构
    # 兼容：project/name/save/... 可在 training 或 output
    project = _pull(t, o, "project", "runs/train")
    name = _pull(t, o, "name", "exp")
    save = _pull(t, o, "save", True)
    save_period = _pull(t, o, "save_period", -1)
    val = _pull(t, o, "val", True)
    exist_ok = _pull(t, o, "exist_ok", False)
    plots = _pull(t, o, "plots", True)
    resume = _pull(t, o, "resume", False)
    close_logger = _pull(t, o, "close_logger", False)

    params: Dict[str, Any] = {
        # 必需
        "data": str(yaml_path),
        "imgsz": m.get("imgsz", 640),
        "epochs": t.get("epochs", 100),
        "batch": t.get("batch", 16),
        "workers": t.get("workers", 8),
        "device": m.get("device", "0"),
        # 训练流程/输出
        "project": project,
        "name": name,
        "save": save,
        "save_period": save_period,
        "val": val,
        "exist_ok": exist_ok,
        "plots": plots,
        "resume": resume,
        "verbose": True,
        # 扩展
        "seed": t.get("seed", 42),
        "amp": t.get("amp", True),
        "deterministic": t.get("deterministic", False),
        # "accumulate": t.get("accumulate", 1), # 报错提示不支持
        # 优化器与 LR
        "optimizer": t.get("optimizer", "auto"),
        "lr0": t.get("lr0", 0.01),
        "lrf": t.get("lrf", 0.01),
        "momentum": t.get("momentum", 0.937),
        "weight_decay": t.get("weight_decay", 0.0005),
        "warmup_epochs": t.get("warmup_epochs", 3.0),
        "warmup_momentum": t.get("warmup_momentum", 0.8),
        "warmup_bias_lr": t.get("warmup_bias_lr", 0.1),
        # 正则与停止
        "dropout": t.get("dropout", 0.0),
        "label_smoothing": t.get("label_smoothing", 0.0),
        "patience": t.get("patience", 50),
        # "ema": t.get("ema", True), # 不再支持直接传参
        # 数据加载策略
        "cache": t.get("cache", False),
        "rect": t.get("rect", False),
        # "shuffle": t.get("shuffle", True),  # YOLO train 不再直接支持 shuffle 参数
        # 任务相关
        "single_cls": t.get("single_cls", False),
        "classes": t.get("classes", None),
        "max_det": t.get("max_det", 300),
        # 其他
        "freeze": t.get("freeze", 0),
        # "close_logger": close_logger, # 不再支持
        # "ema": t.get("ema", True),   # 不再支持
    }

    # 合并损失权重
    loss = t.get("loss") or {}
    for k in ("cls", "box", "dfl"):
        if k in loss:
            params[k] = loss[k]

    # 合并增强
    aug = _merge_augmentation(t, config)
    params.update(aug if aug.get("augment") else {"augment": False})

    return params


def _validate_params(params: Dict[str, Any]) -> None:
    ok = True
    # 基本范围
    def _in(v, lo, hi):  # 包含端点
        return lo <= v <= hi
    if params["epochs"] < 1:
        logger.error("epochs 必须 >=1"); ok = False
    if params["batch"] < 1:
        logger.error("batch 必须 >=1"); ok = False
    if not _in(float(params.get("lrf", 0.01)), 0.0, 1.0):
        logger.error("lrf 需在 (0,1]"); ok = False
    if float(params.get("lr0", 0.01)) <= 0.0 or float(params.get("lr0", 0.01)) > 0.1:
        logger.error("lr0 建议在 (0,0.1]"); ok = False

    # 交互提示
    if params.get("augment"):
        if float(params.get("mosaic", 0.0)) > 0 and params["batch"] < 4:
            logger.warning("开启 mosaic 但 batch<4，建议增大 batch 或降低 mosaic。")
        if float(params.get("degrees", 0.0)) > 30 and params["imgsz"] < 512:
            logger.warning("旋转角度较大且输入较小，可能截断目标；留意。")
        if int(params.get("close_mosaic", 0)) == 0 and float(params.get("mosaic", 0.0)) > 0:
            logger.warning("未设置 close_mosaic，后期可能影响验证稳定。")

    if not ok:
        logger.error("参数校验未通过，请修正后重试。")
        sys.exit(2)


# ------------------ train flow ------------------

def TrainYOLO(config: Dict[str, Any]) -> bool:
    dataset_cfg = config.get("dataset", {})
    dataset_dir = Path(dataset_cfg["output_dir"])
    yaml_path = dataset_dir / "dataset.yaml"
    if not yaml_path.exists():
        logger.error(f"dataset.yaml 不存在: {yaml_path}")
        return False

    model_path = Path(config["model"]["path"])
    model = YOLO(str(model_path))

    params = _collect_train_params(config, yaml_path)
    _validate_params(params)
    _suggest_batch_imgsz(params)

    # 日志摘要
    logger.info("开始训练 | epochs=%s batch=%s imgsz=%s opt=%s lr0=%s lrf=%s",
                params["epochs"], params["batch"], params["imgsz"],
                params.get("optimizer"), params.get("lr0"), params.get("lrf"))
    if params.get("augment"):
        logger.info(
            "增强: deg=%s mosaic=%s close_mosaic=%s fliplr=%s flipud=%s shear=%s persp=%s trans=%s scale=%s mixup=%s copy_paste=%s",
            params.get("degrees"), params.get("mosaic"), params.get("close_mosaic"),
            params.get("fliplr"), params.get("flipud"), params.get("shear"), params.get("perspective"),
            params.get("translate"), params.get("scale"), params.get("mixup"), params.get("copy_paste"),
        )
    else:
        logger.info("增强: 已禁用")

    try:
        results = model.train(**params)
        if hasattr(results, "save_dir"):
            logger.info(f"日志目录: {results.save_dir}")
        return True
    except Exception as e:  # noqa: BLE001
        logger.error(f"训练失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="YOLO training with full YAML support & presets")
    parser.add_argument("--prepare", action="store_true", default=False, help="准备数据")
    parser.add_argument("--config", type=str, default=str(Path(__file__).parent / "train_config.yaml"))
    parser.add_argument("--preset", type=str, default=None, help="增强预设名，覆盖 YAML augmentation.preset")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        logger.error(f"配置文件不存在: {cfg_path}")
        sys.exit(2)

    config = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    prepare_data = args.prepare
    if prepare_data:
        if "dataset" in config:
            ds = config["dataset"]
            processor = DatasetPreprocessor(DatasetPreprocessorConfig(
                input_dir=ds["source_dir"],
                output_dir=ds["output_dir"],
                random_seed=ds.get("random_seed", 42),
                train_ratio=ds["train_ratio"],
                val_ratio=ds["val_ratio"],
                test_ratio=ds["test_ratio"],
            ))
            processor.prepare()

    # 应用预设并训练
    _apply_preset(config, args.preset)
    ok = TrainYOLO(config)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
