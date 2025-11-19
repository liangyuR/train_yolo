# WoT AI

坦克世界 AI 辅助工具

## 项目结构

```
wot_ai/
├── pyproject.toml          # Poetry 配置文件
├── README.md               # 项目说明文档
├── wot_ai/                 # 主包
│   ├── __init__.py
│   ├── data_collection/    # 数据采集模块
│   ├── game_modules/       # 游戏功能模块
│   ├── train_yolo/         # YOLO 模型训练模块
│   └── utils/              # 工具模块
├── tests/                  # 测试文件
└── data/                   # 数据目录（训练数据、模型等）
```

## 安装

### 使用 Poetry（推荐）

1. 安装 Poetry（如果尚未安装）：
```bash
pip install poetry
```

2. 安装项目依赖：
```bash
poetry install
```

3. 激活虚拟环境：
```bash
poetry env activate
C:\Users\11601\AppData\Local\pypoetry\Cache\virtualenvs\wot-ai-aIKBCqeS-py3.11\Scripts\activate.ps1
```

### 使用 pip

```bash
pip install -e .
```

## 依赖

主要依赖包括：
- PyYAML
- opencv-python
- numpy==1.26.4
- mss
- pynput
- loguru
- dxcam>=0.0.5
- ultralytics>=8.0.0
- pytesseract
- pyinstaller

完整依赖列表请查看 `pyproject.toml`。

## 使用

### 数据采集

```bash
python -m wot_ai.data_collection.main
```

### 路径规划

```bash
python -m wot_ai.game_modules.navigation.controller.path_planning_controller
```

### 模型训练

```bash
# 训练小地图检测模型
python -m wot_ai.train_yolo.train_minimap_yolo

# 准备数据集
python -m wot_ai.train_yolo.prepare_minimap_dataset

# 模型推理
python -m wot_ai.train_yolo.inference_minimap_yolo
```

## 开发

### 运行测试

```bash
poetry run pytest tests/
```

### 构建

使用 PyInstaller 构建可执行文件：

```bash
python -m wot_ai.data_collection.build
```

## 许可证

[添加许可证信息]

