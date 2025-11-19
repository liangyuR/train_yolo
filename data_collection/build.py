#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python程序打包脚本
使用 PyInstaller 打包 data_collection 应用程序
"""

import argparse
import os
import shutil
import sys
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional


class BuildConfig:
    """打包配置"""
    
    def __init__(self):
        self.project_root_ = Path(__file__).parent
        self.dist_dir_ = self.project_root_ / "dist"
        self.build_dir_ = self.project_root_ / "build"
        self.spec_dir_ = self.project_root_
        
        # 主程序入口
        self.gui_entry_ = self.project_root_ / "main.py"
        self.cli_entry_ = self.project_root_ / "record_gameplay.py"
        
        # 输出名称
        self.gui_name_ = "WoTDataCollection"
        self.cli_name_ = "WoTRecorder"
        
        # 图标文件（可选）
        self.icon_file_ = None  # 可以后续添加 .ico 文件


def CollectHiddenImports() -> List[str]:
    """
    收集所有需要显式导入的隐藏模块
    
    Returns:
        隐藏导入列表
    """
    hidden_imports = [
        # Tkinter 相关
        'tkinter',
        'tkinter.ttk',
        'tkinter.messagebox',
        'tkinter.filedialog',
        
        # OpenCV
        'cv2',
        'numpy',
        
        # YOLO/Ultralytics
        'ultralytics',
        'ultralytics.models',
        'ultralytics.utils',
        'torch',
        'torchvision',
        
        # 屏幕捕获
        'mss',
        'dxcam',
        'dxcam.camera',
        
        # 输入监听
        'pynput',
        'pynput.keyboard',
        'pynput.mouse',
        
        # 日志
        'loguru',
        
        # YAML
        'yaml',
        'yaml.loader',
        
        # Windows 特定
        'win32api',
        'win32con',
        'win32gui',
        'pywintypes',
        'psutil',
        
        # 内部模块（确保被包含）
        'wot_ai.data_collection',
        'wot_ai.data_collection.core',
        'wot_ai.data_collection.core.config_manager',
        'wot_ai.data_collection.detection',
        'wot_ai.data_collection.detection.game_state_detector',
        'wot_ai.data_collection.detection.yolo_state_detector',
        'wot_ai.data_collection.capture',
        'wot_ai.data_collection.capture.mss_capture',
        'wot_ai.data_collection.capture.dxcam_capture',
        'wot_ai.data_collection.save',
        'wot_ai.data_collection.save.frame_saver',
        'wot_ai.data_collection.save.async_frame_saver',
        'wot_ai.data_collection.save.turbojpeg_saver',
        'wot_ai.data_collection.listeners',
        'wot_ai.data_collection.listeners.global_listener',
        'wot_ai.data_collection.listeners.pynput_listener',
        'wot_ai.data_collection.recording_events',
        'wot_ai.data_collection.recording_overlay',
        'wot_ai.data_collection.global_hotkey',
    ]
    
    return hidden_imports


def CollectDataFiles(project_root: Path) -> List[Tuple[str, str]]:
    """
    收集需要包含的数据文件
    
    Args:
        project_root: 项目根目录
        
    Returns:
        数据文件列表，格式: [(源路径, 目标相对路径), ...]
    """
    data_files = []
    
    # 配置文件目录
    configs_dir = project_root / "configs"
    if configs_dir.exists():
        data_files.append((
            str(configs_dir),
            "configs"
        ))
    
    # 可选：如果存在 YOLO 模型目录，可以包含
    # yolo_models_dir = project_root.parent / "yolo" / "train" / "model"
    # if yolo_models_dir.exists():
    #     data_files.append((
    #         str(yolo_models_dir),
    #         "models"
    #     ))
    
    return data_files


def CheckPyInstaller() -> bool:
    """检查 PyInstaller 是否已安装"""
    try:
        import PyInstaller
        return True
    except ImportError:
        return False


def CleanBuildArtifacts(build_dir: Path, dist_dir: Path, spec_dir: Path):
    """
    清理旧的构建文件
    
    Args:
        build_dir: build 目录
        dist_dir: dist 目录
        spec_dir: spec 文件目录
    """
    print("清理旧的构建文件...")
    
    if build_dir.exists():
        shutil.rmtree(build_dir)
        print(f"  已删除: {build_dir}")
    
    if dist_dir.exists():
        shutil.rmtree(dist_dir)
        print(f"  已删除: {dist_dir}")
    
    # 清理 spec 文件（可选，保留用于调试）
    spec_files = list(spec_dir.glob("*.spec"))
    for spec_file in spec_files:
        if spec_file.name.startswith("WoT"):
            spec_file.unlink()
            print(f"  已删除: {spec_file}")


def BuildGUI(config: BuildConfig, mode: str = "onedir", clean: bool = True) -> bool:
    """
    打包 GUI 应用程序
    
    Args:
        config: 打包配置
        mode: 打包模式 ("onefile" 或 "onedir")
        clean: 是否清理旧的构建文件
        
    Returns:
        是否成功
    """
    if not config.gui_entry_.exists():
        print(f"错误: GUI入口文件不存在: {config.gui_entry_}")
        return False
    
    if clean:
        CleanBuildArtifacts(config.build_dir_, config.dist_dir_, config.spec_dir_)
    
    print(f"\n开始打包 GUI 应用程序 ({mode})...")
    print(f"入口文件: {config.gui_entry_}")
    
    # 收集参数
    hidden_imports = CollectHiddenImports()
    data_files = CollectDataFiles(config.project_root_)
    
    # 构建 PyInstaller 命令
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name", config.gui_name_,
        "--windowed",  # GUI 模式，无控制台窗口
    ]
    
    if mode == "onefile":
        cmd.append("--onefile")
    else:
        cmd.append("--onedir")
    
    # 添加隐藏导入
    for imp in hidden_imports:
        cmd.extend(["--hidden-import", imp])
    
    # 添加数据文件
    for src, dst in data_files:
        if os.name == 'nt':  # Windows
            cmd.extend(["--add-data", f"{src};{dst}"])
        else:  # Unix
            cmd.extend(["--add-data", f"{src}:{dst}"])
    
    # 添加图标（如果有）
    if config.icon_file_ and config.icon_file_.exists():
        cmd.extend(["--icon", str(config.icon_file_)])
    
    # 添加入口文件
    cmd.append(str(config.gui_entry_))
    
    print(f"\n执行命令: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, cwd=config.project_root_)
        print(f"\n✓ GUI 应用程序打包成功!")
        print(f"  输出目录: {config.dist_dir_}")
        if mode == "onefile":
            exe_path = config.dist_dir_ / f"{config.gui_name_}.exe"
        else:
            exe_path = config.dist_dir_ / config.gui_name_ / f"{config.gui_name_}.exe"
        print(f"  可执行文件: {exe_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ 打包失败: {e}")
        return False
    except Exception as e:
        print(f"\n✗ 打包过程出错: {e}")
        import traceback
        traceback.print_exc()
        return False


def BuildCLI(config: BuildConfig, mode: str = "onedir", clean: bool = False) -> bool:
    """
    打包命令行工具
    
    Args:
        config: 打包配置
        mode: 打包模式 ("onefile" 或 "onedir")
        clean: 是否清理旧的构建文件
        
    Returns:
        是否成功
    """
    if not config.cli_entry_.exists():
        print(f"错误: CLI入口文件不存在: {config.cli_entry_}")
        return False
    
    if clean:
        # 只清理 CLI 相关的构建文件
        cli_build_dir = config.build_dir_ / config.cli_name_
        cli_dist_dir = config.dist_dir_ / config.cli_name_
        if cli_build_dir.exists():
            shutil.rmtree(cli_build_dir)
        if cli_dist_dir.exists():
            shutil.rmtree(cli_dist_dir)
    
    print(f"\n开始打包命令行工具 ({mode})...")
    print(f"入口文件: {config.cli_entry_}")
    
    # 收集参数
    hidden_imports = CollectHiddenImports()
    data_files = CollectDataFiles(config.project_root_)
    
    # 构建 PyInstaller 命令
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name", config.cli_name_,
        # CLI 工具保留控制台窗口
    ]
    
    if mode == "onefile":
        cmd.append("--onefile")
    else:
        cmd.append("--onedir")
    
    # 添加隐藏导入
    for imp in hidden_imports:
        cmd.extend(["--hidden-import", imp])
    
    # 添加数据文件
    for src, dst in data_files:
        if os.name == 'nt':  # Windows
            cmd.extend(["--add-data", f"{src};{dst}"])
        else:  # Unix
            cmd.extend(["--add-data", f"{src}:{dst}"])
    
    # 添加图标（如果有）
    if config.icon_file_ and config.icon_file_.exists():
        cmd.extend(["--icon", str(config.icon_file_)])
    
    # 添加入口文件
    cmd.append(str(config.cli_entry_))
    
    print(f"\n执行命令: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, cwd=config.project_root_)
        print(f"\n✓ 命令行工具打包成功!")
        print(f"  输出目录: {config.dist_dir_}")
        if mode == "onefile":
            exe_path = config.dist_dir_ / f"{config.cli_name_}.exe"
        else:
            exe_path = config.dist_dir_ / config.cli_name_ / f"{config.cli_name_}.exe"
        print(f"  可执行文件: {exe_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ 打包失败: {e}")
        return False
    except Exception as e:
        print(f"\n✗ 打包过程出错: {e}")
        import traceback
        traceback.print_exc()
        return False


def VerifyBuild(config: BuildConfig, gui: bool = True, cli: bool = False) -> bool:
    """
    验证打包结果
    
    Args:
        config: 打包配置
        gui: 是否验证 GUI 程序
        cli: 是否验证 CLI 工具
        
    Returns:
        是否验证通过
    """
    print("\n验证打包结果...")
    success = True
    
    if gui:
        gui_exe_onefile = config.dist_dir_ / f"{config.gui_name_}.exe"
        gui_exe_onedir = config.dist_dir_ / config.gui_name_ / f"{config.gui_name_}.exe"
        
        if gui_exe_onefile.exists():
            print(f"  ✓ GUI 单文件: {gui_exe_onefile} ({gui_exe_onefile.stat().st_size / 1024 / 1024:.1f} MB)")
        elif gui_exe_onedir.exists():
            print(f"  ✓ GUI 目录: {gui_exe_onedir}")
            print(f"    大小: {gui_exe_onedir.stat().st_size / 1024 / 1024:.1f} MB")
        else:
            print(f"  ✗ GUI 可执行文件不存在")
            success = False
    
    if cli:
        cli_exe_onefile = config.dist_dir_ / f"{config.cli_name_}.exe"
        cli_exe_onedir = config.dist_dir_ / config.cli_name_ / f"{config.cli_name_}.exe"
        
        if cli_exe_onefile.exists():
            print(f"  ✓ CLI 单文件: {cli_exe_onefile} ({cli_exe_onefile.stat().st_size / 1024 / 1024:.1f} MB)")
        elif cli_exe_onedir.exists():
            print(f"  ✓ CLI 目录: {cli_exe_onedir}")
            print(f"    大小: {cli_exe_onedir.stat().st_size / 1024 / 1024:.1f} MB")
        else:
            print(f"  ✗ CLI 可执行文件不存在")
            success = False
    
    return success


def ParseArgs() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="打包 data_collection 应用程序",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--mode",
        choices=["onefile", "onedir"],
        default="onedir",
        help="打包模式: onefile (单文件) 或 onedir (目录) [默认: onedir]"
    )
    
    parser.add_argument(
        "--target",
        choices=["gui", "cli", "all"],
        default="gui",
        help="打包目标: gui (GUI程序), cli (命令行工具), all (全部) [默认: gui]"
    )
    
    parser.add_argument(
        "--clean",
        action="store_true",
        help="打包前清理旧的构建文件"
    )
    
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="打包后不验证结果"
    )
    
    parser.add_argument(
        "--icon",
        type=str,
        help="图标文件路径 (.ico)"
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    args = ParseArgs()
    
    # 检查 PyInstaller
    if not CheckPyInstaller():
        print("错误: PyInstaller 未安装")
        print("请运行: pip install pyinstaller")
        return 1
    
    # 创建配置
    config = BuildConfig()
    
    # 设置图标（如果提供）
    if args.icon:
        config.icon_file_ = Path(args.icon)
        if not config.icon_file_.exists():
            print(f"警告: 图标文件不存在: {config.icon_file_}")
            config.icon_file_ = None
    
    print("=" * 80)
    print("Python 程序打包脚本")
    print("=" * 80)
    print(f"项目根目录: {config.project_root_}")
    print(f"打包模式: {args.mode}")
    print(f"打包目标: {args.target}")
    print(f"清理构建: {args.clean}")
    print("=" * 80)
    
    success = True
    
    # 打包 GUI
    if args.target in ["gui", "all"]:
        success = BuildGUI(config, mode=args.mode, clean=args.clean) and success
    
    # 打包 CLI
    if args.target in ["cli", "all"]:
        success = BuildCLI(config, mode=args.mode, clean=args.clean and args.target == "cli") and success
    
    # 验证
    if success and not args.no_verify:
        VerifyBuild(
            config,
            gui=args.target in ["gui", "all"],
            cli=args.target in ["cli", "all"]
        )
    
    print("\n" + "=" * 80)
    if success:
        print("✓ 打包完成!")
    else:
        print("✗ 打包失败，请检查错误信息")
    print("=" * 80)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

