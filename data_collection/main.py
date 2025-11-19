"""
简单的配置 GUI - 让非技术用户轻松设置参数
使用 tkinter (Python 内置，无需额外依赖)
"""
import tkinter as tk
from tkinter import ttk, messagebox
import sys
import os
from pathlib import Path
import logging
import threading


def GetDpiScale():
    """获取 DPI 缩放比例（用于高 DPI 屏幕适配）"""
    try:
        # Windows 系统
        if sys.platform == 'win32':
            try:
                import ctypes
                # 尝试获取 DPI
                dpi = ctypes.windll.user32.GetDpiForSystem()
                # 标准 DPI 是 96，缩放比例 = DPI / 96
                scale = dpi / 96.0
                return max(1.0, min(scale, 3.0))  # 限制在 1.0-3.0 之间
            except:
                # 如果获取失败，尝试根据分辨率估算
                try:
                    # 使用 Windows API 获取屏幕分辨率
                    import ctypes
                    user32 = ctypes.windll.user32
                    width = user32.GetSystemMetrics(0)  # SM_CXSCREEN
                    # 假设标准 1080p 是基准，4K (3840) 大约是 2x
                    if width >= 3840:
                        return 2.0
                    elif width >= 2560:
                        return 1.5
                    else:
                        return 1.0
                except:
                    return 1.0
        return 1.0
    except:
        return 1.0

try:
    from wot_ai.data_collection.core.config_manager import ConfigManager
except ImportError:
    try:
        from data_collection.core.config_manager import ConfigManager
    except ImportError:
        try:
            from core.config_manager import ConfigManager
        except ImportError:
            ConfigManager = None
            logging.warning("ConfigManager 不可用")

# 设置日志
logger = logging.getLogger(__name__)


def get_base_path():
    """获取程序基础路径（兼容 PyInstaller 打包）"""
    if getattr(sys, 'frozen', False):
        # 如果是打包后的 exe
        return Path(sys.executable).parent
    else:
        # 如果是 Python 脚本
        return Path(__file__).parent


class ConfigGUI:
    def __init__(self, root):
        self.root = root
        
        # 获取 DPI 缩放比例
        self.dpi_scale_ = GetDpiScale()
        
        # 设置高 DPI 感知（Windows）
        if sys.platform == 'win32':
            try:
                import ctypes
                # 设置 DPI 感知
                ctypes.windll.shcore.SetProcessDpiAwareness(1)  # PROCESS_PER_MONITOR_DPI_AWARE
            except:
                pass
        
        self.root.title("坦克世界 AI - 数据采集配置")
        
        # 根据 DPI 缩放调整窗口大小
        base_width = 550
        base_height = 900
        self.root.geometry(f"{int(base_width * self.dpi_scale_)}x{int(base_height * self.dpi_scale_)}")
        self.root.resizable(True, True)
        self.root.minsize(int(base_width * self.dpi_scale_), int(base_height * self.dpi_scale_))
        
        # 配置文件路径（兼容打包后的 exe）
        base_path = get_base_path()
        config_path = base_path / "configs" / "client_config.yaml"
        
        # 使用 ConfigManager（如果可用）
        if ConfigManager:
            self.config_manager_ = ConfigManager(config_path)
            self.config = self.config_manager_.Load()
        else:
            # 降级到直接文件操作
            self.config_manager_ = None
            self.config_path = config_path
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            self.load_config()
        
        # 录制状态
        self.recording_thread_ = None
        self.recorder_instance_ = None
        self.is_recording_ = False
        
        # 创建界面
        self.create_widgets()
        
        # 绑定窗口关闭事件
        self.root.protocol("WM_DELETE_WINDOW", self.on_exit)
        
    def load_config(self):
        """加载配置文件（降级方案）"""
        if self.config_manager_:
            return  # 已通过 ConfigManager 加载
        
        try:
            import yaml
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f) or {}
        except Exception as e:
            messagebox.showerror("错误", f"无法加载配置文件: {e}")
            self.config = ConfigManager.GetDefaultConfig() if ConfigManager else {
                'capture': {
                    'fps': 5,
                    'mode': 'fullscreen',
                    'fullscreen': {'width': 1920, 'height': 1080}
                }
            }
    
    def save_config(self):
        """保存配置文件"""
        if self.config_manager_:
            return self.config_manager_.Save(self.config)
        
        # 降级方案
        try:
            import yaml
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, allow_unicode=True, default_flow_style=False)
            return True
        except Exception as e:
            messagebox.showerror("错误", f"保存配置失败: {e}")
            return False
    
    def create_widgets(self):
        """创建界面组件"""
        # 根据 DPI 缩放计算字体大小
        title_font_size = int(16 * self.dpi_scale_)
        heading_font_size = int(10 * self.dpi_scale_)
        normal_font_size = int(9 * self.dpi_scale_)
        small_font_size = int(8 * self.dpi_scale_)
        info_font_size = int(9 * self.dpi_scale_)
        
        # 标题
        title_height = int(70 * self.dpi_scale_)
        title_frame = tk.Frame(self.root, bg="#2c3e50", height=title_height)
        title_frame.pack(fill=tk.X)
        title_frame.pack_propagate(False)
        
        title_pady = int(20 * self.dpi_scale_)
        title_label = tk.Label(
            title_frame,
            text="🎮 坦克世界 AI 数据采集工具",
            font=("微软雅黑", title_font_size, "bold"),
            bg="#2c3e50",
            fg="white"
        )
        title_label.pack(pady=title_pady)
        
        # 主配置区域
        main_padding = int(20 * self.dpi_scale_)
        main_frame = ttk.Frame(self.root, padding=main_padding)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 1. 屏幕分辨率显示（自动检测）
        ttk.Label(main_frame, text="屏幕分辨率:", font=("微软雅黑", heading_font_size, "bold")).grid(
            row=0, column=0, sticky=tk.W, pady=(0, int(5 * self.dpi_scale_))
        )
        
        # 自动检测屏幕分辨率
        try:
            import mss
            sct = mss.mss()
            monitor_info = sct.monitors[1]  # 主显示器
            detected_width = monitor_info['width']
            detected_height = monitor_info['height']
            self.detected_resolution = f"{detected_width}x{detected_height}"
        except Exception as e:
            logger.warning(f"无法自动检测分辨率: {e}")
            self.detected_resolution = "1920x1080"
            detected_width, detected_height = 1920, 1080
        
        resolution_info_frame = ttk.Frame(main_frame)
        resolution_info_frame.grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=(0, int(15 * self.dpi_scale_)))
        
        ttk.Label(
            resolution_info_frame,
            text=f"自动检测: {self.detected_resolution}",
            font=("微软雅黑", normal_font_size),
            foreground="#2c3e50"
        ).pack(anchor=tk.W)
        
        ttk.Label(
            resolution_info_frame,
            text="✓ 将自动使用当前屏幕分辨率进行录制",
            font=("微软雅黑", small_font_size),
            foreground="#27ae60"
        ).pack(anchor=tk.W)
        
        # 2. FPS 设置
        ttk.Label(main_frame, text="录制帧率 (FPS):", font=("微软雅黑", heading_font_size, "bold")).grid(
            row=2, column=0, sticky=tk.W, pady=(int(15 * self.dpi_scale_), int(5 * self.dpi_scale_))
        )
        
        fps_frame = ttk.Frame(main_frame)
        fps_frame.grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=(0, int(15 * self.dpi_scale_)))
        
        self.fps_var = tk.IntVar(value=self.config.get('capture', {}).get('fps', 5))
        
        fps_options = [
            (1, "1 FPS (极省空间 - 1秒1帧)"),
            (2, "2 FPS (很省空间)"),
            (5, "5 FPS (推荐 - 省空间)"),
            (10, "10 FPS (平衡)"),
            (15, "15 FPS (更流畅)"),
            (30, "30 FPS (流畅 - 占用较多空间)")
        ]
        
        for fps, label in fps_options:
            ttk.Radiobutton(
                fps_frame,
                text=label,
                variable=self.fps_var,
                value=fps
            ).pack(anchor=tk.W, pady=int(2 * self.dpi_scale_))
        
        # 3. 自动模式设置
        ttk.Label(main_frame, text="自动检测模式:", font=("微软雅黑", heading_font_size, "bold")).grid(
            row=4, column=0, sticky=tk.W, pady=(int(15 * self.dpi_scale_), int(5 * self.dpi_scale_))
        )
        
        auto_mode_frame = ttk.Frame(main_frame)
        auto_mode_frame.grid(row=5, column=0, columnspan=2, sticky=tk.W, pady=(0, int(15 * self.dpi_scale_)))
        
        self.auto_mode_var = tk.BooleanVar(value=self.config.get('auto_mode', False))
        
        auto_checkbox = ttk.Checkbutton(
            auto_mode_frame,
            text="启用自动检测（检测战斗开始/结束，自动录制）",
            variable=self.auto_mode_var
        )
        auto_checkbox.pack(anchor=tk.W, pady=int(2 * self.dpi_scale_))
        
        label_padx = int(20 * self.dpi_scale_)
        ttk.Label(
            auto_mode_frame,
            text="• 检测区域: 屏幕中心靠上1/3区域",
            font=("微软雅黑", small_font_size),
            foreground="#7f8c8d"
        ).pack(anchor=tk.W, padx=(label_padx, 0))
        
        ttk.Label(
            auto_mode_frame,
            text="• 战斗开始时自动开始录制",
            font=("微软雅黑", small_font_size),
            foreground="#7f8c8d"
        ).pack(anchor=tk.W, padx=(label_padx, 0))
        
        ttk.Label(
            auto_mode_frame,
            text="• 胜利/被击败/被击毁时自动停止录制",
            font=("微软雅黑", small_font_size),
            foreground="#7f8c8d"
        ).pack(anchor=tk.W, padx=(label_padx, 0))
        
        # 4. 存储估算
        ttk.Separator(main_frame, orient=tk.HORIZONTAL).grid(
            row=6, column=0, columnspan=2, sticky="ew", pady=int(15 * self.dpi_scale_)
        )
        
        info_padx = int(10 * self.dpi_scale_)
        info_pady = int(10 * self.dpi_scale_)
        self.info_label = tk.Label(
            main_frame,
            text="",
            font=("Consolas", info_font_size),
            justify=tk.LEFT,
            bg="#ecf0f1",
            fg="#34495e",
            padx=info_padx,
            pady=info_pady
        )
        self.info_label.grid(row=7, column=0, columnspan=2, sticky="ew", pady=(0, int(15 * self.dpi_scale_)))
        
        self.update_info()
        
        # 状态显示
        status_pady = int(5 * self.dpi_scale_)
        self.status_label = tk.Label(
            main_frame,
            text="状态: 就绪",
            font=("微软雅黑", normal_font_size),
            fg="#27ae60",
            pady=status_pady
        )
        self.status_label.grid(row=8, column=0, columnspan=2, pady=(int(10 * self.dpi_scale_), status_pady))
        
        # 按钮区域
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=9, column=0, columnspan=2, pady=(int(10 * self.dpi_scale_), 0))
        
        button_width = int(15 * self.dpi_scale_)
        button_padx = int(5 * self.dpi_scale_)
        self.save_button = ttk.Button(
            button_frame,
            text="💾 保存配置",
            command=self.apply_config,
            width=button_width
        )
        self.save_button.pack(side=tk.LEFT, padx=button_padx)
        
        self.record_button = ttk.Button(
            button_frame,
            text="🎬 开始录制",
            command=self.start_recording,
            width=button_width
        )
        self.record_button.pack(side=tk.LEFT, padx=button_padx)
        
        ttk.Button(
            button_frame,
            text="❌ 退出",
            command=self.on_exit,
            width=button_width
        ).pack(side=tk.LEFT, padx=button_padx)
        
        # 绑定变量改变事件
        self.fps_var.trace('w', lambda *args: self.update_info())
    
    def update_info(self):
        """更新存储估算信息"""
        # 使用目标分辨率（960x540）进行估算，而不是捕获分辨率
        target_width = 960
        target_height = 540
        
        fps = self.fps_var.get()
        
        # 估算存储（PNG 压缩后约 1-3 bits per pixel，比 JPEG 大但无损）
        pixels = target_width * target_height
        bytes_per_frame = pixels * 0.2  # 平均压缩率
        
        # 每分钟 (考虑 frame_step=2)
        actual_fps = fps / 2
        mb_per_minute = (bytes_per_frame * actual_fps * 60) / (1024 * 1024)
        
        # 每小时
        mb_per_hour = mb_per_minute * 60
        
        # 根据FPS调整显示信息
        if fps <= 2:
            time_info = f"每场战斗 (约 5 分钟): ~{mb_per_minute * 5 / 1024:.2f} MB"
        elif fps <= 5:
            time_info = f"每场战斗 (约 5 分钟): ~{mb_per_minute * 5:.1f} MB"
        else:
            time_info = f"每场战斗 (约 5 分钟): ~{mb_per_minute * 5 / 1024:.2f} MB"
        
        info_text = f"""📊 存储估算 (frame_step=2):

保存分辨率: {target_width}x{target_height}
录制帧率: {fps} FPS
实际保存: {actual_fps:.1f} FPS

每分钟: ~{mb_per_minute:.1f} MB
每小时: ~{mb_per_hour / 1024:.2f} GB
{time_info}"""
        
        self.info_label.config(text=info_text)
    
    def apply_config(self):
        """应用配置"""
        # 使用检测到的分辨率
        try:
            width, height = map(int, self.detected_resolution.split('x'))
        except:
            messagebox.showerror("错误", "无法获取屏幕分辨率")
            return
            
        fps = self.fps_var.get()
        auto_mode = self.auto_mode_var.get()
        
        # 更新配置
        if 'capture' not in self.config:
            self.config['capture'] = {}
        if 'fullscreen' not in self.config['capture']:
            self.config['capture']['fullscreen'] = {}
        
        self.config['capture']['fullscreen']['width'] = width
        self.config['capture']['fullscreen']['height'] = height
        self.config['capture']['fps'] = fps
        self.config['auto_mode'] = auto_mode
        
        # 保存
        if self.save_config():
            mode_text = "自动模式" if auto_mode else "手动模式（F9/F10）"
            messagebox.showinfo("成功", f"配置已保存！\n\n录制模式: {mode_text}\n\n可以开始录制了。")
    
    def start_recording(self):
        """启动录制（在后台线程中运行）"""
        if self.is_recording_:
            messagebox.showwarning("警告", "录制已在运行中！")
            return
        
        # 先保存配置
        try:
            width, height = map(int, self.detected_resolution.split('x'))
        except:
            messagebox.showerror("错误", "无法获取屏幕分辨率")
            return
            
        fps = self.fps_var.get()
        auto_mode = self.auto_mode_var.get()
        
        if 'capture' not in self.config:
            self.config['capture'] = {}
        if 'fullscreen' not in self.config['capture']:
            self.config['capture']['fullscreen'] = {}
        
        self.config['capture']['fullscreen']['width'] = width
        self.config['capture']['fullscreen']['height'] = height
        self.config['capture']['fps'] = fps
        self.config['auto_mode'] = auto_mode
        
        if not self.save_config():
            return
        
        # 更新UI状态
        self.is_recording_ = True
        if auto_mode:
            status_text = "状态: 录制程序运行中... (自动检测模式)"
        else:
            status_text = "状态: 录制程序运行中... (等待按 F9 开始录制)"
        self.status_label.config(text=status_text, fg="#e67e22")
        self.record_button.config(text="⏸️  录制中...", state="disabled")
        self.save_button.config(state="disabled")
        
        # 在后台线程中启动录制
        def run_recording():
            try:
                # 导入录制模块
                sys.path.insert(0, str(get_base_path()))
                from wot_ai.data_collection.record_gameplay import run_with_config
                
                # 使用配置启动录制
                run_with_config(self.config)
                
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror(
                    "错误", 
                    f"录制过程出错: {e}\n\n详情请查看日志"
                ))
                import traceback
                traceback.print_exc()
            finally:
                # 重置状态
                self.root.after(0, self.reset_recording_state)
        
        self.recording_thread_ = threading.Thread(target=run_recording, daemon=True)
        self.recording_thread_.start()
        
        if auto_mode:
            message_text = (
                "配置已保存！\n\n录制程序已在后台运行。\n\n自动检测模式：\n"
                "• 自动检测战斗开始，开始录制\n"
                "• 自动检测胜利/被击败/被击毁，停止录制\n"
                "• 检测区域: 屏幕中心靠上1/3\n\n"
                "快捷键说明：\n"
                "• F9  - 手动开始录制（覆盖自动检测）\n"
                "• F10 - 手动停止录制（覆盖自动检测）\n"
                "• Ctrl+C - 退出程序\n\n"
                "可以关闭此窗口（录制会继续）"
            )
        else:
            message_text = (
                "配置已保存！\n\n录制程序已在后台运行。\n\n快捷键说明：\n"
                "• F9  - 开始录制\n"
                "• F10 - 停止录制\n"
                "• Ctrl+C - 退出程序\n\n"
                "使用方法：\n"
                "1. 进入游戏战斗\n"
                "2. 按 F9 开始录制\n"
                "3. 正常游戏\n"
                "4. 按 F10 停止录制\n"
                "5. 可重复按 F9/F10 录制多场\n"
                "6. 可以关闭此窗口（录制会继续）"
            )
        
        messagebox.showinfo("录制已启动", message_text)
    
    def reset_recording_state(self):
        """重置录制状态"""
        self.is_recording_ = False
        self.status_label.config(text="状态: 已停止", fg="#e74c3c")
        self.record_button.config(text="🎬 开始录制", state="normal")
        self.save_button.config(state="normal")
        self.recording_thread_ = None
    
    def on_exit(self):
        """退出程序"""
        if self.is_recording_:
            if messagebox.askyesno(
                "确认退出",
                "录制程序正在运行中。\n\n退出将终止录制程序。\n\n确定要退出吗？"
            ):
                self.root.quit()
        else:
            self.root.quit()


def main():
    root = tk.Tk()
    app = ConfigGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

