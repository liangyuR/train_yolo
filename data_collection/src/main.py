"""
简单的配置 GUI - 让非技术用户轻松设置参数
使用 tkinter (Python 内置，无需额外依赖)
"""
import tkinter as tk
from tkinter import ttk, messagebox
import sys
import threading
from pathlib import Path
from loguru import logger

# 允许以 `python src/main.py` 形式直接运行
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.global_config import GetConfig, SaveConfig
from src.record.record_gameplay import run_recorder, RecorderController

class ConfigGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("坦克世界 AI - 数据采集配置")
        
        # 窗口设置
        base_width = 550
        base_height = 700
        self.root.geometry(f"{base_width}x{base_height}")
        self.root.resizable(True, True)
        
        # 加载配置
        self.config = GetConfig()
        
        # 后台录制线程
        self.recorder_thread = None
        self.is_running = False
        
        self.create_widgets()
        
        # 绑定退出
        self.root.protocol("WM_DELETE_WINDOW", self.on_exit)
        
        # 自动启动后台监听
        self.start_background_service()
        
    def create_widgets(self):
        # 标题栏
        title_frame = tk.Frame(self.root, bg="#2c3e50", height=60)
        title_frame.pack(fill=tk.X)
        title_frame.pack_propagate(False)
        
        tk.Label(
            title_frame, 
            text="🎮 坦克世界 AI 数据采集工具",
            font=("微软雅黑", 16, "bold"),
            bg="#2c3e50", fg="white"
        ).pack(pady=15)
        
        # 主内容
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 1. 分辨率检测
        ttk.Label(main_frame, text="屏幕分辨率:", font=("微软雅黑", 10, "bold")).pack(anchor=tk.W)
        
        try:
            import mss
            with mss.mss() as sct:
                mon = sct.monitors[1]
                res_text = f"{mon['width']}x{mon['height']}"
        except Exception:
            res_text = "无法检测 (默认 1920x1080)"
            
        ttk.Label(
            main_frame, 
            text=f"当前主屏: {res_text} (自动全屏捕获)",
            foreground="#27ae60"
        ).pack(anchor=tk.W, pady=(0, 15))

        # 2. FPS 设置
        ttk.Label(main_frame, text="录制帧率 (FPS):", font=("微软雅黑", 10, "bold")).pack(anchor=tk.W)
        
        fps_frame = ttk.Frame(main_frame)
        fps_frame.pack(fill=tk.X, pady=(5, 15))
        
        self.fps_var = tk.DoubleVar(value=self.config['capture'].get('fps', 5.0))
        
        fps_options = [
            (0.2, "0.2 FPS (极省 - 5秒1帧)"),
            (1.0, "1.0 FPS (省空间 - 1秒1帧)"),
            (2.0, "2.0 FPS (推荐 - 训练够用)"),
            (5.0, "5.0 FPS (流畅 - 占用较大)"),
            (10.0, "10.0 FPS (高频 - 仅测试用)")
        ]
        
        for val, label in fps_options:
            ttk.Radiobutton(
                fps_frame, text=label, variable=self.fps_var, value=val
            ).pack(anchor=tk.W)

        # 3. 保存按钮
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=20)
        
        ttk.Button(
            btn_frame, text="💾 应用配置 (需重启生效)", command=self.save_config
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        ttk.Button(
            btn_frame, text="❌ 退出", command=self.on_exit
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # 4. 状态栏
        self.status_var = tk.StringVar(value="初始化中...")
        status_bar = tk.Label(
            self.root, textvariable=self.status_var, 
            bd=1, relief=tk.SUNKEN, anchor=tk.W, padx=5, pady=5,
            font=("Consolas", 9)
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def save_config(self):
        # 更新配置对象
        self.config['capture']['fps'] = self.fps_var.get()
        
        # 尝试自动更新分辨率配置
        try:
            import mss
            with mss.mss() as sct:
                mon = sct.monitors[1]
                self.config['capture']['fullscreen']['width'] = mon['width']
                self.config['capture']['fullscreen']['height'] = mon['height']
        except:
            pass

        if SaveConfig(self.config):
            messagebox.showinfo("成功", "配置已保存！\n请重启程序以应用更改。")
        else:
            messagebox.showerror("错误", "配置保存失败，请检查日志。")

    def start_background_service(self):
        """启动后台录制服务"""
        self.is_running = True
        self.recorder_thread = threading.Thread(target=run_recorder, daemon=True)
        self.recorder_thread.start()
        self.status_var.set("状态: 正在运行 | 按 F9 开始录制 | 按 F10 停止并保存")

    def on_exit(self):
        if messagebox.askokcancel("退出", "确定要退出吗？\n正在录制的数据可能会丢失。"):
            self.is_running = False
            self.root.quit()

def main():
    root = tk.Tk()
    app = ConfigGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
