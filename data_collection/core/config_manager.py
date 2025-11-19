"""
配置管理器
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger


class ConfigManager:
    """配置管理器，处理配置文件的加载和保存"""
    
    def __init__(self, config_path: Path):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path_ = Path(config_path)
        self.config_ = {}
        
        # 确保配置目录存在
        self.config_path_.parent.mkdir(parents=True, exist_ok=True)
    
    def Load(self) -> Dict[str, Any]:
        """
        加载配置文件
        
        Returns:
            配置字典
        """
        try:
            if self.config_path_.exists():
                with open(self.config_path_, 'r', encoding='utf-8') as f:
                    self.config_ = yaml.safe_load(f) or {}
                logger.info(f"已加载配置文件: {self.config_path_}")
            else:
                self.config_ = self.GetDefaultConfig()
                logger.info(f"配置文件不存在，使用默认配置")
            
            return self.config_
        except Exception as e:
            logger.warning(f"加载配置文件失败: {e}，使用默认配置")
            self.config_ = self.GetDefaultConfig()
            return self.config_
    
    def Save(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        保存配置文件
        
        Args:
            config: 配置字典（如果为 None，使用当前配置）
            
        Returns:
            是否保存成功
        """
        try:
            if config is not None:
                self.config_ = config
            
            with open(self.config_path_, 'w', encoding='utf-8') as f:
                yaml.dump(self.config_, f, allow_unicode=True, default_flow_style=False)
            
            logger.info(f"配置已保存: {self.config_path_}")
            return True
        except Exception as e:
            logger.error(f"保存配置文件失败: {e}")
            return False
    
    def Get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值
        
        Args:
            key: 配置键（支持点号分隔，如 'capture.fps'）
            default: 默认值
            
        Returns:
            配置值
        """
        keys = key.split('.')
        value = self.config_
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def Set(self, key: str, value: Any):
        """
        设置配置值
        
        Args:
            key: 配置键（支持点号分隔，如 'capture.fps'）
            value: 配置值
        """
        keys = key.split('.')
        config = self.config_
        
        # 创建嵌套字典
        for k in keys[:-1]:
            if k not in config or not isinstance(config[k], dict):
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    @staticmethod
    def GetDefaultConfig() -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'capture': {
                'fps': 5,
                'mode': 'fullscreen',
                'fullscreen': {
                    'width': 1920,
                    'height': 1080
                }
            },
            'auto_mode': False
        }
    
    def GetConfigPath(self) -> Path:
        """获取配置文件路径"""
        return self.config_path_

