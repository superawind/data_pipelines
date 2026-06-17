#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置管理模块

统一管理系统配置，支持从环境变量、配置文件等多种方式加载配置。
"""

import os
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class Config:
    """系统配置类"""
    
    # 基础路径配置
    project_root: str = field(default_factory=lambda: str(Path(__file__).parent.parent.parent))
    prompts_dir: str = field(default="")
    data_dir: str = field(default="")
    config_dir: str = field(default="")
    
    # LLM配置
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4-turbo-preview"
    openai_base_url: Optional[str] = None
    contract_generation_temperature: float = 0.2
    evaluation_temperature: float = 0.1
    max_tokens_contract: int = 4000
    max_tokens_evaluation: int = 2000
    
    # 代码执行配置
    code_execution_timeout: int = 60
    pytest_timeout: int = 30
    enable_code_sandboxing: bool = True
    
    # 验证配置
    pass_threshold: float = 0.6
    strict_mode: bool = True
    
    # 日志配置
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    # 性能配置
    max_concurrent_requests: int = 5
    request_retry_count: int = 3
    request_timeout: int = 30
    
    def __post_init__(self):
        """初始化后处理"""
        # 设置基础路径
        if not self.prompts_dir:
            self.prompts_dir = os.path.join(self.project_root, "prompts")
        if not self.data_dir:
            self.data_dir = os.path.join(self.project_root, "data")
        if not self.config_dir:
            self.config_dir = os.path.join(self.project_root, "config")
        
        # 从环境变量加载配置
        self._load_from_env()
        
        # 创建必要的目录
        self._ensure_directories()
    
    def _load_from_env(self) -> None:
        """从环境变量加载配置"""
        env_mappings = {
            "OPENAI_API_KEY": "openai_api_key",
            "OPENAI_MODEL": "openai_model", 
            "OPENAI_BASE_URL": "openai_base_url",
            "AI_VERIFIER_LOG_LEVEL": "log_level",
            "AI_VERIFIER_LOG_FILE": "log_file",
            "AI_VERIFIER_STRICT_MODE": "strict_mode",
        }
        
        for env_key, attr_name in env_mappings.items():
            env_value = os.getenv(env_key)
            if env_value is not None:
                # 类型转换
                if attr_name == "strict_mode":
                    env_value = env_value.lower() in ("true", "1", "yes")
                elif attr_name in ["contract_generation_temperature", "evaluation_temperature", "pass_threshold"]:
                    env_value = float(env_value)
                elif attr_name in ["max_tokens_contract", "max_tokens_evaluation", "code_execution_timeout", 
                                  "pytest_timeout", "max_concurrent_requests", "request_retry_count", "request_timeout"]:
                    env_value = int(env_value)
                
                setattr(self, attr_name, env_value)
                logger.debug(f"从环境变量加载配置: {attr_name} = {env_value}")
    
    def _ensure_directories(self) -> None:
        """确保必要的目录存在"""
        dirs = [self.prompts_dir, self.data_dir, self.config_dir]
        
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def load_from_file(cls, config_file: str) -> "Config":
        """
        从配置文件加载配置
        
        Args:
            config_file: 配置文件路径
            
        Returns:
            配置对象
        """
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                config_data = json.load(f)
            
            # 创建配置对象
            config = cls()
            
            # 更新配置
            for key, value in config_data.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                else:
                    logger.warning(f"未知的配置项: {key}")
            
            logger.info(f"成功从文件加载配置: {config_file}")
            return config
            
        except FileNotFoundError:
            logger.warning(f"配置文件不存在: {config_file}，使用默认配置")
            return cls()
        except json.JSONDecodeError as e:
            logger.error(f"配置文件格式错误: {str(e)}")
            return cls()
        except Exception as e:
            logger.error(f"加载配置文件失败: {str(e)}")
            return cls()
    
    def save_to_file(self, config_file: str) -> None:
        """
        保存配置到文件
        
        Args:
            config_file: 配置文件路径
        """
        try:
            config_data = {}
            
            # 获取所有配置项
            for field_name in self.__dataclass_fields__:
                value = getattr(self, field_name)
                if value is not None:
                    config_data[field_name] = value
            
            # 保存到文件
            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"配置已保存到文件: {config_file}")
            
        except Exception as e:
            logger.error(f"保存配置文件失败: {str(e)}")
            raise
    
    def validate(self) -> bool:
        """
        验证配置的有效性
        
        Returns:
            配置是否有效
        """
        errors = []
        
        # 检查必需的目录
        if not os.path.exists(self.prompts_dir):
            errors.append(f"prompts目录不存在: {self.prompts_dir}")
        
        # 检查模型配置
        if not self.openai_model:
            errors.append("未配置OpenAI模型")
        
        # 检查温度参数
        if not (0 <= self.contract_generation_temperature <= 2):
            errors.append("contract_generation_temperature必须在0-2之间")
        
        if not (0 <= self.evaluation_temperature <= 2):
            errors.append("evaluation_temperature必须在0-2之间")
        
        # 检查阈值参数
        if not (0 <= self.pass_threshold <= 1):
            errors.append("pass_threshold必须在0-1之间")
        
        # 检查超时参数
        if self.code_execution_timeout <= 0:
            errors.append("code_execution_timeout必须大于0")
        
        if errors:
            for error in errors:
                logger.error(f"配置验证失败: {error}")
            return False
        
        logger.info("配置验证成功")
        return True
    
    def get_summary(self) -> Dict[str, Any]:
        """
        获取配置摘要
        
        Returns:
            配置摘要字典
        """
        return {
            "openai_model": self.openai_model,
            "has_api_key": bool(self.openai_api_key),
            "strict_mode": self.strict_mode,
            "pass_threshold": self.pass_threshold,
            "log_level": self.log_level,
            "prompts_dir": self.prompts_dir,
            "data_dir": self.data_dir
        } 