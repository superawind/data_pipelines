#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM接口模块

提供统一的LLM调用接口，具体实现由用户自定义。
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def call_llm_api(
    prompt: str,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    json_mode: bool = True
) -> str:
    """
    调用LLM API的统一接口
    
    Args:
        prompt: 输入提示
        temperature: 温度参数，控制输出随机性
        max_tokens: 最大token数
        json_mode: 是否要求JSON格式输出
        
    Returns:
        LLM生成的响应文本
        
    Raises:
        NotImplementedError: 此函数需要用户自行实现
        
    Note:
        此函数需要用户根据具体的LLM服务提供商进行实现，例如：
        - OpenAI API
        - Azure OpenAI
        - Claude API
        - 本地模型API
        等等
    """
    # TODO: 用户需要在这里实现具体的LLM调用逻辑
    pass


def is_llm_available() -> bool:
    """
    检查LLM服务是否可用
    
    Returns:
        bool: LLM服务是否可用
        
    Note:
        此函数需要用户根据具体情况实现，例如：
        - 检查API密钥是否设置
        - 测试API连接
        - 检查模型可用性
        等等
    """
    # TODO: 用户需要在这里实现LLM可用性检查逻辑
    pass 