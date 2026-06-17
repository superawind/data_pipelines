#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
抽象接口定义

定义LLM请求和沙箱执行的抽象接口，为后续具体实现提供标准化接口。
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass


@dataclass
class LLMRequest:
    """LLM请求数据结构"""
    prompt: str
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    json_mode: bool = True
    
    
@dataclass
class LLMResponse:
    """LLM响应数据结构"""
    content: str
    success: bool
    error_message: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None


@dataclass
class CodeExecutionRequest:
    """代码执行请求数据结构"""
    code: str
    timeout: int = 60
    working_directory: Optional[str] = None
    environment_vars: Optional[Dict[str, str]] = None


@dataclass
class CodeExecutionResult:
    """代码执行结果数据结构"""
    success: bool
    result: Any = None
    error_message: Optional[str] = None
    execution_time: float = 0.0
    stdout: str = ""
    stderr: str = ""


class LLMProvider(ABC):
    """LLM服务提供商抽象接口"""
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        检查LLM服务是否可用
        
        Returns:
            是否可用
        """
        pass
    
    @abstractmethod
    def request(self, request: LLMRequest) -> LLMResponse:
        """
        发送LLM请求
        
        Args:
            request: LLM请求对象
            
        Returns:
            LLM响应对象
        """
        pass


class SandboxExecutor(ABC):
    """沙箱代码执行器抽象接口"""
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        检查沙箱环境是否可用
        
        Returns:
            是否可用
        """
        pass
    
    @abstractmethod
    def execute(self, request: CodeExecutionRequest) -> CodeExecutionResult:
        """
        在沙箱环境中执行代码
        
        Args:
            request: 代码执行请求对象
            
        Returns:
            代码执行结果对象
        """
        pass
    
    @abstractmethod
    def execute_verification_code(self, verification_code: str, response_code: str) -> bool:
        """
        执行验证代码，返回验证结果
        
        Args:
            verification_code: 验证代码
            response_code: 待验证的响应代码
            
        Returns:
            验证是否通过（True/False）
        """
        pass


# 全局抽象接口实例（待具体实现注入）
llm_provider: Optional[LLMProvider] = None
sandbox_executor: Optional[SandboxExecutor] = None


def set_llm_provider(provider: LLMProvider) -> None:
    """设置LLM服务提供商"""
    global llm_provider
    llm_provider = provider


def set_sandbox_executor(executor: SandboxExecutor) -> None:
    """设置沙箱执行器"""
    global sandbox_executor
    sandbox_executor = executor


def get_llm_provider() -> LLMProvider:
    """获取LLM服务提供商"""
    if llm_provider is None:
        raise RuntimeError("LLM provider not set. Call set_llm_provider() first.")
    return llm_provider


def get_sandbox_executor() -> SandboxExecutor:
    """获取沙箱执行器"""
    if sandbox_executor is None:
        raise RuntimeError("Sandbox executor not set. Call set_sandbox_executor() first.")
    return sandbox_executor 