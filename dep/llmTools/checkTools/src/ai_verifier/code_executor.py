#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
代码执行器模块（重构版本）

负责在安全的沙箱环境中执行验证代码。
重构后的特性：
1. 使用抽象的沙箱执行器接口
2. 直接执行验证代码，返回 true/false 结果
3. 简化验证逻辑
"""

import logging
import time
from typing import Optional

from .abstractions import get_sandbox_executor, CodeExecutionRequest
from .models import CodeVerificationPoint, CodeVerificationResult
from .config import Config

logger = logging.getLogger(__name__)


class CodeExecutor:
    """代码执行器（重构版本）"""
    
    def __init__(self, config: Config):
        """
        初始化代码执行器
        
        Args:
            config: 配置对象
        """
        self.config = config
    
    def execute_code_verification(
        self, 
        code_verification: CodeVerificationPoint,
        response_code: str
    ) -> CodeVerificationResult:
        """
        执行代码验证
        
        Args:
            code_verification: 代码验证点
            response_code: 待验证的响应代码
            
        Returns:
            代码验证结果
        """
        logger.info(f"开始执行代码验证: {code_verification.function_name}")
        
        try:
            # 检查沙箱执行器是否可用
            executor = get_sandbox_executor()
            if not executor.is_available():
                logger.error("沙箱执行器不可用")
                return CodeVerificationResult(
                    success=False,
                    passed=False,
                    execution_time=0.0,
                    error_message="沙箱执行器不可用"
                )
            
            # 验证代码安全性
            is_safe, safety_error = self._validate_code_safety(response_code)
            if not is_safe:
                logger.warning(f"代码安全检查失败: {safety_error}")
                return CodeVerificationResult(
                    success=False,
                    passed=False,
                    execution_time=0.0,
                    error_message=f"代码安全检查失败: {safety_error}"
                )
            
            # 执行验证代码
            start_time = time.time()
            
            try:
                # 使用抽象的沙箱执行器
                verification_passed = executor.execute_verification_code(
                    verification_code=code_verification.verification_code,
                    response_code=response_code
                )
                
                execution_time = time.time() - start_time
                
                logger.info(f"代码验证完成: {verification_passed}, 用时: {execution_time:.2f}秒")
                
                return CodeVerificationResult(
                    success=True,
                    passed=verification_passed,
                    execution_time=execution_time,
                    details=f"函数 {code_verification.function_name} 验证{'通过' if verification_passed else '失败'}"
                )
                
            except Exception as e:
                execution_time = time.time() - start_time
                error_msg = f"验证执行异常: {str(e)}"
                logger.error(error_msg)
                
                return CodeVerificationResult(
                    success=False,
                    passed=False,
                    execution_time=execution_time,
                    error_message=error_msg
                )
                
        except Exception as e:
            error_msg = f"代码验证过程异常: {str(e)}"
            logger.error(error_msg)
            return CodeVerificationResult(
                success=False,
                passed=False,
                execution_time=0.0,
                error_message=error_msg
            )
    
    def _validate_code_safety(self, code: str) -> tuple[bool, str]:
        """
        验证代码安全性
        
        Args:
            code: 待验证的代码
            
        Returns:
            (是否安全, 错误信息)
        """
        if not self.config.enable_code_sandboxing:
            # 如果未启用沙箱，则跳过安全检查
            return True, ""
        
        # 危险关键词检查
        dangerous_keywords = [
            "__import__", "exec", "eval", "compile", "open", "file",
            "subprocess", "os.system", "os.popen", "os.remove", "os.rmdir",
            "shutil", "socket", "urllib", "requests", "http",
            "sys.exit", "quit", "exit", "input", "raw_input",
            "globals", "locals", "vars", "dir", "setattr", "getattr", "delattr"
        ]
        
        code_lower = code.lower()
        
        for keyword in dangerous_keywords:
            if keyword in code_lower:
                return False, f"检测到危险关键词: {keyword}"
        
        # 检查导入语句
        lines = code.split('\n')
        for i, line in enumerate(lines, 1):
            line = line.strip()
            if line.startswith('import ') or line.startswith('from '):
                # 检查是否导入了危险模块
                dangerous_imports = [
                    'os', 'sys', 'subprocess', 'shutil', 'socket',
                    'urllib', 'requests', 'http', 'ftplib', 'smtplib'
                ]
                
                for dangerous_import in dangerous_imports:
                    if f'import {dangerous_import}' in line or f'from {dangerous_import}' in line:
                        return False, f"第{i}行：不允许导入危险模块 {dangerous_import}"
        
        # 检查代码长度（防止过长的恶意代码）
        if len(code) > 10000:  # 10KB限制
            return False, "代码长度超过限制 (10KB)"
        
        # 检查行数（防止过多的循环或递归）
        if len(lines) > 500:
            return False, "代码行数超过限制 (500行)"
        
        return True, ""
    
    def is_available(self) -> bool:
        """
        检查代码执行器是否可用
        
        Returns:
            是否可用
        """
        try:
            executor = get_sandbox_executor()
            return executor.is_available()
        except Exception as e:
            logger.error(f"检查代码执行器可用性失败: {str(e)}")
            return False


# 模拟沙箱执行器（默认实现，后续可替换）
class MockSandboxExecutor:
    """模拟沙箱执行器（用于测试和开发）"""
    
    def is_available(self) -> bool:
        """检查是否可用"""
        return True
    
    def execute(self, request: CodeExecutionRequest):
        """执行代码请求（暂未实现）"""
        logger.warning("使用模拟沙箱执行器，execute方法未实现")
        from .abstractions import CodeExecutionResult
        return CodeExecutionResult(
            success=False,
            error_message="模拟沙箱执行器未实现execute方法"
        )
    
    def execute_verification_code(self, verification_code: str, response_code: str) -> bool:
        """
        执行验证代码（模拟实现）
        
        Args:
            verification_code: 验证代码
            response_code: 待验证的响应代码
            
        Returns:
            验证是否通过
        """
        logger.info("使用模拟沙箱执行器执行验证代码")
        
        try:
            # 简单的模拟验证：检查响应代码是否包含函数定义
            if 'def ' not in response_code:
                logger.info("模拟验证失败：未发现函数定义")
                return False
            
            # 检查是否有明显的语法错误
            try:
                compile(response_code, '<string>', 'exec')
            except SyntaxError:
                logger.info("模拟验证失败：语法错误")
                return False
            
            # 模拟执行验证代码
            namespace = {}
            try:
                # 执行响应代码
                exec(response_code, namespace)
                
                # 执行验证代码
                exec(verification_code, namespace)
                
                # 获取验证结果
                verification_result = namespace.get('verification_result', False)
                logger.info(f"模拟验证结果: {verification_result}")
                return bool(verification_result)
                
            except Exception as e:
                logger.info(f"模拟验证执行异常: {str(e)}")
                return False
                
        except Exception as e:
            logger.error(f"模拟验证过程异常: {str(e)}")
            return False


# 初始化默认的模拟沙箱执行器
def _initialize_default_executor():
    """初始化默认的模拟执行器"""
    try:
        from .abstractions import set_sandbox_executor
        default_executor = MockSandboxExecutor()
        set_sandbox_executor(default_executor)
        logger.info("初始化默认模拟沙箱执行器")
    except Exception as e:
        logger.error(f"初始化默认沙箱执行器失败: {str(e)}")


# 模块加载时初始化
_initialize_default_executor() 