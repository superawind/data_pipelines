#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM客户端模块

统一的LLM接口，支持OpenAI API调用和模拟模式。
"""

import json
import logging
import time
from typing import Dict, Any, Optional

from .config import Config
from .llm_interface import call_llm_api, is_llm_available

logger = logging.getLogger(__name__)

# 尝试导入OpenAI客户端
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    logger.warning("未安装openai库，将仅支持模拟模式")


class LLMClient:
    """LLM客户端"""
    
    def __init__(self, config: Config):
        """
        初始化LLM客户端
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.client = None
        self.mock_mode = False
        
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        """初始化LLM客户端"""
        if not HAS_OPENAI:
            logger.warning("OpenAI库未安装，使用模拟模式")
            self.mock_mode = True
            return
        
        if not self.config.openai_api_key:
            logger.warning("未配置OpenAI API密钥，使用模拟模式")
            self.mock_mode = True
            return
        
        try:
            # 初始化OpenAI客户端
            client_kwargs = {
                "api_key": self.config.openai_api_key
            }
            
            if self.config.openai_base_url:
                client_kwargs["base_url"] = self.config.openai_base_url
            
            self.client = openai.OpenAI(**client_kwargs)
            
            # 测试连接
            self._test_connection()
            
            logger.info("OpenAI客户端初始化成功")
            
        except Exception as e:
            logger.error(f"OpenAI客户端初始化失败: {str(e)}，切换到模拟模式")
            self.mock_mode = True
            self.client = None
    
    def _test_connection(self) -> None:
        """测试OpenAI连接"""
        try:
            # 发送一个简单的测试请求
            response = self.client.chat.completions.create(
                model=self.config.openai_model,
                messages=[{"role": "user", "content": "测试"}],
                max_tokens=10,
                timeout=self.config.request_timeout
            )
            logger.debug("OpenAI连接测试成功")
        except Exception as e:
            logger.error(f"OpenAI连接测试失败: {str(e)}")
            raise
    
    def generate(
        self, 
        prompt: str, 
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = True
    ) -> str:
        """
        生成响应
        
        Args:
            prompt: 输入提示
            temperature: 温度参数
            max_tokens: 最大token数
            json_mode: 是否使用JSON模式
            
        Returns:
            生成的响应文本
            
        Raises:
            RuntimeError: 生成失败
        """
        if not prompt.strip():
            raise ValueError("提示不能为空")
        
        # 优先尝试使用用户实现的LLM接口
        try:
            if is_llm_available():
                logger.info("使用用户实现的LLM接口")
                return call_llm_api(prompt, temperature, max_tokens, json_mode)
        except (NotImplementedError, Exception) as e:
            logger.debug(f"用户LLM接口不可用: {e}，回退到内置实现")
        
        # 回退到原有逻辑
        if self.mock_mode:
            return self._generate_mock_response(prompt, json_mode)
        
        return self._generate_openai_response(prompt, temperature, max_tokens, json_mode)
    
    def _generate_openai_response(
        self, 
        prompt: str, 
        temperature: Optional[float],
        max_tokens: Optional[int],
        json_mode: bool
    ) -> str:
        """
        使用OpenAI API生成响应
        
        Args:
            prompt: 输入提示
            temperature: 温度参数
            max_tokens: 最大token数
            json_mode: 是否使用JSON模式
            
        Returns:
            生成的响应文本
        """
        # 设置默认参数
        if temperature is None:
            temperature = self.config.evaluation_temperature
        if max_tokens is None:
            max_tokens = self.config.max_tokens_evaluation
        
        # 构建请求参数
        request_kwargs = {
            "model": self.config.openai_model,
            "messages": [{"role": "system", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": self.config.request_timeout
        }
        
        if json_mode:
            request_kwargs["response_format"] = {"type": "json_object"}
        
        # 重试机制
        for attempt in range(self.config.request_retry_count):
            try:
                logger.debug(f"OpenAI API请求开始 (尝试 {attempt + 1}/{self.config.request_retry_count})")
                
                response = self.client.chat.completions.create(**request_kwargs)
                content = response.choices[0].message.content
                
                if not content:
                    raise RuntimeError("OpenAI返回空响应")
                
                logger.debug("OpenAI API请求成功")
                return content
                
            except Exception as e:
                logger.warning(f"OpenAI API请求失败 (尝试 {attempt + 1}): {str(e)}")
                
                if attempt < self.config.request_retry_count - 1:
                    # 指数退避
                    wait_time = 2 ** attempt
                    logger.debug(f"等待 {wait_time} 秒后重试")
                    time.sleep(wait_time)
                else:
                    logger.error("OpenAI API请求最终失败，切换到模拟模式")
                    return self._generate_mock_response(prompt, json_mode)
        
        # 如果到这里说明重试次数用尽
        return self._generate_mock_response(prompt, json_mode)
    
    def _generate_mock_response(self, prompt: str, json_mode: bool) -> str:
        """
        生成模拟响应
        
        Args:
            prompt: 输入提示  
            json_mode: 是否使用JSON模式
            
        Returns:
            模拟的响应文本
        """
        logger.debug("使用模拟模式生成响应")
        
        if json_mode:
            # 根据prompt类型返回不同的模拟响应
            if "验证契约生成器" in prompt or "contract_generator" in prompt.lower():
                return self._get_mock_contract_response()
            elif "评判器" in prompt or "evaluator" in prompt.lower():
                return self._get_mock_evaluation_response(prompt)
            else:
                return '{"response": "这是一个模拟的JSON响应"}'
        else:
            return "这是一个模拟的文本响应，用于演示系统功能。"
    
    def _get_mock_contract_response(self) -> str:
        """获取模拟的验证契约响应"""
        mock_contract = {
            "code_verification": {
                "validation_type": "response_code_test",
                "input_source_description": "模拟：从AI回应中提取Python代码，导入mock_function函数进行功能测试",
                "validation_code": """
import pytest
from user_code import mock_function

def test_basic_functionality():
    # 模拟基本功能测试
    result = mock_function("test")
    assert result is not None, "函数应该返回非空结果"

def test_edge_cases():
    # 模拟边界情况测试
    result = mock_function("")
    assert isinstance(result, (str, int, float, list, dict)), "函数应该返回有效类型"
""",
                "expected_success_criteria": "所有pytest测试用例通过，退出代码为0"
            },
            "llm_verification": {
                "evaluation_points": [
                    {
                        "id": "mock_relevance",
                        "description": "模拟评估点：内容是否相关？",
                        "weight": 1.0
                    },
                    {
                        "id": "mock_quality", 
                        "description": "模拟评估点：质量是否良好？",
                        "weight": 1.0
                    }
                ]
            }
        }
        
        return json.dumps(mock_contract, ensure_ascii=False, indent=2)
    
    def _get_mock_evaluation_response(self, prompt: str) -> str:
        """获取模拟的评估响应"""
        import re
        
        # 尝试从prompt中解析验证契约
        verification_contract = None
        try:
            # 查找verification_contract标记后的JSON内容
            contract_start = prompt.find('{{verification_contract}}')
            if contract_start == -1:
                # 如果没找到模板标记，查找实际替换后的内容
                contract_start = prompt.find('- **验证契约**:')
                if contract_start != -1:
                    # 找到验证契约部分，提取JSON
                    json_start = prompt.find('{', contract_start)
                    if json_start != -1:
                        # 找到匹配的结束括号
                        brace_count = 0
                        json_end = json_start
                        for i, char in enumerate(prompt[json_start:], json_start):
                            if char == '{':
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    json_end = i + 1
                                    break
                        
                        if brace_count == 0:
                            contract_json = prompt[json_start:json_end]
                            verification_contract = json.loads(contract_json)
        except Exception as e:
            logger.debug(f"解析验证契约失败: {e}")
        
        # 构建模拟评估结果
        verification_scores = {}
        total_points = 0
        max_points = 0
        
        # 检查是否有代码验证
        if verification_contract and "code_verification" in verification_contract:
            verification_scores["code_verification_score"] = {
                "score": 1,
                "reasoning": "模拟：代码验证通过，所有要求满足"
            }
            total_points += 1
            max_points += 1
        
        # 处理LLM验证点
        llm_scores = []
        if verification_contract and "llm_verification" in verification_contract:
            evaluation_points = verification_contract["llm_verification"].get("evaluation_points", [])
            for point in evaluation_points:
                llm_scores.append({
                    "id": point["id"],
                    "description": point["description"],
                    "score": 1,
                    "reasoning": f"模拟：{point['description']} - 评估通过"
                })
                total_points += 1
                max_points += 1
        else:
            # 默认的模拟LLM验证点
            llm_scores = [
                {
                    "id": "mock_relevance",
                    "description": "模拟评估点：内容是否相关？",
                    "score": 1,
                    "reasoning": "模拟：内容相关，评估通过"
                },
                {
                    "id": "mock_quality",
                    "description": "模拟评估点：质量是否良好？", 
                    "score": 1,
                    "reasoning": "模拟：质量良好，评估通过"
                }
            ]
            total_points += 2
            max_points += 2
        
        verification_scores["llm_verification_scores"] = llm_scores
        
        mock_evaluation = {
            "verification_scores": verification_scores,
            "summary": {
                "total_points": total_points,
                "max_points": max_points,
                "pass_rate": (total_points / max_points * 100) if max_points > 0 else 0,
                "overall_verdict": "PASSED" if total_points == max_points else "FAILED"
            }
        }
        
        return json.dumps(mock_evaluation, ensure_ascii=False, indent=2)
    
    def is_mock_mode(self) -> bool:
        """
        检查是否为模拟模式
        
        Returns:
            是否为模拟模式
        """
        return self.mock_mode
    
    def get_status(self) -> Dict[str, Any]:
        """
        获取客户端状态
        
        Returns:
            状态信息字典
        """
        return {
            "mock_mode": self.mock_mode,
            "has_openai_client": self.client is not None,
            "model": self.config.openai_model,
            "has_api_key": bool(self.config.openai_api_key)
        } 