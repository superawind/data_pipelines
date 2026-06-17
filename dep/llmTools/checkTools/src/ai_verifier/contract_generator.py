#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证契约生成器模块（重构版本）

负责基于用户提示生成结构化的验证契约。
重构后的特性：
1. 代码验证：生成严格的输入输出测试用例和可直接执行的验证代码
2. LLM验证：定义清晰的验证点，后续一次性评估
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, Optional, List

from .abstractions import get_llm_provider, LLMRequest
from .models import (
    VerificationContract, 
    CodeVerificationPoint, 
    LLMVerificationPoint, 
    CodeTestCase
)
from .config import Config

logger = logging.getLogger(__name__)


class ContractGenerator:
    """验证契约生成器（重构版本）"""
    
    def __init__(self, config: Config):
        """
        初始化契约生成器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self._load_prompt_template()
    
    def _load_prompt_template(self) -> None:
        """加载契约生成器的prompt模板"""
        try:
            prompt_path = Path(self.config.prompts_dir) / "contract_generator.txt"
            with open(prompt_path, "r", encoding="utf-8") as f:
                self.prompt_template = f.read()
            logger.info("成功加载契约生成器prompt模板")
        except FileNotFoundError:
            logger.error(f"找不到prompt模板文件: {prompt_path}")
            raise
        except Exception as e:
            logger.error(f"加载prompt模板失败: {str(e)}")
            raise
    
    def generate_contract(self, user_prompt: str) -> VerificationContract:
        """
        为用户提示生成验证契约
        
        Args:
            user_prompt: 用户的原始提示
            
        Returns:
            验证契约对象
            
        Raises:
            ValueError: 用户提示为空
            RuntimeError: 契约生成失败
        """
        if not user_prompt.strip():
            raise ValueError("用户提示不能为空")
        
        logger.info(f"开始为提示生成验证契约: {user_prompt[:100]}...")
        
        try:
            # 首先判断是否需要代码验证
            if self._should_use_code_verification(user_prompt):
                code_verification = self._generate_code_verification(user_prompt)
            else:
                code_verification = None
            
            # 生成LLM验证点
            llm_verification_points = self._generate_llm_verification_points(user_prompt)
            
            # 构建验证契约
            contract = VerificationContract(
                code_verification=code_verification,
                llm_verification_points=llm_verification_points
            )
            
            logger.info(f"成功生成验证契约，包含 {contract.get_total_points()} 个验证点")
            return contract
            
        except Exception as e:
            logger.error(f"生成验证契约失败: {str(e)}")
            return self._create_fallback_contract(user_prompt)
    
    def _should_use_code_verification(self, user_prompt: str) -> bool:
        """
        判断是否应该使用代码验证
        
        Args:
            user_prompt: 用户提示
            
        Returns:
            是否使用代码验证
        """
        # 检查是否包含明确的函数编写要求
        function_patterns = [
            r'写一?个?函数',
            r'写一?个?方法', 
            r'def\s+\w+',
            r'函数\s*\w+\s*\(',
            r'定义一?个?函数',
            r'实现一?个?函数'
        ]
        
        has_function_request = any(re.search(pattern, user_prompt, re.IGNORECASE) 
                                 for pattern in function_patterns)
        
        if not has_function_request:
            return False
        
        # 检查是否是简单的数学/算法函数
        simple_function_keywords = [
            '加法', '减法', '乘法', '除法', '求和', '平均值',
            '最大值', '最小值', '阶乘', '斐波那契', 
            '排序', '查找', '反转', '计算'
        ]
        
        is_simple_function = any(keyword in user_prompt 
                               for keyword in simple_function_keywords)
        
        # 检查是否包含复杂特征（如果包含这些，则不使用代码验证）
        complex_keywords = [
            'flask', 'django', 'web', 'api', 'http', 'url',
            '网站', '应用', '服务器', '数据库', '框架',
            '类', 'class', '继承', '多文件', '模块'
        ]
        
        has_complex_features = any(keyword in user_prompt.lower() 
                                 for keyword in complex_keywords)
        
        # 最终判断：有函数要求 且 是简单函数 且 没有复杂特征
        return has_function_request and is_simple_function and not has_complex_features
    
    def _generate_code_verification(self, user_prompt: str) -> CodeVerificationPoint:
        """
        生成代码验证点
        
        Args:
            user_prompt: 用户提示
            
        Returns:
            代码验证点对象
        """
        try:
            # 提取函数名
            function_name = self._extract_function_name(user_prompt)
            
            # 生成测试用例
            test_cases = self._generate_test_cases(user_prompt, function_name)
            
            # 生成验证代码
            verification_code = self._generate_verification_code(function_name, test_cases)
            
            return CodeVerificationPoint(
                function_name=function_name,
                test_cases=test_cases,
                verification_code=verification_code,
                description=f"验证函数 {function_name} 的正确性"
            )
            
        except Exception as e:
            logger.error(f"生成代码验证点失败: {str(e)}")
            # 返回基础的代码验证点
            return self._create_basic_code_verification()
    
    def _extract_function_name(self, user_prompt: str) -> str:
        """
        从用户提示中提取函数名
        
        Args:
            user_prompt: 用户提示
            
        Returns:
            函数名
        """
        # 尝试提取明确的函数名
        patterns = [
            r'函数\s*(\w+)\s*\(',
            r'def\s+(\w+)\s*\(',
            r'写一?个?函数\s*(\w+)',
            r'(\w+)\s*\(\s*[^)]*\s*\)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, user_prompt, re.IGNORECASE)
            if match:
                return match.group(1)
        
        # 根据任务类型推断函数名
        if '加法' in user_prompt or '求和' in user_prompt:
            return 'add'
        elif '乘法' in user_prompt:
            return 'multiply'
        elif '除法' in user_prompt:
            return 'divide'
        elif '减法' in user_prompt:
            return 'subtract'
        elif '阶乘' in user_prompt:
            return 'factorial'
        elif '反转' in user_prompt:
            return 'reverse'
        elif '最大值' in user_prompt:
            return 'max_value'
        elif '最小值' in user_prompt:
            return 'min_value'
        else:
            return 'target_function'
    
    def _generate_test_cases(self, user_prompt: str, function_name: str) -> List[CodeTestCase]:
        """
        生成测试用例
        
        Args:
            user_prompt: 用户提示
            function_name: 函数名
            
        Returns:
            测试用例列表
        """
        test_cases = []
        
        # 根据函数类型生成不同的测试用例
        if function_name in ['add', 'subtract', 'multiply', 'divide'] or '加法' in user_prompt or '减法' in user_prompt or '乘法' in user_prompt:
            test_cases = [
                CodeTestCase(
                    name="基础测试",
                    inputs={"a": 2, "b": 3},
                    expected_output=5 if 'add' in function_name or '加法' in user_prompt else None,
                    description="测试基本功能"
                ),
                CodeTestCase(
                    name="零值测试",
                    inputs={"a": 0, "b": 5},
                    expected_output=5 if 'add' in function_name or '加法' in user_prompt else None,
                    description="测试零值情况"
                ),
                CodeTestCase(
                    name="负数测试",
                    inputs={"a": -2, "b": 3},
                    expected_output=1 if 'add' in function_name or '加法' in user_prompt else None,
                    description="测试负数情况"
                )
            ]
            
            # 根据具体操作调整期望输出
            for case in test_cases:
                a, b = case.inputs["a"], case.inputs["b"]
                if '乘法' in user_prompt or 'multiply' in function_name:
                    case.expected_output = a * b
                elif '减法' in user_prompt or 'subtract' in function_name:
                    case.expected_output = a - b
                elif '除法' in user_prompt or 'divide' in function_name:
                    case.expected_output = a / b if b != 0 else None
                else:  # 默认加法
                    case.expected_output = a + b
                    
        elif function_name == 'factorial' or '阶乘' in user_prompt:
            test_cases = [
                CodeTestCase(
                    name="基础测试",
                    inputs={"n": 5},
                    expected_output=120,
                    description="测试5的阶乘"
                ),
                CodeTestCase(
                    name="边界测试",
                    inputs={"n": 0},
                    expected_output=1,
                    description="测试0的阶乘"
                ),
                CodeTestCase(
                    name="小数测试",
                    inputs={"n": 3},
                    expected_output=6,
                    description="测试3的阶乘"
                )
            ]
        else:
            # 通用测试用例
            test_cases = [
                CodeTestCase(
                    name="基础测试",
                    inputs={},
                    expected_output=None,
                    description="基本功能测试"
                )
            ]
        
        return test_cases
    
    def _generate_verification_code(self, function_name: str, test_cases: List[CodeTestCase]) -> str:
        """
        生成验证代码
        
        Args:
            function_name: 函数名
            test_cases: 测试用例列表
            
        Returns:
            可直接执行的验证代码
        """
        # 生成测试代码
        test_code = f"""
def verify_function(response_code: str) -> bool:
    \"\"\"
    验证函数代码的正确性
    
    Args:
        response_code: 待验证的响应代码
        
    Returns:
        验证是否通过 (True/False)
    \"\"\"
    try:
        # 执行响应代码以定义函数
        exec(response_code, globals())
        
        # 检查函数是否存在
        if '{function_name}' not in globals():
            return False
        
        # 获取函数引用
        func = globals()['{function_name}']
        
        # 执行测试用例
        test_results = []
        
"""
        
        # 添加具体的测试用例
        for i, case in enumerate(test_cases):
            if case.expected_output is not None and case.inputs:
                # 构建参数调用
                params = ", ".join(f"{k}={repr(v)}" for k, v in case.inputs.items())
                test_code += f"""
        # 测试用例 {i+1}: {case.description}
        try:
            result_{i} = func({params})
            expected_{i} = {repr(case.expected_output)}
            test_results.append(result_{i} == expected_{i})
        except Exception:
            test_results.append(False)
"""
        
        test_code += """
        # 返回所有测试是否都通过
        return len(test_results) > 0 and all(test_results)
        
    except Exception:
        return False

# 执行验证
verification_result = verify_function(response_code)
"""
        
        return test_code.strip()
    
    def _generate_llm_verification_points(self, user_prompt: str) -> List[LLMVerificationPoint]:
        """
        生成LLM验证点
        
        Args:
            user_prompt: 用户提示
            
        Returns:
            LLM验证点列表
        """
        verification_points = []
        
        # 根据任务类型生成不同的验证点
        if self._should_use_code_verification(user_prompt):
            # 代码任务的LLM验证点
            verification_points = [
                LLMVerificationPoint(
                    id="function_name_accuracy",
                    description="函数名是否符合要求和规范？",
                    evaluation_criteria="检查函数名是否与用户要求一致，是否遵循命名规范"
                ),
                LLMVerificationPoint(
                    id="code_style",
                    description="代码风格是否良好？",
                    evaluation_criteria="检查代码是否清晰、简洁、可读性好"
                )
            ]
        else:
            # 非代码任务的LLM验证点
            if any(keyword in user_prompt for keyword in ['字数', '字以内', '不超过', '限制']):
                verification_points.append(
                    LLMVerificationPoint(
                        id="length_requirement",
                        description="是否符合长度/字数要求？",
                        evaluation_criteria="检查内容长度是否符合用户的具体要求"
                    )
                )
            
            verification_points.extend([
                LLMVerificationPoint(
                    id="content_relevance",
                    description="内容是否与主题相关？",
                    evaluation_criteria="检查回应内容是否切合用户提示的主题和要求"
                ),
                LLMVerificationPoint(
                    id="content_quality",
                    description="内容质量是否良好？",
                    evaluation_criteria="评估内容的准确性、完整性和表达质量"
                ),
                LLMVerificationPoint(
                    id="format_appropriateness",
                    description="格式和结构是否合适？",
                    evaluation_criteria="检查内容的组织结构和格式是否清晰合理"
                )
            ])
        
        return verification_points
    
    def _create_basic_code_verification(self) -> CodeVerificationPoint:
        """
        创建基础的代码验证点
        
        Returns:
            基础代码验证点
        """
        test_case = CodeTestCase(
            name="基础测试",
            inputs={},
            expected_output=None,
            description="基本代码语法检查"
        )
        
        verification_code = """
def verify_function(response_code: str) -> bool:
    \"\"\"基本的代码语法验证\"\"\"
    try:
        # 检查是否包含函数定义
        if 'def ' not in response_code:
            return False
        
        # 尝试执行代码
        exec(response_code, {})
        return True
    except Exception:
        return False

verification_result = verify_function(response_code)
"""
        
        return CodeVerificationPoint(
            function_name="unknown_function",
            test_cases=[test_case],
            verification_code=verification_code,
            description="基本代码验证"
        )
    
    def _create_fallback_contract(self, user_prompt: str) -> VerificationContract:
        """
        创建兜底的验证契约
        
        Args:
            user_prompt: 用户提示
            
        Returns:
            基础的验证契约
        """
        logger.warning("使用兜底契约")
        
        # 创建基础的LLM验证点
        llm_verification_points = [
            LLMVerificationPoint(
                id="basic_relevance",
                description="回应是否与用户提示相关？",
                evaluation_criteria="检查回应是否切合用户的问题或要求"
            ),
            LLMVerificationPoint(
                id="basic_coherence",
                description="回应是否逻辑清晰、表达连贯？",
                evaluation_criteria="评估回应的逻辑性和表达清晰度"
            )
        ]
        
        return VerificationContract(
            code_verification=None,
            llm_verification_points=llm_verification_points
        )
    
    def batch_generate_contracts(self, prompts: List[str]) -> List[VerificationContract]:
        """
        批量生成验证契约
        
        Args:
            prompts: 用户提示列表
            
        Returns:
            验证契约列表
        """
        contracts = []
        
        for i, prompt in enumerate(prompts):
            logger.info(f"处理第{i+1}/{len(prompts)}个提示")
            try:
                contract = self.generate_contract(prompt)
                contracts.append(contract)
            except Exception as e:
                logger.error(f"第{i+1}个提示处理失败: {str(e)}")
                contracts.append(self._create_fallback_contract(prompt))
        
        return contracts 