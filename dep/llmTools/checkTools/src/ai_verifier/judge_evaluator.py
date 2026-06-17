#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
得分制评判器模块（重构版本）

负责调用LLM进行最终的得分评判。
重构后的特性：
1. 一次性评估所有LLM验证点
2. 直接返回JSON格式的各验证点得分
3. 使用抽象的LLM提供商接口
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List

from .abstractions import get_llm_provider, LLMRequest
from .models import (
    VerificationContract, 
    LLMVerificationPoint, 
    LLMVerificationResult,
    CodeVerificationResult
)
from .config import Config

logger = logging.getLogger(__name__)


class JudgeEvaluator:
    """得分制评判器（重构版本）"""
    
    def __init__(self, config: Config):
        """
        初始化评判器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self._load_prompt_template()
    
    def _load_prompt_template(self) -> None:
        """加载评判器的prompt模板"""
        try:
            prompt_path = Path(self.config.prompts_dir) / "score_evaluator.txt"
            with open(prompt_path, "r", encoding="utf-8") as f:
                self.prompt_template = f.read()
            logger.info("成功加载评判器prompt模板")
        except FileNotFoundError:
            logger.error(f"找不到prompt模板文件: {prompt_path}")
            raise
        except Exception as e:
            logger.error(f"加载prompt模板失败: {str(e)}")
            raise
    
    def evaluate_llm_verification_points(
        self,
        original_prompt: str,
        ai_response: str,
        llm_verification_points: List[LLMVerificationPoint],
        code_verification_result: CodeVerificationResult = None
    ) -> LLMVerificationResult:
        """
        评估LLM验证点
        
        Args:
            original_prompt: 原始用户提示
            ai_response: AI回应内容
            llm_verification_points: LLM验证点列表
            code_verification_result: 代码验证结果（可选）
            
        Returns:
            LLM验证结果
        """
        logger.info(f"开始评估 {len(llm_verification_points)} 个LLM验证点")
        
        if not llm_verification_points:
            logger.warning("没有LLM验证点需要评估")
            return LLMVerificationResult(
                point_scores={},
                success=True
            )
        
        try:
            # 检查LLM提供商是否可用
            llm_provider = get_llm_provider()
            if not llm_provider.is_available():
                logger.error("LLM提供商不可用")
                return self._create_fallback_llm_result(llm_verification_points)
            
            # 构建评判prompt
            judge_prompt = self._build_judge_prompt(
                original_prompt, 
                ai_response, 
                llm_verification_points, 
                code_verification_result
            )
            
            # 调用LLM进行评判
            request = LLMRequest(
                prompt=judge_prompt,
                temperature=self.config.evaluation_temperature,
                max_tokens=self.config.max_tokens_evaluation,
                json_mode=True
            )
            
            response = llm_provider.request(request)
            
            if not response.success:
                logger.error(f"LLM评判失败: {response.error_message}")
                return self._create_fallback_llm_result(llm_verification_points)
            
            # 解析响应
            evaluation_data = json.loads(response.content)
            
            # 验证和提取得分
            point_scores = self._extract_point_scores(evaluation_data, llm_verification_points)
            
            logger.info("LLM评判完成")
            return LLMVerificationResult(
                point_scores=point_scores,
                success=True
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"LLM响应不是有效的JSON: {str(e)}")
            return self._create_fallback_llm_result(llm_verification_points)
        except Exception as e:
            logger.error(f"LLM评判失败: {str(e)}")
            return self._create_fallback_llm_result(llm_verification_points)
    
    def _build_judge_prompt(
        self,
        original_prompt: str,
        ai_response: str,
        llm_verification_points: List[LLMVerificationPoint],
        code_verification_result: CodeVerificationResult = None
    ) -> str:
        """
        构建评判prompt
        
        Args:
            original_prompt: 原始用户提示
            ai_response: AI回应
            llm_verification_points: LLM验证点列表
            code_verification_result: 代码验证结果（可选）
            
        Returns:
            完整的评判prompt
        """
        # 格式化代码验证结果
        code_verification_info = ""
        if code_verification_result:
            if code_verification_result.success:
                status = "通过" if code_verification_result.passed else "失败"
                code_verification_info = f"代码验证结果: {status}\n"
                if code_verification_result.details:
                    code_verification_info += f"详细信息: {code_verification_result.details}\n"
            else:
                code_verification_info = f"代码验证执行失败: {code_verification_result.error_message}\n"
        else:
            code_verification_info = "此任务无代码验证部分\n"
        
        # 构建验证点信息
        verification_points_info = "需要评估的验证点:\n"
        for i, point in enumerate(llm_verification_points, 1):
            verification_points_info += f"{i}. ID: {point.id}\n"
            verification_points_info += f"   描述: {point.description}\n"
            verification_points_info += f"   评估标准: {point.evaluation_criteria}\n\n"
        
        # 替换模板变量
        judge_prompt = self.prompt_template.replace(
            "{{original_prompt}}", original_prompt
        ).replace(
            "{{ai_response}}", ai_response
        ).replace(
            "{{code_verification_info}}", code_verification_info
        ).replace(
            "{{verification_points_info}}", verification_points_info
        ).replace(
            "{{point_ids}}", json.dumps([point.id for point in llm_verification_points])
        )
        
        return judge_prompt
    
    def _extract_point_scores(
        self, 
        evaluation_data: Dict[str, Any], 
        llm_verification_points: List[LLMVerificationPoint]
    ) -> Dict[str, float]:
        """
        从评判结果中提取各验证点得分
        
        Args:
            evaluation_data: LLM返回的评判数据
            llm_verification_points: LLM验证点列表
            
        Returns:
            各验证点得分字典 {point_id: score}
        """
        point_scores = {}
        expected_ids = {point.id for point in llm_verification_points}
        
        # 从evaluation_data中提取得分
        if "verification_scores" in evaluation_data:
            scores_data = evaluation_data["verification_scores"]
            
            for point_id in expected_ids:
                if point_id in scores_data:
                    score_value = scores_data[point_id]
                    # 确保得分是0或1
                    if isinstance(score_value, (int, float)) and score_value in [0, 1]:
                        point_scores[point_id] = float(score_value)
                    else:
                        logger.warning(f"验证点 {point_id} 得分无效: {score_value}，使用0分")
                        point_scores[point_id] = 0.0
                else:
                    logger.warning(f"未找到验证点 {point_id} 的得分，使用0分")
                    point_scores[point_id] = 0.0
        else:
            logger.warning("评判结果中未找到verification_scores，所有验证点使用0分")
            for point_id in expected_ids:
                point_scores[point_id] = 0.0
        
        return point_scores
    
    def _create_fallback_llm_result(self, llm_verification_points: List[LLMVerificationPoint]) -> LLMVerificationResult:
        """
        创建兜底的LLM验证结果
        
        Args:
            llm_verification_points: LLM验证点列表
            
        Returns:
            兜底的LLM验证结果
        """
        logger.warning("使用兜底LLM验证结果")
        
        # 所有验证点都给0分
        point_scores = {point.id: 0.0 for point in llm_verification_points}
        
        return LLMVerificationResult(
            point_scores=point_scores,
            success=False,
            error_message="LLM评判系统异常，使用兜底结果"
        )
    
    def is_available(self) -> bool:
        """
        检查评判器是否可用
        
        Returns:
            是否可用
        """
        try:
            llm_provider = get_llm_provider()
            return llm_provider.is_available()
        except Exception as e:
            logger.error(f"检查LLM评判器可用性失败: {str(e)}")
            return False


# 模拟LLM提供商（默认实现，后续可替换）
class MockLLMProvider:
    """模拟LLM提供商（用于测试和开发）"""
    
    def is_available(self) -> bool:
        """检查是否可用"""
        return True
    
    def request(self, request: LLMRequest):
        """处理LLM请求（模拟实现）"""
        logger.info("使用模拟LLM提供商处理评判请求")
        
        try:
            # 从prompt中提取验证点ID
            import re
            point_ids_match = re.search(r'"point_ids":\s*(\[[^\]]+\])', request.prompt)
            if point_ids_match:
                point_ids = json.loads(point_ids_match.group(1))
            else:
                # 尝试其他方式提取验证点
                point_ids = self._extract_point_ids_from_prompt(request.prompt)
            
            # 生成模拟评分（为了测试效果，让大部分验证点通过）
            verification_scores = {}
            for i, point_id in enumerate(point_ids):
                # 模拟评分：让大部分验证点通过，偶尔失败
                score = 1 if i % 3 != 0 else 0  # 2/3的验证点通过
                verification_scores[point_id] = score
                logger.info(f"模拟评分 {point_id}: {score}")
            
            response_content = json.dumps({
                "verification_scores": verification_scores,
                "reasoning": "这是模拟LLM提供商生成的评分结果，为了测试展示效果"
            }, ensure_ascii=False)
            
            from .abstractions import LLMResponse
            return LLMResponse(
                content=response_content,
                success=True
            )
            
        except Exception as e:
            logger.error(f"模拟LLM处理请求失败: {str(e)}")
            from .abstractions import LLMResponse
            return LLMResponse(
                content="{}",
                success=False,
                error_message=f"模拟LLM处理失败: {str(e)}"
            )
    
    def _extract_point_ids_from_prompt(self, prompt: str) -> List[str]:
        """从prompt中提取验证点ID"""
        point_ids = []
        lines = prompt.split('\n')
        for line in lines:
            if 'ID:' in line:
                # 提取 "ID: point_id" 格式的ID
                match = re.search(r'ID:\s*(\w+)', line)
                if match:
                    point_ids.append(match.group(1))
        return point_ids


# 初始化默认的模拟LLM提供商
def _initialize_default_llm_provider():
    """初始化默认的模拟LLM提供商"""
    try:
        from .abstractions import set_llm_provider
        default_provider = MockLLMProvider()
        set_llm_provider(default_provider)
        logger.info("初始化默认模拟LLM提供商")
    except Exception as e:
        logger.error(f"初始化默认LLM提供商失败: {str(e)}")


# 模块加载时初始化
_initialize_default_llm_provider() 