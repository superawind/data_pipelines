#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证模型定义

定义验证过程中使用的数据结构和模型。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum


class VerificationPointType(Enum):
    """验证点类型"""
    CODE = "code"
    LLM = "llm"


@dataclass
class CodeTestCase:
    """代码测试用例"""
    name: str                    # 测试用例名称
    inputs: Dict[str, Any]       # 输入参数
    expected_output: Any         # 期望输出
    description: str             # 测试描述


@dataclass
class CodeVerificationPoint:
    """代码验证点"""
    function_name: str                    # 函数名称
    test_cases: List[CodeTestCase]        # 测试用例列表
    verification_code: str                # 验证代码（可直接执行）
    description: str                      # 验证描述
    weight: float = 1.0                   # 权重


@dataclass
class LLMVerificationPoint:
    """LLM验证点"""
    id: str                              # 验证点ID
    description: str                     # 验证描述
    evaluation_criteria: str             # 评估标准
    weight: float = 1.0                  # 权重


@dataclass
class VerificationContract:
    """验证契约"""
    code_verification: Optional[CodeVerificationPoint] = None    # 代码验证点
    llm_verification_points: List[LLMVerificationPoint] = field(default_factory=list)  # LLM验证点列表
    
    def get_total_points(self) -> int:
        """获取总验证点数量"""
        total = len(self.llm_verification_points)
        if self.code_verification:
            total += 1
        return total
    
    def has_code_verification(self) -> bool:
        """是否包含代码验证"""
        return self.code_verification is not None


@dataclass
class VerificationPointResult:
    """单个验证点结果"""
    point_id: str                        # 验证点ID
    point_type: VerificationPointType    # 验证点类型
    score: float                         # 得分 (0.0 或 1.0)
    max_score: float                     # 最大得分 (通常为1.0)
    passed: bool                         # 是否通过
    details: Optional[str] = None        # 详细信息
    error_message: Optional[str] = None  # 错误信息


@dataclass
class CodeVerificationResult:
    """代码验证结果"""
    success: bool                        # 验证是否成功执行
    passed: bool                         # 是否通过验证
    execution_time: float                # 执行时间
    error_message: Optional[str] = None  # 错误信息
    details: Optional[str] = None        # 详细信息


@dataclass
class LLMVerificationResult:
    """LLM验证结果"""
    point_scores: Dict[str, float]       # 各验证点得分 {point_id: score}
    success: bool                        # 评估是否成功
    error_message: Optional[str] = None  # 错误信息
    
    def get_total_score(self) -> float:
        """获取总得分"""
        return sum(self.point_scores.values())
    
    def get_passed_count(self) -> int:
        """获取通过的验证点数量"""
        return sum(1 for score in self.point_scores.values() if score >= 1.0)


@dataclass
class OverallVerificationResult:
    """整体验证结果"""
    code_result: Optional[CodeVerificationResult] = None        # 代码验证结果
    llm_result: Optional[LLMVerificationResult] = None          # LLM验证结果
    point_results: List[VerificationPointResult] = field(default_factory=list)  # 所有验证点结果
    
    total_points: int = 0                # 总验证点数
    total_score: float = 0.0             # 总得分
    max_score: float = 0.0               # 最大可能得分
    average_score: float = 0.0           # 平均得分
    pass_rate: float = 0.0               # 通过率
    verdict: str = "UNKNOWN"             # 总体判定
    
    processing_time: float = 0.0         # 处理时间
    
    def calculate_scores(self) -> None:
        """计算各种得分指标"""
        if not self.point_results:
            return
            
        self.total_points = len(self.point_results)
        self.total_score = sum(result.score for result in self.point_results)
        self.max_score = sum(result.max_score for result in self.point_results)
        
        if self.max_score > 0:
            self.average_score = self.total_score / self.max_score
            self.pass_rate = (self.total_score / self.max_score) * 100
        else:
            self.average_score = 0.0
            self.pass_rate = 0.0
        
        # 判定结果（可配置阈值，这里使用0.6）
        if self.average_score >= 0.6:
            self.verdict = "PASSED"
        else:
            self.verdict = "FAILED"
    
    def to_simple_scores(self) -> Dict[str, Any]:
        """转换为简化的得分格式"""
        scores = {}
        
        # 添加各个验证点得分
        for result in self.point_results:
            scores[result.point_id] = result.score
        
        # 添加汇总信息
        scores["summary"] = {
            "total_points": self.total_points,
            "total_score": self.total_score,
            "max_score": self.max_score,
            "average_score": round(self.average_score, 3),
            "pass_rate": round(self.pass_rate, 1),
            "verdict": self.verdict
        }
        
        return scores 