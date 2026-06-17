#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
得分计算器模块（重构版本）

负责处理验证结果的得分计算、统计分析和报告生成。
重构后的特性：
1. 适配新的数据模型
2. 计算所有验证点的平均值作为总得分
3. 统一的验证点管理
"""

import logging
from typing import Dict, Any, List, Optional

from .models import (
    OverallVerificationResult,
    VerificationPointResult,
    VerificationPointType,
    CodeVerificationResult,
    LLMVerificationResult,
    VerificationContract
)
from .config import Config

logger = logging.getLogger(__name__)


class ScoreCalculator:
    """得分计算器（重构版本）"""
    
    def __init__(self, config: Config):
        """
        初始化得分计算器
        
        Args:
            config: 配置对象
        """
        self.config = config
    
    def calculate_overall_result(
        self,
        contract: VerificationContract,
        code_result: Optional[CodeVerificationResult] = None,
        llm_result: Optional[LLMVerificationResult] = None
    ) -> OverallVerificationResult:
        """
        计算整体验证结果
        
        Args:
            contract: 验证契约
            code_result: 代码验证结果（可选）
            llm_result: LLM验证结果（可选）
            
        Returns:
            整体验证结果
        """
        logger.info("开始计算整体验证结果")
        
        # 创建整体结果对象
        overall_result = OverallVerificationResult(
            code_result=code_result,
            llm_result=llm_result
        )
        
        # 生成所有验证点结果
        point_results = []
        
        # 添加代码验证点结果
        if contract.has_code_verification() and code_result:
            code_point_result = VerificationPointResult(
                point_id="code_verification",
                point_type=VerificationPointType.CODE,
                score=1.0 if (code_result.success and code_result.passed) else 0.0,
                max_score=1.0,
                passed=code_result.success and code_result.passed,
                details=code_result.details,
                error_message=code_result.error_message if not code_result.success else None
            )
            point_results.append(code_point_result)
        
        # 添加LLM验证点结果
        if llm_result and llm_result.success:
            for point in contract.llm_verification_points:
                score = llm_result.point_scores.get(point.id, 0.0)
                llm_point_result = VerificationPointResult(
                    point_id=point.id,
                    point_type=VerificationPointType.LLM,
                    score=score,
                    max_score=1.0,
                    passed=score >= 1.0,
                    details=f"LLM评估点: {point.description}"
                )
                point_results.append(llm_point_result)
        else:
            # LLM验证失败，所有LLM验证点得0分
            for point in contract.llm_verification_points:
                llm_point_result = VerificationPointResult(
                    point_id=point.id,
                    point_type=VerificationPointType.LLM,
                    score=0.0,
                    max_score=1.0,
                    passed=False,
                    error_message="LLM验证失败"
                )
                point_results.append(llm_point_result)
        
        # 设置验证点结果
        overall_result.point_results = point_results
        
        # 计算得分
        overall_result.calculate_scores()
        
        logger.info(f"整体验证计算完成: {overall_result.verdict}, 平均得分: {overall_result.average_score:.3f}")
        
        return overall_result
    
    def create_simple_score_output(self, overall_result: OverallVerificationResult) -> Dict[str, Any]:
        """
        创建简化的得分输出
        
        Args:
            overall_result: 整体验证结果
            
        Returns:
            简化的得分字典
        """
        return overall_result.to_simple_scores()
    
    def create_detailed_breakdown(self, overall_result: OverallVerificationResult) -> Dict[str, Any]:
        """
        创建详细的得分细分
        
        Args:
            overall_result: 整体验证结果
            
        Returns:
            详细细分字典
        """
        breakdown = {
            "verification_points": [],
            "code_verification": None,
            "llm_verification": [],
            "summary": {
                "total_points": overall_result.total_points,
                "total_score": overall_result.total_score,
                "max_score": overall_result.max_score,
                "average_score": overall_result.average_score,
                "pass_rate": overall_result.pass_rate,
                "verdict": overall_result.verdict
            }
        }
        
        # 添加所有验证点的详细信息
        for point_result in overall_result.point_results:
            point_detail = {
                "point_id": point_result.point_id,
                "point_type": point_result.point_type.value,
                "score": point_result.score,
                "max_score": point_result.max_score,
                "passed": point_result.passed,
                "details": point_result.details,
                "error_message": point_result.error_message
            }
            breakdown["verification_points"].append(point_detail)
            
            # 分类添加到相应的部分
            if point_result.point_type == VerificationPointType.CODE:
                breakdown["code_verification"] = point_detail
            else:
                breakdown["llm_verification"].append(point_detail)
        
        return breakdown
    
    def analyze_failure_patterns(self, overall_result: OverallVerificationResult) -> Dict[str, Any]:
        """
        分析失败模式
        
        Args:
            overall_result: 整体验证结果
            
        Returns:
            失败分析结果
        """
        analysis = {
            "failed_areas": [],
            "failure_reasons": [],
            "improvement_suggestions": [],
            "failure_count": 0,
            "failure_rate": 0.0
        }
        
        failed_points = [point for point in overall_result.point_results if not point.passed]
        analysis["failure_count"] = len(failed_points)
        
        if overall_result.total_points > 0:
            analysis["failure_rate"] = (len(failed_points) / overall_result.total_points) * 100
        
        # 分析不同类型的失败
        code_failures = [p for p in failed_points if p.point_type == VerificationPointType.CODE]
        llm_failures = [p for p in failed_points if p.point_type == VerificationPointType.LLM]
        
        if code_failures:
            analysis["failed_areas"].append("代码验证")
            for failure in code_failures:
                if failure.error_message:
                    analysis["failure_reasons"].append(f"代码验证失败: {failure.error_message}")
                else:
                    analysis["failure_reasons"].append("代码验证不符合要求")
            analysis["improvement_suggestions"].append("检查代码功能实现、语法正确性和测试用例")
        
        if llm_failures:
            analysis["failed_areas"].append(f"LLM验证 ({len(llm_failures)}项)")
            for failure in llm_failures:
                if failure.error_message:
                    analysis["failure_reasons"].append(f"LLM验证失败: {failure.error_message}")
                else:
                    analysis["failure_reasons"].append(f"验证点 {failure.point_id} 不符合要求")
            analysis["improvement_suggestions"].append("重点关注内容质量、格式规范和要求理解")
        
        return analysis
    
    def generate_comprehensive_report(
        self,
        original_prompt: str,
        ai_response: str,
        overall_result: OverallVerificationResult
    ) -> Dict[str, Any]:
        """
        生成综合报告
        
        Args:
            original_prompt: 原始提示
            ai_response: AI回应
            overall_result: 整体验证结果
            
        Returns:
            综合报告
        """
        simple_scores = self.create_simple_score_output(overall_result)
        detailed_breakdown = self.create_detailed_breakdown(overall_result)
        failure_analysis = self.analyze_failure_patterns(overall_result)
        recommendations = self._generate_recommendations(overall_result, failure_analysis)
        
        report = {
            "meta": {
                "prompt_length": len(original_prompt),
                "response_length": len(ai_response),
                "total_verification_points": overall_result.total_points,
                "verification_types": self._get_verification_types_summary(overall_result),
                "config": {
                    "pass_threshold": self.config.pass_threshold,
                    "strict_mode": self.config.strict_mode
                }
            },
            "scores": simple_scores,
            "detailed_breakdown": detailed_breakdown,
            "failure_analysis": failure_analysis,
            "recommendations": recommendations,
            "performance": {
                "processing_time": overall_result.processing_time,
                "code_execution_time": overall_result.code_result.execution_time if overall_result.code_result else 0.0
            }
        }
        
        return report
    
    def _get_verification_types_summary(self, overall_result: OverallVerificationResult) -> Dict[str, int]:
        """
        获取验证类型汇总
        
        Args:
            overall_result: 整体验证结果
            
        Returns:
            验证类型汇总
        """
        summary = {
            "code_verification_points": 0,
            "llm_verification_points": 0
        }
        
        for point in overall_result.point_results:
            if point.point_type == VerificationPointType.CODE:
                summary["code_verification_points"] += 1
            else:
                summary["llm_verification_points"] += 1
        
        return summary
    
    def _generate_recommendations(
        self,
        overall_result: OverallVerificationResult,
        failure_analysis: Dict[str, Any]
    ) -> List[str]:
        """
        生成改进建议
        
        Args:
            overall_result: 整体验证结果
            failure_analysis: 失败分析
            
        Returns:
            建议列表
        """
        recommendations = []
        
        if overall_result.verdict == "FAILED":
            recommendations.append("当前回应未达到通过标准，建议重新审视要求并改进")
            
            # 基于平均分给出具体建议
            if overall_result.average_score < 0.3:
                recommendations.append("得分过低，建议从基础要求开始，逐项检查和改进")
            elif overall_result.average_score < 0.6:
                recommendations.append("得分偏低，建议重点关注失败的验证点并针对性改进")
            
            # 基于失败模式的建议
            if "代码验证" in failure_analysis["failed_areas"]:
                recommendations.append("代码验证失败，建议：1) 检查语法正确性 2) 验证函数逻辑 3) 确保符合命名要求")
            
            if any("LLM验证" in area for area in failure_analysis["failed_areas"]):
                llm_failure_count = len([p for p in overall_result.point_results 
                                       if p.point_type == VerificationPointType.LLM and not p.passed])
                if llm_failure_count > 1:
                    recommendations.append("多项LLM验证失败，建议：1) 仔细阅读要求 2) 检查格式规范 3) 确保内容相关性")
                else:
                    recommendations.append("LLM验证失败，建议根据具体验证点要求进行针对性改进")
        
        elif overall_result.verdict == "PASSED":
            if overall_result.average_score < 0.9:
                recommendations.append("验证通过但仍有改进空间，可进一步优化细节")
            else:
                recommendations.append("验证结果优秀，继续保持高质量输出")
        
        return recommendations
    
    def calculate_batch_statistics(self, results: List[OverallVerificationResult]) -> Dict[str, Any]:
        """
        计算批量统计信息
        
        Args:
            results: 多个验证结果列表
            
        Returns:
            批量统计信息
        """
        if not results:
            return {"error": "没有可统计的结果"}
        
        total_count = len(results)
        passed_count = sum(1 for r in results if r.verdict == "PASSED")
        failed_count = total_count - passed_count
        
        # 计算平均得分
        all_scores = [r.average_score for r in results]
        avg_score = sum(all_scores) / len(all_scores) if all_scores else 0
        
        # 计算各类型验证点的统计
        total_code_points = sum(1 for r in results for p in r.point_results 
                              if p.point_type == VerificationPointType.CODE)
        passed_code_points = sum(1 for r in results for p in r.point_results 
                               if p.point_type == VerificationPointType.CODE and p.passed)
        
        total_llm_points = sum(1 for r in results for p in r.point_results 
                             if p.point_type == VerificationPointType.LLM)
        passed_llm_points = sum(1 for r in results for p in r.point_results 
                              if p.point_type == VerificationPointType.LLM and p.passed)
        
        # 处理时间统计
        processing_times = [r.processing_time for r in results if r.processing_time > 0]
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
        
        return {
            "summary": {
                "total_evaluations": total_count,
                "passed_count": passed_count,
                "failed_count": failed_count,
                "overall_pass_rate": round((passed_count / total_count) * 100, 2),
                "average_score": round(avg_score, 3)
            },
            "verification_points": {
                "code_verification": {
                    "total_points": total_code_points,
                    "passed_points": passed_code_points,
                    "pass_rate": round((passed_code_points / total_code_points) * 100, 2) if total_code_points > 0 else 0
                },
                "llm_verification": {
                    "total_points": total_llm_points,
                    "passed_points": passed_llm_points,
                    "pass_rate": round((passed_llm_points / total_llm_points) * 100, 2) if total_llm_points > 0 else 0
                }
            },
            "performance": {
                "average_processing_time": round(avg_processing_time, 2),
                "min_processing_time": min(processing_times) if processing_times else 0,
                "max_processing_time": max(processing_times) if processing_times else 0
            },
            "score_distribution": {
                "min_score": min(all_scores) if all_scores else 0,
                "max_score": max(all_scores) if all_scores else 0,
                "median_score": sorted(all_scores)[len(all_scores)//2] if all_scores else 0
            }
        } 