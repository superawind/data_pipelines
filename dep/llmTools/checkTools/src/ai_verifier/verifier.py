#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主验证器模块（重构版本）

集成所有验证组件，提供统一的AI回应验证接口。
重构后的特性：
1. 使用新的数据模型和抽象接口
2. 平均值计算的得分系统
3. 简化的验证流程
"""

import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, List, Optional

from .config import Config
from .contract_generator import ContractGenerator
from .code_executor import CodeExecutor
from .judge_evaluator import JudgeEvaluator
from .score_calculator import ScoreCalculator
from .models import (
    VerificationContract,
    OverallVerificationResult,
    CodeVerificationResult,
    LLMVerificationResult
)

logger = logging.getLogger(__name__)


class AIVerifier:
    """AI回应验证器（重构版本）"""
    
    def __init__(self, config: Optional[Config] = None):
        """
        初始化验证器
        
        Args:
            config: 配置对象，如果为None则使用默认配置
        """
        self.config = config or Config()
        
        # 验证配置
        if not self.config.validate():
            logger.warning("配置验证失败，可能影响系统功能")
        
        # 初始化各组件
        self.contract_generator = ContractGenerator(self.config)
        self.code_executor = CodeExecutor(self.config)
        self.judge_evaluator = JudgeEvaluator(self.config)
        self.score_calculator = ScoreCalculator(self.config)
        
        logger.info("AI验证器初始化完成")
    
    def verify_response(
        self,
        prompt: str,
        response: str,
        contract: Optional[VerificationContract] = None
    ) -> Dict[str, Any]:
        """
        验证AI回应
        
        Args:
            prompt: 原始用户提示
            response: AI回应内容
            contract: 预生成的验证契约（可选）
            
        Returns:
            验证结果字典
        """
        start_time = time.time()
        
        logger.info(f"开始验证AI回应: {prompt[:100]}...")
        
        try:
            # 步骤1: 生成或使用验证契约
            if contract is None:
                logger.info("生成验证契约...")
                contract = self.contract_generator.generate_contract(prompt)
            else:
                logger.info("使用预提供的验证契约")
            
            # 步骤2: 执行代码验证（如果需要）
            code_result = None
            if contract.has_code_verification():
                logger.info("执行代码验证...")
                code_result = self.code_executor.execute_code_verification(
                    contract.code_verification, 
                    response
                )
            else:
                logger.info("跳过代码验证（契约中不包含代码验证）")
            
            # 步骤3: LLM评判
            logger.info("执行LLM评判...")
            llm_result = self.judge_evaluator.evaluate_llm_verification_points(
                prompt, 
                response, 
                contract.llm_verification_points,
                code_result
            )
            
            # 步骤4: 计算整体结果
            logger.info("计算整体验证结果...")
            overall_result = self.score_calculator.calculate_overall_result(
                contract, 
                code_result, 
                llm_result
            )
            
            # 设置处理时间
            total_time = time.time() - start_time
            overall_result.processing_time = total_time
            
            # 生成综合报告
            comprehensive_report = self.score_calculator.generate_comprehensive_report(
                prompt, response, overall_result
            )
            
            logger.info(f"验证完成，总耗时: {total_time:.3f}秒，结果: {overall_result.verdict}")
            
            return {
                "success": True,
                "verification_result": overall_result,
                "scores": overall_result.to_simple_scores(),
                "detailed_report": comprehensive_report,
                "contract": contract,
                "processing_time": total_time,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            error_time = time.time() - start_time
            logger.error(f"验证过程失败: {str(e)}")
            
            return {
                "success": False,
                "error": str(e),
                "processing_time": error_time,
                "timestamp": datetime.now().isoformat(),
                "partial_results": {
                    "contract": contract
                }
            }
    
    def verify_batch(
        self,
        data: List[Dict[str, Any]],
        use_existing_contracts: bool = True
    ) -> Dict[str, Any]:
        """
        批量验证AI回应
        
        Args:
            data: 包含prompt和response的数据列表
            use_existing_contracts: 是否使用数据中已有的verify契约
            
        Returns:
            批量验证结果
        """
        start_time = time.time()
        total_count = len(data)
        
        logger.info(f"开始批量验证，共{total_count}条数据")
        
        overall_results = []
        verification_results = []
        success_count = 0
        
        for i, item in enumerate(data, 1):
            logger.info(f"处理第{i}/{total_count}条数据")
            
            try:
                prompt = item.get("prompt", "")
                response = item.get("response", "")
                contract = None
                
                if use_existing_contracts and "verify" in item:
                    # 如果数据中包含验证契约，需要转换为VerificationContract对象
                    contract_data = item["verify"]
                    contract = self._convert_dict_to_contract(contract_data)
                
                if not prompt or not response:
                    logger.warning(f"第{i}条数据缺少prompt或response，跳过")
                    continue
                
                # 单条验证
                result = self.verify_response(prompt, response, contract)
                
                if result["success"]:
                    success_count += 1
                    overall_results.append(result["verification_result"])
                    verification_results.append(result)
                else:
                    logger.error(f"第{i}条数据验证失败: {result.get('error', '未知错误')}")
                
            except Exception as e:
                logger.error(f"第{i}条数据处理异常: {str(e)}")
        
        # 计算批量统计
        if overall_results:
            batch_statistics = self.score_calculator.calculate_batch_statistics(overall_results)
        else:
            batch_statistics = {"error": "没有成功验证的数据"}
        
        total_time = time.time() - start_time
        
        batch_result = {
            "summary": {
                "total_items": total_count,
                "processed_items": len(verification_results),
                "success_count": success_count,
                "failure_count": total_count - success_count,
                "overall_pass_rate": round((success_count / total_count) * 100, 2) if total_count > 0 else 0,
                "total_processing_time": round(total_time, 3),
                "average_time_per_item": round(total_time / total_count, 3) if total_count > 0 else 0
            },
            "statistics": batch_statistics,
            "individual_results": verification_results,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"批量验证完成，成功{success_count}/{total_count}条，总耗时: {total_time:.3f}秒")
        
        return batch_result
    
    def generate_contract(self, prompt: str) -> VerificationContract:
        """
        生成验证契约
        
        Args:
            prompt: 用户提示
            
        Returns:
            验证契约对象
        """
        logger.info("生成单个验证契约")
        return self.contract_generator.generate_contract(prompt)
    
    def generate_contracts_batch(self, prompts: List[str]) -> List[VerificationContract]:
        """
        批量生成验证契约
        
        Args:
            prompts: 提示列表
            
        Returns:
            验证契约列表
        """
        logger.info(f"开始批量生成{len(prompts)}个验证契约")
        contracts = self.contract_generator.batch_generate_contracts(prompts)
        logger.info("批量验证契约生成完成")
        return contracts
    
    def create_simple_score_output(self, verification_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        创建简化的得分输出（用户需求格式）
        
        Args:
            verification_result: 完整的验证结果
            
        Returns:
            简化的得分字典，每个验证点为0或1分
        """
        if not verification_result.get("success", False):
            return {"error": "验证失败", "scores": {}}
        
        # 如果已经有scores字段，直接返回
        if "scores" in verification_result:
            return verification_result["scores"]
        
        # 如果有verification_result对象，转换为简化格式
        if "verification_result" in verification_result:
            overall_result = verification_result["verification_result"]
            if isinstance(overall_result, OverallVerificationResult):
                return overall_result.to_simple_scores()
        
        logger.warning("无法提取简化得分信息")
        return {"error": "无法提取得分信息", "scores": {}}
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        获取系统状态信息
        
        Returns:
            系统状态字典
        """
        # 检查各组件可用性
        code_executor_available = self.code_executor.is_available()
        judge_evaluator_available = self.judge_evaluator.is_available()
        
        config_summary = self.config.get_summary()
        
        return {
            "config": config_summary,
            "components": {
                "contract_generator": "已加载",
                "code_executor": "可用" if code_executor_available else "不可用",
                "judge_evaluator": "可用" if judge_evaluator_available else "不可用",
                "score_calculator": "已加载"
            },
            "status": "运行正常" if code_executor_available and judge_evaluator_available else "部分组件不可用",
            "abstractions": {
                "llm_provider": "已设置" if judge_evaluator_available else "未设置",
                "sandbox_executor": "已设置" if code_executor_available else "未设置"
            }
        }
    
    def validate_contract(self, contract: VerificationContract) -> Dict[str, Any]:
        """
        验证契约有效性
        
        Args:
            contract: 待验证的契约
            
        Returns:
            验证结果
        """
        try:
            # 检查基本结构
            if contract.get_total_points() == 0:
                return {
                    "valid": False,
                    "error": "契约不包含任何验证点"
                }
            
            # 检查代码验证安全性（如果有的话）
            if contract.has_code_verification():
                verification_code = contract.code_verification.verification_code
                is_safe, error_msg = self.code_executor._validate_code_safety(verification_code)
                if not is_safe:
                    return {
                        "valid": False,
                        "error": f"验证代码安全检查失败: {error_msg}"
                    }
            
            return {
                "valid": True, 
                "message": "契约验证通过",
                "total_points": contract.get_total_points(),
                "has_code_verification": contract.has_code_verification(),
                "llm_verification_points": len(contract.llm_verification_points)
            }
            
        except Exception as e:
            return {
                "valid": False,
                "error": f"契约验证失败: {str(e)}"
            }
    
    def _convert_dict_to_contract(self, contract_data: Dict[str, Any]) -> VerificationContract:
        """
        将字典格式的契约转换为VerificationContract对象
        
        Args:
            contract_data: 字典格式的契约数据
            
        Returns:
            VerificationContract对象
        """
        from .models import CodeVerificationPoint, LLMVerificationPoint, CodeTestCase
        
        # 处理代码验证
        code_verification = None
        if "code_verification" in contract_data:
            code_data = contract_data["code_verification"]
            
            # 创建测试用例（简化处理，这里可能需要更复杂的转换逻辑）
            test_cases = []
            if "test_cases" in code_data:
                for case_data in code_data["test_cases"]:
                    test_case = CodeTestCase(
                        name=case_data.get("name", "test"),
                        inputs=case_data.get("inputs", {}),
                        expected_output=case_data.get("expected_output"),
                        description=case_data.get("description", "")
                    )
                    test_cases.append(test_case)
            
            code_verification = CodeVerificationPoint(
                function_name=code_data.get("function_name", "unknown"),
                test_cases=test_cases,
                verification_code=code_data.get("verification_code", ""),
                description=code_data.get("description", "")
            )
        
        # 处理LLM验证点
        llm_verification_points = []
        if "llm_verification_points" in contract_data:
            for point_data in contract_data["llm_verification_points"]:
                llm_point = LLMVerificationPoint(
                    id=point_data.get("id", ""),
                    description=point_data.get("description", ""),
                    evaluation_criteria=point_data.get("evaluation_criteria", ""),
                    weight=point_data.get("weight", 1.0)
                )
                llm_verification_points.append(llm_point)
        
        return VerificationContract(
            code_verification=code_verification,
            llm_verification_points=llm_verification_points
        ) 