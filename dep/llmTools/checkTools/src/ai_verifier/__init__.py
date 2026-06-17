"""
AI回应验证系统 - 工业级混合验证模式（重构版本）

这是一个完整的AI回应验证框架，采用平均值计算的得分系统。
重构后的特性：
1. 智能验证策略：根据任务类型选择验证方式
2. 抽象化接口：支持自定义LLM提供商和沙箱执行器
3. 平均值得分：所有验证点的平均值作为总得分
4. 新数据模型：更清晰的验证结果结构

主要模块:
- models: 数据模型定义
- abstractions: 抽象接口定义
- contract_generator: 验证契约生成器
- code_executor: 代码验证执行器  
- judge_evaluator: LLM评判器
- score_calculator: 得分计算器
- verifier: 主验证器
"""

__version__ = "2.0.0"
__author__ = "AI Verifier Team"

# 核心组件
from .config import Config
from .verifier import AIVerifier

# 数据模型
from .models import (
    VerificationContract,
    CodeVerificationPoint,
    LLMVerificationPoint,
    CodeTestCase,
    VerificationPointResult,
    VerificationPointType,
    CodeVerificationResult,
    LLMVerificationResult,
    OverallVerificationResult
)

# 抽象接口
from .abstractions import (
    LLMProvider,
    SandboxExecutor,
    LLMRequest,
    LLMResponse,
    CodeExecutionRequest,
    CodeExecutionResult,
    set_llm_provider,
    set_sandbox_executor,
    get_llm_provider,
    get_sandbox_executor
)

# 组件（通常不需要直接使用，但可以导入用于扩展）
from .contract_generator import ContractGenerator
from .code_executor import CodeExecutor
from .judge_evaluator import JudgeEvaluator
from .score_calculator import ScoreCalculator

__all__ = [
    # 核心
    "Config",
    "AIVerifier",
    
    # 数据模型
    "VerificationContract",
    "CodeVerificationPoint", 
    "LLMVerificationPoint",
    "CodeTestCase",
    "VerificationPointResult",
    "VerificationPointType",
    "CodeVerificationResult",
    "LLMVerificationResult", 
    "OverallVerificationResult",
    
    # 抽象接口
    "LLMProvider",
    "SandboxExecutor",
    "LLMRequest",
    "LLMResponse",
    "CodeExecutionRequest",
    "CodeExecutionResult",
    "set_llm_provider",
    "set_sandbox_executor",
    "get_llm_provider",
    "get_sandbox_executor",
    
    # 组件
    "ContractGenerator",
    "CodeExecutor",
    "JudgeEvaluator", 
    "ScoreCalculator"
] 