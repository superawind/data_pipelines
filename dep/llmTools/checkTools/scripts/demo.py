#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI验证系统演示脚本

展示系统的主要功能，包括单条验证、批量验证和简化得分输出。
"""

import json
import logging
import sys
from pathlib import Path

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_verifier import AIVerifier, Config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def create_demo_data():
    """创建演示数据"""
    return [
        {
            "prompt": "用Python写一个函数 calculate_average(numbers) 计算数字列表的平均值，要求函数名准确且回应长度不超过150字符",
            "response": "def calculate_average(numbers):\n    if not numbers:\n        return 0\n    return sum(numbers) / len(numbers)"
        },
        {
            "prompt": "写一篇80字以内的科技文章，介绍人工智能，语气要专业",
            "response": "人工智能是计算机科学的前沿领域，通过机器学习、深度学习等技术模拟人类智能。AI在图像识别、自然语言处理、自动驾驶等领域取得突破性进展，正深刻改变着社会生产方式和生活方式，成为推动数字经济发展的重要引擎。"
        }
    ]

def demo_single_verification():
    """演示单条验证"""
    print("\n" + "="*60)
    print("🔍 单条验证演示")
    print("="*60)
    
    # 初始化验证器
    verifier = AIVerifier()
    
    # 测试数据
    prompt = "用Python写一个函数 fibonacci(n) 计算斐波那契数列第n项，要求递归实现"
    response = "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)"
    
    print(f"📝 用户提示: {prompt}")
    print(f"🤖 AI回应: {response}")
    print("\n⚙️ 开始验证...")
    
    # 执行验证
    result = verifier.verify_response(prompt, response)
    
    if result["success"]:
        # 显示简化得分
        simple_scores = verifier.create_simple_score_output(result)
        print("\n📊 验证得分:")
        print(json.dumps(simple_scores, ensure_ascii=False, indent=2))
        
        # 显示详细摘要
        summary = result["result"]["summary"]
        print(f"\n📈 验证摘要:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
    else:
        print(f"❌ 验证失败: {result.get('error', '未知错误')}")

def demo_batch_verification():
    """演示批量验证"""
    print("\n" + "="*60)
    print("📦 批量验证演示")
    print("="*60)
    
    # 初始化验证器
    verifier = AIVerifier()
    
    # 创建演示数据
    demo_data = create_demo_data()
    
    print(f"📋 待验证数据: {len(demo_data)}条")
    
    # 执行批量验证
    batch_result = verifier.verify_batch(demo_data, use_existing_contracts=False)
    
    print(f"\n📊 批量验证结果:")
    summary = batch_result["summary"]
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    print(f"\n📈 统计信息:")
    if "statistics" in batch_result and "summary" in batch_result["statistics"]:
        stats = batch_result["statistics"]["summary"]
        for key, value in stats.items():
            print(f"  {key}: {value}")

def demo_contract_generation():
    """演示验证契约生成"""
    print("\n" + "="*60)
    print("📄 验证契约生成演示")
    print("="*60)
    
    verifier = AIVerifier()
    
    prompt = "写一个100字以内的天气预报，语气要亲切友好"
    
    print(f"📝 用户提示: {prompt}")
    print("\n⚙️ 生成验证契约...")
    
    # 生成契约
    contracts = verifier.generate_contracts_only([prompt])
    
    if contracts:
        contract = contracts[0]
        print(f"\n📄 生成的验证契约:")
        
        # 显示代码验证部分
        code_verification = contract["code_verification"]
        print(f"🔧 代码验证类型: {code_verification['validation_type']}")
        print(f"📥 输入来源: {code_verification['input_source_description']}")
        
        # 显示LLM验证点
        llm_verification = contract["llm_verification"]
        print(f"\n🧠 LLM验证点:")
        for i, point in enumerate(llm_verification["evaluation_points"], 1):
            print(f"  {i}. {point['description']}")

def demo_system_status():
    """演示系统状态"""
    print("\n" + "="*60)
    print("🔧 系统状态演示")
    print("="*60)
    
    verifier = AIVerifier()
    status = verifier.get_system_status()
    
    print(f"🏃 系统状态: {status['status']}")
    print(f"🤖 LLM模式: {'模拟模式' if status['llm_client']['mock_mode'] else 'API模式'}")
    print(f"🔑 API密钥: {'已配置' if status['llm_client']['has_api_key'] else '未配置'}")
    print(f"📁 Prompts目录: {status['config']['prompts_dir']}")
    print(f"📊 通过阈值: {status['config']['pass_threshold']}")

def main():
    """主函数"""
    print("🚀 AI验证系统演示")
    print("=" * 60)
    print("这是一个工业级的AI回应验证系统演示")
    print("支持代码验证和LLM判断的双重验证模式")
    print("每个验证点采用0/1得分制")
    
    try:
        # 系统状态
        demo_system_status()
        
        # 单条验证
        demo_single_verification()
        
        # 验证契约生成
        demo_contract_generation()
        
        # 批量验证
        demo_batch_verification()
        
        print("\n" + "="*60)
        print("✅ 演示完成！")
        print("💡 提示：")
        print("  - 设置OPENAI_API_KEY环境变量可启用真实LLM验证")
        print("  - 当前为模拟模式，展示系统架构和流程")
        print("  - 生产环境建议使用Docker容器运行")
        
    except Exception as e:
        print(f"❌ 演示过程出错: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 