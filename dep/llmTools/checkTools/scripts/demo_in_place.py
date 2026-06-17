#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
就地更新功能演示

演示如何使用--in-place参数进行就地更新工作流程。
"""

import json
import sys
import tempfile
from pathlib import Path

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def create_demo_data():
    """创建演示数据"""
    demo_data = [
        {
            "prompt": "用Python写一个函数 multiply(a, b) 计算两个数的乘积",
            "response": "def multiply(a, b):\n    return a * b"
        },
        {
            "prompt": "写一篇60字以内的关于机器学习的简介",
            "response": "机器学习是人工智能的重要分支，通过算法让计算机从数据中学习规律，无需明确编程。它广泛应用于图像识别、自然语言处理等领域。"
        },
        {
            "prompt": "用Python创建一个简单的类Calculator，包含加法和减法方法",
            "response": "class Calculator:\n    def add(self, a, b):\n        return a + b\n    \n    def subtract(self, a, b):\n        return a - b"
        }
    ]
    return demo_data


def demo_workflow():
    """演示完整的就地更新工作流程"""
    print("🎯 就地更新功能演示\n")
    
    # 创建临时演示文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8') as f:
        demo_data = create_demo_data()
        for item in demo_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
        demo_file = f.name
    
    print(f"📁 创建演示文件: {demo_file}")
    print(f"📊 初始数据: {len(demo_data)} 条\n")
    
    # 显示初始数据状态
    print("=== 初始数据状态 ===")
    with open(demo_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            item = json.loads(line)
            print(f"记录 {i}: prompt='{item['prompt'][:40]}...', fields={list(item.keys())}")
    
    try:
        # 添加脚本模块到路径
        scripts_dir = Path(__file__).parent
        sys.path.insert(0, str(scripts_dir))
        
        from generate_contracts import read_jsonl, generate_contracts_for_data, write_jsonl
        from run_verification import verify_data_items
        from ai_verifier import Config
        
        # 初始化配置（使用模拟模式）
        config = Config()
        config.use_mock_llm = True
        
        # 步骤1：生成验证契约
        print("\n=== 步骤1：生成验证契约 ===")
        data = read_jsonl(demo_file)
        print(f"📖 读取数据: {len(data)} 条")
        
        results = generate_contracts_for_data(data, config)
        write_jsonl(results, demo_file)  # 就地更新
        print(f"✅ 生成验证契约完成，写回原文件")
        
        # 显示添加verify后的状态
        updated_data = read_jsonl(demo_file)
        with_verify = len([item for item in updated_data if 'verify' in item])
        print(f"📋 包含verify字段: {with_verify}/{len(updated_data)} 条")
        
        # 步骤2：执行验证
        print("\n=== 步骤2：执行验证 ===")
        verification_results = verify_data_items(updated_data, config)
        write_jsonl(verification_results, demo_file)  # 就地更新
        print(f"✅ 执行验证完成，写回原文件")
        
        # 显示最终状态
        final_data = read_jsonl(demo_file)
        with_results = len([item for item in final_data if 'verification_result' in item])
        with_scores = len([item for item in final_data if 'scores' in item])
        print(f"📊 包含验证结果: {with_results}/{len(final_data)} 条")
        print(f"📈 包含得分信息: {with_scores}/{len(final_data)} 条")
        
        # 显示最终数据结构
        print("\n=== 最终数据结构 ===")
        sample = final_data[0]
        print(f"📝 示例记录包含字段: {list(sample.keys())}")
        
        # 显示得分示例
        if 'scores' in sample:
            scores = sample['scores']
            print(f"🎯 得分示例: {list(scores.keys())}")
            if 'summary' in scores:
                summary = scores['summary']
                print(f"📋 验证结果: {summary.get('verdict', 'N/A')} ({summary.get('pass_rate', 0)}%)")
        
        # 提供文件路径供查看
        print(f"\n💡 演示完成！可以查看完整结果文件: {demo_file}")
        print(f"💭 使用以下命令查看结果:")
        print(f"   cat {demo_file}")
        print(f"   head -n 1 {demo_file} | jq .  # 查看第一条记录的完整结构")
        
        return demo_file
        
    except ImportError as e:
        print(f"❌ 无法导入必要模块: {e}")
        return None
    except Exception as e:
        print(f"❌ 演示过程出错: {e}")
        import traceback
        traceback.print_exc()
        return None


def show_command_equivalents(demo_file):
    """显示等效的命令行操作"""
    print("\n=== 等效的命令行操作 ===")
    print("上述演示等效于以下命令行操作:\n")
    
    print("# 1. 生成验证契约")
    print(f"python scripts/generate_contracts.py {demo_file} --in-place\n")
    
    print("# 2. 执行验证")
    print(f"python scripts/run_verification.py {demo_file} --in-place --summary report.json\n")
    
    print("# 3. 查看结果")
    print(f"cat {demo_file}")
    print(f"head -n 1 {demo_file} | jq .  # 需要安装jq工具")


def main():
    """主演示函数"""
    print("🎬 AI验证系统 - 就地更新功能演示\n")
    
    demo_file = demo_workflow()
    
    if demo_file:
        show_command_equivalents(demo_file)
        
        print("\n🎉 演示成功完成！")
        print("\n📝 关键特点:")
        print("• 所有操作都在同一个文件中进行")
        print("• 数据逐步完善：原始 → +verify → +验证结果")
        print("• 支持中断和恢复处理")
        print("• 避免文件名混乱")
        
        print(f"\n🗑️  清理演示文件: rm {demo_file}")
    else:
        print("\n❌ 演示失败，请检查环境配置")


if __name__ == "__main__":
    main() 