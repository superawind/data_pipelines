#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成验证契约脚本

读取jsonl格式的数据文件，根据prompt字段生成验证契约，
并将结果保存到verify字段中。
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_verifier import Config
from ai_verifier.contract_generator import ContractGenerator

logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO"):
    """设置日志"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def read_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """
    读取jsonl文件
    
    Args:
        file_path: jsonl文件路径
        
    Returns:
        数据列表
    """
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    item = json.loads(line)
                    data.append(item)
                except json.JSONDecodeError as e:
                    logger.error(f"第{line_num}行JSON解析失败: {e}")
                    
    except FileNotFoundError:
        logger.error(f"文件未找到: {file_path}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"读取文件失败: {e}")
        sys.exit(1)
        
    return data


def write_jsonl(data: List[Dict[str, Any]], file_path: str):
    """
    写入jsonl文件
    
    Args:
        data: 数据列表
        file_path: 输出文件路径
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
                
        logger.info(f"结果已保存到: {file_path}")
    except Exception as e:
        logger.error(f"写入文件失败: {e}")
        sys.exit(1)


def contract_to_dict(contract) -> Dict[str, Any]:
    """
    将VerificationContract对象转换为字典格式
    
    Args:
        contract: VerificationContract对象
        
    Returns:
        字典格式的契约数据
    """
    from ai_verifier.models import VerificationContract
    
    if not isinstance(contract, VerificationContract):
        # 如果已经是字典格式，直接返回
        return contract
    
    contract_dict = {}
    
    # 处理代码验证部分
    if contract.has_code_verification():
        code_verification = contract.code_verification
        test_cases = []
        for case in code_verification.test_cases:
            test_cases.append({
                "name": case.name,
                "inputs": case.inputs,
                "expected_output": case.expected_output,
                "description": case.description
            })
        
        contract_dict["code_verification"] = {
            "function_name": code_verification.function_name,
            "test_cases": test_cases,
            "verification_code": code_verification.verification_code,
            "description": code_verification.description
        }
    
    # 处理LLM验证点
    llm_verification_points = []
    for point in contract.llm_verification_points:
        llm_verification_points.append({
            "id": point.id,
            "description": point.description,
            "evaluation_criteria": point.evaluation_criteria,
            "weight": point.weight
        })
    
    contract_dict["llm_verification_points"] = llm_verification_points
    
    return contract_dict


def generate_contracts_for_data(data: List[Dict[str, Any]], config: Config) -> List[Dict[str, Any]]:
    """
    为数据生成验证契约
    
    Args:
        data: 输入数据列表
        config: 配置对象
        
    Returns:
        包含验证契约的数据列表
    """
    contract_generator = ContractGenerator(config)
    results = []
    
    for i, item in enumerate(data, 1):
        logger.info(f"处理第{i}/{len(data)}条数据")
        
        # 检查必需字段
        if 'prompt' not in item:
            logger.warning(f"第{i}条数据缺少prompt字段，跳过")
            continue
            
        prompt = item['prompt']
        if not prompt or not prompt.strip():
            logger.warning(f"第{i}条数据prompt为空，跳过")
            continue
            
        try:
            # 生成验证契约
            logger.debug(f"为prompt生成验证契约: {prompt[:100]}...")
            verify_contract = contract_generator.generate_contract(prompt)
            
            # 转换为字典格式以便JSON序列化
            verify_contract_dict = contract_to_dict(verify_contract)
            
            # 创建新的数据项
            new_item = item.copy()
            new_item['verify'] = verify_contract_dict
            
            results.append(new_item)
            logger.debug(f"第{i}条数据验证契约生成成功")
            
        except Exception as e:
            logger.error(f"第{i}条数据验证契约生成失败: {e}")
            # 保留原数据，但不添加verify字段
            results.append(item.copy())
            
    return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="生成验证契约")
    parser.add_argument(
        "input_file", 
        help="输入的jsonl文件路径"
    )
    parser.add_argument(
        "-o", "--output", 
        help="输出文件路径（默认为输入文件名_with_contracts.jsonl）"
    )
    parser.add_argument(
        "--in-place", 
        action="store_true",
        help="就地更新原文件，而不是创建新文件"
    )
    parser.add_argument(
        "--log-level", 
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别"
    )
    parser.add_argument(
        "--overwrite-verify",
        action="store_true",
        help="覆盖已存在的verify字段"
    )
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.log_level)
    
    # 确定输出文件路径
    if args.in_place:
        output_file = args.input_file
        logger.info("就地更新模式：将直接更新原文件")
    elif args.output:
        output_file = args.output
    else:
        input_path = Path(args.input_file)
        output_file = str(input_path.parent / f"{input_path.stem}_with_contracts.jsonl")
    
    logger.info(f"输入文件: {args.input_file}")
    logger.info(f"输出文件: {output_file}")
    
    # 读取数据
    logger.info("读取输入数据...")
    all_data = read_jsonl(args.input_file)
    logger.info(f"共读取 {len(all_data)} 条数据")
    
    if not all_data:
        logger.warning("没有读取到有效数据")
        return
    
    # 分离需要处理和已经处理的数据
    need_processing = []
    already_processed = []
    
    for item in all_data:
        if 'verify' not in item or args.overwrite_verify:
            need_processing.append(item)
        else:
            already_processed.append(item)
    
    logger.info(f"需要处理的数据: {len(need_processing)} 条")
    logger.info(f"已有verify字段的数据: {len(already_processed)} 条")
    
    if len(need_processing) == 0:
        logger.info("所有数据都已有verify字段，使用--overwrite-verify参数来强制重新生成")
        return
    
    # 初始化配置
    config = Config()
    
    # 生成验证契约
    logger.info("开始生成验证契约...")
    processed_results = generate_contracts_for_data(need_processing, config)
    
    # 合并结果：保持原始顺序
    final_results = []
    processed_index = 0
    
    for original_item in all_data:
        if 'verify' not in original_item or args.overwrite_verify:
            # 使用处理后的数据
            if processed_index < len(processed_results):
                final_results.append(processed_results[processed_index])
                processed_index += 1
            else:
                final_results.append(original_item)  # 兜底
        else:
            # 保留原有数据
            final_results.append(original_item)
    
    # 写入结果
    write_jsonl(final_results, output_file)
    
    # 统计结果
    with_verify = len([item for item in final_results if 'verify' in item])
    logger.info(f"处理完成！总共 {len(final_results)} 条数据，其中 {with_verify} 条包含验证契约")


if __name__ == "__main__":
    main() 