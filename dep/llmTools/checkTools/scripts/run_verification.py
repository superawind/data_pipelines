#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
执行验证脚本

读取jsonl格式的数据文件，根据verify、prompt、response字段
执行验证并生成验证结果。
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_verifier import AIVerifier, Config

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


def dict_to_contract(contract_dict: Dict[str, Any]):
    """
    将字典格式的契约数据转换为VerificationContract对象
    
    Args:
        contract_dict: 字典格式的契约数据
        
    Returns:
        VerificationContract对象
    """
    from ai_verifier.models import VerificationContract, CodeVerificationPoint, LLMVerificationPoint, CodeTestCase
    
    if hasattr(contract_dict, 'get_total_points'):
        # 如果已经是VerificationContract对象，直接返回
        return contract_dict
    
    # 处理代码验证部分
    code_verification = None
    if "code_verification" in contract_dict:
        code_data = contract_dict["code_verification"]
        
        # 创建测试用例
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
    if "llm_verification_points" in contract_dict:
        for point_data in contract_dict["llm_verification_points"]:
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


def verify_data_items(data: List[Dict[str, Any]], config: Config) -> List[Dict[str, Any]]:
    """
    对数据进行验证
    
    Args:
        data: 输入数据列表
        config: 配置对象
        
    Returns:
        包含验证结果的数据列表
    """
    verifier = AIVerifier(config)
    results = []
    
    success_count = 0
    total_items = len(data)
    
    for i, item in enumerate(data, 1):
        logger.info(f"验证第{i}/{total_items}条数据")
        
        # 检查必需字段
        required_fields = ['prompt', 'response', 'verify']
        missing_fields = [field for field in required_fields if field not in item]
        
        if missing_fields:
            logger.warning(f"第{i}条数据缺少字段: {missing_fields}，跳过")
            new_item = item.copy()
            new_item['verification_error'] = f"缺少必需字段: {missing_fields}"
            results.append(new_item)
            continue
            
        prompt = item['prompt']
        response = item['response']
        verify_contract = item['verify']
        
        if not prompt or not prompt.strip():
            logger.warning(f"第{i}条数据prompt为空，跳过")
            new_item = item.copy()
            new_item['verification_error'] = "prompt为空"
            results.append(new_item)
            continue
            
        if not response or not response.strip():
            logger.warning(f"第{i}条数据response为空，跳过")
            new_item = item.copy()
            new_item['verification_error'] = "response为空"
            results.append(new_item)
            continue
            
        try:
            # 执行验证
            logger.debug(f"验证prompt: {prompt[:100]}...")
            logger.debug(f"验证response: {response[:100]}...")
            
            # 转换契约格式
            contract_obj = dict_to_contract(verify_contract)
            
            verification_result = verifier.verify_response(
                prompt=prompt,
                response=response,
                contract=contract_obj
            )
            
            # 创建新的数据项
            new_item = item.copy()
            
            # 将验证结果转换为可序列化的格式
            if verification_result.get('success', False):
                # 获取简化的得分信息
                simple_scores = verification_result.get('scores', {})
                new_item['scores'] = simple_scores
                
                # 获取详细报告（转换为字典格式）
                detailed_report = verification_result.get('detailed_report', {})
                new_item['verification_result'] = detailed_report
                
                success_count += 1
                logger.debug(f"第{i}条数据验证成功")
            else:
                new_item['verification_error'] = verification_result.get('error', '验证失败')
                logger.warning(f"第{i}条数据验证失败: {new_item['verification_error']}")
            
            results.append(new_item)
            
        except Exception as e:
            logger.error(f"第{i}条数据验证异常: {e}")
            new_item = item.copy()
            new_item['verification_error'] = f"验证异常: {str(e)}"
            results.append(new_item)
            
    logger.info(f"验证完成！成功验证 {success_count}/{total_items} 条数据")
    return results


def generate_summary_report(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    生成汇总报告
    
    Args:
        data: 验证结果数据
        
    Returns:
        汇总报告
    """
    total_items = len(data)
    successful_verifications = 0
    failed_verifications = 0
    error_verifications = 0
    
    # 得分统计
    total_scores = {}
    score_counts = {}
    
    for item in data:
        if 'verification_error' in item:
            error_verifications += 1
        elif 'scores' in item:
            successful_verifications += 1
            scores = item['scores']
            
            # 统计各个验证点的得分
            for key, value in scores.items():
                if key == 'summary':
                    continue
                if key not in total_scores:
                    total_scores[key] = 0
                    score_counts[key] = 0
                total_scores[key] += value
                score_counts[key] += 1
        else:
            failed_verifications += 1
    
    # 计算平均得分
    average_scores = {}
    for key in total_scores:
        if score_counts[key] > 0:
            average_scores[key] = round(total_scores[key] / score_counts[key], 3)
    
    # 计算整体通过率
    passed_items = len([item for item in data 
                       if 'scores' in item and 
                       item['scores'].get('summary', {}).get('verdict') == 'PASSED'])
    
    overall_pass_rate = (passed_items / total_items * 100) if total_items > 0 else 0
    
    summary = {
        "total_items": total_items,
        "successful_verifications": successful_verifications,
        "failed_verifications": failed_verifications,
        "error_verifications": error_verifications,
        "overall_pass_rate": round(overall_pass_rate, 2),
        "average_scores": average_scores,
        "statistics": {
            "verification_success_rate": round((successful_verifications / total_items * 100), 2) if total_items > 0 else 0,
            "passed_items": passed_items
        }
    }
    
    return summary


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="执行验证")
    parser.add_argument(
        "input_file", 
        help="输入的jsonl文件路径（需包含verify字段）"
    )
    parser.add_argument(
        "-o", "--output", 
        help="输出文件路径（默认为输入文件名_verified.jsonl）"
    )
    parser.add_argument(
        "--in-place", 
        action="store_true",
        help="就地更新原文件，而不是创建新文件"
    )
    parser.add_argument(
        "--summary", 
        help="汇总报告输出路径（可选）"
    )
    parser.add_argument(
        "--log-level", 
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别"
    )
    parser.add_argument(
        "--overwrite-result",
        action="store_true",
        help="覆盖已存在的verification_result字段"
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
        output_file = str(input_path.parent / f"{input_path.stem}_verified.jsonl")
    
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
        # 只处理有verify字段的数据
        if 'verify' in item and ('verification_result' not in item or args.overwrite_result):
            need_processing.append(item)
        else:
            already_processed.append(item)
    
    logger.info(f"需要验证的数据: {len(need_processing)} 条")
    logger.info(f"无需验证的数据: {len(already_processed)} 条")
    
    if len(need_processing) == 0:
        logger.info("没有需要验证的数据（缺少verify字段或已有verification_result字段）")
        logger.info("使用--overwrite-result参数来强制重新验证")
        return
    
    # 初始化配置
    config = Config()
    
    # 执行验证
    logger.info("开始执行验证...")
    start_time = time.time()
    processed_results = verify_data_items(need_processing, config)
    elapsed_time = time.time() - start_time
    
    # 合并结果：保持原始顺序
    final_results = []
    processed_index = 0
    
    for original_item in all_data:
        if 'verify' in original_item and ('verification_result' not in original_item or args.overwrite_result):
            # 使用验证后的数据
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
    
    # 生成汇总报告
    summary = generate_summary_report(final_results)
    summary["processing_time_seconds"] = round(elapsed_time, 2)
    
    logger.info("=== 验证汇总 ===")
    logger.info(f"总数据量: {summary['total_items']}")
    logger.info(f"验证成功: {summary['successful_verifications']}")
    logger.info(f"验证失败: {summary['failed_verifications']}")
    logger.info(f"验证异常: {summary['error_verifications']}")
    logger.info(f"整体通过率: {summary['overall_pass_rate']}%")
    logger.info(f"处理耗时: {summary['processing_time_seconds']}秒")
    
    # 保存汇总报告
    if args.summary:
        try:
            with open(args.summary, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            logger.info(f"汇总报告已保存到: {args.summary}")
        except Exception as e:
            logger.error(f"保存汇总报告失败: {e}")


if __name__ == "__main__":
    main() 