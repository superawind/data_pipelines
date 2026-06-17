#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
极速去重基类模块

专为千万和亿级别数据设计，提供最高性能的基础设施
"""

import json
import logging
import sys
import time
import os
import datetime
import multiprocessing as mp
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
from collections import deque
import numpy as np

try:
    from .config import DedupConfig
except ImportError:
    from config import DedupConfig


class DedupPreviewResult:
    """去重预览结果类"""
    
    def __init__(self, indices_to_keep: List[int], indices_to_delete: List[int], stats: Dict[str, Any]):
        self.indices_to_keep = indices_to_keep
        self.indices_to_delete = indices_to_delete
        self.stats = stats
        self.duplicate_pairs = []
        self.similarity_examples = []
        self.threshold_examples = []
    
    def add_duplicate_pairs(self, pairs: List[tuple]):
        """添加重复对"""
        self.duplicate_pairs = pairs
    
    def add_similarity_examples(self, examples: List[tuple]):
        """添加相似度示例"""
        self.similarity_examples = examples
    
    def add_threshold_examples(self, examples: List[tuple]):
        """添加阈值示例"""
        self.threshold_examples = examples
    
    def print_summary(self):
        """打印去重结果摘要"""
        print("\n" + "="*60)
        print("去重结果预览")
        print("="*60)
        
        total = self.stats.get('total_lines', len(self.indices_to_keep) + len(self.indices_to_delete))
        kept = len(self.indices_to_keep)
        deleted = len(self.indices_to_delete)
        ratio = self.stats.get('dedup_ratio', deleted / total if total > 0 else 0)
        
        print(f"📊 去重统计:")
        print(f"   原始数据量: {total:,} 行")
        print(f"   保留数据量: {kept:,} 行")
        print(f"   删除数据量: {deleted:,} 行")
        print(f"   去重比例: {ratio:.2%}")
        
        if 'total_time' in self.stats:
            print(f"   处理耗时: {self.stats['total_time']:.2f} 秒")
        
        if 'lsh_size' in self.stats:
            print(f"   LSH索引大小: {self.stats['lsh_size']}")
        
        if 'used_gpu' in self.stats:
            print(f"   使用GPU: {self.stats['used_gpu']}")
    
    def print_examples(self, max_examples: int = 5):
        """打印去重示例"""
        if not self.duplicate_pairs:
            print("\n✅ 没有发现重复内容")
            return
        
        print(f"\n🔍 重复内容对比示例 (显示前 {min(max_examples, len(self.duplicate_pairs))} 个):")
        for i, (original_idx, deleted_idx, content, similarity) in enumerate(self.duplicate_pairs[:max_examples], 1):
            print(f"   重复对 {i}:")
            print(f"     保留 (索引 {original_idx}): {content[:100]}{'...' if len(content) > 100 else ''}")
            print(f"     删除 (索引 {deleted_idx}): {content[:100]}{'...' if len(content) > 100 else ''}")
            if similarity is not None:
                print(f"     相似度: {similarity:.4f}")
            print()
    
    def print_similarity_analysis(self, max_examples: int = 3):
        """打印相似度分析示例"""
        if not self.similarity_examples:
            return
        
        print(f"\n📈 相似度分析示例 (显示前 {min(max_examples, len(self.similarity_examples))} 个):")
        for i, (idx1, idx2, content1, content2, similarity) in enumerate(self.similarity_examples[:max_examples], 1):
            print(f"   相似度示例 {i}:")
            print(f"     向量 {idx1}: {content1[:80]}{'...' if len(content1) > 80 else ''}")
            print(f"     向量 {idx2}: {content2[:80]}{'...' if len(content2) > 80 else ''}")
            print(f"     相似度: {similarity:.4f}")
            print()
    
    def print_threshold_examples(self, max_examples: int = 3):
        """打印阈值边界示例"""
        if not self.threshold_examples:
            return
        
        print(f"\n⚖️ 阈值边界示例 (显示前 {min(max_examples, len(self.threshold_examples))} 个):")
        for i, (idx1, idx2, content1, content2, similarity) in enumerate(self.threshold_examples[:max_examples], 1):
            print(f"   边界示例 {i}:")
            print(f"     向量 {idx1}: {content1[:80]}{'...' if len(content1) > 80 else ''}")
            print(f"     向量 {idx2}: {content2[:80]}{'...' if len(content2) > 80 else ''}")
            print(f"     相似度: {similarity:.4f} (阈值附近)")
            print()
    
    def ask_for_confirmation(self) -> bool:
        """询问用户是否确认写入文件"""
        print("\n" + "="*60)
        print("是否将去重结果写入文件？")
        print("="*60)
        
        while True:
            response = input("请输入 'y' 确认写入文件，'n' 取消，或 's' 查看详细统计: ").strip().lower()
            
            if response == 'y':
                return True
            elif response == 'n':
                return False
            elif response == 's':
                self.print_detailed_stats()
            else:
                print("请输入 'y', 'n' 或 's'")
    
    def print_detailed_stats(self):
        """打印详细统计信息"""
        print("\n📋 详细统计信息:")
        print(f"   保留索引范围: {min(self.indices_to_keep) if self.indices_to_keep else 'N/A'} - {max(self.indices_to_keep) if self.indices_to_keep else 'N/A'}")
        print(f"   删除索引范围: {min(self.indices_to_delete) if self.indices_to_delete else 'N/A'} - {max(self.indices_to_delete) if self.indices_to_delete else 'N/A'}")
        
        if 'malformed_lines' in self.stats:
            print(f"   格式错误行数: {self.stats['malformed_lines']}")
        
        if 'unique_lines' in self.stats:
            print(f"   唯一行数: {self.stats['unique_lines']}")
        
        if 'duplicate_lines' in self.stats:
            print(f"   重复行数: {self.stats['duplicate_lines']}")


class BaseDeduplicator(ABC):
    """去重器基类"""
    
    def __init__(self, config: DedupConfig):
        self.config = config
        self.num_workers = getattr(config, 'num_workers', None)
        self.run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 动态设置日志目录 - 根据使用的去重器类型决定
        if hasattr(config, 'log_dir') and config.log_dir:
            # 如果配置中明确指定了log_dir，使用配置值
            self.log_dir = Path(config.log_dir)
        else:
            # 根据去重器类名自动判断
            class_name = self.__class__.__name__.lower()
            if 'global' in class_name:
                self.log_dir = Path("global_dedup_logs")
            else:
                self.log_dir = Path("stream_dedup_logs")
        
        # 初始化日志
        self.logger = logging.getLogger(self.__class__.__name__)
        self.setup_logging()
        
        # 性能统计
        self.performance_stats = {}
        
        # 去重效果分析 - 增加到用户需要的数量
        self.duplicate_records = deque(maxlen=1000)  # 1000条重复数据示例
        self.threshold_examples = deque(maxlen=1000)  # 1000条接近阈值的数据
        
        # 记录计数器
        self.duplicate_count = 0
        self.threshold_example_count = 0
        
        # 临时目录设置
        self.temp_dir = str(self.log_dir / "temp_dedup")
        Path(self.temp_dir).mkdir(parents=True, exist_ok=True)
    
    def setup_logging(self):
        """设置日志系统"""
        # 清理现有处理器
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        self.logger.setLevel(logging.INFO)
        
        # 创建日志目录
        self.log_dir.mkdir(exist_ok=True)
        
        # 创建时间戳文件名
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        method_name = getattr(self.config, 'method', 'unknown')
        if hasattr(method_name, 'value'):
            method_name = method_name.value
        
        # 日志文件路径 - 根据去重器类型设置前缀
        class_name = self.__class__.__name__.lower()
        if 'global' in class_name:
            prefix = "global_dedup"
        else:
            prefix = "stream_dedup"
        
        main_log_file = self.log_dir / f"{prefix}_{method_name}_{timestamp}.log"
        duplicate_log_file = self.log_dir / f"duplicates_{method_name}_{timestamp}.jsonl"
        threshold_log_file = self.log_dir / f"threshold_examples_{method_name}_{timestamp}.jsonl"
        performance_log_file = self.log_dir / f"performance_{method_name}_{timestamp}.json"
        
        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        
        # 文件处理器
        file_handler = logging.FileHandler(main_log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        
        # 只在主进程中添加处理器
        if mp.current_process().name == 'MainProcess':
            self.logger.addHandler(console_handler)
            self.logger.addHandler(file_handler)
            
            # 根据模式显示不同的启动消息
            class_name = self.__class__.__name__.lower()
            if 'global' in class_name:
                self.logger.info("全局去重日志系统已启动")
            else:
                self.logger.info("流式去重日志系统已启动")
        
        # 保存日志文件路径
        self.log_files = {
            'main': main_log_file,
            'duplicates': duplicate_log_file,
            'threshold_examples': threshold_log_file,
            'performance': performance_log_file
        }
        
    def _get_mode_suffix(self) -> str:
        """根据去重器类型返回正确的模式后缀"""
        class_name = self.__class__.__name__.lower()
        if 'global' in class_name:
            return 'global'
        else:
            return 'streaming'
    
    def _format_json_output(self, data: dict) -> str:
        """根据field_order格式化JSON输出"""
        if not hasattr(self.config, 'field_order') or not self.config.field_order:
            # 如果没有指定字段顺序，直接输出
            return json.dumps(data, ensure_ascii=False)
        
        # 按照field_order重新排序字段
        ordered_data = {}
        
        # 首先添加field_order中指定的字段
        for field in self.config.field_order:
            if field in data:
                ordered_data[field] = data[field]
        # print('========================')
        # print(data)
        # print('========================')
        
        # 然后添加不在field_order中的其他字段
        if isinstance(data, dict):
            for key, value in data.items():
                if key not in ordered_data:
                    ordered_data[key] = value
        elif isinstance(data, tuple):
            for key, value in json.loads(data[1]).items():
                if key not in ordered_data:
                    ordered_data[key] = value
        else:
            print('数据格式存在问题-------------')
        return json.dumps(ordered_data, ensure_ascii=False)
    
    def log_duplicate_record(self, removed_data: dict, kept_data: dict, similarity: float, method_specific_info: dict = None):
        """记录重复数据详情 - 简洁直观格式，支持prompt_id"""
        if self.duplicate_count >= 1000:
            return
        
        # 提取去重字段内容
        removed_content = self._extract_content_for_log(removed_data)
        kept_content = self._extract_content_for_log(kept_data)
        
        # 提取prompt_id信息
        removed_prompt_id = self._extract_prompt_id(removed_data)
        kept_prompt_id = self._extract_prompt_id(kept_data)
        
        # 创建简洁的日志记录
        method_name = str(self.config.method.value) if hasattr(self.config.method, 'value') else str(self.config.method)
        mode_suffix = self._get_mode_suffix()
        record = {
            'type': 'duplicate',
            'removed_prompt_id': removed_prompt_id,
            'kept_prompt_id': kept_prompt_id,
            'removed_content': removed_content,
            'kept_content': kept_content,
            'similarity': float(similarity),
            'content_fields': self.config.content_keys,
            'method': f"{method_name}_{mode_suffix}",
            'record_id': self.duplicate_count + 1
        }
        
        # 添加方法特定信息
        if method_specific_info:
            record.update(method_specific_info)
        
        self.duplicate_records.append(record)
        self.duplicate_count += 1
        
        # 写入重复记录文件 - 简洁格式
        if 'duplicates' in self.log_files:
            try:
                record_str = json.dumps(record, ensure_ascii=False, indent=2)
                with open(self.log_files['duplicates'], 'a', encoding='utf-8') as f:
                    f.write(f"{record_str}\n")
                    f.write("-" * 80 + "\n")  # 分隔线便于阅读
            except Exception as e:
                self.logger.warning(f"写入重复记录失败: {e}")
    
    def log_threshold_example(self, data1: dict, data2: dict, similarity: float, threshold: float, method_specific_info: dict = None):
        """记录接近阈值但未去重的数据 - 简洁直观格式，支持prompt_id"""
        if self.threshold_example_count >= 1000:
            return
        
        # 提取去重字段内容
        content1 = self._extract_content_for_log(data1)
        content2 = self._extract_content_for_log(data2)
        
        # 提取prompt_id信息
        prompt_id1 = self._extract_prompt_id(data1)
        prompt_id2 = self._extract_prompt_id(data2)
        
        # 创建简洁的阈值边界记录
        method_name = str(self.config.method.value) if hasattr(self.config.method, 'value') else str(self.config.method)
        mode_suffix = self._get_mode_suffix()
        example = {
            'type': 'near_threshold',
            'prompt_id1': prompt_id1,
            'prompt_id2': prompt_id2,
            'content1': content1,
            'content2': content2,
            'similarity': float(similarity),
            'threshold': float(threshold),
            'distance_from_threshold': float(threshold - similarity),
            'content_fields': self.config.content_keys,
            'method': f"{method_name}_{mode_suffix}",
            'example_id': self.threshold_example_count + 1
        }
        
        # 添加方法特定信息
        if method_specific_info:
            example.update(method_specific_info)
        
        self.threshold_examples.append(example)
        self.threshold_example_count += 1
        
        # 写入阈值例子文件 - 简洁格式
        if 'threshold_examples' in self.log_files:
            try:
                example_str = json.dumps(example, ensure_ascii=False, indent=2)
                with open(self.log_files['threshold_examples'], 'a', encoding='utf-8') as f:
                    f.write(f"{example_str}\n")
                    f.write("-" * 80 + "\n")  # 分隔线便于阅读
            except Exception as e:
                self.logger.warning(f"写入阈值例子失败: {e}")
    
    def _extract_prompt_id(self, data: dict) -> str:
        """提取数据中的prompt_id"""
        # 尝试多种可能的prompt_id字段名
        prompt_id_keys = []
        
        # 如果配置中指定了prompt_id_key，优先使用
        if hasattr(self.config, 'prompt_id_key'):
            prompt_id_keys.append(self.config.prompt_id_key)
        
        # 添加常见的ID字段名
        prompt_id_keys.extend(['prompt_id', 'id', 'sample_id', 'data_id', 'index'])
        
        for key in prompt_id_keys:
            if key in data:
                return str(data[key])
        
        # 如果没有找到ID字段，返回空字符串或生成一个
        return ""
    
    def _extract_content_for_log(self, data: dict) -> str:
        """提取用于日志的内容"""
        if hasattr(self.config, 'content_keys') and self.config.content_keys:
            content_parts = []
            for key in self.config.content_keys:
                if key in data:
                    content_parts.append(str(data[key]))
            return ' '.join(content_parts)
        else:
            # 如果没有指定content_keys，尝试常见字段
            for key in ['prompt', 'content', 'text', 'instruction']:
                if key in data:
                    return str(data[key])
            return str(data)
    
    def log_performance_stats(self, stats: Dict[str, Any]):
        """记录性能统计"""
        self.performance_stats.update(stats)
        
        if 'performance' in self.log_files:
            try:
                with open(self.log_files['performance'], 'w', encoding='utf-8') as f:
                    json.dump(self.performance_stats, f, indent=2, ensure_ascii=False)
            except Exception as e:
                self.logger.warning(f"写入性能统计失败: {e}")
    
    def cleanup_temp_files(self):
        """清理临时文件"""
        try:
            import shutil
            temp_dir_path = Path(self.temp_dir)
            if temp_dir_path.exists():
                shutil.rmtree(temp_dir_path)
                self.logger.info("临时文件已清理")
        except Exception as e:
            self.logger.warning(f"清理临时文件失败: {e}")
    
    def validate_similarity_score(self, similarity: float, method_name: str = "unknown") -> float:
        """验证相似度分数是否在有效范围内
        
        Args:
            similarity: 相似度分数
            method_name: 方法名称，用于日志记录
            
        Returns:
            修正后的相似度分数
        """
        if similarity > 1.0001 or similarity < -1.0001:  # 允许小的浮点误差
            self.logger.error(f"❌ {method_name}方法检测到无效的相似度: {similarity}")
            self.logger.error("这表明向量归一化或相似度计算存在问题")
            
            # 限制到有效范围
            corrected_similarity = max(-1.0, min(1.0, similarity))
            self.logger.warning(f"相似度已限制到有效范围: {corrected_similarity}")
            return corrected_similarity
        
        return similarity
    
    def log_vector_norm_info(self, vector: np.ndarray, vector_name: str = "vector") -> None:
        """记录向量norm信息，用于调试
        
        Args:
            vector: 向量
            vector_name: 向量名称
        """
        if hasattr(vector, 'shape') and len(vector.shape) > 1:
            # 多个向量
            norms = np.linalg.norm(vector, axis=1)
            self.logger.info(f"📊 {vector_name} norm统计: 最小={np.min(norms):.6f}, 最大={np.max(norms):.6f}, 平均={np.mean(norms):.6f}")
        else:
            # 单个向量
            norm = np.linalg.norm(vector)
            self.logger.info(f"📊 {vector_name} norm: {norm:.6f}")
    
    @abstractmethod
    def deduplicate(self) -> Tuple[List[int], List[int], Dict[str, Any]]:
        """执行去重，返回保留索引、删除索引和统计信息"""
        pass
    
    def preview_deduplication(self) -> DedupPreviewResult:
        """预览去重结果（可选实现）"""
        # 默认实现：执行完整去重
        keep_indices, remove_indices, stats = self.deduplicate()
        return DedupPreviewResult(keep_indices, remove_indices, stats) 