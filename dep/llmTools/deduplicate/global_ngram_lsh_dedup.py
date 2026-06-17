#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
真正的全局N-gram LSH去重器

特点：
- 真正的全局去重：构建完整的LSH索引
- 智能相似度检测：基于Jaccard相似度
- 批量优化处理：支持大规模数据
- 完整日志记录：详细的重复信息
"""

import os
import sys
import time
import json
import logging
import numpy as np
from typing import List, Dict, Tuple, Set, Any
from tqdm import tqdm
import psutil
import datetime
from pathlib import Path
from collections import defaultdict

# 高性能库
try:
    import xxhash
    hash_func = lambda x: xxhash.xxh64(x.encode()).intdigest()
    FAST_HASH = True
    print("🚀 xxHash极速哈希可用")
except ImportError:
    import hashlib
    hash_func = lambda x: int(hashlib.md5(x.encode()).hexdigest()[:16], 16)
    FAST_HASH = False

try:
    import orjson
    json_loads = orjson.loads
    json_dumps = orjson.dumps
    print("🚀 orjson极速JSON可用")
except ImportError:
    import json
    json_loads = json.loads
    json_dumps = json.dumps

try:
    from datasketch import MinHash, MinHashLSH
    MINHASH_AVAILABLE = True
    print("🚀 MinHash可用")
except ImportError:
    MINHASH_AVAILABLE = False
    print("❌ MinHash不可用，请安装: pip install datasketch")
    sys.exit(1)

try:
    from .base import BaseDeduplicator
    from .config import DedupConfig
except ImportError:
    try:
        from base import BaseDeduplicator
        from config import DedupConfig
    except ImportError:
        # 简单的基类
        class BaseDeduplicator:
            def __init__(self, config):
                self.config = config
                
        class DedupConfig:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)


class GlobalNgramLshDeduplicator(BaseDeduplicator):
    """真正的全局N-gram LSH去重器"""
    
    def __init__(self, config: DedupConfig):
        super().__init__(config)
        
        # 基本配置
        self.input_file = config.input_file
        self.output_file = config.output_file
        self.content_keys = getattr(config, 'content_keys', ['prompt'])
        
        # LSH配置
        self.jaccard_threshold = getattr(config, 'jaccard_threshold', 0.85)
        self.ngram_size = getattr(config, 'ngram_size', 10)
        self.num_permutations = getattr(config, 'num_permutations', 128)
        
        # 性能配置
        self.batch_size = getattr(config, 'batch_size', 10000)
        
        # 检查文件
        if not os.path.exists(self.input_file):
            raise FileNotFoundError(f"输入文件不存在: {self.input_file}")
        
        # 为N-gram LSH添加特有的groups日志文件
        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        method_name = getattr(self.config, 'method', 'unknown')
        if hasattr(method_name, 'value'):
            method_name = method_name.value
        groups_log_file = self.log_dir / f"similar_groups_{method_name}_{timestamp}.jsonl"
        self.log_files['groups'] = groups_log_file

        
        print(f"🌍 真正的全局N-gram LSH去重器")
        print(f"🎯 Jaccard阈值: {self.jaccard_threshold}")
        print(f"📊 N-gram大小: {self.ngram_size}")
        print(f"🔀 排列数: {self.num_permutations}")
        print(f"📦 批处理大小: {self.batch_size}")
        print(f"🏆 支持文本相似度检测")
    

    
    def _extract_content(self, data: dict) -> str:
        """提取用于去重的内容"""
        content_parts = []
        for key in self.content_keys:
            if key in data:
                content_parts.append(str(data[key]))
        return " ".join(content_parts)
    
    def _create_minhash(self, content: str) -> MinHash:
        """创建MinHash"""
        minhash = MinHash(num_perm=self.num_permutations)
        
        # 生成n-grams
        if len(content) >= self.ngram_size:
            ngrams = [content[i:i + self.ngram_size] for i in range(len(content) - self.ngram_size + 1)]
        else:
            ngrams = [content]
        
        for ngram in ngrams:
            minhash.update(ngram.encode('utf-8'))
        
        return minhash
    
    def _calculate_jaccard_similarity(self, content1: str, content2: str) -> float:
        """计算精确的Jaccard相似度"""
        if len(content1) >= self.ngram_size:
            ngrams1 = set(content1[i:i + self.ngram_size] for i in range(len(content1) - self.ngram_size + 1))
        else:
            ngrams1 = {content1}
        
        if len(content2) >= self.ngram_size:
            ngrams2 = set(content2[i:i + self.ngram_size] for i in range(len(content2) - self.ngram_size + 1))
        else:
            ngrams2 = {content2}
        
        intersection = len(ngrams1 & ngrams2)
        union = len(ngrams1 | ngrams2)
        
        return intersection / union if union > 0 else 0.0
    
    def load_data_and_build_lsh(self) -> Tuple[List[Dict], MinHashLSH, Dict[str, Any]]:
        """加载数据并构建全局LSH索引 - 带详细进度显示"""
        print("🔄 加载数据并构建全局LSH索引...")
        
        # 获取文件大小
        file_size = os.path.getsize(self.input_file)
        print(f"📁 文件大小: {file_size / (1024**3):.2f} GB ({file_size:,} 字节)")
        
        # 创建全局LSH索引
        global_lsh = MinHashLSH(
            threshold=self.jaccard_threshold,
            num_perm=self.num_permutations,
            storage_config={'type': 'dict'}
        )
        
        data_list = []
        content_to_id = {}
        id_to_data = {}
        next_id = 0
        bytes_read = 0
        start_time = time.time()
        
        # 创建带总量的进度条
        with open(self.input_file, 'r', encoding='utf-8') as f:
            with tqdm(total=file_size, desc="📖 加载+构建LSH", unit="B", unit_scale=True,
                     bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
                
                for line_idx, line in enumerate(f):
                    line = line.strip()
                    line_bytes = len(line.encode('utf-8')) + 1  # +1 for newline
                    bytes_read += line_bytes
                    pbar.update(line_bytes)
                    
                    if not line:
                        continue
                    
                    try:
                        data = json_loads(line)
                        content = self._extract_content(data)
                        
                        if content:
                            # 创建MinHash
                            minhash = self._create_minhash(content)
                            
                            # 为每个内容分配唯一ID
                            content_id = f"content_{next_id}"
                            next_id += 1
                            
                            # 将数据添加到全局LSH索引
                            global_lsh.insert(content_id, minhash)
                            
                            # 保存映射关系
                            content_to_id[content] = content_id
                            id_to_data[content_id] = {
                                'line_idx': line_idx,
                                'line': line,
                                'content': content,
                                'data': data,
                                'minhash': minhash
                            }
                            
                            data_list.append(data)
                            
                    except Exception as e:
                        self.logger.warning(f"行 {line_idx} 解析失败: {e}")
                    
                    # 每1000行更新一次后缀信息
                    if line_idx % 1000 == 0 and line_idx > 0:
                        elapsed = time.time() - start_time
                        lines_per_sec = line_idx / elapsed
                        mb_per_sec = (bytes_read / (1024**2)) / elapsed
                        pbar.set_postfix({
                            "行数": f"{len(data_list):,}",
                            "LSH项": f"{next_id:,}",
                            "行/s": f"{lines_per_sec:,.0f}",
                            "MB/s": f"{mb_per_sec:.1f}"
                        })
        
        total_time = time.time() - start_time
        final_mb_per_sec = (file_size / (1024**2)) / total_time
        final_lines_per_sec = len(data_list) / total_time
        
        print(f"✅ 数据加载和LSH构建完成!")
        print(f"📊 加载统计:")
        print(f"  📄 总行数: {len(data_list):,}")
        print(f"  🔍 LSH索引项: {next_id:,}")
        print(f"  📁 文件大小: {file_size / (1024**3):.2f} GB")
        print(f"  ⏱️ 总耗时: {total_time:.1f} 秒")
        print(f"  🚀 平均速度: {final_mb_per_sec:.1f} MB/s")
        print(f"  📈 行处理速度: {final_lines_per_sec:,.0f} 行/s")
        
        return data_list, global_lsh, id_to_data
    
    def find_similar_groups(self, global_lsh: MinHashLSH, id_to_data: Dict[str, Any]) -> List[List[str]]:
        """找到相似组"""
        print("🔍 查找相似组...")
        
        visited = set()
        similar_groups = []
        
        for content_id, data_info in tqdm(id_to_data.items(), desc="🎯 分组处理"):
            if content_id in visited:
                continue
            
            # 查找相似项
            similar_ids = global_lsh.query(data_info['minhash'])
            
            if len(similar_ids) > 1:
                # 有相似项，形成一个组
                group = []
                for similar_id in similar_ids:
                    if similar_id not in visited:
                        group.append(similar_id)
                        visited.add(similar_id)
                
                if len(group) > 1:
                    similar_groups.append(group)
            else:
                # 没有相似项，标记为已访问
                visited.add(content_id)
        
        print(f"✅ 相似组查找完成: {len(similar_groups)} 个组")
        
        return similar_groups
    
    def select_best_in_group(self, group: List[str], id_to_data: Dict[str, Any]) -> str:
        """在相似组中选择最佳代表"""
        if len(group) == 1:
            return group[0]
        
        # 简单策略：选择内容最长的
        best_id = group[0]
        best_length = len(id_to_data[best_id]['content'])
        
        for content_id in group[1:]:
            content_length = len(id_to_data[content_id]['content'])
            if content_length > best_length:
                best_id = content_id
                best_length = content_length
        
        return best_id
    
    def global_deduplicate(self) -> Tuple[List[Dict], Dict[str, Any]]:
        """执行全局去重"""
        start_time = time.time()
        
        # 步骤1：加载数据并构建LSH索引
        data_list, global_lsh, id_to_data = self.load_data_and_build_lsh()
        
        # 步骤2：找到相似组
        similar_groups = self.find_similar_groups(global_lsh, id_to_data)
        
        # 步骤3：选择最佳代表并记录重复
        print("🏆 选择最佳代表...")
        unique_data = []
        duplicate_records = []
        kept_ids = set()
        
        for group in tqdm(similar_groups, desc="🎯 处理相似组"):
            # 选择最佳代表
            best_id = self.select_best_in_group(group, id_to_data)
            kept_ids.add(best_id)
            
            # 记录其他成员为重复
            best_data_info = id_to_data[best_id]
            for content_id in group:
                if content_id != best_id:
                    data_info = id_to_data[content_id]
                    
                    # 计算精确相似度
                    similarity = self._calculate_jaccard_similarity(
                        data_info['content'], 
                        best_data_info['content']
                    )
                    
                    duplicate_records.append({
                        'removed_data': data_info['data'],
                        'kept_data': best_data_info['data'],
                        'similarity': similarity,
                        'group_size': len(group),
                        'removed_content_id': content_id,
                        'kept_content_id': best_id,
                        'method': 'ngram_lsh'
                    })
        
        # 收集所有唯一数据（包括没有重复的）
        for content_id, data_info in id_to_data.items():
            if content_id in kept_ids or not any(content_id in group for group in similar_groups):
                unique_data.append(data_info['data'])
        
        # 统计信息
        total_time = time.time() - start_time
        method_name = str(self.config.method.value) if hasattr(self.config.method, 'value') else str(self.config.method)
        stats = {
            'total_items': len(data_list),
            'unique_items': len(unique_data),
            'duplicate_items': len(duplicate_records),
            'similar_groups': len(similar_groups),
            'dedup_ratio': len(duplicate_records) / len(data_list) if data_list else 0,
            'total_time': total_time,
            'method': f'{method_name}_global',
            'mode': 'global'
        }
        
        # 记录日志
        self._save_duplicate_records(duplicate_records)
        self._save_similar_groups(similar_groups, id_to_data)
        self._save_performance_stats(stats)
        
        print(f"\n" + "="*70)
        print(f"🎉 全局N-gram LSH去重完成!")
        print(f"="*70)
        print(f"📊 数据统计:")
        print(f"  📥 输入总数据: {stats['total_items']:,} 行")
        print(f"  ✅ 保留数据: {stats['unique_items']:,} 行")
        print(f"  🗑️  删除数据: {stats['duplicate_items']:,} 行")
        print(f"  📈 去重效果: {stats['dedup_ratio']:.2%} 重复率")
        print(f"  🔗 相似组数: {stats['similar_groups']:,} 个")
        print(f"")
        print(f"⏱️  性能统计:")
        print(f"  🕐 总处理时间: {total_time:.2f} 秒")
        print(f"  🚀 处理速度: {stats['total_items']/total_time:,.0f} 行/秒")
        print(f"  🧠 LSH索引构建: 包含在总时间内")
        print(f"  🎯 算法参数: {self.ngram_size}-gram, {self.jaccard_threshold} 阈值, {self.num_permutations} 排列")
        print(f"")
        print(f"📁 文件信息:")
        print(f"  📂 输入文件: {self.input_file}")
        print(f"  💾 输出文件: {self.output_file}")
        print(f"  📄 日志目录: {self.log_dir}/")
        print(f"  📊 相似组详情: {self.log_files['groups']}")
        print(f"="*70)
        
        return unique_data, stats
    
    def _save_duplicate_records(self, records: List[Dict]):
        """保存重复记录"""
        with open(self.log_files['duplicates'], 'w', encoding='utf-8') as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        print(f"📄 重复记录已保存: {self.log_files['duplicates']}")
    
    def _save_similar_groups(self, groups: List[List[str]], id_to_data: Dict[str, Any]):
        """保存相似组信息"""
        with open(self.log_files['groups'], 'w', encoding='utf-8') as f:
            for group_id, group in enumerate(groups):
                group_info = {
                    'group_id': group_id,
                    'size': len(group),
                    'content_ids': group,
                    'contents': [id_to_data[cid]['content'] for cid in group]
                }
                f.write(json.dumps(group_info, ensure_ascii=False) + '\n')
        print(f"📊 相似组信息已保存: {self.log_files['groups']}")
    
    def _save_performance_stats(self, stats: Dict):
        """保存性能统计"""
        with open(self.log_files['performance'], 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"📈 性能统计已保存: {self.log_files['performance']}")
    
    def deduplicate(self):
        """主入口：执行全局去重并保存结果"""
        unique_data, stats = self.global_deduplicate()
        
        # 保存结果
        print("💾 保存去重结果...")
        with open(self.output_file, 'w', encoding='utf-8') as f:
            for data in tqdm(unique_data, desc="💾 写入结果"):
                # 使用field_order排序的JSON输出
                f.write(self._format_json_output(data) + '\n')
        
        print(f"✅ 去重结果已保存: {self.output_file}")
        return stats


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="真正的全局N-gram LSH去重器")
    parser.add_argument("--input_file", required=True, help="输入文件")
    parser.add_argument("--output_file", required=True, help="输出文件")
    parser.add_argument("--content-keys", nargs="+", default=["prompt"], help="内容字段")
    parser.add_argument("--jaccard-threshold", type=float, default=0.85, help="Jaccard相似度阈值")
    parser.add_argument("--ngram-size", type=int, default=10, help="N-gram大小")
    parser.add_argument("--num-permutations", type=int, default=128, help="MinHash排列数")
    parser.add_argument("--batch-size", type=int, default=10000, help="批处理大小")
    
    args = parser.parse_args()
    
    config = DedupConfig(
        input_file=args.input_file,
        output_file=args.output_file,
        content_keys=args.content_keys,
        jaccard_threshold=args.jaccard_threshold,
        ngram_size=args.ngram_size,
        num_permutations=args.num_permutations,
        batch_size=args.batch_size
    )
    
    deduplicator = GlobalNgramLshDeduplicator(config)
    deduplicator.deduplicate()


if __name__ == "__main__":
    main() 