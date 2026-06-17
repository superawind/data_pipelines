#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
全局Embedding去重器 - LSH优化版本

特点：
- 基于LSH(Locality Sensitive Hashing)的快速相似度检测
- 适用于超大规模数据（千万级、亿级）
- 避免两两比较，大幅提升效率
- 保持全局去重特性
- 可调节精度-速度平衡

核心思想：
1. 使用随机超平面LSH将相似向量映射到相同的bucket
2. 只在同一bucket内进行精确相似度比较
3. 通过多个哈希表提高召回率
"""

import os
import sys
import time
import json
import logging
import numpy as np
from typing import List, Dict, Tuple, Set, Any, Optional
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path

# 尝试导入父类
try:
    from global_embedding_dedup import GlobalEmbeddingDeduplicator
except ImportError:
    # 如果当前文件在deduplicate目录下运行
    from deduplicate.global_embedding_dedup import GlobalEmbeddingDeduplicator

try:
    from dep_code.base import BaseDeduplicator
    from dep_code.config import DedupConfig
except ImportError:
    try:
        from deduplicate.dep_code.base import BaseDeduplicator
        from deduplicate.dep_code.config import DedupConfig
    except ImportError:
        # 如果没有基类，创建简单的基类
        class BaseDeduplicator:
            def __init__(self, config):
                self.config = config
                
        class DedupConfig:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)


class LSHIndex:
    """
    基于随机超平面的LSH索引
    
    原理：
    - 使用随机超平面将向量空间分割
    - 相似向量倾向于落在超平面的同一侧
    - 通过多个哈希表提高召回率
    """
    
    def __init__(self, dimension: int, num_tables: int = 10, hash_size: int = 10):
        """
        初始化LSH索引
        
        Args:
            dimension: 向量维度
            num_tables: 哈希表数量（越多召回越高，但内存和时间开销越大）
            hash_size: 每个哈希表的位数（越大bucket越多越精细）
        """
        self.dimension = dimension
        self.num_tables = num_tables
        self.hash_size = hash_size
        
        # 生成随机超平面（哈希函数）
        # 每个哈希表有hash_size个随机超平面
        self.random_planes = []
        np.random.seed(42)  # 固定随机种子保证可复现性
        for i in range(num_tables):
            planes = np.random.randn(hash_size, dimension)
            # 归一化超平面
            planes = planes / np.linalg.norm(planes, axis=1, keepdims=True)
            self.random_planes.append(planes)
        
        # 存储bucket：{table_id: {hash_value: [vector_indices]}}
        self.tables = [defaultdict(list) for _ in range(num_tables)]
        
        print(f"🔧 LSH索引配置:")
        print(f"  📐 向量维度: {dimension}")
        print(f"  📊 哈希表数量: {num_tables}")
        print(f"  🔢 哈希位数: {hash_size}")
        print(f"  🎯 理论bucket数: {2**hash_size} 个/表")
    
    def _compute_hash(self, vector: np.ndarray, table_id: int) -> int:
        """
        计算向量在指定哈希表中的哈希值
        
        Args:
            vector: 输入向量（已归一化）
            table_id: 哈希表ID
            
        Returns:
            哈希值（整数）
        """
        planes = self.random_planes[table_id]
        # 计算向量与所有超平面的点积
        projections = np.dot(planes, vector)
        # 根据点积正负生成二进制哈希码
        bits = (projections >= 0).astype(int)
        # 将二进制转换为整数
        hash_value = int(''.join(bits.astype(str)), 2)
        return hash_value
    
    def add(self, vector: np.ndarray, index: int):
        """
        添加向量到LSH索引
        
        Args:
            vector: 输入向量（已归一化）
            index: 向量的全局索引
        """
        for table_id in range(self.num_tables):
            hash_value = self._compute_hash(vector, table_id)
            self.tables[table_id][hash_value].append(index)
    
    def add_batch(self, vectors: np.ndarray, start_index: int):
        """
        批量添加向量到LSH索引
        
        Args:
            vectors: 向量矩阵 (n, dimension)
            start_index: 起始全局索引
        """
        for i, vector in enumerate(vectors):
            self.add(vector, start_index + i)
    
    def query(self, vector: np.ndarray, return_set: bool = True) -> Set[int]:
        """
        查询与给定向量可能相似的候选向量索引
        
        Args:
            vector: 查询向量（已归一化）
            return_set: 是否返回集合（自动去重）
            
        Returns:
            候选向量索引集合
        """
        candidates = []
        for table_id in range(self.num_tables):
            hash_value = self._compute_hash(vector, table_id)
            candidates.extend(self.tables[table_id][hash_value])
        
        if return_set:
            return set(candidates)
        return candidates
    
    def get_stats(self) -> Dict[str, Any]:
        """获取LSH索引统计信息"""
        stats = {
            'num_tables': self.num_tables,
            'hash_size': self.hash_size,
            'dimension': self.dimension,
            'total_buckets_per_table': 2**self.hash_size,
            'buckets_used_per_table': [],
            'bucket_size_stats': []
        }
        
        for table_id, table in enumerate(self.tables):
            stats['buckets_used_per_table'].append(len(table))
            
            bucket_sizes = [len(bucket) for bucket in table.values()]
            if bucket_sizes:
                stats['bucket_size_stats'].append({
                    'table_id': table_id,
                    'min': min(bucket_sizes),
                    'max': max(bucket_sizes),
                    'avg': np.mean(bucket_sizes),
                    'median': np.median(bucket_sizes)
                })
        
        return stats


class GlobalEmbeddingLSHDeduplicator(GlobalEmbeddingDeduplicator):
    """
    全局Embedding去重器 - LSH优化版本
    
    继承自GlobalEmbeddingDeduplicator，复用数据加载逻辑
    使用LSH优化大规模数据的去重效率
    """
    
    def __init__(self, config: DedupConfig):
        """初始化LSH优化的全局Embedding去重器"""
        # 调用父类初始化（会加载所有配置和检查FAISS等）
        super().__init__(config)
        
        # LSH特定配置
        self.use_lsh = getattr(config, 'use_lsh', True)  # 默认启用LSH
        self.lsh_num_tables = getattr(config, 'lsh_num_tables', 10)
        self.lsh_hash_size = getattr(config, 'lsh_hash_size', 10)
        
        print(f"\n🚀 LSH优化模式已启用")
        print(f"📊 LSH配置:")
        print(f"  🔢 哈希表数量: {self.lsh_num_tables}")
        print(f"  📏 哈希位数: {self.lsh_hash_size}")
        print(f"  💡 预期加速比: 10-100倍（取决于数据规模）")
    
    def _format_json_output(self, data: dict) -> str:
        """格式化JSON输出（如果父类方法不可用，使用此fallback）"""
        if not hasattr(self.config, 'field_order') or not self.config.field_order:
            return json.dumps(data, ensure_ascii=False)
        
        # 按照field_order重新排序字段
        ordered_data = {}
        for field in self.config.field_order:
            if field in data:
                ordered_data[field] = data[field]
        
        # 添加不在field_order中的其他字段
        for key, value in data.items():
            if key not in ordered_data:
                ordered_data[key] = value
        
        return json.dumps(ordered_data, ensure_ascii=False)
    
    def global_deduplicate(self) -> Tuple[List[Dict], Dict[str, Any]]:
        """执行全局去重 - LSH优化版本"""
        start_time = time.time()
        
        print("🚀 启动全局Embedding去重 - LSH优化模式")
        print("💡 适用于千万级、亿级数据，避免两两比较")
        
        # 步骤1：加载数据和embeddings（复用父类方法）
        data_list, embeddings = self.load_data_and_embeddings()
        
        print(f"\n📊 开始去重处理...")
        print(f"  📁 数据量: {len(data_list):,} 条")
        print(f"  📐 向量维度: {embeddings.shape[1]}")
        print(f"  🎯 相似度阈值: {self.threshold}")
        
        # 步骤2：构建LSH索引
        print("\n🔨 构建LSH索引...")
        lsh_start_time = time.time()
        
        lsh_index = LSHIndex(
            dimension=embeddings.shape[1],
            num_tables=self.lsh_num_tables,
            hash_size=self.lsh_hash_size
        )
        
        # 批量添加向量到LSH索引
        print("📥 添加向量到LSH索引...")
        batch_size_lsh = 10000
        for i in tqdm(range(0, len(embeddings), batch_size_lsh), desc="🔨 构建LSH"):
            batch = embeddings[i:i+batch_size_lsh]
            lsh_index.add_batch(batch, i)
        
        lsh_build_time = time.time() - lsh_start_time
        print(f"✅ LSH索引构建完成，耗时: {lsh_build_time:.1f}秒")
        
        # 打印LSH统计信息
        lsh_stats = lsh_index.get_stats()
        print(f"\n📈 LSH索引统计:")
        print(f"  📊 已使用的bucket数（平均）: {np.mean(lsh_stats['buckets_used_per_table']):.0f} / {lsh_stats['total_buckets_per_table']}")
        if lsh_stats['bucket_size_stats']:
            avg_bucket_size = np.mean([s['avg'] for s in lsh_stats['bucket_size_stats']])
            print(f"  📦 平均bucket大小: {avg_bucket_size:.1f} 个向量")
        
        # 步骤3：基于LSH的去重检测
        print("\n🔍 开始LSH去重检测...")
        dedup_start_time = time.time()
        
        kept_indices = set()
        removed_indices = set()
        duplicate_records = []
        
        total_comparisons = 0  # 统计实际比较次数
        
        with tqdm(total=len(embeddings), desc="🔍 LSH去重", 
                 bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} '
                           '[耗时:{elapsed}, 剩余:{remaining}, {rate_fmt}]') as pbar:
            
            for idx in range(len(embeddings)):
                # 跳过已删除的数据
                if idx in removed_indices:
                    pbar.update(1)
                    continue
                
                # 使用LSH查询候选集
                vector = embeddings[idx]
                candidates = lsh_index.query(vector)
                
                # 过滤候选集：只保留之前处理过且未被删除的数据
                valid_candidates = [c for c in candidates if c < idx and c not in removed_indices]
                total_comparisons += len(valid_candidates)
                
                # 在候选集中进行精确相似度比较
                is_duplicate = False
                best_match_idx = None
                best_similarity = 0.0
                
                for candidate_idx in valid_candidates:
                    # 计算余弦相似度（向量已归一化，直接点积即可）
                    similarity = float(np.dot(vector, embeddings[candidate_idx]))
                    
                    # 验证相似度范围
                    if similarity > 1.0001 or similarity < -1.0001:
                        self.logger.warning(f"无效相似度: {similarity:.6f}，限制到有效范围")
                        similarity = max(-1.0, min(1.0, similarity))
                    
                    if similarity >= self.threshold:
                        is_duplicate = True
                        best_match_idx = candidate_idx
                        best_similarity = similarity
                        break
                    elif similarity >= self.threshold * 0.8:  # 记录接近阈值的样例
                        self.log_threshold_example(
                            data_list[idx],
                            data_list[candidate_idx],
                            similarity,
                            self.threshold,
                            {
                                'embedding_similarity': float(similarity),
                                'threshold': self.threshold,
                                'distance_from_threshold': float(self.threshold - similarity),
                                'vector_dimension': embeddings.shape[1],
                                'method': 'lsh'
                            }
                        )
                
                if is_duplicate:
                    # 标记为重复
                    removed_indices.add(idx)
                    duplicate_records.append({
                        'removed_data': data_list[idx],
                        'kept_data': data_list[best_match_idx],
                        'similarity': float(best_similarity)
                    })
                else:
                    # 保留为唯一数据
                    kept_indices.add(idx)
                
                # 更新进度条
                pbar.update(1)
                if idx % 1000 == 0:
                    current_dedup_ratio = len(removed_indices) / (idx + 1) if idx > 0 else 0
                    pbar.set_postfix({
                        '保留': f'{len(kept_indices):,}',
                        '去重': f'{len(removed_indices):,}',
                        '去重率': f'{current_dedup_ratio:.1%}'
                    })
        
        dedup_time = time.time() - dedup_start_time
        
        # 计算加速效果
        naive_comparisons = len(embeddings) * (len(embeddings) - 1) // 2
        speedup = naive_comparisons / total_comparisons if total_comparisons > 0 else float('inf')
        
        print(f"\n⚡ LSH加速效果:")
        print(f"  🔄 朴素算法需要比较: {naive_comparisons:,} 次")
        print(f"  ⚡ LSH实际比较: {total_comparisons:,} 次")
        print(f"  🚀 加速比: {speedup:.1f}x")
        
        # 生成最终结果
        print("\n📊 生成最终结果...")
        unique_data = []
        for idx in sorted(kept_indices):
            data = data_list[idx].copy()
            # 清理内部字段
            if '__line_idx__' in data:
                del data['__line_idx__']
            if '__embedding_idx__' in data:
                del data['__embedding_idx__']
            unique_data.append(data)
        
        # 统计信息
        total_time = time.time() - start_time
        
        method_name = str(self.config.method.value) if hasattr(self.config.method, 'value') else str(self.config.method)
        stats = {
            'total_items': len(data_list),
            'unique_items': len(unique_data),
            'duplicate_items': len(removed_indices),
            'dedup_ratio': len(removed_indices) / len(data_list) if data_list else 0,
            'total_time': total_time,
            'lsh_build_time': lsh_build_time,
            'dedup_time': dedup_time,
            'processing_speed': len(data_list) / dedup_time if dedup_time > 0 else 0,
            'embedding_dim': embeddings.shape[1],
            'threshold': self.threshold,
            'lsh_num_tables': self.lsh_num_tables,
            'lsh_hash_size': self.lsh_hash_size,
            'total_comparisons': total_comparisons,
            'speedup': speedup,
            'method': f'{method_name}_global_lsh',
            'mode': 'global_lsh'
        }
        
        # 保存日志（复用父类方法）
        # 确保log_files存在
        try:
            if hasattr(self, 'log_files') and self.log_files:
                self._save_duplicate_records(duplicate_records)
                self._save_threshold_examples()
                self._save_performance_stats(stats)
            else:
                print("⚠️  log_files未初始化，跳过详细日志保存")
        except Exception as e:
            print(f"⚠️  保存日志时出错: {e}")
            print("💡 跳过日志保存，继续输出统计结果")
        
        # 输出详细统计
        print(f"\n" + "="*70)
        print(f"🎉 全局Embedding去重完成（LSH优化版）!")
        print(f"="*70)
        print(f"📊 去重统计:")
        print(f"  📥 输入数据: {stats['total_items']:,} 条")
        print(f"  ✅ 保留数据: {stats['unique_items']:,} 条")
        print(f"  🗑️  删除数据: {stats['duplicate_items']:,} 条")
        print(f"  📈 去重效果: {stats['dedup_ratio']:.2%} 重复率")
        print(f"")
        print(f"⏱️  性能统计:")
        print(f"  🕐 总处理时间: {total_time:.1f} 秒")
        print(f"  🔨 LSH索引构建: {lsh_build_time:.1f} 秒")
        print(f"  🔍 去重检测: {dedup_time:.1f} 秒")
        print(f"  🚀 处理速度: {stats['processing_speed']:,.0f} 条/秒")
        print(f"")
        print(f"⚡ LSH优化效果:")
        print(f"  🔢 哈希表数量: {self.lsh_num_tables}")
        print(f"  📏 哈希位数: {self.lsh_hash_size}")
        print(f"  🔄 实际比较次数: {total_comparisons:,}")
        print(f"  🚀 加速比: {speedup:.1f}x")
        print(f"  🎯 相似度阈值: {self.threshold}")
        print(f"")
        print(f"📁 文件信息:")
        print(f"  📂 输入文件: {self.input_file}")
        print(f"  💾 输出文件: {self.output_file}")
        if hasattr(self, 'log_dir'):
            print(f"  📄 日志目录: {self.log_dir}/")
        print(f"="*70)
        
        # 性能建议
        if speedup < 5:
            print(f"\n💡 LSH优化建议:")
            print(f"  - 当前加速比较低，可能因为数据规模较小或LSH参数不当")
            print(f"  - 建议增加 --lsh_num_tables 到 {min(20, self.lsh_num_tables * 2)}")
            print(f"  - 或调整 --lsh_hash_size 到 {max(8, self.lsh_hash_size - 2)}")
        
        return unique_data, stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="全局Embedding去重 - LSH优化版")
    parser.add_argument("--input_file", required=True, help="输入JSONL文件")
    parser.add_argument("--output_file", required=True, help="输出文件")
    parser.add_argument("--embeddings_file", help="NPY格式embeddings文件")
    parser.add_argument("--embeddings_files", nargs="+", help="多个JSONL格式embeddings文件")
    parser.add_argument("--embeddings_format", choices=["npy", "jsonl"], default="npy")
    parser.add_argument("--threshold", type=float, default=0.95, help="相似度阈值")
    parser.add_argument("--lsh_num_tables", type=int, default=10, help="LSH哈希表数量")
    parser.add_argument("--lsh_hash_size", type=int, default=10, help="LSH哈希位数")
    
    args = parser.parse_args()
    
    try:
        from dep_code.config import DedupConfig, EmbeddingFileFormat
    except ImportError:
        from deduplicate.dep_code.config import DedupConfig, EmbeddingFileFormat
    
    config = DedupConfig(
        method="embedding",
        mode="global",
        input_file=args.input_file,
        output_file=args.output_file,
        embeddings_file=args.embeddings_file,
        embeddings_files=args.embeddings_files,
        embeddings_format=EmbeddingFileFormat.NPY if args.embeddings_format == "npy" else EmbeddingFileFormat.JSONL,
        threshold=args.threshold,
        use_lsh=True,
        lsh_num_tables=args.lsh_num_tables,
        lsh_hash_size=args.lsh_hash_size
    )
    
    deduplicator = GlobalEmbeddingLSHDeduplicator(config)
    deduplicator.deduplicate()

