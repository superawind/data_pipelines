#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
去重配置模块

定义三种去重方法的配置参数：
1. 严格哈希去重 - 完全匹配去重
2. N-gram LSH去重 - 文本相似度去重
3. Embedding去重 - 语义相似度去重
"""

import os
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional


class DedupMethod(Enum):
    """去重方法枚举"""
    STRICT_HASH = "strict_hash"  # 完全匹配
    NGRAM_LSH = "ngram_lsh"      # 文本相似度
    EMBEDDING = "embedding"       # 语义相似度


class EmbeddingFileFormat(Enum):
    """Embedding文件格式"""
    NPY = "npy"      # numpy数组文件
    JSONL = "jsonl"  # JSONL格式，支持ID匹配


@dataclass
class DedupConfig:
    """
    去重配置类
    
    提供完整的类型验证和参数范围检查，确保配置的正确性。
    """
    
    # ===== 基础配置 =====
    method: DedupMethod
    """
    去重方法（必选）
    可选值：
        - DedupMethod.STRICT_HASH: 严格哈希去重（完全匹配）
        - DedupMethod.NGRAM_LSH: N-gram LSH去重（文本相似）
        - DedupMethod.EMBEDDING: Embedding去重（语义相似）
    """
    
    input_file: str
    """输入JSONL文件路径（必选）"""
    
    output_file: str
    """输出JSONL文件路径（必选）"""
    
    content_keys: List[str] = field(default_factory=lambda: ["prompt"])
    """
    用于去重的内容字段列表（可选，默认：["prompt"]）
    示例：["prompt"], ["prompt", "instruction"]
    """
    
    field_order: List[str] = field(default_factory=lambda: ["prompt", "instruction", "input", "output"])
    """
    JSON字段的输出顺序（可选）
    默认：["prompt", "instruction", "input", "output"]
    """
    
    # ===== 通用性能配置 =====
    num_workers: Optional[int] = None
    """
    并行worker数量（可选，默认：None=自动检测）
    范围：1-128，推荐：CPU核心数的1-2倍
    """
    
    batch_size: int = 10000
    """
    批处理大小（可选，默认：10000）
    范围：1000-100000，推荐：10000-50000
    """
    
    log_dir: str = None
    """
    日志目录（可选，默认：None=自动选择）
    None时自动创建global_dedup_logs目录
    """
    
    # ===== N-gram LSH配置 =====
    ngram_size: int = 10
    """
    N-gram大小（ngram_lsh方法，默认：10）
    范围：3-20，推荐：8-12
    """
    
    jaccard_threshold: float = 0.85
    """
    Jaccard相似度阈值（ngram_lsh方法，默认：0.85）
    范围：0.0-1.0，推荐：0.8-0.9
    值越高去重越严格
    """
    
    num_permutations: int = 128
    """
    MinHash排列数（ngram_lsh方法，默认：128）
    范围：32-512，推荐：64-256
    值越大精度越高但速度越慢
    """
    
    # ===== Embedding配置 =====
    embeddings_file: Optional[str] = None
    """
    Embedding文件路径（embedding方法必需）
    支持格式：.npy或.jsonl
    """
    
    embeddings_files: Optional[List[str]] = None
    """
    多个Embedding文件路径列表（embedding方法可选）
    仅支持JSONL格式，用于大文件分割场景
    """
    
    embeddings_format: EmbeddingFileFormat = EmbeddingFileFormat.JSONL
    """
    Embedding文件格式（默认：JSONL）
    可选值：
        - EmbeddingFileFormat.NPY: numpy数组格式
        - EmbeddingFileFormat.JSONL: JSONL格式，支持ID匹配
    """
    
    prompt_id_key: str = "prompt_id"
    """
    数据文件中的ID字段名（默认："prompt_id"）
    仅在embeddings_format=JSONL时使用
    """
    
    embedding_id_key: str = "prompt_id"
    """
    Embedding文件中的ID字段名（默认："prompt_id"）
    仅在embeddings_format=JSONL时使用
    """
    
    embedding_vector_key: str = "embedding"
    """
    Embedding文件中的向量字段名（默认："embedding"）
    仅在embeddings_format=JSONL时使用
    """
    
    threshold: float = 0.95
    """
    向量相似度阈值（embedding方法，默认：0.95）
    范围：0.0-1.0，推荐：0.9-0.98
    值越高去重越严格
    """
    
    top_k: int = 10
    """
    Top-K检索数量（embedding方法，默认：10）
    范围：1-100，推荐：5-20
    值越大召回越高但速度越慢
    """
    
    use_gpu: bool = False
    """
    是否使用GPU加速（embedding方法，默认：False）
    可选值：True（需要faiss-gpu）, False（需要faiss-cpu）
    """
    
    gpu_device: int = 0
    """
    GPU设备ID（默认：0）
    范围：0-7（取决于系统GPU数量）
    仅在use_gpu=True时生效
    """
    
    # ===== LSH优化配置（适用于ngram_lsh和embedding） =====
    use_lsh: bool = False
    """
    是否使用LSH优化（默认：False）
    可选值：True（大规模数据加速）, False（精确检索）
    适用于千万级、亿级数据
    """
    
    lsh_num_tables: int = 10
    """
    LSH哈希表数量（默认：10）
    范围：3-50，推荐：5-20
    值越大召回率越高但速度越慢
    """
    
    lsh_hash_size: int = 10
    """
    LSH哈希位数（默认：10）
    范围：6-16，推荐：8-12
    值越大bucket越多，比较次数越少
    """
    
    def __post_init__(self):
        """
        配置验证和自动修正
        
        验证所有配置参数的类型、范围和合法性，确保配置正确。
        """
        # 1. 验证method类型
        if not isinstance(self.method, DedupMethod):
            if isinstance(self.method, str):
                try:
                    self.method = DedupMethod(self.method)
                except ValueError:
                    raise ValueError(
                        f"无效的去重方法: {self.method}\n"
                        f"可选值: {', '.join([m.value for m in DedupMethod])}"
                    )
            else:
                raise TypeError(f"method必须是DedupMethod枚举或字符串，当前类型: {type(self.method)}")
        
        # 2. 验证输入/输出文件
        if not os.path.exists(self.input_file):
            raise FileNotFoundError(f"输入文件不存在: {self.input_file}")
        
        if not self.output_file:
            raise ValueError("output_file不能为空")
        
        # 3. 验证embedding方法的专属配置
        if self.method == DedupMethod.EMBEDDING:
            if not self.embeddings_file and not self.embeddings_files:
                raise ValueError(
                    "embedding方法必须指定embeddings_file或embeddings_files\n"
                    "示例：--embeddings_file embeddings.jsonl"
                )
            
            # 验证embeddings_format类型
            if not isinstance(self.embeddings_format, EmbeddingFileFormat):
                if isinstance(self.embeddings_format, str):
                    try:
                        self.embeddings_format = EmbeddingFileFormat(self.embeddings_format)
                    except ValueError:
                        raise ValueError(
                            f"无效的embedding格式: {self.embeddings_format}\n"
                            f"可选值: {', '.join([f.value for f in EmbeddingFileFormat])}"
                        )
            
            # 验证embedding文件存在性并自动检测格式
            if self.embeddings_file:
                if not os.path.exists(self.embeddings_file):
                    raise FileNotFoundError(f"embedding文件不存在: {self.embeddings_file}")
                # 根据扩展名自动检测格式
                if self.embeddings_file.endswith('.npy'):
                    self.embeddings_format = EmbeddingFileFormat.NPY
                elif self.embeddings_file.endswith('.jsonl'):
                    self.embeddings_format = EmbeddingFileFormat.JSONL
            elif self.embeddings_files:
                for f in self.embeddings_files:
                    if not os.path.exists(f):
                        raise FileNotFoundError(f"embedding文件不存在: {f}")
                self.embeddings_format = EmbeddingFileFormat.JSONL
        
        # 4. 验证content_keys
        if not self.content_keys:
            raise ValueError("content_keys不能为空")
        if not isinstance(self.content_keys, list):
            raise TypeError(f"content_keys必须是列表，当前类型: {type(self.content_keys)}")
        
        # 5. 验证数值参数范围
        # 阈值类参数 (0.0-1.0)
        if not (0.0 <= self.jaccard_threshold <= 1.0):
            raise ValueError(
                f"jaccard_threshold必须在0.0-1.0范围内，当前值: {self.jaccard_threshold}\n"
                f"推荐值: 0.8-0.9"
            )
        
        if not (0.0 <= self.threshold <= 1.0):
            raise ValueError(
                f"threshold必须在0.0-1.0范围内，当前值: {self.threshold}\n"
                f"推荐值: 0.9-0.98"
            )
        
        # N-gram参数
        if not (3 <= self.ngram_size <= 20):
            if self.ngram_size < 1:
                raise ValueError(f"ngram_size必须大于0，当前值: {self.ngram_size}")
            elif self.ngram_size > 20:
                print(f"⚠️  警告: ngram_size={self.ngram_size}较大，可能影响性能。推荐范围: 8-12")
        
        # MinHash参数
        if not (32 <= self.num_permutations <= 512):
            if self.num_permutations < 1:
                raise ValueError(f"num_permutations必须大于0，当前值: {self.num_permutations}")
            elif self.num_permutations > 512:
                print(f"⚠️  警告: num_permutations={self.num_permutations}过大，可能严重影响性能。推荐范围: 64-256")
        
        # Top-K参数
        if not (1 <= self.top_k <= 100):
            if self.top_k < 1:
                raise ValueError(f"top_k必须大于0，当前值: {self.top_k}")
            elif self.top_k > 100:
                print(f"⚠️  警告: top_k={self.top_k}过大，可能严重影响检索速度。推荐范围: 5-20")
        
        # 批处理大小
        if not (1000 <= self.batch_size <= 100000):
            if self.batch_size < 1:
                raise ValueError(f"batch_size必须大于0，当前值: {self.batch_size}")
            elif self.batch_size < 1000:
                print(f"⚠️  警告: batch_size={self.batch_size}过小，可能影响性能。推荐范围: 10000-50000")
            elif self.batch_size > 100000:
                print(f"⚠️  警告: batch_size={self.batch_size}过大，可能占用过多内存。推荐范围: 10000-50000")
        
        # Worker数量
        if self.num_workers is not None:
            if not (1 <= self.num_workers <= 128):
                if self.num_workers < 1:
                    raise ValueError(f"num_workers必须大于0，当前值: {self.num_workers}")
                elif self.num_workers > 128:
                    print(f"⚠️  警告: num_workers={self.num_workers}过大。推荐范围: 1-32")
        
        # GPU设备ID
        if self.gpu_device < 0 or self.gpu_device > 7:
            raise ValueError(f"gpu_device必须在0-7范围内，当前值: {self.gpu_device}")
        
        # LSH参数
        if self.use_lsh:
            if not (3 <= self.lsh_num_tables <= 50):
                if self.lsh_num_tables < 1:
                    raise ValueError(f"lsh_num_tables必须大于0，当前值: {self.lsh_num_tables}")
                elif self.lsh_num_tables > 50:
                    print(f"⚠️  警告: lsh_num_tables={self.lsh_num_tables}过大，可能影响内存和性能。推荐范围: 5-20")
            
            if not (6 <= self.lsh_hash_size <= 16):
                if self.lsh_hash_size < 1:
                    raise ValueError(f"lsh_hash_size必须大于0，当前值: {self.lsh_hash_size}")
                elif self.lsh_hash_size > 16:
                    print(f"⚠️  警告: lsh_hash_size={self.lsh_hash_size}过大，bucket数量会过多。推荐范围: 8-12")
        
        # 6. 类型验证
        if not isinstance(self.use_gpu, bool):
            raise TypeError(f"use_gpu必须是布尔值，当前类型: {type(self.use_gpu)}")
        
        if not isinstance(self.use_lsh, bool):
            raise TypeError(f"use_lsh必须是布尔值，当前类型: {type(self.use_lsh)}") 