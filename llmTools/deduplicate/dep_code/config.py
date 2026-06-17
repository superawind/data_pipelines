#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
流式去重配置模块

定义三种流式去重方法的配置参数：
1. 流式严格哈希去重 (StreamingStrictHashDeduplicator)
2. 流式N-gram LSH去重 (StreamingNgramLshDeduplicator)  
3. 流式Embedding去重 (StreamingEmbeddingDeduplicator)
"""

import os
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional


class DedupMethod(Enum):
    """
    去重方法枚举
    
    三种流式去重算法：
    - STRICT_HASH: 基于内容哈希的完全匹配去重，速度最快
    - NGRAM_LSH: 基于N-gram和LSH的近似匹配去重，平衡速度和精度
    - EMBEDDING: 基于向量嵌入的语义相似度去重，精度最高
    """
    STRICT_HASH = "strict_hash"
    NGRAM_LSH = "ngram_lsh"
    EMBEDDING = "embedding"


class EmbeddingFileFormat(Enum):
    """
    Embedding文件格式枚举
    
    支持的embedding文件格式：
    - NPY: numpy数组文件，形状(n_samples, embedding_dim)
    - JSONL: 每行包含prompt_id和embedding的JSONL文件
    """
    NPY = "npy"
    JSONL = "jsonl"


@dataclass
class DedupConfig:
    """
    流式去重配置类
    
    包含基础配置和三种方法的专用配置
    """
    
    # ===== 基础必需配置 =====
    method: DedupMethod
    """去重方法：strict_hash(完全匹配) | ngram_lsh(近似匹配) | embedding(语义匹配)"""
    
    input_file: str
    """输入文件路径，支持任意大小的JSONL文件，采用流式读取"""
    
    output_file: str
    """输出文件路径，去重后的唯一数据将写入此文件"""
    
    content_keys: List[str] = field(default_factory=lambda: ["prompt"])
    """
    用于去重的内容字段列表
    - 支持多个字段组合去重，如 ["prompt", "instruction"]
    - 多个字段会按空格连接后进行去重判断
    - 默认使用 "prompt" 字段
    """
    
    field_order: List[str] = field(default_factory=lambda: ["prompt", "instruction", "input", "output"])
    """
    JSON字段的输出顺序
    - 用于控制输出文件中字段的排列顺序
    - 只影响输出格式，不影响去重逻辑
    - 默认顺序：prompt -> instruction -> input -> output
    """
    
    # ===== 通用性能配置 =====
    num_workers: Optional[int] = None
    """
    并行处理的worker数量
    - None: 自动检测CPU核心数并优化设置
    - 建议设置为CPU核心数的1-2倍
    - 对于120GB大文件，建议设置为16或更高
    """
    
    chunk_size_mb: int = 16
    """
    分块大小（MB）
    - 控制每个处理块的大小，影响内存使用和并行效率
    - 值越大：减少I/O开销，提升处理速度，但增加内存使用
    - 值越小：降低内存压力，提高并行度，但增加I/O开销
    - 推荐值：16MB（标准），64MB（高性能），128MB（极致性能）
    """
    
    queue_maxsize: int = 8
    """
    队列最大缓存块数
    - 控制内存中同时缓存的数据块数量
    - 总内存使用 = chunk_size_mb × queue_maxsize
    - 值越大：更好的并行流水线，但消耗更多内存
    - 推荐值：8（标准），16（高内存），4（低内存）
    """
    
    buffer_size_mb: int = 2
    """
    I/O缓冲区大小（MB）
    - 控制文件读写的缓冲区大小
    - 值越大：减少系统调用次数，提升I/O性能
    - 推荐值：2MB（标准），8MB（高性能），16MB（极致性能）
    """
    
    log_dir: str = None
    """
    日志文件目录
    - 存储详细的去重日志、重复数据记录、阈值边界示例
    - 包含性能统计和分析报告
            - 默认目录：根据模式自动选择（stream_dedup_logs/ 或 global_dedup_logs/）
    """
    
    # ===== N-gram LSH 专用配置 =====
    ngram_size: int = 10
    """
    N-gram的大小（仅ngram_lsh方法使用）
    - 控制文本分割的粒度，影响相似度检测精度
    - 值越大：精度越高，但计算越慢
    - 值越小：计算越快，但可能产生误匹配
    - 推荐范围：8-12，默认10
    """
    
    jaccard_threshold: float = 0.85
    """
    Jaccard相似度阈值（仅ngram_lsh方法使用）
    - 判定两个文本为相似的最低Jaccard相似度
    - 范围：0.0-1.0，1.0表示完全相同
    - 值越高：去重越严格，保留数据越多
    - 值越低：去重越宽松，删除数据越多
    - 推荐范围：0.8-0.9，默认0.85
    """
    
    num_permutations: int = 128
    """
    MinHash排列数量（仅ngram_lsh方法使用）
    - 控制LSH算法的精度和速度平衡
    - 值越大：精度越高，但内存消耗和计算时间增加
    - 值越小：速度越快，但可能遗漏相似项
    - 推荐范围：64-256，默认128
    """
    
    # ===== Embedding 专用配置 =====
    embeddings_file: Optional[str] = None
    """
    嵌入向量文件路径（embedding方法必需）
    - 支持两种格式：.npy格式和.jsonl格式
    - NPY格式：numpy数组文件，形状(样本数, 向量维度)，向量顺序必须与输入文件行顺序一致
    - JSONL格式：每行包含prompt_id和embedding字段的JSON文件
    - 支持各种嵌入模型：BERT、RoBERTa、Sentence-BERT等
    """
    
    embeddings_files: Optional[List[str]] = None
    """
    多个嵌入向量文件路径列表（embedding方法可选）
    - 当大的embedding文件被分割成多个小文件时使用
    - 支持JSONL格式的多文件加载
    - 如果设置了此参数，将优先使用此参数而不是embeddings_file
    - 例：["embeddings_part1.jsonl", "embeddings_part2.jsonl"]
    """
    
    embeddings_format: EmbeddingFileFormat = EmbeddingFileFormat.NPY
    """
    Embedding文件格式（仅embedding方法使用）
    - NPY: 传统numpy数组格式，要求与输入文件行顺序一致
    - JSONL: 新格式，每行包含prompt_id和embedding，支持基于ID匹配
    - 默认：NPY（向后兼容）
    """
    
    prompt_id_key: str = "prompt_id"
    """
    输入数据中prompt_id字段名（仅embedding方法使用JSONL格式时）
    - 用于与embedding文件中的prompt_id进行匹配
    - 默认：prompt_id
    """
    
    embedding_id_key: str = "prompt_id"
    """
    embedding文件中ID字段名（仅JSONL格式时）
    - embedding JSONL文件中存储ID的字段名
    - 默认：prompt_id
    - 可自定义为其他字段名，如"id", "sample_id"等
    """
    
    embedding_vector_key: str = "embedding"
    """
    embedding文件中向量字段名（仅JSONL格式时）
    - embedding JSONL文件中存储向量的字段名
    - 默认：embedding
    - 可自定义为其他字段名，如"vector", "embeddings"等
    """
    
    threshold: float = 0.95
    """
    向量相似度阈值（仅embedding方法使用）
    - 判定两个向量为相似的最低余弦相似度
    - 范围：0.0-1.0，1.0表示完全相同
    - 值越高：去重越严格，保留数据越多  
    - 值越低：去重越宽松，删除数据越多
    - 推荐范围：0.9-0.98，默认0.95
    """
    
    top_k: int = 10
    """
    向量检索的Top-K数量（仅embedding方法使用）
    - 每次查询时检索最相似的K个候选向量
    - 值越大：召回越高，但计算时间增加
    - 值越小：速度越快，但可能遗漏相似项
    - 推荐范围：5-20，默认10
    """
    
    use_gpu: bool = False
    """
    是否使用GPU加速（仅embedding方法使用）
    - True: 使用GPU加速FAISS向量检索（需要faiss-gpu）
    - False: 使用CPU进行向量检索（需要faiss-cpu）
    - GPU模式下处理速度可提升5-10倍
    """
    
    gpu_device: int = 0
    """
    GPU设备ID（仅在use_gpu=True时使用）
    - 指定使用哪个GPU设备
    - 默认使用第0个GPU设备
    - 多GPU环境下可指定不同设备
    """
    
    # ===== 通用批处理配置 =====
    batch_size: int = 10000
    """
    批处理大小（embedding方法使用）
    - 控制embedding去重时的批量处理大小
    - 值越大：内存使用越多，但可能提高处理效率
    - 值越小：内存使用越少，但可能降低处理效率
    - 推荐范围：5000-50000，默认10000
    """
    
    def __post_init__(self):
        """
        配置验证和自动修正
        
        验证内容：
        1. 输入文件存在性检查
        2. embedding方法的向量文件检查
        3. content_keys字段有效性检查
        4. 参数范围合理性检查
        """
        # 验证输入文件
        if not os.path.exists(self.input_file):
            raise FileNotFoundError(f"输入文件不存在: {self.input_file}")
        
        # 验证embedding方法的向量文件
        if self.method == DedupMethod.EMBEDDING:
            if not self.embeddings_file and not self.embeddings_files:
                raise ValueError("embedding方法必须指定embeddings_file或embeddings_files参数")
            
            if self.embeddings_file:
                if not os.path.exists(self.embeddings_file):
                    raise FileNotFoundError(f"嵌入向量文件不存在: {self.embeddings_file}")
                # 根据文件扩展名自动检测格式
                if self.embeddings_file.endswith('.npy'):
                    self.embeddings_format = EmbeddingFileFormat.NPY
                elif self.embeddings_file.endswith('.jsonl'):
                    self.embeddings_format = EmbeddingFileFormat.JSONL
                else:
                    # 如果没有明确扩展名，保持用户设置的格式
                    pass
            elif self.embeddings_files:
                for f in self.embeddings_files:
                    if not os.path.exists(f):
                        raise FileNotFoundError(f"嵌入向量文件不存在: {f}")
                # 多个文件只支持JSONL格式，自动设置为JSONL
                self.embeddings_format = EmbeddingFileFormat.JSONL
        
        # 验证content_keys不为空
        if not self.content_keys:
            raise ValueError("content_keys不能为空")
        
        # 验证content_keys中的字段都在field_order中
        for key in self.content_keys:
            if key not in self.field_order:
                raise ValueError(f"content_key '{key}' 不在field_order中: {self.field_order}")
        
        # 验证参数范围
        if self.jaccard_threshold < 0.0 or self.jaccard_threshold > 1.0:
            raise ValueError(f"jaccard_threshold必须在0.0-1.0范围内: {self.jaccard_threshold}")
        
        if self.threshold < 0.0 or self.threshold > 1.0:
            raise ValueError(f"threshold必须在0.0-1.0范围内: {self.threshold}")
        
        if self.ngram_size < 1:
            raise ValueError(f"ngram_size必须大于0: {self.ngram_size}")
        
        if self.num_permutations < 1:
            raise ValueError(f"num_permutations必须大于0: {self.num_permutations}")
        
        if self.top_k < 1:
            raise ValueError(f"top_k必须大于0: {self.top_k}")
        
        # 日志目录由BaseDeduplicator在运行时创建，此处不处理
        
        # 性能优化建议
        if self.method == DedupMethod.NGRAM_LSH and self.num_permutations > 256:
            print(f"⚠️  警告：num_permutations={self.num_permutations}较大，可能影响性能")
        
        if self.method == DedupMethod.EMBEDDING and self.top_k > 50:
            print(f"⚠️  警告：top_k={self.top_k}较大，可能影响检索速度") 