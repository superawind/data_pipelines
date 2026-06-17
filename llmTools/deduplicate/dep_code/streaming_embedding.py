#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
流式Embedding去重器 - 零延迟启动，实时进度显示

核心特性：
1. 零延迟启动 - 立即开始处理
2. 实时进度显示 - 不再停滞  
3. 边读边处理 - 流式架构
4. 高效向量搜索 - FAISS加速
5. 心跳机制 - 防止假死
6. GPU加速支持 - 可选CUDA
"""

import os
import sys
import time
import threading
import queue
import hashlib
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from collections import deque
import multiprocessing as mp
from tqdm import tqdm
import signal

# 内存和系统监控
try:
    import psutil
except ImportError:
    psutil = None

# 可选的高性能库
try:
    import xxhash
    FAST_HASH_AVAILABLE = True
except ImportError:
    FAST_HASH_AVAILABLE = False

try:
    import orjson
    FAST_JSON_AVAILABLE = True
except ImportError:
    FAST_JSON_AVAILABLE = False

# FAISS导入 - 优雅处理GPU版本兼容性问题
FAISS_AVAILABLE = False
FAISS_GPU_AVAILABLE = False
faiss = None

try:
    # 首先尝试导入faiss-gpu版本
    import faiss
    FAISS_AVAILABLE = True
    
    # 检查GPU支持
    try:
        if faiss.get_num_gpus() > 0:
            FAISS_GPU_AVAILABLE = True
            print("🚀 FAISS-GPU可用")
        else:
            print("🚀 FAISS可用（CPU版本）")
    except Exception as gpu_check_error:
        print(f"🚀 FAISS可用（GPU检查失败: {gpu_check_error}）")
        FAISS_GPU_AVAILABLE = False
        
except ImportError as e:
    print(f"❌ FAISS不可用: {e}")
    print("💡 请安装: pip install faiss-cpu 或 pip install faiss-gpu")
except AttributeError as e:
    print(f"⚠️  FAISS-GPU版本兼容性问题: {e}")
    print("🔄 尝试回退到CPU版本...")
    
    # 回退到CPU版本
    try:
        # 清除可能有问题的GPU模块
        import sys
        faiss_modules = [name for name in sys.modules.keys() if name.startswith('faiss')]
        for module_name in faiss_modules:
            if module_name in sys.modules:
                del sys.modules[module_name]
        
        # 重新导入，强制使用CPU版本
        import faiss
        FAISS_AVAILABLE = True
        FAISS_GPU_AVAILABLE = False
        print("✅ 成功回退到FAISS-CPU版本")
        
    except Exception as fallback_error:
        print(f"❌ FAISS-CPU回退也失败: {fallback_error}")
        print("💡 解决方案:")
        print("   1. 卸载有问题的faiss-gpu: pip uninstall faiss-gpu")
        print("   2. 重新安装CPU版本: pip install faiss-cpu")
        print("   3. 或重新安装兼容的GPU版本: pip install faiss-gpu")
        FAISS_AVAILABLE = False
        FAISS_GPU_AVAILABLE = False
except Exception as e:
    print(f"❌ FAISS导入发生未知错误: {e}")
    print("💡 建议重新安装: pip install faiss-cpu")
    FAISS_AVAILABLE = False
    FAISS_GPU_AVAILABLE = False


try:
    from .base import BaseDeduplicator
    from .config import DedupConfig
except ImportError:
    from base import BaseDeduplicator
    from config import DedupConfig


class StreamingEmbeddingDeduplicator(BaseDeduplicator):
    """流式Embedding去重器 - 零延迟启动，实时进度显示"""
    
    def __init__(self, config: DedupConfig):
        """
        初始化流式Embedding去重器
        """
        # 检查FAISS可用性
        if not FAISS_AVAILABLE:
            raise ImportError(
                "FAISS不可用！Embedding去重需要FAISS库。\n"
                "请安装：pip install faiss-cpu 或 pip install faiss-gpu\n"
                "如果已安装faiss-gpu但有兼容性问题，请尝试：\n"
                "  pip uninstall faiss-gpu && pip install faiss-cpu"
            )
        
        # 调用父类初始化
        super().__init__(config)
        
        # 性能配置
        self.threshold = getattr(config, 'threshold', 0.95)
        self.top_k = getattr(config, 'top_k', 10)
        self.use_gpu = getattr(config, 'use_gpu', False) and FAISS_GPU_AVAILABLE
        self.gpu_device = getattr(config, 'gpu_device', 0)
        
        # 如果用户请求GPU但不可用，给出警告
        if getattr(config, 'use_gpu', False) and not FAISS_GPU_AVAILABLE:
            if FAISS_GPU_AVAILABLE is False and "AttributeError" in str(globals().get('faiss_import_error', '')):
                self.logger.warning("GPU模式不可用（FAISS-GPU兼容性问题），自动使用CPU模式")
                self.logger.warning("建议：pip uninstall faiss-gpu && pip install faiss-cpu")
            else:
                self.logger.warning("GPU模式不可用，自动使用CPU模式")
            self.use_gpu = False
        
        # 流式处理配置 - 使用用户配置参数
        self.num_workers = getattr(config, 'num_workers', None) or min(6, os.cpu_count() or 4)  # Embedding较CPU密集
        chunk_size_mb = getattr(config, 'chunk_size_mb', 8)  # embedding默认用较小的8MB
        buffer_size_mb = getattr(config, 'buffer_size_mb', 2)
        self.chunk_size = chunk_size_mb * 1024 * 1024   # 用户配置的块大小
        self.buffer_size = buffer_size_mb * 1024 * 1024   # 用户配置的读取缓冲区
        self.write_buffer_size = buffer_size_mb * 2 * 1024 * 1024  # 写入缓冲区是读取缓冲区的2倍
        
        # 队列大小控制 - 使用用户配置
        self.queue_maxsize = getattr(config, 'queue_maxsize', 6)   # embedding默认用较小的队列
        self.result_queue_size = min(16, self.queue_maxsize * 3)  # 结果队列大小基于输入队列
        
        # Embedding配置
        self.embeddings_file = getattr(config, 'embeddings_file', None)
        self.embeddings_files = getattr(config, 'embeddings_files', None)
        
        # 检查embedding文件
        if hasattr(config, 'embeddings_files') and config.embeddings_files:
            # 使用多个embedding文件
            for file_path in config.embeddings_files:
                if not file_path or not os.path.exists(file_path):
                    raise FileNotFoundError(f"Embedding文件不存在: {file_path}")
        elif self.embeddings_file:
            if not os.path.exists(self.embeddings_file):
                raise FileNotFoundError(f"Embedding文件不存在: {self.embeddings_file}")
        else:
            raise ValueError("必须指定embeddings_file或embeddings_files参数")
        
        # 选择最快的处理函数
        if FAST_HASH_AVAILABLE:
            self.hash_func = lambda content: str(xxhash.xxh64(content.encode('utf-8')).intdigest())
            self.logger.info("🚀 xxHash极速哈希可用")
        else:
            self.hash_func = lambda content: hashlib.blake2b(content.encode('utf-8'), digest_size=8).hexdigest()
            self.logger.info("⚠️  使用blake2b哈希（建议: pip install xxhash）")
        
        if FAST_JSON_AVAILABLE:
            self.json_loads = lambda line: orjson.loads(line.encode('utf-8'))
            self.logger.info("🚀 使用orjson极速JSON解析")
        else:
            import json
            self.json_loads = json.loads
            self.logger.info("⚠️  使用标准JSON解析（建议: pip install orjson）")
        
        # 加载embeddings
        self.logger.info(f"📂 加载embedding文件: {self.embeddings_file}")
        start_load = time.time()
        
        # 🔧 大规模优化：使用内存映射而不是全加载
        # 根据配置的格式加载embedding
        if hasattr(config, 'embeddings_files') and config.embeddings_files:
            self.embeddings, self.prompt_id_to_index = self._load_embeddings_from_multiple_jsonl_files()
            self.use_prompt_id_mapping = True
        elif hasattr(config, 'embeddings_format') and config.embeddings_format.value == 'jsonl':
            self.embeddings, self.prompt_id_to_index = self._load_embeddings_from_jsonl_massive()
            self.use_prompt_id_mapping = True
        else:
            # 默认使用NPY格式 - 使用内存映射
            self.embeddings, self.embeddings_normalized = self._load_embeddings_npy_massive()
            self.prompt_id_to_index = {}
            self.use_prompt_id_mapping = False
            
        load_time = time.time() - start_load
        
        self.embedding_dim = self.embeddings.shape[1] if isinstance(self.embeddings, np.ndarray) else len(next(iter(self.embeddings.values())))
        self.logger.info(f"✅ Embedding加载完成: {self.embeddings.shape if isinstance(self.embeddings, np.ndarray) else len(self.embeddings)} 维度={self.embedding_dim}, 耗时{load_time:.1f}秒")
        
        if self.use_prompt_id_mapping:
            self.logger.info(f"📊 加载了 {len(self.prompt_id_to_index)} 个prompt_id映射")
        
        # 初始化FAISS索引
        self._init_faiss_index()
        
        # 统计信息
        self.stats = {
            'total_processed': 0,
            'unique_count': 0,
            'chunks_processed': 0,
            'chunks_read': 0,
            'chunks_written': 0,
            'start_time': 0,
            'processing_speed_mb_s': 0,
            'last_update_time': 0,
            'bytes_processed': 0,
            'last_heartbeat': 0,
            'similarity_comparisons': 0,
            'vector_searches': 0,
            'missing_id_field': 0,
            'missing_embedding': 0
        }
        
        # 控制标志
        self.stop_flag = threading.Event()
        self.debug_mode = True
        
        # 已处理的向量索引和内容映射
        self.processed_vectors = []
        self.vector_to_content = {}
        self.content_to_index = {}
        self.vector_lock = threading.RLock()
        
        self.logger.info(f"🎯 流式Embedding去重器初始化完成，{self.num_workers}个worker，块大小{self.chunk_size//1024//1024}MB")
        self.logger.info(f"🔍 相似度阈值: {self.threshold}, Top-K: {self.top_k}")
    
    def _load_embeddings_npy_massive(self) -> Tuple[np.memmap, bool]:
        """🔧 大规模NPY文件优化：使用内存映射而不是全加载到内存，严格验证数量匹配"""
        self.logger.info(f"📂 大规模NPY文件：使用内存映射加载 {self.embeddings_file}")
        
        file_size = os.path.getsize(self.embeddings_file) / (1024**3)
        available_memory = psutil.virtual_memory().available / (1024**3)
        
        self.logger.info(f"📁 Embedding文件大小: {file_size:.2f} GB")
        self.logger.info(f"💾 可用内存: {available_memory:.2f} GB")
        
        # 先用内存映射快速读取基本信息
        temp_embeddings = np.load(self.embeddings_file, mmap_mode='r')
        embedding_count = len(temp_embeddings)
        self.logger.info(f"🎯 Embedding数量: {embedding_count:,}")
        
        # 验证数据文件行数
        print("📊 验证数据文件与embedding数量匹配...")
        data_line_count = 0
        with open(self.config.input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data_line_count += 1
        
        print(f"📄 数据文件行数: {data_line_count:,}")
        print(f"🎯 Embedding数量: {embedding_count:,}")
        
        # 严格验证数量匹配
        if data_line_count != embedding_count:
            error_msg = (
                f"❌ NPY格式数量不匹配错误！\n"
                f"   📄 数据文件行数: {data_line_count:,}\n"
                f"   🎯 Embedding数量: {embedding_count:,}\n"
                f"   📐 差异: {abs(data_line_count - embedding_count):,} 条\n\n"
                f"💡 NPY格式要求数据和embedding严格按行索引一一对应！\n"
                f"   请确保：\n"
                f"   - 数据文件第i行 ↔ embedding文件第i个向量\n"
                f"   - 两个文件的数量完全相等\n\n"
                f"🔧 解决方案：\n"
                f"   1. 重新生成embedding确保数量匹配\n"
                f"   2. 或使用JSONL格式支持ID匹配: --embeddings_format jsonl"
            )
            self.logger.error(error_msg)
            raise ValueError(f"NPY格式数量不匹配：数据{data_line_count:,}行 vs embedding{embedding_count:,}个")
        
        print("✅ 数量验证通过，开始加载embedding...")
        
        if file_size > available_memory * 0.8:
            self.logger.info("🚀 启用内存映射模式 - 避免内存爆炸")
        
        try:
            # 使用内存映射而不是全加载
            embeddings_mmap = np.memmap(self.embeddings_file, dtype=np.float32, mode='r')
            
            # 自动检测embedding维度（假设是2D数组）
            # 已经加载过temp_embeddings，直接使用其形状
            embedding_shape = temp_embeddings.shape
            
            self.logger.info(f"📊 Embedding形状: {embedding_shape}")
            self.embedding_dim = embedding_shape[1]
            
            # 重新创建正确形状的memmap
            embeddings_mmap = np.memmap(self.embeddings_file, dtype=np.float32, mode='r', shape=embedding_shape)
            
            # 检查是否已经归一化（抽样检查前1000个向量）
            sample_size = min(1000, embedding_shape[0])
            sample_vectors = embeddings_mmap[:sample_size]
            sample_norms = np.linalg.norm(sample_vectors, axis=1)
            
            # 如果大部分向量的模接近1，认为已经归一化
            normalized_ratio = np.sum(np.abs(sample_norms - 1.0) < 0.01) / len(sample_norms)
            is_normalized = normalized_ratio > 0.9
            
            if is_normalized:
                self.logger.info("✅ 检测到向量已归一化")
            else:
                self.logger.info("⚠️ 向量未归一化，将在使用时动态归一化")
            
            self.logger.info(f"✅ 内存映射加载完成: {embedding_shape}, 内存占用: ~0 GB")
            
            return embeddings_mmap, is_normalized
            
        except Exception as e:
            self.logger.error(f"❌ 内存映射失败: {e}")
            self.logger.info("🔄 回退到传统加载方式...")
            # 回退到原来的方式（但会有内存问题）
            embeddings = np.load(self.embeddings_file)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            embeddings = embeddings / norms
            return embeddings, True

    def _load_embeddings_from_jsonl_massive(self) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
        """🔧 大规模JSONL文件优化：分块读取，避免全加载"""
        embedding_id_key = getattr(self.config, 'embedding_id_key', 'prompt_id')
        embedding_vector_key = getattr(self.config, 'embedding_vector_key', 'embedding')
        
        file_size = os.path.getsize(self.embeddings_file) / (1024**3)
        self.logger.info(f"📂 大规模JSONL文件处理: {file_size:.2f} GB")
        self.logger.info(f"📖 分块读取模式 (ID字段: {embedding_id_key}, 向量字段: {embedding_vector_key})")
        
        # 使用字典存储，支持按需访问
        embeddings_dict = {}
        prompt_id_to_index = {}
        
        chunk_size = 10000  # 每次处理10000行
        current_chunk = []
        total_processed = 0
        
        try:
            import psutil
        except ImportError:
            psutil = None
        
        with open(self.embeddings_file, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                
                current_chunk.append((line_idx, line))
                
                if len(current_chunk) >= chunk_size:
                    # 处理当前块
                    processed_count = self._process_embedding_chunk(
                        current_chunk, embeddings_dict, prompt_id_to_index,
                        embedding_id_key, embedding_vector_key
                    )
                    total_processed += processed_count
                    
                    # 内存监控
                    if psutil and line_idx % (chunk_size * 5) == 0:
                        memory_usage = psutil.virtual_memory().percent
                        self.logger.info(f"📊 已处理 {total_processed:,} 个向量，内存使用: {memory_usage:.1f}%")
                        
                        if memory_usage > 85:
                            self.logger.warning("⚠️ 内存使用过高，建议减小块大小或使用NPY格式")
                    
                    current_chunk = []
            
            # 处理最后一块
            if current_chunk:
                processed_count = self._process_embedding_chunk(
                    current_chunk, embeddings_dict, prompt_id_to_index,
                    embedding_id_key, embedding_vector_key
                )
                total_processed += processed_count
        
        self.logger.info(f"✅ 大规模JSONL处理完成: {total_processed:,} 个向量")
        
        return embeddings_dict, prompt_id_to_index
    
    def _process_embedding_chunk(self, chunk, embeddings_dict, prompt_id_to_index, embedding_id_key, embedding_vector_key):
        """处理embedding块"""
        processed_count = 0
        
        for line_idx, line in chunk:
            try:
                data = self.json_loads(line)
                
                if embedding_id_key not in data or embedding_vector_key not in data:
                    continue
                
                prompt_id = str(data[embedding_id_key])
                embedding = data[embedding_vector_key]
                
                # 转换为numpy数组
                if isinstance(embedding, list):
                    embedding = np.array(embedding, dtype=np.float32)
                elif isinstance(embedding, np.ndarray):
                    embedding = embedding.astype(np.float32)
                else:
                    continue
                
                # 归一化向量
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                    embeddings_dict[prompt_id] = embedding
                    prompt_id_to_index[prompt_id] = len(prompt_id_to_index)
                    processed_count += 1
                
            except Exception:
                continue
        
        return processed_count

    def _load_embeddings_from_multiple_jsonl_files(self) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
        """🔧 大规模JSONL文件优化：从多个JSONL文件加载embedding"""
        embedding_id_key = getattr(self.config, 'embedding_id_key', 'prompt_id')
        embedding_vector_key = getattr(self.config, 'embedding_vector_key', 'embedding')
        
        all_embeddings = {}
        prompt_id_to_index = {}
        
        for embeddings_file in self.config.embeddings_files:
            self.logger.info(f"📂 加载embedding文件: {embeddings_file}")
            file_size = os.path.getsize(embeddings_file) / (1024**3)
            self.logger.info(f"📁 Embedding文件大小: {file_size:.2f} GB")
            
            try:
                with open(embeddings_file, 'r', encoding='utf-8') as f:
                    for line_idx, line in enumerate(f):
                        line = line.strip()
                        if not line:
                            continue
                        
                        try:
                            data = self.json_loads(line)
                            
                            if embedding_id_key not in data or embedding_vector_key not in data:
                                self.logger.warning(f"跳过包含无效字段的行: {line}")
                                continue
                            
                            prompt_id = str(data[embedding_id_key])
                            embedding = data[embedding_vector_key]
                            
                            # 转换为numpy数组
                            if isinstance(embedding, list):
                                embedding = np.array(embedding, dtype=np.float32)
                            elif isinstance(embedding, np.ndarray):
                                embedding = embedding.astype(np.float32)
                            else:
                                self.logger.warning(f"跳过非数组类型的embedding: {embedding}")
                                continue
                            
                            # 归一化向量
                            norm = np.linalg.norm(embedding)
                            if norm > 0:
                                embedding = embedding / norm
                                all_embeddings[prompt_id] = embedding
                                prompt_id_to_index[prompt_id] = len(prompt_id_to_index)
                            else:
                                self.logger.warning(f"跳过零向量embedding: {prompt_id}")
                                continue
                            
                        except Exception as e:
                            self.logger.warning(f"处理embedding文件 {embeddings_file} 行 {line_idx} 失败: {e}")
                            continue
            
            except FileNotFoundError:
                self.logger.error(f"embedding文件不存在: {embeddings_file}")
                raise
            except Exception as e:
                self.logger.error(f"加载embedding文件 {embeddings_file} 失败: {e}")
                raise
        
        self.logger.info(f"✅ 从 {len(self.config.embeddings_files)} 个embedding文件加载完成: {len(all_embeddings)} 个prompt_id")
        return all_embeddings, prompt_id_to_index
    
    def _extract_content(self, data: dict) -> str:
        """提取内容用于去重"""
        content_values = []
        
        for key in self.config.content_keys:
            if key in data:
                content_values.append(str(data[key]))
        
        if content_values:
            return ' '.join(content_values)
        else:
            return ""

    def _init_faiss_index(self):
        """初始化FAISS索引"""
        self.logger.info("🔧 初始化FAISS索引...")
        
        # 创建索引 - 使用Inner Product (余弦相似度)
        if self.use_gpu and FAISS_GPU_AVAILABLE:
            try:
                # GPU版本
                res = faiss.StandardGpuResources()
                self.faiss_index = faiss.GpuIndexFlatIP(res, self.embedding_dim)
                self.logger.info(f"🔥 使用GPU索引 (设备{self.gpu_device})")
            except Exception as e:
                self.logger.warning(f"GPU初始化失败，使用CPU: {e}")
                self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
                self.logger.info("💻 使用CPU索引")
        else:
            # CPU版本
            self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
            self.logger.info("💻 使用CPU索引")
        
        self.logger.info("✅ FAISS索引初始化完成")

    def _check_embedding_similarity_with_logging(self, content: str, line_idx: int, data: dict) -> dict:
        """🔧 大规模优化：使用Embedding检查相似度，支持内存映射和分块访问"""
        # 根据embedding文件格式确定查询向量
        if self.use_prompt_id_mapping:
            # 使用prompt_id匹配（JSONL格式，包括多文件）
            prompt_id_key = getattr(self.config, 'prompt_id_key', 'prompt_id')
            if prompt_id_key not in data:
                # 缺少ID字段：跳过此数据，不处理
                self.stats['missing_id_field'] = self.stats.get('missing_id_field', 0) + 1
                if self.stats['missing_id_field'] <= 10:  # 只记录前10个警告
                    self.logger.warning(f"数据缺少ID字段 '{prompt_id_key}': {list(data.keys())[:5]}")
                return {'is_duplicate': False, 'similarity': 0.0, 'matched_content': '', 'matched_data': None, 'skip_data': True}
            
            prompt_id = str(data[prompt_id_key])
            
            # 🔧 大规模优化：从字典中获取embedding（支持多文件JSONL）
            if isinstance(self.embeddings, dict):
                if prompt_id not in self.embeddings:
                    # 缺少embedding：跳过此数据，不处理
                    self.stats['missing_embedding'] = self.stats.get('missing_embedding', 0) + 1
                    # 不记录每个缺失的embedding，数量可能很大
                    return {'is_duplicate': False, 'similarity': 0.0, 'matched_content': '', 'matched_data': None, 'skip_data': True}
                query_vector = self.embeddings[prompt_id].reshape(1, -1)
            else:
                # 传统方式（向后兼容）
                if prompt_id not in self.prompt_id_to_index:
                    # 缺少embedding：跳过此数据，不处理
                    self.stats['missing_embedding'] = self.stats.get('missing_embedding', 0) + 1
                    return {'is_duplicate': False, 'similarity': 0.0, 'matched_content': '', 'matched_data': None, 'skip_data': True}
                embedding_idx = self.prompt_id_to_index[prompt_id]
                query_vector = self.embeddings[embedding_idx:embedding_idx+1].astype(np.float32)
        else:
            # 使用行索引匹配（NPY格式，内存映射）
            # NPY格式已经严格验证数量匹配，不会超出范围
            
            # 🔧 大规模优化：从内存映射中读取单个向量
            query_vector = self.embeddings[line_idx:line_idx+1].astype(np.float32)
            
            # 如果向量未归一化，动态归一化
            if not hasattr(self, 'embeddings_normalized') or not self.embeddings_normalized:
                norm = np.linalg.norm(query_vector)
                if norm > 0:
                    query_vector = query_vector / norm
                else:
                    self.logger.warning(f"查询向量为零向量，跳过")
                    return {'is_duplicate': False, 'similarity': 0.0, 'matched_content': '', 'matched_data': None}
        
        # 对于已归一化的向量，不需要再次归一化
        if not (hasattr(self, 'embeddings_normalized') and self.embeddings_normalized):
            norm = np.linalg.norm(query_vector)
            if norm > 0:
                query_vector = query_vector / norm
            else:
                self.logger.warning(f"查询向量为零向量，跳过")
                return {'is_duplicate': False, 'similarity': 0.0, 'matched_content': '', 'matched_data': None}
        
        max_similarity = 0.0
        best_match_content = ""
        best_match_data = None
        
        with self.vector_lock:
            # 如果索引为空，直接添加
            if self.faiss_index.ntotal == 0:
                self.faiss_index.add(query_vector)
                self.processed_vectors.append(query_vector[0])
                self.vector_to_content[len(self.processed_vectors)-1] = content
                self.content_to_index[content] = len(self.processed_vectors)-1
                
                # 保存数据映射
                if not hasattr(self, 'content_to_data'):
                    self.content_to_data = {}
                self.content_to_data[content] = data
                
                return {'is_duplicate': False, 'similarity': 0.0, 'matched_content': '', 'matched_data': None}
            
            # 搜索相似向量
            similarities, indices = self.faiss_index.search(query_vector, min(self.top_k, self.faiss_index.ntotal))
            self.stats['vector_searches'] += 1
            
            # 分析所有相似度结果
            for sim, idx in zip(similarities[0], indices[0]):
                if idx >= 0:
                    # 🔧 关键修复：验证相似度范围，确保在[-1, 1]范围内
                    if sim > 1.0001 or sim < -1.0001:  # 允许小的浮点误差
                        self.logger.error(f"❌ 检测到无效的余弦相似度: {sim}，这不应该发生！")
                        self.logger.error(f"查询向量norm: {np.linalg.norm(query_vector)}")
                        # 尝试获取存储向量的norm
                        stored_vector = self.processed_vectors[idx] if idx < len(self.processed_vectors) else None
                        if stored_vector is not None:
                            self.logger.error(f"存储向量norm: {np.linalg.norm(stored_vector)}")
                        # 限制到有效范围
                        sim = max(-1.0, min(1.0, sim))
                        self.logger.warning(f"相似度已限制到有效范围: {sim}")
                    
                    # 跟踪最高相似度
                    if sim > max_similarity:
                        max_similarity = sim
                        best_match_content = self.vector_to_content.get(idx, "")
                        # 尝试获取完整数据
                        best_match_data = getattr(self, 'content_to_data', {}).get(best_match_content, {
                            'content': best_match_content,
                            'vector_index': idx
                        })
                    
                    # 检查是否达到重复阈值
                    if sim >= self.threshold:
                        self.stats['similarity_comparisons'] += 1
                        
                        # 记录重复数据
                        if self.duplicate_count < 1000:
                            method_info = {
                                'cosine_similarity': float(sim),
                                'threshold': self.threshold,
                                'embedding_dim': self.embedding_dim,
                                'top_k': self.top_k,
                                'vector_index': int(idx),
                                'faiss_total': self.faiss_index.ntotal,
                                'query_vector_norm': float(np.linalg.norm(query_vector)),
                                'vector_normalized': True,
                                'embedding_format': 'jsonl' if self.use_prompt_id_mapping else 'npy'
                            }
                            
                            self.log_duplicate_record(
                                removed_data=data,
                                kept_data=best_match_data,
                                similarity=float(sim),
                                method_specific_info=method_info
                            )
                        
                        return {
                            'is_duplicate': True, 
                            'similarity': float(sim), 
                            'matched_content': best_match_content,
                            'matched_data': best_match_data
                        }
            
            # 检查是否接近阈值（记录未去重但相似度较高的数据）
            if max_similarity > 0 and max_similarity >= self.threshold * 0.85:  # 85%的阈值作为"接近"
                if self.threshold_example_count < 1000:
                    method_info = {
                        'cosine_similarity': float(max_similarity),
                        'threshold': self.threshold,
                        'embedding_dim': self.embedding_dim,
                        'top_k': self.top_k,
                        'faiss_total': self.faiss_index.ntotal,
                        'query_vector_norm': float(np.linalg.norm(query_vector)),
                        'vector_normalized': True,
                        'embedding_format': 'jsonl' if self.use_prompt_id_mapping else 'npy'
                    }
                    
                    self.log_threshold_example(
                        data1=data,
                        data2=best_match_data,
                        similarity=float(max_similarity),
                        threshold=self.threshold,
                        method_specific_info=method_info
                    )
            
            # 不是重复，添加到索引
            self.faiss_index.add(query_vector)
            self.processed_vectors.append(query_vector[0])
            self.vector_to_content[len(self.processed_vectors)-1] = content
            self.content_to_index[content] = len(self.processed_vectors)-1
            
            # 保存数据映射用于日志记录
            if not hasattr(self, 'content_to_data'):
                self.content_to_data = {}
            self.content_to_data[content] = data
            
            return {
                'is_duplicate': False, 
                'similarity': float(max_similarity), 
                'matched_content': best_match_content,
                'matched_data': best_match_data
            }

    def _read_file_streaming(self, input_queue: queue.Queue):
        """流式读取文件"""
        try:
            with open(self.config.input_file, 'r', encoding='utf-8', buffering=self.buffer_size) as f:
                current_chunk = []
                current_size = 0
                line_num = 0
                total_bytes_read = 0
                
                self.logger.info("📖 文件读取线程已启动")
                file_size = os.path.getsize(self.config.input_file)
                self.logger.info(f"📁 文件大小: {file_size / (1024**3):.2f} GB")
                
                for line in f:
                    if self.stop_flag.is_set():
                        break
                    
                    line = line.strip()
                    if not line:
                        continue
                    
                    line_bytes = len(line.encode('utf-8'))
                    current_chunk.append((line_num, line))
                    current_size += line_bytes
                    total_bytes_read += line_bytes
                    line_num += 1
                    
                    # 当块达到目标大小时，放入队列
                    if current_size >= self.chunk_size:
                        if not self.stop_flag.is_set():
                            input_queue.put(('chunk', current_chunk.copy()))
                            self.stats['chunks_read'] += 1
                            self.stats['bytes_processed'] = total_bytes_read
                            
                            if self.debug_mode and self.stats['chunks_read'] % 5 == 0:
                                progress = (total_bytes_read / file_size) * 100
                                self.logger.info(f"📖 读取进度 {progress:.1f}% - 块{self.stats['chunks_read']}, {total_bytes_read/(1024**2):.1f}MB")
                        
                        current_chunk = []
                        current_size = 0
                
                # 处理最后一个块
                if current_chunk and not self.stop_flag.is_set():
                    input_queue.put(('chunk', current_chunk))
                    self.stats['chunks_read'] += 1
                    self.stats['bytes_processed'] = total_bytes_read
                
                # 发送结束信号
                input_queue.put(('end', None))
                self.logger.info(f"📖 文件读取完成: {total_bytes_read:,} 字节, 共 {self.stats['chunks_read']} 个块")
                
        except Exception as e:
            self.logger.error(f"文件读取错误: {e}")
            input_queue.put(('error', str(e)))

    def _process_chunk_streaming(self, chunk_data: List[Tuple[int, str]], result_queue: queue.Queue):
        """流式处理数据块 - 添加详细的向量相似度分析和记录"""
        try:
            unique_items = []
            processed_count = 0
            skipped_count = 0
            
            for line_num, line in chunk_data:
                try:
                    # 解析JSON
                    if line.strip():
                        data = self.json_loads(line)
                        content = self._extract_content(data)
                        
                        if content:
                            # Embedding去重检查，获取详细相似度信息
                            similarity_result = self._check_embedding_similarity_with_logging(content, line_num, data)
                            
                            # 检查是否需要跳过（缺失embedding或ID字段）
                            if similarity_result.get('skip_data', False):
                                skipped_count += 1
                                continue
                            
                            if not similarity_result['is_duplicate']:
                                unique_items.append((line_num, line))
                            
                        processed_count += 1
                
                except Exception as e:
                    self.logger.warning(f"行 {line_num} 处理失败: {e}")
                    continue
            
            result_queue.put(('result', {
                'unique_items': unique_items,
                'processed_count': processed_count,
                'skipped_count': skipped_count,
                'chunk_id': id(chunk_data)
            }))
            
        except Exception as e:
            self.logger.error(f"块处理错误: {e}")
            result_queue.put(('error', str(e)))

    def _write_results_streaming(self, result_queue: queue.Queue):
        """流式写入结果"""
        try:
            with open(self.config.output_file, 'w', encoding='utf-8', buffering=self.write_buffer_size) as f:
                self.logger.info("💾 结果写入线程已启动")
                
                while True:
                    try:
                        msg_type, data = result_queue.get(timeout=5.0)
                        
                        if msg_type == 'result':
                            unique_items = data['unique_items']
                            
                            for line_num, line in unique_items:
                                f.write(line + '\n')
                            
                            self.stats['unique_count'] += len(unique_items)
                            self.stats['total_processed'] += data['processed_count']
                            self.stats['chunks_written'] += 1
                            
                        elif msg_type == 'end':
                            break
                        elif msg_type == 'error':
                            self.logger.error(f"写入错误: {data}")
                    
                    except queue.Empty:
                        if self.stop_flag.is_set():
                            break
                        continue
                
                self.logger.info(f"💾 写入完成: {self.stats['unique_count']:,} 唯一行")
                
        except Exception as e:
            self.logger.error(f"写入线程错误: {e}")

    def _monitor_progress(self):
        """监控进度并实时显示"""
        pbar = None
        try:
            # 创建进度条
            pbar = tqdm(
                desc="🚀 流式Embedding处理",
                unit='行',
                unit_scale=True,
                bar_format='{desc}: {n_fmt} | {rate_fmt} | {elapsed} | {postfix}',
                position=0,
                leave=True,
                total=None  # 明确设置为None
            )
            
            last_processed = 0
            last_time = time.time()
            heartbeat_counter = 0
            
            while not self.stop_flag.is_set():
                current_time = time.time()
                current_processed = self.stats['total_processed']
                
                # 计算速度
                time_diff = current_time - last_time
                processed_diff = current_processed - last_processed
                
                if time_diff > 0:
                    lines_per_sec = processed_diff / time_diff
                    mb_per_sec = (self.stats['bytes_processed'] / (1024 * 1024)) / (current_time - self.stats['start_time']) if current_time > self.stats['start_time'] else 0
                    self.stats['processing_speed_mb_s'] = mb_per_sec
                
                # 更新进度条
                pbar.n = current_processed
                
                # 确定当前状态
                chunks_in_queue = self.stats['chunks_read'] - self.stats['chunks_processed']
                if chunks_in_queue > 0:
                    status = "处理中"
                    postfix = f"块={self.stats['chunks_processed']}/{self.stats['chunks_read']}, 唯一={self.stats['unique_count']:,}, 速度={lines_per_sec:.0f}行/s, MB/s={mb_per_sec:.1f}, 队列={chunks_in_queue}"
                else:
                    heartbeat_counter += 1
                    status = "等待数据" if self.stats['chunks_read'] > 0 else "启动中"
                    postfix = f"状态={status}, 块={self.stats['chunks_processed']}/{self.stats['chunks_read']}, 队列={chunks_in_queue}, 等待={time_diff:.1f}s"
                
                pbar.set_postfix_str(postfix)
                pbar.refresh()
                
                # 心跳日志
                if heartbeat_counter % 5 == 0 and self.debug_mode:
                    self.stats['last_heartbeat'] = current_time
                
                last_processed = current_processed
                last_time = current_time
                
                # 更新频率：每0.2秒
                time.sleep(0.2)
        
        except Exception as e:
            self.logger.error(f"进度监控错误: {e}")
        finally:
            if pbar is not None:
                pbar.close()

    def deduplicate(self) -> Tuple[List[int], List[int], Dict[str, Any]]:
        """执行流式Embedding去重 - 零延迟启动，实时进度显示"""
        print("🔍 开始流式去重...")
        
        unique_items = []
        total_processed = 0
        total_skipped = 0  # 跳过的数据计数
        error_count = 0
        
        start_time = time.time()
        
        # 启动多线程处理
        print(f"🏃 启动 {self.num_workers} 个处理线程...")
        
        # 创建队列
        input_queue = queue.Queue(maxsize=self.queue_maxsize)
        result_queue = queue.Queue(maxsize=self.result_queue_size)
        
        # 获取文件大小用于进度显示
        total_size = os.path.getsize(self.config.input_file)
        
        with tqdm(total=total_size, desc="🔍 流式去重", unit="B", unit_scale=True,
                 bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
            
            # 启动文件读取线程
            read_thread = threading.Thread(
                target=self._read_file_streaming,
                args=(input_queue,),
                daemon=True
            )
            read_thread.start()
            
            # 处理数据流
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = []
                chunks_submitted = 0
                
                try:
                    while True:
                        try:
                            msg_type, data = input_queue.get(timeout=5.0)
                            
                            if msg_type == 'chunk':
                                # 提交处理任务
                                future = executor.submit(
                                    self._process_chunk_streaming,
                                    data,
                                    result_queue
                                )
                                futures.append(future)
                                chunks_submitted += 1
                                
                            elif msg_type == 'end':
                                print(f"📖 文件读取完成，提交了 {chunks_submitted} 个chunk...")
                                break
                            elif msg_type == 'error':
                                error_count += 1
                                self.logger.error(f"读取错误: {data}")
                                
                        except queue.Empty:
                            # 检查是否还有处理任务在运行
                            if not read_thread.is_alive() and input_queue.empty():
                                break
                            continue
                            
                except KeyboardInterrupt:
                    self.logger.info("用户中断，正在清理...")
                
                # 等待所有任务完成并收集结果
                print(f"⏳ 等待 {len(futures)} 个处理任务完成...")
                for future in as_completed(futures):
                    try:
                        future.result()  # 这会将结果放入result_queue
                    except Exception as e:
                        self.logger.error(f"处理任务失败: {e}")
                        error_count += 1
                
                # 收集所有结果
                print("📊 收集处理结果...")
                while not result_queue.empty():
                    try:
                        msg_type, data = result_queue.get_nowait()
                        if msg_type == 'result':
                            chunk_result = data
                            unique_items.extend(chunk_result['unique_items'])
                            total_processed += chunk_result['processed_count']
                            total_skipped += chunk_result.get('skipped_count', 0)
                            
                            # 更新进度
                            pbar.update(100)  # 简单的进度更新
                            
                    except queue.Empty:
                        break
            
            # 写入最终数据
            if unique_items:
                print(f"💾 写入 {len(unique_items)} 条唯一数据...")
                with open(self.config.output_file, 'w', encoding='utf-8') as f:
                    for _, line in unique_items:
                        f.write(line + '\n')
        
        # 统计信息
        final_unique_count = len(unique_items) if unique_items else 0
        effective_processed = total_processed - total_skipped
        total_time = time.time() - start_time
        
        # 保存统计信息
        method_name = str(self.config.method.value) if hasattr(self.config.method, 'value') else str(self.config.method)
        stats = {
            'input_file': self.config.input_file,
            'output_file': self.config.output_file,
            'method': f'{method_name}_streaming',
            'mode': 'streaming',
            'total_processed': total_processed,
            'total_skipped': total_skipped,
            'missing_id_field': self.stats.get('missing_id_field', 0),
            'missing_embedding': self.stats.get('missing_embedding', 0),
            'unique_count': final_unique_count,
            'duplicate_count': effective_processed - final_unique_count,
            'dedup_ratio': (effective_processed - final_unique_count) / max(1, effective_processed),
            'processing_time': total_time,
            'processing_speed': total_processed / total_time if total_time > 0 else 0,
            'threshold': self.threshold,
            'top_k': self.top_k,
            'embedding_dim': self.embedding_dim,
            'faiss_final_count': self.faiss_index.ntotal if hasattr(self, 'faiss_index') else 0
        }
        
        # 输出统计结果
        print(f"\n" + "="*70)
        print(f"🎉 流式Embedding去重完成!")
        print(f"="*70)
        print(f"📊 处理统计:")
        print(f"  📥 总输入数据: {total_processed:,} 条")
        if total_skipped > 0:
            print(f"  ⏭️  跳过数据: {total_skipped:,} 条 (缺失embedding/ID)")
            print(f"  📊 有效处理: {effective_processed:,} 条")
        print(f"  ✅ 唯一数据: {final_unique_count:,} 条")
        print(f"  🗑️  重复数据: {effective_processed - final_unique_count:,} 条")
        print(f"  📈 去重效果: {stats['dedup_ratio']:.2%} 重复率")
        print(f"")
        if total_skipped > 0:
            print(f"🔍 跳过数据详情:")
            print(f"  ❌ 缺少ID字段: {self.stats.get('missing_id_field', 0):,} 条")
            print(f"  ❌ 缺少embedding: {self.stats.get('missing_embedding', 0):,} 条")
            print(f"")
        print(f"⏱️  性能统计:")
        print(f"  🕐 总处理时间: {total_time:.2f} 秒")
        print(f"  🚀 处理速度: {stats['processing_speed']:,.0f} 条/秒")
        print(f"  🎯 相似度阈值: {self.threshold}")
        print(f"  📊 FAISS索引大小: {stats['faiss_final_count']:,} 向量")
        print(f"")
        print(f"📁 文件信息:")
        print(f"  📂 输入文件: {self.config.input_file}")
        print(f"  💾 输出文件: {self.config.output_file}")
        print(f"  📄 日志目录: {self.log_dir}/")
        print(f"="*70)
        
        return [], [], stats 