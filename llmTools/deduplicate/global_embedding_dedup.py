#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
全局Embedding去重器

特点：
- 真正的全局去重：构建完整的相似度图
- 智能代表选择：在重复组中选择最优数据
- 内存优化：支持大规模数据的分批处理
- 跨块相似度检测：不受数据分块影响
"""

import os
import sys
import time
import json
import logging
import numpy as np
from typing import List, Dict, Tuple, Set, Any, Optional
from tqdm import tqdm
import psutil
import datetime
from pathlib import Path
from collections import defaultdict, deque
import heapq

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
    try:
        from base import BaseDeduplicator
        from config import DedupConfig
    except ImportError:
        # 如果没有基类，创建一个简单的基类
        class BaseDeduplicator:
            def __init__(self, config):
                self.config = config
                
        class DedupConfig:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)


class GlobalEmbeddingDeduplicator(BaseDeduplicator):
    """全局Embedding去重器"""
    
    def __init__(self, config: DedupConfig):
        """初始化全局Embedding去重器"""
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
        
        # 基本文件路径（从config或父类获取）
        self.input_file = config.input_file
        self.output_file = config.output_file
        
        # Embedding相关配置
        self.embeddings_file = getattr(config, 'embeddings_file', None)
        self.embeddings_files = getattr(config, 'embeddings_files', None)
        self.embeddings_format = getattr(config, 'embeddings_format', None)
        self.threshold = getattr(config, 'threshold', 0.95)
        self.top_k = getattr(config, 'top_k', 10)
        self.batch_size = getattr(config, 'batch_size', 10000)
        
        # GPU配置检查
        self.use_gpu = getattr(config, 'use_gpu', False) and FAISS_GPU_AVAILABLE
        self.gpu_device = getattr(config, 'gpu_device', 0)
        
        # 如果用户请求GPU但不可用，给出警告
        if getattr(config, 'use_gpu', False) and not FAISS_GPU_AVAILABLE:
            if FAISS_GPU_AVAILABLE is False:
                print("⚠️  GPU模式不可用（FAISS-GPU兼容性问题），自动使用CPU模式")
                print("💡 建议：pip uninstall faiss-gpu && pip install faiss-cpu")
            else:
                print("⚠️  GPU模式不可用，自动使用CPU模式")
            self.use_gpu = False
        
        # JSONL格式支持
        self.use_embedding_dict = False  # 默认不使用字典（NPY格式）
        
        # 检查embedding格式设置
        if hasattr(self.config, 'embeddings_format'):
            if hasattr(self.config.embeddings_format, 'value'):
                # 处理枚举类型
                format_value = self.config.embeddings_format.value
            else:
                # 处理字符串类型
                format_value = str(self.config.embeddings_format)
            
            if format_value.lower() == 'jsonl':
                self.use_embedding_dict = True
                print("📂 JSONL格式")
            else:
                print("📂 NPY格式")
        elif hasattr(self.config, 'embeddings_files') and self.config.embeddings_files:
            # 如果指定了多个文件，默认为JSONL格式
            self.use_embedding_dict = True
            print("📂 多文件JSONL格式")
        else:
            print("📂 NPY格式")
        
        # 检查文件
        if hasattr(config, 'embeddings_files') and config.embeddings_files:
            # 使用多个embedding文件
            for file_path in config.embeddings_files:
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"Embedding文件不存在: {file_path}")
        elif self.embeddings_file and not os.path.exists(self.embeddings_file):
            raise FileNotFoundError(f"Embedding文件不存在: {self.embeddings_file}")
        
        print(f"🌍 全局Embedding去重器 - 阈值模式")
        print(f"🎯 相似度阈值: {self.threshold}")
        print(f"📦 批处理大小: {self.batch_size}")
        print(f"🔍 Top-K检索: {self.top_k}")
        print(f"💡 专为千万级数据优化，提供实时进度反馈")
        
        # 显示embedding格式信息
        if hasattr(config, 'embeddings_files') and config.embeddings_files:
            print(f"📂 多文件JSONL格式: {len(config.embeddings_files)} 个文件")
        elif hasattr(config, 'embeddings_format') and hasattr(config.embeddings_format, 'value'):
            format_value = config.embeddings_format.value if hasattr(config.embeddings_format, 'value') else str(config.embeddings_format)
            if format_value == 'jsonl':
                print(f"📂 单文件JSONL格式")
            else:
                print(f"📂 NPY格式")
        else:
            print(f"📂 NPY格式（默认）")
    
    def _extract_content(self, data: dict) -> str:
        """提取用于去重的内容"""
        content_parts = []
        for key in self.content_keys:
            if key in data:
                content_parts.append(str(data[key]))
        return " ".join(content_parts)
    
    def load_data_and_embeddings(self) -> Tuple[List[Dict], np.ndarray]:
        """加载数据和embeddings，严格按ID匹配，支持embedding缺失的情况"""
        print("🔄 加载数据和embeddings（基于ID严格匹配）...")
        
        # 步骤1：先加载embedding文件，获取所有有效的ID
        print("📂 第一步：加载embedding文件...")
        emb_start_time = time.time()
        
        embeddings_array = None
        if self.use_embedding_dict:
            # JSONL格式：加载完整的ID-embedding映射
            if hasattr(self.config, 'embeddings_files') and self.config.embeddings_files:
                embeddings_array = self._load_embeddings_from_multiple_jsonl_files()
            else:
                embeddings_array = self._load_embeddings_from_jsonl()
            
            available_ids = set(self.embeddings_dict.keys())
            print(f"✅ 找到 {len(available_ids):,} 个有效的embedding ID")
        else:
            # NPY格式：需要依赖行号对应关系，不适用于此场景
            embeddings_array = self._load_embeddings_from_npy()
            self.logger.warning("NPY格式无法进行ID匹配，假设按行号对应，请确保数据文件和embedding文件顺序一致")
            available_ids = None
        
        embedding_load_time = time.time() - emb_start_time
        print(f"✅ Embedding加载完成: 耗时 {embedding_load_time:.1f}秒")
        
        # 步骤2：加载数据文件，只保留有对应embedding的数据
        print("📖 第二步：加载数据文件并进行ID匹配...")
        
        data_file_size = os.path.getsize(self.input_file)
        print(f"📁 数据文件大小: {data_file_size / (1024**3):.2f} GB ({data_file_size:,} 字节)")
        
        if self.use_embedding_dict:
            matched_data, matched_embeddings = self._load_data_with_embedding_matching(available_ids)
        else:
            # NPY格式的处理
            matched_data, matched_embeddings = self._load_data_for_npy_format(embeddings_array)
        
        print(f"\n📊 ID匹配结果:")
        print(f"  🎯 Embedding文件中的ID数: {len(available_ids) if available_ids else len(embeddings_array):,}")
        print(f"  ✅ 成功匹配的数据: {len(matched_data):,} 条")
        print(f"  📐 最终embedding矩阵: {matched_embeddings.shape}")
        
        if len(matched_data) == 0:
            raise ValueError("没有找到任何可匹配的数据，请检查数据文件和embedding文件的ID字段是否正确")
        
        return matched_data, matched_embeddings
    
    def _load_embeddings_from_npy(self) -> np.ndarray:
        """从NPY文件加载embeddings"""
        # 获取embedding文件大小
        emb_file_size = os.path.getsize(self.embeddings_file)
        print(f"📁 Embedding文件大小: {emb_file_size / (1024**3):.2f} GB ({emb_file_size:,} 字节)")
        
        embeddings = np.load(self.embeddings_file)
        print(f"📊 原始embedding形状: {embeddings.shape}")
        
        # 归一化embeddings
        print("🔧 归一化embeddings...")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        embeddings = embeddings / norms
        
        return embeddings
    
    def _load_embeddings_from_jsonl(self) -> np.ndarray:
        """从单个JSONL文件加载embeddings"""
        embedding_id_key = getattr(self.config, 'embedding_id_key', 'prompt_id')
        embedding_vector_key = getattr(self.config, 'embedding_vector_key', 'embedding')
        
        print(f"📂 从JSONL文件加载embeddings: {self.embeddings_file}")
        print(f"🔑 ID字段: {embedding_id_key}, 向量字段: {embedding_vector_key}")
        
        # 获取embedding文件大小
        emb_file_size = os.path.getsize(self.embeddings_file)
        print(f"📁 Embedding文件大小: {emb_file_size / (1024**3):.2f} GB ({emb_file_size:,} 字节)")
        
        embeddings_dict = {}
        embeddings_list = []
        
        with open(self.embeddings_file, 'r', encoding='utf-8') as f:
            with tqdm(desc="📖 加载JSONL embeddings", unit="行") as pbar:
                for line_idx, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        data = json_loads(line)
                        
                        if embedding_id_key not in data or embedding_vector_key not in data:
                            self.logger.warning(f"行 {line_idx} 缺少必需字段: {embedding_id_key}, {embedding_vector_key}")
                            continue
                        
                        embedding_id = str(data[embedding_id_key])
                        embedding = data[embedding_vector_key]
                        
                        # 转换为numpy数组
                        if isinstance(embedding, list):
                            embedding = np.array(embedding, dtype=np.float32)
                        elif isinstance(embedding, np.ndarray):
                            embedding = embedding.astype(np.float32)
                        else:
                            self.logger.warning(f"行 {line_idx} 向量格式不支持: {type(embedding)}")
                            continue
                        
                        # 归一化向量
                        norm = np.linalg.norm(embedding)
                        if norm > 0:
                            embedding = embedding / norm
                            embeddings_dict[embedding_id] = embedding
                            embeddings_list.append(embedding)
                        
                        pbar.update(1)
                        
                    except Exception as e:
                        self.logger.warning(f"行 {line_idx} 解析失败: {e}")
                        continue
        
        if not embeddings_list:
            raise ValueError("JSONL文件中没有找到有效的embeddings")
        
        embeddings_array = np.array(embeddings_list, dtype=np.float32)
        print(f"✅ JSONL embeddings加载完成: {embeddings_array.shape}")
        
        # 保存ID映射用于后续查找
        self.embeddings_dict = embeddings_dict
        self.use_embedding_dict = True
        
        return embeddings_array
    
    def _load_embeddings_from_multiple_jsonl_files(self) -> np.ndarray:
        """从多个JSONL文件加载embeddings"""
        embedding_id_key = getattr(self.config, 'embedding_id_key', 'prompt_id')
        embedding_vector_key = getattr(self.config, 'embedding_vector_key', 'embedding')
        
        print(f"📂 从多个JSONL文件加载embeddings:")
        for f in self.config.embeddings_files:
            print(f"  - {f}")
        print(f"🔑 ID字段: {embedding_id_key}, 向量字段: {embedding_vector_key}")
        
        embeddings_dict = {}
        embeddings_list = []
        total_processed = 0
        
        for file_path in self.config.embeddings_files:
            if not os.path.exists(file_path):
                self.logger.error(f"文件不存在: {file_path}")
                continue
            
            file_size = os.path.getsize(file_path)
            print(f"📁 处理文件: {file_path} ({file_size / (1024**2):.1f} MB)")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                with tqdm(desc=f"📖 {os.path.basename(file_path)}", unit="行") as pbar:
                    for line_idx, line in enumerate(f):
                        line = line.strip()
                        if not line:
                            continue
                        
                        try:
                            data = json_loads(line)
                            
                            if embedding_id_key not in data or embedding_vector_key not in data:
                                continue
                            
                            embedding_id = str(data[embedding_id_key])
                            embedding = data[embedding_vector_key]
                            
                            # 检查是否重复ID
                            if embedding_id in embeddings_dict:
                                self.logger.warning(f"重复的embedding ID: {embedding_id}，跳过")
                                continue
                            
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
                                embeddings_dict[embedding_id] = embedding
                                embeddings_list.append(embedding)
                                total_processed += 1
                            
                            pbar.update(1)
                            
                        except Exception as e:
                            continue
        
        if not embeddings_list:
            raise ValueError("多个JSONL文件中没有找到有效的embeddings")
        
        embeddings_array = np.array(embeddings_list, dtype=np.float32)
        print(f"✅ 多文件JSONL embeddings加载完成: {embeddings_array.shape} (总共 {total_processed:,} 个向量)")
        
        # 保存ID映射用于后续查找
        self.embeddings_dict = embeddings_dict
        self.use_embedding_dict = True
        
        return embeddings_array
    
    def _get_embedding_by_id_or_index(self, data: Dict, line_idx: int, embeddings: np.ndarray) -> Optional[np.ndarray]:
        """根据ID或索引获取embedding向量"""
        if self.use_embedding_dict:
            # JSONL格式：使用ID匹配
            prompt_id_key = getattr(self.config, 'prompt_id_key', 'prompt_id')
            if prompt_id_key not in data:
                self.logger.warning(f"数据中缺少prompt_id字段 '{prompt_id_key}': {data.keys()}")
                return None
            
            prompt_id = str(data[prompt_id_key])
            if prompt_id not in self.embeddings_dict:
                self.logger.warning(f"embedding文件中未找到prompt_id: {prompt_id}")
                return None
            
            return self.embeddings_dict[prompt_id]
        else:
            # NPY格式：使用行索引
            if line_idx >= len(embeddings):
                self.logger.warning(f"行索引 {line_idx} 超出embedding范围 {len(embeddings)}")
                return None
            
            return embeddings[line_idx]
    
    def global_deduplicate(self) -> Tuple[List[Dict], Dict[str, Any]]:
        """执行全局去重 - 优化的阈值模式（无相似度图）"""
        start_time = time.time()
        
        print("🚀 启动全局Embedding去重 - 阈值模式")
        print("💡 此模式专为千万级数据优化，提供实时进度反馈")
        
        # 步骤1：加载数据
        data_list, embeddings = self.load_data_and_embeddings()
        
        print(f"📊 开始去重处理...")
        print(f"  📁 数据量: {len(data_list):,} 条")
        print(f"  📐 向量维度: {embeddings.shape[1]}")
        print(f"  🎯 相似度阈值: {self.threshold}")
        print(f"  📦 批处理大小: {self.batch_size:,}")
        
        # 步骤2：创建FAISS索引用于快速检索
        print("📚 构建FAISS检索索引...")
        index_start_time = time.time()
        
        index = faiss.IndexFlatIP(embeddings.shape[1])
        if self.use_gpu and faiss.get_num_gpus() > 0:
            index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index)
            print("🚀 使用GPU加速检索")
        
        # 一次性添加所有向量到索引
        index.add(embeddings.astype(np.float32))
        index_time = time.time() - index_start_time
        print(f"✅ FAISS索引构建完成，耗时: {index_time:.1f}秒")
        
        # 步骤3：批量去重处理
        print("🔍 开始批量去重检测...")
        
        kept_indices = set()
        removed_indices = set()
        duplicate_records = []
        processed_count = 0
        
        # 动态调整批处理大小以确保及时的进度反馈
        # 大数据集使用较小的批次以提供更频繁的更新
        if len(embeddings) > 100000:  # 超过10万条数据
            effective_batch_size = min(self.batch_size, 5000)  # 最多5000条一批
            print(f"📦 大数据集优化：批处理大小调整为 {effective_batch_size:,} （原设置: {self.batch_size:,}）")
        elif len(embeddings) > 10000:   # 1万到10万条数据
            effective_batch_size = min(self.batch_size, 2000)  # 最多2000条一批
        else:
            effective_batch_size = min(self.batch_size, 1000)  # 小数据集1000条一批
        
        total_batches = (len(embeddings) + effective_batch_size - 1) // effective_batch_size
        dedup_start_time = time.time()
        
        print(f"📊 批处理配置:")
        print(f"  🎯 总数据量: {len(embeddings):,}")
        print(f"  📦 批处理大小: {effective_batch_size:,}")
        print(f"  📈 总批次数: {total_batches:,}")
        print(f"  ⏱️  预计每批次耗时: {0.1 * effective_batch_size / 1000:.1f}秒")
        
        # 使用详细进度条
        with tqdm(total=len(embeddings), desc="🔍 全局去重检测", 
                 bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} '
                           '[耗时:{elapsed}, 剩余:{remaining}, {rate_fmt}]',
                 mininterval=0.1) as pbar:  # 更频繁的更新间隔
            
            for batch_idx in range(total_batches):
                batch_start_time = time.time()
                start_idx = batch_idx * effective_batch_size
                end_idx = min(start_idx + effective_batch_size, len(embeddings))
                
                # 显示当前批次信息
                if batch_idx % 5 == 0 or batch_idx < 10:  # 前10个批次或每5个批次
                    print(f"📦 处理批次 {batch_idx + 1}/{total_batches} (索引 {start_idx:,}-{end_idx-1:,})")
                
                # 处理当前批次
                batch_embeddings = embeddings[start_idx:end_idx].astype(np.float32)
                
                # 对每个向量进行相似度检索
                similarities, indices = index.search(batch_embeddings, self.top_k + 1)  # +1包含自己
                
                batch_kept = 0
                batch_removed = 0
                
                # 在批次内部添加进度跟踪
                batch_size_current = end_idx - start_idx
                progress_updated = 0  # 跟踪已更新的进度
                
                for i, (sims, idxs) in enumerate(zip(similarities, indices)):
                    global_idx = start_idx + i
                    
                    # 跳过已经被标记为重复的数据
                    if global_idx in removed_indices:
                        continue
                    
                    # 检查是否与之前的数据重复
                    is_duplicate = False
                    best_match_idx = None
                    best_similarity = 0.0
                    
                    for sim, candidate_idx in zip(sims, idxs):
                        # 跳过自己
                        if candidate_idx == global_idx:
                            continue
                        
                        # 验证相似度范围
                        if sim > 1.0001 or sim < -1.0001:
                            self.logger.warning(f"无效相似度: {sim:.6f}，限制到有效范围")
                            sim = max(-1.0, min(1.0, sim))
                        
                        # 只与之前处理的数据比较（保证一致性）
                        if candidate_idx < global_idx and candidate_idx not in removed_indices:
                            if sim >= self.threshold:
                                is_duplicate = True
                                best_match_idx = candidate_idx
                                best_similarity = sim
                                break
                            elif sim >= self.threshold * 0.8:  # 记录接近阈值的样例
                                self.log_threshold_example(
                                    data_list[global_idx], 
                                    data_list[candidate_idx], 
                                    sim, 
                                    self.threshold,
                                    {
                                        'embedding_similarity': float(sim),
                                        'threshold': self.threshold,
                                        'distance_from_threshold': float(self.threshold - sim),
                                        'vector_dimension': embeddings.shape[1]
                                    }
                                )
                    
                    if is_duplicate:
                        # 标记为重复
                        removed_indices.add(global_idx)
                        batch_removed += 1
                        
                        # 记录重复信息
                        duplicate_records.append({
                            'removed_data': data_list[global_idx],
                            'kept_data': data_list[best_match_idx],
                            'similarity': float(best_similarity)
                        })
                    else:
                        # 保留为唯一数据
                        kept_indices.add(global_idx)
                        batch_kept += 1
                    
                    # 每处理200个项目或每10%进度更新一次
                    if (i + 1) % 200 == 0 or (i + 1) % max(1, batch_size_current // 10) == 0:
                        # 计算需要更新的进度
                        progress_to_update = (i + 1) - progress_updated
                        if progress_to_update > 0:
                            pbar.update(progress_to_update)
                            progress_updated = i + 1
                            
                            # 更新进度条显示
                            current_dedup_ratio = len(removed_indices) / (start_idx + i + 1) if (start_idx + i + 1) > 0 else 0
                            pbar.set_postfix({
                                '批次': f'{batch_idx + 1}/{total_batches}',
                                '当前': f'{i+1}/{batch_size_current}',
                                '保留': f'{len(kept_indices):,}',
                                '去重': f'{len(removed_indices):,}',
                                '去重率': f'{current_dedup_ratio:.1%}'
                            })
                
                # 批次结束，更新剩余进度
                remaining_progress = batch_size_current - progress_updated
                if remaining_progress > 0:
                    pbar.update(remaining_progress)
                
                # 更新统计
                processed_count = end_idx
                batch_time = time.time() - batch_start_time
                
                # 批次完成时的详细信息
                elapsed_time = time.time() - dedup_start_time
                avg_batch_time = elapsed_time / (batch_idx + 1)
                remaining_batches = total_batches - (batch_idx + 1)
                estimated_remaining = avg_batch_time * remaining_batches
                
                current_dedup_ratio = len(removed_indices) / processed_count if processed_count > 0 else 0
                processing_speed = processed_count / elapsed_time if elapsed_time > 0 else 0
                
                # 更新最终进度条显示
                pbar.set_postfix({
                    '批次': f'{batch_idx + 1}/{total_batches}',
                    '保留': f'{len(kept_indices):,}',
                    '去重': f'{len(removed_indices):,}',
                    '去重率': f'{current_dedup_ratio:.1%}',
                    '速度': f'{processing_speed:,.0f}/s',
                    '剩余': f'{estimated_remaining/60:.1f}分' if estimated_remaining > 60 else f'{estimated_remaining:.0f}s'
                })
                
                # 详细日志（每5个批次或耗时较长时）
                if (batch_idx + 1) % 5 == 0 or batch_time > 5:
                    self.logger.info(f"批次 {batch_idx + 1}/{total_batches}: "
                                   f"耗时 {batch_time:.1f}s, "
                                   f"本批保留 {batch_kept}, 去重 {batch_removed}, "
                                   f"总计保留 {len(kept_indices):,}, 去重 {len(removed_indices):,}, "
                                   f"去重率 {current_dedup_ratio:.2%}, "
                                   f"处理速度 {processing_speed:,.0f}条/秒")
                
                # 性能警告
                if batch_time > 30:  # 单个批次超过30秒
                    print(f"⚠️  批次 {batch_idx + 1} 耗时较长: {batch_time:.1f}秒")
                    print(f"💡 建议：考虑减小 batch_size 或使用 --use_gpu 加速")
        
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
        dedup_time = time.time() - dedup_start_time
        
        method_name = str(self.config.method.value) if hasattr(self.config.method, 'value') else str(self.config.method)
        stats = {
            'total_items': len(data_list),
            'unique_items': len(unique_data),
            'duplicate_items': len(removed_indices),
            'dedup_ratio': len(removed_indices) / len(data_list) if data_list else 0,
            'total_time': total_time,
            'dedup_time': dedup_time,
            'index_time': index_time,
            'processing_speed': len(data_list) / dedup_time if dedup_time > 0 else 0,
            'embedding_dim': embeddings.shape[1],
            'threshold': self.threshold,
            'batch_size': self.batch_size,
            'method': f'{method_name}_global',
            'mode': 'global'
        }
        
        # 保存日志
        self._save_duplicate_records(duplicate_records)
        self._save_threshold_examples()  # 保存阈值样例
        self._save_performance_stats(stats)
        
        # 输出详细统计
        print(f"\n" + "="*70)
        print(f"🎉 全局Embedding去重完成!")
        print(f"="*70)
        print(f"📊 去重统计:")
        print(f"  📥 输入数据: {stats['total_items']:,} 条")
        print(f"  ✅ 保留数据: {stats['unique_items']:,} 条")
        print(f"  🗑️  删除数据: {stats['duplicate_items']:,} 条")
        print(f"  📈 去重效果: {stats['dedup_ratio']:.2%} 重复率")
        print(f"")
        print(f"⏱️  性能统计:")
        print(f"  🕐 总处理时间: {total_time:.1f} 秒")
        print(f"  📚 索引构建: {index_time:.1f} 秒")
        print(f"  🔍 去重检测: {dedup_time:.1f} 秒")
        print(f"  🚀 处理速度: {stats['processing_speed']:,.0f} 条/秒")
        print(f"  🎯 相似度阈值: {self.threshold}")
        print(f"  📦 批处理大小: {self.batch_size:,}")
        print(f"")
        print(f"📁 文件信息:")
        print(f"  📂 输入文件: {self.input_file}")
        print(f"  💾 输出文件: {self.output_file}")
        print(f"  📄 日志目录: {self.log_dir}/")
        print(f"="*70)
        
        # 性能建议
        if total_time > 1800:  # 超过30分钟
            print(f"💡 性能优化建议:")
            if not self.use_gpu:
                print(f"  - 强烈建议使用 --use_gpu 加速")
            if self.batch_size < 20000:
                print(f"  - 考虑增加 --batch_size 到 {min(50000, self.batch_size * 2):,}")
            if self.top_k > 10:
                print(f"  - 考虑降低 --top_k 到 {max(5, self.top_k // 2)}")
            print(f"  - 考虑使用 streaming 模式获得更好的内存效率")
        
        return unique_data, stats
    
    def _load_data_with_embedding_matching(self, available_ids: set) -> Tuple[List[Dict], np.ndarray]:
        """加载数据文件并与embedding ID进行严格匹配"""
        prompt_id_key = getattr(self.config, 'prompt_id_key', 'prompt_id')
        print(f"🔑 使用ID字段: {prompt_id_key}")
        
        matched_data = []
        matched_embeddings = []
        matched_ids = []
        
        # 统计信息
        total_data_count = 0
        missing_id_field_count = 0
        missing_embedding_count = 0
        embedding_has_no_data_count = 0
        
        data_file_size = os.path.getsize(self.input_file)
        bytes_read = 0
        start_time = time.time()
        
        with open(self.input_file, 'r', encoding='utf-8') as f:
            with tqdm(total=data_file_size, desc="📖 匹配数据和embedding", unit="B", unit_scale=True,
                     bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
                
                for line_idx, line in enumerate(f):
                    line = line.strip()
                    line_bytes = len(line.encode('utf-8')) + 1
                    bytes_read += line_bytes
                    pbar.update(line_bytes)
                    
                    if not line:
                        continue
                    
                    try:
                        data = json_loads(line)
                        total_data_count += 1
                        
                        # 检查是否有ID字段
                        if prompt_id_key not in data:
                            missing_id_field_count += 1
                            if missing_id_field_count <= 10:  # 只记录前10个警告
                                self.logger.warning(f"数据行 {line_idx} 缺少ID字段 '{prompt_id_key}': {list(data.keys())[:5]}")
                            continue
                        
                        prompt_id = str(data[prompt_id_key])
                        
                        # 检查embedding是否存在
                        if prompt_id not in available_ids:
                            missing_embedding_count += 1
                            # 不记录每个缺失的embedding，太多了
                            continue
                        
                        # 匹配成功，添加到结果
                        data['__line_idx__'] = line_idx
                        data['__embedding_idx__'] = len(matched_data)  # 在embedding数组中的索引
                        matched_data.append(data)
                        matched_embeddings.append(self.embeddings_dict[prompt_id])
                        matched_ids.append(prompt_id)
                        
                    except Exception as e:
                        self.logger.warning(f"行 {line_idx} 解析失败: {e}")
                    
                    # 定期更新进度信息
                    if line_idx % 5000 == 0 and line_idx > 0:
                        elapsed = time.time() - start_time
                        lines_per_sec = line_idx / elapsed
                        match_rate = len(matched_data) / total_data_count if total_data_count > 0 else 0
                        pbar.set_postfix({
                            "匹配数": f"{len(matched_data):,}",
                            "匹配率": f"{match_rate:.1%}",
                            "行/s": f"{lines_per_sec:,.0f}"
                        })
        
        # 检查embedding中是否有数据文件中不存在的ID
        data_ids = set(matched_ids)
        embedding_only_ids = available_ids - data_ids
        embedding_has_no_data_count = len(embedding_only_ids)
        
        # 输出详细的匹配统计
        print(f"\n📈 ID匹配详细统计:")
        print(f"  📄 数据文件总行数: {total_data_count:,}")
        print(f"  🎯 embedding文件ID数: {len(available_ids):,}")
        print(f"  ✅ 成功匹配数据: {len(matched_data):,}")
        print(f"  ❌ 缺少ID字段: {missing_id_field_count:,} 条")
        print(f"  ❌ 缺少embedding: {missing_embedding_count:,} 条")
        print(f"  ⚠️  embedding无对应数据: {embedding_has_no_data_count:,} 个ID")
        
        if missing_id_field_count > 0:
            self.logger.warning(f"发现 {missing_id_field_count:,} 条数据缺少ID字段 '{prompt_id_key}'")
        
        if missing_embedding_count > 0:
            match_rate = len(matched_data) / (len(matched_data) + missing_embedding_count) * 100
            self.logger.warning(f"发现 {missing_embedding_count:,} 条数据在embedding文件中缺失（匹配率: {match_rate:.1f}%）")
        
        if embedding_has_no_data_count > 0:
            self.logger.warning(f"发现 {embedding_has_no_data_count:,} 个embedding ID在数据文件中找不到对应数据")
            if embedding_has_no_data_count <= 20:  # 少量时显示具体ID
                self.logger.warning(f"缺失数据的embedding ID示例: {list(embedding_only_ids)[:20]}")
        
        if len(matched_data) == 0:
            raise ValueError(f"没有找到任何匹配的数据！请检查：\n"
                           f"  1. 数据文件中是否有 '{prompt_id_key}' 字段\n"
                           f"  2. embedding文件中是否有对应的ID\n"
                           f"  3. ID格式是否一致（字符串/数字）")
        
        # 转换为numpy数组
        matched_embeddings_array = np.array(matched_embeddings, dtype=np.float32)
        
        return matched_data, matched_embeddings_array
    
    def _load_data_for_npy_format(self, embeddings_array: np.ndarray) -> Tuple[List[Dict], np.ndarray]:
        """为NPY格式加载数据（按行索引对应），严格验证数量匹配"""
        print("⚠️  NPY格式：按行号严格一一对应，验证数量匹配...")
        
        data_file_size = os.path.getsize(self.input_file)
        print(f"📁 数据文件大小: {data_file_size / (1024**3):.2f} GB")
        print(f"🎯 Embedding数量: {len(embeddings_array):,}")
        
        # 第一步：快速统计数据文件总行数
        print("📊 统计数据文件行数...")
        total_data_lines = 0
        with open(self.input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    total_data_lines += 1
        
        print(f"📄 数据文件总行数: {total_data_lines:,}")
        print(f"🎯 Embedding总数量: {len(embeddings_array):,}")
        
        # 严格验证数量匹配
        if total_data_lines != len(embeddings_array):
            error_msg = (
                f"❌ NPY格式数量不匹配错误！\n"
                f"   📄 数据文件行数: {total_data_lines:,}\n"
                f"   🎯 Embedding数量: {len(embeddings_array):,}\n"
                f"   📐 差异: {abs(total_data_lines - len(embeddings_array)):,} 条\n\n"
                f"💡 NPY格式要求数据和embedding严格按行索引一一对应！\n"
                f"   请确保：\n"
                f"   - 数据文件第i行 ↔ embedding文件第i个向量\n"
                f"   - 两个文件的数量完全相等\n\n"
                f"🔧 解决方案：\n"
                f"   1. 重新生成embedding确保数量匹配\n"
                f"   2. 或使用JSONL格式支持ID匹配: --embeddings_format jsonl"
            )
            self.logger.error(error_msg)
            raise ValueError(f"NPY格式数量不匹配：数据{total_data_lines:,}行 vs embedding{len(embeddings_array):,}个")
        
        print("✅ 数量验证通过，开始加载数据...")
        
        # 第二步：加载所有数据
        matched_data = []
        matched_embeddings = []
        
        bytes_read = 0
        start_time = time.time()
        
        with open(self.input_file, 'r', encoding='utf-8') as f:
            with tqdm(total=data_file_size, desc="📖 加载NPY格式数据", unit="B", unit_scale=True,
                     bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
                
                for line_idx, line in enumerate(f):
                    line = line.strip()
                    line_bytes = len(line.encode('utf-8')) + 1
                    bytes_read += line_bytes
                    pbar.update(line_bytes)
                    
                    if not line:
                        continue
                    
                    try:
                        data = json_loads(line)
                        data['__line_idx__'] = line_idx
                        data['__embedding_idx__'] = line_idx  # NPY格式中embedding索引等于行索引
                        
                        matched_data.append(data)
                        matched_embeddings.append(embeddings_array[line_idx])
                        
                    except Exception as e:
                        error_msg = f"NPY格式数据解析失败，行 {line_idx + 1}: {e}"
                        self.logger.error(error_msg)
                        raise ValueError(error_msg)
                    
                    # 定期更新进度信息
                    if line_idx % 10000 == 0 and line_idx > 0:
                        elapsed = time.time() - start_time
                        lines_per_sec = line_idx / elapsed
                        pbar.set_postfix({
                            "已加载": f"{line_idx + 1:,}",
                            "行/s": f"{lines_per_sec:,.0f}"
                        })
        
        # 转换为numpy数组
        matched_embeddings_array = np.array(matched_embeddings, dtype=np.float32)
        
        # 最终验证
        assert len(matched_data) == len(embeddings_array), f"内部错误：加载的数据数量不匹配"
        
        # 输出加载结果
        print(f"\n📊 NPY格式加载成功:")
        print(f"  ✅ 数据和embedding数量: {len(matched_data):,} 条")
        print(f"  📐 向量维度: {matched_embeddings_array.shape[1]}")
        print(f"  💾 内存占用: ~{matched_embeddings_array.nbytes / (1024**2):.1f} MB")
        
        return matched_data, matched_embeddings_array
    
    def _save_threshold_examples(self):
        """保存阈值边界样例"""
        if hasattr(self, 'threshold_examples') and self.threshold_examples:
            with open(self.log_files['threshold_examples'], 'w', encoding='utf-8') as f:
                for example in self.threshold_examples:
                    f.write(json.dumps(example, ensure_ascii=False) + '\n')
            print(f"📄 阈值边界样例已保存: {self.log_files['threshold_examples']}")
    
    def _save_duplicate_records(self, records: List[Dict]):
        """保存重复记录"""
        with open(self.log_files['duplicates'], 'w', encoding='utf-8') as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        print(f"📄 重复记录已保存: {self.log_files['duplicates']}")
    
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
                # 移除内部标记
                if '__line_idx__' in data:
                    del data['__line_idx__']
                # 使用field_order排序的JSON输出
                f.write(self._format_json_output(data) + '\n')
        
        print(f"✅ 去重结果已保存: {self.output_file}")
        return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="全局Embedding去重")
    parser.add_argument("--input_file", required=True, help="输入JSONL文件")
    parser.add_argument("--output_file", required=True, help="输出文件")
    parser.add_argument("--embeddings_file", help="NPY格式embeddings文件")
    parser.add_argument("--embeddings_files", nargs="+", help="多个JSONL格式embeddings文件")
    parser.add_argument("--embeddings_format", choices=["npy", "jsonl"], default="npy")
    parser.add_argument("--threshold", type=float, default=0.95, help="相似度阈值")
    parser.add_argument("--batch_size", type=int, default=10000, help="批处理大小")
    parser.add_argument("--use_gpu", action="store_true", help="使用GPU加速")
    
    args = parser.parse_args()
    
    from .dep_code.config import DedupConfig, EmbeddingFileFormat
    
    config = DedupConfig(
        method="embedding",
        mode="global",
        input_file=args.input_file,
        output_file=args.output_file,
        embeddings_file=args.embeddings_file,
        embeddings_files=args.embeddings_files,
        embeddings_format=EmbeddingFileFormat.NPY if args.embeddings_format == "npy" else EmbeddingFileFormat.JSONL,
        threshold=args.threshold,
        batch_size=args.batch_size,
        use_gpu=args.use_gpu
    )
    
    deduplicator = GlobalEmbeddingDeduplicator(config)
    deduplicator.deduplicate() 