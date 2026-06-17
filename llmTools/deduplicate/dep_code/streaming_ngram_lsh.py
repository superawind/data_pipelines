#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
流式N-gram LSH去重器 - 零延迟启动，实时进度显示

核心特性：
1. 零延迟启动 - 立即开始处理
2. 实时进度显示 - 不再停滞
3. 边读边处理 - 流式架构
4. 高效LSH算法 - 智能相似度检测
5. 心跳机制 - 防止假死
"""

import os
import sys
import time
import threading
import queue
import hashlib
from typing import Dict, List, Tuple, Any, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from collections import deque, defaultdict
import multiprocessing as mp
from tqdm import tqdm
import signal

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

try:
    from datasketch import MinHash, MinHashLSH
    MINHASH_AVAILABLE = True
except ImportError:
    MINHASH_AVAILABLE = False

try:
    from .base import BaseDeduplicator
    from .config import DedupConfig
except ImportError:
    from base import BaseDeduplicator
    from config import DedupConfig


class StreamingNgramLshDeduplicator(BaseDeduplicator):
    """流式N-gram LSH去重器 - 零延迟启动，实时进度显示"""
    
    def __init__(self, config: DedupConfig):
        super().__init__(config)
        
        # 检查依赖库
        if not MINHASH_AVAILABLE:
            raise ImportError("请安装datasketch: pip install datasketch")
        
        # 流式处理配置
        self.num_workers = getattr(config, 'num_workers', None) or min(8, os.cpu_count() or 4)
        chunk_size_mb = getattr(config, 'chunk_size_mb', 16)
        buffer_size_mb = getattr(config, 'buffer_size_mb', 2)
        self.chunk_size = chunk_size_mb * 1024 * 1024  # 用户配置的块大小
        self.buffer_size = buffer_size_mb * 1024 * 1024   # 用户配置的读取缓冲区
        self.write_buffer_size = buffer_size_mb * 2 * 1024 * 1024  # 写入缓冲区是读取缓冲区的2倍
        
        # 队列大小控制
        self.queue_maxsize = getattr(config, 'queue_maxsize', 8)   # 用户配置的队列大小
        self.result_queue_size = min(32, self.queue_maxsize * 4)  # 结果队列大小基于输入队列
        
        # LSH配置
        self.ngram_size = getattr(config, 'ngram_size', 10)
        self.num_permutations = min(getattr(config, 'num_permutations', 128), 128)
        self.jaccard_threshold = getattr(config, 'jaccard_threshold', 0.85)
        
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
            'lsh_comparisons': 0,
            'similarity_checks': 0
        }
        
        # 控制标志
        self.stop_flag = threading.Event()
        self.debug_mode = True
        
        # LSH索引（全局共享）
        self.global_lsh = MinHashLSH(
            threshold=self.jaccard_threshold,
            num_perm=self.num_permutations,
            storage_config={'type': 'dict'}
        )
        self.lsh_lock = threading.RLock()
        
        # 内容到索引映射
        self.content_to_id = {}
        self.id_to_content = {}
        self.next_id = 0
        
        # 计算内存使用
        total_buffer_mb = (self.chunk_size * self.queue_maxsize) // 1024 // 1024
        
        self.logger.info(f"🎯 流式LSH去重器初始化完成")
        self.logger.info(f"🔧 性能配置: {self.num_workers}个worker, {chunk_size_mb}MB块, {self.queue_maxsize}个队列, {buffer_size_mb}MB缓冲区")
        self.logger.info(f"💾 预计内存使用: {total_buffer_mb}MB (块缓存)")
        self.logger.info(f"🔍 LSH配置: {self.num_permutations}排列, {self.ngram_size}-gram, 阈值{self.jaccard_threshold}")

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

    def _generate_ngrams(self, text: str) -> Set[str]:
        """生成N-gram集合"""
        if len(text) < self.ngram_size:
            return {text}
        
        ngrams = set()
        for i in range(len(text) - self.ngram_size + 1):
            ngrams.add(text[i:i + self.ngram_size])
        return ngrams

    def _create_minhash(self, text: str) -> MinHash:
        """创建MinHash签名"""
        minhash = MinHash(num_perm=self.num_permutations)
        ngrams = self._generate_ngrams(text)
        
        for ngram in ngrams:
            minhash.update(ngram.encode('utf-8'))
        
        return minhash

    def _check_similarity_with_logging(self, content: str, data: dict) -> dict:
        """检查内容相似度并记录详细分析信息"""
        if not content.strip():
            return {'is_duplicate': True, 'similarity': 1.0, 'matched_content': '', 'matched_data': None}
            
        minhash = self._create_minhash(content)
        max_similarity = 0.0
        best_match_content = ""
        best_match_data = None
        
        with self.lsh_lock:
            # 查询相似项
            similar_items = self.global_lsh.query(minhash)
            self.stats['lsh_comparisons'] += 1
            
            if similar_items:
                # 发现相似项，进行精确相似度检查
                for similar_id in similar_items:
                    similar_content = self.id_to_content.get(similar_id, "")
                    
                    # 计算Jaccard相似度
                    ngrams1 = self._generate_ngrams(content)
                    ngrams2 = self._generate_ngrams(similar_content)
                    
                    intersection = len(ngrams1 & ngrams2)
                    union = len(ngrams1 | ngrams2)
                    
                    if union > 0:
                        jaccard_sim = intersection / union
                        self.stats['similarity_checks'] += 1
                        
                        # 跟踪最高相似度
                        if jaccard_sim > max_similarity:
                            max_similarity = jaccard_sim
                            best_match_content = similar_content
                            # 尝试从内容映射中获取完整数据
                            best_match_data = getattr(self, 'content_to_data', {}).get(similar_content, {
                                'content': similar_content,
                                'id': similar_id
                            })
                        
                        # 检查是否达到重复阈值
                        if jaccard_sim >= self.jaccard_threshold:
                            # 记录重复数据
                            if self.duplicate_count < 1000:
                                method_info = {
                                    'jaccard_similarity': jaccard_sim,
                                    'threshold': self.jaccard_threshold,
                                    'ngram_size': self.ngram_size,
                                    'num_permutations': self.num_permutations,
                                    'lsh_matches': len(similar_items)
                                }
                                
                                self.log_duplicate_record(
                                    removed_data=data,
                                    kept_data=best_match_data,
                                    similarity=jaccard_sim,
                                    method_specific_info=method_info
                                )
                            
                            return {
                                'is_duplicate': True, 
                                'similarity': jaccard_sim, 
                                'matched_content': similar_content,
                                'matched_data': best_match_data
                            }
            
            # 检查是否接近阈值（记录未去重但相似度较高的数据）
            if max_similarity > 0 and max_similarity >= self.jaccard_threshold * 0.8:  # 80%的阈值作为"接近"
                if self.threshold_example_count < 1000:
                    method_info = {
                        'jaccard_similarity': max_similarity,
                        'threshold': self.jaccard_threshold,
                        'ngram_size': self.ngram_size,
                        'num_permutations': self.num_permutations,
                        'lsh_matches': len(similar_items) if similar_items else 0
                    }
                    
                    self.log_threshold_example(
                        data1=data,
                        data2=best_match_data,
                        similarity=max_similarity,
                        threshold=self.jaccard_threshold,
                        method_specific_info=method_info
                    )
            
            # 不是重复，添加到LSH索引
            content_id = f"content_{self.next_id}"
            self.next_id += 1
            
            self.global_lsh.insert(content_id, minhash)
            self.content_to_id[content] = content_id
            self.id_to_content[content_id] = content
            
            # 保存数据映射用于日志记录
            if not hasattr(self, 'content_to_data'):
                self.content_to_data = {}
            self.content_to_data[content] = data
            
            return {
                'is_duplicate': False, 
                'similarity': max_similarity, 
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
        """流式处理数据块 - 添加详细的相似度分析和记录"""
        try:
            unique_items = []
            processed_count = 0
            
            for line_num, line in chunk_data:
                try:
                    # 解析JSON
                    if line.strip():
                        data = self.json_loads(line)
                        content = self._extract_content(data)
                        
                        if content:
                            # LSH去重检查，获取详细相似度信息
                            similarity_result = self._check_similarity_with_logging(content, data)
                            
                            if not similarity_result['is_duplicate']:
                                unique_items.append((line_num, line))
                            
                        processed_count += 1
                
                except Exception as e:
                    self.logger.warning(f"行 {line_num} 处理失败: {e}")
                    continue
            
            result_queue.put(('result', {
                'unique_items': unique_items,
                'processed_count': processed_count,
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
                desc="🚀 流式LSH处理",
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
        """执行流式LSH去重 - 零延迟启动，实时进度显示"""
        start_time = time.time()
        self.stats['start_time'] = start_time
        
        self.logger.info("🚀 启动流式LSH去重")
        self.logger.info(f"🔧 配置: {self.num_workers}个worker, {self.chunk_size//1024//1024}MB块")
        self.logger.info(f"🔍 LSH配置: {self.num_permutations}排列, {self.ngram_size}-gram, 阈值{self.jaccard_threshold}")
        self.logger.info("⚡ 立即开始处理，实时显示进度...")
        
        # 创建队列
        input_queue = queue.Queue(maxsize=self.queue_maxsize)
        result_queue = queue.Queue(maxsize=self.result_queue_size)
        
        # 启动文件读取线程
        read_thread = threading.Thread(
            target=self._read_file_streaming,
            args=(input_queue,),
            daemon=True
        )
        read_thread.start()
        
        # 启动结果写入线程
        write_thread = threading.Thread(
            target=self._write_results_streaming,
            args=(result_queue,),
            daemon=True
        )
        write_thread.start()
        
        # 启动进度监控线程
        progress_thread = threading.Thread(
            target=self._monitor_progress,
            daemon=True
        )
        progress_thread.start()
        
        # 处理数据流
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            
            while True:
                try:
                    msg_type, data = input_queue.get(timeout=10.0)
                    
                    if msg_type == 'chunk':
                        # 提交处理任务
                        future = executor.submit(
                            self._process_chunk_streaming,
                            data,
                            result_queue
                        )
                        futures.append(future)
                        self.stats['chunks_processed'] += 1
                        
                    elif msg_type == 'end':
                        self.logger.info("📖 文件读取完成，等待处理完成...")
                        break
                    elif msg_type == 'error':
                        self.logger.error(f"读取错误: {data}")
                        break
                        
                except queue.Empty:
                    self.logger.info("等待更多数据...")
                    continue
                except KeyboardInterrupt:
                    self.logger.info("用户中断，正在清理...")
                    self.stop_flag.set()
                    break
            
            # 等待所有处理任务完成
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    self.logger.error(f"处理任务失败: {e}")
        
        # 发送写入结束信号
        result_queue.put(('end', None))
        
        # 等待写入完成
        write_thread.join(timeout=30)
        
        # 停止进度监控
        self.stop_flag.set()
        progress_thread.join(timeout=5)
        
        total_time = time.time() - start_time
        
        # 计算最终统计
        method_name = str(self.config.method.value) if hasattr(self.config.method, 'value') else str(self.config.method)
        final_stats = {
            'total_lines': self.stats['total_processed'],
            'unique_lines': self.stats['unique_count'],
            'duplicate_lines': self.stats['total_processed'] - self.stats['unique_count'],
            'dedup_ratio': (self.stats['total_processed'] - self.stats['unique_count']) / max(self.stats['total_processed'], 1),
            'total_time': total_time,
            'processing_speed_mb_s': self.stats['processing_speed_mb_s'],
            'lines_per_second': self.stats['total_processed'] / total_time if total_time > 0 else 0,
            'lsh_comparisons': self.stats['lsh_comparisons'],
            'similarity_checks': self.stats['similarity_checks'],
            'method': f'{method_name}_streaming',
            'mode': 'streaming'
        }
        
        self.logger.info("🎉 流式LSH去重完成!")
        
        return [], [], final_stats 