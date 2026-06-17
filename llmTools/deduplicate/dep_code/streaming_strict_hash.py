#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
改进版流式极速严格哈希去重器

解决进度条停滞问题：
1. 减少全局锁竞争
2. 改进进度显示机制  
3. 添加详细调试信息
4. 优化队列管理
"""

import os
import sys
import time
import threading
import queue
import hashlib
from typing import Dict, List, Tuple, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from collections import deque
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
    from .base import BaseDeduplicator
    from .config import DedupConfig
except ImportError:
    from base import BaseDeduplicator
    from config import DedupConfig


class StreamingStrictHashDeduplicator(BaseDeduplicator):
    """改进版流式极速严格哈希去重器 - 解决进度条停滞问题"""
    
    def __init__(self, config: DedupConfig):
        super().__init__(config)
        
        # 流式处理配置 - 使用用户配置参数
        self.num_workers = getattr(config, 'num_workers', None) or min(8, os.cpu_count() or 4)
        chunk_size_mb = getattr(config, 'chunk_size_mb', 16)
        buffer_size_mb = getattr(config, 'buffer_size_mb', 2)
        self.chunk_size = chunk_size_mb * 1024 * 1024  # 用户配置的块大小
        self.buffer_size = buffer_size_mb * 1024 * 1024   # 用户配置的读取缓冲区
        self.write_buffer_size = buffer_size_mb * 2 * 1024 * 1024  # 写入缓冲区是读取缓冲区的2倍
        
        # 队列大小控制 - 使用用户配置
        self.queue_maxsize = getattr(config, 'queue_maxsize', 8)   # 用户配置的队列大小
        self.result_queue_size = min(32, self.queue_maxsize * 4)  # 结果队列大小基于输入队列
        
        # 选择最快的处理函数
        if FAST_HASH_AVAILABLE:
            self.hash_func = lambda content: str(xxhash.xxh64(content.encode('utf-8')).intdigest())
            self.logger.info("🚀 使用xxHash极速哈希")
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
        
        # 统计信息 - 增加更多计数器
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
            'last_heartbeat': 0
        }
        
        # 控制标志
        self.stop_flag = threading.Event()
        self.debug_mode = True
        
        # 使用分片哈希表减少锁竞争
        self.num_hash_shards = 16  # 16个分片
        self.global_hashes_shards = [set() for _ in range(self.num_hash_shards)]
        self.shard_locks = [threading.RLock() for _ in range(self.num_hash_shards)]
        
        self.logger.info(f"🎯 流式去重器初始化完成，{self.num_workers}个worker，块大小{self.chunk_size//1024//1024}MB")
    
    def _get_shard_index(self, content_hash: str) -> int:
        """计算哈希分片索引"""
        return hash(content_hash) % self.num_hash_shards
    
    def _is_duplicate(self, content_hash: str) -> bool:
        """检查是否重复（使用分片锁）"""
        shard_idx = self._get_shard_index(content_hash)
        with self.shard_locks[shard_idx]:
            if content_hash in self.global_hashes_shards[shard_idx]:
                return True
            self.global_hashes_shards[shard_idx].add(content_hash)
            return False
    
    def _file_reader_thread(self, filepath: str, chunk_queue: queue.Queue):
        """文件读取线程 - 改进版流式分块读取"""
        try:
            file_size = os.path.getsize(filepath)
            self.logger.info(f"📁 文件大小: {file_size / (1024**3):.2f} GB")
            
            bytes_read = 0
            chunk_id = 0
            last_log_time = time.time()
            
            with open(filepath, 'rb', buffering=self.buffer_size) as f:
                while not self.stop_flag.is_set():
                    read_start = time.time()
                    
                    # 读取一个块
                    chunk_data = f.read(self.chunk_size)
                    if not chunk_data:
                        break
                    
                    # 确保块在行边界结束
                    if len(chunk_data) == self.chunk_size:
                        extra = f.readline()
                        chunk_data += extra
                    
                    bytes_read += len(chunk_data)
                    read_time = time.time() - read_start
                    
                    # 将块加入队列（优化等待时间）
                    queue_start = time.time()
                    try:
                        chunk_queue.put((chunk_id, chunk_data, bytes_read, file_size), timeout=1)
                        chunk_id += 1
                        self.stats['chunks_read'] += 1
                        
                        queue_time = time.time() - queue_start
                        
                        # 更频繁的调试日志（每5个块或每3秒记录一次）
                        if chunk_id % 5 == 0 or time.time() - last_log_time > 3:
                            progress_pct = (bytes_read / file_size) * 100
                            self.logger.info(f"📖 读取进度 {progress_pct:.1f}% - 块{chunk_id}, 读取{read_time:.3f}s, 入队{queue_time:.3f}s")
                            last_log_time = time.time()
                        
                        # 动态调整读取速度（更激进的调整）
                        queue_size = chunk_queue.qsize()
                        if queue_size > self.queue_maxsize // 2:
                            time.sleep(0.01)  # 队列较满时稍微减慢
                        elif queue_size > self.queue_maxsize * 0.8:
                            time.sleep(0.02)  # 队列很满时减慢
                            
                    except queue.Full:
                        if not self.stop_flag.is_set():
                            self.logger.warning(f"⚠️  块队列满（大小: {chunk_queue.qsize()}），暂停读取")
                            time.sleep(0.05)  # 减少等待时间
                
                # 发送结束信号
                chunk_queue.put(None)
                self.logger.info(f"📖 文件读取完成: {bytes_read:,} 字节, 共 {chunk_id} 个块")
                
        except Exception as e:
            self.logger.error(f"❌ 文件读取错误: {e}")
            chunk_queue.put(None)
    
    def _process_chunk_streaming(self, chunk_data: bytes, chunk_id: int) -> Tuple[int, int, Dict[str, str]]:
        """处理单个数据块 - 优化版本，添加去重效果记录"""
        processed_lines = 0
        unique_lines = 0
        local_hashes = {}
        hash_to_data = {}  # 存储第一次见到的数据，用于重复记录
        
        try:
            chunk_text = chunk_data.decode('utf-8', errors='ignore')
            lines = chunk_text.strip().split('\n')
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = self.json_loads(line)
                    content_values = []
                    
                    for key in self.config.content_keys:
                        if key in data:
                            content_values.append(str(data[key]))
                    
                    if content_values:
                        content = ' '.join(content_values)
                        content_hash = self.hash_func(content)
                        
                        processed_lines += 1
                        
                        # 检查是否重复
                        if content_hash not in local_hashes:
                            # 第一次见到这个哈希
                            local_hashes[content_hash] = line + '\n'
                            hash_to_data[content_hash] = data  # 保存数据用于重复记录
                            unique_lines += 1
                        else:
                            # 发现重复数据，记录到日志
                            if self.duplicate_count < 1000:  # 限制记录数量
                                kept_data = hash_to_data[content_hash]
                                removed_data = data
                                
                                # 对于strict_hash，相似度总是1.0（完全相同）
                                similarity = 1.0
                                method_info = {
                                    'hash_algorithm': 'xxHash' if FAST_HASH_AVAILABLE else 'blake2b',
                                    'content_hash': content_hash,
                                    'match_type': 'exact'
                                }
                                
                                self.log_duplicate_record(
                                    removed_data=removed_data,
                                    kept_data=kept_data,
                                    similarity=similarity,
                                    method_specific_info=method_info
                                )
                
                except Exception as e:
                    self.logger.warning(f"行 {processed_lines} 处理失败: {e}")
                    continue
            
        except Exception as e:
            self.logger.warning(f"块 {chunk_id} 处理错误: {e}")
        
        return processed_lines, unique_lines, local_hashes
    
    def _result_writer_thread(self, output_file: str, result_queue: queue.Queue, progress_bar: tqdm):
        """结果写入线程 - 改进进度更新机制"""
        try:
            with open(output_file, 'w', encoding='utf-8', buffering=self.write_buffer_size) as f_out:
                write_buffer = []
                buffer_limit = 1000  # 1000行一次性写入，进一步减小提高响应速度
                last_update_time = time.time()
                last_heartbeat = time.time()
                
                while not self.stop_flag.is_set():
                    try:
                        result = result_queue.get(timeout=0.2)  # 进一步减小超时时间
                        if result is None:  # 结束信号
                            break
                        
                        processed, unique, local_hashes = result
                        
                        # 收集要写入的行
                        for content_hash, line_content in local_hashes.items():
                            if not self._is_duplicate(content_hash):
                                write_buffer.append(line_content)
                        
                        # 批量写入
                        if len(write_buffer) >= buffer_limit:
                            f_out.writelines(write_buffer)
                            f_out.flush()
                            write_buffer.clear()
                            self.stats['chunks_written'] += 1
                        
                        # 更新统计
                        self.stats['total_processed'] += processed
                        self.stats['unique_count'] += unique
                        self.stats['chunks_processed'] += 1
                        self.stats['bytes_processed'] += processed * 500  # 估算字节数
                        
                        # 超频繁更新进度条（每0.2秒更新一次）
                        current_time = time.time()
                        if current_time - last_update_time > 0.2:
                            progress_bar.n = self.stats['total_processed']
                            
                            # 修复速度计算
                            elapsed = current_time - self.stats['start_time']
                            if elapsed > 0:
                                lines_per_sec = self.stats['total_processed'] / elapsed
                                self.stats['processing_speed_mb_s'] = (self.stats['bytes_processed'] / (1024**2)) / elapsed
                            else:
                                lines_per_sec = 0
                            
                            progress_bar.set_postfix({
                                '块': f"{self.stats['chunks_processed']}/{self.stats['chunks_read']}",
                                '唯一': f"{self.stats['unique_count']:,}",
                                '速度': f"{lines_per_sec:.0f}行/s" if lines_per_sec > 0 else "计算中",
                                'MB/s': f"{self.stats['processing_speed_mb_s']:.1f}",
                                '队列': f"{result_queue.qsize()}"
                            })
                            progress_bar.refresh()
                            last_update_time = current_time
                            
                            # 调试日志
                            if self.debug_mode and self.stats['chunks_processed'] % 10 == 0:
                                self.logger.info(f"📊 处理进度: {self.stats['chunks_processed']}块, {self.stats['total_processed']:,}行, {self.stats['unique_count']:,}唯一, {self.stats['processing_speed_mb_s']:.1f}MB/s")
                        
                        # 心跳机制 - 即使没有新数据也要更新显示
                        elif current_time - last_heartbeat > 1.0:
                            progress_bar.set_postfix({
                                '状态': "处理中",
                                '块': f"{self.stats['chunks_processed']}/{self.stats['chunks_read']}",
                                '队列': f"{result_queue.qsize()}",
                                '心跳': f"{int(current_time) % 60}s"
                            })
                            progress_bar.refresh()
                            last_heartbeat = current_time
                        
                    except queue.Empty:
                        # 超时时也要更新进度条显示当前状态
                        current_time = time.time()
                        if current_time - last_update_time > 0.5:
                            progress_bar.set_postfix({
                                '状态': "等待数据",
                                '块': f"{self.stats['chunks_processed']}/{self.stats['chunks_read']}",
                                '队列': f"{result_queue.qsize()}",
                                '等待': f"{current_time - last_update_time:.1f}s"
                            })
                            progress_bar.refresh()
                            last_update_time = current_time
                        continue
                
                # 写入剩余数据
                if write_buffer:
                    f_out.writelines(write_buffer)
                    f_out.flush()
                
                self.logger.info(f"💾 写入完成: {self.stats['unique_count']:,} 唯一行")
                
        except Exception as e:
            self.logger.error(f"❌ 写入错误: {e}")
    
    def deduplicate(self) -> Tuple[List[int], List[int], Dict[str, Any]]:
        """执行改进版流式去重"""
        start_time = time.time()
        self.stats['start_time'] = start_time
        
        self.logger.info("🚀 启动改进版流式极速去重")
        self.logger.info(f"🔧 配置: {self.num_workers}个worker, {self.chunk_size//1024//1024}MB块")
        self.logger.info("🐛 改进模式: 超频繁进度更新，心跳机制，修复速度计算")
        self.logger.info("⚡ 立即开始处理，实时显示进度...")
        
        # 创建队列
        chunk_queue = queue.Queue(maxsize=self.queue_maxsize)
        result_queue = queue.Queue(maxsize=self.result_queue_size)
        
        # 创建进度条（无总量，实时更新）
        progress_bar = tqdm(
            desc="🚀 改进流式处理", 
            unit='行', 
            unit_scale=True,
            bar_format='{desc}: {n_fmt} 行 | {rate_fmt} | {elapsed} | {postfix}',
            position=0, 
            leave=True
        )
        
        try:
            # 启动文件读取线程
            self.logger.info("📖 文件读取线程已启动")
            reader_thread = threading.Thread(
                target=self._file_reader_thread,
                args=(self.config.input_file, chunk_queue)
            )
            reader_thread.start()
            
            # 启动结果写入线程
            self.logger.info("💾 结果写入线程已启动")
            writer_thread = threading.Thread(
                target=self._result_writer_thread,
                args=(self.config.output_file, result_queue, progress_bar)
            )
            writer_thread.start()
            
            # 主处理循环
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                processed_chunks = 0
                bytes_processed = 0
                
                future_to_chunk = {}
                
                while not self.stop_flag.is_set():
                    # 获取新的块
                    try:
                        chunk_item = chunk_queue.get(timeout=0.5)  # 减少超时时间
                        if chunk_item is None:  # 读取完成
                            break
                        
                        chunk_id, chunk_data, bytes_read, file_size = chunk_item
                        bytes_processed = bytes_read
                        
                        # 提交处理任务
                        future = executor.submit(self._process_chunk_streaming, chunk_data, chunk_id)
                        future_to_chunk[future] = chunk_id
                        
                        # 收集完成的任务
                        completed_futures = []
                        for future in list(future_to_chunk.keys()):
                            if future.done():
                                completed_futures.append(future)
                        
                        for future in completed_futures:
                            try:
                                result = future.result()
                                result_queue.put(result)
                                del future_to_chunk[future]
                                processed_chunks += 1
                                
                                # 实时更新处理速度
                                elapsed = time.time() - start_time
                                if elapsed > 0:
                                    self.stats['processing_speed_mb_s'] = (bytes_processed / (1024**2)) / elapsed
                                
                            except Exception as e:
                                self.logger.warning(f"块处理失败: {e}")
                                del future_to_chunk[future]
                    
                    except queue.Empty:
                        # 即使没有新块，也要检查已完成的任务
                        completed_futures = []
                        for future in list(future_to_chunk.keys()):
                            if future.done():
                                completed_futures.append(future)
                        
                        for future in completed_futures:
                            try:
                                result = future.result()
                                result_queue.put(result)
                                del future_to_chunk[future]
                                processed_chunks += 1
                            except Exception as e:
                                self.logger.warning(f"块处理失败: {e}")
                                del future_to_chunk[future]
                        continue
                
                # 等待剩余任务完成
                for future in future_to_chunk:
                    try:
                        result = future.result()
                        result_queue.put(result)
                    except Exception as e:
                        self.logger.warning(f"最终块处理失败: {e}")
            
            # 等待写入完成
            result_queue.put(None)  # 发送结束信号
            writer_thread.join()
            reader_thread.join()
            
            progress_bar.close()
            
            # 最终统计
            total_time = time.time() - start_time
            duplicate_count = self.stats['total_processed'] - self.stats['unique_count']
            dedup_ratio = duplicate_count / self.stats['total_processed'] if self.stats['total_processed'] > 0 else 0
            
            method_name = str(self.config.method.value) if hasattr(self.config.method, 'value') else str(self.config.method)
            final_stats = {
                'total_lines': self.stats['total_processed'],
                'unique_lines': self.stats['unique_count'],
                'duplicate_lines': duplicate_count,
                'dedup_ratio': dedup_ratio,
                'total_time': total_time,
                'processing_speed_mb_s': self.stats['processing_speed_mb_s'],
                'lines_per_second': self.stats['total_processed'] / total_time if total_time > 0 else 0,
                'chunks_processed': self.stats['chunks_processed'],
                'method': f'{method_name}_streaming',
                'mode': 'streaming'
            }
            
            self.logger.info("=" * 80)
            self.logger.info("🎉 改进版流式去重完成!")
            self.logger.info(f"📊 总行数: {final_stats['total_lines']:,}")
            self.logger.info(f"✅ 保留行数: {final_stats['unique_lines']:,}")
            self.logger.info(f"🗑️  删除行数: {final_stats['duplicate_lines']:,}")
            self.logger.info(f"📉 去重比例: {final_stats['dedup_ratio']:.2%}")
            self.logger.info(f"📈 处理速度: {final_stats['processing_speed_mb_s']:.1f} MB/秒")
            self.logger.info(f"📈 行处理速度: {final_stats['lines_per_second']:,.0f} 行/秒")
            self.logger.info(f"🧩 处理块数: {final_stats['chunks_processed']}")
            self.logger.info(f"⏱️  总耗时: {final_stats['total_time']:.1f} 秒")
            self.logger.info("=" * 80)
            
            return [], [], final_stats
            
        except Exception as e:
            self.logger.error(f"❌ 去重过程出错: {e}")
            progress_bar.close()
            raise
        finally:
            self.stop_flag.set() 