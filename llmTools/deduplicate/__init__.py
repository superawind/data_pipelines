#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
流式去重工具包 - 零延迟启动，实时进度显示

支持三种流式去重算法：
- streaming_strict_hash: 流式严格哈希去重
- streaming_ngram_lsh: 流式N-gram LSH去重
- streaming_embedding: 流式Embedding去重

特性：
- 🚀 零延迟启动 - 立即开始处理
- 📊 实时进度显示 - 不再停滞
- 🌊 边读边处理 - 流式架构  
- 💓 心跳机制 - 防止假死

使用方法:
    python -m deduplicate --method strict_hash --input_file data.jsonl --output_file output.jsonl

版本: 4.0.0 (纯流式版)
"""

__version__ = "4.0.0"
__author__ = "AI Assistant"
__email__ = ""
__description__ = "流式数据去重工具包 - 零延迟启动，实时进度显示"

# 导入主要类和函数
try:
    from .dep_code.config import DedupConfig, DedupMethod
    from .dep_code.base import BaseDeduplicator, DedupPreviewResult
    from .dep_code.streaming_strict_hash import StreamingStrictHashDeduplicator
    from .dep_code.streaming_ngram_lsh import StreamingNgramLshDeduplicator  
    from .dep_code.streaming_embedding import StreamingEmbeddingDeduplicator
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'dep_code'))
    
    from config import DedupConfig, DedupMethod
    from base import BaseDeduplicator, DedupPreviewResult
    from streaming_strict_hash import StreamingStrictHashDeduplicator
    from streaming_ngram_lsh import StreamingNgramLshDeduplicator
    from streaming_embedding import StreamingEmbeddingDeduplicator

__all__ = [
    'DedupConfig',
    'DedupMethod', 
    'BaseDeduplicator',
    'DedupPreviewResult',
    'StreamingStrictHashDeduplicator',
    'StreamingNgramLshDeduplicator',
    'StreamingEmbeddingDeduplicator'
] 