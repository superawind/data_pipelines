#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
流式去重核心模块

导出所有可用的流式去重器和配置类
"""

# 导入配置和枚举
from .config import DedupMethod, DedupConfig

# 导入基类
from .base import BaseDeduplicator, DedupPreviewResult

# 导入流式去重器
from .streaming_strict_hash import StreamingStrictHashDeduplicator
from .streaming_ngram_lsh import StreamingNgramLshDeduplicator
from .streaming_embedding import StreamingEmbeddingDeduplicator

# 导出主要接口
__all__ = [
    # 枚举和配置
    'DedupMethod',
    'DedupConfig',
    
    # 基类
    'BaseDeduplicator',
    'DedupPreviewResult',
    
    # 流式去重器
    'StreamingStrictHashDeduplicator',
    'StreamingNgramLshDeduplicator', 
    'StreamingEmbeddingDeduplicator',
]
