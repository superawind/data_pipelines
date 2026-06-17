#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
去重工具包

支持三种去重方法：
- strict_hash: 严格哈希去重（完全匹配）
- ngram_lsh: N-gram LSH去重（文本相似）
- embedding: Embedding去重（语义相似）

使用方法:
    python -m deduplicate --method strict_hash --input_file data.jsonl --output_file output.jsonl

版本: 5.0.0
"""

__version__ = "5.0.0"
__author__ = "AI Assistant"
__description__ = "数据去重工具包"

# 导入主要类和函数
try:
    from .dep_code.config import DedupConfig, DedupMethod
    from .dep_code.base import BaseDeduplicator, DedupPreviewResult
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'dep_code'))
    
    from config import DedupConfig, DedupMethod
    from base import BaseDeduplicator, DedupPreviewResult

__all__ = [
    'DedupConfig',
    'DedupMethod', 
    'BaseDeduplicator',
    'DedupPreviewResult',
]
