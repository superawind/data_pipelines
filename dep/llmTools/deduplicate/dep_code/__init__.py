#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
去重核心模块
"""

# 导入配置和枚举
from .config import DedupMethod, DedupConfig

# 导入基类
from .base import BaseDeduplicator, DedupPreviewResult

# 导出主要接口
__all__ = [
    'DedupMethod',
    'DedupConfig',
    'BaseDeduplicator',
    'DedupPreviewResult',
]
