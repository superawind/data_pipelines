#!/bin/bash
# 严格哈希去重配置示例
# 特点：完全匹配去重，速度最快，零误删

python -m deduplicate \
  --method strict_hash \
  --input_file data/input.jsonl \
  --output_file data/output_dedup.jsonl \
  --content_keys prompt \
  --num_workers 16 \
  --batch_size 50000

