#!/bin/bash
# N-gram LSH去重配置示例
# 特点：文本相似度去重，适合识别相似文本

python -m deduplicate \
  --method ngram_lsh \
  --input_file data/input.jsonl \
  --output_file data/output_dedup.jsonl \
  --content_keys prompt \
  --ngram_size 10 \
  --jaccard_threshold 0.85 \
  --num_permutations 128 \
  --num_workers 16 \
  --batch_size 10000

