#!/bin/bash
# Embedding LSH去重配置示例（LSH优化模式）
# 特点：语义相似度去重，LSH加速，适合千万级、亿级数据

python -m deduplicate \
  --method embedding \
  --input_file data/input.jsonl \
  --output_file data/output_dedup.jsonl \
  --content_keys prompt \
  --embeddings_file data/embeddings.jsonl \
  --embeddings_format jsonl \
  --prompt_id_key prompt_id \
  --embedding_id_key prompt_id \
  --embedding_vector_key embedding \
  --threshold 0.95 \
  --top_k 10 \
  --use_lsh \
  --lsh_num_tables 10 \
  --lsh_hash_size 10 \
  --num_workers 16 \
  --batch_size 10000

# LSH参数调优建议：
# - 高精度要求：--lsh_num_tables 15-20
# - 一般场景：--lsh_num_tables 10
# - 追求速度：--lsh_num_tables 5-8
# - 超大数据集(>1000万)：--lsh_hash_size 10-12
# - 中等数据集：--lsh_hash_size 8-10

# 如果使用多个embedding文件：
# python -m deduplicate \
#   --method embedding \
#   --embeddings_files data/embeddings_part1.jsonl data/embeddings_part2.jsonl \
#   --use_lsh \
#   --lsh_num_tables 10 \
#   --lsh_hash_size 10

