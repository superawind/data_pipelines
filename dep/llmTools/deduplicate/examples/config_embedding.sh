#!/bin/bash
# Embedding去重配置示例（精确模式）
# 特点：语义相似度去重，最高精度，适合中等规模数据

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
  --num_workers 16 \
  --batch_size 10000

# 如果使用NPY格式的embedding文件：
# python -m deduplicate \
#   --method embedding \
#   --input_file data/input.jsonl \
#   --output_file data/output_dedup.jsonl \
#   --embeddings_file data/embeddings.npy \
#   --embeddings_format npy \
#   --threshold 0.95 \
#   --top_k 10

# 如果使用GPU加速：
# 添加参数：--use_gpu --gpu_device 0

