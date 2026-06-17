#!/bin/bash

# 切换到工作目录
cd llmTools || exit

# /code/zhaoxudong03/data_pipelines/training_data/merged_data.jsonl \
python -m deduplicate \
  --method embedding \
  --mode global \
  --input_file /workspace/strict_dep.jsonl \
  --output_file /workspace/embedding_dep_thred_85.jsonl \
  --embeddings_files "/workspace/strict_dep_embeded_0-6B.jsonl" "/workspace/strict_dep_embeded_0-6B-2.jsonl" \
  --embeddings_format jsonl \
  --content_keys prompt \
  --threshold 0.88 \
  --prompt_id_key source \
  --embedding_id_key source \
  --embedding_vector_key embeded \
  --num_workers 16 \
  --top_k 20 \
  --use_gpu \
  --gpu_device 7
  # --jaccard_threshold 0.8 \
  # --ngram_size 10 \
  # --num_permutations 64

  # 严格哈希去重（完全匹配）
python -m deduplicate --method strict_hash --input_file data.jsonl --output_file output.jsonl

# N-gram LSH去重（相似度匹配）  
python -m deduplicate --method ngram_lsh --input_file data.jsonl --output_file output.jsonl \
    --jaccard_threshold 0.85 --ngram_size 10

# Embedding去重（语义匹配）
python -m deduplicate --method embedding --input_file data.jsonl --output_file output.jsonl \
    --embeddings_file embeddings.jsonl --threshold 0.95