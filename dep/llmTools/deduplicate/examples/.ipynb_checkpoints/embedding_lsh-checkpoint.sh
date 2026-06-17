#!/bin/bash
# Embedding LSH去重配置示例（LSH优化模式）
# 特点：语义相似度去重，LSH加速，适合千万级、亿级数据
# 切换到工作目录
cd /code/zhaoxudong03/data_pipelines/dep/llmTools || exit

python -m deduplicate \
  --method embedding \
  --input_file /opt/users/zxd/llmtools/strict_dep_cat_wangfei_prompt_zh_01_strict_dup.jsonl \
  --output_file /opt/users/zxd/llmtools/strict_dep_cat_wangfei_prompt_zh_02_embeded_dup_lsh_test.jsonl \
  --embeddings_files "/opt/users/zxd/llmtools/strict_dep_embeded_0-6B.jsonl" "/opt/users/zxd/llmtools/strict_dep_embeded_0-6B-2.jsonl" "/opt/users/zxd/llmtools/wangfei_prompt_1477w_01_strict_dep_embeded_0-6B.jsonl" \
  --content_keys prompt \
  --embeddings_format jsonl \
  --prompt_id_key source \
  --embedding_id_key source \
  --embedding_vector_key embeded \
  --threshold 0.88 \
  --top_k 20 \
  --use_lsh \
  --lsh_num_tables 20 \
  --lsh_hash_size 12 \
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