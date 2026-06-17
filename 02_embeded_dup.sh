#!/bin/bash
# /mnt/code/zhaoxudong03/data_pipelines/dep/llmTools/deduplicate/examples 新的代码位置
# input_file 是一个 文件，包含两个字段 source , prompt 
# embeddings_files 是多个文件，包含 idx, source , embeded ，理论上两个文件内容应该以source字段一一对应，如果不对应会自动筛选

# 切换到工作目录 
cd llmTools || exit

# # strict_dep.jsonl 共计数据 6721226
# # /code/zhaoxudong03/data_pipelines/training_data/merged_data.jsonl \
# nohup python -m deduplicate \
#   --method embedding \
#   --mode global \
#   --input_file /opt/users/zxd/llmtools/strict_dep_cat_wangfei_prompt_zh_01_strict_dup.jsonl \
#   --output_file /opt/users/zxd/llmtools/strict_dep_cat_wangfei_prompt_zh_02_embeded_dup.jsonl \
#   --embeddings_files "/opt/users/zxd/llmtools/strict_dep_embeded_0-6B.jsonl" "/opt/users/zxd/llmtools/strict_dep_embeded_0-6B-2.jsonl" "/opt/users/zxd/llmtools/wangfei_prompt_1477w_01_strict_dep_embeded_0-6B.jsonl" \
#   --embeddings_format jsonl \
#   --content_keys prompt \
#   --threshold 0.88 \
#   --prompt_id_key source \
#   --embedding_id_key source \
#   --embedding_vector_key embeded \
#   --num_workers 16 \
#   --top_k 20 \
#   --use_gpu \
#   --gpu_device 7 >> embeded_dep.log & 


# strict_dep.jsonl 共计数据 6721226
# /code/zhaoxudong03/data_pipelines/training_data/merged_data.jsonl \
python -m deduplicate \
  --method embedding \
  --mode global \
  --input_file /mnt/code/zhaoxudong03/data_pipelines/training_data_zxd_v2/data/total_202w_only_156w_cat_nvidia-2-stages-204w-total-360w-prompt.jsonl \
  --output_file /mnt/code/zhaoxudong03/data_pipelines/training_data_zxd_v2/data/total_202w_only_156w_cat_nvidia-2_stages_204w_total_360w_02_embeded_dep.jsonl \
  --embeddings_files "/mnt/code/zhaoxudong03/data_pipelines/202w_totals/total_202w_only_156w_embedding.jsonl" "/mnt/code/zhaoxudong03/data_pipelines/training_data_zxd_v2/data/strict_dep_after_204w_02_embedding_0_6B.jsonl" \
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