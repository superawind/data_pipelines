#!/bin/bash

# 切换到工作目录
cd llmTools || exit

# # /code/zhaoxudong03/data_pipelines/training_data/merged_data.jsonl \
# python -m deduplicate \
#   --method strict_hash \
#   --mode global \
#   --input_file /mnt/code/zhaoxudong03/data_pipelines/training_data_zxd/strict_dep_cat_wangfei_prompt_zh.jsonl \
#   --output_file /mnt/code/zhaoxudong03/data_pipelines/training_data_zxd/strict_dep_cat_wangfei_prompt_zh_01_strict_dup.jsonl \
#   --content_keys prompt

# /code/zhaoxudong03/data_pipelines/training_data/merged_data.jsonl \
# 输入只需要 content_keys 默认为 prompt str类型， + source 用来识别数据位置，输出也是这两个字段
# python -m deduplicate \
#   --method strict_hash \
#   --mode global \
#   --input_file /mnt//code/zhaoxudong03/data_pipelines/training_data_zxd_v2/data/strict_dep_cat_nvidia_two_stages_876w.jsonl \
#   --output_file /mnt//code/zhaoxudong03/data_pipelines/training_data_zxd_v2/data/strict_dep_cat_nvidia_two_stages_876w_01_strict_dep.jsonl \
#   --content_keys prompt



python -m deduplicate \
  --method strict_hash \
  --mode global \
  --input_file /mnt/code/zhaoxudong03/zxd_datas/sft_datas_v2/Nemotron-Cascade2-deal/IF/instruction_following_single.jsonl \
  --output_file /mnt/code/zhaoxudong03/zxd_datas/sft_datas_v2/Nemotron-Cascade2-deal/IF/instruction_following_single_01_strict_dep.jsonl \
  --content_keys prompt