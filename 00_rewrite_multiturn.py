# [在此处插入你的多轮对话数据，例如：]
# [
#   {"role": "user", "content": "..."},
#   {"role": "assistant", "content": "..."},
#   {"role": "user", "content": "..."},
#   {"role": "assistant", "content": "..."}
# ]

import os
import json
import time
import random

from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

prompt = """# Role
你是一个顶级的数据清洗与对话润色专家，专门负责优化大模型训练数据集（SFT dataset）。你的任务是对一段标准的多轮对话进行整体改写和精简。

# Goal
在**绝对不改变原文核心主旨、不引入外部知识、严格遵守上下文因果时间线**的前提下，将 Assistant（助理）的回复进行“极限脱水”和润色，使其表达尽可能精简、干练、专业。

# Hard Constraints (铁律，严禁违反)
1. **主旨一致性**：改写后的对话必须保持原有意图、技术要点、逻辑结论和事实完全一致，不能删减核心干货。
2. **严禁信息早泄（核心）**：改写某一轮的 Assistant 回复时，**只能基于该轮及该轮之前的历史信息**。绝对禁止将后续轮次（Future turns）中 User 提到的新概念、新错误日志、或 Assistant 在后文才给出的解决方案，提前泄露到前面的回复中。每一轮对话在时间线上必须是独立的因果链。
3. **上下文极度流畅**：虽然进行了精简，但必须保证 User 提问和 Assistant 回复之间的衔接极其自然，上下文指代清晰，没有生硬的断层。
4. **格式保持**：严格保持原有的多轮对话 JSON/文本格式，仅修改其中的 `content` 文本，不要附加任何额外的解释或碎碎念。

# Optimization Strategy (精简与润色策略)
* **删减客套话**：删掉所有诸如“好的，没问题”、“很高兴为您解答”、“正如您所说”、“抱歉让您久等了”等无意义的礼貌用语和垫话。
* **合并同类项**：将废话连篇的解释转化为一句话结论，或使用清晰的 Markdown 列表（Bullet points）。
* **直奔主题**：Assistant 的第一句话直接回答 User 的核心问题或直接给出代码/命令。
* **语言风味**：保持专业、冷静、技术流的语气（Geek style），惜字如金。

---

# Input Format (输入格式)
请对以下多轮对话进行改写：
{}

# Output Format (输出格式)
请直接输出改写后的标准 JSON 数组，不要包含任何前导词或后续解释。参考输出格式如下：
[
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "..."},
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "..."}
  ...
]
"""

def read_jsonl(path):
    results = []
    try:
        with open(path, 'r', encoding='utf8') as f:
            datas = json.load(f)
            results = datas
    except Exception as e:
        print('----------------------', e)
        res = []
        with open(path, 'r', encoding='utf8') as f:
            for line in f.readlines():
                cur_ = json.loads(line)
                res.append(cur_)

            results = res
    print('data length :::', len(results))
    return results


def get(query, idx):
    openai_api_key = "EMPTY"
    # openai_api_base = "http://172.8.94.114:8011/v1"
    openai_api_base = "http://10.16.80.9:8027/v1"
    # openai_api_base = "http://10.16.80.9:8027/v1"

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    
    # query 传入的是字典

    try:
        completion = client.chat.completions.create(
            model="/workspace/zxd/Qwen/Qwen3-30B-A3B-Instruct-2507",
            messages=[
                # {'role': 'system', 'content': 'You are a helpful assistant.'},
                # {"role": "user", "content": query['instruction']},
                # {"role": "user", "content": prompt+query['prompt']},
                {"role": "user", "content": prompt.format(query['messages'])},
            ],
            # temperature=0.6,
            temperature=0.7,
            top_p=0.8,
            max_tokens=8192,
            extra_body={
                # "chat_template_kwargs": {"enable_thinking": False},
                "top_k":20
                # "separate_reasoning": True
            }
        )
        content = completion.choices[0].message.content
        finish_reason = completion.choices[0].finish_reason
        output_len = completion.usage.completion_tokens
    except Exception as e:
        print('miss an error......................', e)
        content = ''; finish_reason = 'length'; output_len = 0
    query.update({'idx':idx, 'output' : content, 'finish_reason' : finish_reason, 'output_tokens': output_len})
    return query
    # return {'id': idx, 'source': query['source'], 'prompt': query['prompt'], 'output' : content, 'finish_reason' : finish_reason, 'output_tokens': output_len}
    # return {'id': idx, 'instruction':query, 'input': '', 'output' : content, 'finish_reason' : finish_reason, 'output_tokens': output_len}
    # return {'id': idx, 'instruction':query['instruction'], 'choice': query['output'], 'reject' : content, 'finish_reason' : finish_reason, 'output_tokens': output_len}


def get_answer_from_rw_livebench_querys(path, save_path, start_index=0, end_index=None):
    questions = read_jsonl(path)
    if not end_index:
        end_index = len(questions)
        
    cur_datas = questions[start_index: end_index]
    print(len(cur_datas))

    for idx in range(0, len(cur_datas), 5000):
        start = idx
        end = min(start+5000, len(cur_datas))
        cur_ = cur_datas[start: end]
        
        results = []
        with ThreadPoolExecutor(max_workers=300) as executor:  # 可调整线程数
            futures = {executor.submit(get, query, i+idx+start_index): query for i, query in enumerate(cur_)}
            with tqdm(total=len(cur_)) as pbar:
                for future in as_completed(futures):
                    results.append(future.result())
                    pbar.update(1)

        # save_path = '/mnt/code/zhaoxudong03/Train/LLaMA-Factory/data/zxd/datas/livebench_rewrite_Qwen3_2507/rewrite_datas_2507_gen_ans_{}'.format(file)
        with open(save_path, 'a', encoding='utf8') as f_save:
            for elem in results:
                f_save.write(json.dumps(elem, ensure_ascii=False)+'\n')


if __name__ == '__main__':
    # path = '/mnt/code/zhaoxudong03/data_pipelines/training_data_zxd/embedding_dep_thred_90.jsonl'
    # save_path = '/mnt/code/zhaoxudong03/data_pipelines/training_data_zxd/totals_datas_cls_2.jsonl'
    
    # path = '/opt/users/zxd/wangfei_prompt_zh_02_embeded_dup_for_30b_a3b_judge.jsonl'
    # save_path = '/mnt/code/zhaoxudong03/data_pipelines/training_data_zxd/wangfei_prompt_zh_03_30b_a3b_cls_before.jsonl'
    
    path = '/mnt/code/zhaoxudong03/data_pipelines/training_data_zxd_v2/data/total_202w_only_156w_cat_nvidia-2_stages_204w_total_360w_02_embeded_dep_extract_for_30b_a3b_judge.jsonl'
    save_path = '/mnt/code/zhaoxudong03/data_pipelines/training_data_zxd_v2/data/total_202w_only_156w_cat_nvidia-2_stages_204w_total_360w_02_embeded_dep_extract_for_30b_a3b_judge.jsonl_cls_before.jsonl'
    get_answer_from_rw_livebench_querys(path, save_path, start_index=0, end_index=4300000)
    # get_answer_from_rw_livebench_querys(path, save_path, start_index=0, end_index=10)
