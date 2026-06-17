from typing import List, Tuple
import requests
import inspect
import re
import os
import json
import time
import random
from tqdm import tqdm
import pandas as pd
from tqdm import tqdm 
from datasets import load_dataset
# from rllm.rewards.rl_reward import rllm_reward_fn_zxd

import sys
sys.path.insert(0, '/mnt/code/zhaoxudong03/RL/X-Coder-main/rl-recipe/verl/utils/reward_score')
from our_code import compute_score

# 现在 /mnt/code/zhaoxudong03/RL/X-Coder-main/rl-recipe/code-judge 中按照 readme_zxd.md 文件配置好沙箱

from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import openai
from openai import OpenAI

# print(inspect.getfile(rllm_reward_fn_code))

WRITE_LOCK = Lock()

def append_jsonl(file_obj, data):
    with WRITE_LOCK:
        file_obj.write(json.dumps(data, ensure_ascii=False) + '\n')
        file_obj.flush()

                        
def yield_data(path):
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            yield json.loads(line)

def parse_python(s):
    # 正则表达式解释：
    # ```python\n  匹配起始标记
    # (.*?)        核心内容：非贪婪匹配任意字符
    # \n```        匹配结束标记
    pattern = r"```python\n(.*?)\n```"
    
    # re.DOTALL 允许 . 匹配换行符
    matches = re.findall(pattern, s, re.DOTALL)
    return matches


# 开始走rllm-deepcoder的验证流程
def _score_single(data):
    # 这里的score函数可以根据实际情况进行调整，下面是一个示例
    idx = data['id']
    model_gene = data['response']
    input_output = data['input_output']

    score = compute_score(
        model_gene,
        input_output,
        continuous=True
    )
    print(score)
    return data, bool(score)

# /code/zhaoxudong03/data_pipelines/training_data_zxd/step2/clarity_over_8_to_10_and_cg_over_8_to_10/Data_Analysis.jsonl
def get_from_local_api(elem, client=None, max_retries=5, success_file=None, error_file=None):
    last_elem = dict(elem)
    for retry_count in range(1, max_retries + 1):
        cur_elem = dict(elem)
        try:
            completion = client.chat.completions.create(
                model="/workspace/zxd/Qwen3.5-122B-A10B",
                # messages=[
                #     {"role": "user", "content": elem['messages'][0]['content']},
                # ],
                messages=elem['prompt'],
                temperature=0.6,
                top_p=0.95,
                max_tokens=8192,
                extra_body={
                    "top_k":20,
                    "chat_template_kwargs": {"enable_thinking": False},
                    # "chat_template_kwargs": {"thinking": True}
                }
            )
            content = completion.choices[0].message.content
            finish_reason = completion.choices[0].finish_reason
            output_len = completion.usage.completion_tokens
        except Exception as e:
            print('miss an error......................', e)
            content = ''
            finish_reason = 'length'
            output_len = 0

        cur_elem.update(
            {
                "response": content,
                "finish_reason": finish_reason,
                "output_len": output_len,
                "retry_count": retry_count,
            }
        )
        last_elem = cur_elem

        # 验证失败时重新生成，最多重试 max_retries 次。
        data, score = _score_single(cur_elem)
        if score:
            append_jsonl(success_file, data)
            return data

    last_elem["validation_failed"] = True
    append_jsonl(error_file, last_elem)
    return last_elem

def get_ori_datas(path):
    res = []
    with open(path, 'r', encoding='utf8') as f:
        for line in f.readlines():
            cur_ = json.loads(line)
            res.append(cur_['prompt'])
    print('已经打好的数据量为：：：', len(res))
    return res
    
def read_datas(path):
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

def yield_data(path):
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            yield json.loads(line)

def main(path, start_index=0, end_index=None, client=None, max_retries=5, success_path=None, error_path=None):
    # if path.endswith('json') or path.endswith('jsonl'):
    #     datas = read_datas(path)
    # elif path.endswith('csv'):
    #     datas = pd.read_csv(path)
    #     datas = datas.to_dict(orient='records')
    ds = load_dataset('parquet', data_files=path)['train']
    datas = []
    for elem in ds:
        datas.append(elem)

    # random.shuffle(datas)
    if not end_index:
        end_index = len(datas)

    # cur_datas = datas[start_index: end_index]
    cur_datas = datas[:1]
    # print(get(cur_datas[0], openai_api_base, model_type, taskid))
    print('实际需要生成的数量：：：', len(cur_datas))
    print('success_path:::', success_path)
    print('error_path:::', error_path)

    with open(success_path, 'a', encoding='utf8') as success_file, open(error_path, 'a', encoding='utf8') as error_file:
        for idx in range(0, len(cur_datas), 2000):
            start = idx
            end = min(start+2000, len(cur_datas))
            cur_ = cur_datas[start: end]
            
            results = []
            with ThreadPoolExecutor(max_workers=200) as executor:  # 可调整线程数
                futures = {
                    executor.submit(
                        get_from_local_api,
                        query,
                        client,
                        max_retries,
                        success_file,
                        error_file,
                    ): query
                    for i, query in enumerate(cur_)
                }
                with tqdm(total=len(cur_)) as pbar:
                    for future in as_completed(futures):
                        results.append(future.result())
                        pbar.update(1)

            
if __name__ == '__main__':
    # 对是 over_8_to_10 不包含全部 over_9_to_10 的数据进行response 生成
    ori_path = '/mnt/code/zhaoxudong03/zxd_datas/X-Coder/real_rl_data/non_sys_prompt/leetcode_2772.parquet'
    # 验证通过path 和 多次验证失败 path
    success_path = '/mnt/code/zhaoxudong03/zxd_datas/X-Coder/gene_real/xcoder_sandbox_8time_true.jsonl'
    error_path = "/mnt/code/zhaoxudong03/zxd_datas/X-Coder/gene_real/xcoder_sandbox_8time_error.jsonl"
    
    # 1、生成response
    start_index = 0
    end_index = None
    
    openai_api_key = "EMPTY"
    openai_api_base ='http://10.159.0.45:6028/v1'
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    main(ori_path, start_index=start_index, end_index=end_index, client=client, max_retries=1, success_path=success_path, error_path=error_path)


