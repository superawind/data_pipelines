from typing import List, Tuple
import requests
import inspect
import re
import os
import json
import time
import random
import pandas as pd
from tqdm import tqdm 
import datasets

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

# /code/zhaoxudong03/data_pipelines/training_data_zxd/step2/clarity_over_8_to_10_and_cg_over_8_to_10/Data_Analysis.jsonl
def get_from_local_api(elem, client=None, max_retries=5, success_file=None, error_file=None):
    last_elem = dict(elem)
    try:
        completion = client.chat.completions.create(
            model="/workspace/zxd/Qwen3.5-122B-A10B",
            # messages=[
            #     {"role": "user", "content": elem['messages'][0]['content']},
            # ],
            messages=elem['prompt'],
            temperature=0.6,
            top_p=0.95,
            max_tokens=4096,
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

    elem.update({
        'response': content,
        'finish_reason': finish_reason,
        'output_len': output_len,
    })

    return elem

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

def main(path, start_index=0, end_index=None, client=None, max_retries=5, success_path=None, error_path=None):
    datas = []
    for file_name in os.listdir(path):
        if file_name.endswith('parquet'):
            ds = datasets.load_dataset('parquet', data_files=[os.path.join(path, file_name)])['train']
            for elem in ds:
                datas.append(elem)
            #     if len(datas) >= 100:
            #         break
            # break
                # print(elem)
                # return 

    # random.shuffle(datas)
    if not end_index:
        end_index = len(datas)

    # cur_datas = datas[start_index: end_index]
    cur_datas = datas
    # print(get(cur_datas[0], openai_api_base, model_type, taskid))
    print('实际需要生成的数量：：：', len(cur_datas))
    print('success_path:::', success_path)
    print('error_path:::', error_path)

    with open(success_path, 'w', encoding='utf8') as success_file:
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
                    ): query
                    for i, query in enumerate(cur_)
                }
                with tqdm(total=len(cur_)) as pbar:
                    for future in as_completed(futures):
                        success_file.write(json.dumps(future.result(), ensure_ascii=False) + '\n')
                        pbar.update(1)

            
if __name__ == '__main__':
    # 对是 over_8_to_10 不包含全部 over_9_to_10 的数据进行response 生成
    ori_dir = '/mnt/code/zhaoxudong03/zxd_datas/X-Coder/syn_rl_data/xcoder_data/sorted_by_passrate'

    success_path = '/mnt/code/zhaoxudong03/zxd_datas/X-Coder/xcoder_train_modified_swift_122b_distill_1times.jsonl'
    error_path = '/mnt/code/zhaoxudong03/zxd_datas/X-Coder/xcoder_train_modified_swift_122b_distill_1times_error_error.jsonl'
    
    # 1、生成response
    start_index = 0
    end_index = None
    
    openai_api_key = "EMPTY"
    openai_api_base ='http://10.159.0.45:6028/v1'
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    main(ori_dir, start_index=start_index, end_index=end_index, client=client, max_retries=5, success_path=success_path, error_path=error_path)
