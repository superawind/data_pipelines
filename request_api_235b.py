from typing import List, Tuple
import requests
import re
import os
import json
import time
import random
import pandas as pd
from tqdm import tqdm 
from concurrent.futures import ThreadPoolExecutor, as_completed
import openai
from openai import *

# /code/zhaoxudong03/data_pipelines/training_data_zxd/step2/clarity_over_8_to_10_and_cg_over_8_to_10/Data_Analysis.jsonl 
def get_from_local_api(elem, openai_api_base=None, idx=None):
    openai_api_key = "EMPTY"
    openai_api_base = openai_api_base

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    try:
        completion = client.chat.completions.create(
            model="Qwen3-235B-A22B-Instruction-2507-FP8",
            messages=[
                {"role": "user", "content": elem['prompt']},
            ],
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
        content = ''; finish_reason = 'length'; output_len = 0
    
    # elem.update({"model_gene": content})
    return {'id': idx, 'source': elem['source'], 'prompt': elem['prompt'], 'output': content, 'finish_reason': finish_reason, 'output_tokens': output_len}
    # return elem
    
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

def main(path, openai_api_base=None, save_path=None):

    datas = read_datas(path)

    cur_datas = datas
    for idx in range(0, len(cur_datas), 1000):
        start = idx
        end = min(start+1000, len(cur_datas))
        cur_ = cur_datas[start: end]
        
        results = []
        with open(save_path, 'a', encoding='utf8') as f_save:
            with ThreadPoolExecutor(max_workers=300) as executor:  # 可调整线程数
                futures = {executor.submit(get_from_local_api, query, openai_api_base, i): query for i, query in enumerate(cur_)}
                with tqdm(total=len(cur_)) as pbar:
                    for future in as_completed(futures):
                        results.append(future.result())
                        f_save.write(json.dumps(future.result(), ensure_ascii=False)+'\n')
                        pbar.update(1)
            
if __name__ == '__main__':
    # ori_dir = '/mnt/code/zhaoxudong03/data_pipelines/training_data_zxd/wangfei_zh/step2/clarity_over_9_to_10_and_cg_over_9_to_10'
    # save_dir = '/mnt/code/zhaoxudong03/data_pipelines/training_data_zxd/wangfei_zh/step2/res_clarity_over_9_to_10_and_cg_over_9_to_10'
    
    # 对是 over_8_to_10 不包含全部 over_9_to_10 的数据进行response 生成
    cur_path = '/mnt/code/zhaoxudong03/data_pipelines/training_data_zxd/step2/clarity_over_8_to_10_and_cg_over_8_to_10/Data_Analysis.jsonl'
    save_path = '/mnt/code/zhaoxudong03/data_pipelines/training_data_zxd/step2/res_clarity_over_8_to_10_and_cg_over_8_to_10/Data_Analysis.jsonl'

    main(cur_path, openai_api_base='http://10.16.80.154:5027/v1', save_path=save_path)
        # break
    

            
