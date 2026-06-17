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
def get_from_local_api(elem, openai_api_base=None):
    openai_api_key = "EMPTY"
    openai_api_base = openai_api_base

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    try:
        completion = client.chat.completions.create(
            model="/opt/users/models/Qwen3-235B-A22B-Instruct-2507-FP8",
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
    
    elem.update({"model_gene": content})
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
    # ori_datas_dir = '/mnt/code/zhaoxudong03/data_pipelines/training_data_zxd/wangfei_zh/step2/clarity_over_9_to_10_and_cg_over_9_to_10'
    # ori_datas_path = os.path.join(ori_datas_dir, path.split('/')[-1])
    # ori_datas_9_to_10 = get_ori_datas(ori_datas_path)
    # ori_ = 0
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
                # if cur_['prompt'] in ori_datas_9_to_10:
                #     ori_ += 1
                #     continue
                res.append(cur_)

            results = res
    print('data length :::', len(results))
    # print('data length :::', len(results), '已经保存过的数量：：：', ori_)
    return results

def main(path, start_index=0, end_index=None, openai_api_base=None, save_path=None):

    if path.endswith('json') or path.endswith('jsonl'):
        datas = read_datas(path)
    elif path.endswith('csv'):
        datas = pd.read_csv(path)
        datas = datas.to_dict(orient='records')

    # random.shuffle(datas)
    if not end_index:
        end_index = len(datas)

    cur_datas = datas[start_index: end_index]
    # print(get(cur_datas[0], openai_api_base, model_type, taskid))
    print('实际需要生成的数量：：：', len(cur_datas))
    for idx in range(0, len(cur_datas), 1000):
        start = idx
        end = min(start+1000, len(cur_datas))
        cur_ = cur_datas[start: end]
        
        results = []
        with open(save_path, 'a', encoding='utf8') as f_save:
            with ThreadPoolExecutor(max_workers=200) as executor:  # 可调整线程数
                futures = {executor.submit(get_from_local_api, query, openai_api_base): query for i, query in enumerate(cur_)}
                with tqdm(total=len(cur_)) as pbar:
                    for future in as_completed(futures):
                        results.append(future.result())
                        f_save.write(json.dumps(future.result(), ensure_ascii=False)+'\n')
                        pbar.update(1)
            
if __name__ == '__main__':
    # ori_dir = '/mnt/code/zhaoxudong03/data_pipelines/training_data_zxd/wangfei_zh/step2/clarity_over_9_to_10_and_cg_over_9_to_10'
    # save_dir = '/mnt/code/zhaoxudong03/data_pipelines/training_data_zxd/wangfei_zh/step2/res_clarity_over_9_to_10_and_cg_over_9_to_10'
    
    # 对是 over_8_to_10 不包含全部 over_9_to_10 的数据进行response 生成
    ori_dir = '/mnt/code/zhaoxudong03/data_pipelines/training_data_zxd_v2/data/step2/clarity_over_8_to_10_and_cg_over_8_to_10'
    save_dir = '/mnt/code/zhaoxudong03/data_pipelines/training_data_zxd_v2/data/step2/res_clarity_over_8_to_10_and_cg_over_8_to_10'
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    exists = ['Business_&_Finance.jsonl', 'Code.jsonl', 'Creative_Writing.jsonl', 'Daily_Life_&_Health.jsonl', 'Data_Analysis.jsonl', 'General_Knowledge_QA.jsonl', 'Humanities_Arts_&_Social_Sciences.jsonl', 'Instruction_Following_&_Text_Processing.jsonl']
    for path in os.listdir(ori_dir):
        start_index = 0
        end_index = None
        if path in exists:
            print(f'{path} already generate==============================')
            continue
        elif path in ['Math.jsonl']:
            start_index = 135000

        # # over_9_to_10 从 code 后每个 类别都 需要在从 [0, 11000] 条生成。 math ,natural_&_applied_sciences, reasoning_&_logic 三个
        # if path in ['Business_&_Finance.jsonl', 'Code.jsonl']:
        #     continue
        # elif path in ['Math.jsonl', 'Natural_&_Applied_Sciences.jsonl', 'Reasoning_&_Logic.jsonl']:
        #     start_index = 0
        #     end_index = 11000
        cur_path = os.path.join(ori_dir, path)
        save_path = os.path.join(save_dir, path)
        print('save_path:::', save_path)
        main(cur_path, start_index=start_index, end_index=end_index, openai_api_base='http://10.16.80.150:6137/v1', save_path=save_path)
        # break
    

            
