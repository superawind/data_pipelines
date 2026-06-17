import os
import json
import time
import random
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed


def get(query, idx):
    openai_api_key = "EMPTY"
    if idx % 2 == 0:
        openai_api_base = 'http://10.16.80.150:8032/v1'
    else:
        openai_api_base = 'http://10.16.80.150:8031/v1'
    # openai_api_base = random.choice(["http://10.16.80.154:8027/v1", "http://10.16.80.154:8027/v1"])
    # openai_api_base = 

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    
    # query 传入的是字典

    try:
        completion = client.chat.completions.create(
            model="/workspace/zxd/Qwen/Qwen3-30BA3B",
            messages=[
                # {'role': 'system', 'content': 'You are a helpful assistant.'},
                # {"role": "user", "content": query['instruction']},
                {"role": "user", "content": query['instruction']},
            ],
            # temperature=0.6,
            temperature=0.7,
            top_p=0.8,
            max_tokens=8192,
            extra_body={
                "top_k":20
            }
        )
        content = completion.choices[0].message.content
        finish_reason = completion.choices[0].finish_reason
        output_len = completion.usage.completion_tokens
    except Exception as e:
        print('miss an error......................', e)
        content = ''; finish_reason = 'length'; output_len = 0
    
    # return {'id': idx, 'source': query['source'], 'prompt': query['prompt'], 'output' : content, 'finish_reason' : finish_reason, 'output_tokens': output_len}
    return {'id': idx, 'prompt': query['instruction'], 'output' : content, 'finish_reason' : finish_reason, 'output_tokens': output_len}
    # return {'id': idx, 'instruction':query, 'input': '', 'output' : content, 'finish_reason' : finish_reason, 'output_tokens': output_len}
    # return {'id': idx, 'instruction':query['instruction'], 'choice': query['output'], 'reject' : content, 'finish_reason' : finish_reason, 'output_tokens': output_len}


def get_answer_from_rw_livebench_querys(path, start_index=0):
    # path = '/mnt/code/zhaoxudong03/data_pipelines/training_data_zxd/embedding_dep_thred_90.jsonl'
    cur_lst = path.split('/')
    save_path = '/'.join(cur_lst[:-2]) + '/res_rewrite_' + cur_lst[-2]  
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path = os.path.join(save_path, cur_lst[-1])
    questions = []

    with open(path, 'r', encoding='utf8') as f:
        for line in f.readlines():
            sample = json.loads(line)
            # query = sample['prompt']
            questions.append(sample)
    """
        print(os.path.join(path, file))
        # print(save_path)
        # return 
        with open(os.path.join(path, file), 'r', encoding='utf8') as f:
            for line in f.readlines():
                sample = json.loads(line)
                query = sample['rewrite_query']
                if query.startswith('#####改写后的待改写文本开始') and query.endswith('#####改写后的待改写文本结束'):
                    cur_ = query.split('#####改写后的待改写文本开始')[1].split('#####改写后的待改写文本结束')[0]
                    questions.append(cur_)
    """
        
    cur_datas = questions
    print(len(cur_datas))
    # random.shuffle(cur_datas)
    if start_index:
        cur_datas = cur_datas[start_index:]

    print(len(cur_datas))
    # cur_datas = cur_datas[:200]

    for idx in range(0, len(cur_datas), 5000):
        start = idx
        end = min(start+5000, len(cur_datas))
        cur_ = cur_datas[start: end]
        
        results = []
        with ThreadPoolExecutor(max_workers=400) as executor:  # 可调整线程数
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
    # for name in ['Business_&_Finance.jsonl', 'Creative_Writing.jsonl', 'Daily_Life_&_Health.jsonl', 'General_Knowledge_QA.jsonl', 'Humanities_Arts_&_Social_Sciences.jsonl']:
        # get_answer_from_rw_livebench_querys('/mnt/code/zhaoxudong03/data_pipelines/training_data_zxd/step2/clarity_over_8_to_10_and_cg_over_8_to_10/{}'.format(name))
    
    get_answer_from_rw_livebench_querys('/mnt/code/zhaoxudong03/data_pipelines/training_data_zxd/rw_query_code_265k.jsonl')