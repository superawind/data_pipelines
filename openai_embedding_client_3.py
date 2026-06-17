import os 
import openai
import torch 
import json
from tqdm import tqdm
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoTokenizer

def get_ans(sample, idx, tokenizer):
    if len(tokenizer(sample['prompt'])['input_ids']) >= 30000:
        sample.update({'embeded': []})
        return sample
    if len(sample['prompt']) <= 10:
        sample.update({'embeded': []})
        return sample

    client = openai.Client(base_url=f"http://10.16.80.9:8027/v1", api_key="None")

    response = client.embeddings.create(
        model="Qwen3/Qwen3-Embedding-0.6B",
        input=sample['prompt'],
        # dimensions=1024
    )

    embedding = response.data[0].embedding
    # except:
    sample.update({'embeded': embedding})
    return sample



def mul_get(path):
    questions = []; context_over = 0
    file_name = path.split('/')[-1]
    save_path = '/mnt//code/zhaoxudong03/data_pipelines/training_data_zxd/step2/res_clarity_over_8_to_10_and_cg_over_8_to_10_embedding/' + file_name

    # tokenizer = AutoTokenizer.from_pretrained('/workspace/zxd/Qwen3-Embedding-0.6B', trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained('/opt/users/models/Qwen3-Embedding-0.6B', trust_remote_code=True)

    print('-----------begin--------------')
    questions = []
    for line in tqdm(yield_data(path)):
        questions.append(json.loads(line))
    # print('-----------begin--------------', context_over)
    
    # cur_datas = questions[5000000+1055000:]
    cur_datas = questions
    print(len(cur_datas))

    for idx in range(0, len(cur_datas), 5000):
        start = idx
        end = min(start+5000, len(cur_datas))
        cur_ = cur_datas[start: end]
        # ori_ = cur_datas[start: end]
        results = []
        with ThreadPoolExecutor(max_workers=200) as executor:  # 可调整线程数
            futures = {executor.submit(get_ans, query, i+idx, tokenizer): query for i, query in enumerate(cur_)}
            with tqdm(total=len(cur_)) as pbar:
                for future in as_completed(futures):
                    results.append(future.result())
                    pbar.update(1)

        # save_path = '/code/zhaoxudong03/data_pipelines/training_data_zxd/strict_dep_embeded_0-6B.jsonl'
        with open(save_path, 'a', encoding='utf8') as f_save:
            for elem in results:
                if elem['embeded'] == []:
                    continue
                f_save.write(json.dumps(elem, ensure_ascii=False)+'\n')

def yield_data(path):
    count = 10
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            if count >= 100:
                break
            yield line

if __name__ == '__main__':
    dir_path = '/mnt/code/zhaoxudong03/data_pipelines/training_data_zxd/step2/res_clarity_over_8_to_10_and_cg_over_8_to_10'
    file_lst = os.listdir(dir_path)
    for file_name in file_lst:
        cur_path = os.path.join(dir_path, file_name)

        # 读取数据并且生成
        mul_get(cur_path)

