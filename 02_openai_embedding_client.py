import openai
import torch 
import json
from tqdm import tqdm
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoTokenizer

def get_ans(sample, idx, tokenizer):
    if len(tokenizer(sample['prompt'])['input_ids']) >= 30000:
        return {'idx': idx, 'source': sample['source'], 'embeded': []}
    if len(sample['prompt']) <= 10:
        return {'idx': idx, 'source': sample['source'], 'embeded': []}

        # context_over += 1
        # continue
    client = openai.Client(base_url=f"http://10.16.80.9:8027/v1", api_key="None")
    # client = openai.Client(base_url=f"http://172.8.217.79:8028/v1", api_key="None")

    # Text embedding example
    # try:
    response = client.embeddings.create(
        model="Qwen3/Qwen3-Embedding-0.6B",
        input=sample['prompt'],
        # dimensions=1024
    )

    embedding = response.data[0].embedding
    # except:
    #     embedding = []
    # print(f"Text embedding (first 10): {embedding.shape}")
    # print(len(embedding))
    return {'idx': idx, 'source': sample['source'], 'embeded': embedding}

# tokenizer = AutoTokenizer.from_pretrained('/opt/users/models/Qwen3-Embedding-0.6B', trust_remote_code=True)

# res = get_ans({'prompt': '你好，你谁啊，为什么这呀', 'source':''}, 1, tokenizer)
# print(torch.tensor(res['embeded']).shape)

"""
其服务必须加上  --is-embedding
python -m sglang.launch_server --port=8027 --tp-size=2 --trust-remote-code --host 10.16.80.9  --mem-fraction-static 0.88 --model-path /opt/users/models/Qwen3-Embedding-0.6B --is-embedding
"""

def mul_get(path):
    questions = []; context_over = 0
    # tokenizer = AutoTokenizer.from_pretrained('/workspace/zxd/Qwen3-Embedding-0.6B', trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained('/opt/users/models/Qwen3-Embedding-0.6B', trust_remote_code=True)

    print('-----------begin--------------')
    with open(path, 'r', encoding='utf8') as f:
        for line in tqdm(f.readlines()):
            cur_ = json.loads(line)
            # if len(tokenizer(cur_['prompt'])['input_ids']) >= 30000:
            #     context_over += 1
            #     continue
            questions.append(cur_)
    # print('-----------begin--------------', context_over)
    
    # cur_datas = questions[5000000+1055000:]
    start_index = 0
    cur_datas = questions[start_index:]

    print(len(cur_datas))
    for idx in range(0, len(cur_datas), 5000):
        start = idx
        end = min(start+5000, len(cur_datas))
        cur_ = cur_datas[start: end]
        # ori_ = cur_datas[start: end]
        results = []
        with ThreadPoolExecutor(max_workers=300) as executor:  # 可调整线程数
            futures = {executor.submit(get_ans, query, i+idx+start_index, tokenizer): query for i, query in enumerate(cur_)}
            with tqdm(total=len(cur_)) as pbar:
                for future in as_completed(futures):
                    results.append(future.result())
                    pbar.update(1)

        # save_path = '/mnt/code/zhaoxudong03/data_pipelines/training_data_zxd/strict_dep_embeded_0-6B-2.jsonl'
        save_path = '/mnt/code/zhaoxudong03/data_pipelines/training_data_zxd_v2/data/strict_dep_after_204w_02_embedding_0_6B.jsonl'
        # save_path = '/code/zhaoxudong03/data_pipelines/training_data_zxd/strict_dep_embeded_0-6B.jsonl'
        with open(save_path, 'a', encoding='utf8') as f_save:
            for elem in results:
                if elem['embeded'] == []:
                    continue
                f_save.write(json.dumps(elem, ensure_ascii=False)+'\n')

if __name__ == '__main__':
    mul_get('/mnt/code/zhaoxudong03/data_pipelines/training_data_zxd_v2/data/strict_dep_cat_nvidia_two_stages_876w_01_strict_dep_for_embedding.jsonl')
    # mul_get('/mnt/code/zhaoxudong03/data_pipelines/training_data_zxd/strict_dep.jsonl')
    # mul_get('/code/zhaoxudong03/data_pipelines/training_data_zxd/strict_dep.jsonl')

