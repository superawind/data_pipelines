"""
新加入的数据，去重后合为一份数据，该脚本的作用是从过滤后的数据中，抽取出 新数据，然后去单独做 embedding ，便于后续 embedding 去重。
embedding 去重后，也需要 单独对新数据 采用 2507 蒸馏，也需要单独摘出来
"""

import json
from tqdm import tqdm 

def yield_datas(path):
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            yield line

def filter_ori_from_totals(ori_path, total_path, save_path):
    # 获取原始数据 的 prompt
    ori_prompts = []
    for line in tqdm(yield_datas(ori_path)):
        cur_ = json.loads(line)
        ori_prompts.append(cur_['prompt'])
    
    # 新增后，全部数据的 prompt
    total_prompts = []
    total_datas = []
    for line in tqdm(yield_datas(total_path)):
        cur_ = json.loads(line)
        total_prompts.append(cur_['prompt'])
        total_datas.append(cur_)
        
    # 按照 prompt 取差集
    new_prompts = set(total_prompts) - set(ori_prompts)
    print('总数据数量为：：：', len(total_datas))
    print('新数据数量为：：：', len(new_prompts))
    
    # 根据 prompt 获取对应的新数据 
    # 提取完成后，保存，用于后续 embedding 生成 或者 response 生成
    new_datas = []
    count = 0 
    with open(save_path, 'w', encoding='utf8') as f_save:
        for elem in tqdm(total_datas):
            if elem['prompt'] in new_prompts:
                # new_datas.append(elem)
                count += 1
                f_save.write(json.dumps(elem, ensure_ascii=False)+'\n')
    # wangfei_ori_prompt 14780194
    # strict_dup 14779157
    print('save count:::', count)
    
if __name__ == '__main__':
    filter_ori_from_totals('/mnt/code/zhaoxudong03/data_pipelines/training_data_zxd/strict_dep.jsonl', '/mnt/code/zhaoxudong03/data_pipelines/training_data_zxd/strict_dep_cat_wangfei_prompt_zh_01_strict_dup.jsonl', '/mnt/code/zhaoxudong03/data_pipelines/training_data_zxd/wangfei_prompt_zh_01_strict_dup_for_embedding.jsonl')