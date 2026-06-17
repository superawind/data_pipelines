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
from rllm.rewards.rl_reward import rllm_reward_fn_zxd

from concurrent.futures import ThreadPoolExecutor, as_completed
import openai
from openai import *

# print(inspect.getfile(rllm_reward_fn_code))

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
            model="/workspace/zxd/Qwen3.5-122B-A10B",
            messages=[
                {"role": "user", "content": elem['messages'][0]['content']},
            ],
            temperature=0.6,
            top_p=0.95,
            max_tokens=16000,
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

    # cur_datas = datas[start_index: end_index]
    cur_datas = datas
    # print(get(cur_datas[0], openai_api_base, model_type, taskid))
    print('实际需要生成的数量：：：', len(cur_datas))

    for idx in range(0, len(cur_datas), 1000):
        start = idx
        end = min(start+1000, len(cur_datas))
        cur_ = cur_datas[start: end]
        
        results = []
        with open(save_path, 'a', encoding='utf8') as f_save:
            with ThreadPoolExecutor(max_workers=100) as executor:  # 可调整线程数
                futures = {executor.submit(get_from_local_api, query, openai_api_base): query for i, query in enumerate(cur_)}
                with tqdm(total=len(cur_)) as pbar:
                    for future in as_completed(futures):
                        results.append(future.result())
                        f_save.write(json.dumps(future.result(), ensure_ascii=False)+'\n')

                        pbar.update(1)

                        

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
def compute_score(path='/mnt/code/zhaoxudong03/RL/verl_2601/datas/deepcoder/deepcoder_train_modified_swift_122b_distill_1times_for_shaxiang.jsonl'):
    save_for_pred = '/mnt/code/zhaoxudong03/RL/verl_2601/datas/deepcoder/deepcoder_train_modified_swift_122b_distill_1times_for_pred.jsonl'
    save_for_reresponse = '/mnt/code/zhaoxudong03/RL/verl_2601/datas/deepcoder/deepcoder_train_modified_swift_122b_distill_1times_error_for_reresponse.jsonl'
    idx = 0
    f_t = open(save_for_pred, 'a', encoding='utf8')
    f_f = open(save_for_reresponse, 'a', encoding='utf8')
    t, f = 0, 0
    cur_datas = []
    # for data in tqdm(yield_data(path)):
    #     cur_datas.append(data)
    

    # for idx in range(0, len(cur_datas), 100):
    #     start = idx
    #     end = min(start+100, len(cur_datas))
    #     cur_ = cur_datas[start: end]
    #     data_sources = [data['extra_info']['data_source'] for data in cur_]
    #     model_genes = ['```python\n' + data['model_gene'] + '\n```' for data in cur_]
    #     solution_strs = [data['solution'] for data in cur_]
    #     extra_infos = [data['extra_info']['extra_info'] for data in cur_]
        
    #     results = []
    #     with ThreadPoolExecutor(max_workers=100) as executor:  # 可调整线程数
    #         results = list(executor.map(rllm_reward_fn_zxd, data_sources, model_genes, solution_strs, extra_infos))
    #         print(results)

    #     break
    count = 0
    for data in tqdm(yield_data(path)):
        count += 1 # 1433 + 452 = 1885
        if count <= 1885:
            continue
        model_gene = data['model_gene']
        solution_str = data['solution']
        extra_infos_ = data['extra_info']
        extra_info = extra_infos_['extra_info']
        data_source = extra_infos_['data_source']
        # if isinstance(extra_info, str):
        #     ground_truth = {'tests': json.loads(solution_str)}
        # elif isinstance(extra_info, dict):
        #     ground_truth = {'tests': solution_str}
        # else:
        #     print('extra_info type is unexpected:::', type(extra_info))
        #     continue
        
    #     # print('model_res:::', model_gene[:], 'data_source', data_source, 'ground_truth:::', solution_str, 'extra_info:::', extra_info)
    #     # break
        score = rllm_reward_fn_zxd(data_source, '```python\n' + model_gene + '\n```', solution_str, extra_info)
        if score:
            # 模型生成正确，直接保存到文件中，不需要重新生成了
            t += 1
            f_t.write(json.dumps(data, ensure_ascii=False)+'\n')
        else:
            f += 1
            f_f.write(json.dumps(data, ensure_ascii=False)+'\n')
            # 保存到需要重新生成的文件中
        # score = rllm_reward_fn_code(data_source, '```python\n' + model_gene + '\n```', ground_truth)

    # print(f"Total: {t+f}, Correct: {t}, Incorrect: {f}")
    # f_t.close()
    # f_f.close()


# 开始走rllm-deepcoder的验证流程
def _score_single(data):
    model_gene = data['model_gene']
    solution_str = data['solution']
    extra_infos_ = data['extra_info']
    extra_info = extra_infos_['extra_info']
    data_source = extra_infos_['data_source']
    score = rllm_reward_fn_zxd(
        data_source,
        # '```python\n' + model_gene + '\n```',
        model_gene ,
        solution_str,
        extra_info,
    )
    return data, bool(score)


def compute_score1(
    path='/mnt/code/zhaoxudong03/RL/verl/datas/deepcoder/deepcoder_train_modified_swift_122b_distill_1times_for_shaxiang.jsonl',
    max_workers=32,
    skip_count=0,
):
    save_for_pred = '/mnt/code/zhaoxudong03/RL/verl/datas/deepcoder/deepcoder_train_modified_swift_122b_distill_1times_for_pred_true.jsonl'
    save_for_reresponse = '/mnt/code/zhaoxudong03/RL/verl/datas/deepcoder/deepcoder_train_modified_swift_122b_distill_1times_error_for_reresponse_true.jsonl'

    pending_datas = []
    for count, data in enumerate(yield_data(path), start=1):
        if count < skip_count:  # 1433 + 452 = 1885
            continue
        pending_datas.append(data)

    # pending_datas = pending_datas[skip_count:]  # 先测试1000条，确认流程正确
    print(f"Total pending datas to score: {len(pending_datas)}")
    t, f = 0, 0
    with open(save_for_pred, 'a', encoding='utf8') as f_t, open(save_for_reresponse, 'a', encoding='utf8') as f_f:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_score_single, data) for data in pending_datas]
            for future in tqdm(as_completed(futures), total=len(futures)):
                data, score = future.result()
                if score:
                    # 模型生成正确，直接保存到文件中，不需要重新生成了
                    t += 1
                    f_t.write(json.dumps(data, ensure_ascii=False) + '\n')
                else:
                    f += 1
                    f_f.write(json.dumps(data, ensure_ascii=False) + '\n')
                    # 保存到需要重新生成的文件中

    print(f"Total: {t + f}, Correct: {t}, Incorrect: {f}")

            
if __name__ == '__main__':
    # ori_dir = '/mnt/code/zhaoxudong03/data_pipelines/training_data_zxd/wangfei_zh/step2/clarity_over_9_to_10_and_cg_over_9_to_10'
    # save_dir = '/mnt/code/zhaoxudong03/data_pipelines/training_data_zxd/wangfei_zh/step2/res_clarity_over_9_to_10_and_cg_over_9_to_10'
    
    # 对是 over_8_to_10 不包含全部 over_9_to_10 的数据进行response 生成
    # ori_path = '/code/zhaoxudong03/RL/verl_2601/datas/deepcoder/deepcoder_train_modified_swift.jsonl'
    save_path = '/mnt/code/zhaoxudong03/RL/verl/datas/deepcoder/deepcoder_train_modified_swift_122b_distill_1times.jsonl'
    # save_for_reresponse_path = '/code/zhaoxudong03/RL/verl_2601/datas/deepcoder/deepcoder_train_modified_swift_122b_distill_1times_for_reresponse.jsonl'
    # save_for_shaxiang = '/mnt/code/zhaoxudong03/RL/verl_2601/datas/deepcoder/deepcoder_train_modified_swift_122b_distill_1times_for_shaxiang.jsonl'
    # 1、生成response
    start_index = 0
    end_index = None

    print('save_path:::', save_path)
    # main(ori_path, start_index=start_index, end_index=end_index, openai_api_base='http://10.25.117.72:8003/v1', save_path=save_path)

    # 2、解析生成的response，提取代码部分
    # f_shaxiang = open(save_for_shaxiang, 'w', encoding='utf8')
    # totals, success = 0, 0
    # with open('save_for_reresponse_path', 'w', encoding='utf8') as f_save:
    #     for data in yield_data(save_path):
    #         totals += 1
    #         model_res = data['model_gene']
    #         code_parts = parse_python(model_res)
    #         if len(code_parts) == 0:
    #             # 此处需要重新生成
    #             f_save.write(json.dumps(data, ensure_ascii=False)+'\n')
    #             continue

    #         model_res = code_parts[0]  # 取第一个代码块作为最终的生成结果
    #         data['model_gene'] = model_res
    #         f_shaxiang.write(json.dumps(data, ensure_ascii=False)+'\n')
    #         success += 1
    #         # for i, code in enumerate(code_parts):
    #         #     print(code)

    #         # break
    # f_shaxiang.close()
    # print(f"Total: {totals}, Success: {success}")

    # 3、对提取后的代码进行reward计算，验证rllm-deepcoder的reward函数是否合理，不需要第二部提取，提取出来，沙箱反而无法识别代码，无法进行正确的验证
    # compute_score(save_for_shaxiang)
    save_path = '/mnt/code/zhaoxudong03/RL/verl/datas/deepcoder/deepcoder_train_modified_swift_122b_distill_1times.jsonl'
    # compute_score1(save_path, 20, 0)

    path = '/mnt/code/zhaoxudong03/RL/verl/datas/deepcoder/deepcoder_train_modified_swift_122b_distill_1times_for_pred_true.jsonl'
    count = 0
    for elem in yield_path(path):
        count += 1
    print(count)
    # # 4、读取数据，拼接成训练数据格式
    # path = '/mnt/code/zhaoxudong03/RL/verl_2601/datas/deepcoder/deepcoder_train_modified_swift_122b_distill_1times_for_pred.jsonl'
    # count = 0
    # with open('/mnt/code/zhaoxudong03/data_pipelines/training_data/457w/deepcoder_data_deal_shaixiang_18k.jsonl', 'w', encoding='utf8') as f_save:
    #     for data in tqdm(yield_data(path)):
    #         count += 1
    #         messages = {
    #             "messages": [
    #                 {"role": "user", "content": data['messages'][0]['content']},
    #                 {"role": "assistant", "content": data['model_gene']}
    #             ]
    #         }
    #         f_save.write(json.dumps(messages, ensure_ascii=False)+'\n')
        

    # print(f"Total data points: {count}")

# 你是一个优秀的数据筛选助手，你将收到一对问答对，你需要判断问答对是否满足如下规则，具体规则如下：
# 问题：
# {}
# 答案：
# {}
# 规则1: 答案如果是字典，必须满足字典的键全部出现在问题中。
# 规则2: 答案的语言风格必须和问题的语言风格保持一致，例如问答对都是中文，或者都是英文。
# 规则3: 答案整体语言风格保持一致，例如，答案的内容全部是中文，或者全部是英文。
# 规则4: 严禁出现答案中将问题中内容翻译的情况。

# 你必须严格核查这个问答对是否满足上述全部规则，如果全部满足，以json 字典形式，返回{'res': True}，否则，如果存在一条不满足，则返回 {'res': False} 


prompt = '''你是一个极其严苛的数据质量审计专家，专门负责筛选 LLM 训练语料中的问答对。你的任务是根据四项核心指标，像编译器一样扫描数据并识别其中的违规项。
输入数据:
问题: {}
答案: {}

规则如下:
1、逻辑子集规则 (Key Constraint): * 若答案为 JSON/Dictionary 格式，则其所有的 Key (键) 必须能原封不动地在问题中找到。
    判定逻辑: 答案中绝对不能引入任何问题中未提及的新属性或新范畴。

2、语境一致性规则 (Language Consistency):
    外部一致: 答案的主语言必须与问题的主语言完全匹配（例如：问中答中，问英答英）。
    内部一致: 答案内部严禁出现无意义的中英混排（除非是专业术语）。

3、翻译禁止规则 (No Translation):
    严禁答案仅将问题的内容进行“语种互换”（例如：问题是中文，答案是该问题的英文翻译）。
    判定逻辑: 答案必须是针对问题的“解答”，而非“转述”或“翻译”。

输出格式要求:
    必须且仅返回一个 JSON 字典。
    若完全符合上述所有规则：返回 {"res": true}
    若违反任何一项规则：返回 {"res": false, "reason": "简述违规点"}。
'''