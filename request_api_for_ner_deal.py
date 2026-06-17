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

from concurrent.futures import ThreadPoolExecutor, as_completed
import openai
from openai import *

# print(inspect.getfile(rllm_reward_fn_code))
# 1、逻辑子集规则 (Key Constraint): * 若答案为 JSON/Dictionary 格式，则其所有的 Key (键) 必须能原封不动地在问题中找到。
#     判定逻辑: 答案中绝对不能引入任何问题中未提及的新属性或新范畴。


# 3、严格遵守问题要求：
#     如果问题中明确要求回复格式，答案必须严格遵守

# 注意：你只需要按照上述规则判断答案格式是否符合输出要求，禁止解答具体题目，仅做格式判断。
    # 内部一致: 模型答案内部严禁出现无意义的中英混排（除非是专业术语，或者问题中也是中英文混排）。
# 
prompt = '''你是一个极其严苛的数据质量审计专家，专门负责筛选 LLM 训练语料中的问答对。你的任务是根据2项核心规则，扫描数据并识别其中的违规项。
**【输入数据】**:
**用户问题开始**
 %s
**用户问题结束**

**模型答案开始**
 %s
**模型答案结束**

**【规则如下】**:
1、语境一致性规则 (Language Consistency):
    外部一致: 模型答案的主语言必须与用户问题的主语言完全匹配（例如：问中文答中文，问英文答英文）。

2、翻译禁止规则 (No Translation):
    严禁模型答案将用户问题的内容进行“语种互换”（例如：问题是中文，答案是该问题中内容的英文翻译）。
    例如问题中提到要提取年龄，但是答案中是{"age": ...} 自动进行翻译的均为违反规则 

注意：这两项规则都是针对答案中语言的规则，你只需要判断答案的语言是否满足即可，禁止额外判断其他内容

**【输出格式要求】**:
    必须且仅返回一个 JSON 字典。
    若完全符合上述所有规则：返回 {"res": true}
    若违反任何一项规则：返回 {"res": false, "reason": "简述违规点"}。
'''

# /code/zhaoxudong03/data_pipelines/training_data_zxd/step2/clarity_over_8_to_10_and_cg_over_8_to_10/Data_Analysis.jsonl 
def get_from_local_api(elem, openai_api_base=None):
    openai_api_key = "EMPTY"
    openai_api_base = openai_api_base

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    # print(prompt%(elem['messages'][0]['content'], elem['messages'][1]['content']))
    # print('\n\n')
    try:
        completion = client.chat.completions.create(
            model="/workspace/zxd/Qwen3.5-122B-A10B",
            messages=[
                {"role": "user", "content": prompt%(elem['messages'][0]['content'], elem['messages'][1]['content'])},
                # {"role": "user", "content": prompt%('从下面文本中抽取出面试者年龄和身高，我的年龄是18岁，身高171cm，', '{“age”:18, "身高":"171cm"}')},
            ],
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

def main(path, start_index=0, end_index=None, openai_api_base=None, save_path=None, save_path_error=None):

    if path.endswith('json') or path.endswith('jsonl'):
        datas = read_datas(path)
    elif path.endswith('csv'):
        datas = pd.read_csv(path)
        datas = datas.to_dict(orient='records')

    # random.shuffle(datas)
    if not end_index:
        end_index = len(datas)

    # cur_datas = datas[start_index: end_index]
    cur_datas = datas[221723:]
    # print(get(cur_datas[0], openai_api_base, model_type, taskid))
    print('实际需要生成的数量：：：', len(cur_datas))

    for idx in range(0, len(cur_datas), 100000):
        start = idx
        end = min(start+100000, len(cur_datas))
        cur_ = cur_datas[start: end]
        
        results = []
        with open(save_path, 'a', encoding='utf8') as f_save, open(save_path_error, 'w', encoding='utf8') as f_save2:
            with ThreadPoolExecutor(max_workers=100) as executor:  # 可调整线程数
                futures = {executor.submit(get_from_local_api, query, openai_api_base): query for i, query in enumerate(cur_)}
                with tqdm(total=len(cur_)) as pbar:
                    for future in as_completed(futures):
                        results.append(future.result())
                        # 判断如果 为 true
                        try:
                            cur = json.loads(future.result()['model_gene'])
                            if cur['res'] == True:
                                f_save.write(json.dumps(future.result(), ensure_ascii=False)+'\n')
                            else:
                                f_save2.write(json.dumps(future.result(), ensure_ascii=False)+'\n')
                        except:
                            print('---------------生成格式存在问题--------------')
                            f_save2.write(json.dumps(future.result(), ensure_ascii=False)+'\n')
                        pbar.update(1)

                    
def yield_data(path):
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            yield json.loads(line)
      
if __name__ == '__main__':
    # 过滤 IC、Ner中 语言混乱的样本
    ori_path = '/mnt/code/zhaoxudong03/data_pipelines/training_data/457w/datas_domain_ner_dep_70w_format.jsonl'
    save_path = '/mnt/code/zhaoxudong03/data_pipelines/training_data/457w/datas_domain_ner_dep_70w_format_true.jsonl'
    save_path_error = '/mnt/code/zhaoxudong03/data_pipelines/training_data/457w/datas_domain_ner_dep_70w_format_false.jsonl'
    # 1、生成response
    start_index = 0
    end_index = None

    print('save_path:::', save_path)
    main(ori_path, start_index=start_index, end_index=end_index, openai_api_base='http://10.159.0.45:6027/v1', save_path=save_path, save_path_error=save_path_error)

    # 3、对提取后的代码进行reward计算，验证rllm-deepcoder的reward函数是否合理，不需要第二部提取，提取出来，沙箱反而无法识别代码，无法进行正确的验证
    # compute_score(save_for_shaxiang)
    # save_path = '/mnt/code/zhaoxudong03/RL/verl/datas/deepcoder/deepcoder_train_modified_swift_122b_distill_1times.jsonl'
    # compute_score1(save_path, 20, 0)

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

