import json
from tqdm import tqdm 

def read_datas(path, max_nums=None):
    cur_counts = 0
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            cur_counts += 1
            if max_nums and cur_counts > max_nums:
                return {}
            yield line

def save_datas(save_path, method, datas):
    with open(save_path, method, encoding='utf8') as f_save:
        for sample in datas:
            f_save.write(json.dumps(sample, ensure_ascii=False)+'\n')

# 读取数据
def read_datas_am_thinking(path, data_cls):
    # results = []
    # with open(path, 'r', encoding='utf8') as f:
    #     for line in f.readlines():
    mul_ = 0; count = 0; datas = []
    for line in tqdm(read_datas(path)):
        cur_ = json.loads(line)
        if len(cur_['conversations']) == 2:
            count += 1
            instruction = cur_['conversations'][0]['value']
            output = cur_['conversations'][1]['value']
            think_, ans = output.split('<answer>')
            think_ = think_.split('<think>')[1].split('</think>')[0].strip()
            ans_ = ans.split('</answer>')[0]
            sample = {'prompt': instruction, 'think': think_, 'output': ans_, 'data_cls': data_cls, 'source': 'AM-DS-R1-0528-Distilled'}
            datas.append(sample)
        else:
            mul_ += 1
        # break
    print('多轮对话的数量为:::', mul_)

    # print(datas)
    return datas

def read_datas_OmniThough_0528(path, save_path):
    # 数据包括 mathematics, coding, and science, 尚未进行分类
    with open(save_path, 'w', encoding='utf8') as f_save
        for line in read_datas(path):
            cur_ = json.loads(line)
            instruction = cur_['question']
            think_ = cur_['reasoning'][0]['thought']
            output = cur_['reasoning'][0]['solution']
            rv = cur_['reasoning'][0]['Reasoning_Verbosity']['level']
            cd = cur_['reasoning'][0]['Cognitive_Difficulty']['level']
            sample = {'prompt': instruction, 'think': think_, 'output': output, 'rv': rv, 'cd': cd, 'source': 'OmniThought-R1-0528'}
            f_save.wirte(json.dumps(sample, ensure_ascii=False)+'\n')
        

if __name__ == '__main__':
    datas = read_datas_am_thinking('/code/zhaoxudong03/datas/AM-DeepSeek-R1-0528-Distilled/math.jsonl', 'math', )
    # 保存数据
    save_datas('/code/zhaoxudong03/data_pipelines/datas/am-thinking.jsonl', 'a', datas)

                