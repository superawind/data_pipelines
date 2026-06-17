import os
import json
import time
import random
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed


# 对code 数据中，特别清晰度为10， 认知难度为10 的数据进行 query 改写

prompt = """你是一个专业的提示词工程师，擅长将用户输入的问题进行改写和优化，使其更符合语言表达习惯，同时保持原意不变。在改写时需特别注意以下几点：
#### 改写要求 ####
1. **准确保留原始含义**：改写后的问题不得改变用户意图，答案在语义上仍然适用于改写后的问题。
2. **优化语法表达**：使问题语言更通顺、自然、符合逻辑，消除歧义或语病。
3. **避免错误信息**：不得引入任何逻辑错误、常识错误或理论错误。
4. **可用性验证**：改写后的问题应能被大模型正常理解并作出合理回应。
5. **注意**：仅进行问题的改写与优化，禁止解答用户问题

#### 改写方式 ####
1. **语序修改**：在保持语义的情况下通过调整语句顺序
2. **增加、删除或修改非核心内容**：可以向用户输入的提示词中增加、删除或修改无关干扰语句，例如可以增加一句“今天天气真好”等
3. **提示词优化**：可以适当改变提示词的描述内容，使整体更流畅

请根据上述要求，对如下的“用户问题”进行改写，并仅返回改写后的结果，不添加任何额外解释或说明，严禁解答任何问题。
【用户问题】

"""

prompt2 = """**## 角色与目标 (Role & Goal)**
你是一位专业的指令重构与多样化专家。你的目标是接收一个“种子指令”，并严格遵循我定义的一套核心变换策略，通过组合这些策略，生成10个与种子指令主题类别一致、但任务和内容完全不同的全新指令。

**## 核心任务与原则 (Core Task & Principles)**
根据我提供的“种子指令”，生成10个新的、高质量的指令。所有生成的新指令必须严格遵守以下原则：

1.  **主题关联，内容迥异:** 新指令在宏观类别上（如“文本处理”、“数据分析”、“创意写作”）应与原指令保持一致，但其具体的任务内容、输入信息和要求必须有根本性的不同。
2.  **杜绝大段重复:** 新指令的文本描述不应与原指令或其他新指令有任何大段的雷同。目标是创造“形似而神不似”的新指令。
3.  **清晰可执行:** 每个新指令都必须语句通顺、要求合理、任务明确，可以被独立理解和执行。
4.  **强制组合变换:** 每一个新指令都**必须**是多种变换策略组合的结果，以确保其独特性和差异性。

**## 核心变换策略 (Core Variation Strategies)**
你**只能**使用以下策略，并需要将它们进行组合来创造新指令：

1.  **任务类型变换 (Task Type Transformation):**
    *   **说明:** 从根本上改变指令的核心动作。
    *   **示例:**
        *   解答 → 证明
        *   总结 → 扩写
        *   分析 → 创作
        *   提取 → 比较
        *   分类 → 预测

2.  **输入信息替换 (Input Information Replacement):**
    *   **说明:** 完全替换指令中包含的具体输入信息。如果原指令有特定文本、数据或主题，你需要创造一个全新的、不同领域或主题的输入。
    *   **示例:**
        *   原输入: "关于人工智能的最新研究文章..."
        *   新输入: "一段描述古代罗马建筑风格的文本..."
        *   原输入: "特斯拉公司的股票数据..."
        *   新输入: "一个关于全球咖啡消费量的统计数据集..."

3.  **输入/输出格式修改 (I/O Format Modification):**
    *   **说明:** 改变指令处理的数据的结构或格式。
    *   **示例:**
        *   输入格式: `纯文本` → `JSON对象`
        *   输出格式: `一段描述性文字` → `一个Markdown表格`
        *   输出格式: `一个完整的答案` → `一个包含问题、答案和引用来源的列表`

4.  **语言切换 (Language Switch):**
    *   **说明:** 将整个指令（包括任务描述和可能内嵌的输入信息）重写为一个流畅、地道的**中文**指令。

5.  **任务复杂度提升 (Task Complexity Increase):**
    *   **说明:** 在原任务基础上增加额外的步骤、分析维度或综合性要求。
    *   **示例:**
        *   单步任务: "总结这篇文章。"
        *   多步任务: "首先总结这篇文章，然后提取文章中的三个关键论点，并为每个论点提供一个反驳意见。"

**## 强制组合规则 (Mandatory Combination Rule)**
你生成的每一个新指令都**必须**是**至少两种**以上不同“核心变换策略”的组合成果。这是最重要的规则。

*   **组合示例:**
    *   **种子指令:** "Summarize the following English article about climate change."
    *   **组合策略:** `[任务类型变换]` + `[输入信息替换]` + `[语言切换]`
    *   **生成的新指令(可能):** "请根据以下关于中国宋代诗歌发展的中文描述，创作一首模仿那个时期风格的五言律诗。"
    *   *这个例子中，“总结”变为“创作”（任务类型变换），“气候变化文章”变为“宋代诗歌描述”（输入信息替换），并且整个指令变成了中文（语言切换）。*

**## 输出格式要求 (Output Format Requirement)**
你的最终输出**必须**是一个单一的、格式正确的JSON对象。这个JSON对象结构必须极其简单：

*   它只包含一个顶级键：`"generated_prompts"`。
*   这个键的值是一个**字符串列表 (a list of strings)**，列表中包含你生成的10个全新指令。

**你不再需要输出所用的策略或任何解释，只需提供纯粹的指令文本列表。**

**## JSON 输出示例 (Example of the Required Simple JSON Output)**
```json
{
  "generated_prompts": [
    "这是生成的第一个新指令...",
    "这是生成的第二个新指令，它与第一个完全不同...",
    "这是第三个，可能是一个中文指令...",
    "这是第四个，可能是一个要求JSON输出的复杂任务...",
    "这是第五个...",
    "这是第六个...",
    "这是第七个...",
    "这是第八个...",
    "这是第九个...",
    "这是第十个，它可能是一个优化版的元指令..."
  ]
}
```
---

**## 指令执行 (Execution)**
现在，请严格遵守以上所有规则——尤其是复杂的思考过程和极简的JSON输出格式——为我提供的以下“种子指令”生成10个全新的指令。

**种子指令:**

"""

def get(query, idx):
    openai_api_key = "EMPTY"
    # if idx % 2 == 0:
    #     openai_api_base = 'http://10.16.80.154:8027/v1'
    # else:
    #     openai_api_base = 'http://10.16.80.154:3377/v1'
    # openai_api_base = random.choice(["http://10.16.80.154:8027/v1", "http://10.16.80.154:8027/v1"])
    # openai_api_base = 

    # openai_api_base = 'http://10.16.80.140:8027/v1'
    if idx % 2 == 0:
        openai_api_base = 'http://10.16.80.150:8032/v1'
    else:
        openai_api_base = 'http://10.16.80.150:8031/v1'
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
                {"role": "user", "content": prompt2 + query['prompt']},
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
    return {'id': idx, 'source': query['source'], 'prompt': query['prompt'], 'output_rw' : content, 'score': query['score'], 'finish_reason' : finish_reason, 'output_tokens': output_len}
    # return {'id': idx, 'instruction':query, 'input': '', 'output' : content, 'finish_reason' : finish_reason, 'output_tokens': output_len}
    # return {'id': idx, 'instruction':query['instruction'], 'choice': query['output'], 'reject' : content, 'finish_reason' : finish_reason, 'output_tokens': output_len}


def get_answer_from_rw_livebench_querys(path, start_index=0):
    # path = '/mnt/code/zhaoxudong03/data_pipelines/training_data_zxd/embedding_dep_thred_90.jsonl'
    cur_lst = path.split('/')
    save_path = '/'.join(cur_lst[:-2]) + '/res_' + cur_lst[-2]  
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path = os.path.join(save_path, cur_lst[-1])
    questions = []


    with open(path, 'r', encoding='utf8') as f:
        for line in f.readlines():
            sample = json.loads(line)
            # query = sample['prompt']
            questions.append(sample)
        
    cur_datas = questions
    print(len(cur_datas))
    # random.shuffle(cur_datas)
    if start_index:
        cur_datas = cur_datas[start_index:]

    # cur_datas = cur_datas[:100]
    print(len(cur_datas))
    # cur_datas = cur_datas[:200]

    for idx in range(0, len(cur_datas), 5000):
        start = idx
        end = min(start+5000, len(cur_datas))
        cur_ = cur_datas[start: end]
        
        results = []
        with ThreadPoolExecutor(max_workers=300) as executor:  # 可调整线程数
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
    
    # get_answer_from_rw_livebench_querys('/mnt/code/zhaoxudong03/data_pipelines/training_data_zxd/step2/clarity_over_8_to_10_and_cg_over_8_to_10/Data_Analysis.jsonl')
    get_answer_from_rw_livebench_querys('/mnt/code/zhaoxudong03/datas/LeetCodeDataset/LeetCodeDataset-train.jsonl')
    