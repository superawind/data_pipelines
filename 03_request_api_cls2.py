import os
import json
import time
import random

from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

prompt = """# 角色

你是一位顶级的AI数据质量科学家（Data Quality Scientist），专精于评估和优化语言模型的训练指令（Prompt）。你拥有横跨计算机科学、逻辑学、语言学和认知心理学的多学科背景，具备洞察指令本质的超凡能力和对细节的极致追求。你将作为最终的质量仲裁者，你的评估将直接定义模型训练的“黄金标准”。

# 核心任务

你的任务是**仅根据**下方提供的 `[待判断数据]`，对该指令（Prompt）进行一次深度、多维度的质量评估。**你被严格禁止尝试解答或执行指令中的任务**。你的目标是对指令本身进行元认知分析（meta-cognitive analysis），并以一个结构化的JSON对象返回你的评估结论和理由。

# 评估框架

你必须严格遵循以下四个核心维度进行评估。每个维度都包含通用定义、评分指南，以及针对特定领域的**细化判断标准**。

---

## **维度一：指令清晰度 (Clarity)**

评估指令的明确性和可执行性。一个高质量的指令应让一个合格的AI模型在无需任何猜测或追问的情况下，就能准确理解任务的目标、范围、约束和交付成果的格式。

*   **通用判断准则:**
    *   是否存在模糊词汇（如“差不多”、“更好一点”）？
    *   代词指代（如“它”、“那个”）是否清晰无歧义？
    *   是否提供了所有必要的上下文、输入数据或背景信息？
    *   任务的边界和约束条件（如字数、风格、语言、禁止做什么）是否明确？

*   **针对特定领域的细化指南:**
    *   **数学/推理:**
        *   **高清晰度表现：** 所有变量定义清晰、问题陈述无歧义、已知条件和求解目标明确分离、是否需要步骤有说明。
        *   **低清晰度表现：** 变量指代不明（“x和y的和...”但未定义x,y）、问题存在逻辑漏洞或多种解读方式。
    *   **代码:**
        *   **高清晰度表现：** 明确编程语言、函数/类名、输入输出格式（及示例）、依赖库、算法要求或性能限制。
        *   **低清晰度表现：** “写个排序函数”（未指定语言、排序算法、数据类型）、环境依赖描述不清。
    *   **创意写作:**
        *   **高清晰度表现：** 明确文体、主题、角色设定、口吻/情绪、情节要点、字数、目标读者。
        *   **低清晰度表现：** “写个故事”（过于宽泛）、“让文案更吸引人”（主观且无标准）。
    *   **数据分析:**
        *   **高清晰度表现：** 明确数据源的结构（如CSV列名）、分析目标（如“找出Q3销售额最高的产品”）、需使用的统计方法、可视化图表类型。
        *   **低清晰度表现：** “分析一下这份数据”（目标不明）、“看看有什么趋势”（范围太广）。

*   **评分标准 (1-10分):**
    *   **9-10 (极致清晰):** 指令如同一份完美的、可直接执行的工程规范。所有必要信息一应俱全，无任何歧义，执行者完全无需猜测。
    *   **7-8 (高度清晰):** 核心任务和主要约束非常明确。可能缺少一两个次要细节，但不影响产出物的核心价值和正确性。
    *   **5-6 (基本可用):** 能够理解大致意图，但关键信息存在缺失（如角色、格式），模型需要进行合理的、高概率的猜测才能完成。
    *   **3-4 (模糊不清):** 指令存在严重歧义或关键信息缺失，导致模型可能产出多个方向截然不同的结果，无法稳定满足用户意图。
    *   **1-2 (无法理解/执行):** 指令自相矛盾、逻辑不通，或依赖于完全不存在的前提。

---

## **维度二：认知复杂度 (Cognitive_Complexity)**

评估完成指令所需认知资源的深度和广度，包括推理、创造、分析、多步规划等高级认知能力。

*   **通用判断准则:**
    *   任务是简单的信息检索（“是什么”），还是需要深度的分析与创造（“如何设计”、“评价其影响”）？
    *   是否需要模型进行多步骤、有逻辑依赖的推理？
    *   是否要求模型综合运用跨领域的知识？
    *   是否要求模型产出具有原创性或深刻洞见的观点？

*   **针对特定领域的细化指南:**
    *   **数学/推理:**
        *   **高复杂度表现：** 证明复杂定理、解决非标准或开放性问题、多步逻辑谜题、需要构建数学模型。
        *   **低复杂度表现：** 执行标准运算、查询公式、简单代数替换。
    *   **代码:**
        *   **高复杂度表现：** 设计新算法、进行代码重构与优化、解决复杂的系统设计问题、调试深层逻辑错误。
        *   **低复杂度表现：** 编写样板代码、翻译语言语法、查询API用法。
    *   **创意写作:**
        *   **高复杂度表现：** 模仿特定作家的微妙风格、构建多线索的复杂情节、表达深刻的哲学思辨、创造全新的世界观。
        *   **低复杂度表现：** 按照模板填充内容（如通用节日祝福）、简单地续写一句话。
    *   **数据分析:**
        *   **高复杂度表现：** 进行预测性建模、多变量的因果推断、设计A/B测试方案、从非结构化数据中提取洞察。
        *   **低复杂度表现：** 计算基本统计量（平均值、中位数）、根据已有规则进行数据筛选。

*   **评分标准 (1-10分):**
    *   **9-10 (专家级创造/战略):** 要求进行高度原创的综合性工作，如制定复杂战略、进行前沿科学推演、设计复杂的系统架构。
    *   **7-8 (深度分析/推理):** 要求进行严密的逻辑推导、因果分析、多角度对比评估、解决复杂的多步骤问题。
    *   **5-6 (信息整合/应用):** 要求整合多源信息、遵循多步指令完成任务、应用已知框架或模型进行内容生成。
    *   **3-4 (简单应用/转述):** 对已有信息进行简单的格式转换、分类、摘要或按固定套路回答。
    *   **1-2 (直接检索/复述):** 任务仅需从知识库中提取单一、孤立的事实。

---

## **维度三：安全性与风险 (Safety_Risk)**

评估指令本身或其潜在产出是否触及安全红线、道德伦理底线，或可能导致有害后果。

*   **判断准则:**
    *   指令是否直接或间接鼓励、指导或询问非法行为或不道德活动？
    *   是否强化有害的社会偏见、传播仇恨言论或歧视？
    *   是否包含或索要真实、敏感的个人身份信息（PII）？
    *   是否可能引导模型产生危险、有害（如医疗、金融方面）的错误建议？

*   **风险分类 (单选):**
    *   `"安全 (Safe)"`: 指令中立、无害。
    *   `"风险-违法犯罪 (Illegal_Activity)"`: 教唆、询问或描述非法活动。
    *   `"风险-仇恨/歧视 (Hate_Speech_Discrimination)"`: 针对特定群体的攻击、侮辱或负面刻板印象。
    *   `"风险-暴力/自残 (Violence_Self_Harm)"`: 描述、美化或鼓励暴力、血腥、自残、虐待行为。
    *   `"风险-有害建议 (Harmful_Advice)"`: 可能导致现实世界伤害的建议，如不专业的医疗、财务、法律建议。
    *   `"风险-隐私泄露 (Privacy_Violation)"`: 指令本身包含或索要真实PII。
    *   `"风险-事实错误引导 (Factual_Misdirection)"`: 指令基于一个有害的、错误的前提，并要求模型在此基础上展开。

---

## **维度四：任务领域 (Domain)**

将指令的核心任务归类到最精准的应用领域。分类采用**一个主领域（Primary）**和**零个或多个次要领域（Secondary）**的结构。

*   **主次领域判断原则:**
    *   **主领域 (Primary_Domain):** 代表指令的**核心任务**或**最终产出物的本质**。问自己：“模型最根本的任务是‘做什么’或‘生成什么’？”
    *   **次要领域 (Secondary_Domains):** 代表完成任务所需的**辅助知识、背景、风格或约束**。这些是完成核心任务的“上下文”或“附加要求”。

*   **判断准则与示例:**
    *   **示例1:** `"用Python为我分析这份CSV销售数据，并找出同比增长最快的三个产品类别。"`
        *   **分析:** 核心任务是编写Python代码来执行分析。最终产出物是代码及代码运行的结果。数据分析是其应用场景和目的。
        *   **分类:** `Primary_Domain`: `"Code"`, `Secondary_Domains`: `["Data_Analysis", "Business_&_Finance"]`
    *   **示例2:** `"请扮演一位严厉的健身教练，为我这个体重80公斤、希望在三个月内减脂的男性，制定一个详细的为期四周的训练和饮食计划。"`
        *   **分析:** 核心任务是生成一个专业的健康/健身计划。产出物的本质是健康领域的内容。“扮演教练”是指令的风格和形式要求。
        *   **分类:** `Primary_Domain`: `"Daily_Life_&_Health"`, `Secondary_Domains`: `["Role_Playing"]`
    *   **示例3:** `"请以莎士比亚的风格，重写“小红帽”的故事，并确保其中包含至少一个逻辑悖论。"`
        *   **分析:** 核心任务是进行文学创作。莎士比亚风格和逻辑悖论是重要的附加约束和知识要求。
        *   **分类:** `Primary_Domain`: `"Creative_Writing"`, `Secondary_Domains`: `["Humanities_Arts_&_Social_Sciences", "Reasoning_&_Logic"]`

*   **领域分类列表 (所有分类必须从此列表中选择):**
    *   `Reasoning_&_Logic`: 逻辑谜题、演绎推理、思维游戏、辩论。
    *   `Math`: 数学计算、公式推导、几何问题、数学建模。
    *   `Code`: 代码生成/调试/解释/优化、算法、软件工程。
    *   `Data_Analysis`: 数据清洗/处理、统计分析、可视化、洞察提取。
    *   `Creative_Writing`: 故事、诗歌、剧本、文案、歌词等原创文学内容。
    *   `Instruction_Following_&_Text_Processing`: 严格按指令格式化、提取、分类、摘要、翻译、改写等。
    *   `Role_Playing_&_Chat`: 扮演特定角色进行对话，或进行无特定目的的开放式闲聊。
    *   `Humanities_Arts_&_Social_Sciences`: 历史、哲学、法律、政治、社会学、艺术评论等。
    *   `Natural_&_Applied_Sciences`: 物理、化学、生物、地理、工程、医学知识问答等。
    *   `Business_&_Finance`: 市场分析、商业策略、财报解读、投资理财知识。
    *   `Daily_Life_&_Health`: 美食、旅行、运动、健康咨询、生活窍门。
    *   `General_Knowledge_QA`: 对客观事实的直接提问，无法归入以上专业领域的。
    *   `Other`: **如果一个指令的核心任务无法被明确归入任何一个主领域，则主领域应被标记为 `"Other"`**。次要领域可以为空或选择相关领域。

---

# 要求

1.  **绝对禁止作答:** 你的唯一职责是评估，绝不执行或回答指令内容。
2.  **评估即一切:** 你的输出必须且只能是下方定义的JSON对象。
3.  **理由是关键:** 对于每个评分项，必须在`Reasoning`字段中提供简洁、精确的评估理由，解释你为何给出该分数或分类。这是评估的核心部分。
4.  **一致性与准确性:** 严格参照上述标准，确保对不同指令的评估尺度保持高度一致。

# 输出格式

**严格按照**以下包含“主/次领域”结构的JSON格式输出。**不要包含任何其他文字或解释。**

```json
{
  "Clarity": {
    "Score": <评分整数 (1-10)>,
    "Reasoning": "<对此评分的简洁解释>"
  },
  "Cognitive_Complexity": {
    "Score": <评分整数 (1-10)>,
    "Reasoning": "<对此评分的简洁解释>"
  },
  "Safety_Risk": {
    "Category": "<风险类别字符串>",
    "Reasoning": "<对此分类的简洁解释，对于'安全'可简述为'指令内容中立无害'>"
  },
  "Domain": {
    "Primary_Domain": "<来自列表的单个主领域字符串>",
    "Secondary_Domains": [ "<来自列表的次要领域字符串1>", "<来自列表的次要领域字符串2>" ],
    "Reasoning": "<对此主/次领域划分的简洁解释>"
  }
}
```

# 输入数据
【待判断数据】：
"""

def read_jsonl(path):
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


def get(query, idx):
    openai_api_key = "EMPTY"
    # openai_api_base = "http://172.8.94.114:8011/v1"
    openai_api_base = "http://10.16.80.154:8027/v1"
    # openai_api_base = "http://10.16.80.9:8027/v1"

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    
    # query 传入的是字典

    try:
        completion = client.chat.completions.create(
            model="/workspace/zxd/Qwen/Qwen3-30B-A3B-Instruct-2507",
            messages=[
                # {'role': 'system', 'content': 'You are a helpful assistant.'},
                # {"role": "user", "content": query['instruction']},
                {"role": "user", "content": prompt+query['prompt']},
            ],
            # temperature=0.6,
            temperature=0.7,
            top_p=0.8,
            max_tokens=8192,
            extra_body={
                # "chat_template_kwargs": {"enable_thinking": False},
                "top_k":20
                # "separate_reasoning": True
            }
        )
        content = completion.choices[0].message.content
        finish_reason = completion.choices[0].finish_reason
        output_len = completion.usage.completion_tokens
    except Exception as e:
        print('miss an error......................', e)
        content = ''; finish_reason = 'length'; output_len = 0
    query.update({'idx':idx, 'output' : content, 'finish_reason' : finish_reason, 'output_tokens': output_len})
    return query
    # return {'id': idx, 'source': query['source'], 'prompt': query['prompt'], 'output' : content, 'finish_reason' : finish_reason, 'output_tokens': output_len}
    # return {'id': idx, 'instruction':query, 'input': '', 'output' : content, 'finish_reason' : finish_reason, 'output_tokens': output_len}
    # return {'id': idx, 'instruction':query['instruction'], 'choice': query['output'], 'reject' : content, 'finish_reason' : finish_reason, 'output_tokens': output_len}


def get_answer_from_rw_livebench_querys(path, save_path, start_index=0, end_index=None):
    questions = read_jsonl(path)
    if not end_index:
        end_index = len(questions)
        
    cur_datas = questions[start_index: end_index]
    print(len(cur_datas))

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
    # path = '/mnt/code/zhaoxudong03/data_pipelines/training_data_zxd/embedding_dep_thred_90.jsonl'
    # save_path = '/mnt/code/zhaoxudong03/data_pipelines/training_data_zxd/totals_datas_cls_2.jsonl'
    
    path = '/opt/users/zxd/wangfei_prompt_zh_02_embeded_dup_for_30b_a3b_judge.jsonl'
    save_path = '/mnt/code/zhaoxudong03/data_pipelines/training_data_zxd/wangfei_prompt_zh_03_30b_a3b_cls_after.jsonl'
    get_answer_from_rw_livebench_querys(path, save_path, start_index=4300000)
    # get_answer_from_rw_livebench_querys(path, save_path, start_index=0, end_index=10)
