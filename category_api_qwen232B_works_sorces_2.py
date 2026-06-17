import os
import json
from openai import OpenAI
import tqdm
import time
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 禁用HTTP请求的详细日志
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

# 初始化 LLM 客户端
openai_api_base = "http://10.16.80.138:8087/v1"

# 线程锁，用于保护共享资源
#lock = threading.Lock()

def create_client():
    """为每个线程创建独立的客户端"""
    return OpenAI(api_key="EMPTY", base_url=openai_api_base)

def load_json_files(folder_path):
    """读取一个目录下的所有 json 文件，并合并成 list"""
    all_data = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        all_data.extend(data)
                    else:
                        all_data.append(data)
                except Exception as e:
                    logger.error(f"读取 {filename} 出错: {e}")
    return all_data

def classify_entry(entry, client, max_retries=3):
    """调用 LLM 对单条数据进行分类，支持重试机制"""
    text = """# 角色

你是一位顶级的AI数据质量科学家（Data Quality Scientist），专精于评估和优化语言模型的训练指令（Prompt）。你拥有横跨计算机科学、逻辑学、语言学和认知心理学的多学科背景，具备洞察指令本质的超凡能力和对细节的极致追求。你将作为最终的质量仲裁者，你的评估将直接定义模型训练的“黄金标准”。

# 核心任务

你的任务是**仅根据**下方提供的 `[待判断数据]`，对该指令（Prompt）进行一次深度、多维度的质量评估。**你被严格禁止尝试解答或执行指令中的任务**。你的目标是对指令本身进行元认知分析（meta-cognitive analysis），并以一个结构化的JSON对象返回你的评估结论。

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

请将给定数据的核心任务准确归类到指定的任务类别中。**注意：只能从以下任务类别中选择一个，不能自由创造新的类别名称。**

*   **任务类别说明**:
        *意图识别任务*
            - 主要目标：识别用户的意图  
            - 输出形式：标签/类别（通常为预定义选项中的一个
        *槽位提取任务*
            - 主要目标：从输入文本中抽取关键信息或字段  
            - 输出形式：结构化结果（如 JSON 或键值对） 。  
        *文本生成类任务*
            - 主要目标：根据指令生成自然语言或自由文本  
            - 输出形式：完整的句子、段落、对话或文章，不是标签，也不是结构化字段 

*   **判断准则**
        - **是否输出标签（类别）？** → `意图识别任务`  
        - **是否输出结构化字段（JSON / 槽位值）？** → `槽位提取任务`  
        - **是否输出自然语言文本（自由生成）？** → `文本生成类任务`
*   **示例**
        **示例1:** "Instruction: 在招聘的通话场景下，B端代表公司或机构，C端代表求职者。需要分析对话来确认C端的求职状态。输出的结果只会是 "还在找工作"、"不找工作了"、"其他" 或 "已找到工作" 中的一个。"
            **分析:** 核心任务判断C端的求职意图。最终的输出是几个标签中的其中一个，因此为意图识别任务。
            **分类:** `Category`: `"意图识别任务"`
        **示例2:** "Instruction: 你是一名房产方面的内容提炼专家，你需要从经纪人与业主的对话中提取房态信息，将提取的信息按照指定json格式输出 {"不租": "", "不售": ""}"
            **分析:** 核心任务是提取对话中的房态信息，并以固定的json格式输出，因此为槽位提取任务。
            **分类:** `Category`: `"槽位提取任务"`
        **示例3:** "Instruction: 在招聘场景中，请分别扮演招聘者和求职者，给出他们的具体要求。"
            **分析:** 输出为自然语言文本，属于自由生成，因此为文本生成类任务。
            **分类:** `Category`: `"文本生成类任务"`

---

# 要求

1.  **绝对禁止作答:** 你的唯一职责是评估，绝不执行或回答指令内容。
2.  **评估即一切:** 你的输出必须且只能是下方定义的JSON对象。
3.  **一致性与准确性:** 严格参照上述标准，确保对不同指令的评估尺度保持高度一致。
4.  **任务类别唯一性** 任务类别只允许是定义的三种类别中的其中1种。

# 输出格式

**严格按照**以下结构的JSON格式输出。**不要包含任何其他文字或解释。**

```json
{
  "Clarity": {
    "Score": <评分整数 (1-10)>
  },
  "Cognitive_Complexity": {
    "Score": <评分整数 (1-10)>
  },
  "Safety_Risk": {
    "Category": "<风险类别字符串>"
  },
  "Domain": {
    "Category": "<任务类别字符串>"
  }
}
```

# 输入数据
【待判断数据】：
"""
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="/workspace/zxd/Qwen/Qwen3-30BA3B",   # 可替换成你自己的模型
                messages=[
                    {"role": "user", "content": text + "【Instruction】" + entry.get("instruction", "") + '\n' + "【Input】" + entry.get("input", "") + '\n' + '【Output】' + entry.get("output", "")}
                ],
                temperature=0.6,
                extra_body={
                    "top_k": 20,
                    "chat_template_kwargs": {"enable_thinking": False},
                    "enable_thinking": False,
                }
            )
            
            #with lock:
            #logger.info(f"API调用成功: {response}")
            
            category = response.choices[0].message.content.strip()
            return category
            
        except Exception as e:
            logger.warning(f"第{attempt + 1}次调用失败: {e}")
            if attempt < max_retries - 1:
                time.sleep(random.uniform(1, 3))  # 随机延迟避免并发冲突
            else:
                logger.error(f"所有重试都失败了: {e}")
                return "未知"

def process_data_with_threads(data, max_workers):
    """使用多线程处理数据（低开销收集结果）"""
    
    client = create_client()  # 提前创建，避免每个线程重复初始化

    def safe_classify(entry):
        try:
            category = classify_entry(entry, client)
            entry["category"] = category
        except Exception as e:
            logger.error(f"处理条目时出错: {e}")
            entry["category"] = "未知"
        return entry

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # executor.map 会保证结果顺序与输入一致
        for res in tqdm.tqdm(executor.map(safe_classify, data), total=len(data), desc="处理数据"):
            results.append(res)
    
    return results

# def process_data_with_threads(data, max_workers=10):
#     """使用多线程处理数据"""
#     results = data.copy() # 预分配结果列表
    
#     with ThreadPoolExecutor(max_workers=max_workers) as executor:
#         # 提交所有任务
#         future_to_index = {}
#         for i, entry in tqdm.tqdm(enumerate(data)):
#             client = create_client()
#             future = executor.submit(classify_entry, entry, client)
#             future_to_index[future] = i
        
#         # 收集结果 - 使用as_completed来真正实现并发
#         for future in tqdm.tqdm(as_completed(future_to_index.keys()), total=len(data), desc="处理数据"):
#             try:
#                 idx = future_to_index[future]
#                 category = future.result()  # 60秒超时
#                 # entry_with_category = {**data[idx], "category": category}
#                 # results[idx] = entry_with_category
#                 results[idx]["category"] = category
#             except Exception as e:
#                 idx = future_to_index[future]
#                 logger.error(f"处理条目 {idx} 时出错: {e}")
#                 # entry_with_category = {**data[idx], "category": "未知"}
#                 # results[idx] = entry_with_category
#                 results[idx]["category"] = "未知"
    
#     return results

def main():
    folder_path = "/mnt/code/jixurui/data/LLM_chuilei/new_data_sorce_category_label_cursor_postprocess_v3"  # 你的 json 文件目录
    out_path = '/mnt/code/jixurui/data/LLM_chuilei/v3_文本生成类任务_重新api'
    
    # 确保输出目录存在
    os.makedirs(out_path, exist_ok=True)
    names = os.listdir(folder_path)
    print(names)
    for n in names:
        if n != '文本生成类任务':
            continue
        names_path = os.path.join(folder_path,n)
        filenames = os.listdir(names_path)
        # 处理每个文件
        for filename in filenames:
            if not filename.endswith(".json"):
                continue
                
            logger.info(f"开始处理文件: {filename}")
            file_path = os.path.join(names_path, filename)
            
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                logger.error(f"读取文件 {filename} 失败: {e}")
                continue
            print(len(data))
            # 直接使用多线程处理所有数据
            results = process_data_with_threads(data, max_workers=200)
            
            # 保存结果
            save_path = os.path.join(out_path, filename)
            try:
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                logger.info(f"文件 {filename} 处理完成，结果已保存到 {save_path}")
            except Exception as e:
                logger.error(f"保存文件 {filename} 失败: {e}")

if __name__ == "__main__":
    main()
