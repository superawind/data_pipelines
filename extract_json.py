# import json
# import re

path = '/code/zhaoxudong03/data_pipelines/training_data_zxd/totals_datas_cls.jsonl'

# with open(path, 'r', encoding='utf8') as f:
#     for line in f.readlines():
#         cur_ = json.loads(line)
#         cur_out = cur_['output']
#         re.findall('')


import json
import re
from datasets import load_dataset

def extract_json_from_string(text):
    """
    使用正则表达式从字符串中宽松地提取 JSON 字符串。
    这个正则表达式会查找以 '{' 开头，以 '}' 结尾，并且中间可以包含任何字符（包括换行符）的模式。
    """
    # 宽松的 JSON 匹配，考虑多行
    # re.DOTALL 使 '.' 匹配包括换行符在内的所有字符
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        json_str = match.group(0)
        try:
            # 尝试解析提取到的字符串，确保它是有效的 JSON
            json.loads(json_str)
            return json_str
        except json.JSONDecodeError:
            # 如果不是有效的 JSON，则返回 None
            return None
    return None

def process_jsonl_file():
    """
    处理 JSONL 文件，提取 JSON 数据并返回一个列表。
    同时记录提取成功和失败的数量。
    """
    extracted_data = []
    total_lines = 0
    successful_extractions = 0
    failed_extractions = 0

    with open(path, 'r', encoding='utf8') as f:
        for line in f.readlines():
            cur_ = json.loads(line)
            line = cur_['output']

            extracted_json_str = extract_json_from_string(line)
            if extracted_json_str:
                try:
                    extracted_data.append(json.loads(extracted_json_str))
                    successful_extractions += 1
                except json.JSONDecodeError as e:
                    print(f"警告: 第 {line_num} 行提取的字符串不是有效的JSON，忽略。错误: {e}")
                    failed_extractions += 1
            else:
                failed_extractions += 1

    return extracted_data, total_lines, successful_extractions, failed_extractions

def calculate_accuracy(total_attempts, successful_attempts):
    """
    计算提取的准确率。
    """
    if total_attempts == 0:
        return 0.0
    return (successful_attempts / total_attempts) * 100

def test_datas(path):
    datas = load_dataset('parquet', data_files = [path])
    print(datas['train'][34891])

if __name__ == "__main__":
    # 示例 JSONL 文件内容 (请将此保存为 your_data.jsonl)
    # 注意：为了演示，这里直接创建文件。实际使用时，请确保文件存在。
    results = process_jsonl_file()
    print(results[0][0])
    print(results[2])
    # test_datas('/code/zhaoxudong03/datas/DistilQwen_1M/data/train-00008-of-00011.parquet')