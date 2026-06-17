import argparse
import json
from tqdm import tqdm 
from pathlib import Path
from typing import Any, Dict, Iterable, List

"""
1、将数据处理成需要的格式，source + content_keys(默认是 prompt) 字段，用于后续去重处理
2、不区分 单轮对话和多轮对话，多轮对话将全部的 role user 通过 \n 拼接到一起，参与后续去重
"""

def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file_obj:
        for line_number, line in enumerate(file_obj, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc

def extract_user_messages(messages: List[Dict[str, Any]]) -> List[str]:
    prompts: List[str] = []
    for message in messages:
        if message.get("role") == "user":
            prompts.append(message.get("content", ""))
    return prompts


def build_source(file_name: str, index: int) -> str:
    return f"{file_name}__{index}"


def convert_file(input_path: Path, single_output_path: Path, multi_output_path: Path) -> None:
    file_name = input_path.name
    single_output_path.parent.mkdir(parents=True, exist_ok=True)
    multi_output_path.parent.mkdir(parents=True, exist_ok=True)

    single_nums, multi_nums = 0, 0
    with (
        single_output_path.open("w", encoding="utf-8") as single_file,
        # multi_output_path.open("w", encoding="utf-8") as multi_file,
    ):
        for index, record in tqdm(enumerate(iter_jsonl(input_path), start=0)):
            messages = record.get("messages", [])
            user_prompts = extract_user_messages(messages)
            new_source = build_source(file_name, index)

            if len(user_prompts) <= 1:
                single_record = {
                    "source": new_source,
                    "prompt": user_prompts[0] if user_prompts else "",
                }
                single_file.write(json.dumps(single_record, ensure_ascii=False) + "\n")
                single_nums += 1
                # continue
            else:
                multi_nums += 1
                # multi_record = dict(record)
                # multi_record["source"] = new_source
                # multi_file.write(json.dumps(multi_record, ensure_ascii=False) + "\n")
                multi_record = {
                    "source": new_source,
                    "prompt": '\n'.join(user_prompts),
                }
                single_file.write(json.dumps(multi_record, ensure_ascii=False) + "\n")


    print('end::: 单轮数量{}， 多轮数量{}'.format(single_nums, multi_nums))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split dialog JSONL into single-turn and multi-turn outputs."
    )
    parser.add_argument("--input", required=True, help="Input JSONL file path.")
    parser.add_argument(
        "--single-output",
        required=True,
        help="Output JSONL path for single-turn records.",
    )
    parser.add_argument(
        "--multi-output",
        required=True,
        help="Output JSONL path for multi-turn records.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print('输入文件：：：', args.input)
    convert_file(
        Path(args.input),
        Path(args.single_output),
        Path(args.multi_output),
    )


# 运行指令 
# python /mnt/code/zhaoxudong03/data_pipelines/00_deal_data.py \
#  --input /mnt/code/zhaoxudong03/zxd_datas/sft_datas_v2/Nemotron-Cascade2/instruction_following/instruction_following.jsonl \
#  --single-output /mnt/code/zhaoxudong03/zxd_datas/sft_datas_v2/Nemotron-Cascade2-deal/IF/instruction_following_single.jsonl \
#  --multi-output /mnt/code/zhaoxudong03/zxd_datas/sft_datas_v2/Nemotron-Cascade2-deal/IF/instruction_following_multi.jsonl 


# python /mnt/code/zhaoxudong03/data_pipelines/00_deal_data.py \
#  --input /mnt/code/zhaoxudong03/zxd_datas/sft_datas_v2/Nemotron-Cascade2/safety/safety.jsonl \
#  --single-output /mnt/code/zhaoxudong03/zxd_datas/sft_datas_v2/Nemotron-Cascade2-deal/Safe/safety_single.jsonl \
#  --multi-output /mnt/code/zhaoxudong03/zxd_datas/sft_datas_v2/Nemotron-Cascade2-deal/Safe/safety_multi.jsonl 