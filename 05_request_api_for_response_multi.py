import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from openai import OpenAI
from tqdm import tqdm


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


def dump_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        for record in records:
            file_obj.write(json.dumps(record, ensure_ascii=False) + "\n")


def is_multi_turn(messages: List[Dict[str, Any]]) -> bool:
    user_turns = sum(1 for message in messages if message.get("role") == "user")
    return user_turns > 1


def parse_extra_body(extra_body_json: Optional[str]) -> Optional[Dict[str, Any]]:
    if not extra_body_json:
        return None
    return json.loads(extra_body_json)


def request_assistant_reply(
    client: OpenAI,
    model: str,
    history: List[Dict[str, str]],
    temperature: float,
    top_p: float,
    max_tokens: int,
    extra_body: Optional[Dict[str, Any]],
) -> str:
    request_kwargs: Dict[str, Any] = {
        "model": model,
        "messages": history,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    if extra_body is not None:
        request_kwargs["extra_body"] = extra_body

    completion = client.chat.completions.create(**request_kwargs)
    message = completion.choices[0].message
    return message.content or ""


def regenerate_messages(
    messages: List[Dict[str, Any]],
    client: OpenAI,
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    extra_body: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    regenerated_messages: List[Dict[str, Any]] = []

    for message in messages:
        role = message.get("role")
        if role != "assistant":
            regenerated_messages.append(dict(message))
            continue

        history = [
            {"role": item.get("role", ""), "content": item.get("content", "")}
            for item in regenerated_messages
        ]
        new_content = request_assistant_reply(
            client=client,
            model=model,
            history=history,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            extra_body=extra_body,
        )
        new_message = {
            "role": "assistant",
            "content": new_content,
        }
        regenerated_messages.append(new_message)

    return regenerated_messages


def process_record(
    record: Dict[str, Any],
    client: OpenAI,
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    extra_body: Optional[Dict[str, Any]],
    skip_non_multi_turn: bool,
) -> Dict[str, Any]:
    messages = record.get("messages", [])
    if skip_non_multi_turn and not is_multi_turn(messages):
        return dict(record)

    new_record = dict(record)
    new_record["messages"] = regenerate_messages(
        messages=messages,
        client=client,
        model=model,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        extra_body=extra_body,
    )
    return new_record


def regenerate_dataset(
    input_path: Path,
    output_path: Path,
    error_output_path: Optional[Path],
    client: OpenAI,
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    extra_body: Optional[Dict[str, Any]],
    skip_non_multi_turn: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if error_output_path is not None:
        error_output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as output_file:
        error_file = (
            error_output_path.open("w", encoding="utf-8")
            if error_output_path is not None
            else None
        )
        try:
            for index, record in enumerate(
                tqdm(iter_jsonl(input_path), desc="Regenerating dialogs"),
                start=1,
            ):
                try:
                    processed_record = process_record(
                        record=record,
                        client=client,
                        model=model,
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_tokens,
                        extra_body=extra_body,
                        skip_non_multi_turn=skip_non_multi_turn,
                    )
                    output_file.write(json.dumps(processed_record, ensure_ascii=False) + "\n")
                except Exception as exc:
                    if error_file is None:
                        continue
                    error_record = {
                        "record_index": index,
                        "error": str(exc),
                        "record": record,
                    }
                    error_file.write(json.dumps(error_record, ensure_ascii=False) + "\n")
        finally:
            if error_file is not None:
                error_file.close()



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate assistant responses for multi-turn dialog JSONL data via OpenAI-compatible API."
    )
    parser.add_argument("--input", required=True, help="Input JSONL path.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--error-output", help="Optional JSONL path for failed records.")
    parser.add_argument("--model", required=True, help="Model name used by the OpenAI-compatible server.")
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible base URL, e.g. http://host:8000/v1.")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        help="API key for the OpenAI-compatible server. Defaults to OPENAI_API_KEY or EMPTY.",
    )
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument(
        "--extra-body-json",
        help='Optional JSON string passed to extra_body, e.g. \'{"top_k": 20}\'.',
    )
    parser.add_argument(
        "--process-all",
        action="store_true",
        help="Also process non-multi-turn records. By default only multi-turn records are regenerated.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    regenerate_dataset(
        input_path=Path(args.input),
        output_path=Path(args.output),
        error_output_path=Path(args.error_output) if args.error_output else None,
        client=client,
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        extra_body=parse_extra_body(args.extra_body_json),
        skip_non_multi_turn=not args.process_all,
    )
