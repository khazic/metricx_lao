#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable


PROMPT_PATTERNS = [
    re.compile(r"^请将以下句子翻译成[^：:]+[：:]\s*", re.S),
    re.compile(r"^请把以下句子翻译成[^：:]+[：:]\s*", re.S),
    re.compile(r"^请将以下文本翻译成[^：:]+[：:]\s*", re.S),
    re.compile(r"^请把以下文本翻译成[^：:]+[：:]\s*", re.S),
    re.compile(r"^Translate the following .*?:\s*", re.S | re.I),
]


def strip_prompt(text: str) -> str:
    output = text.strip()
    for pattern in PROMPT_PATTERNS:
        updated = pattern.sub("", output, count=1)
        if updated != output:
            return updated.strip()
    return output


def load_rows(path: Path) -> list[dict]:
    if path.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as e:
            raise SystemExit("Reading parquet requires pyarrow to be installed.") from e
        return pq.read_table(path).to_pylist()

    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {e}") from e
            if isinstance(obj, list):
                rows.append({"messages": obj})
            elif isinstance(obj, dict):
                rows.append(obj)
            else:
                raise SystemExit(f"{path}:{line_no}: expected JSON object or list, got {type(obj).__name__}")
    return rows


def first_message_by_role(messages: list[dict], role: str) -> dict | None:
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == role:
            return msg
    return None


def convert_row(row: dict, *, keep_prompt: bool) -> dict:
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise ValueError("missing or invalid `messages`")

    user_msg = first_message_by_role(messages, "user")
    assistant_msg = first_message_by_role(messages, "assistant")
    if user_msg is None or assistant_msg is None:
        raise ValueError("requires at least one user message and one assistant message")

    user_content = user_msg.get("content")
    assistant_content = assistant_msg.get("content")
    if not isinstance(user_content, str) or not isinstance(assistant_content, str):
        raise ValueError("user/assistant content must be strings")

    source = user_content.strip() if keep_prompt else strip_prompt(user_content)
    return {
        "source": source,
        "hypothesis": assistant_content.strip(),
        "reference": "",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert messages-format data into MetricX input JSONL.")
    parser.add_argument("--input", required=True, help="Input .txt/.jsonl/.parquet file")
    parser.add_argument("--output", required=True, help="Output MetricX jsonl path")
    parser.add_argument("--keep-prompt", action="store_true", help="Do not strip translation prompt from user content")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    rows = load_rows(input_path)
    converted = 0
    skipped = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(rows):
            try:
                record = convert_row(row, keep_prompt=args.keep_prompt)
            except ValueError as e:
                skipped += 1
                if skipped <= 10:
                    print(f"skip row {idx}: {e}")
                continue
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            converted += 1

    print(f"input={input_path}")
    print(f"output={output_path}")
    print(f"converted={converted}")
    print(f"skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
