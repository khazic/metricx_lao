# coding=utf-8
"""Batch-scores chat-style translation outputs with a MetricX-24 model.

Input files are expected to contain one JSON array per line, for example:
[
  {"role": "user", "content": "请将以下句子翻译成英语，只输出译文即可，不要输出其他内容：<source>"},
  {"role": "assistant", "content": "<hypothesis>"}
]

The script converts each line into MetricX QE input with:
  source = extracted source text from the user prompt
  hypothesis = assistant content
  reference = ""
"""

import dataclasses
import json
import os
from typing import Any

import datasets
from metricx24 import models
import numpy as np
import torch
import transformers


DEFAULT_PROMPT_PREFIXES = [
    "请将以下句子翻译成英语，只输出译文即可，不要输出其他内容：",
]


@dataclasses.dataclass
class Arguments:
  tokenizer: str = dataclasses.field(
      metadata={"help": "The tokenizer name."},
  )

  model_name_or_path: str = dataclasses.field(
      metadata={"help": "MetricX model path or Hugging Face model id."},
  )

  input_dirs: list[str] = dataclasses.field(
      metadata={
          "help": (
              "One or more directories containing *_result.txt files to score."
          )
      },
  )

  output_dir: str = dataclasses.field(
      metadata={"help": "Directory for summary and per-file predictions."},
  )

  max_input_length: int = dataclasses.field(
      default=1536,
      metadata={"help": "Maximum input length for the model."},
  )

  batch_size: int = dataclasses.field(
      default=1,
      metadata={"help": "Global prediction batch size."},
  )

  file_pattern: str = dataclasses.field(
      default="_result.txt",
      metadata={"help": "Only files ending with this suffix will be scored."},
  )

  prompt_prefixes: list[str] = dataclasses.field(
      default_factory=lambda: list(DEFAULT_PROMPT_PREFIXES),
      metadata={
          "help": (
              "Optional prompt prefixes to strip from the user message before "
              "scoring. If none match, the script falls back to taking the "
              "text after the first colon."
          )
      },
  )

  qe: bool = dataclasses.field(
      default=True,
      metadata={"help": "Run MetricX in QE mode."},
  )


def _extract_source(user_content: str, prompt_prefixes: list[str]) -> str:
  """Extracts the source text from the user prompt."""
  for prompt_prefix in prompt_prefixes:
    if prompt_prefix and user_content.startswith(prompt_prefix):
      return user_content[len(prompt_prefix):].strip()
  if "：" in user_content:
    return user_content.split("：", 1)[1].strip()
  if ":" in user_content:
    return user_content.split(":", 1)[1].strip()
  return user_content.strip()


def _dialog_to_example(
    dialog: list[dict[str, Any]], prompt_prefixes: list[str]
) -> dict[str, str]:
  """Converts one dialog line into MetricX QE input."""
  user_turn = None
  assistant_turn = None
  for turn in dialog:
    role = turn.get("role")
    if role == "user" and user_turn is None:
      user_turn = turn
    elif role == "assistant" and assistant_turn is None:
      assistant_turn = turn

  if user_turn is None or assistant_turn is None:
    raise ValueError("Each line must contain at least one user turn and one assistant turn.")

  source = _extract_source(
      str(user_turn.get("content", "")), prompt_prefixes
  )
  hypothesis = str(assistant_turn.get("content", "")).strip()

  if not source:
    raise ValueError("Parsed source text is empty.")
  if not hypothesis:
    raise ValueError("Assistant content is empty.")

  return {
      "source": source,
      "hypothesis": hypothesis,
      "reference": "",
  }


def _load_examples(
    input_file: str, prompt_prefixes: list[str]
) -> list[dict[str, str]]:
  """Loads and converts a chat-style result file."""
  examples = []
  with open(input_file, "r", encoding="utf-8") as f:
    for line_num, line in enumerate(f, start=1):
      line = line.strip()
      if not line:
        continue
      try:
        dialog = json.loads(line)
      except json.JSONDecodeError as exc:
        raise ValueError(f"{input_file}:{line_num} is not valid JSON.") from exc
      if not isinstance(dialog, list):
        raise ValueError(f"{input_file}:{line_num} must be a JSON array.")
      try:
        examples.append(_dialog_to_example(dialog, prompt_prefixes))
      except ValueError as exc:
        raise ValueError(f"{input_file}:{line_num} parse failed: {exc}") from exc
  return examples


def _build_dataset(
    examples: list[dict[str, str]],
    tokenizer,
    max_input_length: int,
    device,
    is_qe: bool,
):
  """Builds a tokenized dataset compatible with Trainer.predict."""

  def _make_input(example):
    if is_qe:
      example["input"] = (
          "source: "
          + example["source"]
          + " candidate: "
          + example["hypothesis"]
      )
    else:
      example["input"] = (
          "source: "
          + example["source"]
          + " candidate: "
          + example["hypothesis"]
          + " reference: "
          + example["reference"]
      )
    return example

  def _tokenize(example):
    return tokenizer(
        example["input"],
        max_length=max_input_length,
        truncation=True,
        padding=False,
    )

  def _remove_eos(example):
    example["input_ids"] = example["input_ids"][:-1]
    example["attention_mask"] = example["attention_mask"][:-1]
    return example

  ds = datasets.Dataset.from_list(examples)
  ds = ds.map(_make_input)
  ds = ds.map(_tokenize)
  ds = ds.map(_remove_eos)
  ds.set_format(
      type="torch",
      columns=["input_ids", "attention_mask"],
      device=device,
      output_all_columns=True,
  )
  return ds


def _get_input_files(input_dirs: list[str], file_pattern: str) -> list[str]:
  """Collects files to score."""
  input_files = []
  for input_dir in input_dirs:
    for entry in sorted(os.listdir(input_dir)):
      if entry.endswith(file_pattern):
        input_files.append(os.path.join(input_dir, entry))
  return input_files


def _detail_filename(input_file: str) -> str:
  parent = os.path.basename(os.path.dirname(input_file))
  return f"{parent}__{os.path.basename(input_file)}.jsonl"


def main() -> None:
  parser = transformers.HfArgumentParser(Arguments)
  (args,) = parser.parse_args_into_dataclasses()

  os.makedirs(args.output_dir, exist_ok=True)
  details_dir = os.path.join(args.output_dir, "details")
  os.makedirs(details_dir, exist_ok=True)

  if torch.cuda.is_available():
    device = torch.device("cuda")
    per_device_batch_size = max(1, args.batch_size // torch.cuda.device_count())
  else:
    device = torch.device("cpu")
    per_device_batch_size = args.batch_size

  tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer)
  model = models.MT5ForRegression.from_pretrained(
      args.model_name_or_path, torch_dtype="auto"
  )
  model.to(device)
  model.eval()

  training_args = transformers.TrainingArguments(
      output_dir=args.output_dir,
      per_device_eval_batch_size=per_device_batch_size,
      dataloader_pin_memory=False,
  )
  trainer = transformers.Trainer(
      model=model,
      args=training_args,
      data_collator=transformers.DataCollatorWithPadding(tokenizer),
  )

  summary = []
  summary_by_dir = {}
  input_files = _get_input_files(args.input_dirs, args.file_pattern)
  if not input_files:
    raise ValueError("No input files matched the given directories and file pattern.")

  for input_file in input_files:
    examples = _load_examples(input_file, args.prompt_prefixes)
    ds = _build_dataset(
        examples,
        tokenizer,
        args.max_input_length,
        device,
        args.qe,
    )
    predictions, _, _ = trainer.predict(test_dataset=ds)
    scores = np.asarray(predictions, dtype=float).reshape(-1)

    detail_path = os.path.join(details_dir, _detail_filename(input_file))
    with open(detail_path, "w", encoding="utf-8") as out:
      for example, score in zip(examples, scores):
        record = dict(example)
        record["prediction"] = float(score)
        out.write(json.dumps(record, ensure_ascii=False) + "\n")

    item = {
        "input_file": input_file,
        "input_dir": os.path.dirname(input_file),
        "num_examples": len(examples),
        "mean_score": float(np.mean(scores)),
        "median_score": float(np.median(scores)),
        "min_score": float(np.min(scores)),
        "max_score": float(np.max(scores)),
        "detail_file": detail_path,
    }
    summary.append(item)

    input_dir = item["input_dir"]
    if input_dir not in summary_by_dir:
      summary_by_dir[input_dir] = {
          "input_dir": input_dir,
          "num_files": 0,
          "num_examples": 0,
          "weighted_score_sum": 0.0,
      }
    summary_by_dir[input_dir]["num_files"] += 1
    summary_by_dir[input_dir]["num_examples"] += item["num_examples"]
    summary_by_dir[input_dir]["weighted_score_sum"] += (
        item["mean_score"] * item["num_examples"]
    )

  summary_path = os.path.join(args.output_dir, "summary.json")
  with open(summary_path, "w", encoding="utf-8") as out:
    json.dump(summary, out, ensure_ascii=False, indent=2)

  summary_by_dir_path = os.path.join(args.output_dir, "summary_by_dir.json")
  summary_by_dir_items = []
  for item in summary_by_dir.values():
    item["mean_score"] = item["weighted_score_sum"] / item["num_examples"]
    del item["weighted_score_sum"]
    summary_by_dir_items.append(item)
  with open(summary_by_dir_path, "w", encoding="utf-8") as out:
    json.dump(summary_by_dir_items, out, ensure_ascii=False, indent=2)

  for item in summary:
    print(
        json.dumps(
            {
                "input_file": item["input_file"],
                "num_examples": item["num_examples"],
                "mean_score": item["mean_score"],
            },
            ensure_ascii=False,
        )
    )
  print(f"Saved summary to {summary_path}")
  print(f"Saved directory summary to {summary_by_dir_path}")


if __name__ == "__main__":
  main()
