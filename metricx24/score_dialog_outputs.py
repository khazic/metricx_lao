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
  output_dir: str = dataclasses.field(
      metadata={"help": "Directory for summary and per-file predictions."},
  )

  tokenizer: str | None = dataclasses.field(
      default=None,
      metadata={"help": "The tokenizer name."},
  )

  model_name_or_path: str | None = dataclasses.field(
      default=None,
      metadata={"help": "MetricX model path or Hugging Face model id."},
  )

  input_dirs: list[str] = dataclasses.field(
      default_factory=list,
      metadata={
          "help": (
              "One or more directories containing *_result.txt files to score."
          )
      },
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

  shard_index: int = dataclasses.field(
      default=0,
      metadata={"help": "Current shard index, 0-based."},
  )

  num_shards: int = dataclasses.field(
      default=1,
      metadata={"help": "Total number of shards."},
  )

  merge_only: bool = dataclasses.field(
      default=False,
      metadata={"help": "Only merge shard partial summaries."},
  )

  preprocessing_num_workers: int = dataclasses.field(
      default=4,
      metadata={"help": "Number of worker processes for dataset.map."},
  )

  dataloader_num_workers: int = dataclasses.field(
      default=4,
      metadata={"help": "Number of worker processes for the dataloader."},
  )

  bf16: bool = dataclasses.field(
      default=False,
      metadata={"help": "Load and run the model in bfloat16 on supported GPUs."},
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
    is_qe: bool,
    preprocessing_num_workers: int,
):
  """Builds a tokenized dataset compatible with Trainer.predict."""
  preprocessing_num_workers = max(1, preprocessing_num_workers)

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
  map_kwargs = {"num_proc": preprocessing_num_workers}
  ds = ds.map(_make_input, **map_kwargs)
  ds = ds.map(_tokenize, **map_kwargs)
  ds = ds.map(_remove_eos, **map_kwargs)
  ds.set_format(
      type="torch",
      columns=["input_ids", "attention_mask"],
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


def _shard_input_files(
    input_files: list[str], shard_index: int, num_shards: int
) -> list[str]:
  sorted_files = sorted(
      input_files,
      key=lambda input_file: os.path.getsize(input_file),
      reverse=True,
  )
  shards = [[] for _ in range(num_shards)]
  shard_sizes = [0 for _ in range(num_shards)]
  for input_file in sorted_files:
    target_shard = min(range(num_shards), key=lambda idx: shard_sizes[idx])
    shards[target_shard].append(input_file)
    shard_sizes[target_shard] += os.path.getsize(input_file)
  return sorted(shards[shard_index])


def _partials_dir(output_dir: str) -> str:
  return os.path.join(output_dir, "partials")


def _partial_summary_path(
    output_dir: str, shard_index: int, num_shards: int
) -> str:
  filename = f"summary_shard_{shard_index:05d}_of_{num_shards:05d}.json"
  return os.path.join(_partials_dir(output_dir), filename)


def _write_json(output_file: str, payload: Any) -> None:
  dirname = os.path.dirname(output_file)
  if dirname:
    os.makedirs(dirname, exist_ok=True)
  with open(output_file, "w", encoding="utf-8") as out:
    json.dump(payload, out, ensure_ascii=False, indent=2)


def _summarize_by_dir(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
  summary_by_dir = {}
  for item in summary:
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

  summary_by_dir_items = []
  for item in summary_by_dir.values():
    item["mean_score"] = item["weighted_score_sum"] / item["num_examples"]
    del item["weighted_score_sum"]
    summary_by_dir_items.append(item)
  return sorted(summary_by_dir_items, key=lambda item: item["input_dir"])


def _write_final_summaries(output_dir: str, summary: list[dict[str, Any]]) -> None:
  summary = sorted(summary, key=lambda item: item["input_file"])
  summary_path = os.path.join(output_dir, "summary.json")
  summary_by_dir_path = os.path.join(output_dir, "summary_by_dir.json")
  _write_json(summary_path, summary)
  _write_json(summary_by_dir_path, _summarize_by_dir(summary))
  print(f"Saved summary to {summary_path}")
  print(f"Saved directory summary to {summary_by_dir_path}")


def _merge_partial_summaries(output_dir: str, num_shards: int) -> None:
  merged_summary = []
  missing = []
  for shard_index in range(num_shards):
    partial_path = _partial_summary_path(output_dir, shard_index, num_shards)
    if not os.path.exists(partial_path):
      missing.append(partial_path)
      continue
    with open(partial_path, "r", encoding="utf-8") as f:
      merged_summary.extend(json.load(f))

  if missing:
    raise ValueError(
        "Missing shard partial summaries: " + ", ".join(missing)
    )

  _write_final_summaries(output_dir, merged_summary)


def main() -> None:
  parser = transformers.HfArgumentParser(Arguments)
  (args,) = parser.parse_args_into_dataclasses()

  os.makedirs(args.output_dir, exist_ok=True)
  details_dir = os.path.join(args.output_dir, "details")
  os.makedirs(details_dir, exist_ok=True)

  if args.num_shards < 1:
    raise ValueError("--num_shards must be at least 1.")
  if args.shard_index < 0 or args.shard_index >= args.num_shards:
    raise ValueError("--shard_index must be in [0, num_shards).")

  if args.merge_only:
    _merge_partial_summaries(args.output_dir, args.num_shards)
    return

  if not args.tokenizer:
    raise ValueError("--tokenizer is required unless --merge_only is set.")
  if not args.model_name_or_path:
    raise ValueError(
        "--model_name_or_path is required unless --merge_only is set."
    )
  if not args.input_dirs:
    raise ValueError("--input_dirs is required unless --merge_only is set.")

  if torch.cuda.is_available():
    device = torch.device("cuda")
    per_device_batch_size = args.batch_size
  else:
    device = torch.device("cpu")
    per_device_batch_size = args.batch_size

  tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer)
  model_dtype = (
      torch.bfloat16 if args.bf16 and torch.cuda.is_available() else "auto"
  )
  model = models.MT5ForRegression.from_pretrained(
      args.model_name_or_path, torch_dtype=model_dtype
  )
  model.to(device)
  model.eval()

  training_args = transformers.TrainingArguments(
      output_dir=args.output_dir,
      per_device_eval_batch_size=per_device_batch_size,
      dataloader_pin_memory=torch.cuda.is_available(),
      dataloader_num_workers=args.dataloader_num_workers,
      bf16=args.bf16 and torch.cuda.is_available(),
  )
  trainer = transformers.Trainer(
      model=model,
      args=training_args,
      data_collator=transformers.DataCollatorWithPadding(tokenizer),
  )

  summary = []
  input_files = _get_input_files(args.input_dirs, args.file_pattern)
  if not input_files:
    raise ValueError("No input files matched the given directories and file pattern.")
  input_files = _shard_input_files(input_files, args.shard_index, args.num_shards)
  print(
      f"Shard {args.shard_index}/{args.num_shards} processing "
      f"{len(input_files)} files."
  )

  for input_file in input_files:
    examples = _load_examples(input_file, args.prompt_prefixes)
    ds = _build_dataset(
        examples,
        tokenizer,
        args.max_input_length,
        args.qe,
        args.preprocessing_num_workers,
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
  partial_summary_path = _partial_summary_path(
      args.output_dir, args.shard_index, args.num_shards
  )
  _write_json(partial_summary_path, summary)

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
  print(f"Saved shard summary to {partial_summary_path}")

  if args.num_shards == 1:
    _write_final_summaries(args.output_dir, summary)


if __name__ == "__main__":
  main()
