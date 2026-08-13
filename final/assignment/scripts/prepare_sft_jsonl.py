#!/usr/bin/env python3
"""Validate and convert small ME344 SFT JSONL datasets."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


VALID_ROLES = {"system", "user", "assistant"}


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("input_jsonl", type=Path, help="Input JSONL file.")
  parser.add_argument("output_jsonl", type=Path, help="Output JSONL file.")
  parser.add_argument(
      "--output-schema",
      choices=("messages", "prompt_completion"),
      default="messages",
      help="Output schema to write. Default: messages.",
  )
  parser.add_argument(
      "--max-examples",
      type=int,
      default=0,
      help="Optional maximum number of examples to write. 0 means no limit.",
  )
  parser.add_argument(
      "--report",
      type=Path,
      help="Optional JSON path for a content-free dataset quality report.",
  )
  return parser.parse_args()


def require_text(value: Any, field: str, line_number: int) -> str:
  if not isinstance(value, str) or not value.strip():
    raise ValueError(f"line {line_number}: field '{field}' must be a non-empty string")
  return value.strip()


def validate_messages(messages: Any, line_number: int) -> list[dict[str, str]]:
  if not isinstance(messages, list) or not messages:
    raise ValueError(f"line {line_number}: field 'messages' must be a non-empty list")

  normalized = []
  for index, message in enumerate(messages):
    if not isinstance(message, dict):
      raise ValueError(f"line {line_number}: messages[{index}] must be an object")
    role = require_text(message.get("role"), f"messages[{index}].role", line_number)
    content = require_text(message.get("content"), f"messages[{index}].content", line_number)
    if role not in VALID_ROLES:
      raise ValueError(f"line {line_number}: messages[{index}].role must be one of {sorted(VALID_ROLES)}")
    normalized.append({"role": role, "content": content})

  turns = normalized[1:] if normalized[0]["role"] == "system" else normalized
  expected = ["user" if index % 2 == 0 else "assistant" for index in range(len(turns))]
  if not turns or [message["role"] for message in turns] != expected:
    raise ValueError(
        f"line {line_number}: use an optional leading system message, then alternating "
        "user/assistant turns ending with assistant"
    )

  return normalized


def pair_to_messages(prompt: str, completion: str) -> list[dict[str, str]]:
  return [
      {"role": "user", "content": prompt},
      {"role": "assistant", "content": completion},
  ]


def normalize_example(raw: dict[str, Any], line_number: int) -> dict[str, Any]:
  if "messages" in raw:
    messages = validate_messages(raw["messages"], line_number)
    return {
        "messages": messages,
        "prompt": "\n".join(message["content"] for message in messages if message["role"] == "user"),
        "completion": "\n".join(message["content"] for message in messages if message["role"] == "assistant"),
    }

  pair_fields = (
      ("input", "output"),
      ("prompt", "completion"),
      ("question", "answer"),
  )
  for prompt_field, completion_field in pair_fields:
    if prompt_field in raw or completion_field in raw:
      prompt = require_text(raw.get(prompt_field), prompt_field, line_number)
      completion = require_text(raw.get(completion_field), completion_field, line_number)
      return {
          "messages": pair_to_messages(prompt, completion),
          "prompt": prompt,
          "completion": completion,
      }

  raise ValueError(
      f"line {line_number}: expected messages or paired input/output, prompt/completion, or question/answer fields"
  )


def iter_examples(path: Path):
  with path.open("r", encoding="utf-8") as input_file:
    for line_number, line in enumerate(input_file, start=1):
      line = line.strip()
      if not line:
        continue
      try:
        raw = json.loads(line)
      except json.JSONDecodeError as exc:
        raise ValueError(f"line {line_number}: invalid JSON: {exc}") from exc
      if not isinstance(raw, dict):
        raise ValueError(f"line {line_number}: each JSONL row must be an object")
      yield line_number, normalize_example(raw, line_number)


def summarize_lengths(values: list[int]) -> dict[str, float | int]:
  """Return content-free distribution landmarks for plotting and review."""
  ordered = sorted(values)
  last = len(ordered) - 1
  return {
      "min": ordered[0],
      "p25": ordered[round(last * 0.25)],
      "median": ordered[round(last * 0.50)],
      "p75": ordered[round(last * 0.75)],
      "max": ordered[-1],
      "mean": sum(ordered) / len(ordered),
  }


def main() -> int:
  args = parse_args()
  if args.max_examples < 0:
    print("ERROR: --max-examples must be non-negative", file=sys.stderr)
    return 2
  if args.input_jsonl.resolve() == args.output_jsonl.resolve():
    print("ERROR: input and output JSONL paths must differ", file=sys.stderr)
    return 2
  if args.report and args.report.resolve() == args.output_jsonl.resolve():
    print("ERROR: report and output JSONL paths must differ", file=sys.stderr)
    return 2
  if args.report and args.report.resolve() == args.input_jsonl.resolve():
    print("ERROR: report and input JSONL paths must differ", file=sys.stderr)
    return 2

  args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

  count = 0
  role_counts: Counter[str] = Counter()
  prompt_lengths: list[int] = []
  completion_lengths: list[int] = []
  seen_prompts: set[str] = set()
  duplicate_prompts = 0
  try:
    with args.output_jsonl.open("w", encoding="utf-8") as output_file:
      for _, example in iter_examples(args.input_jsonl):
        if args.max_examples and count >= args.max_examples:
          break
        if args.output_schema == "messages":
          row = {"messages": example["messages"]}
        else:
          row = {"prompt": example["prompt"], "completion": example["completion"]}
        output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
        role_counts.update(message["role"] for message in example["messages"])
        prompt_lengths.append(len(example["prompt"]))
        completion_lengths.append(len(example["completion"]))
        if example["prompt"] in seen_prompts:
          duplicate_prompts += 1
        seen_prompts.add(example["prompt"])
        count += 1
  except (OSError, ValueError) as exc:
    print(f"ERROR: {exc}", file=sys.stderr)
    return 1

  if count == 0:
    print("ERROR: no examples were written", file=sys.stderr)
    return 1

  print(f"Wrote {count} examples to {args.output_jsonl}")

  if args.report:
    try:
      output_sha256 = hashlib.sha256(args.output_jsonl.read_bytes()).hexdigest()
      report = {
          "created_at_utc": datetime.now(timezone.utc).isoformat(),
          "input_path": str(args.input_jsonl),
          "output_path": str(args.output_jsonl),
          "output_sha256": output_sha256,
          "output_schema": args.output_schema,
          "examples_written": count,
          "max_examples": args.max_examples or None,
          "message_role_counts": dict(sorted(role_counts.items())),
          "duplicate_prompt_count": duplicate_prompts,
          "prompt_characters": summarize_lengths(prompt_lengths),
          "completion_characters": summarize_lengths(completion_lengths),
          "review_reminder": (
              "Before training, inspect at least 20 examples and check the dataset source, "
              "license, filtering, privacy, and possible test-set overlap."
          ),
      }
      args.report.parent.mkdir(parents=True, exist_ok=True)
      args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
      print(f"ERROR: Could not write {args.report}: {exc}", file=sys.stderr)
      return 1
    print(f"Wrote {args.report}")

  return 0


if __name__ == "__main__":
  raise SystemExit(main())
