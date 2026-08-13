from __future__ import annotations
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from pathlib import Path
import re

_RL_NUMBER = '[-+]?(?:\\d[\\d,]*)(?:\\.\\d+)?(?:[eE][-+]?\\d+)?'
_RL_ANSWER_RE = re.compile(rf"<answer>\s*({_RL_NUMBER})\s*</answer>", re.IGNORECASE)
_RL_FULL_FORMAT_RE = re.compile(
    rf"<reasoning>.+?</reasoning>\s*<answer>\s*{_RL_NUMBER}\s*</answer>\s*(?:<\|im_end\|>)?\s*$",
    re.IGNORECASE | re.DOTALL,
)

def _me344_decimal(value) -> Decimal | None:
    try:
        number = Decimal(str(value).strip().replace(",", ""))
    except (InvalidOperation, ValueError):
        return None
    return number if number.is_finite() else None


def _me344_decimal_text(value) -> str:
    number = _me344_decimal(value)
    if number is None:
        raise ValueError(f"RL answers must be finite numbers; found {value!r}.")
    text = format(number, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in {"-0", "+0"} else text


def _me344_sequence(value) -> list:
    if isinstance(value, (str, bytes)):
        return [value]
    try:
        return list(value)
    except TypeError:
        return [value]


def _me344_targets(answer, count: int) -> list[Decimal | None]:
    values = _me344_sequence(answer)
    if len(values) == 1:
        values *= count
    targets = []
    for raw in values[:count]:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        try:
            decoded = json.loads(str(raw))
        except json.JSONDecodeError:
            decoded = raw
        if isinstance(decoded, list):
            decoded = decoded[0] if decoded else None
        targets.append(_me344_decimal(decoded))
    return targets + [None] * (count - len(targets))


def _me344_prediction(completion: str) -> Decimal | None:
    tagged = _RL_ANSWER_RE.findall(str(completion))
    if tagged:
        return _me344_decimal(tagged[-1])
    numbers = re.findall(_RL_NUMBER, str(completion))
    return _me344_decimal(numbers[-1]) if numbers else None


def _me344_log_reward(name: str, prompts, completions, answer, scores: list[float]) -> None:
    path = os.environ.get("ME344_RL_REWARD_LOG")
    if not path:
        return
    prompt_values = [str(value) for value in prompts]
    completion_values = [str(value) for value in completions]
    answer_values = [str(value) for value in _me344_sequence(answer)]
    batch = hashlib.sha256(
        json.dumps([prompt_values, completion_values, answer_values], sort_keys=True).encode()
    ).hexdigest()[:16]
    event = {
        "batch": batch,
        "function": name,
        "completion_count": len(completion_values),
        "scores": scores,
        "completion_previews": [value.replace("\n", " ")[:160] for value in completion_values],
    }
    with Path(path).open("a", encoding="utf-8") as output:
        output.write(json.dumps(event, ensure_ascii=True) + "\n")


def me344_process_rl_data(dataset_name, model_tokenizer, template_config, tmvp_config, x) -> dict:
    """Adapt a numeric prompt-answer row to the model's native chat template."""
    del dataset_name, template_config, tmvp_config
    question = str(x.get("prompt", x.get("question", ""))).strip()
    answer = _me344_decimal_text(x.get("answer", x.get("expected_answer")))
    if not question:
        raise ValueError("Each RL row needs a non-empty prompt or question.")
    messages = [
        {
            "role": "system",
            "content": (
                "Solve the problem. Put concise work inside <reasoning>...</reasoning>, "
                "then end with exactly <answer>number</answer>."
            ),
        },
        {"role": "user", "content": question},
    ]
    prompt = model_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return {"prompts": prompt, "question": question, "answer": json.dumps([answer])}


def me344_format_reward(prompts, completions, tmvp_config, answer, **kwargs) -> list[float]:
    """Shape partial structure while reserving the maximum for a complete answer block."""
    del tmvp_config, kwargs
    scores = []
    for value in completions:
        text = str(value)
        answer_tag = bool(_RL_ANSWER_RE.search(text))
        reasoning_tags = "<reasoning>" in text.lower() and "</reasoning>" in text.lower()
        if _RL_FULL_FORMAT_RE.search(text):
            scores.append(0.75)
        elif answer_tag and reasoning_tags:
            scores.append(0.5)
        elif answer_tag:
            scores.append(0.25)
        elif "<reasoning>" in text.lower() or "</reasoning>" in text.lower():
            scores.append(0.1)
        else:
            scores.append(0.0)
    _me344_log_reward("format", prompts, completions, answer, scores)
    return scores


def me344_exact_reward(prompts, completions, tmvp_config, answer, **kwargs) -> list[float]:
    """Reward exact numeric correctness even before the model learns the requested tags."""
    del tmvp_config, kwargs
    targets = _me344_targets(answer, len(completions))
    scores = [float(target is not None and _me344_prediction(value) == target) for value, target in zip(completions, targets)]
    _me344_log_reward("exact", prompts, completions, answer, scores)
    return scores


def me344_closeness_reward(prompts, completions, tmvp_config, answer, **kwargs) -> list[float]:
    """Give bounded partial credit to distinct near misses, preserving an exact-answer incentive."""
    del tmvp_config, kwargs
    targets = _me344_targets(answer, len(completions))
    scores = []
    for value, target in zip(completions, targets):
        prediction = _me344_prediction(value)
        if prediction is None or target is None or prediction == target:
            scores.append(0.0)
            continue
        relative_error = float(abs(prediction - target) / max(abs(target), Decimal(1)))
        scores.append(0.1 * max(0.0, 1.0 - min(relative_error, 1.0)))
    _me344_log_reward("closeness", prompts, completions, answer, scores)
    return scores


# MaxText's dataset hook loads this exact public name.
process_data = me344_process_rl_data
