#!/usr/bin/env python3
"""Execution and evidence plumbing for the ME344 final-project notebook."""

from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from dataclasses import dataclass
import datetime as dt
from decimal import Decimal, InvalidOperation
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import tarfile
import time


MODEL_NAME = "qwen3-4b-instruct-2507"
# MaxText exposes this shared 4B architecture under its Thinking-2507 config
# key; the course checkpoint and tokenizer remain the Instruct-2507 release.
MAXTEXT_MODEL_NAME = "qwen3-4b-thinking-2507"
TOKENIZER_PATH = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
ASSIGNMENT_VERSION = "2026.08.02.2"
MAXTEXT_GIT_SHA = "17c7172720ca813b05e5ea248dedd78a0c64612e"
GKE_MAXTEXT_GIT_SHA = "07a3264e056c8d57abf0e00ab567d1f074a52233"
MODEL_PARAMETER_ESTIMATE = 4_022_468_096
V5E_8_USD_PER_HOUR = 9.60
V5E_CHIPS = 8
V5E_HBM_BYTES_PER_CHIP = 16_000_000_000
V5E_PEAK_BF16_TFLOPS_PER_CHIP = 197
V5E_HBM_BANDWIDTH_GIB_PER_SECOND = 800
V5E_ICI_BIDIRECTIONAL_GB_PER_SECOND = 400
MINIPERF_VERSION = "me344-miniperf-v1"
SCALE_BASELINE_GLOBAL_BATCH = 256
SCALE_STRONG_GLOBAL_BATCH = 256
SCALE_WEAK_GLOBAL_BATCH = 512
SCALE_SEQUENCE_LENGTH = 256
MINIPERF_REMAT_POLICIES = {
    "full",
    "save_qkv_proj",
    "save_dot_except_mlpwi",
    "minimal",
}
MINIPERF_ATTENTION_TYPES = {"autoselected", "dot_product"}
BANKING77_REVISION = "8eed2b348ec07d40ffbe91dcc4718e2f0f977714"
GSM8K_REVISION = "740312add88f781978c0658806c59bc2815b9866"
BANKING77_INTENTS = (
    "card_arrival",
    "card_delivery_estimate",
    "card_payment_not_recognised",
    "cash_withdrawal_not_recognised",
    "direct_debit_payment_not_recognised",
    "pending_card_payment",
    "pending_cash_withdrawal",
    "pending_transfer",
    "declined_card_payment",
    "declined_cash_withdrawal",
    "declined_transfer",
    "failed_transfer",
    "transfer_not_received_by_recipient",
    "transfer_timing",
    "cash_withdrawal_charge",
    "card_payment_fee_charged",
    "transfer_fee_charged",
    "top_up_by_card_charge",
    "card_payment_wrong_exchange_rate",
    "wrong_exchange_rate_for_cash_withdrawal",
)
BANKING77_TRAIN_PER_INTENT = 20
BANKING77_RELEASE_PER_INTENT = 2
DOMAIN_RELEASE_PROMPT_COUNT = len(BANKING77_INTENTS) * BANKING77_RELEASE_PER_INTENT
RETENTION_PROMPT_COUNT = 16
GSM8K_RELEASE_PROMPT_COUNT = 32
RELEASE_PROMPT_COUNT = DOMAIN_RELEASE_PROMPT_COUNT + RETENTION_PROMPT_COUNT
REFLECTION_QUESTIONS = (
    (
        "Profile and scaling",
        "What did XProf show, and how did ICI versus DCN affect the scaling results?",
    ),
    ("Training", "What changed after SFT or GRPO?"),
    ("Release", "Which checkpoint would you ship, and what evidence supports it?"),
    ("Next step", "What would you try next with more time or a dataset you care about?"),
)

_RL_NUMBER = r"[-+]?(?:\d[\d,]*)(?:\.\d+)?(?:[eE][-+]?\d+)?"
_RL_ANSWER_RE = re.compile(rf"<answer>\s*({_RL_NUMBER})\s*</answer>", re.IGNORECASE)
_RL_FULL_FORMAT_RE = re.compile(
    rf"<reasoning>.+?</reasoning>\s*<answer>\s*{_RL_NUMBER}\s*</answer>\s*(?:<\|im_end\|>)?\s*$",
    re.IGNORECASE | re.DOTALL,
)


def _maxtext_config(relative_path: str = "base.yml") -> str:
    """Locate a config inside the installed, pinned MaxText package."""
    try:
        import maxtext.configs
    except ImportError as error:
        raise RuntimeError("Install the pinned MaxText package before running TPU work.") from error
    path = Path(maxtext.configs.__file__).resolve().parent / relative_path
    if not path.is_file():
        raise RuntimeError(f"The installed MaxText package is missing {relative_path}.")
    return str(path)


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


@dataclass(frozen=True)
class ProjectConfig:
    """Values a student may change before running the notebook."""

    run_tpu: bool
    student_id: str
    output_directory: str
    submission_uri: str
    base_checkpoint_path: str
    dataset_option: str
    local_input_jsonl: Path
    tpu_name: str
    tpu_zone: str


def project_config_from_env(run_tpu: bool | None = None) -> ProjectConfig:
    """Build the shared notebook configuration from the README environment."""
    if run_tpu is None:
        run_tpu = os.environ.get("ME344_LOCAL_DRY_RUN", "0") != "1"
    return ProjectConfig(
        run_tpu=run_tpu,
        student_id=os.environ.get("ME344_STUDENT_ID", "local-dry-run"),
        output_directory=os.environ.get("ME344_OUTPUT", "gs://replace-me/me344-runs"),
        submission_uri=os.environ.get(
            "ME344_SUBMISSION_URI", "gs://replace-me/me344-submission"
        ),
        base_checkpoint_path=os.environ.get(
            "ME344_BASE_CHECKPOINT",
            "gs://me344-tpu-labs-west4/data/qwen3-4b-instruct-2507/0/items",
        ),
        dataset_option=os.environ.get("ME344_DATASET", "banking77" if run_tpu else "local"),
        local_input_jsonl=Path(
            os.environ.get("ME344_LOCAL_JSONL", "examples/sft_pairs.jsonl")
        ),
        tpu_name=os.environ.get("ME344_TPU_NAME", "YOUR_TPU_NAME"),
        tpu_zone=os.environ.get("ME344_TPU_ZONE", "us-west4-a"),
    )


def benchmark_jax_matmul(matrix_size: int = 4096, repeats: int = 10) -> dict:
    """Time one BF16 matrix multiply on the notebook's default JAX device."""
    if matrix_size <= 0 or repeats <= 0:
        raise ValueError("matrix_size and repeats must be positive.")
    import jax
    import jax.numpy as jnp

    value = jnp.ones((matrix_size, matrix_size), dtype=jnp.bfloat16)
    multiply = jax.jit(lambda left, right: left @ right)
    started = time.perf_counter()
    multiply(value, value).block_until_ready()
    compile_seconds = time.perf_counter() - started
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        multiply(value, value).block_until_ready()
        samples.append(time.perf_counter() - started)
    median_seconds = statistics.median(samples)
    tflops = 2 * matrix_size**3 / median_seconds / 1e12
    result = {
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "matrix_size": matrix_size,
        "compile_plus_first_seconds": compile_seconds,
        "steady_median_milliseconds": 1000 * median_seconds,
        "steady_tflops": tflops,
        "v5e_bf16_peak_percent": 100 * tflops / V5E_PEAK_BF16_TFLOPS_PER_CHIP,
    }
    print(json.dumps(result, indent=2))
    return result


def run_scale_out_benchmark(
    mode: str,
    chips: int,
    result_path: Path,
    handoff_path: Path,
    student_id: str,
    steps: int = 12,
    sequence_length: int = SCALE_SEQUENCE_LENGTH,
) -> Path:
    """Run one bounded MaxText trial across a GKE TPU JobSet."""
    modes = {
        "baseline": {
            "chips": 8,
            "slices": 1,
            "nodes_per_slice": 1,
            "global_batch": SCALE_BASELINE_GLOBAL_BATCH,
            "ici_data_parallelism": 1,
            "dcn_data_parallelism": 1,
            "fabric": "ICI",
        },
        "ici_strong": {
            "chips": 16,
            "slices": 1,
            "nodes_per_slice": 4,
            "global_batch": SCALE_STRONG_GLOBAL_BATCH,
            "ici_data_parallelism": 2,
            "dcn_data_parallelism": 1,
            "fabric": "ICI",
        },
        "ici_weak": {
            "chips": 16,
            "slices": 1,
            "nodes_per_slice": 4,
            "global_batch": SCALE_WEAK_GLOBAL_BATCH,
            "ici_data_parallelism": 2,
            "dcn_data_parallelism": 1,
            "fabric": "ICI",
        },
        "strong": {
            "chips": 16,
            "slices": 2,
            "nodes_per_slice": 1,
            "global_batch": SCALE_STRONG_GLOBAL_BATCH,
            "ici_data_parallelism": 1,
            "dcn_data_parallelism": 2,
            "fabric": "DCN",
        },
        "weak": {
            "chips": 16,
            "slices": 2,
            "nodes_per_slice": 1,
            "global_batch": SCALE_WEAK_GLOBAL_BATCH,
            "ici_data_parallelism": 1,
            "dcn_data_parallelism": 2,
            "fabric": "DCN",
        },
    }
    if mode not in modes or chips != modes[mode]["chips"]:
        raise ValueError("Use baseline/8 or a supported ICI/DCN mode with 16 chips.")
    if not re.fullmatch(r"[a-z0-9]+", student_id):
        raise ValueError("student_id must be a lowercase SUNet ID.")
    if steps < 8:
        raise ValueError("Use at least eight steps so compilation does not dominate every sample.")

    handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
    if handoff.get("student_id") != student_id:
        raise RuntimeError("The scale-out checkpoint belongs to a different SUNet ID.")
    if handoff.get("maxtext_git_sha") != MAXTEXT_GIT_SHA:
        raise RuntimeError("The scale-out checkpoint used a different MaxText commit.")
    if handoff.get("assignment_version") != ASSIGNMENT_VERSION:
        raise RuntimeError("The scale-out checkpoint used a different assignment version.")
    checkpoint_path = str(handoff.get("checkpoint_path", ""))
    if not checkpoint_path.startswith("gs://"):
        raise RuntimeError("The scale-out handoff needs a gs:// checkpoint path.")
    runtime_checkpoint_path = checkpoint_path
    course_bucket_prefix = "gs://me344-tpu-labs-west4/"
    if Path("/gcs").is_dir() and checkpoint_path.startswith(course_bucket_prefix):
        runtime_checkpoint_path = "/gcs/" + checkpoint_path.removeprefix(course_bucket_prefix)
        if not Path(runtime_checkpoint_path).exists():
            raise RuntimeError(f"The GKE checkpoint is not visible at {runtime_checkpoint_path}.")
    next_step = int(handoff.get("next_step", 0))
    if next_step <= 0:
        raise RuntimeError("The scale-out handoff has an invalid next step.")

    spec = modes[mode]
    slices = int(spec["slices"])
    global_batch = int(spec["global_batch"])
    per_device_batch = global_batch / chips
    metrics_path = Path("/tmp") / f"me344-scale-{student_id}-{mode}.jsonl"
    metrics_path.unlink(missing_ok=True)

    try:
        observed_slices = int(os.environ["MEGASCALE_NUM_SLICES"])
        slice_index = int(os.environ["MEGASCALE_SLICE_ID"])
        nodes_per_slice = int(os.environ["ME344_NODES_PER_SLICE"])
        pod_index = int(os.environ["ME344_POD_INDEX"])
        chips_per_worker = int(os.environ["ME344_CHIPS_PER_WORKER"])
    except (KeyError, ValueError) as error:
        raise RuntimeError("The GKE JobSet did not provide valid worker coordinates.") from error
    if observed_slices != slices or nodes_per_slice != spec["nodes_per_slice"]:
        raise RuntimeError(
            f"Expected {slices} slice(s) with {spec['nodes_per_slice']} worker(s) each; "
            f"found {observed_slices} and {nodes_per_slice}."
        )
    if not 0 <= slice_index < slices or not 0 <= pod_index < nodes_per_slice:
        raise RuntimeError(f"Invalid JobSet coordinates: slice {slice_index}, worker {pod_index}.")
    process_count = slices * nodes_per_slice
    process_index = slice_index * nodes_per_slice + pod_index
    if process_count * chips_per_worker != chips:
        raise RuntimeError(
            f"Expected {chips} chips, but {process_count} workers x {chips_per_worker} chips gives "
            f"{process_count * chips_per_worker}."
        )
    if not os.environ.get("MEGASCALE_COORDINATOR_ADDRESS"):
        raise RuntimeError("The GKE JobSet did not provide its coordinator address.")
    mesh = {
        "backend": "tpu",
        "process_index": process_index,
        "process_count": process_count,
        "device_count": chips,
        "local_device_count": chips_per_worker,
        "source": "GKE slice coordinates; MaxText validates the runtime mesh",
    }
    print("ME344_DEVICE_MESH=" + json.dumps(mesh, sort_keys=True))

    print(
        f"{mode}: {slices} slice(s), {mesh['process_count']} workers, {chips} chips, "
        f"global batch {global_batch}, data parallelism over {spec['fabric']}"
    )
    command = [
        sys.executable,
        "-m",
        "maxtext.trainers.pre_train.train",
        _maxtext_config(),
        f"model_name={MAXTEXT_MODEL_NAME}",
        "base_output_directory=/tmp/me344-scale-output",
        f"run_name=me344-{student_id}-{mode}",
        "dataset_type=synthetic",
        "reuse_example_batch=true",
        f"steps={next_step + steps}",
        f"max_target_length={sequence_length}",
        f"per_device_batch_size={per_device_batch}",
        f"num_slices={slices}",
        f"dcn_data_parallelism={spec['dcn_data_parallelism']}",
        f"ici_data_parallelism={spec['ici_data_parallelism']}",
        "ici_fsdp_parallelism=1",
        "ici_tensor_parallelism=8",
        "allow_split_physical_axes=true",
        "learning_rate=0",
        f"load_full_state_path={runtime_checkpoint_path}",
        "enable_checkpointing=true",
        "checkpoint_period=10000",
        "max_num_checkpoints_to_keep=1",
        "async_checkpointing=false",
        "save_checkpoint_on_completion=false",
        "log_config=false",
        f"metrics_file={metrics_path}",
    ]
    started = time.monotonic()
    subprocess.run(command, check=True, timeout=20 * 60)
    wall_seconds = time.monotonic() - started

    if mesh["process_index"] == 0:
        rows = [
            json.loads(line)
            for line in metrics_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        measured = [row for row in rows if "perf/step_time_seconds" in row]
        if len(measured) < 8:
            raise RuntimeError(f"Expected at least eight measured steps in {metrics_path}.")
        steady = measured[-8:]
        step_seconds = statistics.fmean(
            float(row["perf/step_time_seconds"]) for row in steady
        )
        tokens_per_second = statistics.fmean(
            float(row["perf/per_device_tokens_per_sec"]) for row in steady
        ) * chips
        result = {
            "assignment_version": ASSIGNMENT_VERSION,
            "checkpoint_maxtext_git_sha": handoff["maxtext_git_sha"],
            "runtime_maxtext_git_sha": os.environ.get(
                "ME344_RUNTIME_MAXTEXT_SHA", GKE_MAXTEXT_GIT_SHA
            ),
            "student_id": student_id,
            "mode": mode,
            "slices": slices,
            "workers": mesh["process_count"],
            "nodes_per_slice": nodes_per_slice,
            "chips_per_worker": chips_per_worker,
            "chips": chips,
            "fabric": spec["fabric"],
            "topology": os.environ.get("ME344_TPU_TOPOLOGY", "unknown"),
            "global_batch": global_batch,
            "sequence_length": sequence_length,
            "tokens_per_step": global_batch * sequence_length,
            "steady_steps": len(steady),
            "source_checkpoint": checkpoint_path,
            "source_next_step": next_step,
            "command_wall_seconds": wall_seconds,
            "first_step_seconds": float(measured[0]["perf/step_time_seconds"]),
            "steady_step_seconds": step_seconds,
            "tokens_per_second": tokens_per_second,
            "tokens_per_second_per_chip": tokens_per_second / chips,
        }
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print("ME344_SCALE_RESULT=" + json.dumps(result, sort_keys=True))
        print("Saved:", result_path)
    return metrics_path


def summarize_scale_out(
    baseline_result: Path,
    ici_strong_result: Path,
    ici_weak_result: Path,
    strong_result: Path,
    weak_result: Path,
    output: Path = Path("scale_out.png"),
) -> dict:
    """Compare strong and weak scaling over ICI and DCN."""
    trials = {
        "baseline": json.loads(baseline_result.read_text(encoding="utf-8")),
        "ici_strong": json.loads(ici_strong_result.read_text(encoding="utf-8")),
        "ici_weak": json.loads(ici_weak_result.read_text(encoding="utf-8")),
        "strong": json.loads(strong_result.read_text(encoding="utf-8")),
        "weak": json.loads(weak_result.read_text(encoding="utf-8")),
    }
    expected = {
        "baseline": {
            "chips": 8,
            "slices": 1,
            "workers": 1,
            "fabric": "ICI",
            "topology": "2x4",
            "global_batch": SCALE_BASELINE_GLOBAL_BATCH,
        },
        "ici_strong": {
            "chips": 16,
            "slices": 1,
            "workers": 4,
            "fabric": "ICI",
            "topology": "4x4",
            "global_batch": SCALE_STRONG_GLOBAL_BATCH,
        },
        "ici_weak": {
            "chips": 16,
            "slices": 1,
            "workers": 4,
            "fabric": "ICI",
            "topology": "4x4",
            "global_batch": SCALE_WEAK_GLOBAL_BATCH,
        },
        "strong": {
            "chips": 16,
            "slices": 2,
            "workers": 2,
            "fabric": "DCN",
            "topology": "2x4",
            "global_batch": SCALE_STRONG_GLOBAL_BATCH,
        },
        "weak": {
            "chips": 16,
            "slices": 2,
            "workers": 2,
            "fabric": "DCN",
            "topology": "2x4",
            "global_batch": SCALE_WEAK_GLOBAL_BATCH,
        },
    }
    for mode, trial in trials.items():
        if trial.get("mode") != mode:
            raise RuntimeError(f"Expected {mode} result in its result file.")
        if trial.get("assignment_version") != ASSIGNMENT_VERSION:
            raise RuntimeError(f"{mode} used a different assignment version.")
        checkpoint_sha = trial.get(
            "checkpoint_maxtext_git_sha", trial.get("maxtext_git_sha")
        )
        if checkpoint_sha != MAXTEXT_GIT_SHA:
            raise RuntimeError(f"{mode} restored a checkpoint from a different MaxText commit.")
        runtime_sha = trial.get("runtime_maxtext_git_sha")
        if runtime_sha and runtime_sha != GKE_MAXTEXT_GIT_SHA:
            raise RuntimeError(f"{mode} ran with a different GKE MaxText commit.")
        for key in ("chips", "slices", "workers", "global_batch"):
            if int(trial.get(key, 0)) != expected[mode][key]:
                raise RuntimeError(
                    f"{mode} used {key}={trial.get(key)}; this assignment expects "
                    f"{expected[mode][key]}."
                )
        for key in ("fabric", "topology"):
            if trial.get(key) != expected[mode][key]:
                raise RuntimeError(
                    f"{mode} used {key}={trial.get(key)}; this assignment expects "
                    f"{expected[mode][key]}."
                )
        if int(trial.get("sequence_length", 0)) != SCALE_SEQUENCE_LENGTH:
            raise RuntimeError(f"{mode} used a stale sequence length.")
    baseline, ici_strong, ici_weak, strong, weak = (
        trials[name] for name in ("baseline", "ici_strong", "ici_weak", "strong", "weak")
    )
    if len({trial["source_checkpoint"] for trial in trials.values()}) != 1:
        raise RuntimeError("Scaling trials did not restore the same source checkpoint.")
    if len({trial["student_id"] for trial in trials.values()}) != 1:
        raise RuntimeError("Scaling trials came from different students.")
    chip_ratio = strong["chips"] / baseline["chips"]
    strong_speedup = strong["tokens_per_second"] / baseline["tokens_per_second"]
    weak_throughput_ratio = weak["tokens_per_second"] / baseline["tokens_per_second"]
    ici_strong_speedup = ici_strong["tokens_per_second"] / baseline["tokens_per_second"]
    ici_weak_throughput_ratio = ici_weak["tokens_per_second"] / baseline["tokens_per_second"]
    report = {
        "baseline": baseline,
        "ici_strong": ici_strong,
        "ici_weak": ici_weak,
        "strong": strong,
        "weak": weak,
        "ici_strong_speedup": ici_strong_speedup,
        "ici_strong_scaling_efficiency_percent": 100 * ici_strong_speedup / chip_ratio,
        "ici_weak_throughput_ratio": ici_weak_throughput_ratio,
        "ici_weak_scaling_efficiency_percent": 100 * ici_weak_throughput_ratio / chip_ratio,
        "strong_speedup": strong_speedup,
        "strong_scaling_efficiency_percent": 100 * strong_speedup / chip_ratio,
        "weak_throughput_ratio": weak_throughput_ratio,
        "weak_scaling_efficiency_percent": 100 * weak_throughput_ratio / chip_ratio,
        "ici_over_dcn_strong_ratio": ici_strong["tokens_per_second"] / strong["tokens_per_second"],
        "ici_over_dcn_weak_ratio": ici_weak["tokens_per_second"] / weak["tokens_per_second"],
        "design": {
            "sequence_length": SCALE_SEQUENCE_LENGTH,
            "one_host_global_batch": SCALE_BASELINE_GLOBAL_BATCH,
            "strong_global_batch": SCALE_STRONG_GLOBAL_BATCH,
            "weak_global_batch": SCALE_WEAK_GLOBAL_BATCH,
            "one_host_first_tested_oom_global_batch": SCALE_WEAK_GLOBAL_BATCH,
            "reason": (
                "The 16-chip trials use the same model, checkpoint, batches, and logical "
                "data/tensor layout. One 4x4 slice carries data-parallel communication over "
                "ICI; two 2x4 slices carry it over DCN."
            ),
        },
    }
    report_path = output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(report, indent=2))
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is unavailable; the JSON summary was still written to", report_path)
        return report

    labels = ["8 chips\nICI", "16 chips\nICI", "16 chips\nDCN"]
    colors = ["#4285F4", "#34A853", "#EA4335"]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    fixed = [baseline, ici_strong, strong]
    doubled = [baseline, ici_weak, weak]
    axes[0].bar(labels, [trial["tokens_per_second"] for trial in fixed], color=colors)
    axes[0].set(title=f"Strong scaling: batch {SCALE_STRONG_GLOBAL_BATCH}", ylabel="tokens/s")
    axes[1].bar(labels, [trial["tokens_per_second"] for trial in doubled], color=colors)
    axes[1].set(title=f"Weak scaling: batch {SCALE_WEAK_GLOBAL_BATCH} on 16 chips", ylabel="tokens/s")
    figure.suptitle(
        f"ICI strong {report['ici_strong_scaling_efficiency_percent']:.1f}% vs "
        f"DCN strong {report['strong_scaling_efficiency_percent']:.1f}%"
    )
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    print("Graph:", output)
    print("Summary:", report_path)
    return report


@dataclass(frozen=True)
class RunResult:
    log_path: Path
    seconds: float
    estimated_cost: float
    returncode: int


class FinalProject:
    """Runs bounded MaxText stages and builds a reproducible evidence bundle."""

    def __init__(self, config: ProjectConfig):
        self.config = config
        self.model_name = MODEL_NAME
        self.maxtext_model_name = MAXTEXT_MODEL_NAME
        self.tokenizer_path = TOKENIZER_PATH
        self.model_revision = MODEL_REVISION
        self.project_root = self._find_project_root()
        self.assignment_version = self._resolve_assignment_version()
        self.pretrain_python = self.project_root / ".venv-me344-pretrain/bin/python"
        self.posttrain_python = self.project_root / ".venv-me344-posttrain/bin/python"
        self._validate_environment()

        safe_id = re.sub(r"[^a-zA-Z0-9_-]+", "-", config.student_id).strip("-") or "student"
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_id = f"{safe_id}-{stamp.lower()}"
        self.artifact_dir = Path.home() / "me344-artifacts" / self.run_id
        self.data_dir = Path.home() / ".cache" / "me344" / safe_id
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._validate_local_dataset()

        self.session_started = dt.datetime.now(dt.timezone.utc)
        self.commands_path = self.artifact_dir / "commands.sh"
        self.commands_path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n", encoding="utf-8")
        self.ledger_path = self.artifact_dir / "cost_ledger.csv"
        with self.ledger_path.open("w", encoding="utf-8", newline="") as output:
            csv.writer(output).writerow(
                [
                    "kind",
                    "stage",
                    "started_utc",
                    "ended_utc",
                    "wall_seconds",
                    "estimated_usd",
                    "returncode",
                ]
            )

        self.reports: dict = {}
        self.model_config: dict = {}
        self.report: dict = {}
        self.rl_summary: dict = {}
        self.scale_report: dict = {}
        self.capacity_path = self.artifact_dir / "capacity_report.json"
        self.eval_suite_path = self.artifact_dir / "eval_suite.jsonl"
        self.sweep_path = self.artifact_dir / "sweep_results.csv"
        self.xla_flag_path = self.artifact_dir / "xla_flag_ab.csv"
        self.xla_hlo_summary_path = self.artifact_dir / "xla_hlo_summary.json"
        self.miniperf_path = self.artifact_dir / "miniperf_trials.csv"
        self.pretrain_run_name = f"me344-{self.run_id}-c4"
        self.pretrain_checkpoint_path = ""
        self.pretrain_next_step = 0
        self.pretrain_source: dict = {}
        self.sft_run_name = f"me344-{self.run_id}-sft"
        self.sft_checkpoint_path = ""
        self.sft_rollback_path = ""
        self.sft_plan: dict = {}
        self._server: dict | None = None
        self._write_answers()
        self._write_config()
        self._print_startup()

    @staticmethod
    def _find_project_root() -> Path:
        start = Path.cwd().resolve()
        root = next(
            (
                path
                for path in (start, *start.parents)
                if (path / "final_project.ipynb").exists()
                and (path / "scripts/final_project.py").exists()
            ),
            None,
        )
        if root is None:
            raise RuntimeError("Run this notebook from inside me344_final_project.")
        return root

    def _validate_environment(self) -> None:
        if self.config.run_tpu:
            configured = {
                "student_id": self.config.student_id,
                "output_directory": self.config.output_directory,
                "submission_uri": self.config.submission_uri,
                "base_checkpoint_path": self.config.base_checkpoint_path,
                "tpu_name": self.config.tpu_name,
                "tpu_zone": self.config.tpu_zone,
            }
            placeholder = re.compile(r"replace-me|your[_-]|instructor[_-]|[<>]", re.IGNORECASE)
            invalid = [name for name, value in configured.items() if not value or placeholder.search(value)]
            if invalid:
                raise RuntimeError(
                    "Finish the README setup before starting TPU work: " + ", ".join(invalid)
                )
            if not self.config.output_directory.startswith("gs://"):
                raise RuntimeError("ME344_OUTPUT must be a gs:// path.")
            if not self.config.submission_uri.startswith("gs://"):
                raise RuntimeError("ME344_SUBMISSION_URI must be a gs:// path.")
            if not self.config.base_checkpoint_path.startswith("gs://"):
                raise RuntimeError("ME344_BASE_CHECKPOINT must be a gs:// path.")
            missing = [path for path in (self.pretrain_python, self.posttrain_python) if not path.exists()]
            if missing:
                raise RuntimeError(f"Missing course environments: {missing}")
            if self.pretrain_python.absolute() == self.posttrain_python.absolute():
                raise RuntimeError("Pre- and post-training environments must be distinct.")
            if self.config.student_id == "local-dry-run":
                raise RuntimeError("Set your SUNet ID in the README setup block, then restart Jupyter.")
            if "replace-me" in self.config.output_directory or "replace-me" in self.config.base_checkpoint_path:
                raise RuntimeError("Set the instructor-provided output and checkpoint paths.")
            if Path(sys.executable).absolute() != self.posttrain_python.absolute():
                raise RuntimeError("Select the 'ME344 MaxText' Jupyter kernel, then restart.")
            return
        if not self.pretrain_python.exists():
            self.pretrain_python = Path(sys.executable).absolute()
        if not self.posttrain_python.exists():
            self.posttrain_python = Path(sys.executable).absolute()

    @staticmethod
    def _resolve_assignment_version() -> str:
        expected = os.environ.get("ME344_ASSIGNMENT_VERSION", "").strip()
        if expected and expected != ASSIGNMENT_VERSION:
            raise RuntimeError(
                f"Wrong assignment version: expected {expected}, found {ASSIGNMENT_VERSION}."
            )
        return ASSIGNMENT_VERSION

    def _validate_local_dataset(self) -> None:
        if not self.config.run_tpu or self.config.dataset_option != "local":
            return
        local_path = (self.project_root / self.config.local_input_jsonl).resolve()
        rows = sum(1 for line in local_path.open(encoding="utf-8") if line.strip())
        if not 50 <= rows <= 500:
            raise RuntimeError(
                f"Your local SFT file needs 50-500 non-empty rows; found {rows}. "
                "examples/sft_pairs.jsonl only demonstrates the schema, so set "
                "ME344_LOCAL_JSONL to the file you prepared."
            )

    def _write_answers(self) -> None:
        lines = ["# ME344 Final Reflection", "", "Answer each question in one sentence.", ""]
        for index, (label, question) in enumerate(REFLECTION_QUESTIONS, 1):
            lines.extend((f"{index}. **{label}:** {question} TODO", ""))
        (self.artifact_dir / "answers.md").write_text("\n".join(lines), encoding="utf-8")

    def collect_reflections(self) -> Path:
        """Collect the four final responses directly in the notebook."""
        path = self.artifact_dir / "answers.md"
        if not self.config.run_tpu:
            self._skip("reflection prompts")
            return path

        responses = []
        print("Answer each question in one sentence. Press Enter after each answer.\n")
        for index, (label, question) in enumerate(REFLECTION_QUESTIONS, 1):
            response = ""
            while not response:
                response = input(f"{index}. {label}: {question}\n> ").strip()
                if not response:
                    print("Please enter a short answer.")
            responses.append(response)

        lines = ["# ME344 Final Reflection", ""]
        for index, ((label, question), response) in enumerate(
            zip(REFLECTION_QUESTIONS, responses), 1
        ):
            lines.extend((f"{index}. **{label}:** {question}", "", response, ""))
        path.write_text("\n".join(lines), encoding="utf-8")
        print("Saved:", path)
        return path

    def prepare_scale_handoff(self) -> dict:
        """Publish the TPU-VM checkpoint pointer consumed by the GKE JobSets."""
        if not self.config.run_tpu:
            self._skip("TPU-VM checkpoint handoff")
            return {}
        if not self.pretrain_checkpoint_path or self.pretrain_next_step <= 0:
            raise RuntimeError("Run the C4 pre-training pilot before preparing GKE.")

        student_root = self.config.submission_uri.rstrip("/").rsplit("/", 1)[0]
        manifest = {
            "assignment_version": ASSIGNMENT_VERSION,
            "maxtext_git_sha": MAXTEXT_GIT_SHA,
            "student_id": self.config.student_id,
            "model_name": self.maxtext_model_name,
            "checkpoint_type": "MaxText full training state",
            "checkpoint_path": self.pretrain_checkpoint_path,
            "checkpoint_step": self.pretrain_next_step - 1,
            "next_step": self.pretrain_next_step,
            "source": self.pretrain_source,
        }
        local_path = self.artifact_dir / "scale_handoff.json"
        local_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        remote = f"{student_root}/scaling/handoff.json"
        command = ["gcloud", "storage", "cp", str(local_path), remote]
        self._record_command(command)
        uploaded = subprocess.run(command, capture_output=True, text=True, timeout=180)
        if uploaded.returncode:
            raise RuntimeError(
                "Could not publish the checkpoint handoff for GKE:\n"
                + (uploaded.stderr or uploaded.stdout)
            )
        print("Checkpoint:", self.pretrain_checkpoint_path)
        print("GKE handoff:", remote)
        return manifest

    def collect_scale_results(self) -> dict:
        """Bring the required GKE trials into the notebook dashboard."""
        if not self.config.run_tpu:
            self._skip("GKE scale results")
            return {}

        scale_dir = self.artifact_dir / "scaling"
        scale_dir.mkdir(exist_ok=True)
        student_root = self.config.submission_uri.rstrip("/").rsplit("/", 1)[0]
        modes = ("baseline", "ici_strong", "ici_weak", "strong", "weak")
        paths = {mode: scale_dir / f"{mode}.json" for mode in modes}
        failures = []
        for mode, path in paths.items():
            remote = f"{student_root}/scaling/{mode}.json"
            command = ["gcloud", "storage", "cp", remote, str(path)]
            self._record_command(command)
            result = subprocess.run(command, capture_output=True, text=True, timeout=120)
            if result.returncode:
                failures.append((mode, remote, result.stderr or result.stdout))

        if failures:
            print("The TPU VM could not read the GKE result files automatically.")
            print("In a Stanford-node terminal, read the saved results with:")
            for mode in modes:
                print(f"  gcloud storage cat {student_root}/scaling/{mode}.json")
            print("Enter each tokens_per_second value below.")
            manual = {}
            labels = {
                "baseline": f"one slice, batch {SCALE_BASELINE_GLOBAL_BATCH}",
                "ici_strong": f"one 4x4 slice, batch {SCALE_STRONG_GLOBAL_BATCH}",
                "ici_weak": f"one 4x4 slice, batch {SCALE_WEAK_GLOBAL_BATCH}",
                "strong": f"two 2x4 slices, batch {SCALE_STRONG_GLOBAL_BATCH}",
                "weak": f"two 2x4 slices, batch {SCALE_WEAK_GLOBAL_BATCH}",
            }
            for mode in modes:
                value = 0.0
                while value <= 0:
                    try:
                        value = float(input(f"Tokens/s for {labels[mode]}: ").strip())
                    except ValueError:
                        value = 0.0
                    if value <= 0:
                        print("Enter a positive number from the GKE result.")
                chips = 8 if mode == "baseline" else 16
                ici = mode.startswith("ici_") or mode == "baseline"
                manual[mode] = {
                    "assignment_version": ASSIGNMENT_VERSION,
                    "checkpoint_maxtext_git_sha": MAXTEXT_GIT_SHA,
                    "runtime_maxtext_git_sha": GKE_MAXTEXT_GIT_SHA,
                    "student_id": self.config.student_id,
                    "mode": mode,
                    "slices": 1 if ici else 2,
                    "workers": 1 if mode == "baseline" else 4 if ici else 2,
                    "nodes_per_slice": 1 if mode == "baseline" or not ici else 4,
                    "chips_per_worker": 8 if mode == "baseline" or not ici else 4,
                    "chips": chips,
                    "fabric": "ICI" if ici else "DCN",
                    "topology": "4x4" if mode.startswith("ici_") else "2x4",
                    "global_batch": {
                        "baseline": SCALE_BASELINE_GLOBAL_BATCH,
                        "ici_strong": SCALE_STRONG_GLOBAL_BATCH,
                        "ici_weak": SCALE_WEAK_GLOBAL_BATCH,
                        "strong": SCALE_STRONG_GLOBAL_BATCH,
                        "weak": SCALE_WEAK_GLOBAL_BATCH,
                    }[mode],
                    "sequence_length": SCALE_SEQUENCE_LENGTH,
                    "tokens_per_second": value,
                    "tokens_per_second_per_chip": value / chips,
                    "source_checkpoint": self.pretrain_checkpoint_path or "manual-entry",
                    "source": "manual notebook handoff",
                }
                paths[mode].write_text(json.dumps(manual[mode], indent=2) + "\n", encoding="utf-8")

        for mode, path in paths.items():
            trial = json.loads(path.read_text(encoding="utf-8"))
            if trial.get("student_id") != self.config.student_id:
                raise RuntimeError(f"{mode} belongs to a different SUNet ID.")

        self.scale_report = summarize_scale_out(
            paths["baseline"],
            paths["ici_strong"],
            paths["ici_weak"],
            paths["strong"],
            paths["weak"],
            output=self.artifact_dir / "scale_out.png",
        )
        print(
            f"ICI strong: {self.scale_report['ici_strong_speedup']:.2f}x speedup, "
            f"{self.scale_report['ici_strong_scaling_efficiency_percent']:.1f}% efficiency"
        )
        print(
            f"DCN strong: {self.scale_report['strong_speedup']:.2f}x speedup, "
            f"{self.scale_report['strong_scaling_efficiency_percent']:.1f}% efficiency"
        )
        print(
            f"ICI weak: {self.scale_report['ici_weak_throughput_ratio']:.2f}x throughput; "
            f"DCN weak: {self.scale_report['weak_throughput_ratio']:.2f}x"
        )
        return self.scale_report

    def _base_config(self) -> dict:
        return {
            "student_id": self.config.student_id,
            "run_id": self.run_id,
            "assignment_version": self.assignment_version,
            "maxtext_git_sha": MAXTEXT_GIT_SHA,
            "model_name": self.model_name,
            "maxtext_model_name": self.maxtext_model_name,
            "tokenizer_path": self.tokenizer_path,
            "model_revision": self.model_revision,
            "base_output_directory": self.config.output_directory,
            "submission_uri": self.config.submission_uri,
            "base_checkpoint_path": self.config.base_checkpoint_path,
            "dataset_option": self.config.dataset_option,
            "tpu_name": self.config.tpu_name,
            "tpu_zone": self.config.tpu_zone,
            "run_tpu": self.config.run_tpu,
            "pretrain_python": str(self.pretrain_python),
            "posttrain_python": str(self.posttrain_python),
            "session_started_utc": self.session_started.isoformat(),
        }

    def _write_config(self) -> None:
        payload = self._base_config()
        if self.reports:
            payload["device_reports"] = self.reports
        if self.model_config:
            payload["resolved_model"] = self.model_config
        (self.artifact_dir / "config.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def _print_startup(self) -> None:
        print(f"SUNet ID: {self.config.student_id}")
        print(f"Notebook results: {self.artifact_dir}")
        print(f"Cloud output: {self.config.output_directory}")
        print(f"Starting checkpoint: {self.config.base_checkpoint_path}")
        print(f"Pinned MaxText commit: {MAXTEXT_GIT_SHA}")

    def _skip(self, stage: str) -> None:
        print(f"SKIPPED {stage}: TPU-only stage in a local dry run.")

    def _record_command(self, command: list[str]) -> None:
        with self.commands_path.open("a", encoding="utf-8") as output:
            output.write(" ".join(shlex.quote(part) for part in command) + "\n")

    def _append_ledger(
        self,
        kind: str,
        stage: str,
        started: dt.datetime,
        ended: dt.datetime,
        seconds: float,
        cost: float,
        returncode: int | str,
    ) -> None:
        with self.ledger_path.open("a", encoding="utf-8", newline="") as output:
            csv.writer(output).writerow(
                [
                    kind,
                    stage,
                    started.isoformat(),
                    ended.isoformat(),
                    f"{seconds:.2f}",
                    f"{cost:.4f}",
                    returncode,
                ]
            )

    def _run_logged(
        self,
        command: list[str],
        log_name: str,
        stage: str,
        extra_env: dict[str, str] | None = None,
        timeout_minutes: int = 15,
        allow_failure: bool = False,
    ) -> RunResult:
        command = [
            "timeout",
            "--signal=INT",
            "--kill-after=60s",
            f"{timeout_minutes}m",
            *command,
        ]
        self._record_command(command)
        log_path = self.artifact_dir / log_name
        started = dt.datetime.now(dt.timezone.utc)
        start_time = time.monotonic()
        returncode = 127
        tail: deque[str] = deque(maxlen=40)
        env = os.environ.copy()
        # Importing MaxText in the post-training notebook kernel points this
        # variable at that environment's libtpu. Let each child environment
        # discover its own compatible libtpu instead.
        env.pop("TPU_LIBRARY_PATH", None)
        env["MPLBACKEND"] = "Agg"
        env.update(extra_env or {})
        visible = (
            "System Information:",
            "Memory analysis:",
            "Total memory size:",
            "Memstats:",
            "RESOURCE_EXHAUSTED",
            "completed step:",
            "Saved a checkpoint",
            "Train step",
            "Train loop finished",
            "Pre RL Training:",
            "Post RL Training:",
            "RL Training Completed",
            "Generated text:",
            "Output:",
        )
        try:
            with log_path.open("w", encoding="utf-8") as log:
                process = subprocess.Popen(
                    command,
                    cwd=self.project_root,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                assert process.stdout is not None
                for line in process.stdout:
                    log.write(line)
                    tail.append(line.rstrip())
                    if any(term in line for term in visible):
                        print(line.rstrip())
                returncode = process.wait()
        finally:
            ended = dt.datetime.now(dt.timezone.utc)
            seconds = time.monotonic() - start_time
            cost = seconds * V5E_8_USD_PER_HOUR / 3600
            self._append_ledger("command", stage, started, ended, seconds, cost, returncode)
            print(f"{stage} finished in {seconds / 60:.1f} minutes (exit {returncode}).")
        if returncode:
            if allow_failure:
                summary = [
                    line
                    for line in tail
                    if any(
                        term in line.lower()
                        for term in (
                            "resource_exhausted",
                            "out of memory",
                            "hbm allocation",
                            "hbm usage",
                            "timed out",
                            "killed",
                        )
                    )
                ]
                print("\nHandled trial failure:\n" + "\n".join(summary[-3:] or list(tail)[-3:]))
            else:
                print("\nLast log lines:\n" + "\n".join(tail))
            if not allow_failure:
                raise subprocess.CalledProcessError(returncode, command)
        return RunResult(log_path, seconds, cost, returncode)

    @staticmethod
    def _metric_rows(path: Path) -> list[dict]:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not rows:
            raise RuntimeError(f"No metrics in {path}")
        return rows

    @staticmethod
    def _parse_compiled_memory(log_text: str) -> dict[str, float]:
        """Parse JAX's per-device CompiledMemoryStats from an AOT log."""
        byte_fields = {
            key: int(value)
            for key, value in re.findall(r"([a-z_]+_size_in_bytes)=([0-9]+)", log_text)
        }
        required = (
            "argument_size_in_bytes",
            "output_size_in_bytes",
            "temp_size_in_bytes",
            "alias_size_in_bytes",
        )
        if not all(key in byte_fields for key in required):
            return {}

        gib = 2**30
        argument = byte_fields["argument_size_in_bytes"] / gib
        output = byte_fields["output_size_in_bytes"] / gib
        temporary = byte_fields["temp_size_in_bytes"] / gib
        alias = byte_fields["alias_size_in_bytes"] / gib
        return {
            "aot_reported_argument_gib_per_chip": argument,
            "aot_reported_output_gib_per_chip": output,
            "aot_reported_temporary_gib_per_chip": temporary,
            "aot_reported_alias_gib_per_chip": alias,
            "aot_reported_net_output_gib_per_chip": max(0.0, output - alias),
            "aot_reported_total_gib_per_chip": argument + output + temporary - alias,
            "aot_reported_host_temporary_gib": byte_fields.get("host_temp_size_in_bytes", 0) / gib,
        }

    @staticmethod
    def _parse_runtime_hbm(log_text: str) -> dict[str, float]:
        """Parse the runtime allocation snapshot MaxText prints after parameter initialization."""
        matches = re.findall(
            r"Using \(GB\)\s+([0-9]+(?:\.[0-9]+)?)\s*/\s*([0-9]+(?:\.[0-9]+)?)[^\n]*on TPU",
            log_text,
        )
        if not matches:
            return {}
        used = [float(match[0]) for match in matches]
        limits = [float(match[1]) for match in matches]
        return {
            "runtime_hbm_snapshot_gib_per_chip": max(used),
            "runtime_hbm_limit_gib_per_chip": min(limits),
            "runtime_hbm_snapshot_percent_of_limit": 100 * max(used) / min(limits),
        }

    @staticmethod
    def _parse_oom_hbm(log_text: str) -> dict[str, float]:
        """Parse XLA's rounded temporary-memory requirement from an HBM OOM."""
        match = re.search(
            r"total memory required for HLO temporaries \(([0-9.]+)G\) "
            r"exceeds available HBM \(([0-9.]+)G\)",
            log_text,
            flags=re.IGNORECASE,
        )
        if not match:
            return {}
        return {
            "oom_required_temporary_gib_per_chip": float(match.group(1)),
            "oom_available_hbm_gib_per_chip": float(match.group(2)),
        }

    @staticmethod
    def _is_memory_failure(log_text: str) -> bool:
        lower = log_text.lower()
        return any(
            term in lower
            for term in (
                "resource_exhausted",
                "out of memory",
                "ran out of memory",
                "hbm allocation",
                "hbm usage",
            )
        )

    @staticmethod
    def _select_batch_knee(rows: list[dict]) -> dict:
        """Prefer the smallest shape within 1% of peak and below 90% reported memory."""
        if not rows:
            raise ValueError("At least one measured batch is required.")
        peak = max(float(row["total_tokens_per_sec"]) for row in rows)
        near_peak = [
            row
            for row in rows
            if float(row["total_tokens_per_sec"]) >= 0.99 * peak
            and float(row["aot_reported_total_percent_of_limit"]) <= 90
        ]
        if near_peak:
            return min(near_peak, key=lambda row: int(row["global_batch"]))
        within_headroom = [
            row for row in rows if float(row["aot_reported_total_percent_of_limit"]) <= 90
        ]
        return max(within_headroom or rows, key=lambda row: float(row["total_tokens_per_sec"]))

    @staticmethod
    def _write_jsonl(path: Path, rows) -> None:
        with path.open("w", encoding="utf-8") as output:
            for row in rows:
                output.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _require_final_step(self, path: Path, expected: int) -> None:
        observed = int(self._metric_rows(path)[-1]["step"])
        if observed != expected:
            raise RuntimeError(f"{path.name}: expected final step {expected}, observed {observed}")

    def _probe_storage(self) -> None:
        checkpoint = self.config.base_checkpoint_path.rstrip("/") + "/"
        read = subprocess.run(
            ["gcloud", "storage", "ls", checkpoint],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if read.returncode:
            raise RuntimeError(
                "Cannot read the instructor checkpoint with this TPU service account:\n"
                + (read.stderr or read.stdout)
            )

        marker = self.artifact_dir / "storage-preflight.txt"
        marker.write_text(f"{self.run_id}\n", encoding="utf-8")
        remotes = (
            f"{self.config.output_directory.rstrip('/')}/_preflight/{self.run_id}.txt",
            f"{self.config.submission_uri.rstrip('/')}/_preflight/{self.run_id}.txt",
        )
        try:
            for remote in remotes:
                for command in (
                    ["gcloud", "storage", "cp", str(marker), remote],
                    ["gcloud", "storage", "rm", remote],
                ):
                    self._record_command(command)
                    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
                    if result.returncode:
                        raise RuntimeError(
                            "Storage write/delete preflight failed:\n"
                            + (result.stderr or result.stdout)
                        )
        finally:
            marker.unlink(missing_ok=True)

        self.reports["storage"] = {
            "base_checkpoint": self.config.base_checkpoint_path,
            "output_directory": self.config.output_directory,
            "submission_uri": self.config.submission_uri,
            "read_write_delete": "passed",
        }
        print("storage preflight: checkpoint read and output write/delete passed")

    def probe(self) -> dict:
        """Verify the two environments and record the resolved model/mesh."""
        probe_source = r"""
import json, os, platform, jax
from importlib.metadata import distribution
direct_url = json.loads(distribution("maxtext").read_text("direct_url.json") or "{}")
print(json.dumps({
  "python": platform.python_version(), "jax": jax.__version__,
  "maxtext_commit": direct_url.get("vcs_info", {}).get("commit_id", ""),
  "backend": jax.default_backend(), "process_index": jax.process_index(),
  "process_count": jax.process_count(), "device_count": jax.device_count(),
  "local_device_count": jax.local_device_count(),
  "host_ram_bytes": os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"),
  "devices": [str(d) for d in jax.devices()],
}))
"""
        self.reports = {}
        if self.config.run_tpu:
            self._probe_storage()
        for label, python in (
            ("pretrain", self.pretrain_python),
            ("posttrain", self.posttrain_python),
        ):
            env = os.environ.copy()
            env["MPLBACKEND"] = "Agg"
            if self.config.run_tpu:
                env["JAX_PLATFORMS"] = "tpu"
            source = ("import tunix, vllm\n" if label == "posttrain" else "") + probe_source
            result = subprocess.run(
                [str(python), "-c", source],
                cwd=self.project_root,
                env=env,
                capture_output=True,
                text=True,
                timeout=180,
            )
            if result.returncode:
                if self.config.run_tpu:
                    raise RuntimeError(f"{label} device probe failed:\n{result.stderr}")
                self.reports[label] = {"error": (result.stderr or result.stdout).strip()}
                continue
            payload = next(line for line in reversed(result.stdout.splitlines()) if line.startswith("{"))
            self.reports[label] = json.loads(payload)
            print(label, self.reports[label])

        if self.config.run_tpu:
            for label in ("pretrain", "posttrain"):
                report = self.reports[label]
                observed = (
                    report["backend"],
                    report["process_count"],
                    report["device_count"],
                    report["local_device_count"],
                )
                if observed != ("tpu", 1, 8, 8):
                    raise RuntimeError(f"{label}: expected (tpu, 1, 8, 8), got {observed}")
                if report.get("maxtext_commit") != MAXTEXT_GIT_SHA:
                    raise RuntimeError(
                        f"{label}: expected MaxText {MAXTEXT_GIT_SHA}, "
                        f"found {report.get('maxtext_commit') or 'unknown'}. Reinstall from the README."
                    )

        self.model_config = self._probe_model_config()
        self._write_config()
        (self.artifact_dir / "system_info.json").write_text(json.dumps(self.reports, indent=2) + "\n", encoding="utf-8")
        return self.reports

    def _probe_model_config(self) -> dict:
        source = f"""\
import json
from pathlib import Path
import maxtext.configs
from maxtext.configs import pyconfig
base_config = Path(maxtext.configs.__file__).resolve().parent / "base.yml"
c = pyconfig.initialize([
  "probe", str(base_config), "model_name={self.maxtext_model_name}",
  "run_name=probe", "base_output_directory=/tmp/me344-probe",
  "dataset_type=synthetic", "skip_jax_distributed_system=true", "log_config=false",
])
print(json.dumps({{
  "emb_dim": c.emb_dim, "num_decoder_layers": c.num_decoder_layers,
  "num_query_heads": c.num_query_heads, "num_kv_heads": c.num_kv_heads,
  "head_dim": c.head_dim, "mlp_dim": c.mlp_dim, "vocab_size": c.vocab_size,
  "ici_parallelism": list(c.ici_parallelism), "dcn_parallelism": list(c.dcn_parallelism),
}}))
"""
        env = os.environ.copy()
        env["MPLBACKEND"] = "Agg"
        result = subprocess.run(
            [str(self.pretrain_python), "-c", source],
            cwd=self.project_root,
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
        )
        if result.returncode:
            if self.config.run_tpu:
                raise RuntimeError(result.stderr)
            return {"error": (result.stderr or result.stdout).strip()}
        config = json.loads(next(line for line in reversed(result.stdout.splitlines()) if line.startswith("{")))
        print("resolved model:", config)
        return config

    def run_baselines(
        self,
        smoke_steps: int = 5,
        c4_steps: int = 5,
        c4_sequence_length: int = 512,
        c4_global_batch: int = 2,
    ) -> str:
        """Run the pre-training baselines and retain one scale-out checkpoint."""
        if c4_steps < 1:
            raise ValueError("The C4 pilot needs at least one step.")
        self.pretrain_next_step = c4_steps
        self.pretrain_source = {
            "global_batch": c4_global_batch,
            "sequence_length": c4_sequence_length,
            "pilot_steps": c4_steps,
            "checkpoint_after_steps": c4_steps,
            "dataset": "allenai/c4:en",
        }
        self.pretrain_checkpoint_path = (
            f"{self.config.output_directory.rstrip('/')}/{self.pretrain_run_name}/"
            f"checkpoints/{self.pretrain_next_step - 1}/items"
        )
        if not self.config.run_tpu:
            self._skip("synthetic smoke and continued pre-training")
            return self.pretrain_checkpoint_path
        smoke_metrics = self.artifact_dir / "smoke_metrics.jsonl"
        smoke = [
            str(self.pretrain_python),
            "-m",
            "maxtext.trainers.pre_train.train",
            _maxtext_config("tpu/tpu_smoke_test.yml"),
            f"base_output_directory={self.config.output_directory}",
            f"run_name=me344-{self.run_id}-smoke",
            f"steps={smoke_steps}",
            "enable_checkpointing=false",
            "log_config=false",
            f"metrics_file={smoke_metrics}",
        ]
        self._run_logged(smoke, "smoke.log", "smoke")
        self._require_final_step(smoke_metrics, smoke_steps - 1)

        pretrain_metrics = self.artifact_dir / "pretrain_metrics.jsonl"
        pretrain = [
            str(self.pretrain_python),
            "-m",
            "maxtext.trainers.pre_train.train",
            _maxtext_config(),
            f"model_name={self.maxtext_model_name}",
            f"tokenizer_path={self.tokenizer_path}",
            "tokenizer_type=huggingface",
            f"base_output_directory={self.config.output_directory}",
            f"run_name={self.pretrain_run_name}",
            "dataset_type=hf",
            "hf_path=allenai/c4",
            "hf_data_dir=en",
            "train_split=train",
            f"steps={c4_steps}",
            f"max_target_length={c4_sequence_length}",
            "ici_fsdp_parallelism=1",
            "ici_tensor_parallelism=8",
            f"per_device_batch_size={c4_global_batch / 8}",
            "enable_checkpointing=true",
            "checkpoint_period=10000",
            "max_num_checkpoints_to_keep=1",
            "async_checkpointing=false",
            "save_checkpoint_on_completion=true",
            "log_config=false",
            f"metrics_file={pretrain_metrics}",
        ]
        self._run_logged(pretrain, "pretrain.log", "continued-pretrain")
        self._require_final_step(pretrain_metrics, c4_steps - 1)
        checkpoint = subprocess.run(
            ["gcloud", "storage", "ls", f"{self.pretrain_checkpoint_path}/**"],
            capture_output=True,
            text=True,
            timeout=180,
        )
        if checkpoint.returncode:
            raise RuntimeError(
                "The TPU-VM pilot did not write its scale-out checkpoint:\n"
                + (checkpoint.stderr or checkpoint.stdout)
            )
        print("Scale-out checkpoint:", self.pretrain_checkpoint_path)
        return self.pretrain_checkpoint_path

    def run_capacity_lab(
        self,
        global_batch: int = 2,
        sequence_length: int = 512,
    ) -> Path:
        """Compare rough memory math with one successful AOT compile and one AOT OOM."""
        if not self.config.run_tpu:
            self._skip("capacity plan and controlled AOT OOM")
            return self.capacity_path
        if global_batch <= 0 or sequence_length <= 0:
            raise ValueError("global_batch and sequence_length must be positive.")

        parameter_count = MODEL_PARAMETER_ESTIMATE
        checkpoint_bytes = parameter_count * 2
        fp32_parameter_bytes = parameter_count * 4
        persistent_training_bytes = parameter_count * 12
        gradient_bytes = parameter_count * 4
        # FP32 weights, two FP32 Adam moments, and a full FP32 gradient. Activations
        # and compiler temporaries are deliberately excluded from this rough bound.
        rough_peak_bytes = persistent_training_bytes + gradient_bytes
        hbm_gib = V5E_HBM_BYTES_PER_CHIP / 2**30

        cases = []
        specs = [
            ("TP8 AOT", 1, 1, 8, "not_run", global_batch, sequence_length),
            ("DP8 AOT", 8, 1, 1, "not_run", 8, 128),
        ]
        for name, data, fsdp, tensor, observed, batch, length in specs:
            product = data * fsdp * tensor
            state_shards = fsdp * tensor
            per_chip_bytes = rough_peak_bytes / state_shards
            prediction = (
                "invalid_mesh"
                if product != V5E_CHIPS
                else "oom" if per_chip_bytes > V5E_HBM_BYTES_PER_CHIP else "fit_candidate"
            )
            cases.append(
                {
                    "name": name,
                    "data": data,
                    "fsdp": fsdp,
                    "tensor": tensor,
                    "mesh_product": product,
                    "optimistic_state_shards": state_shards,
                    "training_gib_per_chip": per_chip_bytes / 2**30,
                    "prediction": prediction,
                    "observed": observed,
                    "global_batch": batch,
                    "sequence_length": length,
                }
            )

        common = [
            _maxtext_config(),
            f"model_name={self.maxtext_model_name}",
            "base_output_directory=/tmp/me344-capacity",
            "dataset_type=synthetic",
            "steps=2",
            "compile_topology=v5e-8",
            "compile_topology_num_slices=1",
            "enable_checkpointing=false",
            "log_config=false",
        ]

        print(
            "EXPECTED RESULT: TP8 should compile, while the deliberately replicated "
            "DP8 layout should run out of HBM. That OOM confirms the capacity check works."
        )

        tp8_case = cases[0]
        tp8_command = [
            str(self.pretrain_python),
            "-m",
            "maxtext.trainers.pre_train.train_compile",
            *common,
            f"run_name=me344-{self.run_id}-capacity-tp8",
            f"max_target_length={sequence_length}",
            f"per_device_batch_size={global_batch / V5E_CHIPS}",
            "ici_data_parallelism=1",
            "ici_fsdp_parallelism=1",
            "ici_tensor_parallelism=8",
        ]
        tp8 = self._run_logged(
            tp8_command,
            "capacity-tp8-aot.log",
            "capacity-tp8-aot",
            timeout_minutes=10,
            allow_failure=True,
        )
        tp8_text = tp8.log_path.read_text(encoding="utf-8")
        if tp8.returncode == 0:
            tp8_case["observed"] = "fit"
            tp8_case.update(self._parse_compiled_memory(tp8_text))
            if "aot_reported_total_gib_per_chip" not in tp8_case:
                raise RuntimeError("The TP8 AOT compile did not print compiler memory statistics.")
        elif tp8.returncode not in (124, 137) and self._is_memory_failure(tp8_text):
            tp8_case["observed"] = "oom"
            tp8_case.update(self._parse_oom_hbm(tp8_text))
        else:
            raise RuntimeError(
                "The TP8 AOT case failed for a reason other than memory; "
                "inspect capacity-tp8-aot.log before continuing."
            )

        run_name = f"me344-{self.run_id}-capacity-dp8"
        command = [
            str(self.pretrain_python),
            "-m",
            "maxtext.trainers.pre_train.train_compile",
            *common,
            f"run_name={run_name}",
            "max_target_length=128",
            "per_device_batch_size=1",
            "ici_data_parallelism=8",
            "ici_fsdp_parallelism=1",
            "ici_tensor_parallelism=1",
        ]
        result = self._run_logged(
            command,
            "capacity-dp8-aot.log",
            "capacity-dp8-aot",
            timeout_minutes=10,
            allow_failure=True,
        )
        log_text = result.log_path.read_text(encoding="utf-8")
        dp8_case = cases[1]
        if result.returncode == 0:
            dp8_case["observed"] = "fit"
            print("The controlled DP8 case fit. Record it and ask what changed in the compiler or model.")
        elif result.returncode not in (124, 137) and self._is_memory_failure(log_text):
            dp8_case["observed"] = "oom"
            matching = [
                line.strip()
                for line in log_text.splitlines()
                if any(
                    term in line.lower()
                    for term in ("resource_exhausted", "out of memory", "hbm allocation", "hbm usage")
                )
            ]
            dp8_case["evidence_excerpt"] = matching[-1][-500:] if matching else "See capacity-dp8-aot.log"
            print("EXPECTED OOM CONFIRMED: the replicated DP8 state exceeded HBM. This is a successful result.")
        else:
            raise RuntimeError(
                "The controlled AOT case failed for a reason other than memory; "
                "inspect capacity-dp8-aot.log before continuing."
            )

        host_ram_bytes = int(self.reports.get("pretrain", {}).get("host_ram_bytes", 0))
        report = {
            "hardware": {
                "chips": V5E_CHIPS,
                "hbm_decimal_gb_per_chip": V5E_HBM_BYTES_PER_CHIP / 1e9,
                "hbm_gib_per_chip": hbm_gib,
                "hbm_bandwidth_gib_per_second_per_chip": V5E_HBM_BANDWIDTH_GIB_PER_SECOND,
                "ici_bidirectional_gb_per_second_per_chip": V5E_ICI_BIDIRECTIONAL_GB_PER_SECOND,
                "peak_bf16_tflops_per_chip": V5E_PEAK_BF16_TFLOPS_PER_CHIP,
                "aggregate_hbm_gib": V5E_CHIPS * hbm_gib,
                "visible_host_ram_gib": host_ram_bytes / 2**30,
            },
            "model": {
                "parameter_count_estimate": parameter_count,
                "bf16_checkpoint_gib": checkpoint_bytes / 2**30,
                "fp32_parameters_gib": fp32_parameter_bytes / 2**30,
                "persistent_training_state_gib": persistent_training_bytes / 2**30,
                "transient_fp32_gradient_gib": gradient_bytes / 2**30,
                "rough_peak_before_activations_gib": rough_peak_bytes / 2**30,
            },
            "assumptions": [
                "The imported parameter checkpoint is roughly BF16-sized.",
                "MaxText keeps FP32 weights and two FP32 Adam moments; a full FP32 gradient is temporary.",
                "Tensor and FSDP ideally shard dense persistent state; data parallelism replicates it.",
                "Activations, temporary buffers, uneven sharding, padding, and fragmentation are excluded.",
            ],
            "cases": cases,
            "aot_executes_optimizer_steps": False,
        }
        self.capacity_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(
            f"parameters~{parameter_count / 1e9:.3f}B, BF16 checkpoint~{checkpoint_bytes / 2**30:.2f} GiB, "
            f"persistent training state~{persistent_training_bytes / 2**30:.2f} GiB, "
            f"rough peak before activations~{rough_peak_bytes / 2**30:.2f} GiB"
        )
        for case in cases:
            aot_total = case.get("aot_reported_total_gib_per_chip")
            aot_text = f", AOT={aot_total:.2f} GiB/chip" if aot_total is not None else ""
            print(
                f"{case['name']}: mesh={case['mesh_product']}, "
                f"working~{case['training_gib_per_chip']:.2f} GiB/chip, "
                f"predict={case['prediction']}, observed={case['observed']}{aot_text}"
            )
        return self.capacity_path

    def plot_capacity(self) -> None:
        if not self.capacity_path.exists():
            print("No capacity report yet.")
            return
        import matplotlib.pyplot as plt

        report = json.loads(self.capacity_path.read_text(encoding="utf-8"))
        model = report["model"]
        hardware = report["hardware"]
        cases = [case for case in report["cases"] if case["mesh_product"] == V5E_CHIPS]
        figure, axes = plt.subplots(1, 2, figsize=(10, 4.5))

        object_labels = ["BF16\ncheckpoint", "FP32\nweights", "Persistent\ntraining state", "Rough peak\nbefore activations"]
        object_values = [
            model["bf16_checkpoint_gib"],
            model["fp32_parameters_gib"],
            model["persistent_training_state_gib"],
            model["rough_peak_before_activations_gib"],
        ]
        bars = axes[0].bar(object_labels, object_values, color=["#457b9d", "#2a9d8f", "#e9c46a", "#6c757d"])
        axes[0].bar_label(bars, fmt="%.1f", padding=3)
        axes[0].set(ylabel="GiB", title="Whole-model memory estimates")

        case_bars = axes[1].bar(
            [case["name"] for case in cases],
            [case["training_gib_per_chip"] for case in cases],
            color=["#2a9d8f", "#6c757d", "#b23a48"],
        )
        axes[1].axhline(hardware["hbm_gib_per_chip"], color="#111111", linestyle="--", label="HBM/chip")
        axes[1].bar_label(
            case_bars,
            labels=[case["observed"].replace("_", " ") for case in cases],
            padding=3,
        )
        for index, case in enumerate(cases):
            aot_total = case.get("aot_reported_total_gib_per_chip")
            if aot_total is not None:
                axes[1].scatter(
                    index,
                    aot_total,
                    color="#4285F4",
                    marker="D",
                    s=45,
                    zorder=3,
                    label="XLA AOT total",
                )
                axes[1].annotate(
                    f"{aot_total:.1f}",
                    (index, aot_total),
                    xytext=(7, 0),
                    textcoords="offset points",
                    va="center",
                )
        axes[1].set(ylabel="GiB per chip", title="Training working-set estimate")
        axes[1].legend()
        for axis in axes:
            axis.grid(axis="y", alpha=0.2)
        figure.suptitle("Capacity prediction and controlled OOM")
        figure.tight_layout()
        figure.savefig(self.artifact_dir / "capacity.svg")
        plt.show()

    def run_sweep(
        self,
        global_batches: tuple[int, ...] = (4, 8, 16, 32, 64, 128, 256, 512),
        sequence_length: int = 256,
        steps: int = 8,
    ) -> Path:
        """AOT-search the batch/HBM frontier, then measure every shape that fits."""
        if not self.config.run_tpu:
            self._skip("AOT batch and HBM frontier")
            return self.sweep_path
        if not global_batches or any(batch <= 0 for batch in global_batches):
            raise ValueError("global_batches must contain positive values.")
        if any(right <= left for left, right in zip(global_batches, global_batches[1:])):
            raise ValueError("global_batches must be strictly increasing.")

        rows_out = []
        device_count = int(self.reports["pretrain"]["device_count"])
        observed_hbm_limit = V5E_HBM_BYTES_PER_CHIP / 2**30
        for global_batch in global_batches:
            per_device_batch = global_batch / device_count
            run_name = f"me344-{self.run_id}-sweep-gb{global_batch}"
            metrics_path = self.artifact_dir / f"sweep-gb{global_batch}.jsonl"
            common = [
                _maxtext_config(),
                f"model_name={self.maxtext_model_name}",
                f"base_output_directory={self.config.output_directory}",
                f"run_name={run_name}",
                "dataset_type=synthetic",
                "reuse_example_batch=true",
                f"steps={steps}",
                "ici_fsdp_parallelism=1",
                "ici_tensor_parallelism=8",
                f"max_target_length={sequence_length}",
                f"per_device_batch_size={per_device_batch}",
                "learning_rate=0",
                "enable_checkpointing=false",
                "log_config=false",
            ]
            aot_command = [
                str(self.pretrain_python),
                "-m",
                "maxtext.trainers.pre_train.train_compile",
                *common,
                "compile_topology=v5e-8",
                "compile_topology_num_slices=1",
            ]
            aot = self._run_logged(
                aot_command,
                f"sweep-aot-gb{global_batch}.log",
                f"sweep-aot-gb{global_batch}",
                allow_failure=True,
            )
            aot_text = aot.log_path.read_text(encoding="utf-8")
            row = {
                "global_batch": global_batch,
                "per_device_batch": per_device_batch,
                "sequence_length": sequence_length,
                "aot_status": "",
                "benchmark_status": "",
                "aot_wall_sec": aot.seconds,
                "aot_cost_usd": aot.estimated_cost,
                "aot_reported_argument_gib_per_chip": "",
                "aot_reported_output_gib_per_chip": "",
                "aot_reported_temporary_gib_per_chip": "",
                "aot_reported_alias_gib_per_chip": "",
                "aot_reported_net_output_gib_per_chip": "",
                "aot_reported_total_gib_per_chip": "",
                "aot_reported_total_percent_of_limit": "",
                "aot_reported_host_temporary_gib": "",
                "oom_required_temporary_gib_per_chip": "",
                "oom_available_hbm_gib_per_chip": "",
                "runtime_hbm_snapshot_gib_per_chip": "",
                "runtime_hbm_limit_gib_per_chip": observed_hbm_limit,
                "runtime_hbm_snapshot_percent_of_limit": "",
                "command_wall_sec": "",
                "runtime_startup_compile_overhead_sec": "",
                "first_step_sec": "",
                "steady_step_sec": "",
                "total_tokens_per_sec": "",
                "tflops_per_sec_per_device": "",
                "run_cost_usd": "",
                "total_shape_cost_usd": aot.estimated_cost,
                "short_run_cost_per_million_tokens_usd": "",
                "steady_cost_per_million_tokens_usd": "",
            }
            if aot.returncode:
                if aot.returncode in (124, 137) or not self._is_memory_failure(aot_text):
                    raise RuntimeError(
                        f"Batch {global_batch} AOT failed for a reason other than HBM; inspect {aot.log_path.name}."
                    )
                row["aot_status"] = "oom"
                row.update(self._parse_oom_hbm(aot_text))
                rows_out.append(row)
                required = row["oom_required_temporary_gib_per_chip"]
                detail = f": {required:.2f} GiB of temporaries" if required != "" else ""
                print(
                    f"EXPECTED OOM CONFIRMED: global batch {global_batch} is the first shape "
                    f"past the AOT memory boundary{detail}; the observed per-chip limit is "
                    f"{observed_hbm_limit:.2f} GiB. Finding this boundary is the goal."
                )
                break

            memory = self._parse_compiled_memory(aot_text)
            if not memory:
                raise RuntimeError(f"Batch {global_batch} AOT did not produce compiler memory stats.")
            row.update(memory)
            row["aot_status"] = "fit"
            row["aot_reported_total_percent_of_limit"] = (
                100 * float(row["aot_reported_total_gib_per_chip"]) / observed_hbm_limit
            )

            command = [
                str(self.pretrain_python),
                "-m",
                "maxtext.trainers.pre_train.train",
                *common,
                f"metrics_file={metrics_path}",
            ]
            result = self._run_logged(command, f"{run_name}.log", f"sweep-gb{global_batch}")
            self._require_final_step(metrics_path, steps - 1)
            metrics = self._metric_rows(metrics_path)
            steady = metrics[-min(4, len(metrics)) :]
            measured_step_seconds = sum(item["perf/step_time_seconds"] for item in metrics)
            total_tokens = global_batch * sequence_length * steps
            throughput = (
                statistics.fmean(item["perf/per_device_tokens_per_sec"] for item in steady)
                * device_count
            )
            runtime_memory = self._parse_runtime_hbm(result.log_path.read_text(encoding="utf-8"))
            if runtime_memory:
                observed_hbm_limit = runtime_memory["runtime_hbm_limit_gib_per_chip"]
                row.update(runtime_memory)
                row["runtime_hbm_limit_gib_per_chip"] = observed_hbm_limit
                row["aot_reported_total_percent_of_limit"] = (
                    100 * float(row["aot_reported_total_gib_per_chip"]) / observed_hbm_limit
                )
            total_shape_cost = aot.estimated_cost + result.estimated_cost
            row.update(
                {
                    "benchmark_status": "measured",
                    "command_wall_sec": result.seconds,
                    "runtime_startup_compile_overhead_sec": max(
                        0.0, result.seconds - measured_step_seconds
                    ),
                    "first_step_sec": metrics[0]["perf/step_time_seconds"],
                    "steady_step_sec": statistics.fmean(
                        item["perf/step_time_seconds"] for item in steady
                    ),
                    "total_tokens_per_sec": throughput,
                    "tflops_per_sec_per_device": statistics.fmean(
                        item["perf/per_device_tflops_per_sec"] for item in steady
                    ),
                    "run_cost_usd": result.estimated_cost,
                    "total_shape_cost_usd": total_shape_cost,
                    "short_run_cost_per_million_tokens_usd": total_shape_cost
                    * 1_000_000
                    / total_tokens,
                    "steady_cost_per_million_tokens_usd": V5E_8_USD_PER_HOUR
                    * 1_000_000
                    / (3600 * throughput),
                }
            )
            rows_out.append(row)

        if not any(row["benchmark_status"] == "measured" for row in rows_out):
            raise RuntimeError("No batch shape compiled and executed successfully.")
        if rows_out[-1]["aot_status"] != "oom":
            print(f"No OOM through global batch {global_batches[-1]}; this is a lower bound, not the HBM limit.")
        for row in rows_out:
            row["runtime_hbm_limit_gib_per_chip"] = observed_hbm_limit
            if row["aot_reported_total_gib_per_chip"] != "":
                row["aot_reported_total_percent_of_limit"] = (
                    100 * float(row["aot_reported_total_gib_per_chip"]) / observed_hbm_limit
                )
            if row["runtime_hbm_snapshot_gib_per_chip"] != "":
                row["runtime_hbm_snapshot_percent_of_limit"] = (
                    100 * float(row["runtime_hbm_snapshot_gib_per_chip"]) / observed_hbm_limit
                )
        with self.sweep_path.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(rows_out[0]))
            writer.writeheader()
            writer.writerows(rows_out)
        print(self.sweep_path.read_text(encoding="utf-8"))
        return self.sweep_path

    def plot_sweep(self) -> None:
        if not self.sweep_path.exists():
            print("No sweep CSV yet.")
            return
        import matplotlib.pyplot as plt

        rows = list(csv.DictReader(self.sweep_path.open(encoding="utf-8")))
        measured = [row for row in rows if row["benchmark_status"] == "measured"]
        x = [int(row["global_batch"]) for row in measured]

        def values(key: str) -> list[float]:
            return [float(row[key]) for row in measured]

        figure, axes = plt.subplots(2, 3, figsize=(13, 7.5))
        axes = axes.ravel()
        axes[0].plot(x, values("total_tokens_per_sec"), "o-", color="#16697a")
        axes[0].set(ylabel="Tokens/s", title="Throughput")
        axes[1].plot(
            x,
            values("aot_wall_sec"),
            "o--",
            label="AOT compile",
            color="#b23a48",
        )
        axes[1].plot(
            x,
            values("runtime_startup_compile_overhead_sec"),
            "o--",
            label="runtime startup + JIT",
            color="#457b9d",
        )
        axes[1].plot(x, values("first_step_sec"), "o-", label="first step", color="#f4a261")
        axes[1].plot(x, values("steady_step_sec"), "o-", label="steady step", color="#2a9d8f")
        axes[1].set(ylabel="Seconds (log scale)", title="Wall-time decomposition")
        axes[1].set_yscale("log")
        axes[1].legend()
        axes[2].plot(x, values("tflops_per_sec_per_device"), "o-", color="#6a4c93")
        axes[2].axhline(V5E_PEAK_BF16_TFLOPS_PER_CHIP, color="#111111", linestyle="--", label="BF16 peak")
        axes[2].set(ylabel="TFLOP/s/device", title="Device work")
        axes[2].legend()
        axes[3].plot(
            x,
            values("short_run_cost_per_million_tokens_usd"),
            "o--",
            label="8-step + AOT",
            color="#e76f51",
        )
        axes[3].plot(
            x,
            values("steady_cost_per_million_tokens_usd"),
            "o-",
            label="steady-state floor",
            color="#2a9d8f",
        )
        axes[3].set(ylabel="USD / 1M tokens", title="Cost bounds")
        axes[3].set_yscale("log")
        axes[3].legend()

        positions = list(range(len(rows)))
        fit_rows = [row for row in rows if row["aot_status"] == "fit"]
        fit_positions = [index for index, row in enumerate(rows) if row["aot_status"] == "fit"]
        argument = [float(row["aot_reported_argument_gib_per_chip"]) for row in fit_rows]
        temporary = [float(row["aot_reported_temporary_gib_per_chip"]) for row in fit_rows]
        net_output = [float(row["aot_reported_net_output_gib_per_chip"]) for row in fit_rows]
        axes[4].bar(fit_positions, argument, label="arguments/state", color="#457b9d")
        axes[4].bar(fit_positions, temporary, bottom=argument, label="temporaries", color="#e9c46a")
        axes[4].bar(
            fit_positions,
            net_output,
            bottom=[left + right for left, right in zip(argument, temporary)],
            label="net outputs",
            color="#2a9d8f",
        )
        limit = min(float(row["runtime_hbm_limit_gib_per_chip"]) for row in rows)
        axes[4].axhline(limit, color="#111111", linestyle="--", label="HBM limit")
        runtime_positions = [
            index for index, row in enumerate(rows) if row["runtime_hbm_snapshot_gib_per_chip"] != ""
        ]
        runtime_rows = [row for row in rows if row["runtime_hbm_snapshot_gib_per_chip"] != ""]
        axes[4].plot(
            runtime_positions,
            [float(row["runtime_hbm_snapshot_gib_per_chip"]) for row in runtime_rows],
            "o-",
            color="#6a4c93",
            label="post-init snapshot",
        )
        for index, row in enumerate(rows):
            if row["aot_status"] == "oom":
                required = row["oom_required_temporary_gib_per_chip"]
                axes[4].scatter(
                    index,
                    float(required) if required != "" else limit,
                    marker="x",
                    s=90,
                    color="#b23a48",
                    label="OOM temporary requirement",
                )
        axes[4].set_xticks(positions, [row["global_batch"] for row in rows])
        axes[4].set(ylabel="GiB/chip", title="Compiler accounting vs snapshot")
        axes[4].legend(fontsize=8)

        memory = values("aot_reported_total_gib_per_chip")
        axes[5].plot(memory, values("total_tokens_per_sec"), "o-", color="#16697a")
        for batch, memory_gib, throughput in zip(x, memory, values("total_tokens_per_sec")):
            axes[5].annotate(str(batch), (memory_gib, throughput), xytext=(4, 4), textcoords="offset points")
        axes[5].set(
            xlabel="Compiler-reported total GiB/chip",
            ylabel="Tokens/s",
            title="Memory-throughput frontier",
        )
        for index, axis in enumerate(axes):
            if index < 4:
                axis.set_xlabel("Global batch")
                axis.set_xscale("log", base=2)
                axis.set_xticks(x, x)
            axis.grid(alpha=0.25)
        figure.tight_layout()
        figure.savefig(self.artifact_dir / "sweep.svg")
        plt.show()

    def run_xla_flag_ab(
        self,
        sequence_length: int = 256,
        global_batch: int = 4,
        steps: int = 8,
        repeats: int = 2,
    ) -> Path:
        """Compile and time one async-collective flag while holding the workload fixed."""
        if not self.config.run_tpu:
            self._skip("XLA async-collective flag A/B")
            return self.xla_flag_path
        if steps < 5:
            raise ValueError("Use at least five steps so the comparison has steady-state samples.")
        if repeats < 2:
            raise ValueError("Use at least two repeats so run order can be alternated.")

        device_count = int(self.reports["pretrain"]["device_count"])
        per_device_batch = global_batch / device_count
        common_flags = (
            "--xla_enable_async_all_gather=true "
            "--xla_tpu_enable_async_collective_fusion_fuse_all_gather=true"
        )
        cases = (("fusion_off", False), ("fusion_on", True))
        common = [
            _maxtext_config(),
            f"model_name={self.maxtext_model_name}",
            "base_output_directory=/tmp/me344-xla-flag",
            "dataset_type=synthetic",
            "reuse_example_batch=true",
            f"steps={steps}",
            "ici_fsdp_parallelism=1",
            "ici_tensor_parallelism=8",
            f"max_target_length={sequence_length}",
            f"per_device_batch_size={per_device_batch}",
            "learning_rate=0",
            "enable_dropout=false",
            "init_weights_seed=0",
            "data_shuffle_seed=0",
            "enable_checkpointing=false",
            "log_config=false",
        ]
        evidence: dict[str, dict] = {}
        hlo_summaries: dict[str, dict] = {}

        for label, enabled in cases:
            flags = (
                f"{common_flags} "
                f"--xla_tpu_enable_async_collective_fusion={str(enabled).lower()}"
            )
            hlo_dir = self.artifact_dir / f"xla-hlo-{label}"
            shutil.rmtree(hlo_dir, ignore_errors=True)
            hlo_dir.mkdir(parents=True)
            dump_flags = " ".join(
                value
                for value in (
                    os.environ.get("XLA_FLAGS", ""),
                    f"--xla_dump_to={hlo_dir}",
                    "--xla_dump_hlo_as_text",
                )
                if value
            )
            aot_command = [
                str(self.pretrain_python),
                "-m",
                "maxtext.trainers.pre_train.train_compile",
                *common,
                f"run_name=me344-{self.run_id}-{label}-aot",
                "compile_topology=v5e-8",
                "compile_topology_num_slices=1",
                f"compile_xla_flags={flags}",
            ]
            try:
                aot = self._run_logged(
                    aot_command,
                    f"xla-{label}-aot.log",
                    f"xla-{label}-aot",
                    extra_env={"XLA_FLAGS": dump_flags},
                )
                aot_text = aot.log_path.read_text(encoding="utf-8")
                memory = self._parse_compiled_memory(aot_text)
                if not memory:
                    raise RuntimeError(f"{label} AOT did not produce compiler memory stats.")
                hlo_summary = self._summarize_hlo_dump(hlo_dir)
                if hlo_summary.get("status") != "found":
                    raise RuntimeError(f"{label} AOT did not produce an inspectable HLO dump.")
            finally:
                shutil.rmtree(hlo_dir, ignore_errors=True)
            hlo_summaries[label] = hlo_summary
            evidence[label] = {
                "enabled": enabled,
                "flags": flags,
                "aot_seconds": aot.seconds,
                "aot_cost_usd": aot.estimated_cost,
                "memory": memory,
                "hlo": hlo_summary,
                "runs": [],
            }

        for repeat in range(repeats):
            ordered_cases = cases if repeat % 2 == 0 else tuple(reversed(cases))
            for label, _ in ordered_cases:
                metrics_path = self.artifact_dir / f"xla-{label}-r{repeat + 1}.jsonl"
                flags = evidence[label]["flags"]
                command = [
                    str(self.pretrain_python),
                    "-m",
                    "maxtext.trainers.pre_train.train",
                    *common,
                    f"run_name=me344-{self.run_id}-{label}-r{repeat + 1}",
                    f"compile_xla_flags={flags}",
                    f"metrics_file={metrics_path}",
                ]
                result = self._run_logged(
                    command,
                    f"xla-{label}-r{repeat + 1}.log",
                    f"xla-{label}-r{repeat + 1}",
                )
                self._require_final_step(metrics_path, steps - 1)
                metrics = self._metric_rows(metrics_path)
                steady = metrics[-min(4, len(metrics)) :]
                losses = [
                    float(item["learning/loss"])
                    for item in metrics
                    if "learning/loss" in item
                ]
                if not losses or not all(math.isfinite(value) for value in losses):
                    raise RuntimeError(f"{label} repeat {repeat + 1} did not produce finite loss.")
                log_text = result.log_path.read_text(encoding="utf-8")
                runtime_memory = self._parse_runtime_hbm(log_text)
                measured_step_seconds = sum(
                    item["perf/step_time_seconds"] for item in metrics
                )
                evidence[label]["runs"].append(
                    {
                        "repeat": repeat + 1,
                        "command_wall_sec": result.seconds,
                        "startup_compile_overhead_sec": max(
                            0.0, result.seconds - measured_step_seconds
                        ),
                        "first_step_sec": metrics[0]["perf/step_time_seconds"],
                        "steady_step_sec": statistics.fmean(
                            item["perf/step_time_seconds"] for item in steady
                        ),
                        "total_tokens_per_sec": statistics.fmean(
                            item["perf/per_device_tokens_per_sec"] for item in steady
                        )
                        * device_count,
                        "mean_loss": statistics.fmean(losses),
                        "post_init_hbm_snapshot_gib_per_chip": runtime_memory.get(
                            "runtime_hbm_snapshot_gib_per_chip", ""
                        ),
                        "run_cost_usd": result.estimated_cost,
                    }
                )

        rows_out = []
        for label, _ in cases:
            case = evidence[label]
            runs = case["runs"]

            def median(key: str) -> float:
                return statistics.median(float(run[key]) for run in runs)

            hbm_values = [
                float(run["post_init_hbm_snapshot_gib_per_chip"])
                for run in runs
                if run["post_init_hbm_snapshot_gib_per_chip"] != ""
            ]
            rows_out.append(
                {
                    "case": label,
                    "async_collective_fusion": case["enabled"],
                    "compile_xla_flags": case["flags"],
                    "global_batch": global_batch,
                    "sequence_length": sequence_length,
                    "steps": steps,
                    "repeats": repeats,
                    "aot_wall_sec": case["aot_seconds"],
                    "command_wall_sec_median": median("command_wall_sec"),
                    "startup_compile_overhead_sec_median": median(
                        "startup_compile_overhead_sec"
                    ),
                    "first_step_sec_median": median("first_step_sec"),
                    "steady_step_sec": median("steady_step_sec"),
                    "steady_step_sec_min": min(
                        float(run["steady_step_sec"]) for run in runs
                    ),
                    "steady_step_sec_max": max(
                        float(run["steady_step_sec"]) for run in runs
                    ),
                    "total_tokens_per_sec": median("total_tokens_per_sec"),
                    "tokens_per_sec_min": min(
                        float(run["total_tokens_per_sec"]) for run in runs
                    ),
                    "tokens_per_sec_max": max(
                        float(run["total_tokens_per_sec"]) for run in runs
                    ),
                    "mean_loss": median("mean_loss"),
                    "compiler_total_gib_per_chip": case["memory"][
                        "aot_reported_total_gib_per_chip"
                    ],
                    "compiler_temporary_gib_per_chip": case["memory"][
                        "aot_reported_temporary_gib_per_chip"
                    ],
                    "post_init_hbm_snapshot_gib_per_chip": (
                        statistics.median(hbm_values) if hbm_values else ""
                    ),
                    "hlo_sha256": case["hlo"].get("sha256", ""),
                    "hlo_all_gather_start_count": case["hlo"].get(
                        "all_gather_start_count", 0
                    ),
                    "hlo_all_gather_done_count": case["hlo"].get(
                        "all_gather_done_count", 0
                    ),
                    "hlo_all_gather_count": case["hlo"].get("all_gather_count", 0),
                    "total_cost_usd": case["aot_cost_usd"]
                    + sum(float(run["run_cost_usd"]) for run in runs),
                }
            )

        loss_delta = abs(
            float(rows_out[0]["mean_loss"]) - float(rows_out[1]["mean_loss"])
        )
        if loss_delta > 1e-3:
            raise RuntimeError(
                f"XLA flag cases changed the fixed-workload mean loss by {loss_delta:.6f}."
            )
        for row in rows_out:
            row["cross_case_mean_loss_delta"] = loss_delta

        with self.xla_flag_path.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(rows_out[0]))
            writer.writeheader()
            writer.writerows(rows_out)
        self.xla_hlo_summary_path.write_text(
            json.dumps(hlo_summaries, indent=2) + "\n",
            encoding="utf-8",
        )
        print(self.xla_flag_path.read_text(encoding="utf-8"))
        return self.xla_flag_path

    @staticmethod
    def _summarize_hlo_dump(hlo_dir: Path) -> dict:
        """Summarize the largest optimized train-step HLO without retaining the full dump."""
        candidates = [
            path
            for path in hlo_dir.rglob("*after_optimizations.txt")
            if "jit_train_step" in path.name
        ]
        if not candidates:
            candidates = list(hlo_dir.rglob("*.txt"))
        if not candidates:
            return {"status": "missing"}
        hlo_path = max(candidates, key=lambda path: path.stat().st_size)
        hlo_text = hlo_path.read_text(encoding="utf-8", errors="ignore")
        lower = hlo_text.lower()
        relevant_lines = [
            line.strip()
            for line in hlo_text.splitlines()
            if "all-gather" in line.lower() or "async" in line.lower()
        ][:80]
        return {
            "status": "found",
            "source_file": hlo_path.name,
            "bytes": hlo_path.stat().st_size,
            "sha256": hashlib.sha256(hlo_text.encode()).hexdigest(),
            "all_gather_count": lower.count("all-gather"),
            "all_gather_start_count": lower.count("all-gather-start"),
            "all_gather_done_count": lower.count("all-gather-done"),
            "relevant_lines": relevant_lines,
        }

    def plot_xla_flag_ab(self) -> None:
        if not self.xla_flag_path.exists():
            print("No XLA flag A/B results yet.")
            return
        import matplotlib.pyplot as plt

        rows = list(csv.DictReader(self.xla_flag_path.open(encoding="utf-8")))
        labels = ["fusion off", "fusion on"]
        colors = ["#EA4335", "#34A853"]
        step_ms = [1000 * float(row["steady_step_sec"]) for row in rows]
        throughput = [float(row["total_tokens_per_sec"]) for row in rows]
        compiler_memory = [
            float(row["compiler_total_gib_per_chip"])
            if row["compiler_total_gib_per_chip"]
            else math.nan
            for row in rows
        ]
        runtime_memory = [
            float(row["post_init_hbm_snapshot_gib_per_chip"])
            if row["post_init_hbm_snapshot_gib_per_chip"]
            else math.nan
            for row in rows
        ]

        figure, axes = plt.subplots(1, 4, figsize=(14, 3.8))
        bars = axes[0].bar(labels, step_ms, color=colors)
        axes[0].bar_label(bars, fmt="%.1f ms", padding=3)
        axes[0].set(title="Steady step time", ylabel="Milliseconds")

        bars = axes[1].bar(labels, throughput, color=colors)
        axes[1].bar_label(bars, fmt="%.0f", padding=3)
        axes[1].set(title="Throughput", ylabel="Tokens/s")

        x = range(len(labels))
        axes[2].bar(x, compiler_memory, color=colors, alpha=0.8, label="compiler total")
        axes[2].plot(x, runtime_memory, "ko", label="post-init snapshot")
        axes[2].set_xticks(list(x), labels)
        axes[2].set(title="HBM evidence", ylabel="GiB/chip")
        axes[2].legend(fontsize=8)

        width = 0.36
        starts = [int(row["hlo_all_gather_start_count"]) for row in rows]
        dones = [int(row["hlo_all_gather_done_count"]) for row in rows]
        axes[3].bar([value - width / 2 for value in x], starts, width, label="start")
        axes[3].bar([value + width / 2 for value in x], dones, width, label="done")
        axes[3].set_xticks(list(x), labels)
        axes[3].set(title="Optimized HLO evidence", ylabel="Operation tokens")
        axes[3].legend(fontsize=8)

        for axis in axes:
            axis.grid(axis="y", alpha=0.2)
        figure.suptitle("Repeated one-variable XLA flag experiment")
        figure.tight_layout()
        figure.savefig(self.artifact_dir / "xla_flag_ab.svg")
        plt.show()

    @staticmethod
    def _miniperf_row(
        *,
        track: str,
        order: int,
        name: str,
        source: str,
        global_batch: int,
        sequence_length: int,
        tensor_parallelism: int,
        fsdp_parallelism: int,
        remat_policy: str,
        gradient_accumulation_steps: int,
        attention: str,
        hbm_limit: float,
    ) -> dict:
        return {
            "track": track,
            "trial_order": order,
            "name": name,
            "source": source,
            "global_batch": global_batch,
            "sequence_length": sequence_length,
            "tensor_parallelism": tensor_parallelism,
            "fsdp_parallelism": fsdp_parallelism,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "remat_policy": remat_policy,
            "attention": attention,
            "aot_status": "",
            "benchmark_status": "",
            "correctness_gate": "not_run",
            "failure_summary": "",
            "aot_log": "",
            "benchmark_log": "",
            "metrics_file": "",
            "aot_wall_sec": "",
            "aot_cost_usd": "",
            "aot_reported_argument_gib_per_chip": "",
            "aot_reported_output_gib_per_chip": "",
            "aot_reported_temporary_gib_per_chip": "",
            "aot_reported_alias_gib_per_chip": "",
            "aot_reported_net_output_gib_per_chip": "",
            "aot_reported_total_gib_per_chip": "",
            "aot_reported_total_percent_of_limit": "",
            "aot_reported_host_temporary_gib": "",
            "oom_required_temporary_gib_per_chip": "",
            "oom_available_hbm_gib_per_chip": "",
            "runtime_hbm_snapshot_gib_per_chip": "",
            "runtime_hbm_limit_gib_per_chip": hbm_limit,
            "runtime_hbm_snapshot_percent_of_limit": "",
            "command_wall_sec": "",
            "runtime_startup_compile_overhead_sec": "",
            "first_step_sec": "",
            "steady_step_sec": "",
            "total_tokens_per_sec": "",
            "maxtext_total_tokens_per_sec": "",
            "tflops_per_sec_per_device": "",
            "peak_compute_proxy_percent": "",
            "mean_loss": "",
            "mean_total_weights": "",
            "run_cost_usd": "",
            "total_trial_cost_usd": "",
            "steady_cost_per_million_tokens_usd": "",
        }

    @staticmethod
    def _failure_summary(log_text: str) -> str:
        interesting = [
            line.strip()
            for line in log_text.splitlines()
            if any(
                term in line.lower()
                for term in ("error", "valueerror", "indivisible", "resource_exhausted", "traceback")
            )
        ]
        return (interesting[-1] if interesting else log_text.splitlines()[-1] if log_text else "unknown")[:500]

    def _run_miniperf_trial(
        self,
        *,
        track: str,
        order: int,
        name: str,
        global_batch: int,
        sequence_length: int,
        tensor_parallelism: int,
        fsdp_parallelism: int,
        remat_policy: str,
        gradient_accumulation_steps: int,
        attention: str,
        steps: int,
        warmup_steps: int,
        hbm_limit: float,
        run_live: bool,
    ) -> dict:
        device_count = int(self.reports["pretrain"]["device_count"])
        per_device_batch = global_batch / (device_count * gradient_accumulation_steps)
        safe_name = re.sub(r"[^a-z0-9-]+", "-", name.lower()).strip("-")
        prefix = f"miniperf-{track}-{order:02d}-{safe_name}"
        run_name = f"me344-{self.run_id}-{prefix}"
        aot_log = f"{prefix}-aot.log"
        benchmark_log = f"{prefix}.log"
        metrics_file = f"{prefix}.jsonl"
        metrics_path = self.artifact_dir / metrics_file
        row = self._miniperf_row(
            track=track,
            order=order,
            name=name,
            source="miniperf",
            global_batch=global_batch,
            sequence_length=sequence_length,
            tensor_parallelism=tensor_parallelism,
            fsdp_parallelism=fsdp_parallelism,
            remat_policy=remat_policy,
            gradient_accumulation_steps=gradient_accumulation_steps,
            attention=attention,
            hbm_limit=hbm_limit,
        )
        row.update({"aot_log": aot_log, "benchmark_log": benchmark_log, "metrics_file": metrics_file})
        common = [
            _maxtext_config(),
            f"model_name={self.maxtext_model_name}",
            f"base_output_directory={self.config.output_directory}",
            f"run_name={run_name}",
            "dataset_type=synthetic",
            "reuse_example_batch=true",
            f"steps={steps}",
            f"ici_fsdp_parallelism={fsdp_parallelism}",
            f"ici_tensor_parallelism={tensor_parallelism}",
            f"max_target_length={sequence_length}",
            f"per_device_batch_size={per_device_batch}",
            f"gradient_accumulation_steps={gradient_accumulation_steps}",
            f"remat_policy={remat_policy}",
            f"attention={attention}",
            "data_shuffle_seed=0",
            "init_weights_seed=0",
            "learning_rate=0",
            "enable_checkpointing=false",
            "log_config=false",
        ]
        aot = self._run_logged(
            [
                str(self.pretrain_python),
                "-m",
                "maxtext.trainers.pre_train.train_compile",
                *common,
                "compile_topology=v5e-8",
                "compile_topology_num_slices=1",
            ],
            aot_log,
            prefix + "-aot",
            allow_failure=True,
        )
        aot_text = aot.log_path.read_text(encoding="utf-8")
        row.update({"aot_wall_sec": aot.seconds, "aot_cost_usd": aot.estimated_cost})
        if aot.returncode:
            if aot.returncode in {124, 137}:
                raise RuntimeError(f"{name}: AOT timed out or was killed; this is not an OOM result.")
            row["aot_status"] = "oom" if self._is_memory_failure(aot_text) else "invalid"
            row["failure_summary"] = self._failure_summary(aot_text)
            row.update(self._parse_oom_hbm(aot_text))
            row["total_trial_cost_usd"] = aot.estimated_cost
            return row

        memory = self._parse_compiled_memory(aot_text)
        if not memory:
            raise RuntimeError(f"{name}: AOT succeeded without compiler memory statistics.")
        row.update(memory)
        row["aot_status"] = "fit"
        row["aot_reported_total_percent_of_limit"] = (
            100 * float(row["aot_reported_total_gib_per_chip"]) / hbm_limit
        )
        if not run_live:
            row["total_trial_cost_usd"] = aot.estimated_cost
            return row

        result = self._run_logged(
            [
                str(self.pretrain_python),
                "-m",
                "maxtext.trainers.pre_train.train",
                *common,
                f"metrics_file={metrics_path}",
            ],
            benchmark_log,
            prefix,
        )
        self._require_final_step(metrics_path, steps - 1)
        metrics = self._metric_rows(metrics_path)
        measured = metrics[warmup_steps:]
        if len(measured) < 4:
            raise RuntimeError(f"{name}: fewer than four measured steps remain after warmup.")
        measured_step_seconds = sum(float(item["perf/step_time_seconds"]) for item in metrics)
        steady_step = statistics.fmean(float(item["perf/step_time_seconds"]) for item in measured)
        tokens_per_step = global_batch * sequence_length
        throughput = tokens_per_step / steady_step
        maxtext_throughput = (
            statistics.fmean(float(item["perf/per_device_tokens_per_sec"]) for item in measured)
            * device_count
        )
        tflops = statistics.fmean(float(item["perf/per_device_tflops_per_sec"]) for item in measured)
        runtime_memory = self._parse_runtime_hbm(result.log_path.read_text(encoding="utf-8"))
        if runtime_memory:
            row.update(runtime_memory)
            hbm_limit = runtime_memory["runtime_hbm_limit_gib_per_chip"]
            row["aot_reported_total_percent_of_limit"] = (
                100 * float(row["aot_reported_total_gib_per_chip"]) / hbm_limit
            )
        total_cost = aot.estimated_cost + result.estimated_cost
        row.update(
            {
                "benchmark_status": "measured",
                "command_wall_sec": result.seconds,
                "runtime_startup_compile_overhead_sec": max(0.0, result.seconds - measured_step_seconds),
                "first_step_sec": float(metrics[0]["perf/step_time_seconds"]),
                "steady_step_sec": steady_step,
                "total_tokens_per_sec": throughput,
                "maxtext_total_tokens_per_sec": maxtext_throughput,
                "tflops_per_sec_per_device": tflops,
                "peak_compute_proxy_percent": 100 * tflops / V5E_PEAK_BF16_TFLOPS_PER_CHIP,
                "mean_loss": statistics.fmean(float(item["learning/loss"]) for item in measured),
                "mean_total_weights": statistics.fmean(
                    float(item["learning/total_weights"]) for item in measured
                ),
                "run_cost_usd": result.estimated_cost,
                "total_trial_cost_usd": total_cost,
                "steady_cost_per_million_tokens_usd": V5E_8_USD_PER_HOUR
                * 1_000_000
                / (3600 * throughput),
            }
        )
        return row

    @staticmethod
    def _normalize_miniperf_trials(extra_trials: tuple[dict, ...]) -> list[dict]:
        defaults = [
            {"name": "tp8-full", "tensor": 8, "fsdp": 1, "remat": "full"},
            {"name": "tp8-minimal", "tensor": 8, "fsdp": 1, "remat": "minimal"},
            {"name": "tp4-fsdp2", "tensor": 4, "fsdp": 2, "remat": "full"},
            {"name": "fsdp8", "tensor": 1, "fsdp": 8, "remat": "full"},
        ]
        if len(extra_trials) > 4:
            raise ValueError("MiniPerf permits at most four extra trials per paid run.")
        trials = defaults + [dict(trial) for trial in extra_trials]
        names = set()
        normalized = []
        for trial in trials:
            name = str(trial.get("name", ""))
            if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,31}", name) or name in names:
                raise ValueError(f"MiniPerf trial names must be unique lowercase slugs: {name!r}")
            names.add(name)
            tensor = int(trial.get("tensor", 8))
            fsdp = int(trial.get("fsdp", 1))
            remat = str(trial.get("remat", "full"))
            accumulation = int(trial.get("accumulation", 1))
            attention = str(trial.get("attention", "autoselected"))
            if tensor * fsdp != V5E_CHIPS or tensor not in {1, 2, 4, 8}:
                raise ValueError(f"{name}: tensor * fsdp must equal 8 using powers of two.")
            if remat not in MINIPERF_REMAT_POLICIES:
                raise ValueError(f"{name}: unsupported remat policy {remat!r}.")
            if accumulation not in {1, 2, 4}:
                raise ValueError(f"{name}: accumulation must be 1, 2, or 4.")
            if attention not in MINIPERF_ATTENTION_TYPES:
                raise ValueError(f"{name}: attention must be autoselected or dot_product.")
            normalized.append(
                {
                    "name": name,
                    "tensor": tensor,
                    "fsdp": fsdp,
                    "remat": remat,
                    "accumulation": accumulation,
                    "attention": attention,
                }
            )
        return normalized

    def run_miniperf(
        self,
        *,
        enabled: bool = False,
        extra_trials: tuple[dict, ...] = (),
        max_frontier_probes: int = 5,
        sequence_length: int = 256,
        steps: int = 12,
        warmup_steps: int = 4,
    ) -> dict:
        """Run the optional fixed-work MiniPerf challenge and emit a portable scorecard."""
        if not enabled:
            print("SKIPPED optional MiniPerf challenge; set ME344_RUN_MINIPERF=1 to enter.")
            return {}
        if not self.config.run_tpu:
            self._skip("optional MiniPerf challenge")
            return {}
        if not 1 <= max_frontier_probes <= 5:
            raise ValueError("max_frontier_probes must be between 1 and 5.")
        if steps < warmup_steps + 4:
            raise ValueError("MiniPerf needs at least four measured steps after warmup.")
        if not self.sweep_path.exists():
            raise RuntimeError("Run the geometric AOT sweep before MiniPerf.")

        sweep = list(csv.DictReader(self.sweep_path.open(encoding="utf-8")))
        measured_sweep = [row for row in sweep if row["benchmark_status"] == "measured"]
        fixed_batch = 32
        if not any(int(row["global_batch"]) == fixed_batch for row in measured_sweep):
            raise RuntimeError("MiniPerf v1 requires the measured global-batch-32 sweep point.")
        hbm_limit = min(float(row["runtime_hbm_limit_gib_per_chip"]) for row in sweep)
        fit_batches = [int(row["global_batch"]) for row in sweep if row["aot_status"] == "fit"]
        oom_batches = [int(row["global_batch"]) for row in sweep if row["aot_status"] == "oom"]
        if not fit_batches or not oom_batches:
            raise RuntimeError("MiniPerf needs both a successful AOT shape and a larger OOM bound.")
        lower = max(fit_batches)
        upper = min(batch for batch in oom_batches if batch > lower)
        granularity = V5E_CHIPS
        rows = []

        for order, batch in enumerate((lower, upper)):
            source = next(row for row in sweep if int(row["global_batch"]) == batch)
            seed = self._miniperf_row(
                track="frontier",
                order=order,
                name=f"geometric-gb{batch}",
                source="geometric-sweep",
                global_batch=batch,
                sequence_length=sequence_length,
                tensor_parallelism=8,
                fsdp_parallelism=1,
                remat_policy="full",
                gradient_accumulation_steps=1,
                attention="autoselected",
                hbm_limit=hbm_limit,
            )
            for key in (
                "aot_status",
                "aot_wall_sec",
                "aot_cost_usd",
                "aot_reported_argument_gib_per_chip",
                "aot_reported_output_gib_per_chip",
                "aot_reported_temporary_gib_per_chip",
                "aot_reported_alias_gib_per_chip",
                "aot_reported_net_output_gib_per_chip",
                "aot_reported_total_gib_per_chip",
                "aot_reported_total_percent_of_limit",
                "aot_reported_host_temporary_gib",
                "oom_required_temporary_gib_per_chip",
                "oom_available_hbm_gib_per_chip",
            ):
                seed[key] = source[key]
            seed["aot_log"] = f"sweep-aot-gb{batch}.log"
            rows.append(seed)

        low_units = lower // granularity
        high_units = math.ceil(upper / granularity)
        probe_order = 2
        while high_units - low_units > 1 and probe_order - 2 < max_frontier_probes:
            candidate_units = (low_units + high_units) // 2
            candidate = candidate_units * granularity
            row = self._run_miniperf_trial(
                track="frontier",
                order=probe_order,
                name=f"binary-gb{candidate}",
                global_batch=candidate,
                sequence_length=sequence_length,
                tensor_parallelism=8,
                fsdp_parallelism=1,
                remat_policy="full",
                gradient_accumulation_steps=1,
                attention="autoselected",
                steps=1,
                warmup_steps=0,
                hbm_limit=hbm_limit,
                run_live=False,
            )
            rows.append(row)
            if row["aot_status"] == "fit":
                low_units = candidate_units
            elif row["aot_status"] == "oom":
                high_units = candidate_units
            else:
                raise RuntimeError(
                    f"Frontier probe {candidate} was invalid rather than OOM: {row['failure_summary']}"
                )
            probe_order += 1

        trials = self._normalize_miniperf_trials(extra_trials)
        for index, trial in enumerate(trials, start=probe_order):
            rows.append(
                self._run_miniperf_trial(
                    track="throughput",
                    order=index,
                    name=trial["name"],
                    global_batch=fixed_batch,
                    sequence_length=sequence_length,
                    tensor_parallelism=trial["tensor"],
                    fsdp_parallelism=trial["fsdp"],
                    remat_policy=trial["remat"],
                    gradient_accumulation_steps=trial["accumulation"],
                    attention=trial["attention"],
                    steps=steps,
                    warmup_steps=warmup_steps,
                    hbm_limit=hbm_limit,
                    run_live=True,
                )
            )

        benchmark_rows = [row for row in rows if row["track"] == "throughput"]
        baseline = next((row for row in benchmark_rows if row["name"] == "tp8-full"), None)
        if baseline is None or baseline["benchmark_status"] != "measured":
            raise RuntimeError("The fixed TP8/full-remat baseline did not complete.")
        baseline_loss = float(baseline["mean_loss"])
        expected_weights = fixed_batch * sequence_length
        for row in benchmark_rows:
            if row["benchmark_status"] != "measured":
                row["correctness_gate"] = "fail"
                continue
            loss = float(row["mean_loss"])
            weights = float(row["mean_total_weights"])
            direct_tps = float(row["total_tokens_per_sec"])
            reported_tps = float(row["maxtext_total_tokens_per_sec"])
            valid = (
                math.isfinite(loss)
                and abs(loss - baseline_loss) <= max(0.05, abs(baseline_loss) * 0.005)
                and abs(weights - expected_weights) <= 1e-6
                and abs(direct_tps - reported_tps) / direct_tps <= 0.02
            )
            row["correctness_gate"] = "pass" if valid else "fail"

        eligible = [row for row in benchmark_rows if row["correctness_gate"] == "pass"]
        if not eligible:
            raise RuntimeError("No MiniPerf throughput trial passed the fixed-work correctness gate.")
        best = max(eligible, key=lambda row: float(row["total_tokens_per_sec"]))
        lower = low_units * granularity
        upper = high_units * granularity
        boundary_fit = next(
            row for row in rows if row["track"] == "frontier" and int(row["global_batch"]) == lower
        )
        boundary_oom = next(
            row for row in rows if row["track"] == "frontier" and int(row["global_batch"]) == upper
        )
        fit_memory = sorted(
            (
                int(row["global_batch"]),
                float(row["aot_reported_temporary_gib_per_chip"]),
            )
            for row in rows
            if row["track"] == "frontier"
            and row["aot_status"] == "fit"
            and row["aot_reported_temporary_gib_per_chip"] != ""
        )
        memory_monotone = all(
            current[1] >= previous[1] for previous, current in zip(fit_memory, fit_memory[1:])
        )
        source_state = {
            "assignment_version": self.assignment_version,
            "maxtext_git_sha": MAXTEXT_GIT_SHA,
        }
        quality = {}
        evaluation_path = self.artifact_dir / "evaluation_summary.json"
        if evaluation_path.exists():
            evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
            quality = next(model for model in evaluation["models"] if model["label"] == "sft")
            sft_ledger = [
                row
                for row in csv.DictReader(self.ledger_path.open(encoding="utf-8"))
                if row["kind"] == "command" and row["stage"] == "sft"
            ]
            quality = {
                **quality,
                "public_only": True,
                "sft_wall_seconds": sum(float(row["wall_seconds"]) for row in sft_ledger),
                "sft_cost_usd": sum(float(row["estimated_usd"]) for row in sft_ledger),
                "sft_steps": self.sft_plan.get("optimizer_steps"),
            }
        summary = {
            "benchmark_version": MINIPERF_VERSION,
            "student_id": self.config.student_id,
            "run_id": self.run_id,
            "assignment_version": self.assignment_version,
            "maxtext_git_sha": MAXTEXT_GIT_SHA,
            "source_state": source_state,
            "hardware": "v5e-8",
            "fixed_workload": {
                "model": self.model_name,
                "global_batch": fixed_batch,
                "sequence_length": sequence_length,
                "warmup_steps": warmup_steps,
                "measured_steps": steps - warmup_steps,
                "learning_rate": 0,
                "dataset": "synthetic repeated batch",
            },
            "capacity_track": {
                "largest_tested_aligned_fit_global_batch": lower,
                "first_tested_aligned_oom_global_batch": upper,
                "batch_granularity": granularity,
                "remaining_bracket": upper - lower,
                "largest_fit_compiler_temporary_gib_per_chip": float(
                    boundary_fit["aot_reported_temporary_gib_per_chip"]
                ),
                "first_oom_required_temporary_gib_per_chip": float(
                    boundary_oom["oom_required_temporary_gib_per_chip"]
                ),
                "available_hbm_gib_per_chip": hbm_limit,
                "claim": "AOT feasibility bracket, not sampled peak HBM",
                "binary_search_assumption": (
                    "AOT feasibility, not reported memory use, is monotone on the batch-eight grid."
                ),
                "reported_temporary_memory_monotone_over_fit_points": memory_monotone,
            },
            "throughput_track": {
                "winner": best["name"],
                "winning_config": {
                    "tensor_parallelism": int(best["tensor_parallelism"]),
                    "fsdp_parallelism": int(best["fsdp_parallelism"]),
                    "gradient_accumulation_steps": int(best["gradient_accumulation_steps"]),
                    "remat_policy": best["remat_policy"],
                    "attention": best["attention"],
                },
                "tokens_per_second": float(best["total_tokens_per_sec"]),
                "baseline_tokens_per_second": float(baseline["total_tokens_per_sec"]),
                "improvement_over_baseline_percent": 100
                * (float(best["total_tokens_per_sec"]) / float(baseline["total_tokens_per_sec"]) - 1),
                "peak_compute_proxy_percent": float(best["peak_compute_proxy_percent"]),
                "steady_cost_per_million_tokens_usd": float(
                    best["steady_cost_per_million_tokens_usd"]
                ),
                "compiler_reported_total_gib_per_chip": float(
                    best["aot_reported_total_gib_per_chip"]
                ),
                "runtime_post_init_snapshot_gib_per_chip": float(
                    best["runtime_hbm_snapshot_gib_per_chip"]
                ),
                "correctness_gate": best["correctness_gate"],
                "gate_loss_relative_tolerance_percent": 0.5,
                "gate_tps_agreement_tolerance_percent": 2.0,
            },
            "challenge_commands": {
                "wall_seconds": sum(
                    float(row["aot_wall_sec"] or 0) + float(row["command_wall_sec"] or 0)
                    for row in rows
                    if row["source"] == "miniperf"
                ),
                "estimated_cost_usd": sum(
                    float(row["total_trial_cost_usd"] or 0)
                    for row in rows
                    if row["source"] == "miniperf"
                ),
            },
            "quality_track": quality,
        }
        with self.miniperf_path.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (self.artifact_dir / "miniperf_submission.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        self._plot_miniperf(rows, summary)
        print(json.dumps(summary, indent=2))
        return summary

    def _plot_miniperf(self, rows: list[dict], summary: dict) -> None:
        import matplotlib.pyplot as plt

        frontier = sorted(
            (row for row in rows if row["track"] == "frontier"),
            key=lambda row: int(row["global_batch"]),
        )
        benchmark = [row for row in rows if row["track"] == "throughput"]
        measured = [row for row in benchmark if row["benchmark_status"] == "measured"]
        labels = [row["name"] for row in measured]
        positions = list(range(len(measured)))
        colors = ["#2a9d8f" if row["correctness_gate"] == "pass" else "#b23a48" for row in measured]
        figure, axes = plt.subplots(2, 4, figsize=(16, 8.5))
        axes = axes.ravel()

        for status, marker, color in (("fit", "o", "#2a9d8f"), ("oom", "x", "#b23a48")):
            selected = [row for row in frontier if row["aot_status"] == status]
            axes[0].scatter(
                [int(row["global_batch"]) for row in selected],
                [
                    float(
                        row["aot_reported_temporary_gib_per_chip"]
                        or row["oom_required_temporary_gib_per_chip"]
                    )
                    for row in selected
                ],
                marker=marker,
                s=65,
                color=color,
                label=status,
            )
        limit = float(frontier[0]["runtime_hbm_limit_gib_per_chip"])
        axes[0].axhline(limit, color="#111111", linestyle="--", label="HBM limit")
        for row in frontier:
            axes[0].annotate(
                str(row["trial_order"]),
                (
                    int(row["global_batch"]),
                    float(
                        row["aot_reported_temporary_gib_per_chip"]
                        or row["oom_required_temporary_gib_per_chip"]
                    ),
                ),
                xytext=(4, 4),
                textcoords="offset points",
            )
        axes[0].set(title="Binary-search frontier", xlabel="global batch", ylabel="temporary GiB/chip")
        axes[0].legend(fontsize=8)

        bars = axes[1].bar(positions, [float(row["total_tokens_per_sec"]) for row in measured], color=colors)
        axes[1].bar_label(bars, fmt="%.0f", padding=3)
        axes[1].set(title="Fixed-work throughput", ylabel="tokens/s")

        width = 0.38
        axes[2].bar(
            [position - width / 2 for position in positions],
            [float(row["aot_reported_total_gib_per_chip"]) for row in measured],
            width,
            label="compiler report",
            color="#457b9d",
        )
        axes[2].bar(
            [position + width / 2 for position in positions],
            [float(row["runtime_hbm_snapshot_gib_per_chip"]) for row in measured],
            width,
            label="post-init snapshot",
            color="#e9c46a",
        )
        axes[2].axhline(limit, color="#111111", linestyle="--", label="HBM limit")
        axes[2].set(title="Memory evidence", ylabel="GiB/chip")
        axes[2].legend(fontsize=8)

        axes[3].scatter(
            [float(row["aot_reported_total_gib_per_chip"]) for row in measured],
            [float(row["total_tokens_per_sec"]) for row in measured],
            c=colors,
            s=65,
        )
        for row in measured:
            axes[3].annotate(
                row["name"],
                (float(row["aot_reported_total_gib_per_chip"]), float(row["total_tokens_per_sec"])),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )
        axes[3].set(title="TPS-memory tradeoff", xlabel="compiler total GiB/chip", ylabel="tokens/s")

        axes[4].bar(positions, [float(row["peak_compute_proxy_percent"]) for row in measured], color="#6a4c93")
        axes[4].set(title="Peak-compute proxy", ylabel="percent")
        axes[5].bar(
            positions,
            [float(row["steady_cost_per_million_tokens_usd"]) for row in measured],
            color="#e76f51",
        )
        axes[5].set(title="Steady cost", ylabel="USD / 1M tokens")
        axes[6].plot(
            positions,
            [float(row["first_step_sec"]) for row in measured],
            "o--",
            label="first step",
        )
        axes[6].plot(
            positions,
            [float(row["steady_step_sec"]) for row in measured],
            "o-",
            label="steady step",
        )
        axes[6].set_yscale("log")
        axes[6].set(title="Compile vs steady", ylabel="seconds")
        axes[6].legend(fontsize=8)

        quality = summary.get("quality_track", {})
        evaluation_path = self.artifact_dir / "evaluation_summary.json"
        if quality and evaluation_path.exists():
            models = json.loads(evaluation_path.read_text(encoding="utf-8"))["models"]
            quality_bars = axes[7].bar(
                [model["label"] for model in models],
                [float(model["exact_match_percent"]) for model in models],
                color=["#457b9d", "#2a9d8f", "#e9c46a"],
            )
            axes[7].bar_label(quality_bars, fmt="%.1f", padding=3)
            axes[7].set(title="Public quality gate", ylabel="exact match percent", ylim=(0, 105))
        else:
            axes[7].text(0.5, 0.5, "Run evaluation\nfor quality evidence", ha="center", va="center")
            axes[7].set_title("Public quality gate")

        for index, axis in enumerate(axes):
            axis.grid(alpha=0.2)
            if index in {1, 2, 4, 5, 6}:
                axis.set_xticks(positions, labels, rotation=25, ha="right")
        figure.suptitle(f"ME344 MiniPerf: {summary['throughput_track']['winner']} wins fixed-work TPS")
        figure.tight_layout()
        figure.savefig(self.artifact_dir / "miniperf.svg")
        figure.savefig(self.artifact_dir / "miniperf.png", dpi=160)
        plt.show()

    def analyze_compute_bound(self) -> dict:
        """Compare a dense-transformer FLOP lower bound with measured sweep time."""
        if not self.config.run_tpu:
            self._skip("peak-compute bound")
            return {}
        import matplotlib.pyplot as plt

        rows = [
            row
            for row in csv.DictReader(self.sweep_path.open(encoding="utf-8"))
            if row["benchmark_status"] == "measured"
        ]
        results = []
        for row in rows:
            batch = int(row["global_batch"])
            tokens = batch * int(row["sequence_length"])
            approximate_flops = 6 * MODEL_PARAMETER_ESTIMATE * tokens
            peak_seconds = approximate_flops / (
                V5E_CHIPS * V5E_PEAK_BF16_TFLOPS_PER_CHIP * 1e12
            )
            measured_seconds = float(row["steady_step_sec"])
            results.append(
                {
                    "global_batch": batch,
                    "tokens_per_step": tokens,
                    "approximate_training_flops": approximate_flops,
                    "peak_compute_lower_bound_seconds": peak_seconds,
                    "measured_steady_step_seconds": measured_seconds,
                    "time_efficiency_percent": 100 * peak_seconds / measured_seconds,
                    "measured_peak_tflops_percent": 100
                    * float(row["tflops_per_sec_per_device"])
                    / V5E_PEAK_BF16_TFLOPS_PER_CHIP,
                }
            )
        report = {
            "reason": "A lower bound shows whether more tuning can matter; no real step can beat peak hardware.",
            "assumptions": [
                "Dense-transformer training is approximated as 6 * parameters * tokens FLOPs.",
                "All eight chips sustain the published 197 BF16 TFLOP/s peak with no communication or stalls.",
                "The bound is optimistic; measured/peak TFLOPs is a utilization proxy, not model FLOPs utilization.",
            ],
            "cases": results,
        }
        (self.artifact_dir / "compute_bound.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )

        labels = [f"batch {row['global_batch']}" for row in results]
        figure, axes = plt.subplots(1, 2, figsize=(9, 4))
        width = 0.36
        positions = list(range(len(results)))
        axes[0].bar(
            [value - width / 2 for value in positions],
            [row["peak_compute_lower_bound_seconds"] for row in results],
            width,
            label="peak lower bound",
            color="#e9c46a",
        )
        axes[0].bar(
            [value + width / 2 for value in positions],
            [row["measured_steady_step_seconds"] for row in results],
            width,
            label="measured",
            color="#2a9d8f",
        )
        axes[0].set_xticks(positions, labels)
        axes[0].set(title="Step-time bound", ylabel="seconds")
        axes[0].legend()
        axes[1].bar(
            labels,
            [row["measured_peak_tflops_percent"] for row in results],
            color="#6a4c93",
        )
        axes[1].set(title="Measured / peak compute", ylabel="percent")
        for axis in axes:
            axis.grid(axis="y", alpha=0.2)
        figure.tight_layout()
        figure.savefig(self.artifact_dir / "compute_bound.svg")
        plt.show()
        print(json.dumps(report, indent=2))
        return report

    def capture_profile(
        self,
        global_batch: int | None = None,
        sequence_length: int = 256,
        steps: int = 10,
        skip_steps: int = 3,
        profile_steps: int = 3,
    ) -> None:
        """Capture a short steady-state XPlane trace."""
        if not self.config.run_tpu:
            self._skip("XProf capture")
            return
        if global_batch is None:
            measured = [
                row
                for row in csv.DictReader(self.sweep_path.open(encoding="utf-8"))
                if row["benchmark_status"] == "measured"
            ]
            selected = self._select_batch_knee(measured)
            global_batch = int(selected["global_batch"])
            print(
                f"Profiling global batch {global_batch}: the smallest shape within 1% of peak "
                f"with a compiler-reported total below 90% of the HBM limit "
                f"({float(selected['aot_reported_total_percent_of_limit']):.1f}%)."
            )
        run_name = f"me344-{self.run_id}-profile"
        metrics_path = self.artifact_dir / "profile_metrics.jsonl"
        command = [
            str(self.pretrain_python),
            "-m",
            "maxtext.trainers.pre_train.train",
            _maxtext_config(),
            f"model_name={self.maxtext_model_name}",
            f"base_output_directory={self.config.output_directory}",
            f"run_name={run_name}",
            "dataset_type=synthetic",
            "reuse_example_batch=true",
            f"steps={steps}",
            "ici_fsdp_parallelism=1",
            "ici_tensor_parallelism=8",
            f"max_target_length={sequence_length}",
            f"per_device_batch_size={global_batch / 8}",
            "learning_rate=0",
            "enable_checkpointing=false",
            "profiler=xplane",
            f"skip_first_n_steps_for_profiler={skip_steps}",
            f"profiler_steps={profile_steps}",
            "log_config=false",
            f"metrics_file={metrics_path}",
        ]
        self._run_logged(command, "profile.log", "xprof")
        self._require_final_step(metrics_path, steps - 1)

        gcs_tensorboard_dir = f"{self.config.output_directory}/{run_name}/tensorboard"
        local_profile_dir = self.artifact_dir / "xprof_profile"
        local_profile_dir.mkdir(parents=True, exist_ok=True)
        copy = subprocess.run(
            ["gcloud", "storage", "cp", "--recursive", gcs_tensorboard_dir, str(local_profile_dir)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if copy.returncode == 0:
            xprof_logdir = local_profile_dir / "tensorboard"
        else:
            print(
                f"Warning: could not copy the profile locally ({(copy.stderr or copy.stdout).strip()}); "
                "XProf may crash reading it directly from GCS."
            )
            xprof_logdir = gcs_tensorboard_dir

        xprof = (
            f"source {self.project_root}/.venv-me344-pretrain/bin/activate\n"
            f"xprof --logdir={xprof_logdir} --port=6006\n"
            "# Laptop: open http://127.0.0.1:6006 through the existing SSH tunnels.\n"
            "# In XProf, inspect Overview Page, Trace Viewer, and Roofline Analysis.\n"
        )
        (self.artifact_dir / "xprof_command.txt").write_text(xprof, encoding="utf-8")
        print("\nCOPY AND RUN THESE LINES IN A NEW JUPYTER TERMINAL:\n")
        print(xprof)
        print(
            "A traced step can be slower because profiling records extra events. Checkpoint saves and "
            "evaluation can also make individual steps longer, so compare ordinary steady steps."
        )

    @staticmethod
    def _banking_prompt(text: str) -> str:
        allowed = " | ".join(BANKING77_INTENTS)
        return (
            "Route this retail-bank support request. Return exactly one label and no explanation.\n"
            f"Allowed labels: {allowed}\n\nRequest: {text}"
        )

    @staticmethod
    def _take_banking_rows(dataset, per_intent: int) -> dict[str, list[str]]:
        label_names = dataset.features["label"].names
        selected = {intent: [] for intent in BANKING77_INTENTS}
        for row in dataset:
            intent = label_names[int(row["label"])]
            if intent in selected and len(selected[intent]) < per_intent:
                selected[intent].append(row["text"])
            if all(len(rows) == per_intent for rows in selected.values()):
                break
        missing = {intent: len(rows) for intent, rows in selected.items() if len(rows) != per_intent}
        if missing:
            raise RuntimeError(f"Banking77 does not contain the requested stratified slice: {missing}")
        return selected

    def prepare_data(self) -> dict:
        """Pin SFT data and keep task-aligned release rows untouched."""
        source_train = self.data_dir / "source-train.jsonl"
        source_eval = self.data_dir / "source-eval.jsonl"
        train_jsonl = self.data_dir / "messages.jsonl"
        eval_jsonl = self.data_dir / "eval_messages.jsonl"
        option = self.config.dataset_option
        if self.config.run_tpu or option != "local":
            from datasets import load_dataset

        gsm_count = GSM8K_RELEASE_PROMPT_COUNT if option == "gsm8k" else RETENTION_PROMPT_COUNT
        if self.config.run_tpu or option != "local":
            gsm_test = list(
                load_dataset(
                    "openai/gsm8k",
                    "main",
                    split="test",
                    revision=GSM8K_REVISION,
                    streaming=True,
                ).take(gsm_count)
            )
        else:
            gsm_test = [
                {"question": f"What is {value} + {value + 1}?", "answer": f"#### {2 * value + 1}"}
                for value in range(gsm_count)
            ]

        def gsm_suite(group: str) -> list[dict]:
            return [
                {
                    "id": f"gsm8k-{index:02d}",
                    "group": group,
                    "suite": "openai/gsm8k test",
                    "metric": "number",
                    "prompt": row["question"] + "\n\nEnd with exactly `FINAL: <number>`.",
                    "expected": row["answer"].rsplit("####", 1)[-1].strip(),
                }
                for index, row in enumerate(gsm_test)
            ]

        retention_suite = []
        if option == "banking77":
            dataset_id = "PolyAI/banking77"
            revision = BANKING77_REVISION
            license_name = "CC BY 4.0"
            prefix = f"hf://datasets/{dataset_id}@{revision}/data"
            banking_train = load_dataset(
                "parquet",
                data_files={"train": f"{prefix}/train-00000-of-00001.parquet"},
                split="train",
            )
            banking_test = load_dataset(
                "parquet",
                data_files={"test": f"{prefix}/test-00000-of-00001.parquet"},
                split="test",
            )
            training = self._take_banking_rows(banking_train, BANKING77_TRAIN_PER_INTENT + 1)
            release = self._take_banking_rows(banking_test, BANKING77_RELEASE_PER_INTENT)
            train_rows = []
            eval_rows = []
            domain_suite = []
            for intent in BANKING77_INTENTS:
                for text in training[intent][:BANKING77_TRAIN_PER_INTENT]:
                    train_rows.append(
                        {
                            "messages": [
                                {"role": "user", "content": self._banking_prompt(text)},
                                {"role": "assistant", "content": intent},
                            ]
                        }
                    )
                eval_rows.append(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": self._banking_prompt(training[intent][-1]),
                            },
                            {"role": "assistant", "content": intent},
                        ]
                    }
                )
                for index, text in enumerate(release[intent]):
                    domain_suite.append(
                        {
                            "id": f"banking77-{intent}-{index}",
                            "group": "primary",
                            "suite": "PolyAI/banking77 test",
                            "metric": "label",
                            "valid_outputs": list(BANKING77_INTENTS),
                            "prompt": self._banking_prompt(text),
                            "expected": intent,
                        }
                    )
            random.Random(344).shuffle(train_rows)
            random.Random(344).shuffle(eval_rows)
            retention_suite = gsm_suite("retention")
            self.eval_suite = domain_suite + retention_suite
            if len(self.eval_suite) != RELEASE_PROMPT_COUNT:
                raise RuntimeError(f"Expected {RELEASE_PROMPT_COUNT} release rows, found {len(self.eval_suite)}")
            held_out_source = "one unused PolyAI/banking77 train row per selected intent"
            primary_release = f"PolyAI/banking77 test ({len(domain_suite)} untouched rows)"
            retention_release = f"openai/gsm8k test[0:{len(retention_suite)}]"
        elif option == "gsm8k":
            dataset_id = "openai/gsm8k"
            revision = GSM8K_REVISION
            license_name = "MIT"
            training_pool = [
                {"question": row["question"], "answer": row["answer"]}
                for row in load_dataset(
                    dataset_id, "main", split="train", revision=revision, streaming=True
                ).take(144)
            ]
            train_rows, eval_rows = training_pool[:128], training_pool[128:]
            self.eval_suite = gsm_suite("primary")
            held_out_source = "openai/gsm8k train[128:144]"
            primary_release = f"openai/gsm8k test[0:{len(self.eval_suite)}]"
            retention_release = None
        elif option == "ultrachat":
            dataset_id = "HuggingFaceH4/ultrachat_200k"
            revision = "8049631c405ae6576f93f445c6b8166f76f5505a"
            license_name = "MIT"
            train_rows = [
                {"messages": row["messages"]}
                for row in load_dataset(dataset_id, split="train_sft", revision=revision, streaming=True).take(128)
            ]
            eval_rows = [
                {"messages": row["messages"]}
                for row in load_dataset(dataset_id, split="test_sft", revision=revision, streaming=True).take(16)
            ]
            self.eval_suite = gsm_suite("primary")
            held_out_source = "HuggingFaceH4/ultrachat_200k test_sft[0:16]"
            primary_release = "GSM8K retention only; define a task metric before claiming improvement"
            retention_release = None
        elif option == "local":
            dataset_id = "student-provided"
            revision = "local"
            license_name = "student-provided; check the source terms before training"
            local_path = (self.project_root / self.config.local_input_jsonl).resolve()
            rows = [json.loads(line) for line in local_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if self.config.run_tpu:
                random.Random(344).shuffle(rows)
                eval_count = min(16, max(8, len(rows) // 5))
                eval_rows, train_rows = rows[:eval_count], rows[eval_count:]
            else:
                train_rows, eval_rows = rows, rows[:1]
            self.eval_suite = gsm_suite("primary")
            held_out_source = "local deterministic split, seed 344"
            primary_release = "GSM8K retention only; define a task metric before claiming improvement"
            retention_release = None
        else:
            raise ValueError(f"Unknown dataset_option={option}")

        self._write_jsonl(self.eval_suite_path, self.eval_suite)
        self._write_jsonl(source_train, train_rows)
        self._write_jsonl(source_eval, eval_rows * 4)

        def convert(source: Path, destination: Path, report_path: Path) -> dict:
            subprocess.run(
                [
                    str(self.posttrain_python),
                    "scripts/prepare_sft_jsonl.py",
                    str(source),
                    str(destination),
                    "--max-examples",
                    "500",
                    "--report",
                    str(report_path),
                ],
                cwd=self.project_root,
                check=True,
                timeout=180,
            )
            return json.loads(report_path.read_text(encoding="utf-8"))

        report_path = self.artifact_dir / "dataset_report.json"
        eval_report_path = self.artifact_dir / "eval_dataset_report.json"
        self.report = convert(source_train, train_jsonl, report_path)
        convert(source_eval, eval_jsonl, eval_report_path)
        self.report.update(
            {
                "dataset_id": dataset_id,
                "dataset_revision": revision,
                "dataset_license": license_name,
                "manual_review_required": 20,
                "held_out_examples": len(eval_rows),
                "held_out_repetitions": 4,
                "held_out_source": held_out_source,
                "release_suite": primary_release,
                "retention_suite": retention_release,
                "primary_release_examples": sum(
                    row.get("group") == "primary" for row in self.eval_suite
                ),
                "retention_release_examples": len(retention_suite),
                "raw_data_in_submission": False,
            }
        )
        print("\nData review (not copied into the submission):")
        for index, row in enumerate(train_rows[:20], start=1):
            if "messages" in row:
                user = next(
                    (message["content"] for message in row["messages"] if message["role"] == "user"),
                    "",
                )
                answer = next(
                    (
                        message["content"]
                        for message in reversed(row["messages"])
                        if message["role"] == "assistant"
                    ),
                    "",
                )
            else:
                user = str(row.get("question", row.get("prompt", row.get("input", ""))))
                answer = str(row.get("answer", row.get("completion", row.get("output", ""))))
            if "\n\nRequest: " in user:
                user = user.rsplit("\n\nRequest: ", 1)[-1]
            print(f"{index:02d}. {user[:160]} -> {answer[:80]}")
        report_path.write_text(json.dumps(self.report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(self.report, indent=2))
        print(f"held-out SFT rows={len(eval_rows)} x 4 evaluations; release prompts={len(self.eval_suite)}")
        return self.report

    def analyze_token_budget(self, sequence_length: int = 512) -> dict:
        """Measure padding and truncation before choosing a training sequence length."""
        if not self.config.run_tpu:
            self._skip("token-budget bound")
            return {}
        import matplotlib.pyplot as plt
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, revision=self.model_revision)
        rows = self._metric_rows(self.data_dir / "messages.jsonl")
        totals = []
        completions = []

        def token_count(encoded) -> int:
            if hasattr(encoded, "get") and encoded.get("input_ids") is not None:
                encoded = encoded["input_ids"]
            if encoded and isinstance(encoded[0], list):
                encoded = encoded[0]
            return len(encoded)

        for row in rows:
            messages = row["messages"]
            all_tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
            prompt_tokens = tokenizer.apply_chat_template(
                messages[:-1], tokenize=True, add_generation_prompt=True
            )
            totals.append(token_count(all_tokens))
            completions.append(max(0, token_count(all_tokens) - token_count(prompt_tokens)))
        kept = [min(length, sequence_length) for length in totals]
        report = {
            "reason": "Sequence length buys coverage but increases activation memory and padded compute.",
            "examples": len(rows),
            "sequence_length": sequence_length,
            "mean_total_tokens": statistics.fmean(totals),
            "p95_total_tokens": self._percentile(totals, 0.95),
            "max_total_tokens": max(totals),
            "mean_completion_tokens": statistics.fmean(completions),
            "truncated_examples": sum(length > sequence_length for length in totals),
            "truncated_percent": 100 * sum(length > sequence_length for length in totals) / len(totals),
            "padded_token_utilization_percent": 100 * sum(kept) / (len(kept) * sequence_length),
            "padding_tokens_if_unpacked": len(kept) * sequence_length - sum(kept),
            "decision": "Shorten only if saved memory/compute is worth the measured truncation; packing changes this math.",
        }
        (self.artifact_dir / "token_budget.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        figure, axis = plt.subplots(figsize=(7, 4))
        axis.hist(totals, bins=min(16, max(4, len(totals) // 8)), color="#457b9d", alpha=0.85)
        axis.axvline(sequence_length, color="#b23a48", linestyle="--", label=f"limit={sequence_length}")
        axis.set(title="Tokenized SFT examples", xlabel="tokens", ylabel="examples")
        axis.grid(axis="y", alpha=0.2)
        axis.legend()
        figure.tight_layout()
        figure.savefig(self.artifact_dir / "token_budget.svg")
        plt.show()
        print(json.dumps(report, indent=2))
        return report

    def _sft_schedule(
        self,
        steps: int = 200,
        global_batch: int = 4,
        sequence_length: int = 256,
        learning_rate: float = 3e-6,
    ) -> dict:
        """Choose the bounded loader and evaluation schedule used by SFT."""
        examples = int(self.report["examples_written"])
        if not 32 <= examples <= 500:
            raise RuntimeError(f"SFT requires 32-500 training examples after holdout; found {examples}.")
        devices = int(self.reports["posttrain"]["device_count"])
        load_batch = devices if global_batch / devices < 1 else global_batch
        epochs = max(1, math.ceil(steps * load_batch / examples))
        eval_interval = max(5, steps // 8)
        self.sft_plan = {
            "train_examples": examples,
            "held_out_examples": self.report["held_out_examples"],
            "optimizer_steps": steps,
            "effective_global_batch": global_batch,
            "loader_batch": load_batch,
            "loader_epochs": epochs,
            "effective_dataset_passes": steps * global_batch / examples,
            "sequence_length": sequence_length,
            "learning_rate": learning_rate,
            "eval_interval": eval_interval,
            "maximum_training_tokens": steps * global_batch * sequence_length,
        }
        return self.sft_plan

    def run_sft(
        self,
        steps: int = 200,
        global_batch: int = 4,
        sequence_length: int = 256,
        learning_rate: float = 3e-6,
    ) -> str:
        """Run completion-only SFT with held-out loss checkpoints."""
        run_name = self.sft_run_name
        self.sft_checkpoint_path = (
            f"{self.config.output_directory.rstrip('/')}/{run_name}/checkpoints/{steps}/model_params"
        )
        if not self.config.run_tpu:
            self._skip("Tunix SFT")
            return self.sft_checkpoint_path
        self._sft_schedule(steps, global_batch, sequence_length, learning_rate)
        metrics_path = self.artifact_dir / "sft_metrics.jsonl"
        command = self._sft_command(
            steps=steps,
            global_batch=global_batch,
            sequence_length=sequence_length,
            learning_rate=learning_rate,
            epochs=int(self.sft_plan["loader_epochs"]),
            metrics_path=metrics_path,
            eval_interval=int(self.sft_plan["eval_interval"]),
        )
        self._run_logged(command, "sft.log", "sft")
        self._require_final_step(metrics_path, steps)
        checkpoint_root = self.sft_checkpoint_path.rsplit("/model_params", 1)[0]
        listing = subprocess.run(
            ["gcloud", "storage", "ls", f"{checkpoint_root}/**"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if listing.returncode or "optimizer_state" not in listing.stdout:
            raise RuntimeError(f"Full SFT checkpoint not found at {checkpoint_root}:\n{listing.stderr}")
        self._remove_intermediate_sft_checkpoints(run_name, steps)
        self.sft_rollback_path = self.sft_checkpoint_path
        (self.artifact_dir / "checkpoint_paths.txt").write_text(
            f"base={self.config.base_checkpoint_path}\n"
            f"sft_rollback={self.sft_rollback_path}\n"
            f"sft={self.sft_checkpoint_path}\n",
            encoding="utf-8",
        )
        print("SFT checkpoint:", self.sft_checkpoint_path)
        return self.sft_checkpoint_path

    def _sft_command(
        self,
        steps: int,
        global_batch: int,
        sequence_length: int,
        learning_rate: float,
        epochs: int,
        metrics_path: Path,
        eval_interval: int,
    ) -> list[str]:
        return [
            str(self.posttrain_python),
            "-m",
            "maxtext.trainers.post_train.sft.train_sft",
            _maxtext_config("post_train/sft.yml"),
            f"run_name={self.sft_run_name}",
            f"base_output_directory={self.config.output_directory}",
            f"model_name={self.maxtext_model_name}",
            f"tokenizer_path={self.tokenizer_path}",
            "tokenizer_type=huggingface",
            f"load_parameters_path={self.config.base_checkpoint_path}",
            "dataset_type=hf",
            "hf_path=json",
            f"hf_train_files={self.data_dir / 'messages.jsonl'}",
            f"hf_eval_files={self.data_dir / 'eval_messages.jsonl'}",
            "train_split=train",
            "hf_eval_split=train",
            "train_data_columns=['messages']",
            "eval_data_columns=['messages']",
            "ici_fsdp_parallelism=1",
            "ici_tensor_parallelism=8",
            f"per_device_batch_size={global_batch / 8}",
            f"max_target_length={sequence_length}",
            "packing=false",
            f"num_epoch={epochs}",
            f"steps={steps}",
            f"checkpoint_period={steps}",
            "max_num_checkpoints_to_keep=2",
            "async_checkpointing=false",
            f"learning_rate={learning_rate}",
            "sft_train_on_completion_only=true",
            f"eval_interval={eval_interval}",
            f"eval_steps={max(1, math.ceil(int(self.report['held_out_examples']) / 8)) if eval_interval > 0 else -1}",
            "target_eval_loss=0",
            "log_config=false",
            f"metrics_file={metrics_path}",
        ]

    def resume_sft(
        self,
        extra_steps: int = 2,
        global_batch: int = 4,
        sequence_length: int = 256,
        learning_rate: float = 3e-6,
    ) -> str:
        """Resume the same Tunix run and verify step and optimizer continuity."""
        if not self.config.run_tpu:
            self._skip("two-step SFT checkpoint recovery")
            return self.sft_checkpoint_path
        original_rows = [
            row for row in self._metric_rows(self.artifact_dir / "sft_metrics.jsonl") if "learning/loss" in row
        ]
        start_step = int(original_rows[-1]["step"])
        final_step = start_step + extra_steps
        checkpoint_root = self.sft_rollback_path.rsplit("/model_params", 1)[0]
        optimizer = subprocess.run(
            ["gcloud", "storage", "ls", f"{checkpoint_root}/optimizer_state/**"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if optimizer.returncode:
            raise RuntimeError("Rollback checkpoint has no optimizer state; parameter-only loading is not a resume.")

        examples = int(self.report["examples_written"])
        load_batch = int(self.reports["posttrain"]["device_count"])
        epochs = max(1, math.ceil(final_step * load_batch / examples))
        metrics_path = self.artifact_dir / "sft_resume_metrics.jsonl"
        command = self._sft_command(
            steps=final_step,
            global_batch=global_batch,
            sequence_length=sequence_length,
            learning_rate=learning_rate,
            epochs=epochs,
            metrics_path=metrics_path,
            eval_interval=-1,
        )
        self._run_logged(command, "sft-resume.log", "sft-resume")
        resumed = [row for row in self._metric_rows(metrics_path) if "learning/loss" in row]
        observed_steps = [int(row["step"]) for row in resumed]
        expected_steps = list(range(start_step + 1, final_step + 1))
        if observed_steps != expected_steps or not all(math.isfinite(float(row["learning/loss"])) for row in resumed):
            raise RuntimeError(f"Expected resumed steps {expected_steps}, observed {observed_steps}.")

        self.sft_checkpoint_path = (
            f"{self.config.output_directory.rstrip('/')}/{self.sft_run_name}/checkpoints/{final_step}/model_params"
        )
        final_listing = subprocess.run(
            ["gcloud", "storage", "ls", f"{self.sft_checkpoint_path.rsplit('/model_params', 1)[0]}/**"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if final_listing.returncode or "optimizer_state" not in final_listing.stdout:
            raise RuntimeError("The resumed run did not write a full final checkpoint.")
        recovery = {
            "rollback_step": start_step,
            "resumed_steps": observed_steps,
            "optimizer_state_restored": True,
            "optimizer_state_evidence": [
                "optimizer_state existed in the rollback checkpoint",
                "Tunix resumed at the next optimizer step instead of step 1",
                "the final checkpoint contains optimizer_state",
            ],
            "loss_before_resume": original_rows[-1]["learning/loss"],
            "first_loss_after_resume": resumed[0]["learning/loss"],
            "rollback_checkpoint": self.sft_rollback_path,
            "final_checkpoint": self.sft_checkpoint_path,
        }
        (self.artifact_dir / "recovery_report.json").write_text(
            json.dumps(recovery, indent=2) + "\n", encoding="utf-8"
        )
        (self.artifact_dir / "checkpoint_paths.txt").write_text(
            f"base={self.config.base_checkpoint_path}\n"
            f"sft_rollback={self.sft_rollback_path}\n"
            f"sft={self.sft_checkpoint_path}\n",
            encoding="utf-8",
        )
        self._analyze_sft()
        print(json.dumps(recovery, indent=2))
        return self.sft_checkpoint_path

    def _analyze_sft(self) -> None:
        import matplotlib.pyplot as plt

        initial = self._metric_rows(self.artifact_dir / "sft_metrics.jsonl")
        resumed = self._metric_rows(self.artifact_dir / "sft_resume_metrics.jsonl")
        plan = self.sft_plan
        train = [row for row in initial + resumed if "learning/loss" in row]
        held_out = [row for row in initial if "eval/avg_loss" in row]
        losses = [float(row["learning/loss"]) for row in train]
        rolling = [statistics.fmean(losses[max(0, index - 2) : index + 1]) for index in range(len(losses))]
        rolling_by_step = {int(row["step"]): value for row, value in zip(train, rolling)}
        eval_losses = [float(row["eval/avg_loss"]) for row in held_out]
        best_eval_index = min(range(len(eval_losses)), key=eval_losses.__getitem__) if eval_losses else 0
        best_eval_step = int(held_out[best_eval_index]["step"]) if eval_losses else None
        intervals_since_best = len(eval_losses) - 1 - best_eval_index if eval_losses else 0
        recent_improvement = [
            eval_losses[index - 1] - eval_losses[index] for index in range(max(1, len(eval_losses) - 2), len(eval_losses))
        ]
        if len(eval_losses) < 3:
            recommendation = "Collect more held-out measurements before extending training."
        elif intervals_since_best >= 2:
            recommendation = (
                f"Stop or roll back: held-out loss has not set a new best for "
                f"{intervals_since_best} evaluation intervals."
            )
        elif all(delta < 0.02 for delta in recent_improvement):
            recommendation = "Stop or change the experiment: held-out loss has plateaued or regressed."
        else:
            recommendation = "Loss still improves; extend only if the release metric and cost threshold justify it."
        loaded_tokens = sum(float(row.get("learning/total_weights", 0)) for row in train)
        effective_fraction = plan["effective_global_batch"] / plan["loader_batch"]
        initial_steps = max(int(row["step"]) for row in initial if "learning/loss" in row)
        total_steps = max(int(row["step"]) for row in train)
        decision = {
            "loaded_nonpadding_tokens_reported": int(loaded_tokens),
            "estimated_effective_nonpadding_tokens": round(loaded_tokens * effective_fraction),
            "maximum_effective_training_tokens": total_steps
            * plan["effective_global_batch"]
            * plan["sequence_length"],
            "fractional_batch_note": (
                "The MaxText hook counts the full loader batch before loss_fn slices to the effective batch; "
                "the effective token count is therefore an estimate."
            ),
            "last_train_loss_rolling_3": rolling[-1],
            "held_out_loss_by_step": [
                {"step": int(row["step"]), "loss": float(row["eval/avg_loss"])} for row in held_out
            ],
            "best_held_out_loss": min(eval_losses) if eval_losses else None,
            "best_held_out_step": best_eval_step,
            "evaluation_intervals_since_best": intervals_since_best,
            "last_minus_best_held_out_loss": eval_losses[-1] - min(eval_losses) if eval_losses else None,
            "generalization_gap_at_last_eval": (
                eval_losses[-1] - rolling_by_step[int(held_out[-1]["step"])] if eval_losses else None
            ),
            "initial_run_steps": initial_steps,
            "decision_rule": (
                "Stop after two evaluations without a new held-out best, a task-metric regression, "
                "or the cost limit."
            ),
            "recommendation": recommendation,
        }
        (self.artifact_dir / "training_decision.json").write_text(
            json.dumps(decision, indent=2) + "\n", encoding="utf-8"
        )
        figure, axis = plt.subplots(figsize=(8, 4.5))
        axis.plot([row["step"] for row in train], losses, alpha=0.35, label="train batch")
        axis.plot([row["step"] for row in train], rolling, linewidth=2, label="train rolling-3")
        axis.plot(
            [row["step"] for row in held_out],
            eval_losses,
            "o-",
            linewidth=2,
            label="held-out",
        )
        axis.axvline(int(initial[-1]["step"]), linestyle="--", color="#6c757d", label="resume point")
        axis.set(xlabel="optimizer step", ylabel="cross-entropy", title="SFT fit, generalization, and recovery")
        axis.grid(alpha=0.2)
        axis.legend()
        figure.tight_layout()
        figure.savefig(self.artifact_dir / "training_curve.svg")
        plt.show()
        print(json.dumps(decision, indent=2))

    def _remove_intermediate_sft_checkpoints(self, run_name: str, final_step: int) -> None:
        root = f"{self.config.output_directory.rstrip('/')}/{run_name}/checkpoints"
        directories = subprocess.run(
            ["gcloud", "storage", "ls", f"{root}/"],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        ).stdout.splitlines()
        for directory in directories:
            step = directory.rstrip("/").rsplit("/", 1)[-1]
            if step.isdigit() and int(step) != final_step:
                cleanup = ["gcloud", "storage", "rm", "--recursive", directory]
                self._record_command(cleanup)
                subprocess.run(
                    cleanup,
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=300,
                )
                print("Removed intermediate checkpoint:", directory)

    @staticmethod
    def _write_rl_extension(path: Path) -> str:
        functions = (
            _me344_decimal,
            _me344_decimal_text,
            _me344_sequence,
            _me344_targets,
            _me344_prediction,
            _me344_log_reward,
            me344_process_rl_data,
            me344_format_reward,
            me344_exact_reward,
            me344_closeness_reward,
        )
        source = (
            "from __future__ import annotations\n"
            "from decimal import Decimal, InvalidOperation\n"
            "import hashlib\nimport json\nimport os\nfrom pathlib import Path\nimport re\n\n"
            f"_RL_NUMBER = {_RL_NUMBER!r}\n"
            "_RL_ANSWER_RE = re.compile(rf\"<answer>\\s*({_RL_NUMBER})\\s*</answer>\", re.IGNORECASE)\n"
            "_RL_FULL_FORMAT_RE = re.compile(\n"
            "    rf\"<reasoning>.+?</reasoning>\\s*<answer>\\s*{_RL_NUMBER}\\s*</answer>\\s*(?:<\\|im_end\\|>)?\\s*$\",\n"
            "    re.IGNORECASE | re.DOTALL,\n"
            ")\n\n"
            + "\n\n".join(inspect.getsource(function) for function in functions)
            + "\n\n# MaxText's dataset hook loads this exact public name.\n"
            + "process_data = me344_process_rl_data\n"
        )
        path.write_text(source, encoding="utf-8")
        return hashlib.sha256(source.encode()).hexdigest()

    def _prepare_rl_data(self) -> tuple[Path, Path, dict, Path]:
        """Create pinned numeric train/eval Parquet files without packaging raw rows."""
        from datasets import Dataset, load_dataset

        student_path = os.environ.get("ME344_RL_JSONL")
        if student_path:
            source = Path(student_path).expanduser()
            if not source.is_absolute():
                source = self.project_root / source
            rows = [
                json.loads(line)
                for line in source.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if not 24 <= len(rows) <= 200:
                raise ValueError("ME344_RL_JSONL needs 24-200 prompt-answer rows.")
            normalized = []
            for index, row in enumerate(rows, start=1):
                prompt = str(row.get("prompt", row.get("question", ""))).strip()
                if not prompt or len(prompt) > 1_000:
                    raise ValueError(f"RL row {index} needs a prompt of 1-1000 characters.")
                normalized.append({"prompt": prompt, "answer": _me344_decimal_text(row.get("answer"))})
            if len({row["prompt"] for row in normalized}) != len(normalized):
                raise ValueError("ME344_RL_JSONL contains duplicate prompts.")
            random.Random(344).shuffle(normalized)
            eval_count = max(8, len(normalized) // 5)
            eval_rows, train_rows = normalized[:eval_count], normalized[eval_count:]
            provenance = {
                "dataset_id": "student-provided numeric prompts",
                "dataset_revision": hashlib.sha256(source.read_bytes()).hexdigest(),
                "dataset_license": os.environ.get("ME344_RL_LICENSE", "student-provided"),
                "selection": "deterministic 20% holdout with seed 344",
                "student_provided": True,
            }
        else:
            stream = load_dataset(
                "openai/gsm8k",
                "main",
                split="train",
                revision=GSM8K_REVISION,
                streaming=True,
            )
            normalized = []
            scanned = 0
            for row in stream.take(1024):
                scanned += 1
                answer = row["answer"].rsplit("####", 1)[-1].strip()
                try:
                    answer = _me344_decimal_text(answer)
                except ValueError:
                    continue
                operations = row["answer"].count("<<")
                if 5 <= operations <= 9 and len(row["question"]) <= 800:
                    normalized.append({"prompt": row["question"], "answer": answer})
            if len(normalized) < 48:
                raise RuntimeError(f"Pinned GSM8K filter found only {len(normalized)} usable rows.")
            random.Random(344).shuffle(normalized)
            train_rows, eval_rows = normalized[:32], normalized[32:48]
            provenance = {
                "dataset_id": "openai/gsm8k train",
                "dataset_revision": GSM8K_REVISION,
                "dataset_license": "MIT",
                "selection": (
                    f"32 train + 16 held-out rows from {scanned} scanned; "
                    "5-9 annotated arithmetic operations, prompt <=800 characters, seed 344"
                ),
                "student_provided": False,
            }

        train_path = self.data_dir / "rl_train.parquet"
        eval_path = self.data_dir / "rl_eval.parquet"
        extension_path = self.artifact_dir / "rl_task.py"
        Dataset.from_list(train_rows).to_parquet(str(train_path))
        Dataset.from_list(eval_rows).to_parquet(str(eval_path))
        extension_sha = self._write_rl_extension(extension_path)
        report = {
            **provenance,
            "task": "verifiable numeric reasoning",
            "train_examples": len(train_rows),
            "held_out_examples": len(eval_rows),
            "schema": {"prompt": "string", "answer": "finite decimal string"},
            "reward": {
                "format": "0.75 full structure; 0.50 both sections; 0.25 answer tag; 0.10 partial reasoning tag",
                "exact": "1.0 exact numeric answer, tags optional",
                "closeness": "up to 0.10 for an incorrect near miss",
                "maximum": 1.75,
            },
            "release_test_contamination": False,
            "raw_rows_in_submission": False,
            "reward_code": extension_path.name,
            "reward_code_sha256": extension_sha,
        }
        (self.artifact_dir / "rl_dataset_report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, indent=2))
        return train_path, eval_path, report, extension_path

    @staticmethod
    def _training_reward_signal(events_path: Path, generations: int) -> dict:
        events = [
            json.loads(line)
            for line in events_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        grouped: dict[str, dict[str, list[float]]] = {}
        for event in events:
            if int(event["completion_count"]) < generations:
                continue
            grouped.setdefault(event["batch"], {})[event["function"]] = [
                float(value) for value in event["scores"]
            ]
        totals = []
        component_values: dict[str, list[float]] = {}
        required = {"format", "exact", "closeness"}
        for components in grouped.values():
            if set(components) != required:
                continue
            count = len(next(iter(components.values())))
            group_totals = [sum(components[name][index] for name in required) for index in range(count)]
            totals.append(group_totals)
            for name, values in components.items():
                component_values.setdefault(name, []).extend(values)
        spans = [max(values) - min(values) for values in totals]
        return {
            "groups": len(totals),
            "groups_with_relative_signal": sum(span > 1e-9 for span in spans),
            "groups_with_relative_signal_percent": (
                100 * sum(span > 1e-9 for span in spans) / len(spans) if spans else 0.0
            ),
            "mean_group_reward_span": statistics.fmean(spans) if spans else 0.0,
            "mean_total_reward": (
                statistics.fmean(value for group in totals for value in group) if totals else 0.0
            ),
            "component_means": {
                name: statistics.fmean(values) for name, values in sorted(component_values.items())
            },
            "group_total_rewards": totals,
        }

    def _plot_rl_summary(self, summary: dict) -> None:
        import matplotlib.pyplot as plt

        phases = ["pre", "post"]
        figure, axes = plt.subplots(1, 3, figsize=(12, 3.8))
        width = 0.36
        positions = [0, 1]
        axes[0].bar(
            [value - width / 2 for value in positions],
            [summary[phase]["accuracy_percent"] for phase in phases],
            width,
            label="exact",
            color="#457b9d",
        )
        axes[0].bar(
            [value + width / 2 for value in positions],
            [summary[phase]["format_accuracy_percent"] for phase in phases],
            width,
            label="format",
            color="#e9c46a",
        )
        axes[0].set(title="Held-out behavior", ylabel="percent", ylim=(0, 105))
        axes[0].set_xticks(positions, phases)
        axes[0].legend()

        reward_bars = axes[1].bar(
            phases,
            [summary[phase]["mean_reward"] for phase in phases],
            color=["#457b9d", "#2a9d8f"],
        )
        axes[1].bar_label(reward_bars, fmt="%.3f", padding=3)
        axes[1].set(title="Held-out shaped reward", ylabel="mean reward")

        groups = summary["training_signal"]["group_total_rewards"]
        for index, values in enumerate(groups):
            axes[2].scatter([index] * len(values), values, color="#6a4c93", alpha=0.8)
            axes[2].plot([index, index], [min(values), max(values)], color="#6a4c93")
        axes[2].set(title="Training-group reward spread", xlabel="GRPO group", ylabel="total reward")
        for axis in axes:
            axis.grid(alpha=0.2)
        figure.suptitle("ME344 GRPO: reward must vary within a group")
        figure.tight_layout()
        figure.savefig(self.artifact_dir / "rl_reward.svg")
        figure.savefig(self.artifact_dir / "rl_reward.png", dpi=160)
        plt.show()

    def run_grpo(self, updates: int = 12, generations: int = 4) -> dict:
        """Run bounded Tunix GRPO with a verifiable shaped reward."""
        if not 2 <= updates <= 16:
            raise ValueError("GRPO updates must stay between 2 and 16.")
        if generations not in {2, 4}:
            raise ValueError("GRPO generations must be 2 or 4.")
        run_name = f"me344-{self.run_id}-grpo"
        output_root = self.data_dir / "rl-output"
        if not self.config.run_tpu:
            self._skip("bounded shaped-reward GRPO")
            return {}
        train_path, eval_path, _, extension_path = self._prepare_rl_data()
        reward_events = self.artifact_dir / "rl_reward_events.jsonl"
        reward_events.unlink(missing_ok=True)
        # The pinned MaxText RLConfig omits Qwen3's non-multimodal MRoPE default.
        rl_bootstrap = (
            "from maxtext.configs.types import RLConfig;"
            "RLConfig.use_mrope=False;"
            "RLConfig.mrope_section=[24,20,20];"
            "import runpy;"
            "runpy.run_module('maxtext.trainers.post_train.rl.train_rl',run_name='__main__')"
        )
        command = [
            str(self.posttrain_python),
            "-c",
            rl_bootstrap,
            _maxtext_config("post_train/rl.yml"),
            f"run_name={run_name}",
            f"base_output_directory={output_root}",
            f"model_name={self.maxtext_model_name}",
            f"tokenizer_path={self.tokenizer_path}",
            f"load_parameters_path={self.sft_checkpoint_path}",
            "scan_layers=true",
            "chips_per_vm=8",
            "trainer_devices_fraction=1",
            "sampler_devices_fraction=1",
            "use_pathways=false",
            "ici_fsdp_parallelism=1",
            "ici_tensor_parallelism=8",
            "rollout_tensor_parallelism=8",
            "rollout_data_parallelism=1",
            "rollout_expert_parallelism=1",
            "batch_size=1",
            f"num_batches={updates}",
            "num_test_batches=4",
            "eval_batch_size=4",
            f"rl.num_generations={generations}",
            "rl.num_iterations=1",
            "max_prefill_predict_length=256",
            "max_target_length=512",
            "kv_cache_buffer=128",
            "max_num_batched_tokens=1024",
            f"max_num_seqs={generations}",
            "decode_sampling_temperature=1.0",
            "enable_checkpointing=false",
            "async_scheduling=false",
            "hbm_utilization_vllm=0.35",
            "eval_interval=-1",
            "learning_rate=1e-5",
            "learning_rate_schedule_steps=20",
            "warmup_steps_fraction=0",
            "adam_weight_decay=0",
            "dataset_name=parquet",
            "eval_dataset_name=parquet",
            f"hf_train_files={train_path}",
            f"hf_eval_files={eval_path}",
            f"dataset_processor_path={extension_path}",
            f"reward_functions_path={extension_path}",
            "reward_functions=me344_format_reward,me344_exact_reward,me344_closeness_reward",
            "log_period=1",
            "debug=false",
            "log_config=false",
        ]
        result = self._run_logged(
            command,
            "rl.log",
            "grpo",
            extra_env={
                "NEW_MODEL_DESIGN": "1",
                "VLLM_DISABLE_COMPILE_CACHE": "1",
                "ME344_RL_REWARD_LOG": str(reward_events),
            },
            timeout_minutes=25,
        )
        self.rl_summary = self._parse_rl_summary(result.log_path, reward_events, generations)
        if not self.rl_summary["training_signal"]["groups_with_relative_signal"]:
            raise RuntimeError("GRPO sampled no within-group reward differences; do not claim an RL update.")
        print(json.dumps(self.rl_summary, indent=2))
        return self.rl_summary

    def _parse_rl_summary(self, log_path: Path, reward_events: Path, generations: int) -> dict:
        pattern = re.compile(
            r"(Pre|Post) RL Training: corr=(\d+), total=(\d+), accuracy=([0-9.]+)%, "
            r"partial_accuracy=([0-9.]+)%, format_accuracy=([0-9.]+)%, mean_reward=([-+0-9.eE]+)"
        )
        summary = {}
        for (
            phase,
            correct,
            total,
            accuracy,
            partial,
            format_accuracy,
            reward,
        ) in pattern.findall(log_path.read_text(encoding="utf-8")):
            summary[phase.lower()] = {
                "correct": int(correct),
                "total": int(total),
                "accuracy_percent": float(accuracy),
                "partial_accuracy_percent": float(partial),
                "format_accuracy_percent": float(format_accuracy),
                "mean_reward": float(reward),
            }
        if set(summary) != {"pre", "post"}:
            raise RuntimeError("Could not parse both pre- and post-RL summaries from rl.log")
        if not reward_events.exists():
            raise RuntimeError("Custom rewards did not emit an audit log.")
        summary["training_signal"] = self._training_reward_signal(reward_events, generations)
        summary["change"] = {
            "accuracy_percentage_points": summary["post"]["accuracy_percent"]
            - summary["pre"]["accuracy_percent"],
            "format_percentage_points": summary["post"]["format_accuracy_percent"]
            - summary["pre"]["format_accuracy_percent"],
            "mean_reward": summary["post"]["mean_reward"] - summary["pre"]["mean_reward"],
        }
        summary["reward_improved"] = summary["change"]["mean_reward"] > 0
        summary["exact_accuracy_guardrail_passed"] = (
            summary["change"]["accuracy_percentage_points"] >= 0
        )
        (self.artifact_dir / "rl_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        self._plot_rl_summary(summary)
        if summary["reward_improved"] and not summary["exact_accuracy_guardrail_passed"]:
            print(
                "GRPO CHECK: reward and formatting improved, but exact-answer accuracy fell. "
                "Formatting was easier to earn than a correct answer, so keep the SFT checkpoint."
            )
        return summary

    def _server_command(self, checkpoint, port: int) -> list[str]:
        additional = {
            "maxtext_config": {
                "model_name": self.maxtext_model_name,
                "load_parameters_path": str(checkpoint),
                "scan_layers": True,
                "allow_split_physical_axes": True,
                "log_config": False,
            }
        }
        return [
            str(self.posttrain_python.parent / "vllm"),
            "serve",
            self.tokenizer_path,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--seed",
            "42",
            "--max-model-len",
            "768",
            "--gpu-memory-utilization",
            "0.35",
            "--no-enable-prefix-caching",
            "--enable-force-include-usage",
            "--tensor-parallel-size",
            "8",
            "--data-parallel-size",
            "1",
            "--max-num-batched-tokens",
            "1024",
            "--max-num-seqs",
            "8",
            "--hf-overrides",
            json.dumps({"architectures": ["MaxTextForCausalLM"]}),
            "--additional-config",
            json.dumps(additional),
        ]

    def _start_server(self, label: str, checkpoint, port: int = 8000) -> None:
        if self._server is not None:
            raise RuntimeError("Stop the current inference server before starting another.")
        import requests

        command = self._server_command(checkpoint, port)
        self._record_command(command)
        log_path = self.artifact_dir / f"inference-{label}.log"
        log_handle = log_path.open("w", encoding="utf-8")
        env = os.environ.copy()
        env.update(
            {
                "JAX_PLATFORMS": "tpu",
                "NEW_MODEL_DESIGN": "1",
                "SKIP_JAX_PRECOMPILE": "1",
                "VLLM_DISABLE_COMPILE_CACHE": "1",
            }
        )
        started = dt.datetime.now(dt.timezone.utc)
        process = subprocess.Popen(
            command,
            cwd=self.project_root,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self._server = {
            "label": label,
            "port": port,
            "process": process,
            "log_handle": log_handle,
            "log_path": log_path,
            "started": started,
            "start_monotonic": time.monotonic(),
        }
        deadline = time.monotonic() + 600
        try:
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"vLLM exited during startup; inspect {log_path}.")
                try:
                    if requests.get(f"http://127.0.0.1:{port}/health", timeout=3).ok:
                        print(f"{label} server ready at 127.0.0.1:{port}")
                        return
                except requests.RequestException:
                    pass
                time.sleep(2)
            raise TimeoutError(f"vLLM did not become healthy within 10 minutes; inspect {log_path}.")
        except Exception:
            self.stop_server()
            raise

    def stop_server(self) -> None:
        if self._server is None:
            return
        server = self._server
        process = server["process"]
        intentional_stop = process.poll() is None
        if intentional_stop:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        server["log_handle"].close()
        ended = dt.datetime.now(dt.timezone.utc)
        seconds = time.monotonic() - server["start_monotonic"]
        cost = seconds * V5E_8_USD_PER_HOUR / 3600
        self._append_ledger(
            "command",
            f"inference-{server['label']}",
            server["started"],
            ended,
            seconds,
            cost,
            0 if intentional_stop else process.returncode,
        )
        print(f"Stopped the {server['label']} inference server after {seconds / 60:.1f} minutes.")
        self._server = None

    def _chat_request(self, prompt: str, max_tokens: int) -> dict:
        import requests

        if self._server is None:
            raise RuntimeError("The inference server is not running.")
        payload = {
            "model": self.tokenizer_path,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        started = time.monotonic()
        first_token = None
        pieces = []
        usage = {}
        finish_reason = None
        response = requests.post(
            f"http://127.0.0.1:{self._server['port']}/v1/chat/completions",
            json=payload,
            stream=True,
            timeout=(10, 120),
        )
        if not response.ok:
            raise RuntimeError(f"vLLM returned HTTP {response.status_code}: {response.text[:500]}")
        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            usage = event.get("usage") or usage
            choices = event.get("choices") or []
            if choices:
                finish_reason = choices[0].get("finish_reason") or finish_reason
            text = choices[0].get("delta", {}).get("content", "") if choices else ""
            if text:
                first_token = first_token or time.monotonic()
                pieces.append(text)
        ended = time.monotonic()
        generated = "".join(pieces)
        return {
            "generated": generated,
            "ttft_seconds": (first_token or ended) - started,
            "latency_seconds": ended - started,
            "completion_tokens": int(usage.get("completion_tokens") or max(1, len(generated.split()))),
            "usage_reported": bool(usage.get("completion_tokens")),
            "finish_reason": finish_reason,
        }

    def _request_many(self, prompts: list[str], concurrency: int, max_tokens: int) -> tuple[list[dict], float]:
        started = time.monotonic()
        rows = []
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {pool.submit(self._chat_request, prompt, max_tokens): index for index, prompt in enumerate(prompts)}
            for future in as_completed(futures):
                rows.append({"index": futures[future], **future.result()})
        return sorted(rows, key=lambda row: row["index"]), time.monotonic() - started

    @staticmethod
    def _answer_score(case: dict, generated: str) -> tuple[str, bool, bool]:
        expected = case["expected"]
        if case.get("metric") == "label":
            valid = {label.lower(): label for label in case["valid_outputs"]}
            normalized = generated.strip().lower().strip(" `\"'.")
            format_ok = normalized in valid
            matches = [label for label in valid if re.search(rf"\b{re.escape(label)}\b", normalized)]
            predicted = valid.get(normalized, valid[matches[-1]] if matches else "")
            return predicted, predicted == expected, format_ok
        number = r"[-+]?\$?[0-9][0-9,]*(?:\.[0-9]+)?"
        formatted = re.search(rf"FINAL:\s*({number})", generated, flags=re.IGNORECASE)
        candidates = re.findall(number, generated)
        predicted = (formatted.group(1) if formatted else candidates[-1] if candidates else "").replace(",", "")
        target = expected.replace(",", "").replace("$", "").strip()
        try:
            exact = float(predicted.replace("$", "")) == float(target)
        except ValueError:
            exact = predicted.strip() == target
        return predicted, exact, formatted is not None

    @staticmethod
    def _percentile(values: list[float], percentile: float) -> float:
        ordered = sorted(values)
        position = (len(ordered) - 1) * percentile
        lower = math.floor(position)
        upper = math.ceil(position)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    @staticmethod
    def _wilson_interval(successes: int, trials: int, z_score: float = 1.96) -> tuple[float, float]:
        proportion = successes / trials
        denominator = 1 + z_score**2 / trials
        center = (proportion + z_score**2 / (2 * trials)) / denominator
        radius = z_score / denominator * math.sqrt(
            proportion * (1 - proportion) / trials + z_score**2 / (4 * trials**2)
        )
        return center - radius, center + radius

    def run_evaluation_and_serving(
        self,
        eval_prompts: int | None = None,
        benchmark_requests: int = 8,
    ) -> dict:
        """Evaluate base and SFT checkpoints, then leave SFT available for chat."""
        if not self.config.run_tpu:
            self._skip("fixed-suite evaluation and serving benchmark")
            return {}
        suite = [json.loads(line) for line in self.eval_suite_path.read_text(encoding="utf-8").splitlines()]
        if eval_prompts is None:
            eval_prompts = len(suite)
        if not 1 <= eval_prompts <= len(suite):
            raise ValueError(f"eval_prompts must be between 1 and {len(suite)}.")
        if not 1 <= benchmark_requests <= eval_prompts:
            raise ValueError("benchmark_requests must be between 1 and eval_prompts.")
        suite = suite[:eval_prompts]
        checkpoints = (
            ("base", self.config.base_checkpoint_path),
            ("sft", self.sft_checkpoint_path),
        )
        summaries = []
        all_rows: dict[str, list[dict]] = {}
        for label, checkpoint in checkpoints:
            print(f"\nEvaluating {label}: loading {checkpoint}")
            self._start_server(label, checkpoint)
            try:
                generated, _ = self._request_many([row["prompt"] for row in suite], concurrency=4, max_tokens=384)
                scored = []
                for case, output in zip(suite, generated):
                    predicted, exact, format_ok = self._answer_score(case, output["generated"])
                    scored.append(
                        {
                            "label": label,
                            **case,
                            **output,
                            "predicted": predicted,
                            "exact_match": exact,
                            "format_compliance": format_ok,
                        }
                    )
                all_rows[label] = scored
                self._write_jsonl(self.artifact_dir / f"outputs_{label}.jsonl", scored)
                primary = [row for row in scored if row.get("group") == "primary"]
                retention = [row for row in scored if row.get("group") == "retention"]

                def percent(rows: list[dict], key: str) -> float | None:
                    return 100 * sum(row[key] for row in rows) / len(rows) if rows else None

                summaries.append(
                    {
                        "label": label,
                        "examples": len(primary),
                        "exact_match_percent": percent(primary, "exact_match"),
                        "format_compliance_percent": percent(primary, "format_compliance"),
                        "length_limited_percent": 100
                        * sum(row["finish_reason"] == "length" for row in primary)
                        / len(primary),
                        "mean_completion_tokens": statistics.fmean(
                            row["completion_tokens"] for row in primary
                        ),
                        "retention_examples": len(retention),
                        "retention_exact_match_percent": percent(retention, "exact_match"),
                        "retention_format_compliance_percent": percent(
                            retention, "format_compliance"
                        ),
                    }
                )
                latest = summaries[-1]
                math_exact = latest["retention_exact_match_percent"]
                math_text = "n/a" if math_exact is None else f"{math_exact:.1f}%"
                print(
                    f"{label} complete: Banking77 exact={latest['exact_match_percent']:.1f}%, "
                    f"GSM8K exact={math_text}"
                )
            except Exception:
                self.stop_server()
                raise
            if label != checkpoints[-1][0]:
                self.stop_server()

        try:
            regressions = []
            for before, after in (("base", "sft"),):
                for prior, current in zip(all_rows[before], all_rows[after]):
                    if prior["exact_match"] and not current["exact_match"]:
                        regressions.append(
                            {
                                "transition": f"{before}->{after}",
                                "id": current["id"],
                                "expected": current["expected"],
                                "before": prior["generated"][:500],
                                "after": current["generated"][:500],
                            }
                        )
            evaluation = {
                "primary_suite": self.report["release_suite"],
                "retention_suite": self.report.get("retention_suite"),
                "secondary_suite_role": (
                    "GSM8K is the SFT safety check and the separate GRPO target; "
                    "Banking77 remains the startup product metric."
                ),
                "models": summaries,
                "regressions": regressions[:6],
            }
            base_summary, sft_summary = summaries
            primary_change = (
                sft_summary["exact_match_percent"] - base_summary["exact_match_percent"]
            )
            format_change = (
                sft_summary["format_compliance_percent"]
                - base_summary["format_compliance_percent"]
            )
            print(
                f"SFT changed Banking77 exact match by {primary_change:+.1f} points and "
                f"format compliance by {format_change:+.1f} points."
            )
            if primary_change < 0:
                print(
                    "Exact task accuracy fell. Better formatting alone is not enough to ship this checkpoint; "
                    "keep the base model or revise the data and training recipe."
                )
            (self.artifact_dir / "evaluation_summary.json").write_text(
                json.dumps(evaluation, indent=2) + "\n", encoding="utf-8"
            )
            with (self.artifact_dir / "evaluation_results.csv").open(
                "w", encoding="utf-8", newline=""
            ) as output:
                writer = csv.DictWriter(output, fieldnames=list(summaries[0]))
                writer.writeheader()
                writer.writerows(summaries)
        except Exception:
            self.stop_server()
            raise

        serving_rows = []
        prompts = [row["prompt"] for row in suite[:benchmark_requests]]
        try:
            self._request_many(prompts, concurrency=1, max_tokens=64)
            self._request_many(prompts, concurrency=4, max_tokens=64)
            for concurrency in (1, 4):
                results, wall = self._request_many(prompts, concurrency=concurrency, max_tokens=64)
                tokens = sum(row["completion_tokens"] for row in results)
                mean_latency = statistics.fmean(row["latency_seconds"] for row in results)
                requests_per_second = len(results) / wall
                serving_rows.append(
                    {
                        "concurrency": concurrency,
                        "requests": len(results),
                        "warmup_requests": len(prompts),
                        "wall_seconds": wall,
                        "mean_output_tokens": tokens / len(results),
                        "requests_per_second": requests_per_second,
                        "latency_mean_seconds": mean_latency,
                        "ttft_p50_seconds": self._percentile([row["ttft_seconds"] for row in results], 0.50),
                        "ttft_p95_seconds": self._percentile([row["ttft_seconds"] for row in results], 0.95),
                        "latency_p50_seconds": self._percentile(
                            [row["latency_seconds"] for row in results], 0.50
                        ),
                        "latency_p95_seconds": self._percentile(
                            [row["latency_seconds"] for row in results], 0.95
                        ),
                        "output_tokens_per_second": tokens / wall,
                        "usage_reported_for_all": all(row["usage_reported"] for row in results),
                    }
                )
            with (self.artifact_dir / "serving_results.csv").open(
                "w", encoding="utf-8", newline=""
            ) as output:
                writer = csv.DictWriter(output, fieldnames=list(serving_rows[0]))
                writer.writeheader()
                writer.writerows(serving_rows)
        except Exception:
            self.stop_server()
            raise
        print(json.dumps(evaluation, indent=2))
        print(json.dumps(serving_rows, indent=2))
        print("The SFT server is still running for the optional chat. Run project.stop_server() when finished.")
        return {"evaluation": evaluation, "serving": serving_rows}

    def analyze_evaluation_uncertainty(self, target_margin: float = 0.10) -> dict:
        """Put confidence bounds around the deliberately small release suite."""
        if not 0 < target_margin < 1:
            raise ValueError("target_margin must be between 0 and 1.")
        if not self.config.run_tpu:
            self._skip("evaluation uncertainty bound")
            return {}
        import matplotlib.pyplot as plt

        evaluation = json.loads((self.artifact_dir / "evaluation_summary.json").read_text(encoding="utf-8"))
        models = []
        for model in evaluation["models"]:
            trials = int(model["examples"])
            successes = round(trials * float(model["exact_match_percent"]) / 100)
            lower, upper = self._wilson_interval(successes, trials)
            models.append(
                {
                    "label": model["label"],
                    "successes": successes,
                    "trials": trials,
                    "observed_percent": 100 * successes / trials,
                    "wilson_95_low_percent": 100 * lower,
                    "wilson_95_high_percent": 100 * upper,
                }
            )
        worst_case_samples = math.ceil(1.96**2 * 0.25 / target_margin**2)
        report = {
            "reason": "A point estimate without uncertainty can turn random prompt variation into a release claim.",
            "assumption": "Prompts are independent and representative; Wilson intervals do not fix biased data.",
            "target_margin_percentage_points": 100 * target_margin,
            "worst_case_samples_for_target_margin": worst_case_samples,
            "models": models,
            "decision": "Treat this suite as a release smoke test; use a larger representative set for a real launch.",
        }
        (self.artifact_dir / "evaluation_uncertainty.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        labels = [row["label"] for row in models]
        observed = [row["observed_percent"] for row in models]
        figure, axis = plt.subplots(figsize=(7, 4))
        axis.errorbar(
            labels,
            observed,
            yerr=[
                [value - row["wilson_95_low_percent"] for value, row in zip(observed, models)],
                [row["wilson_95_high_percent"] - value for value, row in zip(observed, models)],
            ],
            fmt="o",
            capsize=6,
            color="#2a9d8f",
        )
        axis.set(title="Exact match with 95% Wilson intervals", ylabel="percent", ylim=(0, 100))
        axis.grid(axis="y", alpha=0.2)
        figure.tight_layout()
        figure.savefig(self.artifact_dir / "evaluation_uncertainty.svg")
        plt.show()
        print(json.dumps(report, indent=2))
        return report

    def log_wandb(self) -> dict:
        """Create an offline-first W&B run from the immutable course evidence."""
        if not self.config.run_tpu:
            self._skip("W&B evidence tracking")
            return {}
        import wandb

        mode = os.environ.get("WANDB_MODE", "offline").lower()
        if mode not in {"offline", "online", "disabled"}:
            raise ValueError("WANDB_MODE must be offline, online, or disabled.")
        init_args = {
            "project": os.environ.get("WANDB_PROJECT", "stanford-me344"),
            "name": self.run_id,
            "mode": mode,
            "dir": str(self.artifact_dir),
            "config": self._base_config(),
            "reinit": "finish_previous",
        }
        if os.environ.get("WANDB_ENTITY"):
            init_args["entity"] = os.environ["WANDB_ENTITY"]
        run = wandb.init(**init_args)
        sweep = list(csv.DictReader(self.sweep_path.open(encoding="utf-8")))
        run.define_metric("sweep/*", step_metric="sweep/global_batch")
        for row in sweep:
            payload = {
                "sweep/global_batch": int(row["global_batch"]),
                "sweep/aot_fit": int(row["aot_status"] == "fit"),
                "sweep/aot_oom": int(row["aot_status"] == "oom"),
                "sweep/aot_wall_seconds": float(row["aot_wall_sec"]),
                "sweep/aot_cost_usd": float(row["aot_cost_usd"]),
                "sweep/hbm_limit_gib_per_chip": float(row["runtime_hbm_limit_gib_per_chip"]),
            }
            for source, destination in (
                ("aot_reported_total_gib_per_chip", "sweep/aot_reported_total_gib_per_chip"),
                ("aot_reported_total_percent_of_limit", "sweep/aot_reported_total_percent_of_limit"),
                ("aot_reported_argument_gib_per_chip", "sweep/aot_reported_argument_gib_per_chip"),
                ("aot_reported_temporary_gib_per_chip", "sweep/aot_reported_temporary_gib_per_chip"),
                ("runtime_hbm_snapshot_gib_per_chip", "sweep/runtime_hbm_snapshot_gib_per_chip"),
                ("runtime_hbm_snapshot_percent_of_limit", "sweep/runtime_hbm_snapshot_percent_of_limit"),
                ("oom_required_temporary_gib_per_chip", "sweep/oom_required_temporary_gib_per_chip"),
                ("oom_available_hbm_gib_per_chip", "sweep/oom_available_hbm_gib_per_chip"),
                ("total_tokens_per_sec", "sweep/tokens_per_second"),
                ("tflops_per_sec_per_device", "sweep/tflops_per_second_per_device"),
                ("steady_cost_per_million_tokens_usd", "sweep/steady_usd_per_million_tokens"),
            ):
                if row[source] != "":
                    payload[destination] = float(row[source])
            run.log(payload)
        fit_sweep = [row for row in sweep if row["aot_status"] == "fit"]
        oom_sweep = [row for row in sweep if row["aot_status"] == "oom"]
        run.summary["bounds/largest_aot_fit_global_batch"] = max(
            int(row["global_batch"]) for row in fit_sweep
        )
        run.summary["bounds/recommended_profile_global_batch"] = int(
            self._select_batch_knee([row for row in sweep if row["benchmark_status"] == "measured"])[
                "global_batch"
            ]
        )
        if oom_sweep:
            run.summary["bounds/first_aot_oom_global_batch"] = int(oom_sweep[0]["global_batch"])
        if self.miniperf_path.exists():
            miniperf = list(csv.DictReader(self.miniperf_path.open(encoding="utf-8")))
            run.define_metric("miniperf/*", step_metric="miniperf/trial_order")
            for row in miniperf:
                payload = {
                    "miniperf/trial_order": int(row["trial_order"]),
                    "miniperf/global_batch": int(row["global_batch"]),
                    "miniperf/aot_fit": int(row["aot_status"] == "fit"),
                    "miniperf/aot_oom": int(row["aot_status"] == "oom"),
                    "miniperf/correctness_pass": int(row["correctness_gate"] == "pass"),
                }
                for source, destination in (
                    ("aot_reported_total_gib_per_chip", "miniperf/compiler_total_gib_per_chip"),
                    ("aot_reported_temporary_gib_per_chip", "miniperf/compiler_temporary_gib_per_chip"),
                    ("oom_required_temporary_gib_per_chip", "miniperf/oom_temporary_gib_per_chip"),
                    ("runtime_hbm_snapshot_gib_per_chip", "miniperf/runtime_snapshot_gib_per_chip"),
                    ("total_tokens_per_sec", "miniperf/tokens_per_second"),
                    ("peak_compute_proxy_percent", "miniperf/peak_compute_proxy_percent"),
                    ("steady_cost_per_million_tokens_usd", "miniperf/usd_per_million_tokens"),
                ):
                    if row[source] != "":
                        payload[destination] = float(row[source])
                run.log(payload)
            miniperf_summary = json.loads(
                (self.artifact_dir / "miniperf_submission.json").read_text(encoding="utf-8")
            )
            run.summary["miniperf/winner"] = miniperf_summary["throughput_track"]["winner"]
            run.summary["miniperf/best_tokens_per_second"] = miniperf_summary["throughput_track"][
                "tokens_per_second"
            ]
            run.summary["miniperf/largest_aligned_fit_global_batch"] = miniperf_summary[
                "capacity_track"
            ]["largest_tested_aligned_fit_global_batch"]
            run.log({"miniperf_dashboard": wandb.Image(str(self.artifact_dir / "miniperf.png"))})
        run.define_metric("sft/*", step_metric="sft/step")
        for row in self._metric_rows(self.artifact_dir / "sft_metrics.jsonl") + self._metric_rows(
            self.artifact_dir / "sft_resume_metrics.jsonl"
        ):
            payload = {"sft/step": int(row["step"])}
            if "learning/loss" in row:
                payload["sft/train_loss"] = float(row["learning/loss"])
            if "eval/avg_loss" in row:
                payload["sft/held_out_loss"] = float(row["eval/avg_loss"])
            run.log(payload)

        rl = json.loads((self.artifact_dir / "rl_summary.json").read_text(encoding="utf-8"))
        for phase in ("pre", "post"):
            run.summary[f"rl/{phase}_exact_percent"] = rl[phase]["accuracy_percent"]
            run.summary[f"rl/{phase}_format_percent"] = rl[phase]["format_accuracy_percent"]
            run.summary[f"rl/{phase}_mean_reward"] = rl[phase]["mean_reward"]
        run.summary["rl/groups_with_relative_signal"] = rl["training_signal"][
            "groups_with_relative_signal"
        ]
        run.summary["rl/reward_change"] = rl["change"]["mean_reward"]
        run.log({"rl_reward_evidence": wandb.Image(str(self.artifact_dir / "rl_reward.png"))})

        evaluation = json.loads((self.artifact_dir / "evaluation_summary.json").read_text(encoding="utf-8"))
        for model in evaluation["models"]:
            run.summary[f"release/{model['label']}_exact_match_percent"] = model["exact_match_percent"]
            run.summary[f"release/{model['label']}_format_percent"] = model["format_compliance_percent"]
            run.summary[f"release/{model['label']}_length_limited_percent"] = model[
                "length_limited_percent"
            ]
            if model["retention_exact_match_percent"] is not None:
                run.summary[f"retention/{model['label']}_exact_match_percent"] = model[
                    "retention_exact_match_percent"
                ]
        serving = list(csv.DictReader((self.artifact_dir / "serving_results.csv").open(encoding="utf-8")))
        for row in serving:
            concurrency = row["concurrency"]
            run.summary[f"serving/c{concurrency}_ttft_p95_seconds"] = float(row["ttft_p95_seconds"])
            run.summary[f"serving/c{concurrency}_latency_p95_seconds"] = float(row["latency_p95_seconds"])
            run.summary[f"serving/c{concurrency}_output_tokens_per_second"] = float(
                row["output_tokens_per_second"]
            )
        compute = json.loads((self.artifact_dir / "compute_bound.json").read_text(encoding="utf-8"))
        run.summary["bounds/best_peak_compute_percent"] = max(
            row["measured_peak_tflops_percent"] for row in compute["cases"]
        )
        tokens = json.loads((self.artifact_dir / "token_budget.json").read_text(encoding="utf-8"))
        run.summary["bounds/token_utilization_percent"] = tokens["padded_token_utilization_percent"]
        run.summary["bounds/truncated_examples_percent"] = tokens["truncated_percent"]
        uncertainty = json.loads(
            (self.artifact_dir / "evaluation_uncertainty.json").read_text(encoding="utf-8")
        )
        run.summary["bounds/release_samples_for_10pp"] = uncertainty[
            "worst_case_samples_for_target_margin"
        ]
        scale = json.loads((self.artifact_dir / "scale_out.json").read_text(encoding="utf-8"))
        run.summary["scaling/strong_speedup"] = scale["strong_speedup"]
        run.summary["scaling/strong_efficiency_percent"] = scale[
            "strong_scaling_efficiency_percent"
        ]
        run.summary["scaling/weak_throughput_ratio"] = scale["weak_throughput_ratio"]
        run.summary["scaling/weak_efficiency_percent"] = scale[
            "weak_scaling_efficiency_percent"
        ]
        run.summary["scaling/ici_strong_speedup"] = scale["ici_strong_speedup"]
        run.summary["scaling/ici_strong_efficiency_percent"] = scale[
            "ici_strong_scaling_efficiency_percent"
        ]
        run.summary["scaling/ici_weak_throughput_ratio"] = scale["ici_weak_throughput_ratio"]
        run.summary["scaling/ici_weak_efficiency_percent"] = scale[
            "ici_weak_scaling_efficiency_percent"
        ]
        run.log({"scale_out": wandb.Image(str(self.artifact_dir / "scale_out.png"))})
        run.log({"systems_dashboard": wandb.Image(str(self.artifact_dir / "systems_dashboard.png"))})
        result = {
            "mode": mode,
            "run_id": run.id,
            "url": run.url or None,
            "offline_directory": "wandb" if mode == "offline" else None,
        }
        run.finish()
        (self.artifact_dir / "wandb_run.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        return result

    def build_dashboard(self) -> None:
        """Combine raw evidence into the nine required systems views."""
        if not self.config.run_tpu:
            self._skip("systems dashboard")
            return
        import matplotlib.pyplot as plt

        all_sweep = list(csv.DictReader(self.sweep_path.open(encoding="utf-8")))
        sweep = [row for row in all_sweep if row["benchmark_status"] == "measured"]
        batches = [int(row["global_batch"]) for row in sweep]

        def values(key: str) -> list[float]:
            return [float(row[key]) for row in sweep]

        sft = self._metric_rows(self.artifact_dir / "sft_metrics.jsonl") + self._metric_rows(
            self.artifact_dir / "sft_resume_metrics.jsonl"
        )
        sft_train = [row for row in sft if "learning/loss" in row]
        sft_eval = [row for row in sft if "eval/avg_loss" in row]
        train_losses = [float(row["learning/loss"]) for row in sft_train]
        rolling = [statistics.fmean(train_losses[max(0, index - 2) : index + 1]) for index in range(len(train_losses))]
        evaluation = list(csv.DictReader((self.artifact_dir / "evaluation_results.csv").open(encoding="utf-8")))
        serving = list(csv.DictReader((self.artifact_dir / "serving_results.csv").open(encoding="utf-8")))
        recovery = json.loads((self.artifact_dir / "recovery_report.json").read_text(encoding="utf-8"))
        scale = json.loads((self.artifact_dir / "scale_out.json").read_text(encoding="utf-8"))
        ledger = [row for row in csv.DictReader(self.ledger_path.open(encoding="utf-8")) if row["kind"] == "command"]
        grouped_costs: dict[str, float] = {}
        for row in ledger:
            stage = row["stage"]
            if stage.startswith("sweep-"):
                stage = "AOT batch frontier"
            elif stage.startswith("sft"):
                stage = "SFT + recovery"
            elif stage == "grpo" or stage.startswith("rl"):
                stage = "GRPO"
            elif stage.startswith("inference") or stage.startswith("vllm"):
                stage = "evaluation + serving"
            grouped_costs[stage] = grouped_costs.get(stage, 0.0) + float(row["estimated_usd"])
        dashboard_seconds = (
            dt.datetime.now(dt.timezone.utc) - self.session_started
        ).total_seconds()
        dashboard_cost = dashboard_seconds * V5E_8_USD_PER_HOUR / 3600
        command_cost = sum(grouped_costs.values())
        figure, axes = plt.subplots(3, 3, figsize=(14, 11))
        axes = axes.ravel()
        axes[0].plot(
            batches,
            values("aot_wall_sec"),
            "o--",
            label="AOT compile",
        )
        axes[0].plot(
            batches,
            values("runtime_startup_compile_overhead_sec"),
            "o--",
            label="runtime startup + JIT",
        )
        axes[0].plot(batches, values("first_step_sec"), "o-", label="first step")
        axes[0].plot(batches, values("steady_step_sec"), "o-", label="steady step")
        axes[0].set_title("Wall-time decomposition")
        axes[0].set_yscale("log")
        axes[0].legend()
        axes[1].plot(batches, values("total_tokens_per_sec"), "o-", color="#16697a")
        axes[1].set_title("Training throughput")
        axes[1].set_ylabel("tokens/s")
        axes[2].plot(batches, values("tflops_per_sec_per_device"), "o-", color="#6a4c93")
        axes[2].set_title("Device work")
        axes[2].set_ylabel("TFLOP/s/device")
        scale_trials = [
            scale[name]
            for name in ("baseline", "ici_strong", "strong", "ici_weak", "weak")
        ]
        scale_labels = ["8 ICI\nb256", "16 ICI\nb256", "16 DCN\nb256", "16 ICI\nb512", "16 DCN\nb512"]
        bars = axes[3].bar(
            scale_labels,
            [trial["tokens_per_second"] for trial in scale_trials],
            color=["#4285F4", "#34A853", "#EA4335", "#34A853", "#EA4335"],
        )
        axes[3].bar_label(bars, fmt="%.0f", fontsize=7)
        axes[3].set_title(
            f"GKE scaling: ICI strong {scale['ici_strong_scaling_efficiency_percent']:.0f}% | "
            f"DCN strong {scale['strong_scaling_efficiency_percent']:.0f}%"
        )
        axes[3].set_ylabel("tokens/s")
        axes[4].plot([row["step"] for row in sft_train], rolling, label="train rolling-3")
        axes[4].plot(
            [row["step"] for row in sft_eval],
            [row["eval/avg_loss"] for row in sft_eval],
            "o-",
            label="held-out",
        )
        axes[4].axvline(recovery["rollback_step"], linestyle="--", color="#6c757d", label="resume")
        axes[4].set_title("SFT stopping evidence")
        axes[4].set_ylabel("cross-entropy")
        axes[4].set_xlabel("optimizer step")
        axes[4].legend()
        positions = list(range(len(evaluation)))
        axes[5].bar(
            [position - 0.18 for position in positions],
            [float(row["exact_match_percent"]) for row in evaluation],
            width=0.36,
            label="primary",
            color="#2a9d8f",
        )
        axes[5].bar(
            [position + 0.18 for position in positions],
            [
                float(row["retention_exact_match_percent"])
                if row["retention_exact_match_percent"]
                else math.nan
                for row in evaluation
            ],
            width=0.36,
            label="GSM8K math",
            color="#457b9d",
        )
        axes[5].set_xticks(positions, [row["label"] for row in evaluation])
        axes[5].set_title("Domain gain and retention")
        axes[5].set_ylabel("percent")
        axes[5].legend()
        concurrency = [int(row["concurrency"]) for row in serving]
        for key, label, style in (
            ("ttft_p50_seconds", "TTFT p50", "o-"),
            ("ttft_p95_seconds", "TTFT p95", "o--"),
            ("latency_p50_seconds", "E2E p50", "s-"),
            ("latency_p95_seconds", "E2E p95", "s--"),
        ):
            axes[6].plot(concurrency, [float(row[key]) for row in serving], style, label=label)
        axes[6].set_xticks(concurrency)
        axes[6].set_title("Serving latency")
        axes[6].set(xlabel="concurrency", ylabel="seconds")
        axes[6].legend(fontsize=8)
        cost_labels = ["notebook total", *grouped_costs]
        axes[7].barh(
            cost_labels,
            [dashboard_cost, *grouped_costs.values()],
            color=["#2a9d8f", *(["#b56576"] * len(grouped_costs))],
        )
        axes[7].set_title(f"Estimated TPU cost by stage | commands: ${command_cost:.2f}")
        axes[7].set_xlabel("USD")
        axes[7].tick_params(axis="y", labelsize=7)
        frontier_positions = list(range(len(all_sweep)))
        fit_positions = [index for index, row in enumerate(all_sweep) if row["aot_status"] == "fit"]
        fit_rows = [row for row in all_sweep if row["aot_status"] == "fit"]
        axes[8].bar(
            fit_positions,
            [float(row["aot_reported_total_gib_per_chip"]) for row in fit_rows],
            color="#457b9d",
            label="compiler-reported total",
        )
        runtime_positions = [
            index
            for index, row in enumerate(all_sweep)
            if row["runtime_hbm_snapshot_gib_per_chip"] != ""
        ]
        runtime_rows = [row for row in all_sweep if row["runtime_hbm_snapshot_gib_per_chip"] != ""]
        if runtime_rows:
            axes[8].plot(
                runtime_positions,
                [float(row["runtime_hbm_snapshot_gib_per_chip"]) for row in runtime_rows],
                "o-",
                color="#2a9d8f",
                label="post-init snapshot",
            )
        hbm_limit = min(float(row["runtime_hbm_limit_gib_per_chip"]) for row in all_sweep)
        axes[8].axhline(hbm_limit, color="#111111", linestyle="--", label="HBM limit")
        for index, row in enumerate(all_sweep):
            if row["aot_status"] == "oom":
                required = row["oom_required_temporary_gib_per_chip"]
                axes[8].scatter(
                    index,
                    float(required) if required != "" else hbm_limit,
                    marker="x",
                    s=90,
                    color="#b23a48",
                    label="OOM temporary requirement",
                )
        axes[8].set_xticks(frontier_positions, [row["global_batch"] for row in all_sweep])
        axes[8].set_title("Batch memory evidence")
        axes[8].set_ylabel("GiB per chip")
        axes[8].legend()
        for index, axis in enumerate(axes):
            axis.grid(alpha=0.2)
            if index < 4:
                axis.set_xlabel("global batch")
        figure.suptitle(
            f"ME344 systems evidence: {self.model_name} | "
            f"{dashboard_seconds / 60:.1f} notebook min | est. ${dashboard_cost:.2f}",
            fontsize=15,
        )
        figure.tight_layout(rect=(0, 0, 1, 0.96))
        figure.savefig(self.artifact_dir / "systems_dashboard.svg")
        figure.savefig(self.artifact_dir / "systems_dashboard.png", dpi=160)
        plt.show()

    def package(self) -> Path:
        """Create a small submission while retaining full local evidence."""
        if self._server is not None:
            raise RuntimeError("Stop the inference server before packaging.")
        ended = dt.datetime.now(dt.timezone.utc)
        seconds = (ended - self.session_started).total_seconds()
        cost = seconds * V5E_8_USD_PER_HOUR / 3600 if self.config.run_tpu else 0.0
        self._append_ledger(
            "resource-window",
            "notebook-session",
            self.session_started,
            ended,
            seconds,
            cost,
            "estimate",
        )
        todos = (self.artifact_dir / "answers.md").read_text(encoding="utf-8").count("TODO")
        if self.config.run_tpu and todos:
            raise RuntimeError("Answer the four reflection prompts before packaging.")
        if self.config.run_tpu and not (self.artifact_dir / "scale_out.json").exists():
            raise RuntimeError("Finish the required GKE scaling experiment before packaging.")

        submission_names = (
            "answers.md",
            "systems_dashboard.png",
        )
        submission_paths = [
            self.artifact_dir / name
            for name in submission_names
            if (self.artifact_dir / name).exists()
        ]
        kind = "final" if self.config.run_tpu else "dry-run"
        archive = self.artifact_dir.parent / f"{self.artifact_dir.name}.{kind}.tar.gz"
        with tarfile.open(archive, "w:gz") as output:
            for path in submission_paths:
                output.add(path, arcname=f"{self.run_id}/{path.name}")
        print(f"{kind.title()} archive:", archive)
        print("Submission files:", ", ".join(path.name for path in submission_paths))
        print("Detailed logs remain in:", self.artifact_dir)
        if self.config.run_tpu:
            remote = f"{self.config.submission_uri.rstrip('/')}/{archive.name}"
            command = ["gcloud", "storage", "cp", str(archive), remote]
            self._record_command(command)
            uploaded = subprocess.run(command, capture_output=True, text=True, timeout=180)
            if uploaded.returncode:
                raise RuntimeError(
                    f"Final upload failed. Preserve {archive} with TPU VM scp before deletion:\n"
                    + (uploaded.stderr or uploaded.stdout)
                )
            print("Uploaded final archive:", remote)
            pilot_run = f"{self.config.output_directory.rstrip('/')}/{self.pretrain_run_name}"
            cleanup = ["gcloud", "storage", "rm", "--recursive", pilot_run]
            self._record_command(cleanup)
            removed = subprocess.run(cleanup, capture_output=True, text=True, timeout=300)
            if removed.returncode:
                print("Could not remove the temporary pre-training checkpoint:", pilot_run)
            else:
                print("Removed temporary pre-training checkpoint:", pilot_run)
            print("Next: return to a Stanford-node terminal and follow Submit And Clean Up in the README.")
        else:
            print("Dry run complete; this archive is not a submission.")
        return archive


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="ME344 helper commands")
    subparsers = parser.add_subparsers(dest="command", required=True)

    scale_run = subparsers.add_parser("scale-run", help="run one required GKE scaling trial")
    scale_run.add_argument(
        "--mode",
        choices=("baseline", "ici_strong", "ici_weak", "strong", "weak"),
        required=True,
    )
    scale_run.add_argument("--chips", type=int, choices=(8, 16), required=True)
    scale_run.add_argument("--result", type=Path, required=True)
    scale_run.add_argument("--handoff", type=Path, required=True)
    scale_run.add_argument("--student-id", required=True)
    scale_run.add_argument("--steps", type=int, default=12)

    scale_summary = subparsers.add_parser("scale-summary", help="summarize GKE scale results")
    scale_summary.add_argument("--baseline-result", type=Path, required=True)
    scale_summary.add_argument("--ici-strong-result", type=Path, required=True)
    scale_summary.add_argument("--ici-weak-result", type=Path, required=True)
    scale_summary.add_argument("--strong-result", type=Path, required=True)
    scale_summary.add_argument("--weak-result", type=Path, required=True)
    scale_summary.add_argument("--output", type=Path, default=Path("scale_out.png"))

    args = parser.parse_args()
    if args.command == "scale-run":
        run_scale_out_benchmark(
            args.mode,
            chips=args.chips,
            result_path=args.result,
            handoff_path=args.handoff,
            student_id=args.student_id,
            steps=args.steps,
        )
    else:
        summarize_scale_out(
            args.baseline_result,
            args.ici_strong_result,
            args.ici_weak_result,
            args.strong_result,
            args.weak_result,
            output=args.output,
        )


if __name__ == "__main__":
    _main()
