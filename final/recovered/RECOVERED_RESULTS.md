# ME344 Final Project: recovered results

Recovered on 2026-08-13 from two places, after the TPU VM `tpu-student49` was deleted
and took `~/me344-artifacts/` with it:

1. `session3-cells1-33-full.log` (1502 lines) from gist `8dadecfa12ef5e2b7dfbaada34019e67`
   revision `ad027ede752d6dd5904a4b588880a55e4b65fc13`, the last revision before the log
   was reset for a fourth attempt.
2. `*.json` scaling results pulled from `gs://me344-tpu-labs-west4/final_projects/qanh/scaling/`.

Run id: `qanh-20260810t005140z`. Cells 1 through 33 all completed, 00:51:39Z to 01:28:25Z
(37 minutes). GKE jobs ran 01:37Z to 01:58Z.

## Capacity boundary (cell 6)

Qwen3-4B: 4.022B params, BF16 checkpoint 7.49 GiB, persistent training state 44.95 GiB,
rough peak before activations 59.94 GiB. Per-chip HBM limit observed: 15.75 GiB.

| Layout | Working set per chip | Predicted | Observed | AOT total |
|---|---|---|---|---|
| TP8 (ici_tensor_parallelism=8) | ~7.49 GiB | fit_candidate | fit | 8.23 GiB/chip |
| DP8 (replicated) | ~59.94 GiB | oom | oom (65.47 GiB temporaries) | n/a |

The deliberate DP8 OOM confirms the capacity check works.

## Batch frontier sweep (cell 10, seq len 256)

Rows for gb8 / gb16 / gb32 were cut by the gist writer's output truncation. What survived:

| global_batch | per_device | AOT status | AOT total GiB/chip | % of limit | tok/s | TFLOP/s/device |
|---|---|---|---|---|---|---|
| 4 | 0.5 | fit | 8.23 | 52.2% | n/a | n/a |
| 64 | 8.0 | fit | 10.60 | 67.3% | 16,626 | 50.62 |
| 128 | 16.0 | fit | 12.99 | 82.5% | 16,629 | 50.64 |
| 256 | 32.0 | fit | 17.37 | 110.3% | 16,226 | 49.41 |
| 512 | 64.0 | **oom** | needs 18.43 GiB vs 15.75 available | - | - | - |

Headline: **global batch 512 is the first shape past the memory boundary.** Throughput is
flat from gb64 to gb256 at roughly 16.2k to 16.6k tokens/s and about 50 TFLOP/s/device, so
the extra HBM spent above gb64 buys nothing on this shape.

## SFT (cells 21, 23)

| Metric | Value |
|---|---|
| Train step 200 training loss | 0.007942 |
| Train step 200 eval loss | 0.166139 (perplexity 1.180) |
| Loss before resume | 0.007942 |
| First loss after resume | 0.012250 |
| Resume behavior | Tunix resumed at the next optimizer step, not step 1 |

Checkpoints: step 200, then 202 after the 2 extra resume steps.

## Evaluation, base vs SFT (cell 27)

Primary suite: PolyAI/banking77 test, 40 untouched rows. Retention suite: openai/gsm8k test[0:16].

| Checkpoint | Banking77 exact | Format compliance | GSM8K retention exact |
|---|---|---|---|
| base (qwen3-4b-instruct-2507) | 70.0% | 100.0% | 81.25% |
| SFT (step 202) | **92.5%** | 100.0% | **87.5%** |
| Change | **+22.5 pts** | +0.0 pts | +6.25 pts |

No retention regression. GSM8K went up, not down.

## GRPO (cell 25)

| Metric | Change |
|---|---|
| Exact accuracy | **-12.5 percentage points** |
| Format compliance | +37.5 percentage points |
| Mean reward | +0.285 |
| `reward_improved` | true |
| `exact_accuracy_guardrail_passed` | **false** |

GRPO taught the format, and cost accuracy. The guardrail failing is the evidence for
shipping SFT step 202 rather than the GRPO checkpoint.

## Serving benchmark (cell 27, vLLM on the SFT checkpoint)

| Concurrency | req/s | TTFT p50 | latency p50 | latency p95 | output tok/s |
|---|---|---|---|---|---|
| 1 | 28.0 | 12.1 ms | 33.2 ms | 50.0 ms | 168 |
| 4 | 78.1 | 22.6 ms | 43.8 ms | 60.1 ms | 468 |

4x concurrency gives 2.8x throughput for 1.9x TTFT. Mean output 6 tokens, so this is a
short-generation classification workload, not a long-form one.

## GKE scaling (cells 33 to 35, five JobSet runs)

All five result files are in the bucket and are reusable: `collect_scale_results()` checks
only `student_id`, and `summarize_scale_out()` checks only mode / chips / slices / workers /
fabric / topology / global_batch. Neither cross-checks `source_checkpoint`.

| Run | Chips | Fabric | Batch | s/step | tok/s | tok/s/chip | Efficiency |
|---|---|---|---|---|---|---|---|
| baseline | 8 | ICI 2x4 | 256 | 4.045 | 16,200 | 2025 | reference |
| ici_strong | 16 | ICI 4x4 | 256 | 2.022 | 32,417 | 2026 | 100.1% strong |
| ici_weak | 16 | ICI 4x4 | 512 | 4.061 | 32,272 | 2017 | 99.6% weak |
| strong | 16 | DCN 2x(2x4) | 256 | 3.104 | 21,123 | 1320 | 65.2% strong |
| weak | 16 | DCN 2x(2x4) | 512 | 7.553 | 17,355 | 1085 | 53.6% weak |

ICI scales essentially perfectly. DCN costs 35% on strong scaling and 46% on weak.

### One anomaly worth explaining, not asserting

DCN weak is worse per chip than DCN strong (1085 vs 1320 tok/s/chip). With DP2 x TP8 the
cross-slice all-reduce carries gradients, whose size is set by the model, not the batch, so
doubling per-slice work should amortize that fixed cost better, not worse.

Backing out exposed communication time:

- DCN strong: 3.104 - 2.022 = 1.08 s exposed
- DCN weak: 7.553 - 4.045 = 3.51 s exposed

Same payload, 3.2x the exposed cost. Candidate explanations: lost compute/communication
overlap at per-slice batch 256, HBM pressure forcing rematerialization (gb256 was already
at 110% of the AOT limit on one slice), or DCN contention from other students, since these
ran 01:37Z to 01:58Z on a shared cluster. The GKE jobs did not save XProf traces, so this
cannot be re-profiled from what survives.

## Consistency note

`handoff.json` points at the `20260810t020258z` checkpoint (the fourth attempt) while all
five scaling results reference `20260810t005140z` (the third attempt, the one logged here).
Harmless, since nothing validates checkpoint identity, but worth knowing they disagree.
