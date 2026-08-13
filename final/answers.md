# ME344 Final Reflection

1. **Profile and scaling:** What did XProf show, and how did ICI versus DCN affect the scaling results?

XProf showed a 24.94 TFLOP training step that is 99.07% learnable-weight matmul against only 0.93% attention at sequence length 256, so this shape is a dense-matmul workload rather than an attention-bound one, and the fabric mattered far more than the chip count: doubling from 8 to 16 chips inside one ICI slice held 100.1% strong and 99.6% weak scaling efficiency (2,026 against 2,025 tokens/s/chip), while spreading the same 16 chips across two slices so the data-parallel gradient all-reduce had to cross DCN dropped that to 65.2% strong and 53.6% weak (1,320 and 1,085 tokens/s/chip).

2. **Training:** What changed after SFT or GRPO?

SFT for 200 steps moved banking77 exact match from 70.0% to 92.5% on 40 held-out prompts while holding format compliance at 100% and lifting GSM8K retention from 81.25% to 87.5%, and GRPO on top of it raised format compliance from 37.5% to 93.75% while moving exact accuracy from 10/16 to 12/16, but the held-out loss set its best at step 50 (0.0436) and had drifted to 0.0623 by step 200, so the run's own stopping rule recommends rolling back rather than training further.

3. **Release:** Which checkpoint would you ship, and what evidence supports it?

I would ship the SFT checkpoint at step 202, because it is the only candidate with a domain gain, no retention regression, and a serving measurement behind it (70.0% to 92.5% exact match, 81.25% to 87.5% GSM8K retention, 27.0 requests/s at concurrency 1 and 75.5 requests/s at concurrency 4), and I would explicitly not lean on the GRPO checkpoint even though its guardrail passed this time, because its +12.5 point accuracy change is two examples out of sixteen and came out as -12.5 points on an identically configured earlier run, whereas the 95% Wilson interval on the SFT result (80.1% to 97.4%) still overlaps the base interval (54.6% to 81.9%), so I would treat all of this as a release smoke test rather than a launch decision.

4. **Next step:** What would you try next with more time or a dataset you care about?

I would fix the two measurement gaps before trusting any of these conclusions further: re-run the two DCN trials with XProf enabled, because DCN weak scaling came out worse per chip than DCN strong (1,085 against 1,320 tokens/s/chip) even though the all-reduce payload is fixed by the model rather than the batch, and the 3.2x jump in exposed communication time (3.51 s against 1.08 s) could be lost compute/communication overlap, rematerialization from running at 110% of the AOT memory limit, or contention from other students on a shared fabric; and grow the evaluation suite to the 97 examples the Wilson analysis says a plus-or-minus 10 point margin requires, then evaluate the step-50 checkpoint the stopping rule prefers, which no task metric has ever been run against.
