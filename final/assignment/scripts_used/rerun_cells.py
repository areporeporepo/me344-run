#!/usr/bin/env python
"""Re-run final_project.ipynb cells 2 through 38 headlessly.

Same calls, same arguments, same order as the notebook. Cell 40 is left out
because collect_reflections() prompts for four answers, and cell 44 is optional.
"""
import datetime as dt
import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

START = Path.cwd().resolve()
PROJECT_ROOT = next(
    path for path in (START, *START.parents) if (path / "scripts/final_project.py").exists()
)
sys.path.insert(0, str(PROJECT_ROOT))


def stamp(message: str) -> None:
    print(f"\n@@@ {dt.datetime.now(dt.timezone.utc):%H:%M:%SZ} {message}", flush=True)


from scripts.final_project import project_config_from_env  # noqa: E402

CONFIG = project_config_from_env()                                    # cell 2

from scripts.final_project import FinalProject  # noqa: E402

project = FinalProject(CONFIG)                                        # cell 3
stamp(f"run_id={project.run_id} artifact_dir={project.artifact_dir}")

try:
    stamp("cell 5: probe")
    device_reports = project.probe()

    stamp("cell 7: capacity lab")
    AOT_GLOBAL_BATCH = 2
    AOT_SEQUENCE_LENGTH = 512
    capacity_report = project.run_capacity_lab(
        global_batch=AOT_GLOBAL_BATCH,
        sequence_length=AOT_SEQUENCE_LENGTH,
    )
    project.plot_capacity()

    stamp("cell 9: baselines")
    pretrain_checkpoint = project.run_baselines(
        smoke_steps=5,
        c4_steps=5,
        c4_sequence_length=AOT_SEQUENCE_LENGTH,
        c4_global_batch=AOT_GLOBAL_BATCH,
    )

    stamp("cell 11: batch/HBM frontier sweep")
    SWEEP_BATCHES = (4, 8, 16, 32, 64, 128, 256, 512)
    SWEEP_SEQUENCE_LENGTH = 256
    sweep_csv = project.run_sweep(
        global_batches=SWEEP_BATCHES,
        sequence_length=SWEEP_SEQUENCE_LENGTH,
        steps=8,
    )
    project.plot_sweep()

    stamp("cell 13: compute bound")
    compute_bound = project.analyze_compute_bound()

    stamp("cell 15: XProf capture")
    project.capture_profile(
        sequence_length=SWEEP_SEQUENCE_LENGTH,
        steps=10,
        skip_steps=3,
        profile_steps=3,
    )

    stamp("cell 18: prepare data")
    dataset_report = project.prepare_data()

    stamp("cell 20: token budget")
    token_budget = project.analyze_token_budget(sequence_length=256)

    stamp("cell 22: SFT 200 steps")
    sft_checkpoint = project.run_sft(
        steps=200,
        global_batch=4,
        sequence_length=256,
        learning_rate=3e-6,
    )

    stamp("cell 24: resume SFT")
    sft_checkpoint = project.resume_sft(extra_steps=2)

    stamp("cell 26: GRPO")
    rl_summary = project.run_grpo(
        updates=12,
        generations=4,
    )

    stamp("cell 28: evaluation + serving")
    release_evidence = project.run_evaluation_and_serving(benchmark_requests=8)

    stamp("cell 30: evaluation uncertainty")
    evaluation_bound = project.analyze_evaluation_uncertainty(target_margin=0.10)

    stamp("cell 32: stop server")
    project.stop_server()

    stamp("cell 34: scale handoff")
    scale_handoff = project.prepare_scale_handoff()

    stamp("cell 36: collect GKE scale results (reusing bucket jsons)")
    scale_report = project.collect_scale_results()

    stamp("cell 38: dashboard + wandb")
    project.build_dashboard()
    wandb_run = project.log_wandb()

    stamp("RERUN_COMPLETE")
    Path.home().joinpath(".me344-rerun-complete").write_text(
        f"{project.run_id}\n{project.artifact_dir}\n", encoding="utf-8"
    )
except Exception:
    stamp("RERUN_FAILED")
    traceback.print_exc()
    try:
        project.stop_server()
    except Exception:
        pass
    Path.home().joinpath(".me344-rerun-failed").write_text(
        f"{project.run_id}\n{project.artifact_dir}\n", encoding="utf-8"
    )
    raise
