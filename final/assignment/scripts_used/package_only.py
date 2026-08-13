#!/usr/bin/env python
"""Run cell 40's packaging step against the run that already completed.

FinalProject derives run_id from the wall clock, so a fresh instance would point at an
empty artifact dir. package() only reads run_id, artifact_dir, ledger_path, submission_uri
and session_started, so re-pointing those three attributes reuses the real code path
without re-running any TPU work.
"""
import os
import sys
from pathlib import Path

RUN_ID = "qanh-20260813t194806z"
ANSWERS = Path.home() / "answers_final.md"

os.environ.setdefault("MPLBACKEND", "Agg")
START = Path.cwd().resolve()
PROJECT_ROOT = next(
    p for p in (START, *START.parents) if (p / "scripts/final_project.py").exists()
)
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.final_project import FinalProject, project_config_from_env  # noqa: E402

project = FinalProject(project_config_from_env())
stray = project.artifact_dir

project.run_id = RUN_ID
project.artifact_dir = Path.home() / "me344-artifacts" / RUN_ID
project.ledger_path = project.artifact_dir / "cost_ledger.csv"

if not project.artifact_dir.is_dir():
    raise SystemExit(f"missing artifact dir {project.artifact_dir}")

answers_target = project.artifact_dir / "answers.md"
answers_target.write_text(ANSWERS.read_text(encoding="utf-8"), encoding="utf-8")
print("installed answers from", ANSWERS)
print("TODO count:", answers_target.read_text(encoding="utf-8").count("TODO"))

archive = project.package()
print("ARCHIVE:", archive)

if stray != project.artifact_dir:
    for leftover in sorted(stray.rglob("*"), reverse=True):
        leftover.unlink() if leftover.is_file() else leftover.rmdir()
    stray.rmdir()
    print("removed stray dir", stray)
