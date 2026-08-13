#!/usr/bin/env python3
"""Rebuild and re-upload the ME344 submission archive after editing answers.md.

No TPU needed: the archive is just answers.md + systems_dashboard.png under a
run-id directory, which is exactly what the course's package() produces.

    python3 repackage.py            # rebuild + upload
    python3 repackage.py --dry-run  # rebuild only, show contents
"""
import subprocess
import sys
import tarfile
from pathlib import Path

RUN_ID = "qanh-20260813t194806z"
SUBMISSION_URI = "gs://me344-tpu-labs-west4/final_projects/qanh/submission"
HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
NAMES = ("answers.md", "systems_dashboard.png")

archive = HERE / f"{RUN_ID}.final.tar.gz"
missing = [name for name in NAMES if not (ARTIFACTS / name).exists()]
if missing:
    sys.exit(f"missing required files in {ARTIFACTS}: {missing}")

todos = (ARTIFACTS / "answers.md").read_text(encoding="utf-8").count("TODO")
if todos:
    sys.exit(f"answers.md still has {todos} TODO placeholder(s); answer them first.")

with tarfile.open(archive, "w:gz") as output:
    for name in NAMES:
        output.add(ARTIFACTS / name, arcname=f"{RUN_ID}/{name}")

with tarfile.open(archive) as check:
    for member in check.getmembers():
        print(f"  {member.name}  {member.size} bytes")
print("built:", archive)

if "--dry-run" in sys.argv:
    print("dry run: not uploaded")
    raise SystemExit(0)

remote = f"{SUBMISSION_URI}/{archive.name}"
subprocess.run(["gcloud", "storage", "cp", str(archive), remote], check=True)
print("uploaded:", remote)
