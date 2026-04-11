#!/usr/bin/env python3
"""Sequential training job queue.

Runs train.py for each job defined in a YAML jobs file. Jobs execute one at a
time; the queue stops immediately if any job fails.

Usage:
    python queue.py --jobs path/to/jobs.yaml
    python queue.py --jobs path/to/jobs.yaml --dry-run
    python queue.py --jobs path/to/jobs.yaml --start-from 3

See examples/jobs.yaml for the jobs file format.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import yaml

# Maps jobs.yaml keys to their train.py CLI flags
OVERRIDE_MAP = {
    "max_steps": "--max-steps",
    "output_dir": "--output-dir",
    "data_dir": "--data-dir",
    "pretrained_path": "--pretrained-path",
    "resume": "--resume",
    "log_level": "--log-level",
}


def build_cmd(job: dict) -> list[str]:
    cmd = [sys.executable, "train.py", "--config", job["config"]]
    for key, flag in OVERRIDE_MAP.items():
        if key in job:
            cmd += [flag, str(job[key])]
    if job.get("no_auto_tune"):
        cmd.append("--no-auto-tune")
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a queue of choochoo training jobs sequentially.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--jobs", required=True, metavar="FILE", help="Path to YAML jobs file")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configs and print the job list without running anything",
    )
    parser.add_argument(
        "--start-from",
        type=int,
        default=1,
        metavar="N",
        help="Skip the first N-1 jobs and start from job N (1-indexed, default: 1)",
    )
    args = parser.parse_args()

    jobs_path = Path(args.jobs)
    if not jobs_path.exists():
        sys.exit(f"error: jobs file not found: {jobs_path}")

    with open(jobs_path) as f:
        data = yaml.safe_load(f)

    jobs = data.get("jobs") if data else None
    if not jobs:
        sys.exit("error: no jobs found in jobs file")

    # Validate all config paths exist before touching the GPU
    errors = []
    for i, job in enumerate(jobs, 1):
        cfg = job.get("config")
        if not cfg:
            errors.append(f"  job {i}: missing required 'config' key")
        elif not Path(cfg).exists():
            errors.append(f"  job {i}: config not found: {cfg}")
    if errors:
        sys.exit("Validation failed:\n" + "\n".join(errors))

    total = len(jobs)
    print(f"Queue: {total} job(s) from {jobs_path}")
    for i, job in enumerate(jobs, 1):
        marker = "  " if i >= args.start_from else "  (skip)"
        print(f"{marker}[{i}/{total}] {job['config']}")

    if args.dry_run:
        print("\nDry run — no jobs executed.")
        return

    if args.start_from > 1:
        print(f"\nStarting from job {args.start_from} (skipping {args.start_from - 1}).")

    for i, job in enumerate(jobs, 1):
        if i < args.start_from:
            continue

        cmd = build_cmd(job)
        ts = time.strftime("%H:%M:%S")
        print(f"\n{'='*60}")
        print(f"[{i}/{total}] Starting  {ts}")
        print(f"  config: {job['config']}")
        overrides = {k: job[k] for k in job if k != "config"}
        if overrides:
            print(f"  overrides: {overrides}")
        print(f"{'='*60}")

        result = subprocess.run(cmd)

        if result.returncode != 0:
            sys.exit(
                f"\n[{i}/{total}] FAILED (exit code {result.returncode}). Queue stopped."
            )

        print(f"\n[{i}/{total}] Completed  {time.strftime('%H:%M:%S')}")

    print(f"\nAll {total} job(s) completed successfully.")


if __name__ == "__main__":
    main()
