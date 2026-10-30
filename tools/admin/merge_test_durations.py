"""
Refresh the per-test durations that balance the CPU and GPU test shards.

`gpu_tests.yml` and `cpu_tests.yml` split each backend's tests into four
shards with pytest-split, using `.github/test_durations/<backend>.json` and
`.github/test_durations/cpu-<backend>.json`. Every run on master uploads
the fresh timings of each shard as an artifact (`durations-<backend>-<shard>`
on GPU, `cpu-durations-<backend>-<version>-<shard>` on CPU). This script
downloads those artifacts, unions the four shard files per backend and
rewrites the committed files. Commit the result and open a PR; the PR's own
CI run shows the new shard times.

Requires the GitHub CLI (`gh auth login`).

Usage:
python tools/admin/merge_test_durations.py            # GPU, newest master run
python tools/admin/merge_test_durations.py --cpu      # CPU, newest master run
python tools/admin/merge_test_durations.py --run_id 40967156654
"""

import json
import pathlib
import subprocess
import tempfile

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string("run_id", None, "A run id of the selected workflow.")
flags.DEFINE_string("repo", "keras-team/keras-hub", "GitHub repository.")
flags.DEFINE_bool("cpu", False, "Refresh the CPU durations instead of GPU.")

SHARDS = 4
OUT_DIR = (
    pathlib.Path(__file__).resolve().parents[2] / ".github" / "test_durations"
)
# (workflow file, artifact prefix, backends, committed file prefix). On CPU
# several Keras versions run per backend; the shard files are picked from
# `keras-stable` when present, otherwise from the only version there is.
GPU = ("gpu_tests.yml", "durations", ("jax", "tensorflow", "torch"), "")
CPU = (
    "actions.yml",
    "cpu-durations",
    ("jax", "tensorflow", "torch", "openvino"),
    "cpu-",
)


def gh(*args):
    return subprocess.run(
        ["gh", *args], check=True, capture_output=True, text=True
    ).stdout


def latest_master_run(repo, workflow):
    runs = json.loads(
        gh(
            "run",
            "list",
            "--repo",
            repo,
            "--workflow",
            workflow,
            "--branch",
            "master",
            "--status",
            "completed",
            "--limit",
            "1",
            "--json",
            "databaseId,createdAt,conclusion,headSha",
        )
    )
    if not runs:
        raise SystemExit(f"No completed {workflow} run on master.")
    run = runs[0]
    print(
        f"Using run {run['databaseId']} ({run['createdAt']}, "
        f"{run['conclusion']}, {run['headSha'][:10]})"
    )
    return str(run["databaseId"])


def shard_files(tmp, prefix, backend, out_prefix):
    """The four shard files for `backend`, one Keras version on CPU."""
    files = sorted(tmp.glob(f"{prefix}-{backend}-*/{out_prefix}{backend}.json"))
    by_version = {}
    for path in files:
        # `durations-jax-1` -> "" ; `cpu-durations-jax-keras-stable-1` ->
        # "keras-stable".
        middle = path.parent.name[len(prefix) + len(backend) + 2 :]
        version = middle.rsplit("-", 1)[0] if "-" in middle else ""
        by_version.setdefault(version, []).append(path)
    if not by_version:
        return []
    version = (
        "keras-stable" if "keras-stable" in by_version else min(by_version)
    )
    return by_version[version]


def main(_):
    workflow, prefix, backends, out_prefix = CPU if FLAGS.cpu else GPU
    run_id = FLAGS.run_id or latest_master_run(FLAGS.repo, workflow)
    merged = {}
    with tempfile.TemporaryDirectory() as tmp:
        try:
            gh(
                "run",
                "download",
                run_id,
                "--repo",
                FLAGS.repo,
                "--pattern",
                f"{prefix}-*",
                "--dir",
                tmp,
            )
        except subprocess.CalledProcessError as e:
            raise SystemExit(
                f"{e.stderr.strip()}\nRun {run_id} has no {prefix}-* "
                "artifacts. They are uploaded by master runs and kept for "
                "14 days; pick a newer run with --run_id."
            )
        tmp = pathlib.Path(tmp)
        # Validate every backend before writing anything, so a missing shard
        # cannot leave the committed files half refreshed.
        for backend in backends:
            files = shard_files(tmp, prefix, backend, out_prefix)
            if len(files) != SHARDS:
                raise SystemExit(
                    f"{backend}: expected {SHARDS} shard files, "
                    f"found {len(files)}."
                )
            merged[backend] = {}
            for path in files:
                merged[backend].update(json.loads(path.read_text()))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for backend, durations in merged.items():
        target = OUT_DIR / f"{out_prefix}{backend}.json"
        target.write_text(
            json.dumps(durations, indent=4, sort_keys=True) + "\n"
        )
        total = sum(durations.values()) / 60
        print(f"{backend}: {len(durations)} tests, {total:.1f} min -> {target}")
        for nodeid, seconds in sorted(durations.items(), key=lambda kv: -kv[1])[
            :5
        ]:
            print(f"  {seconds:7.1f}s  {nodeid}")


if __name__ == "__main__":
    app.run(main)
