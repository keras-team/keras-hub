"""
Refresh the per-test durations that balance the GPU test shards.

`gpu_tests.yml` splits each backend's tests into four shards with
pytest-split, using `.github/test_durations/<backend>.json`. Every GPU run
on master uploads the fresh timings of each shard as an artifact named
`durations-<backend>-<shard>`. This script downloads those artifacts, unions
the four shard files per backend and rewrites the committed files. Commit
the result and open a PR; the PR's own GPU run shows the new shard times.

Requires the GitHub CLI (`gh auth login`).

Usage:
python tools/admin/merge_test_durations.py            # newest master run
python tools/admin/merge_test_durations.py --run_id 40967156654
"""

import json
import pathlib
import subprocess
import tempfile

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string("run_id", None, "A `KerasHub GPU Tests` run id.")
flags.DEFINE_string("repo", "keras-team/keras-hub", "GitHub repository.")

BACKENDS = ("jax", "tensorflow", "torch")
SHARDS = 4
OUT_DIR = (
    pathlib.Path(__file__).resolve().parents[2] / ".github" / "test_durations"
)


def gh(*args):
    return subprocess.run(
        ["gh", *args], check=True, capture_output=True, text=True
    ).stdout


def latest_master_run(repo):
    runs = json.loads(
        gh(
            "run",
            "list",
            "--repo",
            repo,
            "--workflow",
            "gpu_tests.yml",
            "--branch",
            "master",
            "--event",
            "push",
            "--status",
            "completed",
            "--limit",
            "1",
            "--json",
            "databaseId,createdAt,conclusion,headSha",
        )
    )
    if not runs:
        raise SystemExit("No completed GPU run on master.")
    run = runs[0]
    print(
        f"Using run {run['databaseId']} ({run['createdAt']}, "
        f"{run['conclusion']}, {run['headSha'][:10]})"
    )
    return str(run["databaseId"])


def main(_):
    run_id = FLAGS.run_id or latest_master_run(FLAGS.repo)
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
                "durations-*",
                "--dir",
                tmp,
            )
        except subprocess.CalledProcessError as e:
            raise SystemExit(
                f"{e.stderr.strip()}\nRun {run_id} has no durations-* "
                "artifacts. They are uploaded by master runs and kept for "
                "14 days; pick a newer run with --run_id."
            )
        tmp = pathlib.Path(tmp)
        # Validate every backend before writing anything, so a missing shard
        # cannot leave the committed files half refreshed.
        for backend in BACKENDS:
            files = sorted(tmp.glob(f"durations-{backend}-*/{backend}.json"))
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
        target = OUT_DIR / f"{backend}.json"
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
