"""Run one already compiled matrix entry; defaults to CPU validation."""
import argparse
import os
import subprocess
import sys

from common import ROOT, read_json, resolve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", required=True)
    parser.add_argument("--index", type=int)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    batch = read_json(args.batch)
    index = args.index
    if index is None:
        index = int(os.environ["SLURM_ARRAY_TASK_ID"])
    if not 0 <= index < batch["count"] or len(batch["entries"]) != batch["count"]:
        raise ValueError("Invalid batch index/count")
    entry = batch["entries"][index]
    if entry["index"] != index:
        raise ValueError("Batch manifest indices changed")
    command = [sys.executable, str(ROOT / "experiments/chloe/formal_runner.py"), "--config", str(resolve(entry["config"]))]
    if args.execute:
        command.append("--execute")
    subprocess.run(command, check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
