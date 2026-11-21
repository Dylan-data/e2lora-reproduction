"""Create a reusable ordered manifest from the existing local MetaMathQA JSON."""
import argparse
from datetime import datetime, timezone

from e2lora.dataset.loading import sample_indices

from common import digest, file_sha256, output_path, read_json, resolve, write_json


def prepare(source, *, count, seed, expected_rows, revision, output):
    source, output = resolve(source), output_path(output)
    source_hash = file_sha256(source)
    if output.exists():
        existing = read_json(output)
        if (existing["source_sha256"], existing["count"], existing["seed"], existing["source_revision"]) != (source_hash, count, seed, revision):
            raise ValueError("Existing manifest uses a different source/seed/count/revision; choose a new path.")
        if existing["indices_sha256"] != digest(existing["indices"]) or existing["indices"] != sample_indices(existing["source_rows"], count, seed):
            raise ValueError("Existing manifest indices are corrupted or use a different sampling algorithm.")
        if expected_rows is not None and existing["source_rows"] != expected_rows:
            raise ValueError("Existing manifest row count differs from expected_rows.")
        return existing
    rows = read_json(source)
    if not isinstance(rows, list) or expected_rows is not None and len(rows) != expected_rows:
        raise ValueError(f"Unexpected MetaMathQA row count; expected {expected_rows}.")
    ids = sample_indices(len(rows), count, seed)
    for i in ids:
        if not isinstance(rows[i].get("query"), str) or not isinstance(rows[i].get("response"), str):
            raise ValueError(f"Invalid query/response at source row {i}.")
    manifest = {"schema_version": 1, "dataset": "MetaMathQA", "source_path": str(source),
                "source_revision": revision, "source_sha256": source_hash, "source_rows": len(rows),
                "count": count, "seed": seed, "sampling_algorithm": "e2lora.dataset.loading.sample_indices / Python random.Random.sample",
                "indices": ids, "indices_sha256": digest(ids), "index_base": 0,
                "status": "candidate_pending_team_manifest_alignment",
                "created_at_utc": datetime.now(timezone.utc).isoformat()}
    write_json(output, manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="/home/msds/yahui003/e2lora-storage/datasets/metamathqa/MetaMathQA-395K.json")
    parser.add_argument("--output", default="outputs/chloe/data/math-100k-seed42.json")
    parser.add_argument("--count", type=int, default=100000)
    parser.add_argument("--expected-rows", type=int, default=395000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--revision", default="aa4f34d3d2d3231299b5b03d9b3e5a20da45aa18")
    args = parser.parse_args()
    manifest = prepare(args.source, count=args.count, seed=args.seed, expected_rows=args.expected_rows,
                       revision=args.revision, output=args.output)
    print(f"{output_path(args.output)}: {manifest['count']} fixed IDs; {manifest['indices_sha256']}")


if __name__ == "__main__":
    main()
