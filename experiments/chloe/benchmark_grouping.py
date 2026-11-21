"""CPU grouping latency. Synthetic results are never Table 10 reproductions."""
import argparse
import json
import os
import platform
import random
import statistics

from common import digest, output_path, read_json, write_json
from e2lora.adapters.grouping import search_intervals


def benchmark(matrix, *, method, warmup, repeats, **options):
    if warmup < 0 or repeats < 1:
        raise ValueError("Invalid warmup/repeat count")
    for _ in range(warmup):
        search_intervals(matrix, method=method, **options)
    samples, last = [], None
    for _ in range(repeats):
        _, last = search_intervals(matrix, method=method, **options)
        samples.append(last["grouping_seconds"] * 1000)
    ordered = sorted(samples)
    return {"method": method, "options": options, "warmup": warmup, "repeats": repeats,
            "milliseconds": {"min": min(samples), "median": statistics.median(samples),
                             "mean": statistics.mean(samples), "max": max(samples),
                             "p95": ordered[min(len(samples) - 1, int(.95 * len(samples)))]},
            "last_search": last}


def benchmark_total(matrices, *, method, warmup, repeats, **options):
    """Per-round sum of kernel times over all families; identical matrices for both methods."""
    if warmup < 0 or repeats < 1:
        raise ValueError("Invalid warmup/repeat count")
    values = []
    for index in range(warmup + repeats):
        milliseconds = sum(search_intervals(matrix, method=method, **options)[1]["grouping_seconds"] * 1000
                           for matrix in matrices.values())
        if index >= warmup:
            values.append(milliseconds)
    return {"method": method, "families": list(matrices), "repeats": repeats, "warmup": warmup,
            "aggregation": "per-round sum of search_intervals kernel times over all provided families",
            "milliseconds": {"median": statistics.median(values), "mean": statistics.mean(values),
                             "sample_std": statistics.stdev(values) if repeats > 1 else 0.0,
                             "min": min(values), "max": max(values)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help='JSON mapping family names to matrices; no gradients loaded')
    parser.add_argument("--sizes", nargs="+", type=int, default=[12, 32])
    parser.add_argument("--penalty-lambda", required=True, type=float)
    parser.add_argument("--mean-policy", required=True, choices=["include_diagonal", "exclude_diagonal"])
    parser.add_argument("--singleton-cost", type=float)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--output", default="outputs/chloe/cpu/grouping-benchmark.json")
    args = parser.parse_args()
    if args.input:
        matrices, source = read_json(args.input), "provided_cached_similarity_matrices"
    else:
        rng = random.Random(101)
        matrices = {}
        for n in args.sizes:
            if n < 1:
                raise ValueError("sizes must be positive")
            matrix = [[0.0] * n for _ in range(n)]
            for i in range(n):
                for j in range(i + 1):
                    matrix[i][j] = matrix[j][i] = (1.0 if i == j else
                        rng.uniform(.75, .95) if i // 4 == j // 4 else rng.uniform(-.2, .2))
            matrices[f"synthetic-n{n}"] = matrix
        source = "synthetic_block_matrices_seed101"
    results = {}
    for name, matrix in matrices.items():
        options = dict(penalty_lambda=args.penalty_lambda, mean_policy=args.mean_policy,
                       singleton_cost=args.singleton_cost)
        results[name] = {"matrix_sha256": digest(matrix), "layers": len(matrix),
                         "greedy": benchmark(matrix, method="greedy", warmup=args.warmup, repeats=args.repeats),
                         "dp": benchmark(matrix, method="dp", warmup=args.warmup, repeats=args.repeats, **options)}
    cpu = platform.processor()
    if os.path.isfile("/proc/cpuinfo"):
        with open("/proc/cpuinfo") as stream:
            cpu = next((line.split(":", 1)[1].strip() for line in stream if line.startswith("model name")), "unknown")
    result = {"run_type": "cpu_grouping_benchmark", "eligible_for_formal_tables": False,
              "source": source, "cpu": cpu, "python": platform.python_version(), "platform": platform.platform(),
              "implementation": "Python scalar loops; one thread; no GPU operations",
              "timing_scope": "validation + prefix/diagonal sums (DP) or thresholds (Greedy) + search + backtracking",
              "excluded": "gradient collection, similarity, rank allocation, loading, adapters, training",
              "results": results, "task_performance": None,
              "total_all_families": {
                  "greedy": benchmark_total(matrices, method="greedy", warmup=args.warmup, repeats=args.repeats),
                  "dp": benchmark_total(matrices, method="dp", warmup=args.warmup, repeats=args.repeats,
                      penalty_lambda=args.penalty_lambda, mean_policy=args.mean_policy, singleton_cost=args.singleton_cost)}}
    write_json(output_path(args.output), result)
    print(json.dumps({name: {method: row[method]["milliseconds"]["median"] for method in ["greedy", "dp"]}
                      for name, row in results.items()}, indent=2))


if __name__ == "__main__":
    main()
