# DP grouping and timing

I implemented continuous DP grouping in `e2lora.adapters.grouping.search_intervals`, alongside the existing greedy anchor/threshold method. I use half-open intervals and minimize `sum(-mean(segment)) + lambda * number_of_groups`, with two-dimensional prefix sums and predecessor backtracking. Search complexity is O(N^2).

I require a finite nonnegative penalty and an explicit diagonal policy. With the diagonal excluded, singleton cost must also be specified. For an exact objective tie, I prefer fewer groups and then the smaller predecessor. I reject empty, nonsquare, asymmetric or nonfinite input matrices.

I checked optimality against an independent oracle that enumerates all 2^(N-1) continuous partitions for N1 through 8 and computes segment means directly. Tests cover four penalties, both diagonal policies, singleton alternatives, negative entries, ties and invalid inputs.

## Development measurements

| Synthetic matrix | Greedy median ms | DP median ms |
|---|---:|---:|
| 12 layers | 0.03855 | 0.07894 |
| 32 layers | 0.20450 | 0.46377 |

I used seed 101, lambda 1, included diagonals,20 warmups and 200 repetitions. These are development measurements, not the paper's Table 10 results.

On the cached 32-layer Q/K/V/O RMI matrices from the mathematical smoke run, I recorded the following with 20 warmups and 1000 repetitions:

| Input | Greedy median ms | DP median ms |
|---|---:|---:|
| Q | 0.20358 | 0.46329 |
| K | 0.20363 | 0.46329 |
| V | 0.20334 | 0.46198 |
| O | 0.20316 | 0.46207 |
| Sum of four searches per repetition | 0.82008 | 1.87910 |

The final row is the median of per-repetition sums, not the sum of four medians. Timing includes input validation, threshold/prefix preparation, search and backtracking; it excludes gradients, similarity calculation, loading, adapter installation and training.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src python experiments/chloe/benchmark_grouping.py --penalty-lambda 1 --mean-policy include_diagonal
```

I can supply actual matrices with `--input`, using JSON shaped as `{family_name: square_matrix}`. Records are in [cpu-grouping-benchmark.json](report/cpu-grouping-benchmark.json) and [math-only-dp-benchmark-20261118.json](report/math-only-dp-benchmark-20261118.json).

The paper's Table 10 reports GLUE/CV task results and dataset-level aggregation. My Llama CPU checks do not replace those experiments. I leave task scores with the corresponding members and agree on the penalty, diagonal/singleton policy and aggregation before comparisons. Reference: [AppendixG](https://proceedings.iclr.cc/paper_files/paper/2026/file/7b434657e8b8c3bfbf1d61d6e0026f90-Paper-Conference.pdf#page=22).
