# Experiment coverage matrix

`reproduction.yaml` uses the JSON-compatible subset of YAML 1.2, so the generator needs only Python standard-library modules. Run `python experiments/matrices/generate_tables.py` from the repository root. It exports `experiment_matrix.csv` and `coverage_table.md`.

The manifest describes suite-specific task/method coverage, integration type, rank-4 variants, ablations and provenance fields. It avoids an unrestricted model/method/task cross-product. The reported profile lists seed 42; the three-seed extension lists 42, 43 and 44 and remains planned. Matrix rows do not assert execution or produce measured means/standard deviations.

This generator does not submit, stop or change any cluster jobs. Existing task scripts continue to read their reference JSON files. To calculate a multi-seed result, first match actual measured records by run ID and configuration; use sample standard deviation with ddof=1.
