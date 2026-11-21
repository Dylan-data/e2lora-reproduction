# Generation-protocol handoff

I use the generation experiments as the reference for model/data identity, preprocessing, optimization and GSM 8 K evaluation. I compare the actual run configuration and artifacts, rather than assuming that matching README settings establish the same protocol.

My current baseline route verifies the normalized ordered 100 K hash and the complete ordered GSM 8 K manifest. I retain the earlier alignment checks in [MEMBER 4_ALIGNMENT_REPORT.md](MEMBER4_ALIGNMENT_REPORT.md). Code/HumanEval items are historical and outside the agreed Math-only scope.

## Information I retain for a reproducible comparison

| Area | Required record |
|---|---|
| Run identity | Entry point, commit/source hashes, environment, model/tokenizer revision and loaded local paths |
| Training set | Source revision/file hash, actual count, ordered row IDs or normalized hashes, preprocessing mapping and disjoint validation selection |
| Training prompt | Exact whitespace, BOS/EOS, response mask, loss normalization, truncation and packing |
| Optimization | Microbatch/accumulation/effective batch, tail handling, epochs/steps, AdamW groups/betas/decay, clipping and executed LR schedule |
| Adapters | Target modules, alpha/dropout, initialization, scaling, shared object identity, total and active group ranks |
| Recovery | Full adapter buffers/parameters, optimizer/scheduler, random states, data cursor and checkpoint selection rule |
| Evaluation | Fixed 1319-question GSM 8 K manifest, exact prompt, decoding/stop rules, scorer version, numerical edge cases and per-question records |
| Ablations | Method controls, metric conventions, gradient sample identities and Figure 5 fixed settings |

The confirmed Table 2 settings were seed 42, one epoch, sequence limit 2048 and 512 generated tokens. I did not infer every other setting from those four values. The completed baselines now specify the remaining choices in their actual configuration; any deliberate implementation difference stays documented.

## Historical external evaluator interface

The older `formal_runner.py` accepts a versioned `config_checkpoint_output_v1` evaluator via an argv list, not a shell string. Placeholders are `{python}`, `{script}`, `{config}`, `{checkpoint}` and `{output}`. The evaluator must reconstruct the saved shared plan and restore complete adapter state, including frozen buffers.

Its output includes formal task identity, protocol/checkpoint/task-ID hashes, complete evaluation count, metric and a score in[0,1], with per-question records. A legacy evaluator with different arguments or checkpoint structure is not interchangeable simply by changing its path. The new suite provides `evaluate_suite.py` for its own format.

```bash
PYTHONPATH=src python experiments/chloe/alignment.py --task math --protocol <chloe-protocol.json> --compare <generation-protocol.json>
```

I version protocol changes and use new output directories. I keep seeds for data selection, training and generation distinguishable, and I add repeats only when the actual seed list is specified.
