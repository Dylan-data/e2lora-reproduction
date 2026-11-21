# Code generation protocol: excluded route

I retain this note as a record of the initially planned code-generation route. The project later excluded HumanEval because an isolated execution environment was unavailable; Code-Feedback was excluded with it. I do not schedule code GPU jobs or let these pending fields block the mathematical experiments.

The public candidate was `m-a-p/Code-Feedback`, revision `f411b16a97c910ac9acf8b0d0948e340aa77cc34`, train split, approximately 66,400 rows. That is not the paper's100 K examples. `training_source=null` in the shared references does not identify a complete training corpus or preprocessing rule.

If I revisit this route, I need a fixed source/file hash, actual example count, filtering and multi-turn expansion rules, stable original-row/turn IDs, normalization, training prompt, EOS/masking/truncation policy and an independent checkpoint. I would record any difference from the paper instead of repeating examples silently to reach 100 K.

HumanEval would additionally require a fixed task dataset, ordered task IDs, prompt/completion handling, sample count, decoding and stopping rules, pass@1 estimator, execution harness revision, dependency lock, isolated runtime and timeouts. I would validate known correct, incorrect and timed-out completions before evaluating generated programs.

The historical placeholder is `configs/code-protocol.pending.json`; it is not a working HumanEval pipeline. Math checkpoints and GSM 8 K scores cannot fill code-result fields. I leave excluded HumanEval results absent rather than entering zero. The active scope and commands are in [README.md](README.md).
