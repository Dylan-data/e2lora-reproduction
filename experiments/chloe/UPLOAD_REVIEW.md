# Source review and branch integration

I keep my experiment code under `experiments/chloe/` and reuse the public adapter modules. I do not change another member's experiment directory. My source export includes code, configuration and small audit records; it excludes models, raw data, credentials, logs and checkpoints.

## Changes I contributed

I added the unified method/rank/sampling runner, explicit suite planning, SLURM submission and recovery, independent checkpoint evaluation, summaries and plots, CPU tests and source-export verification. I retained the original formal baseline entry point for existing run provenance.

The shared additions are `ablation.py`, `grouping.py`, `partial_rank.py`, `preparation.py` and `similarity.py`. The `adaptive.py` change releases temporary gradient objects and spells out non-retained autograd settings; it does not change the entropy or rank formulas. `lora_layers.py` matched main at the integration check.

Before integration I backed up my uncommitted source and tracked diff outside the repository. I merged main at `ae243032b96b175d1a9f83fe3d37e3d515449c39` into `chloe-ablation`, preserving the other members' tree entries, and pushed my branch. I checked that the difference from main was restricted to my directory, the listed adapter changes and ignore rules.

The local sparse view omits four checkpoint blobs already tracked upstream under the generation member's outputs. Their Git history is unchanged. My models, data and checkpoints were not added to the upload. I preserve the existing main ignore rules and exclude my local outputs and backup files.

## Repeating the checks

```bash
python experiments/chloe/export_code.py --output outputs/chloe/review/chloe-source.zip
python experiments/chloe/review_upload.py --manifest /path/to/UPLOAD-MANIFEST.json
git diff --stat
git diff --cached --stat
git diff --cached --check
```

I inspect exact source paths before staging rather than using `git add .`. I do not reset a dirty branch, force-push over another branch, or present a successful source test as a new full GPU experiment. The current protocol, commands and completed scores are in README.md; verification results are in VALIDATION.md.
