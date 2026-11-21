# Saved-plan verification for the initial E2LoRA smoke

I checked the Stage A plan before the training/recovery run. The subsequent Stage B job 54466 completed 20 updates, resumed at update 10 and scored 12/32=37.5%. This page records the older all-trainable plan, not a current partial-rank configuration.

The configuration semantic SHA256 was `24f0919860e6df77fb05477180f6400167099be86311504c22ee006db0422b60`; the prepared-plan file SHA256 was `2e3c321a148ad3fb34c0a7d1c219df7c746a1d0aa52cf3c77cce53270c0f9064`. I matched the source/data/tokenizer fingerprint, recomputed intervals from saved RMI and ranks from saved entropy, and compared all four diagnostics with the installed plan.

| Projection | Inclusive layer intervals | Group ranks | Total rank |
|---|---|---|---:|
| Q | 0-1;2-14;15;16;17-28;29;30;31 | 33,30,33,31,33,29,35,32 | 256 |
| K | 0;1;2-14;15-17;18-26;27;28-30;31 | 31,32,31,32,33,32,33,32 | 256 |
| V | 0-4;5-15;16-21;22-28;29-31 | 39,47,52,58,59 | 255 |
| O | 0-15;16-18;19-31 | 78,87,91 | 256 |

I verified that 24 groups cover all 128 targets once, in depth order and without crossing projection families. Installation on the actual Llama meta architecture gave 48 unique trainable tensors and 6,810,624 parameters. Group members referenced the same A/B objects; base weights, embeddings and the output head were frozen.

All group ranks were trainable and `frozen_rank=0`. V's total 255, rather than 256, explains the 5,120-parameter difference from ordinary LoRA r8. It does not establish a50% saving. Negative entropy warnings were retained under the literal round-based rule.

The two sampling positions were 61 and 89, corresponding to original rows 329588 and 37435. They generated 128*2=256 target backward calls and zero optimizer updates. Stage B reused the prepared plan, saved at update 10 and resumed through 20; it did not recompute gradients.

I preserved the original prepared plan, diagnostics and baseline result directory. The historical 64 G submission request is superseded by the later 30 G user limit. For current commands I use [README.md](README.md), not the old resource request.
