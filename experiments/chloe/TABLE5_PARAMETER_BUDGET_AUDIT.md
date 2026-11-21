# Table 5 parameter-budget audit

I audited the literal round-based shared-adapter budget on November 18,2026, using the saved Job 54457 smoke statistics and the actual Llama architecture on the meta device. I inspected Parameter identity, shapes, trainability and counts without loading 8B weights or taking optimizer steps. The full machine record is [table 5-budget-audit-v1.json](report/table5-budget-audit-v1.json).

## Budget derivation

Equation 7/Algorithm 3 assigns `round(interval_max_entropy / sum_interval_max_entropy * N * base_rank)`, where N is the number of original layers, not the number of groups. Greedy grouping is performed independently for each depth-ordered projection family. I retained negative entropies under the literal rule; I did not shift, take absolute values or repair the total silently.

For one shared interval, A has shape `[rank,d_in]`, B has shape `[d_out,rank]`, and its physical count is `rank*(d_in+d_out)`. Q/O each contribute 8192 parameters per rank and K/V5120. With 32 layers and both factors trained, the unrounded budget is therefore approximately `851968 * base_rank`, regardless of how the total is partitioned across groups.

For K groups, independent nearest-integer rounding changes a family's total rank by at most K/2. The observed 8/8/5/3 groups give a combined parameter-error bound 78,336. Even allowing 32 groups in every family gives 425,984: r8 is at least 6,389,760 and r16 at least 13,205,504 under that bound. Rounding alone cannot explain 3.61 M/7.22 M.

| Projection | Groups | W shape | Parameters per rank | r8 rank sum | r8 parameters | r16 rank sum | r16 parameters |
|---|---:|---|---:|---:|---:|---:|---:|
| q_proj |8 |[4096,4096] |8192 |256 |2,097,152 |512 |4,194,304 |
| k_proj |8 |[1024,4096] |5120 |256 |1,310,720 |513 |2,626,560 |
| v_proj |5 |[1024,4096] |5120 |255 |1,305,600 |511 |2,616,320 |
| o_proj |3 |[4096,4096] |8192 |256 |2,097,152 |512 |4,194,304 |

I obtained 6,810,624 parameters at r8 and 13,631,488 at r16. The former matches the old Stage A/B installation. These cached length 256/two-example statistics are neither author-produced groups nor a fresh formal 2048 sampling pass.

## Group-level counts

The intervals below use zero-based inclusive endpoints. Every row counts one shared A/B pair, not one pair per layer.

### q_proj

| Inclusive layers | r8 rank | r8 A/B shapes | r8 A+B parameters | r16 rank | r16 A/B shapes | r16 A+B parameters |
|---|---:|---|---:|---:|---|---:|
| 0–1 | 33 | [33,4096] / [4096,33] | 135,168+135,168=270,336 | 67 | [67,4096] / [4096,67] | 274,432+274,432=548,864 |
| 2–14 | 30 | [30,4096] / [4096,30] | 122,880+122,880=245,760 | 60 | [60,4096] / [4096,60] | 245,760+245,760=491,520 |
| 15 | 33 | [33,4096] / [4096,33] | 135,168+135,168=270,336 | 66 | [66,4096] / [4096,66] | 270,336+270,336=540,672 |
| 16 | 31 | [31,4096] / [4096,31] | 126,976+126,976=253,952 | 62 | [62,4096] / [4096,62] | 253,952+253,952=507,904 |
| 17–28 | 33 | [33,4096] / [4096,33] | 135,168+135,168=270,336 | 66 | [66,4096] / [4096,66] | 270,336+270,336=540,672 |
| 29 | 29 | [29,4096] / [4096,29] | 118,784+118,784=237,568 | 57 | [57,4096] / [4096,57] | 233,472+233,472=466,944 |
| 30 | 35 | [35,4096] / [4096,35] | 143,360+143,360=286,720 | 70 | [70,4096] / [4096,70] | 286,720+286,720=573,440 |
| 31 | 32 | [32,4096] / [4096,32] | 131,072+131,072=262,144 | 64 | [64,4096] / [4096,64] | 262,144+262,144=524,288 |

### k_proj

| Inclusive layers | r8 rank | r8 A/B shapes | r8 A+B parameters | r16 rank | r16 A/B shapes | r16 A+B parameters |
|---|---:|---|---:|---:|---|---:|
| 0 | 31 | [31,4096] / [1024,31] | 126,976+31,744=158,720 | 63 | [63,4096] / [1024,63] | 258,048+64,512=322,560 |
| 1 | 32 | [32,4096] / [1024,32] | 131,072+32,768=163,840 | 65 | [65,4096] / [1024,65] | 266,240+66,560=332,800 |
| 2–14 | 31 | [31,4096] / [1024,31] | 126,976+31,744=158,720 | 61 | [61,4096] / [1024,61] | 249,856+62,464=312,320 |
| 15–17 | 32 | [32,4096] / [1024,32] | 131,072+32,768=163,840 | 63 | [63,4096] / [1024,63] | 258,048+64,512=322,560 |
| 18–26 | 33 | [33,4096] / [1024,33] | 135,168+33,792=168,960 | 66 | [66,4096] / [1024,66] | 270,336+67,584=337,920 |
| 27 | 32 | [32,4096] / [1024,32] | 131,072+32,768=163,840 | 64 | [64,4096] / [1024,64] | 262,144+65,536=327,680 |
| 28–30 | 33 | [33,4096] / [1024,33] | 135,168+33,792=168,960 | 67 | [67,4096] / [1024,67] | 274,432+68,608=343,040 |
| 31 | 32 | [32,4096] / [1024,32] | 131,072+32,768=163,840 | 64 | [64,4096] / [1024,64] | 262,144+65,536=327,680 |

### v_proj

| Inclusive layers | r8 rank | r8 A/B shapes | r8 A+B parameters | r16 rank | r16 A/B shapes | r16 A+B parameters |
|---|---:|---|---:|---:|---|---:|
| 0–4 | 39 | [39,4096] / [1024,39] | 159,744+39,936=199,680 | 79 | [79,4096] / [1024,79] | 323,584+80,896=404,480 |
| 5–15 | 47 | [47,4096] / [1024,47] | 192,512+48,128=240,640 | 94 | [94,4096] / [1024,94] | 385,024+96,256=481,280 |
| 16–21 | 52 | [52,4096] / [1024,52] | 212,992+53,248=266,240 | 104 | [104,4096] / [1024,104] | 425,984+106,496=532,480 |
| 22–28 | 58 | [58,4096] / [1024,58] | 237,568+59,392=296,960 | 115 | [115,4096] / [1024,115] | 471,040+117,760=588,800 |
| 29–31 | 59 | [59,4096] / [1024,59] | 241,664+60,416=302,080 | 119 | [119,4096] / [1024,119] | 487,424+121,856=609,280 |

### o_proj

| Inclusive layers | r8 rank | r8 A/B shapes | r8 A+B parameters | r16 rank | r16 A/B shapes | r16 A+B parameters |
|---|---:|---|---:|---:|---|---:|
| 0–15 | 78 | [78,4096] / [4096,78] | 319,488+319,488=638,976 | 156 | [156,4096] / [4096,156] | 638,976+638,976=1,277,952 |
| 16–18 | 87 | [87,4096] / [4096,87] | 356,352+356,352=712,704 | 174 | [174,4096] / [4096,174] | 712,704+712,704=1,425,408 |
| 19–31 | 91 | [91,4096] / [4096,91] | 372,736+372,736=745,472 | 182 | [182,4096] / [4096,182] | 745,472+745,472=1,490,944 |

## Interpretation

I count unique physical trainable Parameters, not active forward coordinates. Each group's full stored rank is used in the old all-trainable path. A zero gradient or a forward slice does not make part of a Parameter frozen; optimizer decay or momentum can still change it.

All 48 installations across r8/r16 verified within-group sharing and independence across groups. The base, embeddings and head were frozen. There were no fixed adapter buffers or unused rank suffixes in that audit. Scaling by alpha/rank affects updates, not the count.

The team's later total 256/active 139 policy produces 3,700,736 parameters, but 139 and its frozen initialization are not derived from the literal formula above. I keep that policy, its budget gap and the paper's displayed values distinct. I do not change ranks to fit the paper. The current configured extension is documented in README.md, and the partial-rank GPU acceptance record is in TEAM_RANK_CHECK_README.md.

References: [Equations 5-7](https://proceedings.iclr.cc/paper_files/paper/2026/file/7b434657e8b8c3bfbf1d61d6e0026f90-Paper-Conference.pdf#page=6), [Algorithms 2/3](https://proceedings.iclr.cc/paper_files/paper/2026/file/7b434657e8b8c3bfbf1d61d6e0026f90-Paper-Conference.pdf#page=27), [Table 5](https://proceedings.iclr.cc/paper_files/paper/2026/file/7b434657e8b8c3bfbf1d61d6e0026f90-Paper-Conference.pdf#page=9).
