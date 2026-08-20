# Q-Former oracle residual-steering pilot

## Question

This development pilot asks one narrow causal question: if the correct visible
event attributes are inserted at the native Q-Former output interface, while
the projector, Llama, checkpoint, vocabulary restriction and decoding are held
fixed, does the generated commentary become more factually correct?

The answer is not a final module ranking. The 48 labels are video-only
development annotations produced by Codex, not independent human gold, and the
oracle residual is not a deployable replacement Q-Former.

## Why this interface

The cached Q-Former output has shape `[32, 768]`. The historical projector
accepts 768-dimensional tokens and emits the 4096-dimensional Llama prefix.
The intervention therefore preserves the exact downstream interface. A text
fact prompt would instead change the decoder input language and is retained
only as a non-rankable language upper bound.

The cached `temporal_output` and `qformer_input` are not independent stages in
the current trace: `cache_layers_shard_gpu.py` explicitly clones the latter
into the former. Their identical Stage-2 scores must not be counted as two
separate pieces of evidence.

## Frozen packet and leakage boundary

- Use exactly the locked 48 clips from 36 matches.
- Use four-fold `GroupKFold` by `match_id`.
- Estimate every centroid, direction, scale and clipping threshold from the
  training matches of that fold only.
- A held-out sample contributes only its locked video-only fact value, which
  selects the oracle sign of each direction.
- Exclude tasks with fewer than four positives or four negatives in any
  training fold. The resulting 12-task list is frozen in the JSON config.
- Do not use reference commentary to construct steering vectors or to perform
  the primary factual review.
- If a clip has no clear label among the 12 stable tasks, leave all four
  conditions identical to baseline and exclude that clip from intervention-
  effect denominators; retain it in the packet so sample accounting stays
  explicit.

## Intervention

For fold `f` and fact task `k`, compute the full-tensor class-centroid
difference

`D[f,k] = mean(Q_train | y_k=1) - mean(Q_train | y_k=0)`.

For held-out clip `i`, average the signed directions for its eligible stable
tasks:

`R[i] = mean_k((2*y[i,k]-1) * D[f,k])`.

Run two preregistered doses, `alpha=0.5` and `alpha=1.0`, then cap the total
Frobenius norm at half the training fold's median distance from its Q-Former
centroid. The modified tensor is `Q'[i] = Q[i] + clip(alpha*R[i])`.

Four paired conditions are generated in one later authorized run: the original
cached Q-Former output, a deterministic norm-matched cyclic-task control, and
the two oracle doses. The control breaks the task-to-direction mapping while
preserving each sample's perturbation norm.

## Downstream controls

The original epoch-11 `llama_proj` and Llama weights remain unchanged. All
conditions use the same restricted vocabulary, BOS/EOS/PAD handling, maximum
length and deterministic five-beam decoding. Deterministic decoding is used so
paired differences cannot be attributed to sampling noise; reproducing the
sampled Stage-1 string is not an assertion of this pilot.

No dataset loader, visual encoder forward, Q-Former forward, training,
backward, optimizer, scheduler or checkpoint write is allowed.

## Reading the result

The primary result is paired video-fact consistency review on the 48 clips.
The pilot is a positive rescue only if an oracle condition adds at least 5/48
central-fact-correct clips, the number improved is at least twice the number
regressed, and the norm-matched control does not obtain the same gain.

A positive result means Q-Former-output information loss is a causal
contributor under this intervention. It does not prove that Q-Former is the
only bottleneck. A negative result also does not exonerate Q-Former, because a
centroid direction may be an inadequate way to insert the missing fact.

Final ranking requires capacity- and supervision-matched interventions for
the competing modules and replication on new match-disjoint videos.

## Execution boundary

The JSON config has `execution_authorized=false`. Static preflight may inspect
metadata and labels on CPU. It may not load the model, query or use a GPU, or
generate commentary. A later GPU run requires the attribute-probe trigger to
pass and a separate executable worker/config under the applicable GPU
authorization rules. Every later output must be new and the run must stop at
`REVIEW_REQUIRED`.
