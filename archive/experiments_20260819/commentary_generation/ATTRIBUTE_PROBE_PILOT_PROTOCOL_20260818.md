# SoccerMaster attribute probe pilot

Status: 48-clip video-only development labels are locked. Frozen feature
extraction is awaiting a fresh GPU resource report and explicit approval.

## Question

Does the frozen SoccerMaster `epoch_19` backbone already expose enough
information for five decomposed video attributes?

- event
- action
- result
- play phase
- per-slot observability

The comparison is fixed to three readouts from the same video forward:

1. temporal mean of `global_features`;
2. the full `[T,D]` `global_features` sequence;
3. spatially pooled `[T,D]` `local_features_late`.

The full local patch tensor must not be written to disk.

## Development packet

`prepare_attribute_probe_pilot_120.py` first built a broad 120-clip candidate
pool. `prepare_attribute_probe_pilot_48.py` then selected an action/result-led
48-clip pilot covering 36 matches. Selection was stratified by the existing
reference-relative weak event label. Those weak labels were retained only in
the coordinator manifest and were not exposed during video annotation.

`video_only_labels.json` was locked before the coordinator manifest was opened.
It records:

- multi-label event and action values;
- result and play phase;
- separate evidence status for event, action, result, and phase;
- clip-level observable / ambiguous / invisible status;
- confidence and a short video-evidence note.

Empty values are not negative labels. Probe losses must be masked whenever the
corresponding slot is not observable or not applicable.

The labels are explicitly `codex_video_only_development`, not independent human
gold. They can screen architecture choices but cannot support a benchmark or a
paper-level accuracy claim.

## Probe implementation

`extract_attribute_probe_features_gpu.py` performs one inference-only pass over
the 48 clips using the frozen high-resolution `epoch_19` vision backbone. It
writes only:

- `global_features.mean(time)`;
- full `global_features [T,D]`;
- `local_features_late.mean(spatial_patches) [T,D]`.

`run_attribute_probe_cpu.py` uses four-fold match-grouped out-of-fold probes.
The sequence inputs are converted to a fixed temporal-pyramid descriptor before
fold-local PCA and L2 logistic regression. This keeps the readout light and
avoids fitting a large temporal network to 48 clips. Factual targets use only
the samples whose corresponding head evidence is clear. Because there is only
one invisible clip, the pilot estimates observable versus not-fully-observable;
it does not claim a reliable three-way observability result.

This is a development pilot, not an independent holdout. Reference-derived
labels may be used for sampling and later disagreement analysis, but not as a
substitute for video facts.

## Execution boundary

Packet preparation and contact-sheet annotation were CPU-only. Frozen feature
extraction requires a fresh `nvidia-smi` report and separate approval. Probe
training is CPU-only. No stage writes a model checkpoint or modifies the frozen
backbone.
