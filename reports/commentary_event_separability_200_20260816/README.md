# Fixed-200 commentary event separability

This is a CPU-only, reference-relative silver-label diagnostic. It does not establish video-ground-truth event labels or train SoccerMaster.

## Label extraction

- Coverage before rare merging: 187/200 (93.5%)
- Multi-category ambiguity: 54/200 (27.0%)
- Raw `other`: 13/200
- Rare rule: raw primary classes with fewer than 10 samples are merged into `other_rare`; no sample is excluded.
- Final class counts: `{"corner": 39, "foul_or_free_kick": 26, "goal": 19, "other": 13, "other_rare": 25, "pass_or_build_up": 22, "shot_or_save": 29, "substitution": 16, "yellow_card": 11}`

Primary selection uses the fixed salience order and regex dictionary stored verbatim in `result.json`. All category hits, vetoes, and final labels are also stored per sample; there are no manual overrides.

## Leakage-safe results

| Representation | Macro-F1 | Balanced accuracy | Shuffled-label macro-F1 | Majority macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| mean_pooled | 0.4326 | 0.4355 | 0.1307 | 0.0363 |
| query_slot_preserving | 0.4675 | 0.4663 | 0.1254 | 0.0363 |

Every StandardScaler, PCA, and logistic classifier is fit only on each training fold. The five stratified folds and all randomized operations use fixed seed 20260816. Full confusion matrices and fold records are in `result.json`.

## Interpretation boundary

Above-baseline out-of-fold performance supports reference-event separability in these cached post-Q-Former/post-projector prefixes for this fixed sample only. It is not evidence of video factual correctness, causal event encoding, full-test performance, or raw Q-Former-state separability.
