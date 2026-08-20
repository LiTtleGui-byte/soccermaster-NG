# Fixed-200 commentary event separability v2

This is the single formal run of the frozen v2 reference-relative silver-label protocol. It is CPU-only and does not establish video-ground-truth events or train SoccerMaster.

## Label extraction

- Coverage: 186/200 (93.0%)
- Selected-sentence ambiguity: 26/200 (13.0%)
- Whole-reference multi-event: 49/200 (24.5%)
- Raw `other`: 14/200
- Rare rule: raw primary classes with fewer than 10 samples merge into `other_rare`; no sample is excluded.
- Final class counts: `{"corner": 34, "cross": 10, "foul_or_free_kick": 28, "other": 14, "other_rare": 27, "pass_or_build_up": 33, "shot_or_save": 28, "substitution": 16, "yellow_card": 10}`

The sentence rule, complete dictionary, all hits, vetoes, and final labels are stored in `result.json`. There are no manual overrides.

## Leakage-safe results

| Representation | Macro-F1 | Balanced accuracy | Shuffled-label macro-F1 | Majority macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| mean_pooled | 0.4798 | 0.4601 | 0.1102 | 0.0323 |
| query_slot_preserving | 0.4963 | 0.4990 | 0.1286 | 0.0323 |

Every StandardScaler, PCA, and logistic classifier is fit only on each training fold. The five stratified folds and all randomized operations use seed 20260816. Full confusion matrices and fold records are in `result.json`.

## Boundary

These results apply only to one fixed 200-sample post-Q-Former/post-projector cache and automatically extracted reference labels. They do not establish video factual correctness, causal event encoding, full-test performance, or raw Q-Former-state separability.
