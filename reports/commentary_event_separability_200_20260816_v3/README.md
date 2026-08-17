# Fixed-200 commentary event separability v3

This is the single formal run of the frozen v3 reference-relative silver-label protocol. It is CPU-only and does not establish video-ground-truth events or train SoccerMaster.

## Label extraction

- Coverage: 191/200 (95.5%)
- Selected-sentence ambiguity: 27/200 (13.5%)
- Whole-reference multi-event: 50/200 (25.0%)
- Raw `other`: 9/200
- Rare rule: raw primary classes with fewer than 10 samples merge into `other_rare`; no sample is excluded.
- Final class counts: `{"corner": 34, "cross": 10, "foul_or_free_kick": 29, "other_rare": 35, "pass_or_build_up": 34, "shot_or_save": 28, "substitution": 19, "yellow_card": 11}`

The complete frozen dictionary, sentence audit, all hits, vetoes, and final labels are stored in `result.json`. There are no manual overrides.

## Leakage-safe results

| Representation | Macro-F1 | Balanced accuracy | Shuffled-label macro-F1 | Majority macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| mean_pooled | 0.6169 | 0.6072 | 0.1321 | 0.0562 |
| query_slot_preserving | 0.5657 | 0.5643 | 0.0847 | 0.0562 |

Every StandardScaler, PCA, and logistic classifier is fit only on each training fold. Five-fold stratification and randomized operations use seed 20260816. Full confusion matrices and fold records are in `result.json`.

## Boundary

These results apply only to one fixed 200-sample post-Q-Former/post-projector cache and automatically extracted reference labels. They do not establish video factual correctness, causal event encoding, full-test performance, or raw Q-Former-state separability.
