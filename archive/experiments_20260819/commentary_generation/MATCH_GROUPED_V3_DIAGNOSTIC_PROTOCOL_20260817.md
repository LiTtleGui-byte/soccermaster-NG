# Fixed-200 v3 match-grouped correction protocol

Frozen: 2026-08-17 UTC, before the corrected metrics were run.

This is a post-hoc leakage correction, not a new blind model experiment. The v3 labels, features, representations, PCA dimensions, classifier, seed, fold count, shuffled-label baseline, majority baseline, and metrics remain unchanged. The only intended evaluation change is replacing clip-level `StratifiedKFold` with `StratifiedGroupKFold`, where a group is the parent match directory of each video path.

Required assertions:

- all 200 manifest/cache/E1/E2 identities remain exact;
- all 200 samples remain assigned to exactly one test fold;
- every fold has zero match overlap between train and test;
- fold-local scaler, PCA, and classifier fitting remains unchanged;
- no video, model, checkpoint, Torch, GPU, inference, generation, or SoccerMaster training is used;
- the original v3 result remains unchanged and is reported beside the corrected result.

The corrected result may be worse, unstable across classes, or fail to support five balanced folds. It must be reported as observed and must not be tuned by changing the seed, groups, labels, or fold count after metrics are visible.

Output: `reports/commentary_event_separability_200_20260817_v3_match_grouped/`.
