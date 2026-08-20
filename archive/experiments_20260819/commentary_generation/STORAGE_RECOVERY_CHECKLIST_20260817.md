# Commentary diagnostics: storage recovery checklist

This checklist does not authorize access, GPU use, inference, or training. Run each stage only after NAS/GPFS is reported available and the applicable project authorization is obtained.

Later on 2026-08-17, node 200's `/remote-home` was confirmed restored as the
read-only `gpfsdata` GPFS mount. The verified 202-to-NAS bridge in
`STORAGE_BRIDGE_202_TO_200_20260817.md` remains a frozen asset bundle and
provenance/fallback record; mount recovery does not silently switch any
existing runtime or armed configuration away from that bundle. `/mnt/nas2`
video access remains separate.

Current 2026-08-17 state: `/mnt/nas2` is restored on node 200. The recorded
train/test roots and one fixed-200 video are readable; `ffprobe` successfully
opened that 30.022-second video. The dedicated model bundle is at
`/mnt/nas/tianlin/SoccerMasterAssets/commentary_generation_epoch11_20260817`.
Treat source videos and verified model files as immutable experiment inputs.

The CPU-only asset checks in items 1–3 below have passed. Items 4–8 remain
gates for the future Locked Match Holdout or layer-cache run where applicable.

## CPU-only access preflight

1. Confirm the intended NAS video roots and GPFS model/checkpoint roots resolve to the identities already recorded in `assets.json` and the fixed manifests.
2. Check a small fixed set of video paths for readable metadata; do not recursively scan or decode the full dataset as the first action.
3. Confirm the Llama, SigLIP2, BERT, generation checkpoint, visual backbone, tokenizer, and restricted-vocabulary assets are readable and unchanged in identity.
4. Recover the relevant generation/backbone training manifests and derive match IDs using the same parent-directory rule as the development inventory.
5. Reject every Locked Match Holdout candidate whose match or clip appears in relevant training data or the current 49 development matches.
6. Freeze the new-match holdout manifest before inspecting model outputs or judge results.
7. Confirm new local output directories are absent and estimate free space before any layer cache. Local patch features require a separate budget and remain subset-only by default.
8. Review the exact cache/generation entry for remote writes, implicit downloads, model cache writes, and overwrite behavior.

## First minimal real run

1. Select one development clip, not the Locked Match Holdout.
2. Reconfirm its frame indices, preprocessing, checkpoint, tokenizer, and decode identity.
3. Before any GPU action, run `nvidia-smi`, report every relevant GPU and process, and obtain explicit authorization for the exact command.
4. Capture one sample through every layer in `LAYER_CACHE_CONTRACT_20260817.json`.
5. Stop after checking identity, shape, dtype, finite values, output isolation, and absence of residual GPU processes.

## Expansion order

1. Event-balanced, match-grouped development subset.
2. Full development set only for layers whose storage budget is accepted.
3. Probe screening.
4. Interface-matched interventions.
5. Three-seed winner confirmation.
6. One final Locked Match Holdout evaluation.

Do not skip directly from restored paths to full inference or training.
