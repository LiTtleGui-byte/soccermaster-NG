# 202 remote-home to 200 NAS bridge

## Decision

Use node 202 as a temporary read source for assets that live under
`/remote-home`. The verified bundle now lives at
`/mnt/nas/tianlin/SoccerMasterAssets/commentary_generation_epoch11_20260817`.
Do not recreate `/remote-home` with ad-hoc system symlinks. Runtime entries
receive the bundle root explicitly through
`SOCCERMASTER_COMMENTARY_ASSET_ROOT`.

Later on 2026-08-17, gpu200's `/remote-home` was confirmed restored as the
read-only `gpfsdata` GPFS mount. The completed bridge remains the verified
input and provenance record for configurations already pointed at the NAS
bundle. Do not silently switch a frozen or armed run back to `/remote-home`;
such a change requires a new configuration identity and any applicable
re-arming or authorization.

This bridge does **not** carry the video dataset. The recorded video roots live
under `/mnt/nas2`, not `/remote-home`. On 2026-08-17 the NAS was restored on
node 200 and a fixed-200 video passed a read-only `ffprobe` check. For frozen
configurations already using this bridge, the path split remains:

1. copy `/remote-home` model/metadata assets from node 202 into the dedicated
   NAS asset root;
2. read videos directly and read-only from the restored `/mnt/nas2` mount.

## Required confirmations before copying

- free space on that filesystem;
- source ownership and permissions permit these project assets to be copied.

Resolved locally on 2026-08-17:

- source alias: `GPU202-tianlin` (`172.16.11.202:15654`);
- proxy: `bastion-DB13` (`202.120.39.165:58122`, user `jumpserver`);
- destination: `gpu200` (`172.16.10.200`);
- staging root: `/home/tianlin/commentary_assets_202_bridge_20260817`;
- staging directories exist and `/home/tianlin` had about 177 GB free;
- `/raid` had more space but was mounted read-only, so it was rejected.

The first pull preflight used an obsolete `/etc/hosts` route
(`172.16.10.202:22`) and timed out before any remote command. The user then
provided the actual ProxyJump route. The bastion endpoint completed an SSH
banner/key handshake. The original Mac identities were not present on gpu200,
so a dedicated gpu200 Ed25519 identity was created and configured for both SSH
hosts. The dedicated key has since been authorized on both accounts and
batch-mode authentication passed. Exact restricted authorization lines are in
`GPU200_GPU202_BRIDGE_PUBLIC_KEY_20260817.md`.

The Linux-compatible SSH config is installed at `/home/tianlin/.ssh/config`
with mode 600. The macOS-only `UseKeychain` option was intentionally omitted.
The observed bastion ED25519 fingerprint is:

`SHA256:/p460eWah5vDQdtccbtOAanO24ix08XIPlM9MJVz96M`

The dedicated key was authorized on both remote accounts. Batch-mode login to
the bastion and GPU202 passed; GPU202 reported hostname `gpu202` and user
`tianlin`.

The measured minimum bundle is 58,218,399,525 bytes (58.22 decimal GB), or
72.77 GB with 25% headroom. Phase-0 metadata has been copied locally, both JSON
files parse, and recorded hashes match where historical hashes exist.

The host mount at `/mnt/nas` is writable. A create/write/rename/delete probe
passed under `/mnt/nas/tianlin/SoccerMasterAssets`; the probe was removed.
Managed Codex commands may still require an approved host-level command for
NAS writes. The user reports tens of TB free on the physical NAS, so SSHFS
free-space output is not used as a capacity authority.

## Completed transfer and verification

The complete 58.22 GB decimal bundle was transferred on 2026-08-17. The first
archive-style rsync copied content but returned code 23 because SSHFS rejected
group metadata changes. A resumable rerun using
`-rt --partial --no-owner --no-group --no-perms` completed with exit code 0.

- final bundle inventory: 117 files, 58,218,293,029 bytes;
- visual backbone SHA-256: `fc64d2acbabb5c20a3e0bf996906954c81838d8495b7724a9862199c0af4c977`;
- generation checkpoint SHA-256: `e1ff7fef61a480576d52f4c2761ccedca16d8af3ccd6cdc39a83d36fc5a32317`;
- checksum rsync dry runs for Llama, BERT and SigLIP2 reported zero files to
  transfer;
- CPU-only preflight passed and is recorded in
  `reports/commentary_nas_bundle_preflight_20260817/result.json`;
- the epoch-11 checkpoint opened with CPU mmap and contained 953 state keys;
- no forward, generation, training or GPU query was performed by the preflight.
- the frozen manifest was copied into the NAS bundle's `manifests/` directory;
  its SHA-256 is `1d56f2e0b8e7602dc54d7d3266b5cc7db0668e00a806b10fe01a692516370bac`.

Do not use a home-directory variable, `/`, `/remote-home`, or a broad shared
directory as the destination.

## Transfer phases

### Phase 0: metadata first

Copy the two MatchTime annotation JSON files and `match_time.pkl`. Their known
combined size is about 12.9 MB. Use `classification_train.json` to reject
training-overlapping matches before selecting a Locked Match Holdout.

### Phase 1: minimum model bundle

Copy the complete Llama, BERT and SigLIP2 model directories, plus:

- generation checkpoint: 17,615,455,530 bytes;
- visual backbone: 1,435,281,181 bytes.

The three directory sizes were not recorded. Measure them on node 202 and
confirm node 200 has the measured total plus at least 25% working headroom
before starting. The historical CPU checkpoint load peaked near 38 GB RSS;
the first GPU run previously reserved about 20 GB of GPU memory.

### Phase 2: one development video preflight

Read exactly one already-known development clip from the fixed manifest on the
restored NAS. This is only for path, decode, preprocessing and layer-contract
validation. It must not come from the future Locked Match Holdout. Do not copy
the video to local staging unless NAS I/O is later shown to be a bottleneck.

### Phase 3: balanced development subset

After the one-clip capture passes, read only the event-balanced, match-grouped
development subset selected for layer caching from NAS. Do not stage all 3,256
videos by default. The historical 200-clip subset occupied 1,208,700,836 bytes.

### Phase 4: new-match holdout

Select new matches only after comparing candidates with both the relevant
training manifest and the current 49 development matches. Freeze the manifest
before producing model outputs or running judge evaluation. Keep the selected
NAS paths logically separate and read-only during model selection.

## Safe copy pattern

For a future refresh, run a dry run first. The direct pull route is now valid:

```bash
rsync -rt --no-owner --no-group --no-perms --dry-run --itemize-changes GPU202-tianlin:/remote-home/share/huggingface/Meta-Llama-3-8B-Instruct/ /mnt/nas/tianlin/SoccerMasterAssets/commentary_generation_epoch11_20260817/models/Meta-Llama-3-8B-Instruct/
```

After checking the resolved source and destination, use a resumable copy:

```bash
rsync -rt --partial --no-owner --no-group --no-perms --info=progress2 GPU202-tianlin:/remote-home/share/huggingface/Meta-Llama-3-8B-Instruct/ /mnt/nas/tianlin/SoccerMasterAssets/commentary_generation_epoch11_20260817/models/Meta-Llama-3-8B-Instruct/
```

Do not remove `--dry-run` until the resolved destination is confirmed.

Apply the same pattern to each manifest entry. Copy into a new staging
directory. Do not use `--delete`, overwrite an existing verified bundle, or
write back to node 202.

## Verification before use

1. Record `du -sb` for every source directory and destination directory.
2. Record SHA-256 for small files and both large checkpoint files. The old
   audit deferred hashes for the two large files, so the source hash computed
   on node 202 becomes the transfer identity.
3. Compare the already-recorded hashes for Llama config/tokenizer files and
   `match_time.pkl` against `assets.json`.
4. Parse both annotation JSON files locally.
5. Decode only the fixed development clip from NAS on CPU.
6. Update runtime paths explicitly to the verified staging paths.
7. Before model/GPU use, run `nvidia-smi`, report all processes, and obtain
   authorization for the exact command.

## Stop conditions

Stop without inference if a source resolves differently from the manifest, a
size/hash differs after transfer, the destination lacks headroom, the NAS
video becomes unavailable, or the proposed holdout overlaps training/development
matches.
