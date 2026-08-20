# Reproduction

New baseline reproduction work belongs here and must reference the frozen code
in `baseline/`. The completed G0-G10 harness history is preserved at
`archive/reproduction_20260819/`, with its outputs under
`runs/reports_legacy_20260819/` and `runs/outputs_legacy_20260819/`.

Current migration smoke:

- `smokes/soccermaster_five_heads.py`: one fixed 30-frame SNGS-10004 inference over all five task heads.
- `smokes/run_soccermaster_five_heads.sh`: fixed environment, timeout, heartbeat and isolated output launcher.
