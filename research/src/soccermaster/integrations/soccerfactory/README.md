# SoccerFactory adapters

This package contains project-owned interface code. It does not replace the
algorithms vendored under `vendor/soccerfactory/`.

Current boundary:

```text
SoccerFactory Step-3 detection/image tables
  -> step3_to_training.convert_step3_to_training_frames
  -> SoccerMaster per-frame training PKL schema
```

Keep frozen model implementations in `baseline/code/soccerfactory/`,
experiment-specific logic in `research/experiments/`, and historical Gate
launchers in `archive/reproduction_20260819/`.
New reusable format conversion belongs here.
