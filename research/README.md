# Editable research workspace

`research/src/soccermaster/` is the code to modify. The existing implementation
is currently grouped under `models/` and `data/`; new backbones and tasks should
enter through `backbones/` and `tasks/`, then be connected by `pipelines/`.

```text
src/soccermaster/       editable package
configs/                current baseline and asset registry only
experiments/            one directory per research question
tools/                  reusable non-model utilities
tests/                  targeted checks for current changes
```

Historical experiments and Gate harnesses were moved to `archive/`; they are
evidence, not templates for new work.
