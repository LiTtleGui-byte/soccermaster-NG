# Frozen baseline

This directory preserves the unmodified reference side of the project.

- `code/soccermaster/` was copied from the read-only upstream Soccer-Backbone.
- `code/soccerfactory/` contains the vendored TrackLab, sn-gamestate, and Refiner snapshot.
- `configs/` preserves the upstream configuration set.
- `checkpoints` is a lightweight link to `assets/checkpoints/official/`; model files are not duplicated.

Do not develop new methods here. All editable work belongs in `research/`.
