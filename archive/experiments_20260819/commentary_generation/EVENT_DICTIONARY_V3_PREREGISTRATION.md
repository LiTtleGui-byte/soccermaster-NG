# Event dictionary v3 preregistration

Frozen: 2026-08-16 UTC

Status: preregistered after a text-only audit of the 14 v2 raw `other` and 27
v2 `other_rare` samples. No v3 label counts, folds, features, classifiers, or
metrics have been computed.

## Evidence boundary

The v1 and v2 results were already known, so this is not a blind
preregistration. The changes below are limited to five reference sentences for
which the v2 dictionary has a general morphological or phrase omission. The
rules are phrased generally; sample IDs are not used by the extractor and
manual overrides remain forbidden.

No prefix values, videos, models, checkpoints, or predictions were inspected
for this audit. The decision is based only on reference text and v2 dictionary
hits.

## Audit conclusion

The 27 v2 `other_rare` samples are coherent rare-event groups:

| Raw event | Count | Decision |
| --- | ---: | --- |
| goal | 9 | keep rare and merged |
| offside | 9 | keep rare and merged |
| injury | 5 | keep rare and merged |
| penalty | 2 | keep rare and merged |
| restart | 1 | keep rare and merged |
| throw_in | 1 | keep rare and merged |

The threshold remains raw count `< 10` because five-fold evaluation otherwise
has fewer than two examples per class per fold in expectation. It must not be
lowered merely to make goal or offside visible as separate v3 classes.

Among the 14 v2 raw `other` references:

- five have a general dictionary omission: two substitution constructions,
  one inflected yellow-card construction, one explicit trip/foul construction,
  and one explicit long-ball/pass construction;
- five describe dribbling, challenges, or open-play duels for which no v2 class
  exists;
- three describe generic ball progression/clearance that cannot be safely
  distinguished as pass, cross, or clearance from the text rule alone;
- one says only “set piece” and does not identify whether it is a corner or free
  kick.

Only the first group is eligible for v3 changes. The other nine remain `other`.

## Frozen v3 changes

V3 inherits the complete v2 sentence splitting, earliest-event-sentence
selection, salience order, dictionary, vetoes, ambiguity reporting, rare-class
rule, representations, cross-validation, classifier, baselines, and metrics.
Only the following exact dictionary edits are allowed.

### `substitution`

Append exactly:

```python
[
    r"\b(?:is being|was|will be) substituted\b",
    r"\bwill be replaced by\b",
]
```

### `yellow_card`

Replace v2 pattern:

```python
r"\breceives? a yellow card\b"
```

with exactly:

```python
r"\breceiv(?:e|es|ed|ing) a yellow card\b"
```

No bare `yellow card` pattern may be restored; conditional warnings must remain
unmatched.

### `foul_or_free_kick`

Append exactly:

```python
r"\btrips? an opponent\b"
```

Do not add bare `slide tackle`, because a slide tackle is not necessarily a
foul without the explicit trip construction.

### `pass_or_build_up`

Append exactly:

```python
r"\bsends? a long ball\b"
```

No patterns may be added for generic “ball into the area”, “attempt”, “clear”,
“challenge”, “blocked”, “set piece”, or “goal kick”; those phrases do not
uniquely identify one current primary category.

## Frozen non-changes

- Do not add a dribble/duel/clearance category.
- Do not manually relabel any sample.
- Do not change sentence splitting or category priority.
- Do not lower the rare threshold or exclude samples.
- Do not change seed, fold count, representation, scaling, PCA, classifier,
  baseline, or metric.
- Do not modify or overwrite v1/v2 scripts or results.

## Frozen future run

If separately authorized, v3 must use:

- script: `experiments/commentary_generation/event_separability_200_v3.py`;
- output: `reports/commentary_event_separability_200_20260816_v3/`;
- the identical local fixed-200 prefix cache and manifest/E1/E2 identity checks;
- seed 20260816, five-fold stratified out-of-fold evaluation, the same two
  representations, fold-local preprocessing, logistic probe, shuffled-label
  baseline, majority baseline, macro-F1, balanced accuracy, and confusion;
- one CPU-only formal run with explicit local Python, empty
  `CUDA_VISIBLE_DEVICES`, fixed `PYTHONPATH`/`LD_LIBRARY_PATH`, timeout,
  heartbeat, no overwrite, and no fallback.

The v3 result must be reported even if coverage or scores become worse. Once v3
metrics are visible, this frozen rule set cannot be edited and rerun under the
v3 name.
