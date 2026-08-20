# Event dictionary v2 preregistration

Frozen: 2026-08-16 UTC

Status: preregistered from a text-only review; v2 labels, features, cross-validation,
and metrics have not been computed.

## Scope and evidence boundary

This document freezes a narrower primary-event extraction protocol before any
v2 separability result is produced. The review used the 54 references that v1
marked ambiguous, their v1 regex hits, and the already known v1 report. It did
not inspect per-sample prefix values or fit a v2 classifier. This is therefore
not a blind preregistration: v1 metrics were already known. The v2 changes below
are justified only by identifiable text-rule false positives, not by whether a
change might improve a downstream score.

Labels remain reference-relative silver labels. No video is opened, no event is
manually assigned, and no per-sample override is allowed.

## Review decision

A v2 rule set is necessary. Four v1 rules are too broad:

1. `\bscores?\b` treats infinitive phrases such as “chance to score” as a goal.
2. `\bgoal\s*[!-]` treats “shot towards goal - this time ...” as a goal event.
3. bare `corner` treats spatial phrases such as “bottom right corner” as a
   corner-kick event.
4. bare `yellow card` treats a conditional warning (“will get a yellow card if
   ...”) as an awarded card.

The review also showed that a reference often describes an initiating event in
one sentence and a secondary consequence in the next, especially a blocked
cross/pass/shot followed by a corner. A whole-reference global priority can
therefore replace the narrated primary event with its later consequence.

## Frozen v2 extraction algorithm

1. Lowercase the reference without otherwise rewriting it.
2. Split it with `re.split(r"(?<=[.!?])\s+", reference.strip())` and discard
   empty pieces.
3. Run the event dictionary independently on each sentence, in original order.
4. Select the earliest sentence with at least one non-vetoed category hit.
5. Within that sentence, select the first hit category in the unchanged fixed
   salience order below. Do not use textual hit position to break the tie.
6. If no sentence has a hit, assign raw primary `other`.
7. Preserve two ambiguity fields:
   - `primary_sentence_ambiguous`: more than one category hits the selected
     sentence;
   - `whole_reference_multi_event`: more than one category hits anywhere in the
     full reference.
8. Merge every raw primary category with fewer than 10 samples into
   `other_rare`. Retain all 200 samples; exclusions and manual overrides remain
   forbidden.

The frozen salience order remains:

```text
goal
penalty
yellow_card
substitution
offside
foul_or_free_kick
shot_or_save
corner
cross
throw_in
injury
pass_or_build_up
restart
```

## Exact dictionary changes

All categories not listed here inherit their v1 patterns and veto behavior
verbatim from `event_separability_200.py` as it existed when this document was
frozen.

### `goal`

Replace the v1 goal patterns with exactly:

```python
[
    r"\bgoo+a+l\s*!",
    r"^goal\s*[!-]",
    r"\bscores\b",
    r"\bown goal\b",
    r"\bscore is now\b",
    r"\bscore is \d+\s*:\s*\d+\b",
    r"\bmakes? it \d+\s*:\s*\d+\b",
    r"\binside the (?:left|right|middle) post\b",
]
```

Apply the existing goal vetoes to the current sentence:

```python
[r"\bdisallowed goal\b", r"\bgoal, no wait\b"]
```

This retains explicit goal announcements, scored-goal verbs, own goals, score
updates, and the existing precise “inside the ... post” construction. It no
longer treats “score” as a goal without third-person `scores`, or a mid-sentence
“goal -” construction as an announcement.

### `corner`

Replace bare `r"\bcorner(?: kick)?\b"` with exactly:

```python
[
    r"\bcorner kick\b",
    r"\b(?:take|takes|taking|deliver|delivers|will take|will deliver|win|wins|won|get|gets|got|have|has|had|earn|earns|earned|force|forces|forced) (?:a |the )?corner\b",
    r"\bawarded a corner\b",
    r"\b(?:the|this|resulting|another) corner\b",
    r"\bcorner (?:by|from|flag)\b",
    r"\bfrom (?:a |the )?corner\b",
    r"\bfor a corner\b",
]
```

These patterns require a set-piece construction or corner-award context and do
not match “bottom right corner” by itself.

### `yellow_card`

Replace the v1 yellow-card patterns with exactly:

```python
[
    r"\byellow-coloured card\b",
    r"\breceives? a yellow card\b",
    r"\bgets? booked\b",
    r"\bis booked\b",
    r"\bis cautioned\b",
    r"\b(?:is|was) shown a yellow card\b",
    r"\bshows? (?:him |her )?a yellow card\b",
    r"\bthat will be a yellow card\b",
]
```

The bare phrase is removed. A card must be described as received, booked,
cautioned, shown, or directly awarded by “that will be”. Conditional future
warnings are not card events.

## Known text-only consequences, not v2 results

The following v1 false-positive families motivated the general rules and must
not be implemented as item-specific exceptions:

- “chance/opportunity to score” references currently promoted to `goal`;
- “shot towards goal - ...” currently promoted to `goal`;
- “bottom right corner” currently adding a false `corner` hit;
- “will get a yellow card if ...” currently adding a false card hit;
- later-sentence corner awards currently overriding an earlier narrated
  cross/pass/shot under global priority.

No predicted v2 class counts, confusion matrix, or metric is frozen here. Those
must be produced by the implementation without hand adjustment.

## Frozen future evaluation protocol

If the user authorizes a v2 run, it must use:

- script: `experiments/commentary_generation/event_separability_200_v2.py`;
- output: `reports/commentary_event_separability_200_20260816_v2/`;
- the identical fixed 200 prefix cache and exact manifest/E1/E2 identity checks;
- seed 20260816 and the identical five `StratifiedKFold` splits generated from
  the v2 final labels;
- the same mean-pooled and query-slot-preserving transforms, with every scaler,
  PCA, and classifier fit only on its training fold;
- the same logistic probe, fixed-seed shuffled-training-label baseline,
  training-fold majority baseline, macro-F1, balanced accuracy, and confusion
  matrix;
- exactly one formal CPU-only run with the existing explicit environment,
  timeout, heartbeat, no overwrite, and no fallback policy.

The v2 result must be reported even if its scores are lower. After metrics are
visible, this dictionary cannot be edited and rerun under the v2 name.
