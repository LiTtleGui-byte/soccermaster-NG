# Fixed-200 independent video-event annotation protocol

Status: prepared, not executed

This protocol is the required bridge from reference-relative silver labels to
independently observed video labels. It covers the same fixed 200 clips as the
event-separability experiments, but annotators must not see references, model
outputs, prefix features, v1/v2/v3 labels, or each other's work.

## Roles

- Annotator A and Annotator B independently watch all 200 clips in separately
  shuffled orders and lock their CSV files before comparison.
- The adjudicator sees the two locked annotations only for disagreements or
  `indeterminate` decisions, watches those clips independently, and records one
  final decision.
- None of the three roles may use the reference commentary or generated text.

The same person must not fill more than one role. An AI-generated text label is
not a substitute for either human annotator.

## Primary-event rule

Choose the single observable event whose decisive moment is closest to the
temporal center of the clip. When events form an inseparable causal chain at the
same moment, choose the clearest terminal outcome:

- `goal` over the preceding shot, pass, cross, or set piece;
- `yellow_card` or `red_card` over the preceding foul when the card is visibly
  shown in the clip;
- `shot_or_save` over its assisting pass or cross;
- `corner` only when the clip shows the corner being taken; an awarded corner
  after a block remains secondary to the shot/cross/pass that caused it;
- `substitution` over an associated injury only when the substitution process
  is visibly underway;
- use `indeterminate` when the decisive event cannot be observed reliably.

Do not infer an event from filename, crowd reaction alone, scoreboard change
alone, or knowledge of the match.

## Event families

Exactly one `primary_event` must be selected from:

| Event family | Inclusion boundary |
| --- | --- |
| `goal` | The ball visibly crosses the goal line as a valid or apparent scored goal. |
| `shot_or_save` | A shot, miss, defensive block, goalkeeper save, or shot outcome other than a visible goal. |
| `foul_or_free_kick` | A visible foul decision or a free kick being taken. |
| `yellow_card` | The referee visibly shows a yellow card. |
| `red_card` | The referee visibly shows a red card or second-yellow dismissal. |
| `penalty` | A penalty kick is visibly prepared or taken; a foul merely occurring in the penalty area is not enough. |
| `offside` | An offside decision is visibly signaled or play is clearly stopped for offside. |
| `corner` | A corner kick is visibly taken. |
| `cross` | A deliberate lateral or wide delivery into a threatening central area without a later terminal event taking precedence. |
| `pass_or_build_up` | Open-play passing or possession development without a later terminal event taking precedence. |
| `substitution` | A player exchange is visibly underway. |
| `injury` | A player receives attention or is visibly unable to continue, without a visible substitution taking precedence. |
| `throw_in` | A throw-in is visibly prepared or taken. |
| `restart` | A half or match restart is visibly underway. |
| `duel_or_dribble` | A primary one-on-one challenge, tackle, dribble, or dispossession without a foul or later terminal event. |
| `goal_kick` | A goal kick is visibly prepared or taken. |
| `clearance_or_interception` | A defensive clearance or interception is the primary completed action. |
| `other_observable` | A clear football event exists but none of the canonical families applies; explanation is mandatory. |
| `indeterminate` | No reliable primary event can be identified from the clip. |

`secondary_events` is a semicolon-separated list from the same vocabulary,
excluding the selected primary event. It may be empty.

## Observability and confidence

`observability` must be one of:

- `clear`: the decisive event is shown directly;
- `partial`: material evidence is visible, but the decisive instant or signal is
  partly obscured or cut;
- `not_observable`: the proposed event is not actually visible; primary must be
  `indeterminate`.

`confidence` must be `high`, `medium`, or `low`. Confidence describes the
annotator's decision, not model probability. `low` confidence does not by itself
force `indeterminate`, but notes are mandatory.

## Locking, agreement, and adjudication

1. Each annotator completes every row and signs the declaration at the top of
   their CSV by setting `annotation_complete=yes` in all rows.
2. The coordinator validates vocabulary, required fields, unique IDs, and 200
   completed rows before opening either file for comparison.
3. Agreement requires identical primary events and neither row being
   `indeterminate`.
4. Every disagreement and every `indeterminate` row goes to adjudication.
5. The adjudicator receives only the clip and the two proposed event families,
   not references or model results.
6. The final dataset records both original decisions, the adjudicated label,
   and whether agreement was direct or adjudicated. Original decisions are
   never overwritten.

Report raw agreement, Cohen's kappa, per-family counts, the disagreement matrix,
and adjudication rate before running the separability probe. Do not tune the
ontology after seeing probe metrics.

## Safety and current blocker

The local packet builder only copies video path strings from existing local
JSON; it does not open or stat any video. Actual annotation requires explicit
authorization to access the fixed 200 NAS video paths and two independent human
annotators plus a separate adjudicator. Those conditions are not satisfied by
preparing this protocol.
