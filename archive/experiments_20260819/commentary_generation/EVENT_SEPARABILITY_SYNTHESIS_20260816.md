# Fixed-200 event separability synthesis

Date: 2026-08-16 UTC

## Question answered

The fixed 200 post-Q-Former, post-projector visual prefixes contain enough
reference-event-associated structure for a leakage-safe linear probe to beat
both majority and fixed-seed shuffled-label baselines. The current evidence does
not show that preserving query-slot identity is consistently better than simple
mean pooling.

All conclusions are reference-relative. They do not establish the event visible
in the video, generation faithfulness, causal encoding, or full-test behavior.

## Protocol lineage

| Version | Main label change | Final classes | Mean-pooled macro-F1 | Slot-preserving macro-F1 |
| --- | --- | ---: | ---: | ---: |
| v1 | whole-reference global priority, broad regex | 9 | 0.4326 | 0.4675 |
| v2 | earliest event sentence and narrower goal/corner/card rules | 9 | 0.4798 | 0.4963 |
| v3 | five preregistered phrase/morphology repairs | 8 | 0.6169 | 0.5657 |

These rows are not a model-improvement series. Labels, class membership, and
stratified folds changed between versions. V3 changed only six raw primary
labels from v2, but this reduced raw `other` from 14 to 9. The frozen `< 10`
rule then merged the entire `other` class into `other_rare`, causing 15 final
label changes and reducing the task from nine to eight classes. This threshold
discontinuity can explain a substantial part of the v3 score jump.

As a descriptive sensitivity check only, collapsing v2 `other` predictions and
targets into `other_rare` gives:

| V2 representation on collapsed 8-label vocabulary | Macro-F1 | Balanced accuracy |
| --- | ---: | ---: |
| Mean pooled | 0.5118 | 0.4987 |
| Query-slot preserving | 0.5427 | 0.5489 |

Those probes were still trained with v2 labels and folds, so this is not a
replacement v2 result or a controlled v2-v3 comparison.

## Final v3 label audit

- Coverage: 191/200 (95.5%).
- Selected-primary-sentence ambiguity: 27/200 (13.5%).
- Whole-reference multi-event: 50/200 (25.0%).
- Final counts: `other_rare=35`, `corner=34`, `pass_or_build_up=34`,
  `foul_or_free_kick=29`, `shot_or_save=28`, `substitution=19`,
  `yellow_card=11`, `cross=10`.
- All 200 samples are retained; there are no manual overrides.

The 27 v2 rare samples were coherent goal, offside, injury, penalty, restart,
and throw-in references. They were merged because of the preregistered sample
threshold, not because their texts were unrecognizable. V3 also leaves nine raw
`other` references: five dribble/challenge/open-play duels, three generic
progression/clearance descriptions, and one unspecified set piece. Assigning
them to existing categories would require debatable manual semantics, so no v4
dictionary iteration is justified from this sample.

## Final v3 probe results

| Representation | Macro-F1 | Balanced accuracy | Shuffled-label macro-F1 | Majority macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| Mean pooled | 0.6169 | 0.6072 | 0.1321 | 0.0562 |
| Query-slot preserving | 0.5657 | 0.5643 | 0.0847 | 0.0562 |

Both representations remain far above their controls. The evidence therefore
supports event-associated separability in the cached prefix space.

V3 paired correctness over the same 200 out-of-fold positions is:

- both correct: 100;
- mean pooled only: 25;
- query-slot preserving only: 15;
- both wrong: 60;
- prediction agreement: 70.5%.

An exact two-sided binomial test over the 40 discordant correctness outcomes is
`p=0.15386`. It does not establish a reliable accuracy difference. The pooled
advantage is concentrated in `other_rare` (9 pooled-only versus 2 slot-only),
with smaller advantages in pass/build-up and shot/save. Query slots do not add
a consistent event-separability benefit under the current sample size and label
protocol.

## Class behavior

The most consistently separable semantic class is substitution. In v3 its
recall is 0.947 for pooled and 1.000 for slot-preserving. Corner and
foul/free-kick are also comparatively strong. Yellow-card recall is 0.636 for
both.

The unresolved boundary is open-play semantics:

- cross recall is only 0.200 pooled and 0.100 slot-preserving;
- pass/build-up recall is 0.471 and 0.412;
- shot/save recall is 0.571 and 0.500;
- the dominant errors exchange `other_rare`, pass/build-up, cross, shot/save,
  and corner.

This can reflect overlapping reference narratives, genuine visual similarity,
or both. Silver labels alone cannot separate those explanations.

## Final interpretation

Confirmed:

- The fixed projected prefixes are not event-agnostic: leakage-safe linear
  probes consistently beat both controls.
- Much of the easiest structure concerns discrete stoppage/set-piece events,
  especially substitutions, corners, fouls, and cards.
- Event-taxonomy details materially affect reported macro metrics.

Not established:

- Query-slot preservation is superior to mean pooling.
- The probe predicts video-ground-truth events rather than reference-writing
  conventions.
- The result generalizes to all 3,256 references, another seed, the raw
  768-dimensional Q-Former state, or another checkpoint.

The dictionary iteration stops at v3. Further progress requires labels that are
independent of these reference-text heuristics, not more post-result regex
tuning.
