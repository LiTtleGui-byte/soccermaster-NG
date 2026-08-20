# Soccer Commentary Event Evidence

本文件维护解说研究使用的统一术语；它不是项目目录或运行状态说明。

This context defines the language used to distinguish reference-derived event labels from independently observed video events in SoccerMaster commentary research.

## Project and dataset names

**SoccerMaster**:
A soccer-specific vision foundation model used here as the visual backbone for downstream commentary generation.
_Avoid_: Commentary dataset, commentary generator

**SoccerFactory**:
The pretraining data resource assembled for SoccerMaster from automatically generated spatial annotations and existing soccer video datasets.
_Avoid_: Commentary source, MatchTime

**MatchTime**:
A training dataset of soccer video and commentary pairs produced by correcting and filtering temporal pairings inherited from SoccerNet-Caption.
_Avoid_: SoccerFactory, clip-comprehensive commentary dataset

**SN-Caption-test-align**:
The manually timestamp-aligned commentary benchmark built from 49 matches and used as the test source in MatchTime.
_Avoid_: MatchTime training split, SoccerFactory

**MatchVoice**:
The automatic commentary generator trained from the MatchTime data and adapted here to use SoccerMaster visual features.
_Avoid_: SoccerMaster, MatchTime

## Evidence sources

**Video Clip**:
A bounded sequence of match footage presented as the sole evidence for independent event annotation.
_Avoid_: Sample, reference

**Reference Commentary**:
The existing natural-language description paired with a video clip; it may mention multiple events and is not treated as video truth.
_Avoid_: Ground truth caption

**Event-Anchored Commentary**:
A commentary sentence paired around a particular commentary timestamp and primarily intended to describe the associated incident, without a requirement to cover every meaningful event visible in the surrounding clip.
_Avoid_: Clip-comprehensive commentary

**Clip-Comprehensive Commentary**:
A concise account of all independently meaningful, non-replay events directly supported by the full Video Clip.
_Avoid_: Longer reference, event-anchored commentary

**Reference-Relative Silver Label**:
An event label derived automatically from Reference Commentary rather than independently observed from the Video Clip.
_Avoid_: Ground truth, true label

**Adjudicated Video Label**:
The final label produced from two independent video annotations and, when required, a separate adjudication decision.
_Avoid_: Absolute ground truth, silver label

## Event language

**Observable Event**:
A football action or officiating outcome that can be identified from the Video Clip without using Reference Commentary or model output.
_Avoid_: Mention, cue

**Primary Event**:
The single Observable Event whose decisive moment is closest to the clip's temporal center; for an inseparable causal chain, it is the clearest terminal outcome.
_Avoid_: First mentioned event, reference event

**Secondary Event**:
Another Observable Event in the same clip that is not selected as the Primary Event.
_Avoid_: Alternative primary label

**Event Family**:
A canonical football category used consistently across annotators, such as substitution, goal, offside, or shot/save.
_Avoid_: Keyword, regex class

**Indeterminate Event**:
A clip for which the Primary Event cannot be identified reliably because the decisive action is absent, obscured, ambiguous, or outside the clip.
_Avoid_: Other, negative sample

## Annotation language

**Independent Annotation**:
An annotation created from the Video Clip without access to Reference Commentary, model output, prefix features, another annotator's decision, or silver labels.
_Avoid_: Review, correction

**Reference-Support Review**:
An assessment made with access to both the Video Clip and Reference Commentary of whether the clip supports the reference's claims.
_Avoid_: Independent annotation, video ground truth

**Candidate-Grounding Review**:
An assessment made with access to a Video Clip and one generated commentary candidate, but without access to Reference Commentary or the candidate's model identity, of which candidate claims the clip supports.
_Avoid_: Reference match, independent annotation

**Agreement**:
Two Independent Annotations selecting the same Primary Event family while both judge the event observable.
_Avoid_: Similar labels

**Adjudication**:
A third, blinded decision made only after the two Independent Annotations are locked and disagree or include an Indeterminate Event.
_Avoid_: Majority vote, label repair

## Alignment language

**Temporal Pairing**:
The association between a bounded Video Clip and the Reference Commentary that is intended to describe the same match moment.
_Avoid_: Alignment

**Representation Alignment**:
The learned association between video and text representations, evaluated independently of natural-language generation.
_Avoid_: Commentary alignment, factual grounding

**Video-Grounded Commentary**:
Generated commentary whose claimed primary event, action, result, and role relationships are directly supported by the Video Clip.
_Avoid_: Reference match, fluent commentary

**Video Fact Record**:
A structured, video-only account of the observable primary event, action, result, and role relationships, with unobservable fields left explicitly unknown.
_Avoid_: Reference summary, generated caption

**Unsupported Claim**:
A claim made by generated commentary for which the Video Clip provides insufficient evidence; omitting an unobservable detail is not an Unsupported Claim.
_Avoid_: Reference mismatch, missing detail

**Anonymous Entity Placeholder**:
A task-level token such as `[PLAYER] ([TEAM])`, `[TEAM]`, or `[REFEREE]` used in final commentary when an entity's real identity is unavailable or intentionally masked.
_Avoid_: Shirt-colour identity, guessed name

**Core Event Error**:
A generated claim that identifies the primary event incorrectly or asserts a different primary event from the one supported by the Video Clip.
_Avoid_: Wording difference, incomplete detail

## Diagnostic language

**Oracle Intervention**:
A diagnostic replacement of one module's output with the best available independently justified information while holding the remaining chain fixed.
_Avoid_: Ablation, module improvement

**Causal Bottleneck Ranking**:
An uncertainty-aware ordering of modules by the change in Video-Grounded Commentary quality caused by their Oracle Interventions, including material interactions between modules.
_Avoid_: Worst-module guess, metric ranking

**Interface-Matched Intervention**:
An Oracle Intervention whose replacement obeys the original module boundary and is given the same supervision and optimization budget as replacements being compared with it.
_Avoid_: Unmatched upper bound, prompt shortcut

**Locked Match Holdout**:
A set of clips from matches absent from all development, selection, and relevant model-training data, whose results remain hidden until one final evaluation.
_Avoid_: Remaining test clips, clip-only holdout
