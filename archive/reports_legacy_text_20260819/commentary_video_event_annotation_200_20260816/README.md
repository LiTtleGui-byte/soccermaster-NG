# Fixed-200 video-event annotation packet

Status: prepared, not annotated.

`annotator_a.csv` and `annotator_b.csv` contain the same 200 fixed video path
strings in different fixed random orders. They intentionally exclude reference
commentary, model output, silver labels, and prefix features. Follow the linked
protocol before comparing the two completed files.

Preparing this packet did not open, stat, decode, or otherwise access any video.
Actual annotation remains blocked on explicit NAS/video authorization and the
availability of two independent human annotators plus a separate adjudicator.

For the user-approved single-review workflow, follow `USER_REVIEW_GUIDE.md` and
fill only `annotator_a.csv`. That result is a single independent video review,
not a two-annotator consensus label set.
