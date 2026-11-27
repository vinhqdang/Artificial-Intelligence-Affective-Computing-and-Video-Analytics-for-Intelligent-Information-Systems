# Extension experiments specification

Inputs: RAVDESS features `/home/user/data/features/` (incl. new `qda_*`, `qdv_*`, cafe5/street5/h264 files),
CREMA-D features `/home/user/data/features_cremad/`. Reuse helpers from `code/run_experiments.py`,
`code/run_qag.py`, `code/run_slg.py`, `code/run_revision.py` (import; do not modify). Seed 2026. Outputs `results/ext_*`.

Common model recipe (as before): StandardScaler + multinomial logistic regression, C from {0.01, 0.1, 1}
by inner 4-fold grouped-by-actor CV (macro-F1), temperature on inner out-of-fold logits, late fusion by
summed log-probabilities, F-all temperature fitted out-of-fold. MCT = training on clean + one degraded copy
(audio: snr10; visual: compress), exactly as in run_revision.py.

## X1. Replication on CREMA-D (within corpus)
- 7 outer folds grouped by actor (13 actors each, stratified by sex as far as possible).
- Clean-trained and MCT models for A0, A1, V0, V1; F-all; F-cheap(A0,V0); LF(A0,A1).
- Test conditions: clean; audio snr10, snr0, cafe5, street5; video compress, lowlight, h264.
- Report accuracy, macro-F1 (cluster-bootstrap CI over actors), mean max-probability, ECE (15 bins).
- Fusion gain: F-all vs best single tier, Wilcoxon over actors.
- A1-first cascade (operating point on training OOF, within 1 pp of OOF F-all): accuracy, escalation, cost
  (cost from CREMA-D timings.csv component means), per condition.
- Fairness: female-minus-male accuracy gap of F-all with bootstrap CI.

## X2. Cross-corpus transfer (6 shared classes: neutral, happy, sad, angry, fearful, disgust)
- Train on all RAVDESS clips of the six classes (clean; and MCT), test on all CREMA-D clips (all conditions);
  and train on all CREMA-D clips, test on RAVDESS (six classes).
- Report accuracy, macro-F1, mean max-probability, ECE for A1, V1, F-all; compare with within-corpus results.
- Treat the corpus change as a natural shift: fit the input monitors on the training corpus and report the
  share of target-corpus clean clips flagged (per-clip P99) and whether ADWIN fires on a stream that switches
  from 240 source-corpus test clips to 240 target-corpus clips (source clips from a held-out fold).

## X3. Monitors: content vs quality descriptors vs speaker-adaptive
Monitor recipe (as run_qag): StandardScaler + PCA (95% variance) + Mahalanobis, P99 threshold from inner
grouped OOF, standardised log q, ADWIN (delta 0.002), recent-15 rule with margin 1.0 for state changes.
Monitors per channel:
- CONTENT: prosody (audio), MobileNet (visual) features [as before].
- QD: quality descriptors (audio `qda`, visual `qdv`).
- SPK: speaker-adaptive content monitor: within each speaker session, content features are z-normalised
  with the mean/SD of the first 10 clips of that session (enrolment), then scored against a reference fitted
  on training actors processed the same way; the detector is reset at every session boundary
  (session boundaries known, as in a call or consultation). Clips during enrolment are not scored.
- QD+SPK: alarm if either fires.
Scenarios (each corpus; 20 seeds per fold):
- S1: 240 clean -> 240 degraded (shuffled clips; all degradation conditions incl. cafe5/street5/h264).
- S4: speaker-blocked clean streams (clips grouped by actor; RAVDESS: as in run_integrated; CREMA-D: 20
  speakers x 24 clips = 480 clips per stream, test-fold actors, random speaker order).
- S5: speaker-blocked degraded: as S4 but a degradation starts at a random clip between 120 and 360
  (possibly mid-session) and persists to the end.
Report per monitor/scenario/condition: detection rate, median/IQR delay, false marks per 1000 clips,
share of streams with any false mark, attribution (single-channel conditions only).

## X4. Gating with the best monitor
For the monitor with the best trade-off (detection vs false marks; choose on RAVDESS, apply to CREMA-D),
evaluate stream-level gating (run_slg state machine) with clean-trained models and envelope-aware gating
with MCT models (envelope monitors fitted on MCT training data, same monitor type) on S1 and S4/S5 for
both corpora: accuracy on degraded and clean segments, gated share on clean speaker-blocked streams.

## Figures
- ext_crosscorpus.pdf: accuracy (clean) within vs cross corpus for A1, V1, F-all, both directions.
- ext_monitors.pdf: detection rate vs false marks per 1000 clips (speaker-blocked) for the four monitor types,
  both corpora.
- ext_realistic.pdf: accuracy under realistic degradations (cafe5, street5, h264) for clean-trained and MCT,
  both corpora.
