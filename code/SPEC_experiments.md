# Experiment specification

Inputs: `/home/user/data/features/` (see SPEC_extraction.md). Outputs: `results/` (JSON/CSV) and
`manuscript/figures/` (PDF figures). Everything seeded (seed 2026) and re-runnable with one command:
`python code/run_experiments.py`.

## Protocol
- 8 classes (RAVDESS emotions), 1440 clips, 24 actors.
- Subject-independent 6-fold CV: fold k holds out actors {4k+1..4k+4} (2 male, 2 female). All numbers
  are pooled over the 6 test folds (each clip predicted exactly once) unless stated.
- Classifier per feature set: StandardScaler + multinomial LogisticRegression (L2, lbfgs, max_iter 3000).
  C selected from {0.01, 0.1, 1.0} by inner grouped (by actor) 4-fold CV on the training actors (macro-F1).
- wav2vec2 features: mean over the 13 hidden-state layers ("layer average"). Separately, a descriptive
  layer-wise probe: accuracy of each single layer (outer CV, C=0.1), for a supplementary figure.
- Temperature scaling per model: obtain out-of-fold (inner grouped CV) training probabilities, fit one
  temperature T>0 minimising NLL; apply to test logits. Use calibrated probabilities everywhere below.

## Models ("tiers")
- A0 prosody, A1 w2v, V0 mnet, V1 clip (trained, clean data).
- V1-ZS: CLIP zero-shot. Text prompts, 8 emotions with adjectives {neutral, calm, happy, sad, angry,
  fearful, disgusted, surprised}, templates ["a photo of a {} face.", "a photo of a person looking {}.",
  "a close-up photo of a {} person."]; averaged normalised text embedding per class;
  clip score = mean over frames of cosine similarity * 100; softmax. No training.
- Late fusion (LF) of a set of modalities: sum of log calibrated probabilities (equal weights), softmax.
  - F-cheap = LF(A0, V0); F-full = LF(A1, V1); F-all = LF(A0, A1, V0, V1).
  - EF-full: early fusion = concatenated A1+V1 features, single LR (comparison).
- Confidence-weighted fusion (CWF): weight of modality m for a clip = 1 - H(p_m)/log(8) (normalised
  entropy), weights renormalised to sum to number of modalities; fused = softmax(sum_m w_m log p_m).

## Cost model
- From timings.csv (clean): mean per-clip CPU seconds for audio decode, video decode+face detection,
  A0, A1, V0, V1. Report a table with mean and std, and real-time factor (seconds of compute per
  second of media, using duration_s).
- Cost(F-cheap) = audio decode + video decode/face + A0 + V0. Cost of escalation adds A1 + V1.
  Cost(F-all) = all components.

## Cascade (the proposed artifact)
- Stage 1: F-cheap. Confidence c = max calibrated probability. If c >= tau, emit Stage-1 label;
  else run A1, V1 and emit F-all label.
- Variant B: Stage 1 = A0 only (audio-first; cost = audio decode + A0); escalation runs everything.
- Sweep tau over the full range (escalation rate 0..100% in steps of 2%) -> accuracy, macro-F1, mean cost.
  Baseline: random escalation at the same rate (expected value = mix of stage-1 and F-all correctness; compute
  by averaging 50 random draws).
- Operating point: choose tau on the TRAINING out-of-fold predictions of each outer fold as the smallest
  escalation rate whose OOF accuracy is within 1.0 percentage point of OOF F-all accuracy; apply to test.
  Report test accuracy, macro-F1, escalation rate, mean cost, cost saving vs F-all.

## Statistics
- Per-actor accuracy (24 values) for main systems; paired Wilcoxon signed-rank tests:
  F-all vs best single tier; cascade(op) vs F-all; F-all vs F-full; CWF vs LF under each degradation.
- 95% cluster bootstrap CI (resample actors, 2000 reps) for accuracy and macro-F1 of each system.

## Robustness to streaming degradations (train clean, test degraded)
- Conditions: audio snr20, snr10, snr0 (video clean); video compress, lowlight, blur (audio clean);
  combined snr10+compress; combined snr0+lowlight.
- For each: accuracy/macro-F1 of A0, A1, V0, V1, F-cheap, F-all (LF), F-all (CWF), cascade at the
  operating tau (escalation rate reported; it should adapt automatically), and mean stage-1 confidence.

## Label-free drift detection (streaming simulation)
- For each outer fold and each degradation condition, and for 20 seeds: stream = 240 test clips clean
  (random order) followed by the same 240 clips under the degraded condition (new random order).
- Monitored signal: stage-1 confidence c_t (no labels). Detectors: Page-Hinkley on (1-c_t)
  (own implementation, delta=0.005, lambda tuned once = 5.0, alpha=1) and ADWIN (river, delta=0.002) on c_t.
  Also a "clean-only" control stream (480 clean clips) to estimate false alarms.
- Report per condition: detection rate, median and IQR detection delay (clips after change),
  false alarms per 1000 clean clips.
- Accuracy shift accompanying the drift (label-based, for reference).

## Fairness and calibration
- By sex (actor parity): accuracy, macro-F1, ECE (15 bins) for each system; gap = |male - female|.
- Per-emotion recall for F-all and cascade.
- Cluster-bootstrap CI for the sex gap of F-all.

## Explainability
- Modality attribution for F-all: for clip i with predicted class y, contribution of modality m =
  log p_m(y) - mean_k log p_m(k). Share_m = max(contribution,0)/sum. Report mean audio share
  (A0+A1) vs visual share (V0+V1) per true emotion class (figure).
- Top 10 prosody features by mean |standardised coefficient| over folds (table, descriptive).

## Figures (PDF, matplotlib, serif font, width suitable for a single column ~ 3.5in or double 7in)
1. cost_accuracy.pdf: accuracy vs mean CPU cost per clip, cascade sweep (two variants), random
   escalation baseline, single tiers and fusions as points, operating point marked.
2. robustness.pdf: grouped bars of accuracy per condition for A1, V1, F-all LF, F-all CWF, cascade.
3. drift_stream.pdf: one example stream (rolling mean of confidence, change point, PH and ADWIN alarms).
4. confusion.pdf: row-normalised confusion matrix of F-all.
5. attribution.pdf: audio vs visual share per emotion.
6. layer_probe.pdf: wav2vec2 layer-wise accuracy.

## Outputs
- results/summary.json with all numbers, plus CSV tables: main_results.csv, cost.csv, cascade_sweep.csv,
  robustness.csv, drift.csv, fairness.csv, per_actor.csv, per_class.csv, stats.csv, prosody_top.csv.
