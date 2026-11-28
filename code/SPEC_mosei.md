# Cycle 6 specification: CMU-MOSEI (spontaneous, in-the-wild speech)

## Data
- Segmented clips: MMSA mirror of CMU-MOSEI raw segments, `Raw.zip` at
  https://huggingface.co/datasets/tamb2203579/CMU-MOSEI/resolve/main/Raw.zip (27 GB, deflate64 entries
  `Raw/<video_id>/<clip_id>.mp4`). Do NOT download the whole archive (disk). Extract only the selected
  members with `remotezip` + `zipfile-deflate64` (both installed), several threads, retries with backoff,
  into `/home/user/data/mosei/clips/<video_id>/<clip_id>.mp4`. Verify each file with ffprobe (has audio and
  video streams, duration within 0.5 s of `dur`); log and exclude failures.
- Labels and subset: `code/mosei_select.py` -> `/home/user/data/mosei/subset.csv` (2346 clips, 1259 videos;
  classes neutral 600, happy 600, sad 501, angry 645; columns incl. video_id, clip_id, split, dur, emotion
  intensities, cls, emotion (RAVDESS codes 1/3/4/5), clip_key).
- Speaker identity is not distributed with the labels; the video is the grouping unit (most videos have one
  speaker). Sex is not available.

## Extraction (extend `code/extract_extension.py` with `--part mosei`; do not change RAVDESS/CREMA-D paths)
- Same decoding, every-6th-frame sampling, YuNet face detection with the same fallback, extractors A0, A1,
  V0, V1, CLIP zero-shot not needed, quality descriptors (audio and video), seeds from clip id (use clip_key
  with '/' replaced by '_' as clip id).
- To bound cost on long or high-resolution clips: at most 24 sampled frames per clip (uniformly subsampled from
  the every-6th-frame set if more), and frames larger than 1280x720 are downscaled to fit 1280x720 before
  everything else (record original resolution).
- Audio conditions: clean, snr10, snr0, cafe5, street5. Video conditions: clean, compress, lowlight, h264
  (exact functions as for CREMA-D).
- Output `/home/user/data/features_mosei/` with the same file naming as CREMA-D; `meta.csv` with clip_id,
  video_id, clip_num, split, emotion, cls, dur, duration_s, n_frames, width, height, face_rate_<vcond>;
  `timings.csv` for clean; `excluded.csv`. Resumable per block of 50 clips; 4 workers x 1 torch thread
  (nothing else runs) unless memory is short.
- Verify: no NaN/Inf, row counts, face-detection rates (expect < 1 for in-the-wild video; report distribution),
  QD shifts in the expected direction, timing summary.

## Experiments (extend `code/run_extension_exp.py` with corpus `mosei`)
- Within-corpus: 5 outer folds grouped by video, stratified by class (StratifiedGroupKFold, seed 2026); inner
  4-fold grouped CV as before. Models A0, A1, V0, V1, F-all, F-cheap(A0,V0), LF(A0,A1); clean-trained and MCT.
  Conditions as listed. Metrics: accuracy, macro-F1 (cluster bootstrap over videos), mean confidence, ECE.
  Fusion gain vs best single (Wilcoxon over folds is too small: use paired cluster bootstrap over videos for
  the accuracy difference, 2000 resamples, report CI). A1-first cascade with costs from MOSEI timings.
- Cross-corpus (4 shared classes neutral, happy, sad, angry): train on RAVDESS (4 classes) and on CREMA-D
  (4 classes), test on MOSEI; and train on MOSEI, test on each. Same metrics; within-corpus 4-class baselines.
  Monitor shift: monitors (CONTENT, QD) fitted on the source corpus, share of clean MOSEI clips flagged,
  ADWIN on 240 source -> 240 MOSEI clips.
- Monitors on MOSEI (CONTENT, QD; SPK only if sessions allow: sessions = videos; videos with fewer than 12
  clips cannot be enrolled, so report SPK as not applicable): S1 (240 clean -> 240 degraded, random clips of
  the test fold), S4 (480 clean clips grouped by video, random video order: the stream changes recording
  every 1-2 clips; draw videos without replacement from the test fold, then repeat the fold's clips in a
  second random video order if fewer than 480 clips), S5 (S4 with degradation from a random clip in 120-360).
  20 seeds per fold. Also report the per-clip P99 flag rate on clean test clips (expected about 1% if the
  in-the-wild heterogeneity is captured by the training videos).
- Gating with QD monitors as in X4 (clean-trained SLG; MCT + envelope SLG) on S1 and S4.
- Outputs: `results/ext_mosei_*.csv`, figures `manuscript/figures/ext_mosei.pdf` (accuracy by condition
  CT vs MCT; monitor detection vs false marks) and update `ext_crosscorpus.pdf` only if it remains readable
  (otherwise `ext_crosscorpus_mosei.pdf`).
