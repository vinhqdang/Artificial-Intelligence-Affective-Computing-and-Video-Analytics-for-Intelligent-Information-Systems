# Extension specification: second corpus, realistic degradations, quality-descriptor monitors

## E1. Quality descriptors (QD)
Speaker-invariant descriptors of signal quality, intended for input monitoring. They describe the
recording channel, not the person or the emotion.

Audio (16 kHz mono waveform; frames of 25 ms, hop 10 ms):
- frame log-energy (dB) percentiles p5, p10, p50, p90, p95; dynamic range p95 - p10 (an SNR proxy);
- spectral flatness: mean, p10, p90;
- noise-floor spectrum: mean log power in 8 mel bands over the 10% lowest-energy frames;
- high-frequency ratio: energy above 4 kHz / total (mean over frames);
- spectral entropy: mean;
- clipping fraction: share of samples with |x| > 0.99 of max representable.
Video (each sampled frame, full frame AND the 224x224 face crop, grey level):
- mean luminance, luminance SD, fraction of dark pixels (< 30), fraction of saturated pixels (> 245);
- log Laplacian variance, mean Tenengrad (Sobel magnitude squared);
- blockiness: mean absolute difference across 8-pixel block boundaries divided by the mean absolute
  difference inside blocks (horizontal and vertical averaged);
- noise estimate: SD of (frame - median3(frame));
- mean HSV saturation.
  Each frame-level descriptor is summarised by mean and SD over the clip's sampled frames.

## E2. Realistic degradations (new conditions)
- `cafe5`: DEMAND PCAFETER (channel 1, 16 kHz) noise, random segment (seeded by clip id), added at 5 dB SNR.
- `street5`: DEMAND STRAFFIC (channel 1) noise, same procedure, 5 dB SNR.
- `h264`: the original video re-encoded with ffmpeg: `-vf scale=320:180 -c:v libx264 -b:v 100k -maxrate 100k -bufsize 200k -preset veryfast`,
  decoded, frames upscaled to the original size before face detection.
DEMAND files: `/home/user/data/demand/PCAFETER/ch01.wav`, `/home/user/data/demand/STRAFFIC/ch01.wav`
(check actual paths after unzip).

## E3. RAVDESS additions
- Existing conditions: compute QD for audio conditions clean, snr20, snr10, snr0 and video conditions
  clean, compress, lowlight, blur, reusing the exact degradation functions and seeds of
  `extract_features.py` (import them).
- New conditions cafe5, street5 (audio: A0, A1, audio QD) and h264 (video: face detection, V0, V1, video QD).
- Output into `/home/user/data/features/` with the same row order as `meta.csv`:
  `qda_<acond>.npy`, `qdv_<vcond>.npy`, `prosody_<new>.npy`, `w2v_<new>.npy`, `mnet_h264.npy`, `clip_h264.npy`,
  `clipframes_h264.pkl`, `face_rate_h264` added to a separate `meta_ext.csv`.

## E4. CREMA-D subset
- Files listed in `/home/user/data/cremad/subset_files.txt` (2178 clips, 91 actors, sentences DFA, ITH, TAI, IWW,
  six emotions ANG, DIS, FEA, HAP, NEU, SAD), videos in `/home/user/data/cremad/flv/*.flv`.
  Actor sex from `/home/user/data/cremad/meta/VideoDemographics.csv`.
- Same decoding, frame sampling (every 6th frame), face detection, extractors (A0, A1, V0, V1) and QD.
- Audio conditions: clean, snr10, snr0, cafe5, street5. Video conditions: clean, compress, lowlight, h264.
- Output `/home/user/data/features_cremad/` with `meta.csv` (clip_id, actor, sex, emotion, sentence, level,
  n_frames, duration_s, face_rate_<vcond>) and the same file naming as RAVDESS, plus `timings.csv` for clean.
- Emotion mapping to RAVDESS codes: NEU=1, HAP=3, SAD=4, ANG=5, FEA=6, DIS=7.

## General
- Resumable per actor (RAVDESS) / per block of 200 clips (CREMA-D). Parallel workers allowed (2 workers x 2 threads).
- Deterministic seeds (md5 of clip id), as in `extract_features.py`.
- Verify: no NaN/Inf, row counts, face rates, QD distributions plausibly shift with degradation
  (e.g., audio dynamic range falls with noise; blockiness rises with compress/h264; luminance falls with lowlight).
