# Feature extraction specification

Goal: extract audio and visual features from the RAVDESS audio-visual speech corpus
under clean and degraded streaming conditions, plus per-extractor CPU cost.

## Data
- Archives: `/home/user/data/raw/Video_Speech_Actor_XX.zip` (XX = 01..24).
- Use only files whose name starts with `01-` (full audio-visual), 60 per actor, 1440 total.
- Filename fields: modality-vocal-emotion-intensity-statement-repetition-actor.
  Emotion codes 1..8 = neutral, calm, happy, sad, angry, fearful, disgust, surprised.
  Odd actor = male, even = female.

## Decoding
- Audio: `ffmpeg -i f -ac 1 -ar 16000 -f f32le -` piped to numpy float32.
- Video: OpenCV; take every 6th frame (about 5 fps).

## Degradation conditions (deterministic, seed derived from clip id)
Video conditions applied to full frame BEFORE face detection:
- `clean`
- `compress`: downscale to 320x180, JPEG encode quality 8, decode, upscale to original size.
- `lowlight`: f = 255*((f/255)**2.2)*0.35 + N(0, 6) noise, clip to uint8.
- `blur`: Gaussian blur, kernel 21, sigma 6.
Audio conditions: `clean`, `snr20`, `snr10`, `snr0` (additive white Gaussian noise at the given SNR in dB relative to the clip's signal power).

## Face crop
- YuNet `/home/user/data/face_detection_yunet_2023mar.onnx` (cv2.FaceDetectorYN) on frame resized to 640x360, score threshold 0.6.
- Largest face; expand box by 15%; crop from the full-resolution frame; resize to 224x224 RGB.
- If no face: reuse the last valid box in the clip; if none yet, centre square crop. Record per-clip face detection rate per condition.

## Extractors
- A0 `prosody` (librosa, sr 16 kHz): 40 MFCC + delta + delta-delta (mean, std), RMS (mean, std), ZCR (mean, std),
  spectral centroid/bandwidth/rolloff (mean, std), spectral contrast 7 bands (mean, std), chroma 12 (mean),
  F0 via `librosa.yin(fmin=65,fmax=400)` (mean, std, 10th and 90th percentile). Float32 vector.
- A1 `w2v`: `facebook/wav2vec2-base` (transformers), `output_hidden_states=True`, mean over time for each of the 13 hidden states -> (13, 768) float16.
- V0 `mnet`: torchvision `mobilenet_v3_small` IMAGENET1K_V1, features after avgpool (576-d) per face crop; clip vector = concat(mean, std) over frames (1152).
- V1 `clip`: `openai/clip-vit-base-patch32` `get_image_features`, L2-normalised per frame (512). Save per-frame embeddings (pad to max frames with NaN or save as object list) AND clip vector concat(mean, std) (1024).

## Timing
- On the `clean` condition only, record per-clip wall time (time.perf_counter) for: audio decode, video decode+face detection, A0, A1, V0, V1. torch threads = 4 unless running parallel workers (then document thread count). Record the thread setting in the timing file.

## Output (`/home/user/data/features/`)
- Resumable: process actor by actor, write `actor_XX.pkl` (or npz) and skip existing.
- Final merge: `meta.csv` (clip_id, actor, sex, emotion, intensity, statement, repetition, n_frames, duration_s, face_rate_<cond>),
  `prosody_<acond>.npy`, `w2v_<acond>.npy` (N,13,768), `mnet_<vcond>.npy`, `clip_<vcond>.npy`, `clipframes_<vcond>.pkl`, `timings.csv`.
  Row order identical to meta.csv (sorted by clip_id).
