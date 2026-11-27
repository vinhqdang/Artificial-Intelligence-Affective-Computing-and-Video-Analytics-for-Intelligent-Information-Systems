"""Single-threaded CPU time of the quality descriptors per clip (results/ext_qd_timing.csv).

Audio: a 3.7 s, 16 kHz waveform. Video: 19 sampled frames (RAVDESS, 1280x720) or 13 (CREMA-D, 480x360),
each with a 224x224 face crop, as in extract_extension.py. Mean of 10 repetitions.
"""
import time
import numpy as np
import cv2
import extract_extension as E

cv2.setNumThreads(1)
rng = np.random.default_rng(2026)
rows = []
y = (0.1 * rng.standard_normal(int(16000 * 3.7))).astype(np.float32)
E.audio_qd(y)
t = time.perf_counter()
for _ in range(10):
    E.audio_qd(y)
rows.append(("audio", "3.7 s at 16 kHz", (time.perf_counter() - t) / 10))
crop = (rng.random((224, 224, 3)) * 255).astype(np.uint8)
for name, (w, h), n in [("video RAVDESS", (1280, 720), 19), ("video CREMA-D", (480, 360), 13)]:
    f = (rng.random((h, w, 3)) * 255).astype(np.uint8)
    t = time.perf_counter()
    for _ in range(10):
        for _ in range(n):
            E.img_qd(f, False)
            E.img_qd(crop, True)
    rows.append((name, f"{n} frames {w}x{h} + crops", (time.perf_counter() - t) / 10))
with open("../results/ext_qd_timing.csv", "w") as fh:
    fh.write("descriptor,input,seconds_per_clip\n")
    for r in rows:
        fh.write(f"{r[0]},{r[1]},{r[2]:.4f}\n")
        print(r)
