"""Select the CMU-MOSEI subset used in Cycle 6.

Inputs (in /home/user/data/mosei/): label.csv (MMSA segment list with video_id, clip_id, split) and
CMU_MOSEI_Labels.csd saved as Labels.csd (CMU Multimodal SDK; per video, intervals and 7 labels:
sentiment, happiness, sadness, anger, surprise, disgust, fear; emotion intensities 0-3).
The MMSA clip ids of a video, sorted numerically, correspond one-to-one to the SDK intervals sorted by
start time (verified: sentiment labels agree for all 22,850 segments of the 3,222 videos with equal counts).

Single-label classes: neutral = all six emotion intensities 0; happy / sad / angry = that emotion is the
largest, at least 1.0 and at least 0.5 above the second largest. Other segments are not used.
Duration 3-10 s. All sad and angry segments are kept; happy and neutral are sampled (600 each, seed 2026).
Output: /home/user/data/mosei/subset.csv and code/mosei_subset.csv (same content).
"""
import h5py
import numpy as np
import pandas as pd

D = "/home/user/data/mosei/"
d = h5py.File(D + "Labels.csd", "r")["All Labels"]["data"]
lab = pd.read_csv(D + "label.csv", dtype={"video_id": str})
rows, checked, agree = [], 0, 0
for vid, gr in lab.groupby("video_id"):
    if vid not in d:
        continue
    F = d[vid]["features"][:]
    I = d[vid]["intervals"][:]
    o = np.argsort(I[:, 0], kind="stable")
    F, I = F[o], I[o]
    gr = gr.sort_values("clip_id")
    if len(gr) != len(F):
        continue
    checked += len(gr)
    agree += int(np.sum(np.abs(F[:, 0] - gr.label.values) < 1e-3))
    for (_, r), f, iv in zip(gr.iterrows(), F, I):
        rows.append((vid, int(r.clip_id), r["mode"], iv[0], iv[1], iv[1] - iv[0], *f))
assert agree == checked, (agree, checked)
df = pd.DataFrame(rows, columns=["video_id", "clip_id", "split", "start", "end", "dur",
                                 "sentiment", "happy", "sad", "angry", "surprise", "disgust", "fear"])
E = df[["happy", "sad", "angry", "surprise", "disgust", "fear"]].values
mx, arg, sec = E.max(1), E.argmax(1), np.sort(E, 1)[:, -2]
names = np.array(["happy", "sad", "angry", "surprise", "disgust", "fear"])
cls = np.where(mx == 0, "neutral", np.where((mx >= 1) & (mx - sec >= 0.5), names[arg], "other"))
df["cls"] = cls
df = df[(df.dur >= 3) & (df.dur <= 10) & df.cls.isin(["neutral", "happy", "sad", "angry"])]
rng = np.random.default_rng(2026)
keep = []
for c, g in df.groupby("cls"):
    keep.append(g if c in ("sad", "angry") else g.iloc[rng.permutation(len(g))[:600]])
sub = pd.concat(keep).sort_values(["video_id", "clip_id"]).reset_index(drop=True)
sub["emotion"] = sub.cls.map({"neutral": 1, "happy": 3, "sad": 4, "angry": 5})  # RAVDESS codes
sub["clip_key"] = sub.video_id + "/" + sub.clip_id.astype(str)
sub.to_csv(D + "subset.csv", index=False)
sub.to_csv("mosei_subset.csv", index=False)
print(f"verified {agree}/{checked} segments; subset {len(sub)} clips, {sub.video_id.nunique()} videos")
print(sub.cls.value_counts().to_string())
