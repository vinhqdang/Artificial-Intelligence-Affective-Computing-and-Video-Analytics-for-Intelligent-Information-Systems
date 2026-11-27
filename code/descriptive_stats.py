"""Descriptive statistics of the corpus and the zero-shot CLIP confusion matrix."""
import glob
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

FEAT = Path("/home/user/data/features")
CACHE = Path("/home/user/data/cache")
OUT = Path(__file__).resolve().parents[1] / "results"
EMO = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]


def main():
    meta = pd.read_csv(FEAT / "meta.csv")
    rows = [
        ("n_clips", len(meta)),
        ("duration_mean_s", meta.duration_s.mean()),
        ("duration_sd_s", meta.duration_s.std()),
        ("frames_per_clip_mean", meta.n_frames.mean()),
        ("frames_per_clip_sd", meta.n_frames.std()),
    ]
    for c in [c for c in meta.columns if c.startswith("face_rate_")]:
        rows.append((c + "_min", meta[c].min()))
        rows.append((c + "_mean", meta[c].mean()))
    pd.DataFrame(rows, columns=["statistic", "value"]).to_csv(OUT / "dataset_stats.csv", index=False)

    y = meta.emotion.values - 1
    probs = np.zeros((len(meta), len(EMO)))
    for f in sorted(glob.glob(str(CACHE / "V1ZS_fold*.pkl"))):
        d = pickle.load(open(f, "rb"))
        probs[d["test_idx"]] = d["test_probs"]["clean"]
    pred = probs.argmax(1)
    cm = pd.crosstab(pd.Categorical([EMO[i] for i in y], EMO), pd.Categorical([EMO[i] for i in pred], EMO),
                     rownames=["true"], colnames=["predicted"], dropna=False)
    cm.to_csv(OUT / "zs_confusion.csv")
    share = pd.Series(pred).value_counts(normalize=True).rename(lambda i: EMO[i])
    share.rename("share_of_predictions").to_csv(OUT / "zs_prediction_share.csv")
    print(cm)
    print(share.round(3))


if __name__ == "__main__":
    main()
