#!/usr/bin/env python3
"""Experiments for the audio-visual emotion cascade study (protocol in code/SPEC_experiments.md).

Run from the repository root:  python3 code/run_experiments.py
"""
import json
import pickle
import time
import warnings
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from scipy.optimize import minimize_scalar
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from river.drift import ADWIN

warnings.filterwarnings("ignore", category=ConvergenceWarning)

ROOT = Path(__file__).resolve().parents[1]
FEAT = Path("/home/user/data/features")
CACHE = Path("/home/user/data/cache")
RES = ROOT / "results"
FIG = ROOT / "manuscript" / "figures"

SEED = 2026
N_CLASSES = 8
EMOTIONS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
ZS_ADJ = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgusted", "surprised"]
ZS_TEMPLATES = ["a photo of a {} face.", "a photo of a person looking {}.",
                "a close-up photo of a {} person."]
N_FOLDS = 6
N_INNER = 4
C_GRID = [0.01, 0.1, 1.0]
LAYER_C = 0.1
N_LAYERS = 13
N_BOOT = 2000
GRID = np.round(np.arange(0, 51) * 0.02, 2)
N_RANDOM = 50
OP_TOL = 0.01
TEMP_BOUNDS = (0.05, 20.0)
PH_DELTA, PH_LAMBDA, PH_ALPHA = 0.005, 5.0, 1.0
ADWIN_DELTA = 0.002
N_DRIFT_SEEDS = 20
HALF = 240
N_JOBS = 4

AUDIO_CONDS = ["clean", "snr20", "snr10", "snr0"]
VIDEO_CONDS = ["clean", "compress", "lowlight", "blur"]
# condition -> (audio condition, video condition)
CONDITIONS = OrderedDict([
    ("clean", ("clean", "clean")),
    ("snr20", ("snr20", "clean")),
    ("snr10", ("snr10", "clean")),
    ("snr0", ("snr0", "clean")),
    ("compress", ("clean", "compress")),
    ("lowlight", ("clean", "lowlight")),
    ("blur", ("clean", "blur")),
    ("snr10+compress", ("snr10", "compress")),
    ("snr0+lowlight", ("snr0", "lowlight")),
])
DEG_CONDS = [c for c in CONDITIONS if c != "clean"]

TRAINED = ["A0", "A1", "V0", "V1", "EF"]
MODEL_MCONDS = {"A0": AUDIO_CONDS, "A1": AUDIO_CONDS, "V0": VIDEO_CONDS, "V1": VIDEO_CONDS,
                "V1ZS": VIDEO_CONDS, "EF": ["clean"]}

PAL = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "vermil": "#D55E00",
       "sky": "#56B4E9", "purple": "#CC79A7", "yellow": "#F0E442", "black": "#000000",
       "grey": "#999999"}


# ----------------------------------------------------------------------------
# Data and protocol helpers
# ----------------------------------------------------------------------------
def load_meta():
    meta = pd.read_csv(FEAT / "meta.csv")
    y = meta["emotion"].values.astype(int) - 1
    actor = meta["actor"].values.astype(int)
    male = meta["sex"].values == "male"
    return meta, y, actor, male


def fold_splits(actor):
    n = len(actor)
    out = []
    for k in range(N_FOLDS):
        te = np.where(np.isin(actor, np.arange(4 * k + 1, 4 * k + 5)))[0]
        tr = np.setdiff1d(np.arange(n), te)
        out.append((tr, te))
    return out


def mcond(model, cond):
    """Modality-level condition name used by a model under a full condition."""
    a, v = CONDITIONS[cond]
    if model in ("A0", "A1"):
        return a
    if model in ("V0", "V1", "V1ZS"):
        return v
    return "clean"


def load_modality(model, mc):
    if model == "A0":
        return np.load(FEAT / f"prosody_{mc}.npy").astype(np.float32)
    if model == "A1":
        return np.load(FEAT / f"w2v_{mc}.npy").astype(np.float32).mean(axis=1)
    if model == "V0":
        return np.load(FEAT / f"mnet_{mc}.npy").astype(np.float32)
    if model == "V1":
        return np.load(FEAT / f"clip_{mc}.npy").astype(np.float32)
    raise ValueError(model)


def model_inputs(model):
    if model == "EF":
        return {"clean": np.hstack([load_modality("A1", "clean"), load_modality("V1", "clean")])}
    return {mc: load_modality(model, mc) for mc in MODEL_MCONDS[model]}


# ----------------------------------------------------------------------------
# Numerics and metrics
# ----------------------------------------------------------------------------
def log_softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def softmax(z):
    return np.exp(log_softmax(z))


def log_probs(p):
    return np.log(np.clip(p, 1e-12, 1.0))


def fit_temperature(logits, y):
    idx = np.arange(len(y))

    def nll(T):
        return -log_softmax(logits / T)[idx, y].mean()

    res = minimize_scalar(nll, bounds=TEMP_BOUNDS, method="bounded")
    return float(res.x)


def conf_matrix(y, p):
    return np.bincount(y * N_CLASSES + p, minlength=N_CLASSES * N_CLASSES).reshape(N_CLASSES, N_CLASSES)


def acc_from_cm(cm):
    return np.diagonal(cm, axis1=-2, axis2=-1).sum(-1) / cm.sum(axis=(-2, -1))


def macro_f1_from_cm(cm):
    tp = np.diagonal(cm, axis1=-2, axis2=-1).astype(float)
    fp = cm.sum(axis=-2) - tp
    fn = cm.sum(axis=-1) - tp
    den = 2 * tp + fp + fn
    f1 = np.where(den > 0, 2 * tp / np.where(den > 0, den, 1.0), 0.0)
    return f1.mean(axis=-1)


def actor_cms(y, p, actor):
    out = np.zeros((24, N_CLASSES, N_CLASSES), dtype=np.int64)
    for a in range(1, 25):
        m = actor == a
        out[a - 1] = conf_matrix(y[m], p[m])
    return out


def per_actor_acc(y, p, actor):
    return np.array([np.mean(p[actor == a] == y[actor == a]) for a in range(1, 25)])


def cluster_boot(cm_actor, seed):
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, cm_actor.shape[0], size=(N_BOOT, cm_actor.shape[0]))
    cmb = cm_actor[draws].sum(axis=1)
    acc = acc_from_cm(cmb)
    f1 = macro_f1_from_cm(cmb)
    return np.percentile(acc, [2.5, 97.5]), np.percentile(f1, [2.5, 97.5])


def sex_gap_boot(cm_actor, male_idx, female_idx, seed):
    rng = np.random.default_rng(seed)
    dm = rng.choice(male_idx, size=(N_BOOT, len(male_idx)))
    df = rng.choice(female_idx, size=(N_BOOT, len(female_idx)))
    gap = acc_from_cm(cm_actor[dm].sum(axis=1)) - acc_from_cm(cm_actor[df].sum(axis=1))
    return np.percentile(gap, [2.5, 97.5])


def ece(conf, correct, n_bins=15):
    b = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for i in range(n_bins):
        m = b == i
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


# ----------------------------------------------------------------------------
# Fusion
# ----------------------------------------------------------------------------
def fuse_lf(plist):
    return softmax(sum(log_probs(p) for p in plist))


def fuse_cwf(plist):
    m = len(plist)
    P = np.stack(plist)
    L = np.stack([log_probs(p) for p in plist])
    H = -(P * L).sum(axis=-1) / np.log(N_CLASSES)
    w = 1.0 - H
    s = w.sum(axis=0, keepdims=True)
    ok = s > 1e-12
    w = np.where(ok, w * m / np.where(ok, s, 1.0), 1.0)
    return softmax((w[..., None] * L).sum(axis=0))


# ----------------------------------------------------------------------------
# Trained tiers (run in worker processes)
# ----------------------------------------------------------------------------
def fit_model_fold(model, k):
    path = CACHE / f"{model}_fold{k}.pkl"
    if path.exists():
        return str(path)
    _, y, actor, _ = load_meta()
    tr, te = fold_splits(actor)[k]
    inputs = model_inputs(model)
    Xc = inputs["clean"]
    ytr = y[tr]
    gtr = actor[tr]
    with threadpool_limits(limits=1):
        cv_logits = {}
        for C in C_GRID:
            oof = np.zeros((len(tr), N_CLASSES))
            for itr, iva in GroupKFold(N_INNER).split(tr, ytr, gtr):
                pipe = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))
                pipe.fit(Xc[tr[itr]], ytr[itr])
                oof[iva] = pipe.decision_function(Xc[tr[iva]])
            cv_logits[C] = oof
        cv_f1 = {C: float(macro_f1_from_cm(conf_matrix(ytr, cv_logits[C].argmax(1)))) for C in C_GRID}
        C_best = max(C_GRID, key=lambda c: cv_f1[c])
        T = fit_temperature(cv_logits[C_best], ytr)
        pipe = make_pipeline(StandardScaler(), LogisticRegression(C=C_best, max_iter=3000))
        pipe.fit(Xc[tr], ytr)
        test_probs = {mc: softmax(pipe.decision_function(X[te]) / T) for mc, X in inputs.items()}
    result = dict(model=model, fold=k, C=C_best, T=T, cv_macro_f1=cv_f1, train_idx=tr, test_idx=te,
                  oof_probs=softmax(cv_logits[C_best] / T), test_probs=test_probs,
                  coef=pipe[-1].coef_)
    with open(path, "wb") as f:
        pickle.dump(result, f)
    return str(path)


def layer_probe_fold(layer, k):
    _, y, actor, _ = load_meta()
    tr, te = fold_splits(actor)[k]
    X = np.load(FEAT / "w2v_clean.npy")[:, layer, :].astype(np.float32)
    with threadpool_limits(limits=1):
        pipe = make_pipeline(StandardScaler(), LogisticRegression(C=LAYER_C, max_iter=3000))
        pipe.fit(X[tr], y[tr])
        pred = pipe.predict(X[te])
    return layer, te, pred


# ----------------------------------------------------------------------------
# Zero-shot CLIP (V1-ZS)
# ----------------------------------------------------------------------------
def zs_text_embeddings():
    import torch
    from transformers import CLIPModel, CLIPTokenizer
    name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(name).eval()
    tok = CLIPTokenizer.from_pretrained(name)
    embs = []
    with torch.no_grad():
        for adj in ZS_ADJ:
            texts = [t.format(adj) for t in ZS_TEMPLATES]
            enc = tok(texts, padding=True, return_tensors="pt")
            out = model.get_text_features(**enc)
            if not torch.is_tensor(out):
                out = out.pooler_output
            out = out / out.norm(dim=-1, keepdim=True)
            e = out.mean(dim=0)
            e = e / e.norm()
            embs.append(e.numpy().astype(np.float64))
    return np.stack(embs)


def zs_logits(text_emb, vcond):
    frames = pickle.load(open(FEAT / f"clipframes_{vcond}.pkl", "rb"))
    out = np.zeros((len(frames), N_CLASSES))
    for i, f in enumerate(frames):
        f = f.astype(np.float64)
        f = f / np.linalg.norm(f, axis=1, keepdims=True)
        out[i] = (f @ text_emb.T).mean(axis=0) * 100.0
    return out


# ----------------------------------------------------------------------------
# Cascade
# ----------------------------------------------------------------------------
def operating_point(conf, p1, pF, y):
    """Smallest escalation rate on the grid whose accuracy is within OP_TOL of F-all, plus threshold."""
    n = len(y)
    order = np.argsort(conf, kind="stable")
    cs = conf[order]
    accF = np.mean(pF == y)
    curve = []
    chosen = None
    for r in GRID:
        k = int(round(r * n))
        pred = p1.copy()
        sel = order[:k]
        pred[sel] = pF[sel]
        a = np.mean(pred == y)
        curve.append(a)
        if chosen is None and a >= accF - OP_TOL - 1e-12:
            chosen = (float(r), k)
    r_star, k_star = chosen
    tau = float(cs[k_star]) if k_star < n else 1.0 + 1e-9
    return r_star, tau, np.array(curve), float(accF)


def cascade_pred(conf, p1, pF, tau):
    esc = conf < tau
    return np.where(esc, pF, p1), esc


def stage1_probs(variant, get):
    if variant == "A":
        return fuse_lf([get("A0"), get("V0")])
    return get("A0")


F_ALL = ["A0", "A1", "V0", "V1"]


def cascade_run(variant, get, tau_arr):
    S = stage1_probs(variant, get)
    FULL = fuse_lf([get(m) for m in F_ALL])
    conf = S.max(axis=1)
    pred, esc = cascade_pred(conf, S.argmax(1), FULL.argmax(1), tau_arr)
    final_prob = np.where(esc[:, None], FULL, S)
    return dict(pred=pred, esc=esc, conf=conf, S=S, FULL=FULL, prob=final_prob)


# ----------------------------------------------------------------------------
# Drift detectors
# ----------------------------------------------------------------------------
def page_hinkley_alarms(x, delta=PH_DELTA, lam=PH_LAMBDA, alpha=PH_ALPHA):
    alarms = []
    n = 0
    mean = 0.0
    m = 0.0
    M = np.inf
    for t, xt in enumerate(x):
        n += 1
        mean += (xt - mean) / n
        m = alpha * m + (xt - mean - delta)
        M = min(M, m)
        if m - M > lam:
            alarms.append(t)
            n, mean, m, M = 0, 0.0, 0.0, np.inf
    return alarms


def adwin_alarms(x, delta=ADWIN_DELTA):
    ad = ADWIN(delta=delta)
    alarms = []
    for t, xt in enumerate(x):
        ad.update(float(xt))
        if ad.drift_detected:
            alarms.append(t)
    return alarms


# ----------------------------------------------------------------------------
# Cost model
# ----------------------------------------------------------------------------
def cost_tables(meta):
    tim = pd.read_csv(FEAT / "timings.csv")
    tim = tim[tim["threads"] == 2].merge(meta[["clip_id", "duration_s"]], on="clip_id")
    aud = tim["audio_decode_s"].values
    vid = tim["video_decode_face_s"].values
    a0, a1 = tim["a0_prosody_s"].values, tim["a1_w2v_s"].values
    v0, v1 = tim["v0_mnet_s"].values, tim["v1_clip_s"].values
    dur = tim["duration_s"].values
    comps = OrderedDict([
        ("audio_decode", aud), ("video_decode_face", vid), ("A0 prosody", a0),
        ("A1 w2v", a1), ("V0 mnet", v0), ("V1 clip", v1),
    ])
    pipes = OrderedDict([
        ("A0 alone (audio decode + A0)", aud + a0),
        ("A1 alone (audio decode + A1)", aud + a1),
        ("V0 alone (video decode + V0)", vid + v0),
        ("V1 alone (video decode + V1)", vid + v1),
        ("F-cheap stage 1 (audio + video + A0 + V0)", aud + vid + a0 + v0),
        ("Escalation add-on (A1 + V1)", a1 + v1),
        ("F-full (audio + video + A1 + V1)", aud + vid + a1 + v1),
        ("F-all (all components)", aud + vid + a0 + a1 + v0 + v1),
        ("Variant B stage 1 (audio + A0)", aud + a0),
    ])
    rows = []
    for name, x in list(comps.items()) + list(pipes.items()):
        ratio = x / dur
        rows.append(dict(component=name, n_clips=len(x), mean_s=float(x.mean()), std_s=float(x.std(ddof=1)),
                         median_s=float(np.median(x)), rtf_mean=float(ratio.mean()),
                         rtf_median=float(np.median(ratio))))
    cost_df = pd.DataFrame(rows)
    C = {k: float(v.mean()) for k, v in pipes.items()}
    C["components"] = {k: float(v.mean()) for k, v in comps.items()}
    return cost_df, C, len(dur), float(dur.mean())


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------
def set_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
    })


def save_fig(fig, name):
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)


def fig_cost_accuracy(sweep_df, singles, ops):
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    styles = {"A": (PAL["blue"], "-"), "B": (PAL["orange"], "-")}
    for v in ["A", "B"]:
        d = sweep_df[sweep_df.variant == v]
        ax.plot(d.mean_cost_s, d.accuracy, styles[v][1], color=styles[v][0],
                label=f"Cascade variant {v} (sweep)")
    d = sweep_df[sweep_df.variant == "A"]
    ax.plot(d.mean_cost_s, d.random_accuracy, "--", color=PAL["grey"], label="Random escalation (A)")
    for _, r in singles.iterrows():
        ax.scatter(r.cost_s, r.accuracy, s=14, color=PAL["black"], zorder=3)
        ax.annotate(r["name"], (r.cost_s, r.accuracy), textcoords="offset points", xytext=(3, -6),
                    fontsize=6)
    for i, (_, r) in enumerate(ops.iterrows()):
        ax.scatter(r.cost_s, r.accuracy, marker="*", s=60, color=PAL["vermil"], zorder=4)
        ax.annotate(r["name"], (r.cost_s, r.accuracy), textcoords="offset points",
                    xytext=(-22, 4 + 9 * i), fontsize=6, color=PAL["vermil"])
    ax.set_xlabel("Mean CPU seconds per clip")
    ax.set_ylabel("Accuracy")
    ax.legend(frameon=False, loc="lower right")
    save_fig(fig, "cost_accuracy.pdf")


def fig_robustness(rob_df):
    systems = ["A1", "V1", "F-all (LF)", "F-all (CWF)", "Cascade A (op)"]
    colours = [PAL["sky"], PAL["green"], PAL["blue"], PAL["orange"], PAL["vermil"]]
    conds = list(CONDITIONS.keys())
    fig, ax = plt.subplots(figsize=(7, 2.6))
    w = 0.16
    x = np.arange(len(conds))
    for i, (s, c) in enumerate(zip(systems, colours)):
        vals = [rob_df[(rob_df.condition == cd) & (rob_df.system == s)].accuracy.values[0] for cd in conds]
        ax.bar(x + (i - 2) * w, vals, w, color=c, label=s)
    ax.set_xticks(x)
    ax.set_xticklabels(conds, rotation=20, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 0.9)
    ax.legend(frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.13))
    save_fig(fig, "robustness.pdf")


def fig_drift(conf, ph_alarms, ad_alarms, title):
    t = np.arange(len(conf))
    roll = pd.Series(conf).rolling(20, min_periods=1).mean().values
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    ax.plot(t, roll, color=PAL["blue"], lw=1, label="Confidence (rolling mean, 20)")
    ax.axvline(HALF, color=PAL["black"], ls="--", lw=0.8, label="Change point")
    ax.scatter(ph_alarms, [0.16] * len(ph_alarms), marker="v", s=16, color=PAL["vermil"], label="Page-Hinkley")
    ax.scatter(ad_alarms, [0.24] * len(ad_alarms), marker="x", s=16, color=PAL["green"], label="ADWIN")
    ax.set_xlabel("Clip index in stream")
    ax.set_ylabel("Stage-1 confidence")
    ax.set_ylim(0.1, 1.02)
    ax.set_title(title, fontsize=8)
    ax.legend(frameon=False, fontsize=6, loc="upper left", bbox_to_anchor=(0.0, 0.62))
    save_fig(fig, "drift_stream.pdf")


def fig_confusion(cm_norm):
    fig, ax = plt.subplots(figsize=(3.5, 3.1))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(N_CLASSES))
    ax.set_yticks(range(N_CLASSES))
    ax.set_xticklabels(EMOTIONS, rotation=45, ha="right")
    ax.set_yticklabels(EMOTIONS)
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            v = cm_norm[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5,
                    color="white" if v > 0.6 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    save_fig(fig, "confusion.pdf")


def fig_attribution(att_df):
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    x = np.arange(len(att_df))
    ax.bar(x - 0.2, att_df.audio_share, 0.4, color=PAL["blue"], label="Audio (A0+A1)")
    ax.bar(x + 0.2, att_df.visual_share, 0.4, color=PAL["orange"], label="Visual (V0+V1)")
    ax.set_xticks(x)
    ax.set_xticklabels(att_df.emotion, rotation=45, ha="right")
    ax.set_ylabel("Mean share of attribution")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    save_fig(fig, "attribution.pdf")


def fig_layer_probe(layer_df):
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    yerr = np.vstack([layer_df.accuracy - layer_df.acc_ci_low, layer_df.acc_ci_high - layer_df.accuracy])
    ax.errorbar(layer_df.layer, layer_df.accuracy, yerr=yerr, marker="o", ms=3, lw=1, capsize=2,
                color=PAL["blue"])
    ax.set_xlabel("wav2vec2 hidden-state layer (0 = input projection)")
    ax.set_ylabel("Accuracy (6-fold actor CV)")
    ax.set_xticks(layer_df.layer)
    save_fig(fig, "layer_probe.pdf")


# ----------------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------------
def main():
    t_start = time.perf_counter()
    for d in (CACHE, RES, FIG):
        d.mkdir(parents=True, exist_ok=True)
    meta, y, actor, male = load_meta()
    N = len(y)
    splits = fold_splits(actor)
    male_actor_idx = np.array([a - 1 for a in range(1, 25) if a % 2 == 1])
    female_actor_idx = np.array([a - 1 for a in range(1, 25) if a % 2 == 0])

    # ---- 1. trained tiers and layer probe (parallel, cached per fold) -----
    jobs = [(m, k) for m in TRAINED for k in range(N_FOLDS) if not (CACHE / f"{m}_fold{k}.pkl").exists()]
    layer_path = CACHE / "layer_probe.pkl"
    tasks = [delayed(fit_model_fold)(m, k) for m, k in jobs]
    if not layer_path.exists():
        tasks += [delayed(layer_probe_fold)(l, k) for l in range(N_LAYERS) for k in range(N_FOLDS)]
    if tasks:
        print(f"Training {len(jobs)} tier-fold models and layer probes on {N_JOBS} workers ...", flush=True)
        outs = Parallel(n_jobs=N_JOBS)(tasks)
        if not layer_path.exists():
            layer_pred = np.zeros((N_LAYERS, N), dtype=int)
            for out in outs:
                if isinstance(out, tuple):
                    layer, te, pred = out
                    layer_pred[layer, te] = pred
            with open(layer_path, "wb") as f:
                pickle.dump(layer_pred, f)
        print(f"  done at {time.perf_counter() - t_start:.1f}s", flush=True)
    layer_pred = pickle.load(open(layer_path, "rb"))

    results = {m: [pickle.load(open(CACHE / f"{m}_fold{k}.pkl", "rb")) for k in range(N_FOLDS)]
               for m in TRAINED}

    # ---- 2. zero-shot CLIP (V1-ZS), temperature fit on training actors only ----
    text_emb = zs_text_embeddings()
    zs_path = CACHE / "V1ZS_logits.pkl"
    if zs_path.exists():
        zs = pickle.load(open(zs_path, "rb"))
    else:
        zs = {vc: zs_logits(text_emb, vc) for vc in VIDEO_CONDS}
        with open(zs_path, "wb") as f:
            pickle.dump(zs, f)
    results["V1ZS"] = []
    for k, (tr, te) in enumerate(splits):
        T = fit_temperature(zs["clean"][tr], y[tr])
        res = dict(model="V1ZS", fold=k, T=T, train_idx=tr, test_idx=te,
                   oof_probs=softmax(zs["clean"][tr] / T),
                   test_probs={vc: softmax(zs[vc][te] / T) for vc in VIDEO_CONDS})
        with open(CACHE / f"V1ZS_fold{k}.pkl", "wb") as f:
            pickle.dump(res, f)
        results["V1ZS"].append(res)

    # ---- 3. assemble pooled calibrated test probabilities ---------------------
    P = {}
    for m, mconds in MODEL_MCONDS.items():
        P[m] = {}
        for mc in mconds:
            arr = np.full((N, N_CLASSES), np.nan)
            for k in range(N_FOLDS):
                arr[results[m][k]["test_idx"]] = results[m][k]["test_probs"][mc]
            assert not np.isnan(arr).any(), (m, mc)
            P[m][mc] = arr

    def PR(model, cond):
        return P[model][mcond(model, cond)]

    # ---- 4. cascade operating points from training out-of-fold predictions -----
    TAU = {v: np.full(N, np.nan) for v in ("A", "B")}
    op_folds = {v: [] for v in ("A", "B")}
    for k, (tr, te) in enumerate(splits):
        ytr = y[tr]
        G = lambda m, kk=k: results[m][kk]["oof_probs"]
        FULL_oof = fuse_lf([G(m) for m in F_ALL])
        for v in ("A", "B"):
            S = stage1_probs(v, G)
            r_star, tau, curve, accF = operating_point(S.max(1), S.argmax(1), FULL_oof.argmax(1), ytr)
            TAU[v][te] = tau
            op_folds[v].append(dict(fold=k + 1, r_star=r_star, tau=tau, oof_acc_fall=accF,
                                    oof_acc_at_op=float(curve[GRID.tolist().index(r_star)])))
    assert not np.isnan(TAU["A"]).any() and not np.isnan(TAU["B"]).any()

    # ---- 5. system outputs for each condition ---------------------------------
    def system_outputs(cond):
        get = lambda m: PR(m, cond)
        lf_cheap = fuse_lf([get("A0"), get("V0")])
        lf_full = fuse_lf([get("A1"), get("V1")])
        lf_all = fuse_lf([get(m) for m in F_ALL])
        cwf_all = fuse_cwf([get(m) for m in F_ALL])
        cA = cascade_run("A", get, TAU["A"])
        cB = cascade_run("B", get, TAU["B"])
        out = OrderedDict()
        out["A0"] = dict(pred=get("A0").argmax(1), prob=get("A0"))
        out["A1"] = dict(pred=get("A1").argmax(1), prob=get("A1"))
        out["V0"] = dict(pred=get("V0").argmax(1), prob=get("V0"))
        out["V1"] = dict(pred=get("V1").argmax(1), prob=get("V1"))
        out["F-cheap (LF)"] = dict(pred=lf_cheap.argmax(1), prob=lf_cheap,
                                   stage1_conf=float(lf_cheap.max(1).mean()))
        out["F-full (LF)"] = dict(pred=lf_full.argmax(1), prob=lf_full)
        out["F-all (LF)"] = dict(pred=lf_all.argmax(1), prob=lf_all)
        out["F-all (CWF)"] = dict(pred=cwf_all.argmax(1), prob=cwf_all)
        out["Cascade A (op)"] = dict(pred=cA["pred"], prob=cA["prob"], esc=float(cA["esc"].mean()),
                                     stage1_conf=float(cA["conf"].mean()), esc_mask=cA["esc"])
        out["Cascade B (op)"] = dict(pred=cB["pred"], prob=cB["prob"], esc=float(cB["esc"].mean()),
                                     stage1_conf=float(cB["conf"].mean()), esc_mask=cB["esc"])
        return out

    sys_by_cond = {cond: system_outputs(cond) for cond in CONDITIONS}
    clean = sys_by_cond["clean"]

    # ---- 6. costs --------------------------------------------------------------
    cost_df, COST, n_timed, mean_dur = cost_tables(meta)
    C_CHEAP = COST["F-cheap stage 1 (audio + video + A0 + V0)"]
    C_ALL = COST["F-all (all components)"]
    C_B1 = COST["Variant B stage 1 (audio + A0)"]

    def cascade_cost(esc_rate, c1):
        return (1 - esc_rate) * c1 + esc_rate * C_ALL

    # ---- 7. main results with cluster bootstrap CIs ---------------------------
    zs_prob = P["V1ZS"]["clean"]
    ef_prob = P["EF"]["clean"]
    main_sys = OrderedDict()
    main_sys["A0 prosody"] = dict(pred=clean["A0"]["pred"], prob=clean["A0"]["prob"],
                                  cost=COST["A0 alone (audio decode + A0)"])
    main_sys["A1 w2v"] = dict(pred=clean["A1"]["pred"], prob=clean["A1"]["prob"],
                              cost=COST["A1 alone (audio decode + A1)"])
    main_sys["V0 mnet"] = dict(pred=clean["V0"]["pred"], prob=clean["V0"]["prob"],
                               cost=COST["V0 alone (video decode + V0)"])
    main_sys["V1 clip"] = dict(pred=clean["V1"]["pred"], prob=clean["V1"]["prob"],
                               cost=COST["V1 alone (video decode + V1)"])
    main_sys["V1-ZS"] = dict(pred=zs_prob.argmax(1), prob=zs_prob,
                             cost=COST["V1 alone (video decode + V1)"])
    main_sys["EF-full"] = dict(pred=ef_prob.argmax(1), prob=ef_prob,
                               cost=COST["F-full (audio + video + A1 + V1)"])
    main_sys["F-cheap (LF)"] = dict(pred=clean["F-cheap (LF)"]["pred"], prob=clean["F-cheap (LF)"]["prob"],
                                    cost=C_CHEAP)
    main_sys["F-full (LF)"] = dict(pred=clean["F-full (LF)"]["pred"], prob=clean["F-full (LF)"]["prob"],
                                   cost=COST["F-full (audio + video + A1 + V1)"])
    main_sys["F-all (LF)"] = dict(pred=clean["F-all (LF)"]["pred"], prob=clean["F-all (LF)"]["prob"],
                                  cost=C_ALL)
    main_sys["F-all (CWF)"] = dict(pred=clean["F-all (CWF)"]["pred"], prob=clean["F-all (CWF)"]["prob"],
                                   cost=C_ALL)
    for name, cost1 in (("Cascade A (op)", C_CHEAP), ("Cascade B (op)", C_B1)):
        esc = clean[name]["esc"]
        main_sys[name] = dict(pred=clean[name]["pred"], prob=clean[name]["prob"],
                              cost=cascade_cost(esc, cost1), esc=esc)

    main_rows = []
    cm_actor_main = {}
    per_actor_main = {}
    for name, d in main_sys.items():
        cm_a = actor_cms(y, d["pred"], actor)
        cm_actor_main[name] = cm_a
        per_actor_main[name] = per_actor_acc(y, d["pred"], actor)
        acc, f1 = acc_from_cm(conf_matrix(y, d["pred"])), macro_f1_from_cm(conf_matrix(y, d["pred"]))
        acc_ci, f1_ci = cluster_boot(cm_a, SEED)
        main_rows.append(dict(system=name, accuracy=float(acc), acc_ci_low=acc_ci[0], acc_ci_high=acc_ci[1],
                              macro_f1=float(f1), f1_ci_low=f1_ci[0], f1_ci_high=f1_ci[1],
                              escalation_rate=d.get("esc", np.nan), mean_cost_s=d["cost"], n_clips=N))
    main_df = pd.DataFrame(main_rows)
    main_df.to_csv(RES / "main_results.csv", index=False)
    print("\nMain results (pooled 6-fold test predictions):")
    print(main_df.round(4).to_string(index=False), flush=True)

    best_single = max(["A0 prosody", "A1 w2v", "V0 mnet", "V1 clip"],
                      key=lambda s: main_sys[s]["pred"].__eq__(y).mean())

    # ---- 8. cost table -----------------------------------------------------------
    cost_df.to_csv(RES / "cost.csv", index=False)

    # ---- 9. cascade sweep (pooled test) and random baseline ----------------------
    rng = np.random.default_rng(SEED + 1)
    sweep_rows = []
    op_pooled = {}
    S_A = stage1_probs("A", lambda m: PR(m, "clean"))
    S_B = stage1_probs("B", lambda m: PR(m, "clean"))
    FULL_clean = fuse_lf([PR(m, "clean") for m in F_ALL])
    for v, S, c1 in (("A", S_A, C_CHEAP), ("B", S_B, C_B1)):
        conf = S.max(1)
        order = np.argsort(conf, kind="stable")
        p1, pF = S.argmax(1), FULL_clean.argmax(1)
        for r in GRID:
            k_esc = int(round(r * N))
            mask = np.zeros(N, dtype=bool)
            mask[order[:k_esc]] = True
            pred = np.where(mask, pF, p1)
            acc, f1 = acc_from_cm(conf_matrix(y, pred)), macro_f1_from_cm(conf_matrix(y, pred))
            esc_rate = mask.mean()
            tau_g = float(conf[order[k_esc]]) if k_esc < N else 1.0
            r_acc, r_f1 = [], []
            for _ in range(N_RANDOM):
                rm = np.zeros(N, dtype=bool)
                rm[rng.permutation(N)[:k_esc]] = True
                rp = np.where(rm, pF, p1)
                r_acc.append(np.mean(rp == y))
                r_f1.append(macro_f1_from_cm(conf_matrix(y, rp)))
            sweep_rows.append(dict(variant=v, escalation_rate=float(r), tau_global=tau_g,
                                   accuracy=float(acc), macro_f1=float(f1),
                                   mean_cost_s=cascade_cost(esc_rate, c1),
                                   saving_vs_F_all=1 - cascade_cost(esc_rate, c1) / C_ALL,
                                   random_accuracy=float(np.mean(r_acc)), random_macro_f1=float(np.mean(r_f1))))
    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv(RES / "cascade_sweep.csv", index=False)

    op_rows = []
    for v, name, c1 in (("A", "Cascade A (op)", C_CHEAP), ("B", "Cascade B (op)", C_B1)):
        esc = clean[name]["esc_mask"]
        cost = cascade_cost(esc.mean(), c1)
        op_rows.append(dict(variant=v, escalation_rate=float(esc.mean()), accuracy=float(np.mean(clean[name]["pred"] == y)),
                            macro_f1=float(macro_f1_from_cm(conf_matrix(y, clean[name]["pred"]))),
                            mean_cost_s=cost, saving_vs_F_all=1 - cost / C_ALL,
                            F_all_accuracy=float(np.mean(clean["F-all (LF)"]["pred"] == y)),
                            F_all_macro_f1=float(macro_f1_from_cm(conf_matrix(y, clean["F-all (LF)"]["pred"])))))
    op_df = pd.DataFrame(op_rows)
    print("\nCascade operating points (pooled test):")
    print(op_df.round(4).to_string(index=False), flush=True)

    # ---- 10. robustness -----------------------------------------------------------
    rob_rows = []
    rob_actor = {}
    for cond in CONDITIONS:
        so = sys_by_cond[cond]
        for name in ["A0", "A1", "V0", "V1", "F-cheap (LF)", "F-all (LF)", "F-all (CWF)",
                     "Cascade A (op)", "Cascade B (op)"]:
            d = so[name]
            rob_rows.append(dict(condition=cond, system=name,
                                 accuracy=float(np.mean(d["pred"] == y)),
                                 macro_f1=float(macro_f1_from_cm(conf_matrix(y, d["pred"]))),
                                 escalation_rate=d.get("esc", np.nan),
                                 mean_stage1_conf=d.get("stage1_conf", np.nan)))
        rob_actor[cond] = {"LF": per_actor_acc(y, so["F-all (LF)"]["pred"], actor),
                           "CWF": per_actor_acc(y, so["F-all (CWF)"]["pred"], actor)}
    rob_df = pd.DataFrame(rob_rows)
    rob_df.to_csv(RES / "robustness.csv", index=False)
    print("\nRobustness (accuracy):")
    print(rob_df.pivot(index="condition", columns="system", values="accuracy").round(3).to_string(), flush=True)

    # ---- 11. drift detection -----------------------------------------------------
    conf_clean = clean["F-cheap (LF)"]["prob"].max(1)
    pred_clean = clean["F-cheap (LF)"]["pred"]
    ctrl_runs = []
    drift_runs = []
    for k, (tr, te) in enumerate(splits):
        for seed in range(N_DRIFT_SEEDS):
            r0 = np.random.default_rng([SEED, 7, k, seed])
            o1, o2 = r0.permutation(te), r0.permutation(te)
            cc = np.concatenate([conf_clean[o1], conf_clean[o2]])
            ctrl_runs.append(dict(fold=k, seed=seed, ph=page_hinkley_alarms(1 - cc), ad=adwin_alarms(cc)))
            for ci, cond in enumerate(DEG_CONDS):
                so = sys_by_cond[cond]
                conf_deg = so["F-cheap (LF)"]["prob"].max(1)
                pred_deg = so["F-cheap (LF)"]["pred"]
                r1 = np.random.default_rng([SEED, 11, k, seed, ci])
                p1, p2 = r1.permutation(te), r1.permutation(te)
                conf = np.concatenate([conf_clean[p1], conf_deg[p2]])
                corr1 = pred_clean[p1] == y[p1]
                corr2 = pred_deg[p2] == y[p2]
                drift_runs.append(dict(
                    condition=cond, fold=k, seed=seed,
                    ph=page_hinkley_alarms(1 - conf), ad=adwin_alarms(conf),
                    acc_clean=float(corr1.mean()), acc_deg=float(corr2.mean()),
                    conf_clean=float(conf[:HALF].mean()), conf_deg=float(conf[HALF:].mean()),
                    stream=conf if (k == 0 and seed == 0) else None))
    ctrl_clean_clips = len(ctrl_runs) * 2 * len(splits[0][1])
    drift_rows = []
    for cond in DEG_CONDS:
        runs = [r for r in drift_runs if r["condition"] == cond]
        for det, key in (("Page-Hinkley", "ph"), ("ADWIN", "ad")):
            first = []
            pre = 0
            for r in runs:
                pre += sum(1 for t in r[key] if t < HALF)
                post = [t for t in r[key] if t >= HALF]
                if post:
                    first.append(post[0] - HALF)
            ctrl_alarms = sum(len(r[key]) for r in ctrl_runs)
            drift_rows.append(dict(
                condition=cond, detector=det, n_runs=len(runs),
                detection_rate=len(first) / len(runs),
                delay_median=float(np.median(first)) if first else np.nan,
                delay_q25=float(np.percentile(first, 25)) if first else np.nan,
                delay_q75=float(np.percentile(first, 75)) if first else np.nan,
                false_alarms_per_1000_clean=1000.0 * ctrl_alarms / ctrl_clean_clips,
                pre_change_alarms_per_1000=1000.0 * pre / (len(runs) * HALF),
                acc_clean_segment=float(np.mean([r["acc_clean"] for r in runs])),
                acc_degraded_segment=float(np.mean([r["acc_deg"] for r in runs])),
                conf_clean_segment=float(np.mean([r["conf_clean"] for r in runs])),
                conf_degraded_segment=float(np.mean([r["conf_deg"] for r in runs]))))
    drift_df = pd.DataFrame(drift_rows)
    drift_df.to_csv(RES / "drift.csv", index=False)
    print("\nDrift detection:")
    print(drift_df.round(3).to_string(index=False), flush=True)

    example = [r for r in drift_runs if r["condition"] == "snr0" and r["fold"] == 0 and r["seed"] == 0][0]

    # ---- 12. fairness and calibration ---------------------------------------------
    fair_rows = []
    fair_systems = OrderedDict()
    for name in ["A0 prosody", "A1 w2v", "V0 mnet", "V1 clip", "EF-full", "F-cheap (LF)", "F-full (LF)",
                 "F-all (LF)", "F-all (CWF)", "Cascade A (op)", "Cascade B (op)"]:
        if name in main_sys:
            fair_systems[name] = main_sys[name]
    for name, d in fair_systems.items():
        pred, prob = d["pred"], d["prob"]
        conf = prob.max(1)
        correct = (pred == y).astype(float)
        row = dict(system=name)
        for tag, mask in (("male", male), ("female", ~male)):
            row[f"acc_{tag}"] = float(correct[mask].mean())
            row[f"macro_f1_{tag}"] = float(macro_f1_from_cm(conf_matrix(y[mask], pred[mask])))
            row[f"ece_{tag}"] = ece(conf[mask], correct[mask])
        row["acc_gap"] = abs(row["acc_male"] - row["acc_female"])
        row["macro_f1_gap"] = abs(row["macro_f1_male"] - row["macro_f1_female"])
        row["ece_gap"] = abs(row["ece_male"] - row["ece_female"])
        gci = sex_gap_boot(cm_actor_main[name], male_actor_idx, female_actor_idx, SEED)
        row["acc_gap_ci_low"], row["acc_gap_ci_high"] = gci[0], gci[1]
        fair_rows.append(row)
    fair_df = pd.DataFrame(fair_rows)
    fair_df.to_csv(RES / "fairness.csv", index=False)
    print("\nFairness (sex):")
    print(fair_df.round(3).to_string(index=False), flush=True)

    perclass_rows = []
    cm_fall = conf_matrix(y, clean["F-all (LF)"]["pred"])
    cm_casA = conf_matrix(y, clean["Cascade A (op)"]["pred"])
    for c in range(N_CLASSES):
        m = y == c
        perclass_rows.append(dict(
            emotion=EMOTIONS[c], n=int(m.sum()),
            recall_F_all=float(cm_fall[c, c] / cm_fall[c].sum()),
            recall_cascade_A=float(cm_casA[c, c] / cm_casA[c].sum()),
            precision_F_all=float(cm_fall[c, c] / max(cm_fall[:, c].sum(), 1)),
            precision_cascade_A=float(cm_casA[c, c] / max(cm_casA[:, c].sum(), 1))))
    per_class_df = pd.DataFrame(perclass_rows)
    per_class_df.to_csv(RES / "per_class.csv", index=False)

    # ---- 13. statistics -------------------------------------------------------------
    def wil(a, b):
        d = a - b
        if np.all(d == 0):
            return np.nan, 1.0
        r = stats.wilcoxon(a, b)
        return float(r.statistic), float(r.pvalue)

    stat_rows = []
    pairs = [
        (f"F-all (LF) vs best single tier ({best_single})", "F-all (LF)", best_single),
        ("Cascade A (op) vs F-all (LF)", "Cascade A (op)", "F-all (LF)"),
        ("Cascade B (op) vs F-all (LF)", "Cascade B (op)", "F-all (LF)"),
        ("F-all (LF) vs F-full (LF)", "F-all (LF)", "F-full (LF)"),
    ]
    for label, a, b in pairs:
        s, p = wil(per_actor_main[a], per_actor_main[b])
        stat_rows.append(dict(comparison=label, mean_diff=float(np.mean(per_actor_main[a] - per_actor_main[b])),
                              median_diff=float(np.median(per_actor_main[a] - per_actor_main[b])),
                              wilcoxon_W=s, p_value=p))
    cwf_p = []
    cwf_rows = []
    for cond in DEG_CONDS:
        s, p = wil(rob_actor[cond]["CWF"], rob_actor[cond]["LF"])
        cwf_p.append(p)
        cwf_rows.append(dict(comparison=f"F-all CWF vs LF under {cond}",
                             mean_diff=float(np.mean(rob_actor[cond]["CWF"] - rob_actor[cond]["LF"])),
                             median_diff=float(np.median(rob_actor[cond]["CWF"] - rob_actor[cond]["LF"])),
                             wilcoxon_W=s, p_value=p))
    holm_adj = holm(cwf_p)
    for row, adj in zip(cwf_rows, holm_adj):
        row["p_holm_cwf_family"] = adj
    stat_rows += cwf_rows
    stats_df = pd.DataFrame(stat_rows)
    stats_df.to_csv(RES / "stats.csv", index=False)
    print("\nWilcoxon tests:")
    print(stats_df.round(4).to_string(index=False), flush=True)

    # per-actor table
    pa = pd.DataFrame({"actor": np.arange(1, 25),
                       "sex": ["male" if a % 2 == 1 else "female" for a in range(1, 25)]})
    for name in ["A0 prosody", "A1 w2v", "V0 mnet", "V1 clip", "F-cheap (LF)", "F-all (LF)", "F-all (CWF)",
                 "Cascade A (op)", "Cascade B (op)"]:
        pa[name] = per_actor_main[name]
    for cond in DEG_CONDS:
        pa[f"F-all LF {cond}"] = rob_actor[cond]["LF"]
        pa[f"F-all CWF {cond}"] = rob_actor[cond]["CWF"]
    pa.to_csv(RES / "per_actor.csv", index=False)

    # ---- 14. explainability -----------------------------------------------------------
    yhat = clean["F-all (LF)"]["pred"]
    logs = {m: log_probs(PR(m, "clean")) for m in F_ALL}
    contrib = {m: np.maximum(logs[m][np.arange(N), yhat] - logs[m].mean(axis=1), 0) for m in F_ALL}
    tot = sum(contrib.values())
    share = {m: np.where(tot > 0, contrib[m] / np.where(tot > 0, tot, 1.0), 0.25) for m in F_ALL}
    audio_sh = share["A0"] + share["A1"]
    visual_sh = share["V0"] + share["V1"]
    att_rows = []
    for c in range(N_CLASSES):
        m = y == c
        att_rows.append(dict(emotion=EMOTIONS[c], n=int(m.sum()), audio_share=float(audio_sh[m].mean()),
                             visual_share=float(visual_sh[m].mean()),
                             audio_share_std=float(audio_sh[m].std(ddof=1)),
                             visual_share_std=float(visual_sh[m].std(ddof=1))))
    att_df = pd.DataFrame(att_rows)
    att_df.to_csv(RES / "attribution.csv", index=False)
    print("\nAttribution (F-all, mean audio vs visual share by true class):")
    print(att_df.round(3).to_string(index=False), flush=True)

    # prosody features
    names = []
    for pre in ("mfcc", "dmfcc", "ddmfcc"):
        for stat in ("mean", "std"):
            names += [f"{pre}{i}_{stat}" for i in range(40)]
    names += ["rms_mean", "rms_std", "zcr_mean", "zcr_std", "centroid_mean", "centroid_std",
              "bandwidth_mean", "bandwidth_std", "rolloff_mean", "rolloff_std"]
    for stat in ("mean", "std"):
        names += [f"contrast{b}_{stat}" for b in range(7)]
    names += [f"chroma{c}_mean" for c in range(12)]
    names += ["f0_mean", "f0_std", "f0_p10", "f0_p90"]
    assert len(names) == 280
    coefs = np.stack([r["coef"] for r in results["A0"]])  # (folds, 8, 280)
    importance = np.abs(coefs).mean(axis=1).mean(axis=0)
    signed = coefs.mean(axis=0)
    top_idx = np.argsort(-importance)[:10]
    prosody_df = pd.DataFrame({
        "rank": np.arange(1, 11),
        "feature": [names[i] for i in top_idx],
        "mean_abs_std_coef": importance[top_idx],
        "class_largest_positive_coef": [EMOTIONS[int(np.argmax(signed[:, i]))] for i in top_idx],
        "class_most_negative_coef": [EMOTIONS[int(np.argmin(signed[:, i]))] for i in top_idx],
    })
    prosody_df.to_csv(RES / "prosody_top.csv", index=False)
    print("\nTop prosody features:")
    print(prosody_df.round(4).to_string(index=False), flush=True)

    # layer probe
    layer_rows = []
    for l in range(N_LAYERS):
        p = layer_pred[l]
        cm_a = actor_cms(y, p, actor)
        acc = float(acc_from_cm(conf_matrix(y, p)))
        f1 = float(macro_f1_from_cm(conf_matrix(y, p)))
        ci_acc, _ = cluster_boot(cm_a, SEED)
        layer_rows.append(dict(layer=l, accuracy=acc, acc_ci_low=ci_acc[0], acc_ci_high=ci_acc[1],
                               macro_f1=f1))
    layer_df = pd.DataFrame(layer_rows)
    layer_df.to_csv(RES / "layer_probe.csv", index=False)
    print("\nLayer probe (C=0.1, clean):")
    print(layer_df.round(4).to_string(index=False), flush=True)

    # ---- 15. figures ---------------------------------------------------------------
    set_style()
    single_pts = pd.DataFrame([
        dict(name="A0", cost_s=COST["A0 alone (audio decode + A0)"], accuracy=main_sys["A0 prosody"]["pred"].__eq__(y).mean()),
        dict(name="A1", cost_s=COST["A1 alone (audio decode + A1)"], accuracy=main_sys["A1 w2v"]["pred"].__eq__(y).mean()),
        dict(name="V0", cost_s=COST["V0 alone (video decode + V0)"], accuracy=main_sys["V0 mnet"]["pred"].__eq__(y).mean()),
        dict(name="V1", cost_s=COST["V1 alone (video decode + V1)"], accuracy=main_sys["V1 clip"]["pred"].__eq__(y).mean()),
        dict(name="F-cheap", cost_s=C_CHEAP, accuracy=main_sys["F-cheap (LF)"]["pred"].__eq__(y).mean()),
        dict(name="F-all", cost_s=C_ALL, accuracy=main_sys["F-all (LF)"]["pred"].__eq__(y).mean()),
    ])
    op_pts = pd.DataFrame([
        dict(name="op A", cost_s=float(op_df.loc[op_df.variant == "A", "mean_cost_s"].iloc[0]),
             accuracy=float(op_df.loc[op_df.variant == "A", "accuracy"].iloc[0])),
        dict(name="op B", cost_s=float(op_df.loc[op_df.variant == "B", "mean_cost_s"].iloc[0]),
             accuracy=float(op_df.loc[op_df.variant == "B", "accuracy"].iloc[0])),
    ])
    fig_cost_accuracy(sweep_df, single_pts, op_pts)
    fig_robustness(rob_df)
    fig_drift(example["stream"], [t for t in example["ph"]], [t for t in example["ad"]],
              "Example stream: snr0, fold 1, seed 0")
    cm_norm = cm_fall / cm_fall.sum(axis=1, keepdims=True)
    fig_confusion(cm_norm)
    fig_attribution(att_df)
    fig_layer_probe(layer_df)

    # ---- 16. summary.json -------------------------------------------------------------
    runtime = time.perf_counter() - t_start
    fold_info = {m: [dict(fold=r["fold"] + 1, C=r.get("C"), T=r["T"],
                          cv_macro_f1=r.get("cv_macro_f1")) for r in results[m]] for m in results}
    summary = dict(
        runtime_seconds=runtime,
        seed=SEED,
        n_clips=N,
        n_actors=24,
        best_single_tier=best_single,
        main_results=main_rows,
        cost=cost_df.to_dict(orient="records"),
        cost_constants_s=COST,
        cost_timed_clips=n_timed,
        mean_duration_s=mean_dur,
        cascade_operating_points_pooled=op_rows,
        cascade_operating_points_per_fold=op_folds,
        cascade_sweep=sweep_rows,
        robustness=rob_rows,
        drift=drift_rows,
        fairness=fair_rows,
        per_class=perclass_rows,
        stats=stat_rows,
        attribution=att_rows,
        prosody_top10=prosody_df.to_dict(orient="records"),
        layer_probe=layer_rows,
        fold_hyperparameters=fold_info,
    )

    def to_py(o):
        if isinstance(o, dict):
            return {str(k): to_py(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [to_py(v) for v in o]
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return o

    with open(RES / "summary.json", "w") as f:
        json.dump(to_py(summary), f, indent=2)
    print(f"\nTotal runtime: {runtime:.1f} s", flush=True)


if __name__ == "__main__":
    main()
