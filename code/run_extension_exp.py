#!/usr/bin/env python3
"""Extension experiments X1-X4 (protocol in code/SPEC_ext_experiments.md).

X1  within-corpus replication (RAVDESS 8-class on the six existing folds; CREMA-D 6-class on
    seven actor-grouped, sex-stratified folds), clean-trained and MCT models, F-all, cheap and
    A0/A1 fusions, A1-first cascade, fusion gain, sex gap.
X2  cross-corpus transfer on the six shared classes (neutral, happy, sad, angry, fearful, disgust),
    both directions, plus within-corpus comparators on the same six classes and input-monitor
    shift (flag rates and ADWIN firing on a corpus switch).
X3  input monitors CONTENT, QD, SPK (speaker-adaptive) and QD+SPK on S1 (clean -> degraded),
    S4 (speaker-blocked clean) and S5 (speaker-blocked, degradation from a random onset).
X4  stream-level gating with the best monitor (chosen on RAVDESS, applied to both corpora):
    clean-trained + SLG and MCT + envelope SLG on S1, S4 and S5.

Helpers are imported from run_experiments (R), run_qag (Q), run_slg (S), run_revision (RV),
run_envelope (RE) and run_integrated (I); none of these files is modified.

MOSEI (--corpus mosei, or --corpus all for all three corpora; protocol in code/SPEC_mosei.md) runs the
same X1-X4 machinery with 4 classes (neutral, happy, sad, angry), 5 outer folds grouped by video
(StratifiedGroupKFold, seed 2026), no sex-based fairness, no speaker monitor (SPK not applicable), fusion
gain by a paired cluster bootstrap over videos, X2 in four directions on the 4 shared classes, and X4 with
the QD monitor on S1 and S4. MOSEI outputs go to results/ext_mosei_* and manuscript/figures/ext_mosei*.pdf.

Run from the repository root:
    python3 code/run_extension_exp.py --corpus ravdess --part all
    python3 code/run_extension_exp.py --corpus both --part all      # needs CREMA-D features
    python3 code/run_extension_exp.py --corpus mosei --part all     # needs MOSEI features
    python3 code/run_extension_exp.py --corpus all --part all
"""
import argparse
import json
import pickle
import sys
import time
import warnings
import zlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_experiments as R  # noqa: E402
import run_qag as Q  # noqa: E402
import run_slg as S  # noqa: E402
import run_revision as RV  # noqa: E402
import run_envelope as RE  # noqa: E402
import run_integrated as I  # noqa: E402

warnings.filterwarnings("ignore", category=ConvergenceWarning)

SEED = R.SEED                      # 2026
HALF = R.HALF                      # 240 clips per segment
N_INNER = R.N_INNER
C_GRID = R.C_GRID
N_BOOT = R.N_BOOT
ECE_BINS = 15
SPK_ENROL = 10                     # enrolment clips per speaker session (not scored)
CREMA_SESSIONS = 20                # S4/S5 CREMA-D streams: 20 sessions x 24 clips = 480 clips
CREMA_PER_SESSION = 24
ONSET_LO, ONSET_HI = 120, 360      # S5 degradation onset drawn uniformly in [120, 360]
CREMA_FOLDS = 7
CH = ("audio", "visual")
MODES = ("clean", "mct")
MODELS = ["A0", "A1", "V0", "V1"]
FUSED = {"F-all": MODELS, "F-cheap(A0,V0)": ["A0", "V0"], "LF(A0,A1)": ["A0", "A1"]}
SHARED = [1, 3, 4, 5, 6, 7]        # RAVDESS codes of neutral, happy, sad, angry, fearful, disgust
SHARED4 = [1, 3, 4, 5]             # neutral, happy, sad, angry (RAVDESS codes; used for MOSEI)
MOSEI_MONS = ["CONTENT", "QD"]     # SPK not applicable on MOSEI (no speaker sessions)
MOSEI_FOLDS = 5
MOSEI_STREAM = 480                 # clips per S4 / S5 stream
BASE_CORPORA = ("ravdess", "cremad")
MOSEI_COND_ORDER = ["clean", "snr10", "snr0", "cafe5", "street5", "compress", "lowlight", "h264",
                    "snr10+compress", "snr0+lowlight"]
AUD_ALL = ["clean", "snr20", "snr10", "snr0", "cafe5", "street5"]
VID_ALL = ["clean", "compress", "lowlight", "blur", "h264"]
COMBOS = [("snr10", "compress"), ("snr0", "lowlight")]
MONS = ["CONTENT", "QD", "SPK", "QD+SPK"]
SPK_MONS = ("SPK", "QD+SPK")
VARIANTS = [("clean-trained, no gate", "clean", None), ("clean-trained, SLG", "clean", "clean"),
            ("MCT, no gate", "mct", None), ("MCT, envelope SLG", "mct", "env")]
DIRECTIONS = {"ravdess": "ravdess->cremad", "cremad": "cremad->ravdess"}
TIM_COLS = ["audio_decode_s", "video_decode_face_s", "a0_prosody_s", "a1_w2v_s", "v0_mnet_s", "v1_clip_s"]
PAL = R.PAL

CFG = SimpleNamespace(
    feat_ravdess=Path("/home/user/data/features"),
    feat_cremad=Path("/home/user/data/features_cremad"),
    feat_mosei=Path("/home/user/data/features_mosei"),
    cache=Path("/home/user/data/cache_ext"),
    res=R.RES, fig=R.FIG, seeds=R.N_DRIFT_SEEDS, n_jobs=2, t0=time.perf_counter(),
)


def log(msg):
    print(f"[{time.perf_counter() - CFG.t0:8.1f}s] {msg}", flush=True)


def seed_of(key):
    return SEED + zlib.crc32(key.encode()) % 100000


# ----------------------------------------------------------------------------
# Corpora, registries and tasks
# ----------------------------------------------------------------------------
def discover(feat, cands, prefixes):
    return [c for c in cands if all((feat / f"{p}_{c}.npy").exists() for p in prefixes)]


def build_registry(acond, vcond):
    """Condition name -> (audio condition, visual condition), clean first; combined pairs if both exist."""
    reg = {"clean": ("clean", "clean")}
    for a in acond:
        if a != "clean":
            reg[a] = (a, "clean")
    for v in vcond:
        if v != "clean":
            reg[v] = ("clean", v)
    for a, v in COMBOS:
        if a in acond and v in vcond:
            reg[f"{a}+{v}"] = (a, v)
    return reg


def cremad_folds(actor, male, n_folds=CREMA_FOLDS, seed=SEED):
    """Actor-grouped folds, dealt round-robin within each sex so that folds are sex-balanced."""
    rng = np.random.default_rng(seed)
    fold_of = {}
    for group in (sorted(set(actor[male].tolist())), sorted(set(actor[~male].tolist()))):
        for i, a in enumerate(rng.permutation(group)):
            fold_of[int(a)] = i % n_folds
    fa = np.array([fold_of[int(a)] for a in actor])
    return [(np.where(fa != k)[0], np.where(fa == k)[0]) for k in range(n_folds)]


def cost_for(pk):
    """Per-clip component means (s) from timings.csv, same definitions as R.cost_tables."""
    tim = pd.read_csv(pk.feat / "timings.csv")
    if "threads" in tim.columns and (tim["threads"] == 2).any():
        tim = tim[tim["threads"] == 2]
    m = {c: float(tim[c].mean()) for c in TIM_COLS}
    aud, vid, a0, a1, v0, v1 = (m[c] for c in TIM_COLS)
    return dict(C_AUD=aud, C_VID=vid, C_A0=a0, C_A1=a1, C_V0=v0, C_V1=v1,
                C_ALL=aud + vid + a0 + a1 + v0 + v1, C_CHEAP=aud + vid + a0 + v0, C_A1ALONE=aud + a1)


def make_pack(name):
    pk = SimpleNamespace(name=name)
    if name == "ravdess":
        pk.feat = CFG.feat_ravdess
        meta = pd.read_csv(pk.feat / "meta.csv")
        pk.actor = meta["actor"].values.astype(int)
        pk.male = meta["sex"].values == "male"
        pk.emotion = meta["emotion"].values.astype(int)
        pk.y8 = pk.emotion - 1
        pk.folds_full = R.fold_splits(pk.actor)
    elif name == "cremad":
        pk.feat = CFG.feat_cremad
        meta = pd.read_csv(pk.feat / "meta.csv")
        pk.actor = meta["actor"].values.astype(int)
        pk.male = meta["sex"].str.lower().values == "male"
        pk.emotion = meta["emotion"].values.astype(int)
        pk.y8 = None
        pk.folds_full = cremad_folds(pk.actor, pk.male)
    else:  # mosei: groups are videos; no sex, no speaker sessions
        pk.feat = CFG.feat_mosei
        meta = pd.read_csv(pk.feat / "meta.csv")
        _, pk.actor = np.unique(meta["video_id"].astype(str).values, return_inverse=True)
        pk.actor = pk.actor.astype(int)
        pk.male = None
        pk.emotion = meta["emotion"].values.astype(int)
        pk.y8 = None
        pk.folds_full = mosei_folds(pk.actor, pk.emotion)
    pk.y4 = np.array([SHARED4.index(e) if e in SHARED4 else -1 for e in pk.emotion])
    if name == "mosei":
        pk.y6 = np.full(len(pk.emotion), -1)
    else:
        pk.y6 = np.array([SHARED.index(e) if e in SHARED else -1 for e in pk.emotion])
    pk.n = len(meta)
    pk.acond = discover(pk.feat, AUD_ALL, ("prosody", "w2v", "qda"))
    pk.vcond = discover(pk.feat, VID_ALL, ("mnet", "clip", "qdv"))
    assert "clean" in pk.acond and "clean" in pk.vcond, f"{name}: clean condition missing"
    pk.reg = build_registry(pk.acond, pk.vcond)
    pk.cost = cost_for(pk)
    # speakers with fewer than 24 clips in a test fold are sampled with replacement in CREMA-D S4/S5
    pk.under24 = None if name == "mosei" else [int(sum(1 for a in np.unique(pk.actor[te]) if np.sum(pk.actor[te] == a) < CREMA_PER_SESSION))
                  for _, te in pk.folds_full]
    return pk


def make_tasks(pk):
    """Main within-corpus task of a pack: RAVDESS 8-class, CREMA-D 6-class, MOSEI 4-class."""
    if pk.name == "mosei":
        return mosei_task(pk)
    if pk.name == "ravdess":
        return dict(name="ravdess8", pack=pk, y=pk.y8, K=8, valid=np.ones(pk.n, bool), folds=pk.folds_full)
    return dict(name="cremad6", pack=pk, y=pk.y6, K=6, valid=pk.y6 >= 0, folds=pk.folds_full)


def mosei_folds(group, emotion, n_folds=MOSEI_FOLDS, seed=SEED):
    """Outer folds grouped by video and stratified by class (StratifiedGroupKFold, seed 2026)."""
    sgk = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    X = np.zeros(len(emotion))
    return [(np.sort(tr), np.sort(te)) for tr, te in sgk.split(X, emotion, group)]


def shared4_task(pk):
    """Four shared classes (neutral, happy, sad, angry) on RAVDESS or CREMA-D: same folds, shared clips only."""
    valid = pk.y4 >= 0
    folds = [(tr[valid[tr]], te[valid[te]]) for tr, te in pk.folds_full]
    return dict(name=f"{pk.name}4", pack=pk, y=pk.y4, K=4, valid=valid, folds=folds)


def mosei_task(pk):
    """MOSEI main task: 4 classes, all clips, the five video-grouped folds."""
    return dict(name="mosei4", pack=pk, y=pk.y4, K=4, valid=np.ones(pk.n, bool), folds=pk.folds_full)


def ravdess6_task(pk):
    """Six shared classes on RAVDESS, same six folds, restricted to the shared clips."""
    valid = pk.y6 >= 0
    folds = [(tr[valid[tr]], te[valid[te]]) for tr, te in pk.folds_full]
    return dict(name="ravdess6", pack=pk, y=pk.y6, K=6, valid=valid, folds=folds)


def cm_k(y, p, K):
    return np.bincount(np.asarray(y) * K + np.asarray(p), minlength=K * K).reshape(K, K)


def mc_of(reg, tier, cond):
    return reg[cond][0] if tier in ("A0", "A1") else reg[cond][1]


def load_mod(feat, tier, mc):
    """Same feature definitions as R.load_modality (A1 = mean over the 13 layers)."""
    if tier == "A0":
        return np.load(feat / f"prosody_{mc}.npy").astype(np.float32)
    if tier == "A1":
        return np.load(feat / f"w2v_{mc}.npy").astype(np.float32).mean(axis=1)
    if tier == "V0":
        return np.load(feat / f"mnet_{mc}.npy").astype(np.float32)
    if tier == "V1":
        return np.load(feat / f"clip_{mc}.npy").astype(np.float32)
    raise ValueError(tier)


def tier_mcs(pk, tier):
    return pk.acond if tier in ("A0", "A1") else pk.vcond


# ----------------------------------------------------------------------------
# Trained tiers (cached per fit)
# ----------------------------------------------------------------------------
def fit_core(Xc, ytr, gtr, K, C=None, Xdeg=None):
    """Clean (C selected on inner GroupKFold by macro-F1) or MCT (stacked clean + degraded copy, C given).
    Mirrors R.fit_model_fold and RV.mct_fold."""
    if Xdeg is None:
        Xs, ys, gs = Xc, ytr, gtr
    else:
        Xs = np.vstack([Xc, Xdeg])
        ys = np.concatenate([ytr, ytr])
        gs = np.concatenate([gtr, gtr])
    cands = C_GRID if C is None else [C]
    oofs, f1s = {}, {}
    for c in cands:
        oof = np.zeros((len(ys), K))
        for itr, iva in GroupKFold(N_INNER).split(Xs, ys, gs):
            pipe = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=3000))
            pipe.fit(Xs[itr], ys[itr])
            oof[iva] = pipe.decision_function(Xs[iva])
        oofs[c] = oof
        f1s[c] = float(R.macro_f1_from_cm(cm_k(ys, oof.argmax(1), K)))
    Cb = max(cands, key=lambda c: f1s[c])
    T = R.fit_temperature(oofs[Cb], ys)
    pipe = make_pipeline(StandardScaler(), LogisticRegression(C=Cb, max_iter=3000))
    pipe.fit(Xs, ys)
    return dict(C=Cb, T=T, oof=oofs[Cb], ys=ys, pipe=pipe, cv_f1=f1s)


def fit_job(spec):
    """One tier fit: training rows (feat_tr, tr), test rows (feat_te, te, modality conditions te_mcs)."""
    path = Path(spec["cache"]) / f"{spec['key']}.pkl"
    if path.exists():
        with open(path, "rb") as fh:
            return spec["id"], pickle.load(fh)
    tier, mode = spec["tier"], spec["mode"]
    ftr, fte = Path(spec["ftr"]), Path(spec["fte"])
    tr, te = spec["tr"], spec["te"]
    with threadpool_limits(limits=1):
        Xc = load_mod(ftr, tier, "clean")[tr]
        Xd = load_mod(ftr, tier, RV.MCT_TRAIN_MC[tier])[tr] if mode == "mct" else None
        core = fit_core(Xc, spec["y"], spec["g"], spec["K"], C=spec["C"], Xdeg=Xd)
        test = {mc: core["pipe"].decision_function(load_mod(fte, tier, mc)[te]) for mc in spec["te_mcs"]}
    out = dict(key=spec["key"], tier=tier, mode=mode, C=core["C"], T=core["T"], oof=core["oof"],
               ys=core["ys"], test=test, cv_f1=core["cv_f1"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(out, fh)
    return spec["id"], out


def run_specs(specs):
    if CFG.n_jobs == 1:
        pairs = [fit_job(s) for s in specs]
    else:
        pairs = Parallel(n_jobs=CFG.n_jobs)(delayed(fit_job)(s) for s in specs)
    return dict(pairs)


def fit_within(task, modes=MODES):
    """Fits for every tier, fold and mode of a within-corpus task. Keys: (mode, tier, fold index)."""
    pk = task["pack"]
    res = {}
    for mode in modes:
        specs = []
        for k, (tr, te) in enumerate(task["folds"]):
            for tier in MODELS:
                C = None
                if mode == "mct":
                    C = res[("clean", tier, k)]["C"]
                specs.append(dict(id=(mode, tier, k), key=f"{task['name']}_{tier}_{mode}_f{k}",
                                  cache=str(CFG.cache), ftr=str(pk.feat), fte=str(pk.feat), tr=tr, te=te,
                                  y=task["y"][tr], g=pk.actor[tr], K=task["K"], tier=tier, mode=mode, C=C,
                                  te_mcs=tier_mcs(pk, tier)))
        log(f"fitting {task['name']} {mode}: {len(specs)} tier fits")
        res.update(run_specs(specs))
    return res


def fit_cross(src, tgt, modes=MODES, tag="x"):
    """Train on all valid source clips, test on all valid target clips (one fold)."""
    sp_pk, tg_pk = src["pack"], tgt["pack"]
    tr = np.flatnonzero(src["valid"])
    te = np.flatnonzero(tgt["valid"])
    res = {}
    for mode in modes:
        specs = []
        for tier in MODELS:
            C = res[("clean", tier, 0)]["C"] if mode == "mct" else None
            specs.append(dict(id=(mode, tier, 0), key=f"{tag}_{sp_pk.name}2{tg_pk.name}_{tier}_{mode}",
                              cache=str(CFG.cache), ftr=str(sp_pk.feat), fte=str(tg_pk.feat), tr=tr, te=te,
                              y=src["y"][tr], g=sp_pk.actor[tr], K=src["K"], tier=tier, mode=mode, C=C,
                              te_mcs=tier_mcs(tg_pk, tier)))
        log(f"fitting cross {sp_pk.name}->{tg_pk.name} {mode}")
        res.update(run_specs(specs))
    return res


def assemble(reg, res, te_list, N, K, modes=MODES):
    """Pooled calibrated probabilities P[mode][name][cond] (N,K) filled on the test rows of each fold.
    Temperatures: per tier from inner OOF; fused systems get a temperature fitted on fused OOF logits
    (as in run_qag.py). Returns P and the per-fold temperatures."""
    P, Tinfo = {}, {}
    names = list(MODELS) + list(FUSED)
    for mode in modes:
        P[mode] = {n: {c: np.full((N, K), np.nan) for c in reg} for n in names}
        Tinfo[mode] = dict(T={t: [] for t in MODELS}, Tf={n: [] for n in FUSED})
        for k, te in enumerate(te_list):
            outs = {t: res[(mode, t, k)] for t in MODELS}
            for t in MODELS:
                o = outs[t]
                Tinfo[mode]["T"][t].append(o["T"])
                for c in reg:
                    P[mode][t][c][te] = R.softmax(o["test"][mc_of(reg, t, c)] / o["T"])
            for name, tiers in FUSED.items():
                Lo = sum(R.log_probs(R.softmax(outs[t]["oof"] / outs[t]["T"])) for t in tiers)
                Tf = R.fit_temperature(Lo, outs[tiers[0]]["ys"])
                Tinfo[mode]["Tf"][name].append(Tf)
                for c in reg:
                    Lt = sum(R.log_probs(R.softmax(outs[t]["test"][mc_of(reg, t, c)] / outs[t]["T"]))
                             for t in tiers)
                    P[mode][name][c][te] = R.softmax(Lt / Tf)
    return P, Tinfo


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def eval_metrics(parr, y, idx, actor, K, seed):
    pr = parr[idx]
    pred, conf = pr.argmax(1), pr.max(1)
    yy, aa = y[idx], actor[idx]
    corr = (pred == yy).astype(float)
    cm = cm_k(yy, pred, K)
    ids = np.unique(aa)
    cma = np.stack([cm_k(yy[aa == a], pred[aa == a], K) for a in ids])
    (alo, ahi), (flo, fhi) = R.cluster_boot(cma, seed)
    row = dict(n_clips=int(len(idx)), accuracy=float(corr.mean()), acc_ci_low=float(alo),
               acc_ci_high=float(ahi), macro_f1=float(R.macro_f1_from_cm(cm)), f1_ci_low=float(flo),
               f1_ci_high=float(fhi), mean_max_prob=float(conf.mean()),
               ece_15=float(R.ece(conf, corr, ECE_BINS)))
    return row, cma, ids


def per_actor_acc(parr, y, idx, actor, ids):
    pred = parr[idx].argmax(1)
    yy, aa = y[idx], actor[idx]
    return np.array([np.mean(pred[aa == a] == yy[aa == a]) for a in ids])


def sex_gap(cma, ids, male_of, seed):
    """Female-minus-male accuracy with an actor-cluster bootstrap CI."""
    mp = np.array([i for i, a in enumerate(ids) if male_of[a]])
    fp = np.array([i for i, a in enumerate(ids) if not male_of[a]])
    point = R.acc_from_cm(cma[fp].sum(0)) - R.acc_from_cm(cma[mp].sum(0))
    rng = np.random.default_rng(seed)
    dm = rng.choice(mp, size=(N_BOOT, len(mp)))
    df = rng.choice(fp, size=(N_BOOT, len(fp)))
    gap = R.acc_from_cm(cma[df].sum(axis=1)) - R.acc_from_cm(cma[dm].sum(axis=1))
    lo, hi = np.percentile(gap, [2.5, 97.5])
    return float(point), float(lo), float(hi), float(R.acc_from_cm(cma[mp].sum(0))), float(R.acc_from_cm(cma[fp].sum(0)))


def save_rows(df, name, keys):
    """Write rows to results/<name>, replacing earlier rows with the same key values."""
    CFG.res.mkdir(parents=True, exist_ok=True)
    path = CFG.res / name
    if path.exists() and len(df):
        old = pd.read_csv(path)
        old = old[~old.set_index(keys).index.isin(df.set_index(keys).index)]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(path, index=False)
    log(f"wrote {path} ({len(df)} rows)")


# ----------------------------------------------------------------------------
# X1 within-corpus
# ----------------------------------------------------------------------------
def fusion_boot_rows(pk, P, y, eval_idx, reg):
    """MOSEI fusion gain: F-all minus the best single tier (same training mode and condition), point estimate
    on the pooled test set; paired cluster bootstrap over videos (R.N_BOOT resamples), percentile 95% CI."""
    rows = []
    yy = y[eval_idx]
    uv, inv = np.unique(pk.actor[eval_idx], return_inverse=True)
    nv = np.bincount(inv, minlength=len(uv)).astype(float)
    for mode in MODES:
        for c in reg:
            accs = {t: float(np.mean(P[mode][t][c][eval_idx].argmax(1) == yy)) for t in MODELS}
            best = max(MODELS, key=lambda t: accs[t])
            cF = (P[mode]["F-all"][c][eval_idx].argmax(1) == yy).astype(float)
            cS = (P[mode][best][c][eval_idx].argmax(1) == yy).astype(float)
            sF = np.bincount(inv, weights=cF, minlength=len(uv))
            sS = np.bincount(inv, weights=cS, minlength=len(uv))
            rng = np.random.default_rng(seed_of(f"fus|{pk.name}|{mode}|{c}"))
            draws = rng.integers(0, len(uv), size=(R.N_BOOT, len(uv)))
            gains = (sF[draws].sum(1) - sS[draws].sum(1)) / nv[draws].sum(1)
            lo, hi = np.percentile(gains, [2.5, 97.5])
            point = (sF.sum() - sS.sum()) / nv.sum()
            rows.append(dict(corpus=pk.name, training=mode, condition=c, F_all_accuracy=float(cF.mean()),
                             best_single=best, best_single_accuracy=accs[best],
                             mean_gain_pp=float(100 * point), gain_ci_low_pp=float(100 * lo),
                             gain_ci_high_pp=float(100 * hi), n_videos=int(len(uv)), n_boot=int(R.N_BOOT),
                             method="paired cluster bootstrap over videos"))
    return pd.DataFrame(rows)


def run_x1(task, res=None):
    pk, reg, K, N = task["pack"], task["pack"].reg, task["K"], task["pack"].n
    if res is None:
        res = fit_within(task)
    te_list = [te for _, te in task["folds"]]
    P, Tinfo = assemble(reg, res, te_list, N, K)
    eval_idx = np.sort(np.concatenate(te_list))
    y = task["y"]
    rows, cmas = [], {}
    for mode in MODES:
        for name in list(MODELS) + list(FUSED):
            for c in reg:
                row, cma, ids = eval_metrics(P[mode][name][c], y, eval_idx, pk.actor, K,
                                             seed_of(f"x1|{task['name']}|{mode}|{name}|{c}"))
                cmas[(mode, name, c)] = (cma, ids)
                rows.append(dict(corpus=pk.name, task=task["name"], training=mode, model=name,
                                 condition=c, **row))
    x1 = pd.DataFrame(rows)

    # fusion gain: F-all against the best single tier of the same training mode, Wilcoxon over actors
    # (MOSEI: paired cluster bootstrap over videos, see fusion_boot_rows)
    frows = []
    for mode in (MODES if pk.name != "mosei" else []):
        pvals = []
        for c in reg:
            accs = {t: float(np.mean(P[mode][t][c][eval_idx].argmax(1) == y[eval_idx])) for t in MODELS}
            best = max(MODELS, key=lambda t: accs[t])
            ids = cmas[(mode, "F-all", c)][1]
            fa = per_actor_acc(P[mode]["F-all"][c], y, eval_idx, pk.actor, ids)
            fb = per_actor_acc(P[mode][best][c], y, eval_idx, pk.actor, ids)
            stat, p = RV.wilcoxon(fa, fb)
            pvals.append(1.0 if np.isnan(p) else p)
            acc_f = float(np.mean(P[mode]["F-all"][c][eval_idx].argmax(1) == y[eval_idx]))
            frows.append(dict(corpus=pk.name, training=mode, condition=c, F_all_accuracy=acc_f,
                              best_single=best, best_single_accuracy=accs[best],
                              mean_gain_pp=float(100 * np.mean(fa - fb)), wilcoxon_stat=stat, p_value=p))
        for r, a in zip([r for r in frows if r["training"] == mode], R.holm(pvals)):
            r["p_holm"] = float(a)
    fus = pd.DataFrame(frows) if pk.name != "mosei" else fusion_boot_rows(pk, P, y, eval_idx, reg)

    # A1-first cascade, clean-trained: operating point from training OOF (within OP_TOL of F-all)
    casc = {c: np.full(N, -1) for c in reg}
    esc = {c: np.zeros(N, bool) for c in reg}
    oof_rows = []
    for k, (tr, te) in enumerate(task["folds"]):
        oA1 = res[("clean", "A1", k)]
        ytr = y[tr]
        Sconf = R.softmax(oA1["oof"] / oA1["T"])
        Lf = sum(R.log_probs(R.softmax(res[("clean", t, k)]["oof"] / res[("clean", t, k)]["T"]))
                 for t in MODELS)
        r_star, tau, _, accF = R.operating_point(Sconf.max(1), Sconf.argmax(1), Lf.argmax(1), ytr)
        oof_rows.append(dict(fold=k + 1, oof_escalation=r_star, tau=tau, oof_acc_F_all=accF))
        for c in reg:
            conf = R.softmax(oA1["test"][mc_of(reg, "A1", c)] / oA1["T"]).max(1)
            e = conf < tau
            p1 = P["clean"]["A1"][c][te].argmax(1)
            pF = P["clean"]["F-all"][c][te].argmax(1)
            casc[c][te] = np.where(e, pF, p1)
            esc[c][te] = e
    crow = []
    C_A1, C_ALL = pk.cost["C_A1ALONE"], pk.cost["C_ALL"]
    for c in reg:
        e = esc[c][eval_idx]
        acc_c = float(np.mean(casc[c][eval_idx] == y[eval_idx]))
        rate = float(e.mean())
        cost = (1 - rate) * C_A1 + rate * C_ALL
        crow.append(dict(corpus=pk.name, condition=c, cascade_accuracy=acc_c,
                         stage1_A1_accuracy=float(np.mean(P["clean"]["A1"][c][eval_idx].argmax(1) == y[eval_idx])),
                         F_all_accuracy=float(np.mean(P["clean"]["F-all"][c][eval_idx].argmax(1) == y[eval_idx])),
                         escalation_rate=rate, cascade_mean_cost_s=cost, F_all_cost_s=C_ALL,
                         A1_stage_cost_s=C_A1, cost_saving_share=1 - cost / C_ALL,
                         tau_oof_mean=float(np.mean([o["tau"] for o in oof_rows])),
                         oof_escalation_mean=float(np.mean([o["oof_escalation"] for o in oof_rows]))))
    casc_df = pd.DataFrame(crow)

    # sex gap of F-all (female minus male) with actor-cluster bootstrap
    male_of = ({int(a): bool(pk.male[pk.actor == a][0]) for a in np.unique(pk.actor)}
               if pk.male is not None else {})
    grow = []
    for mode in (MODES if pk.male is not None else []):
        for c in reg:
            cma, ids = cmas[(mode, "F-all", c)]
            gap, lo, hi, am, af = sex_gap(cma, ids, male_of, seed_of(f"gap|{pk.name}|{mode}|{c}"))
            grow.append(dict(corpus=pk.name, training=mode, condition=c, acc_male=am, acc_female=af,
                             gap_female_minus_male=gap, gap_ci_low=lo, gap_ci_high=hi))
    gap_df = pd.DataFrame(grow)

    # temperatures and calibration checks
    summ = dict(
        corpus=pk.name, task=task["name"], n_eval=int(len(eval_idx)),
        C={f"{mode}:{t}": [float(res[(mode, t, k)]["C"]) for k in range(len(task["folds"]))]
           for mode in MODES for t in MODELS},
        T={f"{mode}:{t}": [float(v) for v in Tinfo[mode]["T"][t]] for mode in MODES for t in MODELS},
        T_fused={f"{mode}:{n}": [float(v) for v in Tinfo[mode]["Tf"][n]] for mode in MODES for n in FUSED},
        clean_F_all_accuracy=float(x1[(x1.training == "clean") & (x1.model == "F-all") & (x1.condition == "clean")].accuracy.iloc[0]),
        clean_F_all_snr10_accuracy=float(x1[(x1.training == "clean") & (x1.model == "F-all") & (x1.condition == "snr10")].accuracy.iloc[0])
        if "snr10" in reg else None,
        cascade_oof=oof_rows, cost=pk.cost,
        conditions=list(reg.keys()),
    )
    return x1, fus, casc_df, gap_df, summ


# ----------------------------------------------------------------------------
# X2 cross-corpus
# ----------------------------------------------------------------------------
def monitor_shift(src, tgt, label=None):
    """Input monitors fitted on the source corpus (leave one source fold out) and applied to the target
    clean clips; ADWIN on a stream of 240 held-out source clips followed by 240 target clips."""
    sp, tg = src, tgt
    bank_t = load_bank(tg, clean_only=True)
    rows, per_fold = [], []
    for k, (tr_f, te_f) in enumerate(sp.folds_full):
        bank_s = load_bank(sp, clean_only=True)
        for kind in ("CONTENT", "QD"):
            for ch in CH:
                Xs = bank_s[(kind, ch)]["clean"]
                Xt = bank_t[(kind, ch)]["clean"]
                m = Q.monitor_fold(Xs, tr_f, sp.actor)
                ref = dict(model=m["model"], theta=m["theta"]["p99"], mu=m["oof_mu"], sd=m["oof_sd"])
                lq_t = np.log(Q.monitor_d(ref["model"], Xt) / ref["theta"])
                lq_s = np.log(Q.monitor_d(ref["model"], Xs) / ref["theta"])
                z_t = (lq_t - ref["mu"]) / ref["sd"]
                z_s = (lq_s - ref["mu"]) / ref["sd"]
                tgt_clean = np.arange(tg.n)
                fired, pre, delays = [], [], []
                for s in range(CFG.seeds):
                    src_c = np.random.default_rng([SEED, 111, k, s]).permutation(te_f)[:HALF]
                    tgt_c = np.random.default_rng([SEED, 112, k, s]).choice(tgt_clean, HALF, replace=False)
                    al = np.asarray(R.adwin_alarms(np.concatenate([z_s[src_c], z_t[tgt_c]])))
                    post = al[al >= HALF]
                    fired.append(post.size > 0)
                    pre.append(bool((al < HALF).any()))
                    if post.size:
                        delays.append(int(post[0] - HALF))
                per_fold.append(dict(fold=k + 1, monitor=kind, channel=ch,
                                     target_clean_flag_rate=float(np.mean(lq_t[tgt_clean] > 0)),
                                     source_heldout_flag_rate=float(np.mean(lq_s[te_f] > 0)),
                                     switch_fire_share=float(np.mean(fired)),
                                     pre_switch_alarm_share=float(np.mean(pre)),
                                     median_delay_clips=float(np.median(delays)) if delays else np.nan))
    pf = pd.DataFrame(per_fold)
    for (kind, ch), g in pf.groupby(["monitor", "channel"]):
        rows.append(dict(direction=label or DIRECTIONS[sp.name], monitor=kind, channel=ch, n_folds=len(g),
                         target_clean_flag_rate_mean=g.target_clean_flag_rate.mean(),
                         target_clean_flag_rate_sd=g.target_clean_flag_rate.std(ddof=1),
                         source_heldout_flag_rate_mean=g.source_heldout_flag_rate.mean(),
                         switch_fire_share=g.switch_fire_share.mean(),
                         pre_switch_alarm_share=g.pre_switch_alarm_share.mean(),
                         median_delay_clips=g.median_delay_clips.median()))
    return pd.DataFrame(rows), pf


def run_cross_block(ravdess_task, cremad_task):
    """X2: both directions with clean and MCT training; comparators are within-corpus CV on the same
    six classes (ravdess6 from the RAVDESS six folds, cremad6 from the seven folds)."""
    out_rows = []
    cross_info = {}
    for src, tgt in ((ravdess_task, cremad_task), (cremad_task, ravdess_task)):
        res = fit_cross(src, tgt)
        tg = tgt["pack"]
        te = np.flatnonzero(tgt["valid"])
        P, Tinfo = assemble(tg.reg, res, [te], tg.n, 6)
        direction = DIRECTIONS[src["pack"].name]
        for mode in MODES:
            for name in ["A1", "V1", "F-all"]:
                for c in tg.reg:
                    row, _, _ = eval_metrics(P[mode][name][c], tgt["y"], te, tg.actor, 6,
                                             seed_of(f"x2|{direction}|{mode}|{name}|{c}"))
                    out_rows.append(dict(direction=direction, setting=f"cross-{mode}", target=tg.name,
                                         model=name, condition=c, **row))
        cross_info[direction] = dict(T={f"{m}:{t}": float(res[(m, t, 0)]["T"]) for m in MODES for t in MODELS},
                                     C={f"{m}:{t}": float(res[(m, t, 0)]["C"]) for m in MODES for t in MODELS})
    # within-corpus comparators on the same six classes
    for task in (ravdess_task, cremad_task):
        tg = task["pack"]
        res = fit_within(task)
        P, _ = assemble(tg.reg, res, [te for _, te in task["folds"]], tg.n, 6)
        ev = np.sort(np.concatenate([te for _, te in task["folds"]]))
        for mode in MODES:
            for name in ["A1", "V1", "F-all"]:
                for c in tg.reg:
                    row, _, _ = eval_metrics(P[mode][name][c], task["y"], ev, tg.actor, 6,
                                             seed_of(f"x2w|{task['name']}|{mode}|{name}|{c}"))
                    out_rows.append(dict(direction=f"within:{task['name']}", setting=f"within-{mode}",
                                         target=tg.name, model=name, condition=c, **row))
    return pd.DataFrame(out_rows), cross_info


def run_cross_mosei(packs):
    """X2 on MOSEI: the four shared classes. Trains on RAVDESS or CREMA-D (clean and MCT) and tests on MOSEI,
    and trains on MOSEI and tests on RAVDESS or CREMA-D. Within-corpus 4-class comparators for all three
    corpora, and monitor shift (source-fitted CONTENT and QD) for the four directions."""
    tasks = {"ravdess": shared4_task(packs["ravdess"]), "cremad": shared4_task(packs["cremad"]),
             "mosei": mosei_task(packs["mosei"])}
    out_rows, info, shifts = [], {}, []
    for s_name, t_name in (("ravdess", "mosei"), ("cremad", "mosei"), ("mosei", "ravdess"), ("mosei", "cremad")):
        src, tgt = tasks[s_name], tasks[t_name]
        direction = f"{s_name}->{t_name}"
        tg = tgt["pack"]
        te = np.flatnonzero(tgt["valid"])
        res = fit_cross(src, tgt, tag="x4")
        P, _ = assemble(tg.reg, res, [te], tg.n, 4)
        for mode in MODES:
            for name in MODELS + ["F-all"]:
                for c in tg.reg:
                    row, _, _ = eval_metrics(P[mode][name][c], tgt["y"], te, tg.actor, 4,
                                             seed_of(f"x4x|{direction}|{mode}|{name}|{c}"))
                    out_rows.append(dict(direction=direction, setting=f"cross-{mode}", target=t_name,
                                         model=name, condition=c, **row))
        info[direction] = dict(T={f"{m}:{t}": float(res[(m, t, 0)]["T"]) for m in MODES for t in MODELS},
                               C={f"{m}:{t}": float(res[(m, t, 0)]["C"]) for m in MODES for t in MODELS})
        ms, _ = monitor_shift(src["pack"], tg, label=direction)
        shifts.append(ms)
    for name in ("ravdess", "cremad", "mosei"):
        task = tasks[name]
        tg = task["pack"]
        res = fit_within(task)
        P, _ = assemble(tg.reg, res, [te for _, te in task["folds"]], tg.n, 4)
        ev = np.sort(np.concatenate([te for _, te in task["folds"]]))
        for mode in MODES:
            for name2 in MODELS + ["F-all"]:
                for c in tg.reg:
                    row, _, _ = eval_metrics(P[mode][name2][c], task["y"], ev, tg.actor, 4,
                                             seed_of(f"x4w|{name}|{mode}|{name2}|{c}"))
                    out_rows.append(dict(direction=f"within:{task['name']}", setting=f"within-{mode}",
                                         target=name, model=name2, condition=c, **row))
    return pd.DataFrame(out_rows), info, pd.concat(shifts, ignore_index=True)


# ----------------------------------------------------------------------------
# X3 / X4 monitors on streams
# ----------------------------------------------------------------------------
def load_bank(pk, clean_only=False):
    """Content (prosody / mnet) and quality-descriptor (qda / qdv) features per modality condition."""
    bank = {}
    for ch, conds, names in (("audio", pk.acond, ("prosody", "qda")), ("visual", pk.vcond, ("mnet", "qdv"))):
        cs = ["clean"] if clean_only else conds
        for kind, nm in zip(("CONTENT", "QD"), names):
            bank[(kind, ch)] = {mc: np.load(pk.feat / f"{nm}_{mc}.npy").astype(np.float64) for mc in cs}
    return bank


def zscore(ref, X):
    lq = np.log(Q.monitor_d(ref["model"], X) / ref["theta"])
    return (lq - ref["mu"]) / ref["sd"]


def build_ctx(pk, k, spk=True):
    """Monitor references for fold k (both modes): clip-level CONTENT / QD and speaker-adaptive SPK."""
    tr, _ = pk.folds_full[k]
    actor = pk.actor
    ctx = SimpleNamespace(bank=load_bank(pk), zc={"clean": {}, "env": {}}, spk={"clean": {}, "env": {}})
    for ch in CH:
        mc_tr = RE.TRAIN_MC[ch]
        for kind in ("CONTENT", "QD"):
            X = ctx.bank[(kind, ch)]
            m = Q.monitor_fold(X["clean"], tr, actor)
            ref_c = dict(model=m["model"], theta=m["theta"]["p99"], mu=m["oof_mu"], sd=m["oof_sd"])
            e = RE.envelope_fold(X["clean"], X[mc_tr], tr, actor)
            ref_e = dict(model=e["model"], theta=e["theta"], mu=e["mu"], sd=e["sd"])
            ctx.zc["clean"][(kind, ch)] = {mc: zscore(ref_c, X[mc]) for mc in X}
            ctx.zc["env"][(kind, ch)] = {mc: zscore(ref_e, X[mc]) for mc in X}
        if not spk:
            continue
        Fc = ctx.bank[("CONTENT", ch)]["clean"]
        Fd = ctx.bank[("CONTENT", ch)][mc_tr]
        floor = 1e-3 * Fc[tr].std(0)
        floor = np.where(floor > 0, floor, 1e-12)
        Xn_c, Xn_d = np.zeros_like(Fc), np.zeros_like(Fd)
        for a in np.unique(actor[tr]):
            ca = tr[actor[tr] == a]
            order = np.random.default_rng([SEED, 101, k, int(a)]).permutation(ca)
            ec, ed = Fc[order[:SPK_ENROL]], Fd[order[:SPK_ENROL]]
            Xn_c[ca] = (Fc[ca] - ec.mean(0)) / np.maximum(ec.std(0, ddof=1), floor)
            Xn_d[ca] = (Fd[ca] - ed.mean(0)) / np.maximum(ed.std(0, ddof=1), floor)
        m = Q.monitor_fold(Xn_c, tr, actor)
        ctx.spk["clean"][ch] = dict(model=m["model"], theta=m["theta"]["p99"], mu=m["oof_mu"],
                                    sd=m["oof_sd"], floor=floor)
        e = RE.envelope_fold(Xn_c, Xn_d, tr, actor)
        ctx.spk["env"][ch] = dict(model=e["model"], theta=e["theta"], mu=e["mu"], sd=e["sd"], floor=floor)
    return ctx


def gather(table, mcp, rows):
    out = None
    for mc in np.unique(mcp):
        m = mcp == mc
        v = table[mc][rows[m]]
        if out is None:
            out = np.empty((len(rows),) + v.shape[1:], dtype=v.dtype)
        out[m] = v
    return out


def blocked_stream_mosei(pk, te, k, s):
    """MOSEI S4 stream: the fold's videos in random order, each video's clips contiguous (the recording
    changes every 1-3 clips). If one pass gives fewer than 480 clips, the videos are drawn again in a second
    random order (the fold's clips repeat); the stream is cut at 480 clips. Returns clips and video bounds."""
    rng = np.random.default_rng([SEED, 83, k, s])
    vids = np.unique(pk.actor[te])
    parts, sizes, total = [], [], 0
    while total < MOSEI_STREAM:
        for v in rng.permutation(vids):
            cv = rng.permutation(te[pk.actor[te] == v])
            parts.append(cv)
            sizes.append(len(cv))
            total += len(cv)
    clips = np.concatenate(parts)[:MOSEI_STREAM]
    bounds, start = [], 0
    for n in sizes:
        if start >= MOSEI_STREAM:
            break
        end = min(start + n, MOSEI_STREAM)
        bounds.append((start, end))
        start = end
    return clips, bounds


def blocked_stream(pk, te, k, s):
    """Speaker-blocked clean stream of 480 clips with session bounds (start, end)."""
    if pk.name == "mosei":
        return blocked_stream_mosei(pk, te, k, s)
    rng = np.random.default_rng([SEED, 81, k, s])
    if pk.name == "ravdess":
        acts1, p1 = I.block_presentation(te, pk.actor, rng)
        acts2, p2 = I.block_presentation(te, pk.actor, rng, avoid=acts1)
        clips = np.concatenate([p1, p2])
        sizes = [int(np.sum(pk.actor[te] == a)) for a in list(acts1) + list(acts2)]
    else:
        acts = np.unique(pk.actor[te])
        order = []
        while len(order) < CREMA_SESSIONS:
            order.extend(rng.permutation(acts).tolist())
        parts, sizes = [], []
        for a in order[:CREMA_SESSIONS]:
            ca = te[pk.actor[te] == a]
            parts.append(rng.choice(ca, CREMA_PER_SESSION, replace=len(ca) < CREMA_PER_SESSION))
            sizes.append(CREMA_PER_SESSION)
        clips = np.concatenate(parts)
    bounds, start = [], 0
    for n in sizes:
        bounds.append((start, start + n))
        start += n
    assert start == len(clips)
    return clips, bounds


def make_stream(pk, scn, cond, k, s, clips, conds, onset, sess):
    CONDS = list(pk.reg.keys())
    mc = {ch: np.array([pk.reg[c][i] for c in CONDS])[conds] for i, ch in enumerate(CH)}
    single = None
    if scn != "S4":
        ac, vc = pk.reg[cond]
        if ac != "clean" and vc == "clean":
            single = "audio"
        elif vc != "clean" and ac == "clean":
            single = "visual"
    return dict(scn=scn, cond=cond, fold=k, seed=s, clips=clips, conds=conds, onset=onset, sess=sess,
                mc=mc, single=single)


def build_streams(pk, k, seeds):
    CONDS = list(pk.reg.keys())
    _, te = pk.folds_full[k]
    out = []
    for s in range(seeds):
        for ci in range(1, len(CONDS)):
            r1 = np.random.default_rng([SEED, 61, k, s, ci])
            p1, p2 = r1.permutation(te), r1.permutation(te)
            h = min(HALF, len(te))
            clips = np.concatenate([p1[:h], p2[:h]])
            conds = np.r_[np.zeros(h, int), np.full(h, ci)]
            out.append(make_stream(pk, "S1", CONDS[ci], k, s, clips, conds, h, None))
        clips, sess = blocked_stream(pk, te, k, s)
        T = len(clips)
        out.append(make_stream(pk, "S4", "video-blocked" if pk.name == "mosei" else "speaker-blocked", k, s,
                               clips, np.zeros(T, int), None, sess))
        for ci in range(1, len(CONDS)):
            onset = int(np.random.default_rng([SEED, 91, k, s, ci]).integers(ONSET_LO, ONSET_HI + 1))
            conds = np.where(np.arange(T) >= onset, ci, 0)
            out.append(make_stream(pk, "S5", CONDS[ci], k, s, clips, conds, onset, sess))
    return out


def spk_stream_z(ctx, st, ch, mode):
    """Speaker-adaptive z: per session, features standardised with the first 10 clips of that session;
    only scored clips (after enrolment) get a value, the rest are NaN."""
    clips, T = st["clips"], len(st["clips"])
    bank = ctx.bank[("CONTENT", ch)]
    F = gather(bank, st["mc"][ch], clips)
    ref = ctx.spk[mode][ch]
    Fn = np.empty_like(F)
    idx = []
    for a, b in st["sess"]:
        enr = F[a:a + SPK_ENROL]
        Fn[a:b] = (F[a:b] - enr.mean(0)) / np.maximum(enr.std(0, ddof=1), ref["floor"])
        idx.append(np.arange(a + SPK_ENROL, b))
    idx = np.concatenate(idx)
    z = np.full(T, np.nan)
    d = Q.monitor_d(ref["model"], Fn[idx])
    z[idx] = (np.log(d / ref["theta"]) - ref["mu"]) / ref["sd"]
    return z


def traces(ctx, st, mon, mode):
    """Channel states in effect per clip (bool arrays) for monitor mon in reference mode."""
    T = len(st["clips"])
    if mon in ("CONTENT", "QD"):
        zA = gather(ctx.zc[mode][(mon, "audio")], st["mc"]["audio"], st["clips"])
        zV = gather(ctx.zc[mode][(mon, "visual")], st["mc"]["visual"], st["clips"])
        eff, _, _, _ = S.run_slg({"audio": zA, "visual": zV})
        return eff["audio"], eff["visual"]
    if st["sess"] is None:
        return None
    zs = {ch: spk_stream_z(ctx, st, ch, mode) for ch in CH}
    eA, eV = np.zeros(T, bool), np.zeros(T, bool)
    for a, b in st["sess"]:
        a2 = a + SPK_ENROL
        if a2 >= b:
            continue
        seg = slice(a2, b)
        eff, _, _, _ = S.run_slg({"audio": zs["audio"][seg], "visual": zs["visual"][seg]})
        eA[seg], eV[seg] = eff["audio"], eff["visual"]
    if mon == "SPK":
        return eA, eV
    qA, qV = traces(ctx, st, "QD", mode)
    return eA | qA, eV | qV


def rising(e):
    prev = np.concatenate([[False], e[:-1]])
    return np.flatnonzero(e & ~prev)


def deg_truth(st):
    T = len(st["clips"])
    pos = np.arange(T)
    out = {}
    for ch in CH:
        d = st["mc"][ch] != "clean"
        out[ch] = d & (pos >= st["onset"]) if st["onset"] is not None else np.zeros(T, bool)
    return out


def stream_metrics(eA, eV, deg, onset, single):
    eff = {"audio": eA, "visual": eV}
    marks = {ch: rising(eff[ch]) for ch in CH}
    false = sum(int(np.sum(~deg[ch][marks[ch]])) for ch in CH)
    dpos = np.flatnonzero((eA & deg["audio"]) | (eV & deg["visual"]))
    post = []
    if onset is not None:
        post = sorted((int(p), ch) for ch in CH for p in marks[ch] if p >= onset)
    attrib = np.nan
    if post and single is not None:
        first = post[0][0]
        attrib = float(single in {ch for p, ch in post if p == first})
    detected = bool(dpos.size > 0) if onset is not None else False
    return dict(n_clips=len(eA), false_marks=false, false_any=bool(false > 0), detected=detected,
                delay=float(dpos[0] - onset) if detected else np.nan, has_post=bool(post), attrib=attrib)


def p99_worker(pk, k):
    """Per-clip P99 flag rate on the clean test clips of fold k (monitor fitted on the training clips)."""
    tr, te = pk.folds_full[k]
    bank = load_bank(pk, clean_only=True)
    rows = []
    for kind in ("CONTENT", "QD"):
        for ch in CH:
            X = bank[(kind, ch)]["clean"]
            m = Q.monitor_fold(X, tr, pk.actor)
            lq = np.log(Q.monitor_d(m["model"], X[te]) / m["theta"]["p99"])
            rows.append(dict(corpus=pk.name, monitor=kind, channel=ch, fold=k + 1, n_test_clips=int(len(te)),
                             flag_rate=float(np.mean(lq > 0))))
    return rows


def x3_worker(pk, k, seeds, mons=MONS):
    ctx = build_ctx(pk, k, spk=pk.name != "mosei")
    rows = []
    for st in build_streams(pk, k, seeds):
        deg = deg_truth(st)
        for mon in mons:
            if mon in SPK_MONS and st["sess"] is None:
                continue
            eA, eV = traces(ctx, st, mon, "clean")
            m = stream_metrics(eA, eV, deg, st["onset"], st["single"])
            rows.append(dict(monitor=mon, scenario=st["scn"], cond=st["cond"], fold=k + 1, seed=st["seed"], **m))
    return rows


def aggregate_x3(df, corpus):
    out = []
    for (mon, scn, cond), g in df.groupby(["monitor", "scenario", "cond"], sort=False):
        nclips = g.n_clips.sum()
        row = dict(corpus=corpus, monitor=mon, scenario=scn, condition=cond, n_streams=len(g),
                   false_marks_per_1000=float(g.false_marks.sum() / nclips * 1000),
                   share_streams_any_false=float(g.false_any.mean()))
        if scn != "S4":
            det = g[g.detected.astype(bool)]
            dl = det.delay.values
            att = g.attrib.dropna().values
            row.update(detection_rate=float(g.detected.mean()),
                       delay_median=float(np.median(dl)) if dl.size else np.nan,
                       delay_q25=float(np.percentile(dl, 25)) if dl.size else np.nan,
                       delay_q75=float(np.percentile(dl, 75)) if dl.size else np.nan,
                       attribution_share=float(att.mean()) if att.size else np.nan,
                       n_attribution=int(att.size))
        out.append(row)
    return pd.DataFrame(out)


def choose_monitor(x3_all):
    """Pre-specified rule (fixed before the CREMA-D run): score = mean S5 detection rate over degraded
    conditions minus the S4 share of streams with any false mark; highest score wins."""
    rav = x3_all[x3_all.corpus == "ravdess"]
    scores = {}
    for mon in MONS:
        g5 = rav[(rav.monitor == mon) & (rav.scenario == "S5")]
        g4 = rav[(rav.monitor == mon) & (rav.scenario == "S4")]
        if g5.empty or g4.empty:
            continue
        det = float(g5.detection_rate.mean())
        fp = float(g4.share_streams_any_false.iloc[0])
        scores[mon] = dict(mean_S5_detection=det, S4_share_any_false=fp, score=det - fp)
    best = max(scores, key=lambda m: scores[m]["score"])
    return best, scores


def x4_worker(pk, k, mon, seeds, LPS, scen=None):
    ctx = build_ctx(pk, k, spk=pk.name != "mosei")
    y = pk.y_main
    rows = []
    for st in build_streams(pk, k, seeds):
        if scen is not None and st["scn"] not in scen:
            continue
        if mon in SPK_MONS and st["sess"] is None:
            continue
        T = len(st["clips"])
        pos = np.arange(T)
        onset = st["onset"] if st["onset"] is not None else T
        cseg, dseg = pos < onset, pos >= onset
        yt = y[st["clips"]]
        for vname, models, gate in VARIANTS:
            if gate is None:
                aS = vS = np.ones(T, bool)
            else:
                eA, eV = traces(ctx, st, mon, gate)
                aS, vS = Q.active_masks(eA, eV)
            L = {t: LPS[models][t][st["conds"], st["clips"]] for t in MODELS}
            pred = (aS[:, None] * (L["A0"] + L["A1"]) + vS[:, None] * (L["V0"] + L["V1"])).argmax(1)
            cor = pred == yt
            rows.append(dict(monitor=mon, variant=vname, scenario=st["scn"], condition=st["cond"],
                             fold=k + 1, seed=st["seed"], n_clean=int(cseg.sum()),
                             cor_clean=int(cor[cseg].sum()), n_deg=int(dseg.sum()),
                             cor_deg=int(cor[dseg].sum()), gA_clean=int((~aS[cseg]).sum()),
                             gV_clean=int((~vS[cseg]).sum()),
                             gAny_clean=int((~(aS & vS))[cseg].sum())))
    return rows


def aggregate_x4(df, corpus):
    g = df.groupby(["monitor", "variant", "scenario", "condition"], sort=False)
    out = g.agg(n_streams=("seed", "size"), n_clean=("n_clean", "sum"), cor_clean=("cor_clean", "sum"),
                n_deg=("n_deg", "sum"), cor_deg=("cor_deg", "sum"), gA_clean=("gA_clean", "sum"),
                gV_clean=("gV_clean", "sum"), gAny_clean=("gAny_clean", "sum")).reset_index()
    out.insert(0, "corpus", corpus)
    out["acc_clean_segment"] = out.cor_clean / out.n_clean.replace(0, np.nan)
    out["acc_degraded_segment"] = out.cor_deg / out.n_deg.replace(0, np.nan)
    out["gated_audio_clean"] = out.gA_clean / out.n_clean.replace(0, np.nan)
    out["gated_visual_clean"] = out.gV_clean / out.n_clean.replace(0, np.nan)
    out["gated_any_clean"] = out.gAny_clean / out.n_clean.replace(0, np.nan)
    return out.drop(columns=["cor_clean", "cor_deg", "gA_clean", "gV_clean", "gAny_clean"])


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def style():
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "legend.fontsize": 6,
                         "xtick.labelsize": 6, "ytick.labelsize": 6, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42, "savefig.bbox": "tight"})


def make_figures():
    style()
    CFG.fig.mkdir(parents=True, exist_ok=True)
    rd = CFG.res

    def load(name):
        p = rd / name
        return pd.read_csv(p) if p.exists() else None

    # ext_crosscorpus: clean accuracy, within-corpus vs cross-corpus, both directions
    x2 = load("ext_x2_crosscorpus.csv")
    if x2 is not None:
        x2 = x2[x2.condition == "clean"]
        targets = [t for t in ("cremad", "ravdess") if (x2.target == t).any()]
        if targets:
            fig, axes = plt.subplots(len(targets), 1, figsize=(3.4, 2.1 * len(targets)), squeeze=False)
            for ax, t in zip(axes[:, 0], targets):
                wit = "within:cremad6" if t == "cremad" else "within:ravdess6"
                src = "ravdess->cremad" if t == "cremad" else "cremad->ravdess"
                series = [("within-corpus", x2[(x2.direction == wit) & (x2.setting == "within-clean")], PAL["blue"]),
                          ("cross, clean-trained", x2[(x2.direction == src) & (x2.setting == "cross-clean")], PAL["orange"]),
                          ("cross, MCT", x2[(x2.direction == src) & (x2.setting == "cross-mct")], PAL["green"])]
                models = ["A1", "V1", "F-all"]
                w = 0.26
                for i, (lab, d, col) in enumerate(series):
                    vals, lo, hi = [], [], []
                    for m in models:
                        r = d[d.model == m]
                        acc = float(r.accuracy.iloc[0]) if len(r) else np.nan
                        vals.append(acc)
                        lo.append(acc - float(r.acc_ci_low.iloc[0]) if len(r) else 0)
                        hi.append(float(r.acc_ci_high.iloc[0]) - acc if len(r) else 0)
                    xs = np.arange(len(models)) + (i - 1) * w
                    ax.bar(xs, vals, w, color=col, label=lab, yerr=[lo, hi], capsize=1.5, error_kw=dict(lw=0.6))
                chance = 1 / 6
                ax.axhline(chance, color=PAL["grey"], ls=":", lw=0.8)
                ax.set_xticks(np.arange(len(models)))
                ax.set_xticklabels(models)
                ax.set_ylim(0, 1)
                ax.set_ylabel("accuracy (clean)")
                ax.set_title(f"target: {'CREMA-D' if t == 'cremad' else 'RAVDESS'} (6 classes)")
                ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
            fig.tight_layout()
            fig.savefig(CFG.fig / "ext_crosscorpus.pdf")
            plt.close(fig)
            log("wrote ext_crosscorpus.pdf")

    # ext_monitors: detection (S5, mean over degraded conditions) vs false marks (S4)
    x3 = load("ext_x3_monitors.csv")
    if x3 is not None:
        corpora = [c for c in ("ravdess", "cremad") if (x3.corpus == c).any()]
        if corpora:
            fig, axes = plt.subplots(len(corpora), 1, figsize=(3.4, 2.3 * len(corpora)), squeeze=False)
            styles = {"CONTENT": (PAL["blue"], "o"), "QD": (PAL["orange"], "s"),
                      "SPK": (PAL["green"], "^"), "QD+SPK": (PAL["purple"], "D")}
            for ax, c in zip(axes[:, 0], corpora):
                for mon in MONS:
                    g5 = x3[(x3.corpus == c) & (x3.monitor == mon) & (x3.scenario == "S5")]
                    g4 = x3[(x3.corpus == c) & (x3.monitor == mon) & (x3.scenario == "S4")]
                    if g5.empty or g4.empty:
                        continue
                    col, mk = styles[mon]
                    xv = float(g4.false_marks_per_1000.iloc[0])
                    yv = float(g5.detection_rate.mean())
                    ax.scatter(xv, yv, color=col, marker=mk, s=28, label=mon, zorder=3)
                ax.set_xlabel("false marks per 1000 clips (S4, speaker-blocked clean)")
                ax.set_ylabel("detection rate (S5, mean)")
                ax.set_ylim(0, 1.02)
                ax.set_title("RAVDESS" if c == "ravdess" else "CREMA-D")
                ax.grid(alpha=0.3, lw=0.4)
            axes[0, 0].legend(frameon=False, ncol=2, loc="lower right")
            fig.tight_layout()
            fig.savefig(CFG.fig / "ext_monitors.pdf")
            plt.close(fig)
            log("wrote ext_monitors.pdf")

    # ext_realistic: F-all accuracy under realistic degradations, clean-trained vs MCT
    x1 = load("ext_x1_within.csv")
    if x1 is not None:
        corpora = [c for c in ("ravdess", "cremad") if (x1.corpus == c).any()]
        if corpora:
            conds = ["clean", "cafe5", "street5", "h264"]
            fig, axes = plt.subplots(len(corpora), 1, figsize=(3.4, 2.1 * len(corpora)), squeeze=False)
            for ax, c in zip(axes[:, 0], corpora):
                w = 0.36
                for i, (mode, col, lab) in enumerate([("clean", PAL["blue"], "clean-trained"),
                                                       ("mct", PAL["vermil"], "MCT")]):
                    vals, lo, hi = [], [], []
                    for cond in conds:
                        r = x1[(x1.corpus == c) & (x1.training == mode) & (x1.model == "F-all") & (x1.condition == cond)]
                        if r.empty:
                            vals.append(np.nan); lo.append(0); hi.append(0)
                            continue
                        a = float(r.accuracy.iloc[0])
                        vals.append(a)
                        lo.append(a - float(r.acc_ci_low.iloc[0]))
                        hi.append(float(r.acc_ci_high.iloc[0]) - a)
                    xs = np.arange(len(conds)) + (i - 0.5) * w
                    ax.bar(xs, vals, w, color=col, label=lab, yerr=[lo, hi], capsize=1.5,
                           error_kw=dict(lw=0.6), hatch="//" if mode == "mct" else None, edgecolor="white")
                ax.set_xticks(np.arange(len(conds)))
                ax.set_xticklabels(conds)
                ax.set_ylim(0, 1)
                ax.set_ylabel("F-all accuracy")
                ax.set_title("RAVDESS (8 classes)" if c == "ravdess" else "CREMA-D (6 classes)")
                ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.14))
            fig.tight_layout()
            fig.savefig(CFG.fig / "ext_realistic.pdf")
            plt.close(fig)
            log("wrote ext_realistic.pdf")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def make_mosei_figures():
    """ext_mosei.pdf: F-all accuracy by condition (clean-trained vs MCT) and monitor detection vs false marks.
    ext_crosscorpus_mosei.pdf: four-class cross-corpus accuracy with the within-corpus comparators."""
    style()
    CFG.fig.mkdir(parents=True, exist_ok=True)

    def load(name):
        p = CFG.res / name
        return pd.read_csv(p) if p.exists() else None

    x1 = load("ext_mosei_x1_within.csv")
    x3 = load("ext_mosei_x3_monitors.csv")
    if x1 is not None:
        conds = [c for c in MOSEI_COND_ORDER if (x1.condition == c).any()]
        npan = 2 if x3 is not None else 1
        fig, axes = plt.subplots(npan, 1, figsize=(3.4, 2.3 * npan + 0.3), squeeze=False)
        ax = axes[0, 0]
        w = 0.38
        for i, (mode, col, lab) in enumerate([("clean", PAL["blue"], "clean-trained"), ("mct", PAL["vermil"], "MCT")]):
            vals, lo, hi = [], [], []
            for cond in conds:
                r = x1[(x1.training == mode) & (x1.model == "F-all") & (x1.condition == cond)]
                a = float(r.accuracy.iloc[0]) if len(r) else np.nan
                vals.append(a)
                lo.append(a - float(r.acc_ci_low.iloc[0]) if len(r) else 0)
                hi.append(float(r.acc_ci_high.iloc[0]) - a if len(r) else 0)
            xs = np.arange(len(conds)) + (i - 0.5) * w
            ax.bar(xs, vals, w, color=col, label=lab, yerr=[lo, hi], capsize=1.2, error_kw=dict(lw=0.5),
                   hatch="//" if mode == "mct" else None, edgecolor="white")
        ax.axhline(0.25, color=PAL["grey"], ls=":", lw=0.8)
        ax.set_xticks(np.arange(len(conds)))
        ax.set_xticklabels(conds, rotation=40, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel("F-all accuracy (4 classes)")
        ax.set_title("CMU-MOSEI: clean-trained vs MCT")
        ax.legend(frameon=False, ncol=2, loc="upper right")
        if x3 is not None:
            ax = axes[1, 0]
            styles = {"CONTENT": (PAL["blue"], "o"), "QD": (PAL["orange"], "s")}
            for mon in MOSEI_MONS:
                g5 = x3[(x3.monitor == mon) & (x3.scenario == "S5")]
                g4 = x3[(x3.monitor == mon) & (x3.scenario == "S4")]
                if g5.empty or g4.empty:
                    continue
                col, mk = styles[mon]
                ax.scatter(float(g4.false_marks_per_1000.iloc[0]), float(g5.detection_rate.mean()),
                           color=col, marker=mk, s=28, label=mon, zorder=3)
            ax.set_xlabel("false marks per 1000 clips (S4, video-blocked clean)")
            ax.set_ylabel("detection rate (S5, mean)")
            ax.set_ylim(0, 1.02)
            ax.grid(alpha=0.3, lw=0.4)
            ax.legend(frameon=False, loc="lower right")
        fig.tight_layout()
        fig.savefig(CFG.fig / "ext_mosei.pdf")
        plt.close(fig)
        log("wrote ext_mosei.pdf")

    x2 = load("ext_mosei_x2_crosscorpus.csv")
    if x2 is not None:
        panels = [
            ("target: CMU-MOSEI", [("within-corpus", "within:mosei4", "within-clean", PAL["blue"]),
                                   ("from RAVDESS", "ravdess->mosei", "cross-clean", PAL["orange"]),
                                   ("from CREMA-D", "cremad->mosei", "cross-clean", PAL["green"])]),
            ("target: RAVDESS (4 classes)", [("within-corpus", "within:ravdess4", "within-clean", PAL["blue"]),
                                             ("from MOSEI", "mosei->ravdess", "cross-clean", PAL["orange"])]),
            ("target: CREMA-D (4 classes)", [("within-corpus", "within:cremad4", "within-clean", PAL["blue"]),
                                             ("from MOSEI", "mosei->cremad", "cross-clean", PAL["orange"])]),
        ]
        models = ["A1", "V1", "F-all"]
        fig, axes = plt.subplots(len(panels), 1, figsize=(3.4, 2.0 * len(panels) + 0.4), squeeze=False)
        for ax, (title, series) in zip(axes[:, 0], panels):
            w = 0.8 / len(series)
            for i, (lab, direction, setting, col) in enumerate(series):
                d = x2[(x2.direction == direction) & (x2.setting == setting) & (x2.condition == "clean")]
                vals, lo, hi = [], [], []
                for m in models:
                    r = d[d.model == m]
                    a = float(r.accuracy.iloc[0]) if len(r) else np.nan
                    vals.append(a)
                    lo.append(a - float(r.acc_ci_low.iloc[0]) if len(r) else 0)
                    hi.append(float(r.acc_ci_high.iloc[0]) - a if len(r) else 0)
                xs = np.arange(len(models)) + (i - (len(series) - 1) / 2) * w
                ax.bar(xs, vals, w, color=col, label=lab, yerr=[lo, hi], capsize=1.2, error_kw=dict(lw=0.5))
            ax.axhline(0.25, color=PAL["grey"], ls=":", lw=0.8)
            ax.set_xticks(np.arange(len(models)))
            ax.set_xticklabels(models)
            ax.set_ylim(0, 1)
            ax.set_ylabel("accuracy (clean)")
            ax.set_title(title)
            ax.legend(frameon=False, ncol=len(series), loc="upper center", bbox_to_anchor=(0.5, -0.12))
        fig.tight_layout()
        fig.savefig(CFG.fig / "ext_crosscorpus_mosei.pdf")
        plt.close(fig)
        log("wrote ext_crosscorpus_mosei.pdf")


def out_name(corpus, base):
    """Results file for a corpus: RAVDESS and CREMA-D keep the ext_* names, MOSEI writes ext_mosei_*."""
    return base if corpus != "mosei" else base.replace("ext_", "ext_mosei_", 1)


def summary_name(corpus):
    return "ext_summary.json" if corpus != "mosei" else "ext_mosei_summary.json"


def update_summary(section, corpus, value, name="ext_summary.json"):
    path = CFG.res / name
    data = json.loads(path.read_text()) if path.exists() else {}
    data.setdefault(section, {})[corpus] = Q.to_py(value)
    path.write_text(json.dumps(data, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", choices=["ravdess", "cremad", "mosei", "both", "all"], default="ravdess")
    ap.add_argument("--part", choices=["x1", "x2", "x3", "x4", "all"], default="all")
    ap.add_argument("--seeds", type=int, default=R.N_DRIFT_SEEDS)
    ap.add_argument("--n-jobs", type=int, default=2)
    ap.add_argument("--feat-ravdess", type=Path, default=CFG.feat_ravdess)
    ap.add_argument("--feat-cremad", type=Path, default=CFG.feat_cremad)
    ap.add_argument("--feat-mosei", type=Path, default=CFG.feat_mosei)
    ap.add_argument("--cache-dir", type=Path, default=CFG.cache)
    ap.add_argument("--results-dir", type=Path, default=CFG.res)
    ap.add_argument("--fig-dir", type=Path, default=CFG.fig)
    args = ap.parse_args()
    CFG.feat_ravdess, CFG.feat_cremad, CFG.feat_mosei = args.feat_ravdess, args.feat_cremad, args.feat_mosei
    CFG.cache, CFG.res, CFG.fig = args.cache_dir, args.results_dir, args.fig_dir
    CFG.seeds, CFG.n_jobs = args.seeds, args.n_jobs
    CFG.cache.mkdir(parents=True, exist_ok=True)
    CFG.res.mkdir(parents=True, exist_ok=True)
    log(f"corpus={args.corpus} part={args.part} seeds={CFG.seeds} n_jobs={CFG.n_jobs}")

    corpora = {"both": ["ravdess", "cremad"], "all": ["ravdess", "cremad", "mosei"]}.get(args.corpus, [args.corpus])
    feats = {"ravdess": CFG.feat_ravdess, "cremad": CFG.feat_cremad, "mosei": CFG.feat_mosei}
    for n in corpora:
        if not (feats[n] / "meta.csv").exists():
            raise SystemExit(f"{n} features not found at {feats[n]} (meta.csv missing); nothing was written")
    parts = ["x1", "x2", "x3", "x4"] if args.part == "all" else [args.part]
    packs = {}
    for name in corpora:
        packs[name] = make_pack(name)
        pk = packs[name]
        log(f"{name}: {pk.n} clips, conditions audio={pk.acond} visual={pk.vcond}, "
            f"regs={list(pk.reg)}, cost F-all={pk.cost['C_ALL']:.4f}s")
    if "ravdess" in packs:
        # the cost mirror must reproduce R.cost_tables on the real RAVDESS features
        if CFG.feat_ravdess == R.FEAT:
            _, COST, _, _ = R.cost_tables(pd.read_csv(R.FEAT / "meta.csv"))
            chk = abs(COST["F-all (all components)"] - packs["ravdess"].cost["C_ALL"])
            assert chk < 1e-12, "cost mirror differs from R.cost_tables"
            log(f"cost mirror matches R.cost_tables (diff {chk:.2e})")
    t_start = time.perf_counter()

    tasks = {n: make_tasks(packs[n]) for n in corpora}
    if "x1" in parts:
        for n in corpora:
            tk = tasks[n]
            x1, fus, casc, gap, summ = run_x1(tk)
            save_rows(x1, out_name(n, "ext_x1_within.csv"), ["corpus", "task", "training", "model", "condition"])
            save_rows(fus, out_name(n, "ext_x1_fusion.csv"), ["corpus", "training", "condition"])
            save_rows(casc, out_name(n, "ext_x1_cascade.csv"), ["corpus", "condition"])
            if n != "mosei":
                save_rows(gap, "ext_x1_fairness.csv", ["corpus", "training", "condition"])
            update_summary("x1", n, summ, summary_name(n))
            if n == "ravdess" and CFG.feat_ravdess == R.FEAT:
                ref = pd.read_csv(R.RES / "main_results.csv").set_index("system")
                ours = float(x1[(x1.training == "clean") & (x1.model == "F-all") & (x1.condition == "clean")].accuracy.iloc[0])
                log(f"check: clean F-all accuracy {ours:.6f} vs main_results {ref.loc['F-all (LF)', 'accuracy']:.6f}")
    if "x2" in parts:
        if "ravdess" in corpora and "cremad" in corpora:
            x2, cinfo = run_cross_block(ravdess6_task(packs["ravdess"]), tasks["cremad"])
            save_rows(x2, "ext_x2_crosscorpus.csv", ["direction"])
            ms, pf = monitor_shift(packs["ravdess"], packs["cremad"])
            ms2, pf2 = monitor_shift(packs["cremad"], packs["ravdess"])
            save_rows(pd.concat([ms, ms2], ignore_index=True), "ext_x2_monitor_shift.csv", ["direction", "monitor", "channel"])
            update_summary("x2", "cross", cinfo)
        elif "ravdess" in corpora:
            log("X2: cross-corpus rows need both corpora; computing the RAVDESS six-class comparator only")
            tk = ravdess6_task(packs["ravdess"])
            res = fit_within(tk)
            P, _ = assemble(packs["ravdess"].reg, res, [te for _, te in tk["folds"]], packs["ravdess"].n, 6)
            ev = np.sort(np.concatenate([te for _, te in tk["folds"]]))
            rows = []
            for mode in MODES:
                for name in ["A1", "V1", "F-all"]:
                    for c in packs["ravdess"].reg:
                        row, _, _ = eval_metrics(P[mode][name][c], tk["y"], ev, packs["ravdess"].actor, 6,
                                                 seed_of(f"x2w|ravdess6|{mode}|{name}|{c}"))
                        rows.append(dict(direction="within:ravdess6", setting=f"within-{mode}",
                                         target="ravdess", model=name, condition=c, **row))
            save_rows(pd.DataFrame(rows), "ext_x2_crosscorpus.csv", ["direction"])
        elif "mosei" not in corpora:
            log("X2 needs the RAVDESS corpus (comparator and cross-corpus rows); skipped")
        if "mosei" in corpora:
            for n in ("ravdess", "cremad"):
                if n not in packs:
                    if not (feats[n] / "meta.csv").exists():
                        raise SystemExit(f"X2 on MOSEI needs {n} features at {feats[n]}; nothing was written")
                    packs[n] = make_pack(n)
            x2m, cinfo_m, msm = run_cross_mosei(packs)
            save_rows(x2m, "ext_mosei_x2_crosscorpus.csv", ["direction"])
            save_rows(msm, "ext_mosei_x2_monitor_shift.csv", ["direction", "monitor", "channel"])
            update_summary("x2", "mosei", cinfo_m, "ext_mosei_summary.json")
    if "x3" in parts:
        for n in corpora:
            pk = packs[n]
            mons = MOSEI_MONS if n == "mosei" else MONS
            log(f"X3 {n}: streams and monitors on {len(pk.folds_full)} folds, {CFG.seeds} seeds, monitors {mons}")
            res_rows = Parallel(n_jobs=CFG.n_jobs)(delayed(x3_worker)(pk, k, CFG.seeds, mons)
                                                   for k in range(len(pk.folds_full)))
            df = pd.DataFrame([r for rows in res_rows for r in rows])
            stream_file = "ext_mosei_x3_streams.csv.gz" if n == "mosei" else f"ext_x3_streams_{n}.csv.gz"
            df.to_csv(CFG.res / stream_file, index=False, compression="gzip")
            agg = aggregate_x3(df, n)
            save_rows(agg, out_name(n, "ext_x3_monitors.csv"), ["corpus"])
            if n == "mosei":
                flag_rows = Parallel(n_jobs=CFG.n_jobs)(delayed(p99_worker)(pk, k)
                                                        for k in range(len(pk.folds_full)))
                save_rows(pd.DataFrame([r for rows in flag_rows for r in rows]), "ext_mosei_x3_p99_flags.csv",
                          ["corpus", "monitor", "channel", "fold"])
                vid_sizes = pd.Series(pk.actor).value_counts()
                info = dict(seeds_per_fold=CFG.seeds, folds=len(pk.folds_full), monitors=mons,
                            spk="not applicable: no speaker sessions; videos with fewer than 12 clips cannot be enrolled",
                            videos=int(len(vid_sizes)), videos_under_12_clips=int((vid_sizes < 12).sum()),
                            n_videos_per_fold=[int(len(np.unique(pk.actor[te]))) for _, te in pk.folds_full])
            else:
                info = dict(seeds_per_fold=CFG.seeds, folds=len(pk.folds_full),
                            test_speakers_under_24_clips_per_fold=pk.under24)
            if n == "ravdess":
                best, scores = choose_monitor(pd.read_csv(CFG.res / "ext_x3_monitors.csv"))
                (CFG.res / "ext_x3_choice.json").write_text(json.dumps(
                    dict(chosen=best, rule="score = mean S5 detection rate over degraded conditions "
                         "minus S4 share of streams with any false mark", scores=scores), indent=1))
                log(f"X3 monitor chosen on RAVDESS: {best} {scores}")
            update_summary("x3", n, info, summary_name(n))
    if "x4" in parts:
        mon_rav = None
        if any(n != "mosei" for n in corpora):
            choice_path = CFG.res / "ext_x3_choice.json"
            if not choice_path.exists():
                raise SystemExit("X4 needs ext_x3_choice.json (run X3 on RAVDESS first)")
            mon_rav = json.loads(choice_path.read_text())["chosen"]
            log(f"X4 gating with monitor {mon_rav}")
        for n in corpora:
            pk = packs[n]
            tk = tasks[n]
            mon = "QD" if n == "mosei" else mon_rav
            scen = ("S1", "S4") if n == "mosei" else None
            log(f"X4 {n}: gating with monitor {mon}" + (f", scenarios {scen}" if scen else ""))
            res = fit_within(tk)
            P, _ = assemble(pk.reg, res, [te for _, te in tk["folds"]], pk.n, tk["K"])
            LPS = {m: {t: np.stack([np.log(np.clip(P[m][t][c], 1e-12, 1.0)) for c in pk.reg]) for t in MODELS}
                   for m in MODES}
            pk.y_main = tk["y"]
            out = Parallel(n_jobs=CFG.n_jobs)(delayed(x4_worker)(pk, k, mon, CFG.seeds, LPS, scen)
                                              for k in range(len(pk.folds_full)))
            df = pd.DataFrame([r for rows in out for r in rows])
            agg = aggregate_x4(df, n)
            save_rows(agg, out_name(n, "ext_x4_gating.csv"), ["corpus"])
            info = dict(monitor=mon, streams_per_variant=int((df.variant == VARIANTS[0][0]).sum()),
                        variants=[v[0] for v in VARIANTS])
            if scen:
                info["scenarios"] = list(scen)
            update_summary("x4", n, info, summary_name(n))
    log(f"parts done in {time.perf_counter() - t_start:.1f}s")
    if any(n in BASE_CORPORA for n in corpora):
        make_figures()
    if "mosei" in corpora:
        make_mosei_figures()


if __name__ == "__main__":
    main()
