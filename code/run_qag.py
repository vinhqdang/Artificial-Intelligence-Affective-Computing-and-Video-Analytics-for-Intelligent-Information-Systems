#!/usr/bin/env python3
"""Quality-aware gating (QAG): design iteration 2 of the audio-visual emotion cascade.

Reuses the cached per-fold models and the helper functions of code/run_experiments.py
(import guarded by its __main__ check). Run from the repository root:

    python3 code/run_qag.py
"""
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_experiments as R  # noqa: E402

SEED = R.SEED
FEAT, CACHE, RES, FIG = R.FEAT, R.CACHE, R.RES, R.FIG
N_CLASSES = R.N_CLASSES
N_FOLDS = R.N_FOLDS
HALF = R.HALF
CONDITIONS = R.CONDITIONS
DEG_CONDS = R.DEG_CONDS
F_ALL = R.F_ALL
PRO_CONDS = R.AUDIO_CONDS
VIS_CONDS = R.VIDEO_CONDS
CH = ("audio", "visual")
DETS = ("PH", "ADWIN")

PCA_VAR = 0.95
PCT_MAIN, PCT_SENS = 99.0, 99.9
PH_DELTA, PH_LAMBDA, PH_ALPHA = 0.005, 5.0, 1.0   # applied to the standardised log q signal
N_DRIFT_SEEDS = R.N_DRIFT_SEEDS
PH_LAMBDA_GRID = [5.0, 10.0, 20.0, 40.0]
N_TIME_REPS = 50
N_SINGLE_REPS = 3

# channel(s) whose degradation is present in each condition (ground truth for attribution)
DEGRADED_CH = {"clean": set(), "snr20": {"audio"}, "snr10": {"audio"}, "snr0": {"audio"},
               "compress": {"visual"}, "lowlight": {"visual"}, "blur": {"visual"},
               "snr10+compress": {"audio", "visual"}, "snr0+lowlight": {"audio", "visual"}}
# oracle: channel dropped when it is the only degraded one; combined conditions keep all modalities
ORACLE_DROP = {"snr20": "audio", "snr10": "audio", "snr0": "audio",
               "compress": "visual", "lowlight": "visual", "blur": "visual"}
SYSTEMS = ["F-all (LF)", "F-all (QAG)", "Oracle gate", "Cascade A (baseline)", "Cascade-QAG"]
COLOURS = {"F-all (LF)": R.PAL["blue"], "F-all (QAG)": R.PAL["orange"], "Oracle gate": R.PAL["green"],
           "Cascade A (baseline)": R.PAL["sky"], "Cascade-QAG": R.PAL["vermil"]}


# ----------------------------------------------------------------------------
# Input monitors
# ----------------------------------------------------------------------------
def fit_monitor(X):
    sc = StandardScaler().fit(X)
    pca = PCA(n_components=PCA_VAR, svd_solver="full").fit(sc.transform(X))
    return sc, pca


def monitor_d(model, X):
    """Mahalanobis distance in the retained PCA space: sum_j z_j^2 / explained_variance_j."""
    sc, pca = model
    Z = pca.transform(sc.transform(X))
    return (Z ** 2 / pca.explained_variance_).sum(axis=1)


def monitor_fold(X, tr, actor):
    """Threshold from inner GroupKFold(4) over training actors; final model on all training actors."""
    gtr = actor[tr]
    oof = np.zeros(len(tr))
    for itr, iva in GroupKFold(R.N_INNER).split(tr, groups=gtr):
        oof[iva] = monitor_d(fit_monitor(X[tr[itr]]), X[tr[iva]])
    theta = {"p99": float(np.percentile(oof, PCT_MAIN)), "p999": float(np.percentile(oof, PCT_SENS))}
    lq = np.log(oof / theta["p99"])
    return dict(model=fit_monitor(X[tr]), theta=theta, oof_mu=float(lq.mean()),
                oof_sd=float(lq.std(ddof=1)), n_oof=len(tr))


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def lf_logits(plist):
    """Late-fusion logits: sum of log probabilities (softmax of this equals the LF probabilities)."""
    return sum(R.log_probs(p) for p in plist)


def active_masks(fa, fv):
    """Audio channel active unless only it is flagged; visual likewise. Both or neither flagged -> all."""
    both_or_none = fa == fv
    return (~fa) | both_or_none, (~fv) | both_or_none


def add_cal(store, L, T, y):
    conf_b = R.softmax(L).max(1)
    conf_a = R.softmax(L / T).max(1)
    corr = (L.argmax(1) == y).astype(float)
    store["conf_b"].append(conf_b)
    store["conf_a"].append(conf_a)
    store["corr"].append(corr)


def to_py(o):
    if isinstance(o, dict):
        return {str(k): to_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_py(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------
def fig_robustness_qag(rob):
    conds = list(CONDITIONS.keys())
    fig, ax = plt.subplots(figsize=(7, 2.6))
    w = 0.16
    x = np.arange(len(conds))
    for i, s in enumerate(SYSTEMS):
        vals = [rob[(rob.condition == c) & (rob.system == s)].accuracy.values[0] for c in conds]
        ax.bar(x + (i - 2) * w, vals, w, color=COLOURS[s], label=s)
    ax.set_xticks(x)
    ax.set_xticklabels(conds, rotation=20, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 0.9)
    ax.legend(frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.13))
    R.save_fig(fig, "robustness_qag.pdf")


def fig_monitor_scores(lq_by_ch, flag_by_ch):
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.5), sharey=True)
    for ax, ch, conds, title in ((axes[0], "audio", PRO_CONDS, "Audio monitor (prosody)"),
                                 (axes[1], "visual", VIS_CONDS, "Visual monitor (mnet)")):
        data = [lq_by_ch[ch][c] for c in conds]
        bp = ax.boxplot(data, showfliers=False, patch_artist=True, widths=0.6)
        colour = R.PAL["blue"] if ch == "audio" else R.PAL["orange"]
        for patch in bp["boxes"]:
            patch.set_facecolor(colour)
            patch.set_alpha(0.35)
            patch.set_edgecolor("black")
        ax.axhline(0, color=R.PAL["vermil"], ls="--", lw=0.8)
        ax.set_xticks(np.arange(1, len(conds) + 1))
        ax.set_xticklabels([f"{c}\n{100 * flag_by_ch[ch][c]:.1f}%" for c in conds], fontsize=6)
        ax.set_xlabel("Condition (flagged clips, P99 threshold)", fontsize=6)
        ax.set_title(title)
    axes[0].set_ylabel("log q (threshold at 0)")
    R.save_fig(fig, "monitor_scores.pdf")


def fig_drift_qag(examples):
    fig, axes = plt.subplots(len(examples), 1, figsize=(3.5, 4.8), sharex=True)
    for ax, (cond, ex) in zip(axes, examples.items()):
        t = np.arange(HALF * 2)
        for i, (ch, colour) in enumerate((("audio", R.PAL["blue"]), ("visual", R.PAL["orange"]))):
            roll = pd.Series(ex["lq"][ch]).rolling(20, min_periods=1).mean().values
            ax.plot(t, roll, color=colour, lw=1, label=f"log q, {ch} monitor (rolling 20)")
            ph_t = ex["alarms"][(ch, "PH")]
            ad_t = ex["alarms"][(ch, "ADWIN")]
            ax.scatter(ph_t, [-0.45 - 0.14 * i] * len(ph_t), marker="v", s=12, color=colour)
            ax.scatter(ad_t, [-0.25 - 0.14 * i] * len(ad_t), marker="x", s=14, color=colour)
        ax.axvline(HALF, color=R.PAL["black"], ls="--", lw=0.8)
        ax.axhline(0, color=R.PAL["grey"], lw=0.6)
        ax.set_title(f"Example stream: {cond}, fold 1, seed 0", fontsize=8)
        ax.set_ylabel("log q")
        ax.set_ylim(-0.7, 1.6)
    axes[0].scatter([], [], marker="v", s=12, color="black", label="Page-Hinkley alarms")
    axes[0].scatter([], [], marker="x", s=14, color="black", label="ADWIN alarms")
    axes[0].legend(frameon=False, fontsize=5.5, loc="upper left", ncol=2)
    axes[-1].set_xlabel("Clip index in stream (change point at 240)")
    R.save_fig(fig, "drift_qag.pdf")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    t_start = time.perf_counter()
    for d in (RES, FIG):
        d.mkdir(parents=True, exist_ok=True)
    meta, y, actor, male = R.load_meta()
    N = len(y)
    splits = R.fold_splits(actor)
    cost_df, COST, n_timed, mean_dur = R.cost_tables(meta)
    comp = COST["components"]
    C_AUD, C_VID = comp["audio_decode"], comp["video_decode_face"]
    C_A0, C_A1, C_V0, C_V1 = comp["A0 prosody"], comp["A1 w2v"], comp["V0 mnet"], comp["V1 clip"]
    C_CHEAP = COST["F-cheap stage 1 (audio + video + A0 + V0)"]
    C_ALL = COST["F-all (all components)"]

    X_pro = {mc: np.load(FEAT / f"prosody_{mc}.npy").astype(np.float64) for mc in PRO_CONDS}
    X_vis = {mc: np.load(FEAT / f"mnet_{mc}.npy").astype(np.float64) for mc in VIS_CONDS}
    for X in list(X_pro.values()) + list(X_vis.values()):
        assert X.shape[0] == N and np.isfinite(X).all()
    Xmon = {"audio": X_pro, "visual": X_vis}

    res = {m: [pickle.load(open(CACHE / f"{m}_fold{k}.pkl", "rb")) for k in range(N_FOLDS)] for m in F_ALL}
    for m in F_ALL:
        for k, (tr, te) in enumerate(splits):
            assert np.array_equal(res[m][k]["train_idx"], tr) and np.array_equal(res[m][k]["test_idx"], te)

    # ---- pooled containers --------------------------------------------------------
    P = {m: {mc: np.full((N, N_CLASSES), np.nan) for mc in R.MODEL_MCONDS[m]} for m in F_ALL}
    Tc, Ta = np.full(N, np.nan), np.full(N, np.nan)
    tauQ, tauA = np.full(N, np.nan), np.full(N, np.nan)
    LQ = {ch: {mc: np.full(N, np.nan) for mc in Xmon[ch]} for ch in CH}
    LQsens = {ch: {mc: np.full(N, np.nan) for mc in Xmon[ch]} for ch in CH}
    Z = {ch: {mc: np.full(N, np.nan) for mc in Xmon[ch]} for ch in CH}
    mons = {ch: [] for ch in CH}
    mon_rows, tau_rows, temp_rows = [], [], []
    cal_oof = {name: dict(conf_b=[], conf_a=[], corr=[]) for name in ("F-cheap", "F-all")}

    # ---- 1. per outer fold: temperatures, cascade thresholds, monitors -------------
    for k, (tr, te) in enumerate(splits):
        ytr = y[tr]
        G = {m: res[m][k]["oof_probs"] for m in F_ALL}
        for m in F_ALL:
            for mc in R.MODEL_MCONDS[m]:
                P[m][mc][te] = res[m][k]["test_probs"][mc]

        Lc_oof = lf_logits([G["A0"], G["V0"]])
        La_oof = lf_logits([G[m] for m in F_ALL])
        Tc_k = R.fit_temperature(Lc_oof, ytr)
        Ta_k = R.fit_temperature(La_oof, ytr)
        Tc[te], Ta[te] = Tc_k, Ta_k
        add_cal(cal_oof["F-cheap"], Lc_oof, Tc_k, ytr)
        add_cal(cal_oof["F-all"], La_oof, Ta_k, ytr)

        conf_c = R.softmax(Lc_oof / Tc_k).max(1)
        r_q, tq, _, accF_q = R.operating_point(conf_c, Lc_oof.argmax(1), La_oof.argmax(1), ytr)
        S_A = R.stage1_probs("A", lambda m: G[m])
        FULL_oof = R.fuse_lf([G[m] for m in F_ALL])
        r_a, ta, _, accF_a = R.operating_point(S_A.max(1), S_A.argmax(1), FULL_oof.argmax(1), ytr)
        tauQ[te], tauA[te] = tq, ta
        tau_rows.append(dict(fold=k + 1, T_cheap=Tc_k, T_all=Ta_k, tau_qag=tq, oof_escalation_qag=r_q,
                             oof_acc_fall=accF_q, tau_baseline=ta, oof_escalation_baseline=r_a,
                             oof_acc_fall_baseline=accF_a))

        for ch in CH:
            mon = monitor_fold(Xmon[ch]["clean"], tr, actor)
            mons[ch].append(mon["model"])
            for mc, X in Xmon[ch].items():
                d = monitor_d(mon["model"], X[te])
                LQ[ch][mc][te] = np.log(d / mon["theta"]["p99"])
                LQsens[ch][mc][te] = np.log(d / mon["theta"]["p999"])
                Z[ch][mc][te] = (LQ[ch][mc][te] - mon["oof_mu"]) / mon["oof_sd"]
            mon_rows.append(dict(
                fold=k + 1, channel=ch, dim=int(X.shape[1]), n_components=int(mon["model"][1].n_components_),
                n_oof_train=mon["n_oof"], theta_p99=mon["theta"]["p99"], theta_p999=mon["theta"]["p999"],
                oof_logq_mean=mon["oof_mu"], oof_logq_sd=mon["oof_sd"],
                clean_test_flag_rate_p99=float(np.mean(LQ[ch]["clean"][te] > 0)),
                clean_test_flag_rate_p999=float(np.mean(LQsens[ch]["clean"][te] > 0))))
        temp_rows.append(dict(fold=k + 1, T_cheap=Tc_k, T_all=Ta_k))
    for name in cal_oof:
        for key in cal_oof[name]:
            cal_oof[name][key] = np.concatenate(cal_oof[name][key])
    for arr in [P[m][mc] for m in F_ALL for mc in P[m]] + [Tc, Ta, tauQ, tauA]:
        assert not np.isnan(arr).any()
    for ch in CH:
        for mc in Xmon[ch]:
            assert not np.isnan(LQ[ch][mc]).any()
    mon_df = pd.DataFrame(mon_rows)
    tau_df = pd.DataFrame(tau_rows)
    print("\nMonitor thresholds per fold:")
    print(mon_df.round(4).to_string(index=False), flush=True)
    print("\nTemperatures and cascade thresholds per fold:")
    print(tau_df.round(4).to_string(index=False), flush=True)

    # ---- 2. monitor timing (fold-1 models, all 1440 clips) -------------------------
    timing_rows = []
    single_us = {}
    for ch in CH:
        model = mons[ch][0]
        X = Xmon[ch]["clean"]
        monitor_d(model, X)
        batch = []
        for _ in range(N_TIME_REPS):
            t0 = time.perf_counter()
            monitor_d(model, X)
            batch.append(time.perf_counter() - t0)
        singles = []
        for _ in range(N_SINGLE_REPS):
            t0 = time.perf_counter()
            for i in range(N):
                monitor_d(model, X[i:i + 1])
            singles.append(time.perf_counter() - t0)
        batch_us = float(np.median(batch)) / N * 1e6
        single_us[ch] = float(np.median(singles)) / N * 1e6
        timing_rows.append(dict(monitor=ch, clips=N, batch_median_total_ms=float(np.median(batch)) * 1e3,
                                batch_us_per_clip=batch_us, single_clip_loop_us_per_clip=single_us[ch]))
    MON_COST_S = (single_us["audio"] + single_us["visual"]) * 1e-6  # conservative: per-clip calls
    timing_rows.append(dict(monitor="both (single-clip, added to QAG cost)", clips=N,
                            batch_median_total_ms=np.nan, batch_us_per_clip=np.nan,
                            single_clip_loop_us_per_clip=single_us["audio"] + single_us["visual"]))
    timing_df = pd.DataFrame(timing_rows)
    print("\nMonitor scoring time:")
    print(timing_df.round(3).to_string(index=False), flush=True)

    # ---- 3. system outputs -------------------------------------------------------------
    def flags(c, pct=PCT_MAIN):
        ac, vc = CONDITIONS[c]
        src = LQ if pct == PCT_MAIN else LQsens
        return src["audio"][ac] > 0, src["visual"][vc] > 0

    def logs(c):
        ac, vc = CONDITIONS[c]
        out = {}
        for m in F_ALL:
            out[m] = R.log_probs(P[m][R.mcond(m, c)])
        return out

    def qag(c, pct=PCT_MAIN):
        ac, vc = CONDITIONS[c]
        fa, fv = flags(c, pct)
        aA, vA = active_masks(fa, fv)
        lg = logs(c)
        L1 = aA[:, None] * lg["A0"] + vA[:, None] * lg["V0"]
        L1_t = L1 / Tc[:, None]
        conf1 = R.softmax(L1_t).max(1)
        esc = (conf1 < tauQ) | fa | fv
        LQf = aA[:, None] * (lg["A0"] + lg["A1"]) + vA[:, None] * (lg["V0"] + lg["V1"])
        pQ = R.softmax(LQf / Ta[:, None])
        esc_cost = (C_A1 * aA + C_V1 * vA) * esc
        full_cost = (C_A1 * aA + C_V1 * vA)
        return dict(fa=fa, fv=fv, aA=aA, vA=vA, esc=esc,
                    pred_fall=LQf.argmax(1), prob_fall=pQ,
                    pred_casc=np.where(esc, LQf.argmax(1), L1.argmax(1)),
                    prob_casc=np.where(esc[:, None], pQ, R.softmax(L1_t)),
                    cost_fall=C_CHEAP + full_cost.mean() + MON_COST_S,
                    cost_casc=C_CHEAP + esc_cost.mean() + MON_COST_S)

    def oracle(c):
        drop = ORACLE_DROP.get(c)
        aA, vA = drop != "audio", drop != "visual"
        lg = logs(c)
        L = np.zeros((N, N_CLASSES))
        if aA:
            L += lg["A0"] + lg["A1"]
        if vA:
            L += lg["V0"] + lg["V1"]
        cost = (C_AUD if aA else 0.0) + (C_VID if vA else 0.0) \
            + (C_A0 + C_A1 if aA else 0.0) + (C_V0 + C_V1 if vA else 0.0)
        return dict(pred=L.argmax(1), prob=R.softmax(L / Ta[:, None]), cost=cost,
                    aA=aA, vA=vA)

    def baseline_cascade(c):
        get = lambda m: P[m][R.mcond(m, c)]
        out = R.cascade_run("A", get, tauA)
        r = out["esc"].mean()
        return dict(pred=out["pred"], esc=out["esc"], esc_rate=float(r),
                    cost=(1 - r) * C_CHEAP + r * C_ALL)

    def lf_all(c):
        get = lambda m: P[m][R.mcond(m, c)]
        return R.fuse_lf([get(m) for m in F_ALL]).argmax(1)

    # ---- 4. robustness table -------------------------------------------------------------
    rob_rows, preds = [], {}
    for c in CONDITIONS:
        q = qag(c)
        o = oracle(c)
        b = baseline_cascade(c)
        preds[c] = {"F-all (LF)": lf_all(c), "F-all (QAG)": q["pred_fall"], "Oracle gate": o["pred"],
                    "Cascade A (baseline)": b["pred"], "Cascade-QAG": q["pred_casc"]}
        meta_rows = {
            "F-all (LF)": (C_ALL, np.nan),
            "F-all (QAG)": (q["cost_fall"], np.nan),
            "Oracle gate": (o["cost"], np.nan),
            "Cascade A (baseline)": (b["cost"], b["esc_rate"]),
            "Cascade-QAG": (q["cost_casc"], float(q["esc"].mean())),
        }
        for s in SYSTEMS:
            pr = preds[c][s]
            cost, esc_rate = meta_rows[s]
            rob_rows.append(dict(condition=c, system=s, accuracy=float(np.mean(pr == y)),
                                 macro_f1=float(R.macro_f1_from_cm(R.conf_matrix(y, pr))),
                                 escalation_rate=esc_rate, mean_cost_s=float(cost),
                                 flag_rate_audio=float(q["fa"].mean()), flag_rate_visual=float(q["fv"].mean())))
    rob_df = pd.DataFrame(rob_rows)
    print("\nRobustness (QAG, P99 thresholds):")
    print(rob_df.round(4).to_string(index=False), flush=True)

    # check that the baseline reproduces results/robustness.csv
    ref = pd.read_csv(RES / "robustness.csv")
    for c in CONDITIONS:
        for s, name in (("F-all (LF)", "F-all (LF)"), ("Cascade A (baseline)", "Cascade A (op)")):
            a = rob_df[(rob_df.condition == c) & (rob_df.system == s)].accuracy.values[0]
            b_ = ref[(ref.condition == c) & (ref.system == name)].accuracy.values[0]
            assert abs(a - b_) < 1e-12, (c, s, a, b_)
    print("Baseline check: F-all (LF) and Cascade A accuracies match results/robustness.csv.", flush=True)

    # accuracy by flag pattern (diagnostic: where the channel gate helps or hurts)
    pat_rows = []
    for c in CONDITIONS:
        q = qag(c)
        lg = logs(c)
        LF = sum(lg[m] for m in F_ALL).argmax(1)
        QP = (q["aA"][:, None] * (lg["A0"] + lg["A1"]) + q["vA"][:, None] * (lg["V0"] + lg["V1"])).argmax(1)
        patterns = (("both flagged", q["fa"] & q["fv"]), ("audio flagged only", q["fa"] & ~q["fv"]),
                    ("visual flagged only", ~q["fa"] & q["fv"]), ("none flagged", ~q["fa"] & ~q["fv"]))
        for name, m in patterns:
            n = int(m.sum())
            pat_rows.append(dict(condition=c, pattern=name, n_clips=n,
                                 LF_accuracy=float((LF[m] == y[m]).mean()) if n else np.nan,
                                 QAG_accuracy=float((QP[m] == y[m]).mean()) if n else np.nan))
    pat_df = pd.DataFrame(pat_rows)
    print("\nAccuracy by flag pattern (LF on all four vs QAG active set):")
    print(pat_df.round(4).to_string(index=False), flush=True)

    # per-actor accuracies and Wilcoxon QAG vs LF
    pa_rows = []
    for c in CONDITIONS:
        lf_a = R.per_actor_acc(y, preds[c]["F-all (LF)"], actor)
        qa_a = R.per_actor_acc(y, preds[c]["F-all (QAG)"], actor)
        cq_a = R.per_actor_acc(y, preds[c]["Cascade-QAG"], actor)
        for a in range(24):
            pa_rows.append(dict(condition=c, actor=a + 1, F_all_LF=lf_a[a], F_all_QAG=qa_a[a],
                                Cascade_QAG=cq_a[a]))
    pa_df = pd.DataFrame(pa_rows)

    def wil(a, b):
        d = a - b
        if np.all(d == 0):
            return np.nan, 1.0
        r = stats.wilcoxon(a, b)
        return float(r.statistic), float(r.pvalue)

    st_rows, pvals = [], []
    for c in CONDITIONS:
        sub = pa_df[pa_df.condition == c]
        d = sub.F_all_QAG.values - sub.F_all_LF.values
        W, p = wil(sub.F_all_QAG.values, sub.F_all_LF.values)
        pvals.append(p)
        st_rows.append(dict(condition=c, mean_diff_QAG_minus_LF=float(d.mean()),
                            median_diff=float(np.median(d)), n_actors_better=int((d > 0).sum()),
                            n_actors_worse=int((d < 0).sum()), wilcoxon_W=W, p_value=p))
    holm_adj = R.holm(pvals)
    for row, adj in zip(st_rows, holm_adj):
        row["p_holm_9_conditions"] = float(adj)
    st_df = pd.DataFrame(st_rows)
    print("\nWilcoxon signed-rank, F-all QAG vs F-all LF (per-actor accuracy, Holm over 9 conditions):")
    print(st_df.round(4).to_string(index=False), flush=True)

    # ---- 5. P99.9 sensitivity ------------------------------------------------------------
    sens_rows = []
    for c in CONDITIONS:
        for tag, pct in (("P99", PCT_MAIN), ("P99.9", PCT_SENS)):
            q = qag(c, pct)
            sens_rows.append(dict(
                condition=c, threshold=tag,
                F_all_QAG_accuracy=float(np.mean(q["pred_fall"] == y)),
                F_all_QAG_macro_f1=float(R.macro_f1_from_cm(R.conf_matrix(y, q["pred_fall"]))),
                Cascade_QAG_accuracy=float(np.mean(q["pred_casc"] == y)),
                Cascade_QAG_macro_f1=float(R.macro_f1_from_cm(R.conf_matrix(y, q["pred_casc"]))),
                Cascade_QAG_escalation=float(q["esc"].mean()), Cascade_QAG_mean_cost_s=float(q["cost_casc"]),
                flag_rate_audio=float(q["fa"].mean()), flag_rate_visual=float(q["fv"].mean())))
    sens_df = pd.DataFrame(sens_rows)
    print("\nThreshold sensitivity (P99 vs P99.9):")
    print(sens_df.round(4).to_string(index=False), flush=True)

    # ---- 6. clean operating point of Cascade-QAG ----------------------------------------
    qc = qag("clean")
    b_clean = baseline_cascade("clean")
    acc = lambda p: float(np.mean(p == y))
    f1 = lambda p: float(R.macro_f1_from_cm(R.conf_matrix(y, p)))
    op_rows = [
        dict(system="F-all (LF)", escalation_rate=np.nan, accuracy=acc(preds["clean"]["F-all (LF)"]),
             macro_f1=f1(preds["clean"]["F-all (LF)"]), mean_cost_s=C_ALL, saving_vs_F_all=0.0),
        dict(system="Cascade A (baseline, op)", escalation_rate=b_clean["esc_rate"],
             accuracy=acc(b_clean["pred"]), macro_f1=f1(b_clean["pred"]), mean_cost_s=b_clean["cost"],
             saving_vs_F_all=1 - b_clean["cost"] / C_ALL),
        dict(system="Cascade-QAG (op)", escalation_rate=float(qc["esc"].mean()),
             accuracy=acc(qc["pred_casc"]), macro_f1=f1(qc["pred_casc"]), mean_cost_s=qc["cost_casc"],
             saving_vs_F_all=1 - qc["cost_casc"] / C_ALL),
    ]
    op_df = pd.DataFrame(op_rows)
    conf_esc = float((R.softmax(lf_logits([P["A0"]["clean"], P["V0"]["clean"]]) / Tc[:, None]).max(1)
                      < tauQ).mean())
    op_df["note"] = ""
    op_df.loc[op_df.system == "Cascade-QAG (op)", "note"] = (
        f"confidence-triggered {conf_esc:.4f}; flag-triggered {float((qc['fa'] | qc['fv']).mean()):.4f}")
    print("\nClean operating point (pooled test):")
    print(op_df.round(4).to_string(index=False), flush=True)

    # ---- 7. calibration -----------------------------------------------------------------
    cal_rows = []
    clean_L = {"F-cheap": lf_logits([P["A0"]["clean"], P["V0"]["clean"]]),
               "F-all": lf_logits([P["A0"]["clean"], P["A1"]["clean"], P["V0"]["clean"], P["V1"]["clean"]])}
    for name, T in (("F-cheap", Tc), ("F-all", Ta)):
        L = clean_L[name]
        for stage, logits in (("before", L), ("after", L / T[:, None])):
            conf = R.softmax(logits).max(1)
            corr = (logits.argmax(1) == y).astype(float)
            e_all = R.ece(conf, corr)
            e_m = R.ece(conf[male], corr[male])
            e_f = R.ece(conf[~male], corr[~male])
            cal_rows.append(dict(system=name, stage=stage, split="test pooled (clean)", ece_overall=e_all,
                                 ece_male=e_m, ece_female=e_f, ece_sex_gap=abs(e_m - e_f),
                                 mean_conf=float(conf.mean()), accuracy=float(corr.mean())))
        for stage in ("before", "after"):
            conf = cal_oof[name]["conf_b" if stage == "before" else "conf_a"]
            corr = cal_oof[name]["corr"]
            cal_rows.append(dict(system=name, stage=stage, split="training OOF (T fitted on these)",
                                 ece_overall=R.ece(conf, corr), ece_male=np.nan, ece_female=np.nan,
                                 ece_sex_gap=np.nan, mean_conf=float(conf.mean()),
                                 accuracy=float(corr.mean())))
    cal_df = pd.DataFrame(cal_rows)
    print("\nECE (15 bins), clean condition:")
    print(cal_df.round(4).to_string(index=False), flush=True)

    # ---- 8. drift detection --------------------------------------------------------------
    def ph(x):
        return R.page_hinkley_alarms(x, delta=PH_DELTA, lam=PH_LAMBDA, alpha=PH_ALPHA)

    def ad(x):
        return R.adwin_alarms(x)

    ctrl_counts = {(ch, d): 0 for ch in CH for d in DETS}
    n_ctrl_runs = 0
    for k, (tr, te) in enumerate(splits):
        for s in range(N_DRIFT_SEEDS):
            r0 = np.random.default_rng([SEED, 51, k, s])
            o1, o2 = r0.permutation(te), r0.permutation(te)
            n_ctrl_runs += 1
            for ch in CH:
                x = np.concatenate([Z[ch]["clean"][o1], Z[ch]["clean"][o2]])
                ctrl_counts[(ch, "PH")] += len(ph(x))
                ctrl_counts[(ch, "ADWIN")] += len(ad(x))
    ctrl_clips = n_ctrl_runs * 2 * len(splits[0][1])

    drift_runs = []
    examples = {}
    for ci, c in enumerate(DEG_CONDS):
        mc_of = {"audio": CONDITIONS[c][0], "visual": CONDITIONS[c][1]}
        for k, (tr, te) in enumerate(splits):
            for s in range(N_DRIFT_SEEDS):
                r1 = np.random.default_rng([SEED, 61, k, s, ci])
                p1, p2 = r1.permutation(te), r1.permutation(te)
                alarms, lq_s = {}, {}
                for ch in CH:
                    x = np.concatenate([Z[ch]["clean"][p1], Z[ch][mc_of[ch]][p2]])
                    alarms[(ch, "PH")] = ph(x)
                    alarms[(ch, "ADWIN")] = ad(x)
                    lq_s[ch] = np.concatenate([LQ[ch]["clean"][p1], LQ[ch][mc_of[ch]][p2]])
                drift_runs.append(dict(cond=c, fold=k, seed=s, alarms=alarms))
                if k == 0 and s == 0 and c in ("snr10", "blur"):
                    examples[c] = dict(lq=lq_s, alarms=alarms)

    def first_post(alarms, ch, d):
        post = [t for t in alarms[(ch, d)] if t >= HALF]
        return post[0] - HALF if post else None

    def first_post_t(alarm_list):
        post = [t for t in alarm_list if t >= HALF]
        return post[0] - HALF if post else None

    drift_rows, attr_rows = [], []
    for c in DEG_CONDS:
        runs = [r for r in drift_runs if r["cond"] == c]
        for d in DETS:
            for ch in CH:
                pre = sum(sum(1 for t in r["alarms"][(ch, d)] if t < HALF) for r in runs)
                delays = [v for v in (first_post(r["alarms"], ch, d) for r in runs) if v is not None]
                drift_rows.append(dict(
                    condition=c, monitor=ch, detector=d, n_runs=len(runs),
                    detection_rate=len(delays) / len(runs),
                    delay_median=float(np.median(delays)) if delays else np.nan,
                    delay_q25=float(np.percentile(delays, 25)) if delays else np.nan,
                    delay_q75=float(np.percentile(delays, 75)) if delays else np.nan,
                    false_alarms_per_1000_clean=1000.0 * ctrl_counts[(ch, d)] / ctrl_clips,
                    pre_change_alarms_per_1000=1000.0 * pre / (len(runs) * HALF),
                    degraded_channel=("yes" if ch in DEGRADED_CH[c] else "no")))
            n_det, n_ok = 0, 0
            for r in runs:
                cands = [(t, ch) for ch in CH for t in r["alarms"][(ch, d)] if t >= HALF]
                if not cands:
                    continue
                t_min = min(t for t, _ in cands)
                tied = {ch for t, ch in cands if t == t_min}
                n_det += 1
                n_ok += int(bool(tied & DEGRADED_CH[c]))
            attr_rows.append(dict(condition=c, detector=d, n_runs=len(runs), n_detected=n_det,
                                  attribution_accuracy=(n_ok / n_det) if n_det else np.nan))
    drift_df = pd.DataFrame(drift_rows)
    attr_df = pd.DataFrame(attr_rows)
    print("\nDrift detection with QAG monitors (standardised log q):")
    print(drift_df.round(3).to_string(index=False), flush=True)
    print("\nAttribution (first post-change alarm comes from the degraded channel):")
    print(attr_df.round(3).to_string(index=False), flush=True)

    # Page-Hinkley lambda sensitivity (delta fixed at 0.005); lambda = 5 is the specified setting
    lam_rows = []
    for lam in PH_LAMBDA_GRID:
        def f_ph(x, lam=lam):
            return R.page_hinkley_alarms(x, delta=PH_DELTA, lam=lam, alpha=PH_ALPHA)
        ctrl = {ch: 0 for ch in CH}
        for k, (tr, te) in enumerate(splits):
            for s in range(N_DRIFT_SEEDS):
                r0 = np.random.default_rng([SEED, 51, k, s])
                o1, o2 = r0.permutation(te), r0.permutation(te)
                for ch in CH:
                    ctrl[ch] += len(f_ph(np.concatenate([Z[ch]["clean"][o1], Z[ch]["clean"][o2]])))
        for ci, c in enumerate(DEG_CONDS):
            mc_of = {"audio": CONDITIONS[c][0], "visual": CONDITIONS[c][1]}
            alarms_by = {ch: [] for ch in CH}
            for k, (tr, te) in enumerate(splits):
                for s in range(N_DRIFT_SEEDS):
                    r1 = np.random.default_rng([SEED, 61, k, s, ci])
                    p1, p2 = r1.permutation(te), r1.permutation(te)
                    for ch in CH:
                        x = np.concatenate([Z[ch]["clean"][p1], Z[ch][mc_of[ch]][p2]])
                        alarms_by[ch].append(f_ph(x))
            for ch in CH:
                al = alarms_by[ch]
                delays = [v for v in (first_post_t(a) for a in al) if v is not None]
                lam_rows.append(dict(
                    lam=lam, condition=c, monitor=ch, degraded_channel=("yes" if ch in DEGRADED_CH[c] else "no"),
                    detection_rate=len(delays) / len(al),
                    delay_median=float(np.median(delays)) if delays else np.nan,
                    false_alarms_per_1000_clean=1000.0 * ctrl[ch] / ctrl_clips))
    lam_df = pd.DataFrame(lam_rows)
    print("\nPage-Hinkley lambda sensitivity (delta = 0.005, standardised log q):")
    print(lam_df.round(3).to_string(index=False), flush=True)

    # ---- 9. outputs -----------------------------------------------------------------------
    mon_df.to_csv(RES / "qag_monitor_thresholds.csv", index=False)
    tau_df.to_csv(RES / "qag_thresholds_temperatures.csv", index=False)
    cal_df.to_csv(RES / "qag_calibration.csv", index=False)
    rob_df.to_csv(RES / "qag_robustness.csv", index=False)
    pa_df.to_csv(RES / "qag_per_actor.csv", index=False)
    st_df.to_csv(RES / "qag_stats.csv", index=False)
    sens_df.to_csv(RES / "qag_sensitivity.csv", index=False)
    op_df.to_csv(RES / "qag_cascade_op.csv", index=False)
    drift_df.to_csv(RES / "qag_drift.csv", index=False)
    attr_df.to_csv(RES / "qag_drift_attribution.csv", index=False)
    lam_df.to_csv(RES / "qag_drift_ph_lambda.csv", index=False)
    pat_df.to_csv(RES / "qag_flag_patterns.csv", index=False)
    timing_df.to_csv(RES / "qag_monitor_timing.csv", index=False)

    # ---- 10. figures ------------------------------------------------------------------------
    R.set_style()
    fig_robustness_qag(rob_df)
    lq_by = {ch: {mc: LQ[ch][mc] for mc in Xmon[ch]} for ch in CH}
    flag_by = {ch: {mc: float(np.mean(LQ[ch][mc] > 0)) for mc in Xmon[ch]} for ch in CH}
    fig_monitor_scores(lq_by, flag_by)
    fig_drift_qag(examples)

    summary = dict(
        seed=SEED, n_clips=N, runtime_seconds=time.perf_counter() - t_start,
        config=dict(pca_variance=PCA_VAR, threshold_percentiles=[PCT_MAIN, PCT_SENS],
                    inner_folds=R.N_INNER, ph=dict(delta=PH_DELTA, lam=PH_LAMBDA, alpha=PH_ALPHA,
                                                   signal="standardised log q"),
                    adwin_delta=R.ADWIN_DELTA, drift_seeds=N_DRIFT_SEEDS,
                    monitor_cost_s_added_to_qag=MON_COST_S, cost_constants_s=dict(
                        stage1=C_CHEAP, F_all=C_ALL, audio_decode=C_AUD, video_decode_face=C_VID,
                        A0=C_A0, A1=C_A1, V0=C_V0, V1=C_V1)),
        monitor_thresholds=mon_rows, thresholds_temperatures=tau_rows,
        calibration=cal_rows, robustness=rob_rows, stats=st_rows, sensitivity=sens_rows,
        clean_operating_point=op_rows, drift=drift_rows, drift_attribution=attr_rows,
        monitor_timing=timing_rows, flag_patterns=pat_rows,
        ph_lambda_sensitivity=lam_rows,
    )
    import json
    with open(RES / "qag_summary.json", "w") as f:
        json.dump(to_py(summary), f, indent=2)
    print(f"\nTotal runtime: {time.perf_counter() - t_start:.1f} s", flush=True)


if __name__ == "__main__":
    main()
