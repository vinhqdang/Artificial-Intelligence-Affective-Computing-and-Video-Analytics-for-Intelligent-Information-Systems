#!/usr/bin/env python3
"""Stream-level gating (SLG): design iteration 3 of the audio-visual emotion cascade.

Per stream, one ADWIN per channel (prosody = audio, mnet = visual) runs on the standardised
log q signal of that channel's input monitor. A channel is marked DEGRADED when its ADWIN
signals a change and the mean of its most recent 15 standardised values exceeds 1.0; a
DEGRADED channel is marked HEALTHY again when its ADWIN signals a change and that mean is
below 1.0. Marks take effect from the next clip. Predictions are late fusion over the active
models. Monitors, temperatures and cached model outputs are rebuilt with the code path of
code/run_qag.py (which imports code/run_experiments.py). Run from the repository root:

    python3 code/run_slg.py
"""
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from river.drift import ADWIN
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_qag as Q  # noqa: E402
R = Q.R

SEED = R.SEED
FEAT, CACHE, RES, FIG = R.FEAT, R.CACHE, R.RES, R.FIG
HALF = R.HALF                       # 240 clips per segment in S1
N_FOLDS, N_SEEDS = R.N_FOLDS, R.N_DRIFT_SEEDS
CH = ("audio", "visual")
F_ALL = R.F_ALL
THR = 1.0                           # decision threshold on the recent-15 mean of standardised log q
WIN = 15
ROLL_SIG, ROLL_ACC = 10, 20
N_S2 = 160                          # clips per segment in S2
SEG2 = (N_S2, 2 * N_S2)             # degraded interval in S2
N_CTRL = 2 * HALF                   # clips per control stream (S3)
COND_NAMES = list(R.CONDITIONS.keys())          # index 0 = clean
DEG = R.DEG_CONDS
CI_OF = {c: i for i, c in enumerate(COND_NAMES)}
DEGRADED_CH = dict(Q.DEGRADED_CH, control=set())
DROP_A = np.array([Q.ORACLE_DROP.get(c) == "audio" for c in COND_NAMES])
DROP_V = np.array([Q.ORACLE_DROP.get(c) == "visual" for c in COND_NAMES])
SYSTEMS = ["LF", "QAG", "SLG", "Oracle", "Oracle_runqag_cost"]
LABEL = {"LF": "LF (F-all)", "QAG": "Clip QAG (P99)", "SLG": "SLG", "Oracle": "Oracle gate",
         "Oracle_runqag_cost": "Oracle gate (run_qag cost convention)"}
SEG_NAMES = ("clean_first", "deg", "clean_last")


# ----------------------------------------------------------------------------
# Setup: features, cached probabilities, per-fold monitors and temperatures
# ----------------------------------------------------------------------------
def build_state():
    meta, y, actor, male = R.load_meta()
    N = len(y)
    splits = R.fold_splits(actor)
    _, COST, _, _ = R.cost_tables(meta)
    comp = COST["components"]
    costs = dict(C_AUD=comp["audio_decode"], C_VID=comp["video_decode_face"],
                 C_A0=comp["A0 prosody"], C_A1=comp["A1 w2v"], C_V0=comp["V0 mnet"], C_V1=comp["V1 clip"],
                 C_ALL=COST["F-all (all components)"])
    timing = pd.read_csv(RES / "qag_monitor_timing.csv")
    mon_s = float(timing.loc[timing.monitor.str.startswith("both"), "single_clip_loop_us_per_clip"].values[0]) * 1e-6
    costs["MON"] = mon_s

    X_pro = {mc: np.load(FEAT / f"prosody_{mc}.npy").astype(np.float64) for mc in R.AUDIO_CONDS}
    X_vis = {mc: np.load(FEAT / f"mnet_{mc}.npy").astype(np.float64) for mc in R.VIDEO_CONDS}
    Xmon = {"audio": X_pro, "visual": X_vis}

    res = {m: [pickle.load(open(CACHE / f"{m}_fold{k}.pkl", "rb")) for k in range(N_FOLDS)] for m in F_ALL}
    for m in F_ALL:
        for k, (tr, te) in enumerate(splits):
            assert np.array_equal(res[m][k]["train_idx"], tr) and np.array_equal(res[m][k]["test_idx"], te)

    P = {m: {mc: np.full((N, R.N_CLASSES), np.nan) for mc in R.MODEL_MCONDS[m]} for m in F_ALL}
    Zg = {ch: {mc: np.full(N, np.nan) for mc in Xmon[ch]} for ch in CH}
    LQg = {ch: {mc: np.full(N, np.nan) for mc in Xmon[ch]} for ch in CH}
    Ta_rows = []
    for k, (tr, te) in enumerate(splits):
        for m in F_ALL:
            for mc in R.MODEL_MCONDS[m]:
                P[m][mc][te] = res[m][k]["test_probs"][mc]
        G = {m: res[m][k]["oof_probs"] for m in F_ALL}
        Ta_k = R.fit_temperature(Q.lf_logits([G[m] for m in F_ALL]), y[tr])
        Ta_rows.append(dict(fold=k + 1, T_all=Ta_k))
        for ch in CH:
            mon = Q.monitor_fold(Xmon[ch]["clean"], tr, actor)
            for mc, X in Xmon[ch].items():
                d = Q.monitor_d(mon["model"], X[te])
                LQg[ch][mc][te] = np.log(d / mon["theta"]["p99"])
                Zg[ch][mc][te] = (LQg[ch][mc][te] - mon["oof_mu"]) / mon["oof_sd"]
    for m in F_ALL:
        for mc in P[m]:
            assert not np.isnan(P[m][mc]).any()
    for ch in CH:
        for mc in Xmon[ch]:
            assert not np.isnan(Zg[ch][mc]).any() and not np.isnan(LQg[ch][mc]).any()

    # per-condition views: index ci in COND_NAMES
    LPS = {m: np.stack([R.log_probs(P[m][R.mcond(m, c)]) for c in COND_NAMES]) for m in F_ALL}
    ZS = {ch: np.stack([Zg[ch][R.CONDITIONS[c][0 if ch == "audio" else 1]] for c in COND_NAMES]) for ch in CH}
    LQS = {ch: np.stack([LQg[ch][R.CONDITIONS[c][0 if ch == "audio" else 1]] for c in COND_NAMES]) for ch in CH}
    return dict(y=y, actor=actor, splits=splits, costs=costs, LPS=LPS, ZS=ZS, LQS=LQS,
                Ta=pd.DataFrame(Ta_rows), N=N)


# ----------------------------------------------------------------------------
# Stream construction (same permutation tags as run_qag for S1 and S3)
# ----------------------------------------------------------------------------
def make_streams(st):
    streams = []
    for k, (tr, te) in enumerate(st["splits"]):
        for s in range(N_SEEDS):
            r0 = np.random.default_rng([SEED, 51, k, s])               # S3 control, as run_qag
            o1, o2 = r0.permutation(te), r0.permutation(te)
            streams.append(dict(scn="S3", cond="control", fold=k, seed=s,
                                clips=np.concatenate([o1, o2]), conds=np.zeros(N_CTRL, int),
                                segs=dict(clean_first=(0, N_CTRL))))
            for ci, c in enumerate(DEG):
                r1 = np.random.default_rng([SEED, 61, k, s, ci])       # S1, as run_qag drift streams
                p1, p2 = r1.permutation(te), r1.permutation(te)
                streams.append(dict(scn="S1", cond=c, fold=k, seed=s,
                                    clips=np.concatenate([p1, p2]),
                                    conds=np.r_[np.zeros(HALF, int), np.full(HALF, CI_OF[c])],
                                    segs=dict(clean_first=(0, HALF), deg=(HALF, 2 * HALF))))
                r2 = np.random.default_rng([SEED, 71, k, s, ci])       # S2 recovery
                q1, q2, q3 = (r2.permutation(te) for _ in range(3))
                streams.append(dict(scn="S2", cond=c, fold=k, seed=s,
                                    clips=np.concatenate([q1[:N_S2], q2[:N_S2], q3[:N_S2]]),
                                    conds=np.r_[np.zeros(N_S2, int), np.full(N_S2, CI_OF[c]),
                                                np.zeros(N_S2, int)],
                                    segs=dict(clean_first=(0, N_S2), deg=(N_S2, 2 * N_S2),
                                              clean_last=(2 * N_S2, 3 * N_S2))))
    return streams


# ----------------------------------------------------------------------------
# SLG state machine
# ----------------------------------------------------------------------------
def run_slg(z):
    """z: dict channel -> standardised log q per clip. Returns per-clip DEGRADED state in effect
    for each clip (decisions at clip t apply from clip t+1), events, ADWIN alarm times, final state."""
    T = len(z["audio"])
    state = {ch: False for ch in CH}
    det = {ch: ADWIN(delta=R.ADWIN_DELTA) for ch in CH}
    eff = {ch: np.zeros(T, dtype=bool) for ch in CH}
    events, drifts = [], {ch: [] for ch in CH}
    for t in range(T):
        for ch in CH:
            eff[ch][t] = state[ch]
        for ch in CH:
            det[ch].update(float(z[ch][t]))
            if det[ch].drift_detected:
                drifts[ch].append(t)
                m15 = float(z[ch][max(0, t - WIN + 1): t + 1].mean())
                if not state[ch] and m15 > THR:
                    state[ch] = True
                    events.append((t, ch, "DEGRADED", m15))
                elif state[ch] and m15 < THR:
                    state[ch] = False
                    events.append((t, ch, "HEALTHY", m15))
    return eff, events, drifts, dict(state)


def simulate(st, c):
    clips, conds, y = st["clips"], st["conds"], c["y"]
    z = {ch: c["ZS"][ch][conds, clips] for ch in CH}
    eff, events, drifts, final = run_slg(z)
    aD, vD = eff["audio"], eff["visual"]
    aS, vS = Q.active_masks(aD, vD)
    fa = c["LQS"]["audio"][conds, clips] > 0
    fv = c["LQS"]["visual"][conds, clips] > 0
    aQ, vQ = Q.active_masks(fa, fv)
    dA, dV = DROP_A[conds], DROP_V[conds]
    L = {m: c["LPS"][m][conds, clips] for m in F_ALL}
    Aall, Vall = L["A0"] + L["A1"], L["V0"] + L["V1"]
    pred = {
        "LF": (Aall + Vall).argmax(1),
        "QAG": (aQ[:, None] * Aall + vQ[:, None] * Vall).argmax(1),
        "SLG": (aS[:, None] * Aall + vS[:, None] * Vall).argmax(1),
        "Oracle": (~dA[:, None] * Aall + ~dV[:, None] * Vall).argmax(1),
    }
    pred["Oracle_runqag_cost"] = pred["Oracle"]
    K = c["costs"]
    T = len(clips)
    cost = {
        "LF": np.full(T, K["C_ALL"]),
        "QAG": K["C_ALL"] - K["C_A1"] * (~aQ) - K["C_V1"] * (~vQ) + K["MON"],
        "SLG": K["C_ALL"] - K["C_A1"] * (~aS) - K["C_V1"] * (~vS) + K["MON"],
        "Oracle": K["C_ALL"] - K["C_A1"] * dA - K["C_V1"] * dV,
        "Oracle_runqag_cost": (K["C_AUD"] + K["C_A0"] + K["C_A1"]) * (~dA)
                              + (K["C_VID"] + K["C_V0"] + K["C_V1"]) * (~dV),
    }
    cor = {s: pred[s] == y[clips] for s in SYSTEMS}
    return dict(cor=cor, pred=pred, cost={s: float(cost[s].mean()) for s in SYSTEMS},
                eff=dict(audio=aD, visual=vD), active=dict(audio=aS, visual=vS),
                active_qag=dict(audio=aQ, visual=vQ), events=events, drifts=drifts,
                final=final, z=z, clips=clips, conds=conds)


# ----------------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------------
def stream_frame(streams, outs):
    rows = []
    for st, o in zip(streams, outs):
        for s in SYSTEMS:
            rec = dict(scn=st["scn"], cond=st["cond"], fold=st["fold"] + 1, seed=st["seed"], system=s,
                       cost=o["cost"][s])
            for seg in SEG_NAMES:
                rec[seg] = float(o["cor"][s][slice(*st["segs"][seg])].mean()) if seg in st["segs"] else np.nan
            rows.append(rec)
    return pd.DataFrame(rows)


def scenario_table(sdf):
    g = sdf.groupby(["scn", "cond", "system"], sort=False)
    out = g.agg(n_streams=("fold", "size"), acc_clean_first=("clean_first", "mean"),
                acc_degraded=("deg", "mean"), acc_clean_last=("clean_last", "mean"),
                mean_cost_s=("cost", "mean")).reset_index()
    out["system"] = out["system"].map(LABEL)
    return out


def per_fold_table(sdf):
    d = sdf[sdf.scn.isin(["S1", "S2"]) & sdf.system.isin(["LF", "SLG", "QAG", "Oracle"])]
    t = d.groupby(["scn", "cond", "fold", "system"])["deg"].mean().unstack("system").reset_index()
    t["diff_SLG_minus_LF"] = t["SLG"] - t["LF"]
    return t


def per_actor_tables(streams, outs, actor):
    acc = {}
    for st, o in zip(streams, outs):
        if st["scn"] not in ("S1", "S2"):
            continue
        a, b = st["segs"]["deg"]
        aid = actor[st["clips"][a:b]]
        key = (st["scn"], st["cond"])
        if key not in acc:
            acc[key] = {s: [np.zeros(25), np.zeros(25)] for s in ("LF", "SLG")}
        for s in ("LF", "SLG"):
            acc[key][s][0] += np.bincount(aid, weights=o["cor"][s][a:b].astype(float), minlength=25)
            acc[key][s][1] += np.bincount(aid, minlength=25)
    rows = []
    for (scn, cond), d in acc.items():
        for ai in range(1, 25):
            rows.append(dict(scn=scn, cond=cond, actor=ai,
                             LF=d["LF"][0][ai] / d["LF"][1][ai], SLG=d["SLG"][0][ai] / d["SLG"][1][ai]))
    return pd.DataFrame(rows)


def wilcoxon_tables(pa):
    rows = []
    for scn in ("S1", "S2"):
        conds = [c for c in DEG if ((pa.scn == scn) & (pa.cond == c)).any()]
        pvals, block = [], []
        for c in conds:
            sub = pa[(pa.scn == scn) & (pa.cond == c)].sort_values("actor")
            d = sub.SLG.values - sub.LF.values
            if np.all(d == 0):
                W, p = np.nan, 1.0
            else:
                r = stats.wilcoxon(sub.SLG.values, sub.LF.values)
                W, p = float(r.statistic), float(r.pvalue)
            pvals.append(p)
            block.append(dict(scn=scn, condition=c, mean_diff_SLG_minus_LF=float(d.mean()),
                              median_diff=float(np.median(d)), n_actors_better=int((d > 0).sum()),
                              n_actors_worse=int((d < 0).sum()), wilcoxon_W=W, p_value=p))
        adj = R.holm(pvals)
        for row, a in zip(block, adj):
            row["p_holm_8_conditions"] = float(a)
        rows += block
    return pd.DataFrame(rows)


def detection_table(streams, outs):
    rows = []
    for c in DEG:
        idx = [i for i, st in enumerate(streams) if st["scn"] == "S1" and st["cond"] == c]
        truth = DEGRADED_CH[c]
        delays, firsts, pre, wrong = [], [], 0, 0
        for i in idx:
            ev = outs[i]["events"]
            if any(e[0] < HALF and e[2] == "DEGRADED" for e in ev):
                pre += 1
            wrong += sum(1 for e in ev if e[2] == "DEGRADED" and e[1] not in truth)
            post = [e for e in ev if e[0] >= HALF and e[2] == "DEGRADED"]
            if post:
                delays.append(post[0][0] + 1 - HALF)     # degraded clips processed ungated
                firsts.append(post[0][1])
        n, d = len(idx), np.array(delays, float)
        rows.append(dict(
            condition=c, n_streams=n, detection_rate=len(d) / n,
            delay_median=float(np.median(d)) if len(d) else np.nan,
            delay_q25=float(np.percentile(d, 25)) if len(d) else np.nan,
            delay_q75=float(np.percentile(d, 75)) if len(d) else np.nan,
            first_mark_audio=sum(f == "audio" for f in firsts), first_mark_visual=sum(f == "visual" for f in firsts),
            first_mark_on_degraded_channel=float(np.mean([f in truth for f in firsts])) if firsts else np.nan,
            pre_change_mark_rate=pre / n,
            wrong_channel_degraded_marks_per_1000_clips=1000.0 * wrong / (n * 2 * HALF)))
    return pd.DataFrame(rows)


def recovery_table(streams, outs):
    rows = []
    for c in DEG:
        idx = [i for i, st in enumerate(streams) if st["scn"] == "S2" and st["cond"] == c]
        n_inst = n_rec = n_drop_streams = n_all_rec = n_det = 0
        det_delay, rec_delay = [], []
        for i in idx:
            o = outs[i]
            ev = o["events"]
            d_ev = [e for e in ev if SEG2[0] <= e[0] < SEG2[1] and e[2] == "DEGRADED"]
            if d_ev:
                n_det += 1
                det_delay.append(d_ev[0][0] + 1 - SEG2[0])
            dropped = [ch for ch in CH if o["eff"][ch][SEG2[1]]]
            if not dropped:
                continue
            n_drop_streams += 1
            all_rec = True
            for ch in dropped:
                n_inst += 1
                h = [e for e in ev if e[1] == ch and e[2] == "HEALTHY" and e[0] >= SEG2[1]]
                if h and not o["final"][ch]:
                    n_rec += 1
                    rec_delay.append(h[0][0] + 1 - SEG2[1])
                else:
                    all_rec = False
            n_all_rec += int(all_rec)
        rd = np.array(rec_delay, float)
        dd = np.array(det_delay, float)
        rows.append(dict(
            condition=c, n_streams=len(idx), detected_in_degraded_segment=n_det / len(idx),
            detection_delay_median=float(np.median(dd)) if len(dd) else np.nan,
            streams_with_channel_dropped_at_return=n_drop_streams,
            channel_instances_dropped_at_return=n_inst, channel_instances_recovered=n_rec,
            recovered_fraction_channels=(n_rec / n_inst) if n_inst else np.nan,
            recovered_fraction_streams=(n_all_rec / n_drop_streams) if n_drop_streams else np.nan,
            recovery_delay_median=float(np.median(rd)) if len(rd) else np.nan,
            recovery_delay_q25=float(np.percentile(rd, 25)) if len(rd) else np.nan,
            recovery_delay_q75=float(np.percentile(rd, 75)) if len(rd) else np.nan))
    return pd.DataFrame(rows)


def control_table(streams, outs, sdf):
    idx = [i for i, st in enumerate(streams) if st["scn"] == "S3"]
    n_clips = len(idx) * N_CTRL
    deg_events = {"audio": 0, "visual": 0}
    clips_any_marked = clips_any_dropped = clips_qag_dropped = 0
    streams_with_event = 0
    zero_streams, zero_identical, zero_acc_equal = 0, 0, 0
    for i in idx:
        o = outs[i]
        ev = [e for e in o["events"] if e[2] == "DEGRADED"]
        for e in ev:
            deg_events[e[1]] += 1
        streams_with_event += int(len(ev) > 0)
        aD, vD = o["eff"]["audio"], o["eff"]["visual"]
        aS, vS = o["active"]["audio"], o["active"]["visual"]
        clips_any_marked += int((aD | vD).sum())
        clips_any_dropped += int((~aS | ~vS).sum())
        aQ, vQ = o["active_qag"]["audio"], o["active_qag"]["visual"]
        clips_qag_dropped += int((~aQ | ~vQ).sum())
        if len(ev) == 0:
            zero_streams += 1
            zero_identical += int(np.array_equal(o["pred"]["SLG"], o["pred"]["LF"]))
            zero_acc_equal += int(o["cor"]["SLG"].mean() == o["cor"]["LF"].mean())
    sub = sdf[sdf.scn == "S3"].groupby("system")
    rows = []
    for s in SYSTEMS[:3]:
        rows.append(dict(system=LABEL[s], accuracy_clean=float(sub.clean_first.mean()[s]),
                         mean_cost_s=float(sub.cost.mean()[s])))
    ct = pd.DataFrame(rows)
    summary = dict(
        n_streams=len(idx), n_clips=n_clips,
        pct_clips_any_channel_marked_degraded=100.0 * clips_any_marked / n_clips,
        pct_clips_any_model_dropped_SLG=100.0 * clips_any_dropped / n_clips,
        pct_clips_any_model_dropped_QAG=100.0 * clips_qag_dropped / n_clips,
        degraded_marks_per_1000_clips=1000.0 * (deg_events["audio"] + deg_events["visual"]) / n_clips,
        degraded_marks_audio_per_1000_clips=1000.0 * deg_events["audio"] / n_clips,
        degraded_marks_visual_per_1000_clips=1000.0 * deg_events["visual"] / n_clips,
        pct_streams_with_any_mark=100.0 * streams_with_event / len(idx),
        streams_without_any_mark=zero_streams,
        zero_mark_streams_SLG_pred_identical_to_LF=zero_identical,
        zero_mark_streams_SLG_acc_equal_LF=zero_acc_equal)
    return ct, summary


def fig_slg_stream(examples):
    R.set_style()
    fig = plt.figure(figsize=(7, 3.9))
    gs = GridSpec(2, 2, height_ratios=[2.3, 1.0], hspace=0.12, wspace=0.28, figure=fig)
    blue, orange, vermil, green = R.PAL["blue"], R.PAL["orange"], R.PAL["vermil"], R.PAL["green"]
    for j, (cond, ex) in enumerate(examples.items()):
        ax = fig.add_subplot(gs[0, j])
        axa = fig.add_subplot(gs[1, j], sharex=ax)
        t = np.arange(len(ex["clips"]))
        for ch, colour in (("audio", blue), ("visual", orange)):
            ax.plot(t, pd.Series(ex["z"][ch]).rolling(ROLL_SIG, min_periods=1).mean().values,
                    color=colour, lw=1.0)
        ax.axvspan(*SEG2, color=R.PAL["grey"], alpha=0.18, lw=0)
        ax.axhline(0, color=R.PAL["grey"], lw=0.6)
        ax.axhline(THR, color=R.PAL["black"], ls=":", lw=0.7)
        for (tt, ch, st, _) in ex["events"]:
            ax.axvline(tt + 1, color=R.PAL["black"] if st == "DEGRADED" else green,
                       ls="-" if st == "DEGRADED" else "--", lw=0.8)
        ax.set_title(f"{cond} (fold 1, seed 0)")
        ax.tick_params(labelbottom=False)
        if j == 0:
            ax.set_ylabel("standardised log q\n(rolling mean 10)")
        acc_lf = pd.Series(ex["cor"]["LF"].astype(float)).rolling(ROLL_ACC, min_periods=1).mean().values
        acc_slg = pd.Series(ex["cor"]["SLG"].astype(float)).rolling(ROLL_ACC, min_periods=1).mean().values
        axa.plot(t, acc_lf, color=blue, lw=1.0, label="LF (F-all)")
        axa.plot(t, acc_slg, color=vermil, lw=1.0, label="SLG")
        axa.axvspan(*SEG2, color=R.PAL["grey"], alpha=0.18, lw=0)
        axa.set_ylim(-0.05, 1.05)
        axa.set_xlabel("Clip index in stream")
        if j == 0:
            axa.set_ylabel("accuracy\n(rolling 20)")
    handles = [Line2D([], [], color=blue, lw=1.0, label="audio monitor"),
               Line2D([], [], color=orange, lw=1.0, label="visual monitor"),
               Line2D([], [], color=R.PAL["black"], lw=0.8, label="SLG marks DEGRADED"),
               Line2D([], [], color=green, lw=0.8, ls="--", label="SLG marks HEALTHY"),
               Line2D([], [], color=R.PAL["grey"], lw=6, alpha=0.4, label="true degraded interval"),
               Line2D([], [], color=blue, lw=1.0, label="LF (F-all) accuracy"),
               Line2D([], [], color=vermil, lw=1.0, label="SLG accuracy")]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=6,
               bbox_to_anchor=(0.5, 1.06))
    R.save_fig(fig, "slg_stream.pdf")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    t0 = time.perf_counter()
    for d in (RES, FIG):
        d.mkdir(parents=True, exist_ok=True)
    st = build_state()
    y, actor, splits, K = st["y"], st["actor"], st["splits"], st["costs"]

    # checks against run_qag outputs
    ta_ref = pd.read_csv(RES / "qag_thresholds_temperatures.csv")
    ta_diff = float(np.max(np.abs(st["Ta"].T_all.values - ta_ref.T_all.values)))
    assert ta_diff < 1e-9, ta_diff

    streams = make_streams(st)
    outs = [simulate(s, st) for s in streams]

    # streaming ADWIN alarms must equal the batch helper used in run_qag
    n_checked = 0
    for s, o in zip(streams, outs):
        for ch in CH:
            assert o["drifts"][ch] == R.adwin_alarms(o["z"][ch]), (s["scn"], s["cond"], s["fold"], s["seed"], ch)
            n_checked += 1

    sdf = stream_frame(streams, outs)
    scen = scenario_table(sdf)
    pf = per_fold_table(sdf)
    pa = per_actor_tables(streams, outs, actor)
    st_df = wilcoxon_tables(pa)
    det = detection_table(streams, outs)
    rec = recovery_table(streams, outs)
    ctrl_ct, ctrl_sum = control_table(streams, outs, sdf)

    # sanity checks against run_qag robustness tables (S1 degraded/clean segments equal full test sets)
    rob = pd.read_csv(RES / "robustness.csv")
    qrob = pd.read_csv(RES / "qag_robustness.csv")
    checks = []
    s1_all = sdf[sdf.scn == "S1"]
    for c in COND_NAMES:
        for sysname, ref_df, ref_name in (("LF", rob, "F-all (LF)"), ("QAG", qrob, "F-all (QAG)"),
                                          ("Oracle", qrob, "Oracle gate")):
            if c == "clean":   # clean segment of every S1 stream is a full permutation of the test set
                seg = "clean_first"
                got = s1_all[s1_all.system == sysname][seg].mean()
            else:
                seg = "deg"
                got = s1_all[(s1_all.cond == c) & (s1_all.system == sysname)][seg].mean()
            ref = ref_df[(ref_df.condition == c) & (ref_df.system == ref_name)].accuracy.values[0]
            ok = abs(got - ref) < 1e-12 if c != "clean" else abs(got - ref) < 1e-12
            checks.append(dict(check=f"S1 {sysname} {seg} accuracy vs run_qag table", condition=c,
                               slg_pipeline=float(got), run_qag_table=float(ref), match=bool(ok)))
    s3_lf = sdf[(sdf.scn == "S3") & (sdf.system == "LF")].clean_first.mean()
    ref_clean = rob[(rob.condition == "clean") & (rob.system == "F-all (LF)")].accuracy.values[0]
    checks.append(dict(check="S3 LF accuracy vs run_qag clean LF", condition="control",
                       slg_pipeline=float(s3_lf), run_qag_table=float(ref_clean),
                       match=bool(abs(s3_lf - ref_clean) < 1e-12)))
    assert all(c["match"] for c in checks), [c for c in checks if not c["match"]]
    print("Checks against run_qag tables passed:", len(checks), flush=True)

    # ---- outputs -----------------------------------------------------------------------
    events_rows = []
    for s, o in zip(streams, outs):
        for (t, ch, new, m15) in o["events"]:
            events_rows.append(dict(scenario=s["scn"], condition=s["cond"], fold=s["fold"] + 1, seed=s["seed"],
                                    clip_index=t, effective_from=t + 1, channel=ch, new_state=new,
                                    mean_last15=m15, channel_degraded_truth=ch in DEGRADED_CH[s["cond"]]))
    ev_df = pd.DataFrame(events_rows)
    scen.to_csv(RES / "slg_scenarios.csv", index=False)
    det.to_csv(RES / "slg_detection.csv", index=False)
    rec.to_csv(RES / "slg_recovery.csv", index=False)
    ctrl_ct.to_csv(RES / "slg_control.csv", index=False)
    pf.to_csv(RES / "slg_per_fold.csv", index=False)
    st_df.to_csv(RES / "slg_stats.csv", index=False)
    pa.to_csv(RES / "slg_per_actor.csv", index=False)
    ev_df.to_csv(RES / "slg_events.csv", index=False)

    # figure: S2 recovery, fold 1 seed 0, snr10 and compress
    examples = {}
    for c in ("snr10", "compress"):
        i = next(i for i, s in enumerate(streams) if s["scn"] == "S2" and s["cond"] == c
                 and s["fold"] == 0 and s["seed"] == 0)
        o = outs[i]
        examples[c] = dict(clips=streams[i]["clips"], z=o["z"], events=o["events"], cor=o["cor"])
    fig_slg_stream(examples)

    summary = dict(
        seed=SEED, n_streams=len(streams), n_clips=int(sum(len(s["clips"]) for s in streams)),
        runtime_seconds=time.perf_counter() - t0,
        config=dict(threshold=THR, window=WIN, adwin_delta=R.ADWIN_DELTA, signal="standardised log q (OOF)",
                    S1="240 clean -> 240 degraded", S2="160 clean -> 160 degraded -> 160 clean",
                    S3="480 clean", inner_folds=R.N_INNER, folds=N_FOLDS, seeds=N_SEEDS,
                    monitor_cost_s_added_to_gated_systems=K["MON"],
                    cost_constants_s={k: float(v) for k, v in K.items() if k != "MON"},
                    temperatures=st["Ta"].to_dict(orient="records")),
        adwin_alarm_identity_checked=n_checked,
        checks=checks, scenarios=scen.to_dict(orient="records"), detection=det.to_dict(orient="records"),
        recovery=rec.to_dict(orient="records"), control=ctrl_sum, control_table=ctrl_ct.to_dict(orient="records"),
        stats=st_df.to_dict(orient="records"), per_fold=pf.to_dict(orient="records"),
        n_gating_events=len(ev_df))
    with open(RES / "slg_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))

    pd.set_option("display.width", 250)
    print("\nScenario table (mean over 6 folds x 20 seeds):")
    print(scen.round(4).to_string(index=False))
    print("\nDetection (S1):")
    print(det.round(3).to_string(index=False))
    print("\nRecovery (S2):")
    print(rec.round(3).to_string(index=False))
    print("\nControl (S3):")
    print(ctrl_ct.round(4).to_string(index=False))
    for k, v in ctrl_sum.items():
        print(f"  {k}: {v}")
    print("\nPer-fold degraded accuracy (mean over seeds):")
    print(pf.round(4).to_string(index=False))
    print("\nWilcoxon SLG vs LF, per-actor degraded accuracy:")
    print(st_df.round(4).to_string(index=False))
    print(f"\nADWIN alarm identity checked on {n_checked} channel-streams; gating events: {len(ev_df)}")
    print(f"Total runtime: {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
