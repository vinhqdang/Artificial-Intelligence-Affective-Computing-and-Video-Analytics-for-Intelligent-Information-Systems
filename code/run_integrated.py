#!/usr/bin/env python3
"""Integrated system (IS) evaluated on streams, seed 2026.

IS is an audio-first cascade with input monitors and stream-level gating:
  * always: audio decode, A0 (prosody), A1 (wav2vec2) and the audio monitor (on A0's input);
  * stage 1 = late fusion LF(A0, A1), temperature fitted per fold on training OOF, threshold tau
    per fold from run_experiments.operating_point; escalate to F-all when confidence < tau;
  * a visual monitor (on V0's input) is observed only when V0 is computed;
  * per-channel ADWIN and the recent-15 rule of code/run_slg.py gate the two channels.

Variant IS-A: video is decoded only when the visual path needs it (escalations, the visual path,
both-degraded, and every 8th clip while the visual channel is degraded).
Variant IS-B: video is decoded and V0 computed on every clip.

Baselines on the same streams: F-all late fusion, A1-first cascade (no monitors), a static
LF(A0,A1)-first cascade (reference for the sanity checks), and SLG on F-all (code/run_slg.py).
Streams S1-S3 reuse run_slg.make_streams; S4 is the speaker-blocked control defined here.

Run from the repository root:  python3 code/run_integrated.py
"""
import json
import pickle
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from river.drift import ADWIN
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_qag as Q  # noqa: E402
import run_slg as S  # noqa: E402
R = Q.R

SEED = R.SEED
RES, FIG, CACHE = R.RES, R.FIG, R.CACHE
F_ALL = R.F_ALL
N_FOLDS, HALF = R.N_FOLDS, R.HALF
N_SEEDS = R.N_DRIFT_SEEDS
THR, WIN = S.THR, S.WIN
N_S2 = S.N_S2
N_S4 = 2 * HALF
PROBE_EVERY = 8
CH = ("audio", "visual")
COND_NAMES = S.COND_NAMES
DEG = S.DEG
N_COND = len(COND_NAMES)
N_CLASSES = R.N_CLASSES
SEG_NAMES = ("clean_first", "deg", "clean_last")
METRICS = ("acc", "cost", "esc", "esc_out", "vdec", "gated", "abst")
SYS = ["F-all LF", "A1 cascade", "Static LF(A0,A1) cascade", "SLG (F-all)", "IS-A", "IS-B"]
MAIN = ["F-all LF", "A1 cascade", "SLG (F-all)", "IS-A", "IS-B"]
GATED_SYS = ["SLG (F-all)", "IS-A", "IS-B"]

# modes of IS at each clip
M_S1, M_ESC, M_VIS, M_S1D, M_BOTH = 0, 1, 2, 3, 4
MODE_NAME = {M_S1: "stage1", M_ESC: "escalated F-all", M_VIS: "visual path LF(V0,V1)",
             M_S1D: "stage1, visual degraded (probe)", M_BOTH: "F-all unreliable (both degraded)"}


# ----------------------------------------------------------------------------
# Integrated state machine
# ----------------------------------------------------------------------------
def run_is(za, zv, conf, tau, variant, gate=True):
    """Per-clip modes, effective channel states and visual observations for one stream.

    Decisions at clip t are based on observations up to t-1 and apply from clip t.
    The audio channel is observed every clip; the visual channel only when V0 is computed.
    """
    T = len(za)
    zs = {"audio": za, "visual": zv}
    state = {ch: False for ch in CH}
    det = {ch: ADWIN(delta=R.ADWIN_DELTA) for ch in CH}
    hist = {ch: [] for ch in CH}
    eff = {ch: np.zeros(T, dtype=bool) for ch in CH}
    mode = np.zeros(T, dtype=np.int8)
    obs_v = np.zeros(T, dtype=bool)
    events, alarms = [], {ch: [] for ch in CH}
    for t in range(T):
        aD, vD = state["audio"], state["visual"]
        eff["audio"][t], eff["visual"][t] = aD, vD
        if aD and vD:
            m = M_BOTH
        elif aD:
            m = M_VIS
        elif vD:
            m = M_S1D
        elif conf[t] < tau[t]:
            m = M_ESC
        else:
            m = M_S1
        video = (variant == "B") or (m in (M_ESC, M_VIS, M_BOTH)) or (m == M_S1D and t % PROBE_EVERY == 0)
        mode[t] = m
        obs_v[t] = video
        for ch in (("audio", "visual") if video else ("audio",)):
            z = float(zs[ch][t])
            hist[ch].append(z)
            det[ch].update(z)
            if det[ch].drift_detected:
                alarms[ch].append(t)
                if gate:
                    m15 = float(np.mean(hist[ch][-WIN:]))
                    if not state[ch] and m15 > THR:
                        state[ch] = True
                        events.append((t, ch, "DEGRADED", m15))
                    elif state[ch] and m15 < THR:
                        state[ch] = False
                        events.append((t, ch, "HEALTHY", m15))
    return dict(eff=eff, mode=mode, obs_v=obs_v, events=events, alarms=alarms, final=dict(state))


def is_cost(mode, obs_v, K, mon_a, mon_v):
    """Sum of the mean component costs executed for each clip, plus the monitors evaluated."""
    a1 = (mode != M_VIS).astype(float)
    v1 = np.isin(mode, (M_ESC, M_VIS, M_BOTH)).astype(float)
    ov = obs_v.astype(float)
    return (K["C_AUD"] + K["C_A0"] + mon_a + a1 * K["C_A1"]
            + ov * (K["C_VID"] + K["C_V0"] + mon_v) + v1 * K["C_V1"])


def is_pred(mode, pS, pV, pF):
    return np.where(np.isin(mode, (M_ESC, M_BOTH)), pF, np.where(mode == M_VIS, pV, pS))


# ----------------------------------------------------------------------------
# Streams: S1-S3 from run_slg, S4 speaker-blocked control defined here
# ----------------------------------------------------------------------------
def block_presentation(te, actor, rng, avoid=None):
    actors_te = np.unique(actor[te])
    while True:
        acts = rng.permutation(actors_te)
        if avoid is None or not np.array_equal(acts, avoid):
            break
    clips = np.concatenate([rng.permutation(te[actor[te] == a]) for a in acts])
    return acts, clips


def make_s4(st):
    out = []
    actor = st["actor"]
    for k, (tr, te) in enumerate(st["splits"]):
        for s in range(N_SEEDS):
            rng = np.random.default_rng([SEED, 81, k, s])
            acts1, p1 = block_presentation(te, actor, rng)
            _, p2 = block_presentation(te, actor, rng, avoid=acts1)
            out.append(dict(scn="S4", cond="speaker-blocked", fold=k, seed=s,
                            clips=np.concatenate([p1, p2]), conds=np.zeros(N_S4, int),
                            segs=dict(clean_first=(0, N_S4))))
    return out


# ----------------------------------------------------------------------------
# Per-stream evaluation of all systems
# ----------------------------------------------------------------------------
def evaluate_stream(s, st, A, K, mon_a, mon_v):
    ci, n = s["conds"], s["clips"]
    T = len(n)
    yn = st["y"][n]
    za = st["ZS"]["audio"][ci, n]
    zv = st["ZS"]["visual"][ci, n]
    conf, tau = A["CONF_S"][ci, n], A["TAU_S"][n]
    pS, pA1, pV, pF = A["PRED_S"][ci, n], A["PRED_A1"][ci, n], A["PRED_V"][ci, n], A["PRED_F"][ci, n]
    conf_a1, tau_a1 = A["CONF_A1"][ci, n], A["TAU_A1"][n]
    ones, zeros, nan = np.ones(T), np.zeros(T), np.full(T, np.nan)
    out = OrderedDict()
    out["F-all LF"] = dict(pred=pF, cost=np.full(T, K["C_ALL"]), esc=nan, esc_out=nan, vdec=ones,
                           gated=zeros, abst=nan)
    e1 = conf_a1 < tau_a1
    out["A1 cascade"] = dict(pred=np.where(e1, pF, pA1),
                             cost=K["C_AUD"] + K["C_A1"] + e1 * (K["C_ALL"] - K["C_AUD"] - K["C_A1"]),
                             esc=e1.astype(float), esc_out=e1.astype(float), vdec=e1.astype(float),
                             gated=zeros, abst=nan)
    pred_st, es = R.cascade_pred(conf, pS, pF, tau)
    out["Static LF(A0,A1) cascade"] = dict(
        pred=pred_st, cost=K["C_AUD"] + K["C_A0"] + K["C_A1"] + es * (K["C_VID"] + K["C_V0"] + K["C_V1"]),
        esc=es.astype(float), esc_out=es.astype(float), vdec=es.astype(float), gated=zeros, abst=nan)

    # SLG on F-all: run_slg state machine, run_slg cost convention (both monitors every clip)
    eff_slg, ev_slg, drifts_slg, fin_slg = S.run_slg({"audio": za, "visual": zv})
    aS, vS = Q.active_masks(eff_slg["audio"], eff_slg["visual"])
    pred_slg = (aS[:, None] * A["LOG_A"][ci, n] + vS[:, None] * A["LOG_V"][ci, n]).argmax(1)
    cost_slg = K["C_ALL"] - K["C_A1"] * (~aS) - K["C_V1"] * (~vS) + mon_a + mon_v
    out["SLG (F-all)"] = dict(pred=pred_slg, cost=cost_slg, esc=nan, esc_out=nan, vdec=ones,
                              gated=(eff_slg["audio"] | eff_slg["visual"]).astype(float), abst=nan)
    extras = {"SLG (F-all)": dict(events=ev_slg, eff=eff_slg, obs_v=np.ones(T, bool), alarms=drifts_slg,
                                  final=fin_slg)}

    for name, variant in (("IS-A", "A"), ("IS-B", "B")):
        r = run_is(za, zv, conf, tau, variant, gate=True)
        mode = r["mode"]
        vis_path = np.isin(mode, (M_ESC, M_VIS, M_BOTH))
        out[name] = dict(pred=is_pred(mode, pS, pV, pF),
                         cost=is_cost(mode, r["obs_v"], K, mon_a, mon_v),
                         esc=(conf < tau).astype(float), esc_out=vis_path.astype(float),
                         vdec=r["obs_v"].astype(float),
                         gated=(r["eff"]["audio"] | r["eff"]["visual"]).astype(float),
                         abst=(mode == M_BOTH).astype(float))
        extras[name] = dict(events=r["events"], eff=r["eff"], obs_v=r["obs_v"], alarms=r["alarms"],
                            final=r["final"], mode=mode)
    for name in out:
        out[name]["cor"] = out[name]["pred"] == yn
    return out, extras, dict(za=za, zv=zv, yn=yn, pS=pS, pV=pV, pF=pF, conf=conf, tau=tau)


# ----------------------------------------------------------------------------
# Aggregation helpers
# ----------------------------------------------------------------------------
def pick(events, ch=None, state=None, lo=None, hi=None):
    out = []
    for e in events:
        if ch is not None and e[1] != ch:
            continue
        if state is not None and e[2] != state:
            continue
        if lo is not None and e[0] < lo:
            continue
        if hi is not None and e[0] >= hi:
            continue
        out.append(e)
    return out


def qstats(v):
    v = np.asarray(v, float)
    if len(v) == 0:
        return dict(median=np.nan, q25=np.nan, q75=np.nan)
    return dict(median=float(np.median(v)), q25=float(np.percentile(v, 25)), q75=float(np.percentile(v, 75)))


def detection_table(extras, streams):
    rows = []
    for c in DEG:
        idx = [i for i, s in enumerate(streams) if s["scn"] == "S1" and s["cond"] == c]
        for sysname in ("IS-A", "IS-B", "SLG (F-all)"):
            for ch in CH:
                truth = ch in Q.DEGRADED_CH[c]
                delays, obs_delays, pre_marks, all_marks = [], [], 0, 0
                det_n = 0
                for i in idx:
                    ex = extras[i][sysname]
                    ev = ex["events"]
                    all_marks += len(pick(ev, ch=ch, state="DEGRADED"))
                    pre_marks += len(pick(ev, ch=ch, state="DEGRADED", hi=HALF))
                    post = pick(ev, ch=ch, state="DEGRADED", lo=HALF)
                    if truth and post:
                        det_n += 1
                        t_mark = post[0][0]
                        delays.append(t_mark + 1 - HALF)
                        if ch == "visual":
                            obs_delays.append(int(ex["obs_v"][HALF:t_mark + 1].sum()))
                n = len(idx)
                dq = qstats(delays)
                rows.append(dict(
                    system=sysname, condition=c, channel=ch, degraded_truth=truth, n_streams=n,
                    detection_rate=(det_n / n) if truth else np.nan,
                    delay_median=dq["median"], delay_q25=dq["q25"], delay_q75=dq["q75"],
                    visual_obs_delay_median=qstats(obs_delays)["median"] if ch == "visual" else np.nan,
                    pre_change_marks_per_1000_clips=1000.0 * pre_marks / (n * HALF),
                    all_marks_per_1000_clips=1000.0 * all_marks / (n * 2 * HALF)))
    return pd.DataFrame(rows)


def recovery_table(extras, streams):
    rows = []
    t_ret = 2 * N_S2
    for c in DEG:
        idx = [i for i, s in enumerate(streams) if s["scn"] == "S2" and s["cond"] == c]
        for sysname in ("IS-A", "IS-B", "SLG (F-all)"):
            for ch in CH:
                truth = ch in Q.DEGRADED_CH[c]
                det_n, det_delay = 0, []
                n_inst, n_rec, rec_delay = 0, 0, []
                for i in idx:
                    ex = extras[i][sysname]
                    ev = ex["events"]
                    if truth:
                        post = pick(ev, ch=ch, state="DEGRADED", lo=N_S2, hi=t_ret)
                        if post:
                            det_n += 1
                            det_delay.append(post[0][0] + 1 - N_S2)
                    if ex["eff"][ch][t_ret]:
                        n_inst += 1
                        h = pick(ev, ch=ch, state="HEALTHY", lo=t_ret)
                        if h and not ex["final"][ch]:
                            n_rec += 1
                            rec_delay.append(h[0][0] + 1 - t_ret)
                dd, rd = qstats(det_delay), qstats(rec_delay)
                n = len(idx)
                rows.append(dict(
                    system=sysname, condition=c, channel=ch, degraded_truth=truth, n_streams=n,
                    detection_rate_in_degraded_segment=(det_n / n) if truth else np.nan,
                    detection_delay_median=dd["median"], detection_delay_q25=dd["q25"],
                    detection_delay_q75=dd["q75"],
                    instances_marked_degraded_at_return=n_inst,
                    recovered_fraction=(n_rec / n_inst) if n_inst else np.nan,
                    recovery_delay_median=rd["median"], recovery_delay_q25=rd["q25"],
                    recovery_delay_q75=rd["q75"]))
    return pd.DataFrame(rows)


def control_table(sdf, extras, streams):
    rows = []
    for scn in ("S3", "S4"):
        idx = [i for i, s in enumerate(streams) if s["scn"] == scn]
        n_clips = len(idx) * len(streams[idx[0]]["clips"])
        sub = sdf[(sdf.scn == scn) & (sdf.segment == "clean_first")]
        for sysname in SYS:
            d = sub[sub.system == sysname]
            row = dict(scenario=scn, system=sysname, n_streams=len(idx),
                       accuracy=float(d.acc.mean()), mean_cost_s=float(d.cost.mean()),
                       escalation_rate_confidence=float(d.esc.mean()),
                       escalation_rate_visual_output=float(d.esc_out.mean()),
                       video_decoded_fraction=float(d.vdec.mean()),
                       fraction_clips_gated=float(d.gated.mean()),
                       fraction_abstained=float(d["abst"].mean()) if sysname.startswith("IS") else np.nan)
            if sysname in GATED_SYS:
                na = nv = 0
                streams_with = 0
                for i in idx:
                    ev = [e for e in extras[i][sysname]["events"] if e[2] == "DEGRADED"]
                    na += sum(1 for e in ev if e[1] == "audio")
                    nv += sum(1 for e in ev if e[1] == "visual")
                    streams_with += int(len(ev) > 0)
                row.update(false_marks_audio_per_1000=1000.0 * na / n_clips,
                           false_marks_visual_per_1000=1000.0 * nv / n_clips,
                           false_marks_total_per_1000=1000.0 * (na + nv) / n_clips,
                           pct_streams_with_any_mark=100.0 * streams_with / len(idx))
            else:
                row.update(false_marks_audio_per_1000=np.nan, false_marks_visual_per_1000=np.nan,
                           false_marks_total_per_1000=np.nan, pct_streams_with_any_mark=np.nan)
            rows.append(row)
    return pd.DataFrame(rows)


def wilcoxon_table(pa):
    rows = []
    for sysname in ("A1 cascade", "SLG (F-all)", "IS-A", "IS-B"):
        pvals, block = [], []
        for c in DEG:
            sub = pa[pa.condition == c].sort_values("actor")
            d = sub[sysname].values - sub["F-all LF"].values
            if np.all(d == 0):
                W, p = np.nan, 1.0
            else:
                r = stats.wilcoxon(sub[sysname].values, sub["F-all LF"].values)
                W, p = float(r.statistic), float(r.pvalue)
            pvals.append(p)
            block.append(dict(comparison=f"{sysname} vs F-all LF", condition=c,
                              mean_diff=float(d.mean()), median_diff=float(np.median(d)),
                              n_actors_better=int((d > 0).sum()), n_actors_worse=int((d < 0).sum()),
                              wilcoxon_W=W, p_value=p))
        adj = R.holm(pvals)
        for row, a in zip(block, adj):
            row["p_holm_8_conditions"] = float(a)
        rows += block
    return pd.DataFrame(rows)


def fig_integrated(ctrl, deg):
    """ctrl: control_table rows; deg: S1 degraded-segment rows with columns cond, system, acc."""
    R.set_style()
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7, 2.9), gridspec_kw=dict(width_ratios=[1, 1.7]))
    colours = {"F-all LF": R.PAL["blue"], "A1 cascade": R.PAL["sky"], "SLG (F-all)": R.PAL["orange"],
               "IS-A": R.PAL["vermil"], "IS-B": R.PAL["green"]}
    for sysname in MAIN:
        r = ctrl[(ctrl.scenario == "S3") & (ctrl.system == sysname)].iloc[0]
        ax.scatter(r.mean_cost_s, r.accuracy, color=colours[sysname], s=26, zorder=3, label=sysname)
    ax.set_xlabel("Mean CPU seconds per clip")
    ax.set_ylabel("Accuracy")
    ax.set_title("Clean streams (S3)", fontsize=8)
    ax.legend(frameon=False, fontsize=6, loc="lower right")
    x = np.arange(len(DEG))
    w = 0.16
    for i, sysname in enumerate(MAIN):
        vals = [deg[(deg.cond == c) & (deg.system == sysname)].acc.values[0] for c in DEG]
        bx.bar(x + (i - 2) * w, vals, w, color=colours[sysname], label=sysname)
    bx.set_xticks(x)
    bx.set_xticklabels(DEG, rotation=25, ha="right")
    bx.set_ylabel("Accuracy (degraded segment, S1)")
    bx.set_ylim(0, 0.9)
    bx.legend(frameon=False, ncol=5, fontsize=5.5, loc="upper center", bbox_to_anchor=(0.5, 1.14))
    R.save_fig(fig, "integrated.pdf")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    t0 = time.perf_counter()
    for d in (RES, FIG):
        d.mkdir(parents=True, exist_ok=True)
    st = S.build_state()
    y, actor, splits, N = st["y"], st["actor"], st["splits"], st["N"]
    K = dict(st["costs"])
    tim = pd.read_csv(RES / "qag_monitor_timing.csv")
    mon_a = float(tim.loc[tim.monitor == "audio", "single_clip_loop_us_per_clip"].iloc[0]) * 1e-6
    mon_v = float(tim.loc[tim.monitor == "visual", "single_clip_loop_us_per_clip"].iloc[0]) * 1e-6
    assert abs(mon_a + mon_v - K["MON"]) < 1e-12

    # ---- cached probabilities for every condition index ---------------------------------
    res = {m: [pickle.load(open(CACHE / f"{m}_fold{k}.pkl", "rb")) for k in range(N_FOLDS)] for m in F_ALL}
    for m in F_ALL:
        for k, (tr, te) in enumerate(splits):
            assert np.array_equal(res[m][k]["train_idx"], tr) and np.array_equal(res[m][k]["test_idx"], te)
    PR = {m: np.full((N_COND, N, N_CLASSES), np.nan) for m in F_ALL}
    for m in F_ALL:
        for k, (tr, te) in enumerate(splits):
            for ci, c in enumerate(COND_NAMES):
                PR[m][ci, te] = res[m][k]["test_probs"][R.mcond(m, c)]
        assert not np.isnan(PR[m]).any()
    LOG = {m: R.log_probs(PR[m]) for m in F_ALL}

    # ---- per-fold temperature and thresholds from training OOF ---------------------------
    TAU_S, T_S, TAU_A1 = (np.full(N, np.nan) for _ in range(3))
    thr_rows = []
    for k, (tr, te) in enumerate(splits):
        ytr = y[tr]
        G = {m: res[m][k]["oof_probs"] for m in F_ALL}
        pF_oof = R.fuse_lf([G[m] for m in F_ALL]).argmax(1)
        L = Q.lf_logits([G["A0"], G["A1"]])
        T = R.fit_temperature(L, ytr)
        conf = R.softmax(L / T).max(1)
        r_s, tau_s, _, accF = R.operating_point(conf, L.argmax(1), pF_oof, ytr)
        r_a, tau_a, _, _ = R.operating_point(G["A1"].max(1), G["A1"].argmax(1), pF_oof, ytr)
        TAU_S[te], T_S[te], TAU_A1[te] = tau_s, T, tau_a
        thr_rows.append(dict(fold=k + 1, T_stage1=T, tau_stage1=tau_s, r_star_stage1=r_s,
                             oof_acc_fall=accF, tau_A1_cascade=tau_a, r_star_A1_cascade=r_a))
    thr_df = pd.DataFrame(thr_rows)
    assert not np.isnan(TAU_S).any() and not np.isnan(TAU_A1).any()

    # the A1 thresholds must reproduce the existing A1 row of run_cascade_orders
    ref_op = pd.read_csv(RES / "cascade_orders_op.csv")
    ref_tau_a1 = np.array([float(x) for x in ref_op.loc[ref_op.stage1 == "A1", "tau_per_fold"].iloc[0].split(";")])
    tau_a1_diff = float(np.max(np.abs(thr_df.tau_A1_cascade.values - ref_tau_a1)))
    assert tau_a1_diff < 1e-12, tau_a1_diff

    # per-condition arrays indexed [condition, clip]
    Ls = LOG["A0"] + LOG["A1"]
    Lf = (LOG["A0"] + LOG["A1"]) + (LOG["V0"] + LOG["V1"])   # same summation order as run_slg
    A = dict(
        CONF_S=R.softmax(Ls / T_S[None, :, None]).max(-1),
        TAU_S=TAU_S, TAU_A1=TAU_A1,
        CONF_A1=PR["A1"].max(-1),
        PRED_S=Ls.argmax(-1), PRED_A1=PR["A1"].argmax(-1),
        PRED_V=(LOG["V0"] + LOG["V1"]).argmax(-1), PRED_F=Lf.argmax(-1),
        LOG_A=Ls, LOG_V=LOG["V0"] + LOG["V1"])

    # ---- streams ------------------------------------------------------------------------
    streams = S.make_streams(st) + make_s4(st)
    rows, extras, cor_checks = [], [], []
    pa_sum = {}
    pa_cnt = {}
    n_sim_lf_mismatch = n_sim_slg_mismatch = 0
    max_sim_cost_diff = 0.0
    alarm_audio_ok = alarm_visual_ok = 0
    gate_off_pred_ok = gate_off_checked = 0
    gate_off_cost_ok = 0
    zero_mark_streams = {"IS-A": 0, "IS-B": 0}
    zero_mark_pred_ok = {"IS-A": 0, "IS-B": 0}
    gate_off_static_max_cost_resid = 0.0
    for s in streams:
        out, ex, arr = evaluate_stream(s, st, A, K, mon_a, mon_v)
        extras.append(ex)

        # S.simulate is run_slg's own evaluation of the same stream: must agree
        sim = S.simulate(s, st)
        n_sim_lf_mismatch += int(np.sum(sim["cor"]["LF"] != out["F-all LF"]["cor"]))
        n_sim_slg_mismatch += int(np.sum(sim["cor"]["SLG"] != out["SLG (F-all)"]["cor"]))
        max_sim_cost_diff = max(max_sim_cost_diff, abs(sim["cost"]["SLG"] - out["SLG (F-all)"]["cost"].mean()))

        for name in ("IS-A", "IS-B"):
            e = ex[name]
            # audio detector sees every clip: identical to the batch helper
            if e["alarms"]["audio"] == R.adwin_alarms(arr["za"]):
                alarm_audio_ok += 1
            obs = np.where(e["obs_v"])[0]
            if e["alarms"]["visual"] == [int(obs[i]) for i in R.adwin_alarms(arr["zv"][obs])]:
                alarm_visual_ok += 1

        if s["scn"] in ("S3", "S4"):
            static = out["Static LF(A0,A1) cascade"]["pred"]
            static_cost = out["Static LF(A0,A1) cascade"]["cost"]
            esc_static = out["Static LF(A0,A1) cascade"]["esc"]
            for name, variant in (("IS-A", "A"), ("IS-B", "B")):
                off = run_is(arr["za"], arr["zv"], arr["conf"], arr["tau"], variant, gate=False)
                pred_off = is_pred(off["mode"], arr["pS"], arr["pV"], arr["pF"])
                gate_off_checked += 1
                gate_off_pred_ok += int(np.array_equal(pred_off, static))
                cost_off = is_cost(off["mode"], off["obs_v"], K, mon_a, mon_v)
                if variant == "A":   # video only on escalations, monitors as executed
                    expected = static_cost + mon_a + mon_v * off["obs_v"].astype(float)
                else:                # video on every clip: extra video cost on non-escalated clips
                    expected = static_cost + mon_a + mon_v + (K["C_VID"] + K["C_V0"]) * (1 - esc_static)
                resid = float(np.max(np.abs(cost_off - expected)))
                gate_off_static_max_cost_resid = max(gate_off_static_max_cost_resid, resid)
                gate_off_cost_ok += int(resid < 1e-12)
            if s["scn"] == "S3":
                for name in ("IS-A", "IS-B"):
                    if len(ex[name]["events"]) == 0:
                        zero_mark_streams[name] += 1
                        zero_mark_pred_ok[name] += int(np.array_equal(out[name]["pred"], static))

        for sysname, d in out.items():
            for seg, (a, b) in s["segs"].items():
                row = dict(scn=s["scn"], cond=s["cond"], fold=s["fold"] + 1, seed=s["seed"],
                           system=sysname, segment=seg)
                for m in METRICS:
                    v = d["cor"].astype(float) if m == "acc" else d[m]
                    row[m] = float(np.mean(v[a:b]))
                rows.append(row)
            if s["scn"] == "S1":
                a, b = s["segs"]["deg"]
                aid = actor[s["clips"][a:b]]
                key = (s["cond"], sysname)
                if key not in pa_sum:
                    pa_sum[key] = np.zeros(25)
                    pa_cnt[key] = np.zeros(25)
                pa_sum[key] += np.bincount(aid, weights=d["cor"][a:b].astype(float), minlength=25)
                pa_cnt[key] += np.bincount(aid, minlength=25)

    sdf = pd.DataFrame(rows)

    # per-actor degraded accuracy
    pa_rows = []
    for c in DEG:
        for a in range(1, 25):
            rec = dict(condition=c, actor=a)
            for sysname in SYS:
                key = (c, sysname)
                rec[sysname] = float(pa_sum[key][a] / pa_cnt[key][a])
            pa_rows.append(rec)
    pa = pd.DataFrame(pa_rows)
    pa = pa[["condition", "actor"] + SYS]

    scen = sdf.groupby(["scn", "cond", "system", "segment"], sort=False).agg(
        n_streams=("fold", "size"), acc=("acc", "mean"), cost=("cost", "mean"),
        esc_confidence=("esc", "mean"), esc_visual_output=("esc_out", "mean"),
        video_decoded=("vdec", "mean"), gated=("gated", "mean"), abstained=("abst", "mean")).reset_index()
    scen["n_streams"] = scen["n_streams"].astype(int)
    det = detection_table(extras, streams)
    rec = recovery_table(extras, streams)
    ctrl = control_table(sdf, extras, streams)
    st_df = wilcoxon_table(pa)

    # ---- checks ------------------------------------------------------------------------
    checks = [
        dict(check="A1 cascade taus equal run_cascade_orders A1 row (max abs diff)", value=tau_a1_diff,
             expected="< 1e-12", passed=bool(tau_a1_diff < 1e-12)),
        dict(check="F-all LF per-clip correctness equals run_slg.simulate (mismatched clips)",
             value=n_sim_lf_mismatch, expected=0, passed=n_sim_lf_mismatch == 0),
        dict(check="SLG per-clip correctness equals run_slg.simulate (mismatched clips)",
             value=n_sim_slg_mismatch, expected=0, passed=n_sim_slg_mismatch == 0),
        dict(check="SLG mean cost equals run_slg.simulate (max abs diff, s)", value=max_sim_cost_diff,
             expected="< 1e-9", passed=bool(max_sim_cost_diff < 1e-9)),
        dict(check="IS audio ADWIN alarms equal batch adwin_alarms on every clip (streams ok / total)",
             value=alarm_audio_ok, expected=2 * len(streams), passed=alarm_audio_ok == 2 * len(streams)),
        dict(check="IS visual ADWIN alarms equal batch adwin_alarms on observed clips (streams ok / total)",
             value=alarm_visual_ok, expected=2 * len(streams), passed=alarm_visual_ok == 2 * len(streams)),
        dict(check="No gating (S3, S4): IS predictions equal static LF(A0,A1) cascade (streams ok / checked)",
             value=gate_off_pred_ok, expected=gate_off_checked, passed=gate_off_pred_ok == gate_off_checked),
        dict(check="No gating (S3, S4): IS cost = static cost + audio monitor + visual monitor when observed (streams ok / checked)",
             value=gate_off_cost_ok, expected=gate_off_checked, passed=gate_off_cost_ok == gate_off_checked),
        dict(check="S3 IS-A streams with no DEGRADED mark: predictions equal static cascade (ok / streams)",
             value=zero_mark_pred_ok["IS-A"], expected=zero_mark_streams["IS-A"],
             passed=zero_mark_pred_ok["IS-A"] == zero_mark_streams["IS-A"]),
        dict(check="S3 IS-B streams with no DEGRADED mark: predictions equal static cascade (ok / streams)",
             value=zero_mark_pred_ok["IS-B"], expected=zero_mark_streams["IS-B"],
             passed=zero_mark_pred_ok["IS-B"] == zero_mark_streams["IS-B"]),
    ]
    checks_df = pd.DataFrame(checks)
    assert all(c["passed"] for c in checks), checks_df

    # reference: A1 cascade cost under run_cascade_orders convention (1-e) C_S + e C_ALL, S3 clean streams
    e_ref = sdf[(sdf.scn == "S3") & (sdf.system == "A1 cascade")].esc.mean()
    c_ref = (1 - e_ref) * (K["C_AUD"] + K["C_A1"]) + e_ref * K["C_ALL"]
    conv_row = dict(check="A1 cascade mean cost on S3 under run_cascade_orders convention (reference only)",
                    value=float(c_ref), expected="", passed=True)
    checks_df = pd.concat([checks_df, pd.DataFrame([conv_row])], ignore_index=True)

    # ---- outputs -----------------------------------------------------------------------
    ev_rows = []
    for s, ex in zip(streams, extras):
        for name in ("IS-A", "IS-B", "SLG (F-all)"):
            for (t, ch, new, m15) in ex[name]["events"]:
                ev_rows.append(dict(system=name, scenario=s["scn"], condition=s["cond"], fold=s["fold"] + 1,
                                    seed=s["seed"], clip_index=t, effective_from=t + 1, channel=ch,
                                    new_state=new, mean_last15=m15,
                                    channel_degraded_truth=(ch in Q.DEGRADED_CH[s["cond"]])))
    ev_df = pd.DataFrame(ev_rows)

    scen.to_csv(RES / "int_scenarios.csv", index=False)
    det.to_csv(RES / "int_detection.csv", index=False)
    rec.to_csv(RES / "int_recovery.csv", index=False)
    ctrl.to_csv(RES / "int_control.csv", index=False)
    st_df.to_csv(RES / "int_stats.csv", index=False)
    pa.to_csv(RES / "int_per_actor.csv", index=False)
    thr_df.to_csv(RES / "int_thresholds.csv", index=False)
    checks_df.to_csv(RES / "int_checks.csv", index=False)
    ev_df.to_csv(RES / "int_events.csv", index=False)

    deg_fig = scen[(scen.scn == "S1") & (scen.segment == "deg")]
    fig_integrated(ctrl, deg_fig)

    summary = dict(
        seed=SEED, n_streams=len(streams), n_clips=int(sum(len(s["clips"]) for s in streams)),
        runtime_seconds=time.perf_counter() - t0,
        config=dict(stage1="LF(A0,A1), per-fold temperature (training OOF), tau via operating_point",
                    visual_probe_every=PROBE_EVERY, threshold=THR, window=WIN, adwin_delta=R.ADWIN_DELTA,
                    monitor_cost_audio_s=mon_a, monitor_cost_visual_s=mon_v,
                    cost_constants_s={k: float(v) for k, v in K.items() if k != "MON"},
                    cost_convention="incremental: components executed for the clip (escalations pay only "
                                    "the components they add)",
                    folds=N_FOLDS, seeds=N_SEEDS),
        thresholds=thr_rows, checks=checks_df.to_dict(orient="records"),
        scenarios=scen.to_dict(orient="records"), detection=det.to_dict(orient="records"),
        recovery=rec.to_dict(orient="records"), control=ctrl.to_dict(orient="records"),
        stats=st_df.to_dict(orient="records"), n_gating_events=len(ev_df))
    with open(RES / "int_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 50)
    print("\nThresholds and temperatures per fold:")
    print(thr_df.round(4).to_string(index=False))
    print("\nChecks:")
    print(checks_df.to_string(index=False))
    print("\nScenario table:")
    print(scen.round(4).to_string(index=False))
    print("\nDetection (S1):")
    print(det.round(4).to_string(index=False))
    print("\nRecovery (S2):")
    print(rec.round(4).to_string(index=False))
    print("\nControl (S3, S4):")
    print(ctrl.round(4).to_string(index=False))
    print("\nWilcoxon vs F-all LF, per-actor degraded accuracy:")
    print(st_df.round(4).to_string(index=False))
    print(f"\nGating events: {len(ev_df)}; total runtime {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
