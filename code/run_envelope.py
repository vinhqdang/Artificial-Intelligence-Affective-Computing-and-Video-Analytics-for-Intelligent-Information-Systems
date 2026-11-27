#!/usr/bin/env python3
"""Envelope-aware gating on multi-condition-trained (MCT) models (seed 2026).

Models: MCT audio models (clean + snr10 training clips) and MCT visual models (clean + compress).
Two monitor sets per fold, both PCA-Mahalanobis with P99 thresholds as in code/run_qag.py:
  (i) clean-fit monitors: fitted on clean training features (as run_qag / run_slg);
  (ii) envelope-fit monitors: fitted on the MCT training features (audio: prosody clean + snr10;
       visual: mnet clean + compress), threshold from inner GroupKFold(4) by actor on the stacked data,
       standardisation of log q from the inner OOF.
Gating uses the run_slg ADWIN / recent-15 / margin-1.0 state machine on the chosen monitor set.
Clean-fit monitors are used for alerting (detection tables) and as a gating baseline.

Imports code/run_experiments.py, code/run_qag.py, code/run_slg.py, code/run_revision.py and
code/run_revision_mct.py without modifying them. Run from the repository root:

    python3 code/run_envelope.py
"""
import json
import pickle
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.model_selection import GroupKFold

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_experiments as R  # noqa: E402
import run_qag as Q  # noqa: E402
import run_slg as SL  # noqa: E402
import run_revision as RV  # noqa: E402
import run_revision_mct as RM  # noqa: E402

RES, FIG, FEAT = R.RES, R.FIG, R.FEAT
CONDS = list(R.CONDITIONS.keys())
DEG = R.DEG_CONDS
MODELS = ["A0", "A1", "V0", "V1"]
CH = ("audio", "visual")
TRAIN_MC = {"audio": "snr10", "visual": "compress"}
ENV_TRAIN_SET = {"audio": {"clean", "snr10"}, "visual": {"clean", "compress"}}
N_CLASSES = R.N_CLASSES
TOL = 1e-9


def monitor_features(ch, mc):
    name = "prosody" if ch == "audio" else "mnet"
    return np.load(FEAT / f"{name}_{mc}.npy").astype(np.float64)


def envelope_fold(Xc, Xd, tr, actor):
    """Envelope monitor fitted on stacked clean + degraded training clips of the training actors."""
    Xs = np.vstack([Xc[tr], Xd[tr]])
    gs = np.concatenate([actor[tr], actor[tr]])
    oof = np.zeros(len(Xs))
    for itr, iva in GroupKFold(R.N_INNER).split(Xs, groups=gs):
        oof[iva] = Q.monitor_d(Q.fit_monitor(Xs[itr]), Xs[iva])
    theta = float(np.percentile(oof, Q.PCT_MAIN))
    lq = np.log(oof / theta)
    return dict(model=Q.fit_monitor(Xs), theta=theta, mu=float(lq.mean()),
                sd=float(lq.std(ddof=1)), n_oof=len(Xs), oof_flag_rate=float(np.mean(oof > theta)))


def build_envelope(splits, actor, N):
    """Per-condition standardised log q of the envelope monitors on each test fold (pooled)."""
    LQ = {ch: {mc: np.full(N, np.nan) for mc in (R.AUDIO_CONDS if ch == "audio" else R.VIDEO_CONDS)} for ch in CH}
    Z = {ch: {mc: np.full(N, np.nan) for mc in LQ[ch]} for ch in CH}
    fold_rows = []
    for k, (tr, te) in enumerate(splits):
        for ch in CH:
            Xc = monitor_features(ch, "clean")
            Xd = monitor_features(ch, TRAIN_MC[ch])
            mon = envelope_fold(Xc, Xd, tr, actor)
            for mc in LQ[ch]:
                X = monitor_features(ch, mc)
                d = Q.monitor_d(mon["model"], X[te])
                LQ[ch][mc][te] = np.log(d / mon["theta"])
                Z[ch][mc][te] = (LQ[ch][mc][te] - mon["mu"]) / mon["sd"]
            fold_rows.append(dict(fold=k + 1, channel=ch, theta_p99=mon["theta"], oof_mu=mon["mu"],
                                  oof_sd=mon["sd"], n_train_stacked=mon["n_oof"],
                                  oof_flag_rate_p99=mon["oof_flag_rate"]))
    for ch in CH:
        for mc in LQ[ch]:
            assert not np.isnan(Z[ch][mc]).any() and not np.isnan(LQ[ch][mc]).any()
    return LQ, Z, pd.DataFrame(fold_rows)


def stack_by_condition(per_mc, ch):
    idx = 0 if ch == "audio" else 1
    return np.stack([per_mc[ch][R.CONDITIONS[c][idx]] for c in CONDS])


def per_actor_deg(streams, outs, system, actor):
    """Per-actor accuracy on the degraded segment of S1 streams, per condition."""
    acc = {}
    for s, o in zip(streams, outs):
        if s["scn"] != "S1":
            continue
        a, b = s["segs"]["deg"]
        aid = actor[s["clips"][a:b]]
        d = acc.setdefault(s["cond"], [np.zeros(25), np.zeros(25)])
        d[0] += np.bincount(aid, weights=o["cor"][system][a:b].astype(float), minlength=25)
        d[1] += np.bincount(aid, minlength=25)
    return {c: d[0][1:] / d[1][1:] for c, d in acc.items()}


def wil(a, b):
    d = a - b
    if np.all(d == 0):
        return np.nan, 1.0
    r = stats.wilcoxon(a, b)
    return float(r.statistic), float(r.pvalue)


def scen_table(streams, outs_by_key, specs):
    """specs: label -> (outs key, system). Returns degraded/clean-first accuracy and cost per S1 condition."""
    rows = []
    for label, (key, system) in specs.items():
        sdf = SL.stream_frame(streams, outs_by_key[key])
        sub = sdf[(sdf.scn == "S1") & (sdf.system == system)]
        for c in DEG:
            d = sub[sub.cond == c]
            rows.append(dict(system=label, condition=c, n_streams=len(d),
                             acc_clean_first=float(d.clean_first.mean()), acc_degraded=float(d.deg.mean()),
                             mean_cost_s=float(d.cost.mean())))
    return pd.DataFrame(rows)


def main():
    t0 = time.perf_counter()
    R.RES.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    st = SL.build_state()
    y, actor, splits, K, N = st["y"], st["actor"], st["splits"], st["costs"], st["N"]
    checks = []

    def check(name, ours, ref, tol=TOL):
        diff = float(np.max(np.abs(np.asarray(ours, dtype=float) - np.asarray(ref, dtype=float))))
        checks.append(dict(check=name, max_abs_diff=diff, tolerance=tol, passed=bool(diff <= tol)))

    # ---- MCT models (same code path as run_revision_mct) --------------------------------------
    meta_cache = {m: [pickle.load(open(R.CACHE / f"{m}_fold{k}.pkl", "rb")) for k in range(R.N_FOLDS)] for m in MODELS}
    jobs = [(m, k) for m in MODELS for k in range(R.N_FOLDS)]
    outs = Parallel(n_jobs=R.N_JOBS)(
        delayed(RM.mct_full)(m, splits[k][0], splits[k][1], meta_cache[m][k]["C"], RV.MCT_TRAIN_MC[m], y, actor)
        for m, k in jobs)
    PM = {m: {mc: np.full((N, N_CLASSES), np.nan) for mc in R.MODEL_MCONDS[m]} for m in MODELS}
    for (m, k), o in zip(jobs, outs):
        for mc, p in o["probs"].items():
            PM[m][mc][splits[k][1]] = p
    for m in MODELS:
        for mc in PM[m]:
            assert not np.isnan(PM[m][mc]).any()
    st_mct = dict(st)
    st_mct["LPS"] = {m: np.stack([R.log_probs(PM[m][R.mcond(m, c)]) for c in CONDS]) for m in MODELS}

    # ---- envelope monitors -----------------------------------------------------------------
    LQ_env, Z_env, env_fold_df = build_envelope(splits, actor, N)
    st_env = dict(st_mct)
    st_env["ZS"] = {ch: stack_by_condition(Z_env, ch) for ch in CH}

    # clean-fit monitor flags per mc (from run_slg state, per condition row)
    def flags_cf(ch, mc):
        idx = 0 if ch == "audio" else 1
        ci = next(i for i, c in enumerate(CONDS) if R.CONDITIONS[c][idx] == mc)
        return st["LQS"][ch][ci]

    # ---- flag rates (static, pooled test) ---------------------------------------------------
    flag_rows = []
    for ch in CH:
        for mc in (R.AUDIO_CONDS if ch == "audio" else R.VIDEO_CONDS):
            for mon_name, lq in (("envelope", LQ_env[ch][mc]), ("clean-fit", flags_cf(ch, mc))):
                flag_rows.append(dict(channel=ch, monitor=mon_name, mc=mc,
                                      in_training_envelope=mc in ENV_TRAIN_SET[ch],
                                      flag_rate_p99=float(np.mean(lq > 0)),
                                      mean_log_q=float(np.mean(lq)), n_clips=N))
    flag_df = pd.DataFrame(flag_rows)
    flag_df.to_csv(RES / "env_flag_rates.csv", index=False)
    print("\nStatic flag rates (pooled test, P99):")
    print(flag_df.round(4).to_string(index=False), flush=True)

    # ---- streams and simulations -------------------------------------------------------------
    streams = SL.make_streams(st)
    outs_k = {
        "clean": [SL.simulate(s, st) for s in streams],
        "mct_cf": [SL.simulate(s, st_mct) for s in streams],
        "mct_env": [SL.simulate(s, st_env) for s in streams],
    }
    print(f"\nSimulated {len(streams)} streams x 3 states ({time.perf_counter() - t0:.1f} s)", flush=True)

    # ---- checks against existing run_slg / run_revision outputs --------------------------------
    ref_slg = pd.read_csv(RES / "slg_scenarios.csv")
    sdf_clean = SL.stream_frame(streams, outs_k["clean"])
    for c in DEG:
        for sys_, label in (("LF", "LF (F-all)"), ("SLG", "SLG")):
            d = sdf_clean[(sdf_clean.scn == "S1") & (sdf_clean.cond == c) & (sdf_clean.system == sys_)]
            r = ref_slg[(ref_slg.scn == "S1") & (ref_slg.cond == c) & (ref_slg.system == label)].iloc[0]
            check(f"clean-trained {sys_} S1 degraded vs slg_scenarios.csv ({c})", [d.deg.mean()],
                  [r.acc_degraded], 1e-12)
            check(f"clean-trained {sys_} S1 cost vs slg_scenarios.csv ({c})", [d.cost.mean()],
                  [r.mean_cost_s], 1e-9)
    sdf_mct_cf = SL.stream_frame(streams, outs_k["mct_cf"])
    ref_mct2 = pd.read_csv(RES / "rev_mct2_slg_scenarios.csv")
    for c in DEG:
        d = sdf_mct_cf[(sdf_mct_cf.scn == "S1") & (sdf_mct_cf.cond == c) & (sdf_mct_cf.system == "SLG")]
        r = ref_mct2[(ref_mct2.regime == "MCT") & (ref_mct2.system == "SLG") & (ref_mct2.cond == c)].iloc[0]
        check(f"MCT clean-fit SLG S1 degraded vs rev_mct2_slg_scenarios.csv ({c})", [d.deg.mean()],
              [r.acc_degraded], 1e-12)
        check(f"MCT clean-fit SLG S1 cost vs rev_mct2_slg_scenarios.csv ({c})", [d.cost.mean()],
              [r.mean_cost_s], 1e-9)
        d_lf = sdf_mct_cf[(sdf_mct_cf.scn == "S1") & (sdf_mct_cf.cond == c) & (sdf_mct_cf.system == "LF")]
        rob = pd.read_csv(RES / "rev_mct.csv")
        rr = rob[(rob.regime == "MCT") & (rob.system == "F-all (LF)") & (rob.condition == c)].iloc[0]
        check(f"MCT LF S1 degraded vs rev_mct.csv ({c})", [d_lf.deg.mean()], [rr.accuracy], 1e-12)
    # SLG marks depend only on the monitors, so the clean-fit marks are identical in both regimes
    for a, b in zip(outs_k["clean"], outs_k["mct_cf"]):
        assert a["events"] == b["events"]
    ref_det = pd.read_csv(RES / "slg_detection.csv")
    det_clean = SL.detection_table(streams, outs_k["clean"])
    for c in DEG:
        for col in ("detection_rate", "delay_median", "pre_change_mark_rate",
                    "wrong_channel_degraded_marks_per_1000_clips"):
            a_ = det_clean[det_clean.condition == c][col].values[0]
            b_ = ref_det[ref_det.condition == c][col].values[0]
            check(f"clean-fit detection {col} vs slg_detection.csv ({c})", [a_], [b_], 1e-9)
    ref_ctrl = pd.read_csv(RES / "slg_control.csv")
    ct_clean, sum_clean = SL.control_table(streams, outs_k["clean"], sdf_clean)
    check("clean-trained control SLG accuracy vs slg_control.csv",
          [ct_clean[ct_clean.system == "SLG"].accuracy_clean.values[0]],
          [ref_ctrl[ref_ctrl.system == "SLG"].accuracy_clean.values[0]], 1e-12)
    ref_rec = pd.read_csv(RES / "slg_recovery.csv")
    rec_clean = SL.recovery_table(streams, outs_k["clean"])
    for c in DEG:
        check(f"clean-fit recovery fraction vs slg_recovery.csv ({c})",
              [rec_clean[rec_clean.condition == c].recovered_fraction_channels.values[0]],
              [ref_rec[ref_rec.condition == c].recovered_fraction_channels.values[0]], 1e-9)

    # ---- S1 scenario tables -----------------------------------------------------------------
    specs = OrderedDict([
        ("clean-trained LF", ("clean", "LF")),
        ("clean-trained SLG (clean-fit)", ("clean", "SLG")),
        ("MCT LF", ("mct_cf", "LF")),
        ("MCT+SLG (clean-fit)", ("mct_cf", "SLG")),
        ("MCT+SLG (envelope)", ("mct_env", "SLG")),
    ])
    scen = scen_table(streams, outs_k, specs)
    scen.to_csv(RES / "env_scenarios.csv", index=False)
    print("\nS1 scenario (mean over 6 folds x 20 seeds):")
    print(scen.round(4).to_string(index=False), flush=True)

    # ---- detection, control, recovery per monitor set -----------------------------------------
    det_rows = []
    for mon_name, key in (("clean-fit (alerting view)", "mct_cf"), ("envelope (gating view)", "mct_env")):
        d = SL.detection_table(streams, outs_k[key]).copy()
        d.insert(0, "monitor", mon_name)
        det_rows.append(d)
    det_df = pd.concat(det_rows, ignore_index=True)
    det_df.to_csv(RES / "env_detection.csv", index=False)
    print("\nDetection (S1):")
    print(det_df.round(3).to_string(index=False), flush=True)

    ctrl_rows, ctrl_sum = [], {}
    for mon_name, key in (("clean-fit", "mct_cf"), ("envelope", "mct_env")):
        sdf_k = SL.stream_frame(streams, outs_k[key])
        ct, summ = SL.control_table(streams, outs_k[key], sdf_k)
        ct = ct.copy()
        ct.insert(0, "monitor", mon_name)
        ctrl_rows.append(ct)
        ctrl_sum[mon_name] = summ
    ctrl_df = pd.concat(ctrl_rows, ignore_index=True)
    ctrl_df.to_csv(RES / "env_control.csv", index=False)
    print("\nControl (S3):")
    print(ctrl_df.round(4).to_string(index=False))
    for mon_name, summ in ctrl_sum.items():
        print(f"  {mon_name}: {summ}")

    rec_rows = []
    for mon_name, key in (("clean-fit", "mct_cf"), ("envelope", "mct_env")):
        r_ = SL.recovery_table(streams, outs_k[key]).copy()
        r_.insert(0, "monitor", mon_name)
        rec_rows.append(r_)
    rec_df = pd.concat(rec_rows, ignore_index=True)
    rec_df.to_csv(RES / "env_recovery.csv", index=False)
    print("\nRecovery (S2):")
    print(rec_df.round(3).to_string(index=False), flush=True)

    # ---- per-actor Wilcoxon, Holm over 8 -----------------------------------------------------
    pa_lf = per_actor_deg(streams, outs_k["mct_cf"], "LF", actor)
    pa_cf = per_actor_deg(streams, outs_k["mct_cf"], "SLG", actor)
    pa_env = per_actor_deg(streams, outs_k["mct_env"], "SLG", actor)
    stat_rows = []
    for label, pa_b in (("MCT+SLG (envelope) vs MCT LF", pa_env), ("MCT+SLG (clean-fit) vs MCT LF", pa_cf)):
        pvals, block = [], []
        for c in DEG:
            d = pa_b[c] - pa_lf[c]
            W, p = wil(pa_b[c], pa_lf[c])
            pvals.append(p)
            block.append(dict(comparison=label, condition=c, mean_diff=float(d.mean()),
                              median_diff=float(np.median(d)), n_actors_better=int((d > 0).sum()),
                              n_actors_worse=int((d < 0).sum()), wilcoxon_W=W, p_value=p))
        for row, adj in zip(block, R.holm(pvals)):
            row["p_holm_8_conditions"] = float(adj)
        stat_rows += block
    stat_df = pd.DataFrame(stat_rows)
    stat_df.to_csv(RES / "env_stats.csv", index=False)
    print("\nPer-actor Wilcoxon, degraded accuracy (Holm over 8):")
    print(stat_df.round(4).to_string(index=False), flush=True)

    # ---- figure ----------------------------------------------------------------------------
    R.set_style()
    series = [("clean-trained LF", "clean-trained LF", R.PAL["grey"]),
              ("clean-trained SLG (clean-fit)", "clean-trained SLG (clean-fit)", R.PAL["sky"]),
              ("MCT LF", "MCT LF", R.PAL["blue"]),
              ("MCT+SLG (clean-fit)", "MCT+SLG (clean-fit)", R.PAL["orange"]),
              ("MCT+SLG (envelope)", "MCT+SLG (envelope)", R.PAL["vermil"])]
    fig, ax = plt.subplots(figsize=(7, 2.7))
    x = np.arange(len(DEG))
    w = 0.16
    for i, (label, key, colour) in enumerate(series):
        vals = [scen[(scen.system == key) & (scen.condition == c)].acc_degraded.values[0] for c in DEG]
        ax.bar(x + (i - 2) * w, vals, w, color=colour, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(DEG, rotation=20, ha="right")
    ax.set_ylabel("Degraded-segment accuracy")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.15), fontsize=6)
    R.save_fig(fig, "envelope.pdf")

    summary = dict(
        seed=R.SEED, n_streams=len(streams), runtime_seconds=time.perf_counter() - t0,
        notes=dict(
            envelope="Per fold, envelope monitors fitted on stacked clean + degraded training clips "
                     "(audio: prosody clean + snr10; visual: mnet clean + compress), P99 from inner "
                     "GroupKFold(4) by actor, log q standardised with inner OOF mean and sd.",
            gating="run_slg state machine (ADWIN delta 0.002, recent-15 mean threshold 1.0). Predictions are "
                   "MCT late fusion over active models. Clean-fit and envelope rows differ only in monitors.",
            alerting="Clean-fit monitors are reported as the alerting view (detection tables).",
            control_false_alarms="Degraded marks per 1000 clips in S3 clean control streams.",
            clean_fit_marks="Clean-fit SLG marks are identical in the clean-trained and MCT regimes (asserted)."),
        envelope_folds=env_fold_df.to_dict(orient="records"),
        flag_rates=flag_rows,
        scenarios=scen.to_dict(orient="records"),
        detection=det_df.to_dict(orient="records"),
        control=ctrl_df.to_dict(orient="records"),
        control_summary=ctrl_sum,
        recovery=rec_df.to_dict(orient="records"),
        stats=stat_rows,
        checks=checks)
    with open(RES / "env_summary.json", "w") as f:
        json.dump(RV.to_py(summary), f, indent=2)
    n_fail = sum(not c["passed"] for c in checks)
    print(f"\nSanity checks: {len(checks) - n_fail}/{len(checks)} passed", flush=True)
    if n_fail:
        raise SystemExit("sanity checks FAILED")
    print(f"Total runtime: {time.perf_counter() - t0:.1f} s", flush=True)


if __name__ == "__main__":
    main()
