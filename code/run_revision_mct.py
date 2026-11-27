#!/usr/bin/env python3
"""Follow-up revision analyses with multi-condition-trained (MCT) models (seed 2026).

(a) Confidence per condition for A0, A1, V0, V1, LF(A0,A1) and F-all, raw and with a per-fold
    temperature fitted on the MCT out-of-fold fused logits; clean-trained rows for reference.
(b) Oracle gate (active models chosen from the known condition) with MCT and clean-trained models.
(c) Stream-level gating (run_slg machinery: same streams, monitors and ADWIN state machine) applied
    to MCT test probabilities, scenario S1 only (240 clean then 240 degraded clips).
(d) Cycle-1 style cascade with stage 1 = A1 (MCT), tau from MCT operating points.

Imports code/run_experiments.py, code/run_qag.py, code/run_cascade_orders.py, code/run_slg.py and
code/run_revision.py without modifying them. Run from the repository root:

    python3 code/run_revision_mct.py
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_experiments as R  # noqa: E402
import run_qag as RQ  # noqa: E402
import run_cascade_orders as RCO  # noqa: E402
import run_slg as SL  # noqa: E402
import run_revision as RV  # noqa: E402

RES = R.RES
MODELS = ["A0", "A1", "V0", "V1"]
F_ALL = R.F_ALL
CONDS = list(R.CONDITIONS.keys())
N_CLASSES = R.N_CLASSES
TOL = 1e-9


def mct_full(model, tr, te, C, train_mc, y, actor):
    """MCT model on stacked clean + degraded training clips; returns test probs and inner OOF probs."""
    X_clean = R.load_modality(model, "clean")
    X_deg = R.load_modality(model, train_mc)
    Xs = np.vstack([X_clean[tr], X_deg[tr]])
    ys = np.concatenate([y[tr], y[tr]])
    gs = np.concatenate([actor[tr], actor[tr]])
    with threadpool_limits(limits=1):
        oof = np.zeros((len(ys), N_CLASSES))
        for itr, iva in GroupKFold(R.N_INNER).split(Xs, ys, gs):
            pipe = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))
            pipe.fit(Xs[itr], ys[itr])
            oof[iva] = pipe.decision_function(Xs[iva])
        T = R.fit_temperature(oof, ys)
        pipe = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))
        pipe.fit(Xs, ys)
        probs = {mc: R.softmax(pipe.decision_function((X_clean if mc == "clean" else R.load_modality(model, mc))[te]) / T)
                 for mc in R.MODEL_MCONDS[model]}
    return dict(probs=probs, T=T, C=C, oof=R.softmax(oof / T))


def wil(a, b):
    d = a - b
    if np.all(d == 0):
        return np.nan, 1.0
    r = stats.wilcoxon(a, b)
    return float(r.statistic), float(r.pvalue)


def main():
    t0 = time.perf_counter()
    meta, y, actor, male = R.load_meta()
    N = len(y)
    splits = R.fold_splits(actor)
    cost_df, COST, _, _ = R.cost_tables(meta)
    comp = COST["components"]
    C_ALL = COST["F-all (all components)"]
    C_AUD, C_VID = comp["audio_decode"], comp["video_decode_face"]
    C_A0, C_A1, C_V0, C_V1 = comp["A0 prosody"], comp["A1 w2v"], comp["V0 mnet"], comp["V1 clip"]
    C_S_A1 = RCO.stage1_cost(("A1",), comp)
    checks = []

    def check(name, ours, ref, tol=TOL):
        diff = float(np.max(np.abs(np.asarray(ours, dtype=float) - np.asarray(ref, dtype=float))))
        checks.append(dict(check=name, max_abs_diff=diff, tolerance=tol, passed=bool(diff <= tol)))

    # ---- clean-trained cached models ---------------------------------------------------
    res = {m: [pickle.load(open(R.CACHE / f"{m}_fold{k}.pkl", "rb")) for k in range(R.N_FOLDS)] for m in MODELS}
    P = {}
    for m in MODELS:
        P[m] = {}
        for mc in R.MODEL_MCONDS[m]:
            arr = np.full((N, N_CLASSES), np.nan)
            for k in range(R.N_FOLDS):
                arr[res[m][k]["test_idx"]] = res[m][k]["test_probs"][mc]
            P[m][mc] = arr

    def PC(m, cond):
        return P[m][R.mcond(m, cond)]

    Ta_clean = np.full(N, np.nan)
    for k, (tr, te) in enumerate(splits):
        Ta_clean[te] = R.fit_temperature(RQ.lf_logits([res[m][k]["oof_probs"] for m in F_ALL]), y[tr])

    # ---- MCT models ---------------------------------------------------------------------------
    jobs = [(m, k) for m in MODELS for k in range(R.N_FOLDS)]
    outs = Parallel(n_jobs=R.N_JOBS)(
        delayed(mct_full)(m, splits[k][0], splits[k][1], res[m][k]["C"], RV.MCT_TRAIN_MC[m], y, actor)
        for m, k in jobs)
    PM = {m: {mc: np.full((N, N_CLASSES), np.nan) for mc in R.MODEL_MCONDS[m]} for m in MODELS}
    OOF = {m: [None] * R.N_FOLDS for m in MODELS}
    mct_T = {m: [None] * R.N_FOLDS for m in MODELS}
    for (m, k), o in zip(jobs, outs):
        te = splits[k][1]
        for mc, p in o["probs"].items():
            PM[m][mc][te] = p
        OOF[m][k] = o["oof"]
        mct_T[m][k] = o["T"]
    for m in MODELS:
        for mc in PM[m]:
            assert not np.isnan(PM[m][mc]).any()

    def PMc(m, cond):
        return PM[m][R.mcond(m, cond)]

    # per-fold temperature for the MCT F-all: fitted on the stacked OOF fused logits (both copies)
    Ta_mct = np.full(N, np.nan)
    tau_rows = []
    for k, (tr, te) in enumerate(splits):
        ys = np.concatenate([y[tr], y[tr]])
        L_oof = sum(R.log_probs(OOF[m][k]) for m in MODELS)
        Ta_mct[te] = R.fit_temperature(L_oof, ys)

    # ======================================================================
    # (a) Confidence per condition
    # ======================================================================
    conf_rows = []
    for regime, getter, Ta in (("clean-trained", PC, Ta_clean), ("MCT", PMc, Ta_mct)):
        for cond in CONDS:
            L_all = sum(R.log_probs(getter(m, cond)) for m in F_ALL)
            systems = OrderedDict([
                ("A0", getter("A0", cond)), ("A1", getter("A1", cond)),
                ("V0", getter("V0", cond)), ("V1", getter("V1", cond)),
                ("LF(A0,A1) raw", R.fuse_lf([getter("A0", cond), getter("A1", cond)])),
                ("F-all LF raw", R.softmax(L_all)),
                ("F-all LF temp-scaled", R.softmax(L_all / Ta[:, None])),
            ])
            for name, prob in systems.items():
                conf = prob.max(1)
                corr = (prob.argmax(1) == y).astype(float)
                conf_rows.append(dict(regime=regime, condition=cond, system=name, n_clips=N,
                                      mean_max_prob=float(conf.mean()), accuracy=float(corr.mean()),
                                      mean_conf_minus_acc=float(conf.mean() - corr.mean()),
                                      ece_15=R.ece(conf, corr, 15)))
    conf_df = pd.DataFrame(conf_rows)
    conf_df.to_csv(RES / "rev_mct2_confidence.csv", index=False)
    print("\n(a) Confidence per condition:")
    print(conf_df.round(4).to_string(index=False), flush=True)

    ref_conf = pd.read_csv(RES / "rev_confidence.csv")
    for _, r in conf_df[conf_df.regime == "clean-trained"].iterrows():
        ref = ref_conf[(ref_conf.condition == r.condition) & (ref_conf.system == r.system)].iloc[0]
        check(f"clean-trained confidence vs rev_confidence.csv ({r.system}, {r.condition})",
              [r.mean_max_prob, r.accuracy, r.ece_15], [ref.mean_max_prob, ref.accuracy, ref.ece_15])

    # ======================================================================
    # (b) Oracle gate, MCT and clean-trained
    # ======================================================================
    def oracle_eval(getter, cond):
        drop = RQ.ORACLE_DROP.get(cond)
        aA, vA = drop != "audio", drop != "visual"
        L = np.zeros((N, N_CLASSES))
        if aA:
            L += R.log_probs(getter("A0", cond)) + R.log_probs(getter("A1", cond))
        if vA:
            L += R.log_probs(getter("V0", cond)) + R.log_probs(getter("V1", cond))
        cost = (C_AUD if aA else 0.0) + (C_VID if vA else 0.0) \
            + (C_A0 + C_A1 if aA else 0.0) + (C_V0 + C_V1 if vA else 0.0)
        active = "+".join(([ "A0", "A1"] if aA else []) + (["V0", "V1"] if vA else []))
        return L.argmax(1), float(cost), active

    orc_rows = []
    for regime, getter in (("clean-trained", PC), ("MCT", PMc)):
        for cond in CONDS:
            pred, cost, active = oracle_eval(getter, cond)
            L_all = sum(R.log_probs(getter(m, cond)) for m in F_ALL).argmax(1)
            orc_rows.append(dict(regime=regime, condition=cond, active_models=active,
                                 accuracy=float(np.mean(pred == y)),
                                 macro_f1=float(R.macro_f1_from_cm(R.conf_matrix(y, pred))),
                                 mean_cost_s=cost,
                                 F_all_LF_accuracy=float(np.mean(L_all == y))))
    orc_df = pd.DataFrame(orc_rows)
    orc_df.to_csv(RES / "rev_mct2_oracle.csv", index=False)
    print("\n(b) Oracle gate:")
    print(orc_df.round(4).to_string(index=False), flush=True)

    qrob = pd.read_csv(RES / "qag_robustness.csv")
    for cond in CONDS:
        r = orc_df[(orc_df.regime == "clean-trained") & (orc_df.condition == cond)].iloc[0]
        q = qrob[(qrob.condition == cond) & (qrob.system == "Oracle gate")].iloc[0]
        check(f"clean-trained oracle vs qag_robustness.csv ({cond})",
              [r.accuracy, r.macro_f1, r.mean_cost_s], [q.accuracy, q.macro_f1, q.mean_cost_s])

    mct_rob = pd.read_csv(RES / "rev_mct.csv")
    for cond in CONDS:
        ref = mct_rob[(mct_rob.regime == "MCT") & (mct_rob.system == "F-all (LF)") & (mct_rob.condition == cond)]
        check(f"MCT F-all accuracy vs rev_mct.csv ({cond})",
              [np.mean(np.argmax(sum(R.log_probs(PMc(m, cond)) for m in F_ALL), 1) == y)],
              [ref.accuracy.values[0]])

    # ======================================================================
    # (c) Stream-level gating with MCT probabilities, scenario S1
    # ======================================================================
    st = SL.build_state()
    streams = [s for s in SL.make_streams(st) if s["scn"] == "S1"]
    st_clean = st
    st_mct = dict(st)
    st_mct["LPS"] = {m: np.stack([R.log_probs(PMc(m, c)) for c in CONDS]) for m in MODELS}

    def run_regime(stx):
        outs_ = [SL.simulate(s, stx) for s in streams]
        sdf = SL.stream_frame(streams, outs_)
        scen = SL.scenario_table(sdf)
        pa = SL.per_actor_tables(streams, outs_, actor)
        stt = SL.wilcoxon_tables(pa)
        return outs_, sdf, scen, stt

    outs_c, sdf_c, scen_c, stt_c = run_regime(st_clean)
    outs_m, sdf_m, scen_m, stt_m = run_regime(st_mct)

    # the SLG marks depend only on the monitors, which are identical in both regimes
    for oc, om in zip(outs_c, outs_m):
        assert oc["events"] == om["events"]
    # clean-trained reproduction of run_qag / run_slg reference numbers
    # every S1 stream opens with the full test fold (clean_first); its degraded segment is one condition
    ref_rob = pd.read_csv(RES / "robustness.csv")
    for cond in CONDS:
        if cond == "clean":
            seg = "clean_first"
            sub_c = sdf_c[(sdf_c.scn == "S1") & (sdf_c.system == "LF")]
            sub_m = sdf_m[(sdf_m.scn == "S1") & (sdf_m.system == "LF")]
        else:
            seg = "deg"
            sub_c = sdf_c[(sdf_c.scn == "S1") & (sdf_c.cond == cond) & (sdf_c.system == "LF")]
            sub_m = sdf_m[(sdf_m.scn == "S1") & (sdf_m.cond == cond) & (sdf_m.system == "LF")]
        ref = ref_rob[(ref_rob.condition == cond) & (ref_rob.system == "F-all (LF)")].accuracy.values[0]
        check(f"S1 clean-trained LF {seg} vs robustness.csv ({cond})", [sub_c[seg].mean()], [ref], 1e-12)
        ref_m = mct_rob[(mct_rob.regime == "MCT") & (mct_rob.system == "F-all (LF)") & (mct_rob.condition == cond)]
        check(f"S1 MCT LF {seg} vs rev_mct.csv ({cond})", [sub_m[seg].mean()], [ref_m.accuracy.values[0]], 1e-12)

    scen_rows = []
    for regime, scen in (("clean-trained", scen_c), ("MCT", scen_m)):
        s1 = scen[(scen.scn == "S1") & scen.system.isin(["LF (F-all)", "SLG"])].copy()
        s1.insert(0, "regime", regime)
        scen_rows.append(s1)
    scen_out = pd.concat(scen_rows, ignore_index=True)
    scen_out = scen_out.drop(columns=["scn"])
    scen_out.to_csv(RES / "rev_mct2_slg_scenarios.csv", index=False)
    print("\n(c) SLG scenario S1 (mean over 6 folds x 20 seeds):")
    print(scen_out.round(4).to_string(index=False), flush=True)

    stat_rows = []
    for regime, stt in (("clean-trained", stt_c), ("MCT", stt_m)):
        s1 = stt[stt.scn == "S1"].copy()
        s1.insert(0, "regime", regime)
        stat_rows.append(s1)
    stat_out = pd.concat(stat_rows, ignore_index=True).drop(columns=["scn"])
    stat_out.to_csv(RES / "rev_mct2_slg_stats.csv", index=False)
    print("\n(c) Wilcoxon SLG vs LF, per-actor degraded accuracy (Holm over 8):")
    print(stat_out.round(4).to_string(index=False), flush=True)

    # ======================================================================
    # (d) Cycle-1 style cascade with stage 1 = A1 (MCT)
    # ======================================================================
    tau_st = np.full(N, np.nan)
    tau_cl = np.full(N, np.nan)
    tau_ref = np.full(N, np.nan)
    for k, (tr, te) in enumerate(splits):
        n_tr = len(tr)
        ys = np.concatenate([y[tr], y[tr]])
        S_oof = OOF["A1"][k]
        FULL_oof = R.softmax(sum(R.log_probs(OOF[m][k]) for m in MODELS))
        r_s, t_s, _, _ = R.operating_point(S_oof.max(1), S_oof.argmax(1), FULL_oof.argmax(1), ys)
        Sc, Fc = S_oof[:n_tr], FULL_oof[:n_tr]
        r_c, t_c, _, _ = R.operating_point(Sc.max(1), Sc.argmax(1), Fc.argmax(1), y[tr])
        Sr = res["A1"][k]["oof_probs"]
        Fr = R.fuse_lf([res[m][k]["oof_probs"] for m in F_ALL])
        r_r, t_r, _, _ = R.operating_point(Sr.max(1), Sr.argmax(1), Fr.argmax(1), y[tr])
        tau_st[te], tau_cl[te], tau_ref[te] = t_s, t_c, t_r
        tau_rows.append(dict(fold=k + 1, tau_MCT_stacked=t_s, r_star_MCT_stacked=r_s,
                             tau_MCT_clean_copy=t_c, r_star_MCT_clean_copy=r_c,
                             tau_clean_trained=t_r, r_star_clean_trained=r_r))
    tau_df = pd.DataFrame(tau_rows)
    tau_df.to_csv(RES / "rev_mct2_cascade_folds.csv", index=False)

    def casc_eval(S_getter, tau_arr, cond, regime, tau_src):
        S = S_getter("A1", cond)
        FULL = R.fuse_lf([S_getter(m, cond) for m in F_ALL])
        conf = S.max(1)
        p1 = S.argmax(1)
        pF = FULL.argmax(1)
        esc = conf < tau_arr
        pred = np.where(esc, pF, p1)
        cost_clip = np.where(esc, C_ALL, C_S_A1)
        e = float(esc.mean())
        cost = float(cost_clip.mean())
        return dict(regime=regime, tau_source=tau_src, condition=cond, escalation_rate=e,
                    accuracy=float(np.mean(pred == y)),
                    macro_f1=float(R.macro_f1_from_cm(R.conf_matrix(y, pred))),
                    mean_cost_s=cost, saving_vs_F_all=1 - cost / C_ALL,
                    F_all_LF_accuracy=float(np.mean(FULL.argmax(1) == y)))

    casc_rows = []
    for cond in CONDS:
        casc_rows.append(casc_eval(PMc, tau_st, cond, "MCT", "MCT stacked OOF (primary)"))
        casc_rows.append(casc_eval(PMc, tau_cl, cond, "MCT", "MCT clean-copy OOF (sensitivity)"))
        casc_rows.append(casc_eval(PC, tau_ref, cond, "clean-trained", "clean-trained OOF (reference)"))
    casc_df = pd.DataFrame(casc_rows)
    casc_df.to_csv(RES / "rev_mct2_cascade.csv", index=False)
    print("\n(d) Cascade, stage 1 = A1:")
    print(casc_df.round(4).to_string(index=False), flush=True)
    print("\nPer-fold taus:")
    print(tau_df.round(4).to_string(index=False), flush=True)

    op_ref = pd.read_csv(RES / "cascade_orders_op.csv").set_index("stage1").loc["A1"]
    ref_row = casc_df[(casc_df.regime == "clean-trained") & (casc_df.condition == "clean")].iloc[0]
    check("clean-trained A1 cascade accuracy vs cascade_orders_op.csv", [ref_row.accuracy], [op_ref.accuracy])
    check("clean-trained A1 cascade escalation vs cascade_orders_op.csv", [ref_row.escalation_rate],
          [op_ref.escalation_rate])
    check("clean-trained A1 cascade mean cost vs cascade_orders_op.csv", [ref_row.mean_cost_s],
          [op_ref.mean_cost_s])

    sanity_df = pd.DataFrame(checks)
    n_fail = int((~sanity_df.passed).sum())
    print(f"\nSanity checks: {len(checks) - n_fail}/{len(checks)} passed", flush=True)
    if n_fail:
        print(sanity_df[~sanity_df.passed].to_string(index=False), flush=True)
        raise SystemExit("sanity checks FAILED")

    summary = dict(
        seed=R.SEED, n_clips=N, runtime_seconds=time.perf_counter() - t0,
        notes=dict(
            mct_temperature="Per fold, the fused MCT temperature is fitted on the stacked (clean + degraded) "
                            "inner OOF logits of the four MCT models.",
            tau="Primary: operating_point on the stacked MCT OOF (both copies). Sensitivity: clean copy only.",
            cost="Escalated clips charged C_ALL; others charged the A1 stage-1 cost (as run_cascade_orders).",
            slg="Monitors are the clean-fit input monitors of run_qag; SLG marks depend only on them, so "
                "the marks are identical in both regimes (asserted).",
            oracle="Known-condition active set, as in run_qag.oracle.",
        ),
        mct_T={m: [float(t) for t in mct_T[m]] for m in MODELS},
        mct_C={m: [float(res[m][k]["C"]) for k in range(R.N_FOLDS)] for m in MODELS},
        fused_T_mct=[float(Ta_mct[te[0]]) for _, te in splits],
        confidence=conf_rows, oracle=orc_rows, slg_scenarios=scen_out.to_dict(orient="records"),
        slg_stats=stat_out.to_dict(orient="records"), cascade=casc_rows,
        cascade_folds=tau_rows, sanity=checks)
    with open(RES / "rev_mct2_summary.json", "w") as f:
        json.dump(RV.to_py(summary), f, indent=2)
    print(f"\nTotal runtime: {time.perf_counter() - t0:.1f} s", flush=True)


if __name__ == "__main__":
    main()
