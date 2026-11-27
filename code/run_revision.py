#!/usr/bin/env python3
"""Revision analyses for the audio-visual emotion cascade study (seed 2026).

1. Confidence under degradation for per-model, two-model and four-model systems.
2. Multi-condition training (MCT) baseline: audio models see clean + snr10, visual models clean + compress.
3. Nested stage-1 selection for the cascade, using training out-of-fold predictions only.
4. Sex-stratified cluster bootstrap of the female-minus-male accuracy gap.
5. Exact random-escalation baseline for the eight stage-1 sets of the cascade sweep.

Reuses the per-fold caches and helpers of code/run_experiments.py, code/run_cascade_orders.py and
code/run_qag.py (imported, not modified). Run from the repository root:

    python3 code/run_revision.py
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
import run_cascade_orders as RCO  # noqa: E402
import run_qag as RQ  # noqa: E402

SEED = R.SEED
N_CLASSES = R.N_CLASSES
N_FOLDS = R.N_FOLDS
N_INNER = R.N_INNER
N_BOOT = 2000
GRID = R.GRID
CONDS = list(R.CONDITIONS.keys())
F_ALL = R.F_ALL
MODELS = ["A0", "A1", "V0", "V1"]
MCT_TRAIN_MC = {"A0": "snr10", "A1": "snr10", "V0": "compress", "V1": "compress"}
SEEN_CONDS = {"clean", "snr10", "compress", "snr10+compress"}
ECE_BINS = 15
TOL = 1e-9


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def wilcoxon(a, b):
    d = a - b
    if np.all(d == 0):
        return np.nan, 1.0
    r = stats.wilcoxon(a, b)
    return float(r.statistic), float(r.pvalue)


def acc_f1(y, pred):
    cm = R.conf_matrix(y, pred)
    return float(R.acc_from_cm(cm)), float(R.macro_f1_from_cm(cm))


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


def mct_fold(model, tr, te, C, train_mc, y, actor):
    """Train one modality model on stacked clean and degraded training clips of the training actors.

    C is the value selected for this model and fold in the clean-only cache (shortcut, see report).
    Temperature: fitted on inner GroupKFold(N_INNER) out-of-fold logits of the stacked set, groups = actor.
    """
    X_clean = R.load_modality(model, "clean")
    X_deg = R.load_modality(model, train_mc)
    Xs = np.vstack([X_clean[tr], X_deg[tr]])
    ys = np.concatenate([y[tr], y[tr]])
    gs = np.concatenate([actor[tr], actor[tr]])
    with threadpool_limits(limits=1):
        oof = np.zeros((len(ys), N_CLASSES))
        for itr, iva in GroupKFold(N_INNER).split(Xs, ys, gs):
            pipe = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))
            pipe.fit(Xs[itr], ys[itr])
            oof[iva] = pipe.decision_function(Xs[iva])
        T = R.fit_temperature(oof, ys)
        pipe = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))
        pipe.fit(Xs, ys)
        probs = {}
        for mc in R.MODEL_MCONDS[model]:
            X = X_clean if mc == "clean" else R.load_modality(model, mc)
            probs[mc] = R.softmax(pipe.decision_function(X[te]) / T)
    return dict(model=model, fold=None, T=T, C=C, probs=probs)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    t_start = time.perf_counter()
    R.RES.mkdir(parents=True, exist_ok=True)
    meta, y, actor, male = R.load_meta()
    N = len(y)
    splits = R.fold_splits(actor)
    cost_df, COST, _, _ = R.cost_tables(meta)
    comp = COST["components"]
    C_ALL = COST["F-all (all components)"]
    C_CHEAP = COST["F-cheap stage 1 (audio + video + A0 + V0)"]

    male_actors = sorted(set(actor[male].tolist()))
    female_actors = sorted(set(actor[~male].tolist()))
    assert male_actors == [a for a in range(1, 25) if a % 2 == 1], "sex mapping differs from run_experiments"
    assert female_actors == [a for a in range(1, 25) if a % 2 == 0]
    male_idx = np.array([a - 1 for a in male_actors])
    female_idx = np.array([a - 1 for a in female_actors])

    # ---- cached clean-trained models: pooled calibrated test probabilities ----------
    res = {m: [pickle.load(open(R.CACHE / f"{m}_fold{k}.pkl", "rb")) for k in range(N_FOLDS)] for m in MODELS}
    for m in MODELS:
        for k, (tr, te) in enumerate(splits):
            assert np.array_equal(res[m][k]["train_idx"], tr) and np.array_equal(res[m][k]["test_idx"], te)
    P = {}
    for m in MODELS:
        P[m] = {}
        for mc in R.MODEL_MCONDS[m]:
            arr = np.full((N, N_CLASSES), np.nan)
            for k in range(N_FOLDS):
                arr[res[m][k]["test_idx"]] = res[m][k]["test_probs"][mc]
            assert not np.isnan(arr).any(), (m, mc)
            P[m][mc] = arr

    def PR(m, cond):
        return P[m][R.mcond(m, cond)]

    # F-all temperature, fitted per fold on OOF fused logits exactly as in run_qag.py
    Ta = np.full(N, np.nan)
    Ta_fold = []
    for k, (tr, te) in enumerate(splits):
        L_oof = RQ.lf_logits([res[m][k]["oof_probs"] for m in F_ALL])
        Tk = R.fit_temperature(L_oof, y[tr])
        Ta[te] = Tk
        Ta_fold.append(Tk)
    assert not np.isnan(Ta).any()

    sanity = []

    def check(name, ours, ref, tol):
        diff = float(np.max(np.abs(np.asarray(ours, dtype=float) - np.asarray(ref, dtype=float))))
        sanity.append(dict(check=name, max_abs_diff=diff, tolerance=tol, passed=bool(diff <= tol)))

    # ======================================================================
    # 1. Confidence under degradation
    # ======================================================================
    conf_rows = []
    for cond in CONDS:
        L_all = sum(R.log_probs(PR(m, cond)) for m in F_ALL)
        systems = OrderedDict([
            ("A0", PR("A0", cond)),
            ("A1", PR("A1", cond)),
            ("V0", PR("V0", cond)),
            ("V1", PR("V1", cond)),
            ("LF(A0,A1) raw", R.fuse_lf([PR("A0", cond), PR("A1", cond)])),
            ("LF(A0,V0) raw", R.fuse_lf([PR("A0", cond), PR("V0", cond)])),
            ("F-all LF raw", R.softmax(L_all)),
            ("F-all LF temp-scaled", R.softmax(L_all / Ta[:, None])),
        ])
        for name, prob in systems.items():
            conf = prob.max(1)
            corr = (prob.argmax(1) == y).astype(float)
            conf_rows.append(dict(condition=cond, system=name, n_clips=N,
                                  mean_max_prob=float(conf.mean()), accuracy=float(corr.mean()),
                                  mean_conf_minus_acc=float(conf.mean() - corr.mean()),
                                  ece_15=R.ece(conf, corr, ECE_BINS)))
    conf_df = pd.DataFrame(conf_rows)
    conf_df.to_csv(R.RES / "rev_confidence.csv", index=False)
    print("\n[1] Confidence under degradation:")
    print(conf_df.round(4).to_string(index=False), flush=True)

    raw_clean = conf_df[(conf_df.condition == "clean") & (conf_df.system == "F-all LF raw")].iloc[0]
    temp_clean = conf_df[(conf_df.condition == "clean") & (conf_df.system == "F-all LF temp-scaled")].iloc[0]
    main_fall_acc = float(pd.read_csv(R.RES / "main_results.csv").set_index("system").loc["F-all (LF)", "accuracy"])
    check("clean F-all LF raw accuracy vs main_results.csv", [raw_clean.accuracy], [main_fall_acc], 1e-12)
    check("temperature preserves F-all accuracy (clean)", [temp_clean.accuracy], [raw_clean.accuracy], 1e-12)
    rob_ref = pd.read_csv(R.RES / "robustness.csv")
    for cond in CONDS:
        ref = rob_ref[(rob_ref.condition == cond) & (rob_ref.system == "F-all (LF)")].accuracy.values[0]
        mine = conf_df[(conf_df.condition == cond) & (conf_df.system == "F-all LF raw")].accuracy.values[0]
        check(f"F-all LF raw accuracy vs robustness.csv ({cond})", [mine], [ref], 1e-12)

    # ======================================================================
    # 2. Multi-condition training baseline
    # ======================================================================
    jobs = [(m, k) for m in MODELS for k in range(N_FOLDS)]
    outs = Parallel(n_jobs=R.N_JOBS)(
        delayed(mct_fold)(m, splits[k][0], splits[k][1], res[m][k]["C"], MCT_TRAIN_MC[m], y, actor)
        for m, k in jobs)
    P_mct = {m: {mc: np.full((N, N_CLASSES), np.nan) for mc in R.MODEL_MCONDS[m]} for m in MODELS}
    mct_hp = {m: [] for m in MODELS}
    for (m, k), out in zip(jobs, outs):
        te = splits[k][1]
        for mc, pr in out["probs"].items():
            P_mct[m][mc][te] = pr
        mct_hp[m].append(dict(fold=k + 1, C=out["C"], T=out["T"], train_mc=MCT_TRAIN_MC[m]))
    for m in MODELS:
        for mc in P_mct[m]:
            assert not np.isnan(P_mct[m][mc]).any(), (m, mc)

    def PM(m, cond):
        return P_mct[m][R.mcond(m, cond)]

    mct_rows = []
    pa_clean_fall, pa_mct_fall = {}, {}
    for regime, getter in (("clean-trained", PR), ("MCT", PM)):
        for cond in CONDS:
            seen = "yes" if cond in SEEN_CONDS else "no"
            systems = OrderedDict([(m, getter(m, cond)) for m in MODELS])
            systems["F-all (LF)"] = R.fuse_lf([getter(m, cond) for m in F_ALL])
            for name, prob in systems.items():
                pred = prob.argmax(1)
                acc, f1 = acc_f1(y, pred)
                pa = R.per_actor_acc(y, pred, actor)
                row = dict(regime=regime, system=name, condition=cond, seen_in_training=seen,
                           accuracy=acc, macro_f1=f1, n_clips=N,
                           acc_actor_mean=float(pa.mean()), acc_actor_min=float(pa.min()),
                           acc_actor_max=float(pa.max()))
                for a in range(24):
                    row[f"acc_actor_{a + 1:02d}"] = float(pa[a])
                mct_rows.append(row)
                if name == "F-all (LF)":
                    (pa_clean_fall if regime == "clean-trained" else pa_mct_fall)[cond] = pa
    mct_df = pd.DataFrame(mct_rows)
    mct_df.to_csv(R.RES / "rev_mct.csv", index=False)
    print("\n[2] MCT vs clean-trained (accuracy, macro-F1 by system):")
    print(mct_df[["regime", "system", "condition", "seen_in_training", "accuracy", "macro_f1",
                  "acc_actor_min", "acc_actor_max"]].round(4).to_string(index=False), flush=True)

    stat_rows, pvals = [], []
    for cond in CONDS:
        a, b = pa_mct_fall[cond], pa_clean_fall[cond]
        d = a - b
        W, p = wilcoxon(a, b)
        pvals.append(p)
        stat_rows.append(dict(condition=cond, seen_in_training=("yes" if cond in SEEN_CONDS else "no"),
                              clean_trained_F_all_acc=float(b.mean()), MCT_F_all_acc=float(a.mean()),
                              mean_diff_MCT_minus_clean=float(d.mean()), median_diff=float(np.median(d)),
                              n_actors_MCT_better=int((d > 0).sum()), n_actors_MCT_worse=int((d < 0).sum()),
                              wilcoxon_W=W, p_value=p))
    holm_adj = R.holm(pvals)
    for row, adj in zip(stat_rows, holm_adj):
        row["p_holm_9_conditions"] = float(adj)
    mct_stats_df = pd.DataFrame(stat_rows)
    mct_stats_df.to_csv(R.RES / "rev_mct_stats.csv", index=False)
    print("\n[2] Wilcoxon, MCT F-all minus clean-trained F-all (per-actor accuracy, Holm over 9):")
    print(mct_stats_df.round(4).to_string(index=False), flush=True)

    clean_ref = mct_df[(mct_df.regime == "clean-trained") & (mct_df.system == "F-all (LF)")
                       & (mct_df.condition == "clean")].accuracy.values[0]
    check("clean-trained F-all clean accuracy vs main_results.csv", [clean_ref], [main_fall_acc], 1e-12)

    # ======================================================================
    # 3. Nested stage-1 selection
    # ======================================================================
    comp_cost = OrderedDict((name, RCO.stage1_cost(models, comp)) for name, models in RCO.SETS.items())
    S_pool = OrderedDict((name, RCO.stage1_from(list(models), lambda m: P[m]["clean"]))
                         for name, models in RCO.SETS.items())
    FULL_pool = R.fuse_lf([P[m]["clean"] for m in F_ALL])
    pF = FULL_pool.argmax(1)

    nested_rows = []
    tau_by_fold = {name: [None] * N_FOLDS for name in RCO.SETS}
    chosen = []
    pred_n = np.full(N, -1)
    esc_n = np.zeros(N, dtype=bool)
    cost_n = np.zeros(N)
    for k, (tr, te) in enumerate(splits):
        G = lambda m, kk=k: res[m][kk]["oof_probs"]
        FULL_oof = R.fuse_lf([G(m) for m in F_ALL])
        cand = OrderedDict()
        for name, models in RCO.SETS.items():
            S_oof = RCO.stage1_from(list(models), G)
            r_star, tau, _, _ = R.operating_point(S_oof.max(1), S_oof.argmax(1), FULL_oof.argmax(1), y[tr])
            oof_cost = (1 - r_star) * comp_cost[name] + r_star * C_ALL
            cand[name] = dict(r_star=r_star, tau=tau, oof_cost=oof_cost)
            tau_by_fold[name][k] = tau
        best = min(cand, key=lambda n: cand[n]["oof_cost"])
        chosen.append(best)
        S = S_pool[best]
        conf = S.max(1)[te]
        p1 = S.argmax(1)[te]
        pFt = pF[te]
        pr, es = R.cascade_pred(conf, p1, pFt, cand[best]["tau"])
        pred_n[te] = pr
        esc_n[te] = es
        cost_n[te] = np.where(es, C_ALL, comp_cost[best])
        row = dict(level="fold", fold=k + 1, chosen_stage1=best, chosen_stage1_cost_s=comp_cost[best],
                   tau=cand[best]["tau"], oof_r_star=cand[best]["r_star"],
                   oof_mean_cost_s=cand[best]["oof_cost"],
                   test_escalation_rate=float(es.mean()), test_accuracy=float(np.mean(pr == y[te])),
                   test_F_all_accuracy=float(np.mean(pFt == y[te])))
        for name in RCO.SETS:
            row[f"oof_cost_{name}"] = cand[name]["oof_cost"]
            row[f"oof_r_star_{name}"] = cand[name]["r_star"]
            row[f"tau_{name}"] = cand[name]["tau"]
        nested_rows.append(row)

    pred_n_acc, pred_n_f1 = acc_f1(y, pred_n)
    pa_n = R.per_actor_acc(y, pred_n, actor)
    pa_F = R.per_actor_acc(y, pF, actor)
    W_n, p_n = wilcoxon(pa_n, pa_F)
    e_n = float(esc_n.mean())
    mean_cost_n = float(cost_n.mean())
    pooled = dict(level="pooled", fold=None, chosen_stage1=" | ".join(chosen),
                  test_accuracy=pred_n_acc, test_macro_f1=pred_n_f1, test_escalation_rate=e_n,
                  mean_cost_s=mean_cost_n, saving_vs_F_all=1 - mean_cost_n / C_ALL,
                  F_all_accuracy=float(np.mean(pF == y)), F_all_mean_cost_s=C_ALL,
                  mean_acc_diff_vs_F_all=float(np.mean(pa_n - pa_F)),
                  wilcoxon_W_vs_F_all=W_n, p_value_vs_F_all=p_n,
                  n_folds_chosen=json.dumps({s: chosen.count(s) for s in RCO.SETS if chosen.count(s)}))
    nested_df = pd.DataFrame(nested_rows + [pooled])
    nested_df.to_csv(R.RES / "rev_nested.csv", index=False)
    print("\n[3] Nested stage-1 selection, per fold:")
    print(nested_df[nested_df.level == "fold"][["fold", "chosen_stage1", "chosen_stage1_cost_s", "tau",
                                                "oof_r_star", "oof_mean_cost_s", "test_escalation_rate",
                                                "test_accuracy", "test_F_all_accuracy"]].round(4)
          .to_string(index=False), flush=True)
    print("[3] Pooled:", {k_: (round(v, 4) if isinstance(v, float) else v) for k_, v in pooled.items()},
          flush=True)

    # sanity: each candidate, with its own per-fold taus, reproduces cascade_orders_op.csv
    op_ref = pd.read_csv(R.RES / "cascade_orders_op.csv").set_index("stage1")
    for name in RCO.SETS:
        tau_vec = np.full(N, np.nan)
        for k, (tr, te) in enumerate(splits):
            tau_vec[te] = tau_by_fold[name][k]
        S = S_pool[name]
        pr_c, es_c = R.cascade_pred(S.max(1), S.argmax(1), pF, tau_vec)
        cost_c = np.where(es_c, C_ALL, comp_cost[name]).mean()
        ref = op_ref.loc[name]
        check(f"cascade_orders_op accuracy ({name})", [np.mean(pr_c == y)], [ref.accuracy], 1e-9)
        check(f"cascade_orders_op escalation ({name})", [es_c.mean()], [ref.escalation_rate], 1e-9)
        check(f"cascade_orders_op mean cost ({name})", [cost_c], [ref.mean_cost_s], 1e-9)
        ref_tau = np.array([float(x) for x in ref.tau_per_fold.split(";")])
        check(f"cascade_orders_op per-fold tau ({name})", np.array(tau_by_fold[name], dtype=float), ref_tau, 1e-9)
        check(f"stage-1 cost ({name})", [comp_cost[name]], [ref.stage1_cost_s], 1e-9)

    # ======================================================================
    # 4. Sex-stratified cluster bootstrap of the female-minus-male gap
    # ======================================================================
    fair_systems = OrderedDict([
        ("A0 prosody", PR("A0", "clean")),
        ("A1 w2v", PR("A1", "clean")),
        ("V0 mnet", PR("V0", "clean")),
        ("V1 clip", PR("V1", "clean")),
        ("F-cheap (LF)", R.fuse_lf([PR("A0", "clean"), PR("V0", "clean")])),
        ("F-all (LF)", FULL_pool),
    ])
    rng = np.random.default_rng(SEED)
    dm = rng.choice(male_idx, size=(N_BOOT, len(male_idx)))
    df_ = rng.choice(female_idx, size=(N_BOOT, len(female_idx)))
    gap_draws, gap_point, acc_sex = {}, {}, {}
    for name, prob in fair_systems.items():
        pred = prob.argmax(1)
        cm_a = R.actor_cms(y, pred, actor)
        acc_f = float(R.acc_from_cm(cm_a[female_idx].sum(axis=0)))
        acc_m = float(R.acc_from_cm(cm_a[male_idx].sum(axis=0)))
        acc_sex[name] = (acc_f, acc_m)
        gap_point[name] = acc_f - acc_m
        am = R.acc_from_cm(cm_a[dm].sum(axis=1))
        af = R.acc_from_cm(cm_a[df_].sum(axis=1))
        gap_draws[name] = af - am

    fair_rows = []
    for name in fair_systems:
        lo, hi = np.percentile(gap_draws[name], [2.5, 97.5])
        fair_rows.append(dict(quantity="accuracy gap (female minus male)", contrast=name,
                              acc_female=acc_sex[name][0], acc_male=acc_sex[name][1],
                              estimate=gap_point[name], ci_low=float(lo), ci_high=float(hi), n_boot=N_BOOT))
    for other in ("A1 w2v", "V1 clip"):
        d_point = gap_point["F-all (LF)"] - gap_point[other]
        d_draw = gap_draws["F-all (LF)"] - gap_draws[other]
        lo, hi = np.percentile(d_draw, [2.5, 97.5])
        fair_rows.append(dict(quantity="difference of gaps (F-all minus %s)" % other, contrast=f"F-all (LF) minus {other}",
                              acc_female=np.nan, acc_male=np.nan, estimate=float(d_point),
                              ci_low=float(lo), ci_high=float(hi), n_boot=N_BOOT))
    fair_df = pd.DataFrame(fair_rows)
    fair_df.to_csv(R.RES / "rev_fairness.csv", index=False)
    print("\n[4] Fairness (female minus male; cluster bootstrap over actors, stratified by sex):")
    print(fair_df.round(4).to_string(index=False), flush=True)

    # ======================================================================
    # 5. Exact random-escalation baseline for the eight stage-1 sets
    # ======================================================================
    sweep_ref = pd.read_csv(R.RES / "cascade_orders_sweep.csv")
    rand_rows = []
    below_summary = {}
    for name in RCO.SETS:
        S = S_pool[name]
        conf = S.max(1)
        p1 = S.argmax(1)
        acc1 = float(np.mean(p1 == y))
        accF_ = float(np.mean(pF == y))
        order = np.argsort(conf, kind="stable")
        below = []
        for r in GRID:
            k_esc = int(round(r * N))
            mask = np.zeros(N, dtype=bool)
            mask[order[:k_esc]] = True
            pred = np.where(mask, pF, p1)
            acc_c = float(np.mean(pred == y))
            f = k_esc / N
            acc_r = (1 - f) * acc1 + f * accF_
            is_below = bool(acc_c < acc_r - 1e-12)
            if is_below:
                below.append(float(r))
            ref_row = sweep_ref[(sweep_ref.stage1 == name) & np.isclose(sweep_ref.escalation_rate, r)]
            check(f"cascade_orders_sweep accuracy ({name}, r={r:.2f})", [acc_c], [ref_row.accuracy.values[0]], 1e-12)
            e_eff = f
            rand_rows.append(dict(
                stage1=name, escalation_rate=float(r), n_escalated=k_esc, escalation_rate_effective=e_eff,
                stage1_accuracy=acc1, F_all_accuracy=accF_,
                random_expected_accuracy=float(acc_r), confidence_cascade_accuracy=acc_c,
                cascade_minus_random=float(acc_c - acc_r), cascade_below_random=is_below,
                mean_cost_s=(1 - e_eff) * comp_cost[name] + e_eff * C_ALL))
        below_summary[name] = below
    rand_df = pd.DataFrame(rand_rows)
    rand_df.to_csv(R.RES / "rev_random.csv", index=False)
    print("\n[5] Random vs confidence-ordered escalation (pooled test, exact random expectation):")
    print(rand_df.round(4).to_string(index=False), flush=True)
    print("\nRates at which the confidence-ordered cascade is below random:")
    for name, rs in below_summary.items():
        print(f"  {name}: {rs if rs else 'none'}", flush=True)

    # ======================================================================
    # Outputs
    # ======================================================================
    fails = [s for s in sanity if not s["passed"]]
    sanity_df = pd.DataFrame(sanity)
    n_checks = len(sanity)
    print(f"\nSanity checks: {n_checks - len(fails)}/{n_checks} passed", flush=True)
    if fails:
        print(pd.DataFrame(fails).to_string(index=False), flush=True)
        raise SystemExit("sanity checks FAILED")

    summary = dict(
        seed=SEED, n_clips=N, n_actors=24, n_boot=N_BOOT, runtime_seconds=time.perf_counter() - t_start,
        notes=dict(
            mct_C="Per model and fold, C is taken from the clean-only cache (shortcut); the temperature is refit "
                  "on the stacked training set's inner OOF logits.",
            mct_inference_cost="Same feature inputs, model class and cost constants as the clean-trained models: "
                               "no additional inference cost; MCT changes training data only.",
            fairness_sign="All gaps are female minus male accuracy (signed). Bootstrap uses common actor draws "
                          "across systems so differences of gaps are paired.",
            nested_cost="Escalated clips are charged C_ALL and non-escalated clips their stage-1 cost, "
                        "as in run_cascade_orders.py.",
            random_baseline="Expected accuracy of a uniformly random escalation set of size k = round(r*N) is "
                            "exactly (1-k/N)*acc_stage1 + (k/N)*acc_F-all.",
        ),
        F_all_temperature_per_fold=Ta_fold,
        mct_hyperparameters=mct_hp,
        confidence=conf_rows,
        mct=mct_rows,
        mct_stats=stat_rows,
        nested_folds=nested_rows,
        nested_pooled=pooled,
        fairness=fair_rows,
        random=rand_rows,
        random_below_random_rates={k_: v for k_, v in below_summary.items()},
        sanity=sanity,
    )
    with open(R.RES / "rev_summary.json", "w") as f:
        json.dump(to_py(summary), f, indent=2)
    print(f"\nTotal runtime: {time.perf_counter() - t_start:.1f} s", flush=True)


if __name__ == "__main__":
    main()
