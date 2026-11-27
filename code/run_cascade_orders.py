#!/usr/bin/env python3
"""Stage-1 configuration study for the audio-visual emotion cascade (seed 2026).

Evaluates which set of models forms the stage-1 late fusion S (escalating to F-all when the
stage-1 confidence is below a threshold). Reuses the per-fold caches and helpers of
code/run_experiments.py; that module is imported, not modified.

Run from the repository root:  python3 code/run_cascade_orders.py
"""
import json
import pickle
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_experiments as R  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

F_ALL = ["A0", "A1", "V0", "V1"]
SETS = OrderedDict([
    ("A0", ("A0",)),
    ("V0", ("V0",)),
    ("A0+V0", ("A0", "V0")),
    ("A1", ("A1",)),
    ("A0+A1", ("A0", "A1")),
    ("A1+V0", ("A1", "V0")),
    ("A0+A1+V0", ("A0", "A1", "V0")),
    ("V1", ("V1",)),
])
PLOT_SETS = ["A0+V0", "A1", "A0+A1", "A1+V0", "A0"]
ISO_THRESHOLDS = [0.80, 0.82]
COMP_NAME = {"A0": "A0 prosody", "A1": "A1 w2v", "V0": "V0 mnet", "V1": "V1 clip"}
TOL = 1e-9


def stage1_cost(models, comp):
    """Decode costs needed by the set plus the cost of each model in it (mean CPU seconds per clip)."""
    c = 0.0
    if any(m.startswith("A") for m in models):
        c += comp["audio_decode"]
    if any(m.startswith("V") for m in models):
        c += comp["video_decode_face"]
    for m in models:
        c += comp[COMP_NAME[m]]
    return float(c)


def stage1_from(models, get):
    if len(models) == 1:
        return get(models[0])
    return R.fuse_lf([get(m) for m in models])


def load_fold_results(model):
    return [pickle.load(open(R.CACHE / f"{model}_fold{k}.pkl", "rb")) for k in range(R.N_FOLDS)]


def pooled_test(results, model, n):
    arr = np.full((n, R.N_CLASSES), np.nan)
    for res in results[model]:
        arr[res["test_idx"]] = res["test_probs"]["clean"]
    assert not np.isnan(arr).any(), model
    return arr


def wilcoxon(a, b):
    d = a - b
    if np.all(d == 0):
        return np.nan, 1.0
    r = stats.wilcoxon(a, b)
    return float(r.statistic), float(r.pvalue)


def pareto_flags(df):
    order = df.sort_values(["mean_cost_s", "accuracy"], ascending=[True, False]).index
    best = -np.inf
    flags = {}
    for i in order:
        a = df.at[i, "accuracy"]
        flags[i] = bool(a > best + 1e-15)
        best = max(best, a)
    return pd.Series(flags).reindex(df.index).values


def main():
    meta, y, actor, male = R.load_meta()
    N = len(y)
    splits = R.fold_splits(actor)
    cost_df, COST, _, _ = R.cost_tables(meta)
    comp = COST["components"]
    C_ALL = COST["F-all (all components)"]
    C_CHEAP = COST["F-cheap stage 1 (audio + video + A0 + V0)"]

    results = {m: load_fold_results(m) for m in ["A0", "A1", "V0", "V1", "EF", "V1ZS"]}
    P = {m: pooled_test(results, m, N) for m in ["A0", "A1", "V0", "V1", "EF", "V1ZS"]}

    FULL_test = R.fuse_lf([P[m] for m in F_ALL])
    pF = FULL_test.argmax(1)
    accF = float(np.mean(pF == y))
    pa_F = R.per_actor_acc(y, pF, actor)
    cm_F = R.actor_cms(y, pF, actor)
    acc_ci_F, f1_ci_F = R.cluster_boot(cm_F, R.SEED)
    f1_F = float(R.macro_f1_from_cm(R.conf_matrix(y, pF)))

    sweep_rows, op_rows, iso_rows = [], [], []
    op_detail = {}
    wil_p = []
    for name, models in SETS.items():
        C_S = stage1_cost(models, comp)
        models = list(models)

        # ---- full sweep on pooled test (global ranking by stage-1 max probability) ----
        S_test = stage1_from(models, lambda m: P[m])
        conf = S_test.max(1)
        p1 = S_test.argmax(1)
        order = np.argsort(conf, kind="stable")
        for r in R.GRID:
            k_esc = int(round(r * N))
            mask = np.zeros(N, dtype=bool)
            mask[order[:k_esc]] = True
            pred = np.where(mask, pF, p1)
            e = mask.mean()
            tau_g = float(conf[order[k_esc]]) if k_esc < N else 1.0
            sweep_rows.append(dict(
                stage1=name, escalation_rate=float(r), tau_global=tau_g,
                stage1_cost_s=C_S,
                accuracy=float(np.mean(pred == y)),
                macro_f1=float(R.macro_f1_from_cm(R.conf_matrix(y, pred))),
                mean_cost_s=(1 - e) * C_S + e * C_ALL,
                saving_vs_F_all=1 - ((1 - e) * C_S + e * C_ALL) / C_ALL))

        # ---- operating point: per-fold tau from training OOF predictions -------------
        tau_arr = np.full(N, np.nan)
        folds = []
        for k, (tr, te) in enumerate(splits):
            G = lambda m, kk=k: results[m][kk]["oof_probs"]
            FULL_oof = R.fuse_lf([G(m) for m in F_ALL])
            S_oof = stage1_from(models, G)
            r_star, tau, curve, accF_oof = R.operating_point(S_oof.max(1), S_oof.argmax(1),
                                                             FULL_oof.argmax(1), y[tr])
            tau_arr[te] = tau
            folds.append(dict(fold=k + 1, r_star=r_star, tau=tau, oof_acc_fall=accF_oof,
                              oof_acc_at_op=float(curve[R.GRID.tolist().index(r_star)])))
        assert not np.isnan(tau_arr).any()
        pred_op, esc = R.cascade_pred(conf, p1, pF, tau_arr)
        e = float(esc.mean())
        cost_op = (1 - e) * C_S + e * C_ALL
        pa = R.per_actor_acc(y, pred_op, actor)
        cm = R.actor_cms(y, pred_op, actor)
        acc_ci, f1_ci = R.cluster_boot(cm, R.SEED)
        s_w, p_w = wilcoxon(pa, pa_F)
        wil_p.append(p_w)
        row = dict(
            stage1=name, stage1_cost_s=C_S, mean_cost_s=cost_op, saving_vs_F_all=1 - cost_op / C_ALL,
            escalation_rate=e,
            accuracy=float(np.mean(pred_op == y)), acc_ci_low=acc_ci[0], acc_ci_high=acc_ci[1],
            macro_f1=float(R.macro_f1_from_cm(R.conf_matrix(y, pred_op))),
            f1_ci_low=f1_ci[0], f1_ci_high=f1_ci[1],
            mean_acc_diff_vs_F_all=float(np.mean(pa - pa_F)),
            wilcoxon_W_vs_F_all=s_w, p_value_vs_F_all=p_w,
            tau_per_fold=";".join(repr(float(f["tau"])) for f in folds),
            r_star_per_fold=";".join(f"{f['r_star']:.2f}" for f in folds),
        )
        for a in range(24):
            row[f"acc_actor_{a + 1:02d}"] = float(pa[a])
        op_rows.append(row)
        op_detail[name] = dict(folds=folds, pred=pred_op, esc=esc, conf=conf, p1=p1)

        # ---- iso-accuracy on the test sweep (descriptive, test-set based) ----
        sw = pd.DataFrame([s for s in sweep_rows if s["stage1"] == name])
        for thr in ISO_THRESHOLDS:
            ok = sw[sw.accuracy >= thr - 1e-12]
            if len(ok):
                best = ok.sort_values("mean_cost_s").iloc[0]
                iso_rows.append(dict(stage1=name, accuracy_threshold=thr, reachable=True,
                                     min_mean_cost_s=float(best.mean_cost_s),
                                     escalation_rate=float(best.escalation_rate),
                                     accuracy_at_point=float(best.accuracy),
                                     macro_f1_at_point=float(best.macro_f1)))
            else:
                iso_rows.append(dict(stage1=name, accuracy_threshold=thr, reachable=False,
                                     min_mean_cost_s=np.nan, escalation_rate=np.nan,
                                     accuracy_at_point=np.nan, macro_f1_at_point=np.nan))

    # F-all reference row in the operating-point table
    f_row = dict(stage1="F-all (no cascade)", stage1_cost_s=C_ALL, mean_cost_s=C_ALL, saving_vs_F_all=0.0,
                 escalation_rate=1.0, accuracy=accF, acc_ci_low=acc_ci_F[0], acc_ci_high=acc_ci_F[1],
                 macro_f1=f1_F, f1_ci_low=f1_ci_F[0], f1_ci_high=f1_ci_F[1], mean_acc_diff_vs_F_all=0.0,
                 wilcoxon_W_vs_F_all=np.nan, p_value_vs_F_all=np.nan, tau_per_fold="", r_star_per_fold="")
    for a in range(24):
        f_row[f"acc_actor_{a + 1:02d}"] = float(pa_F[a])
    op_df = pd.DataFrame([f_row] + op_rows)
    op_df["p_holm_vs_F_all"] = np.nan
    op_df.loc[op_df.stage1 != "F-all (no cascade)", "p_holm_vs_F_all"] = R.holm(wil_p)
    op_df.to_csv(R.RES / "cascade_orders_op.csv", index=False)

    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv(R.RES / "cascade_orders_sweep.csv", index=False)

    iso_df = pd.DataFrame(iso_rows)
    iso_df.to_csv(R.RES / "cascade_orders_iso.csv", index=False)

    # ---- Pareto frontier over sweep points and single / fused systems ----------------
    sys_defs = OrderedDict([
        ("A0 alone", (COST["A0 alone (audio decode + A0)"], P["A0"].argmax(1))),
        ("A1 alone", (COST["A1 alone (audio decode + A1)"], P["A1"].argmax(1))),
        ("V0 alone", (COST["V0 alone (video decode + V0)"], P["V0"].argmax(1))),
        ("V1 alone", (COST["V1 alone (video decode + V1)"], P["V1"].argmax(1))),
        ("V1-ZS", (COST["V1 alone (video decode + V1)"], P["V1ZS"].argmax(1))),
        ("EF-full", (COST["F-full (audio + video + A1 + V1)"], P["EF"].argmax(1))),
        ("F-cheap (LF)", (C_CHEAP, R.fuse_lf([P["A0"], P["V0"]]).argmax(1))),
        ("F-full (LF)", (COST["F-full (audio + video + A1 + V1)"], R.fuse_lf([P["A1"], P["V1"]]).argmax(1))),
        ("F-all (LF)", (C_ALL, pF)),
        ("F-all (CWF)", (C_ALL, R.fuse_cwf([P[m] for m in F_ALL]).argmax(1))),
    ])
    pts = []
    for _, r in sweep_df.iterrows():
        pts.append(dict(kind="cascade", name=f"{r.stage1} @ {r.escalation_rate:.2f}",
                        escalation_rate=r.escalation_rate, mean_cost_s=r.mean_cost_s,
                        accuracy=r.accuracy, macro_f1=r.macro_f1))
    for nm, (c, pred) in sys_defs.items():
        pts.append(dict(kind="system", name=nm, escalation_rate=np.nan, mean_cost_s=c,
                        accuracy=float(np.mean(pred == y)),
                        macro_f1=float(R.macro_f1_from_cm(R.conf_matrix(y, pred)))))
    pareto_df = pd.DataFrame(pts)
    pareto_df["on_frontier"] = pareto_flags(pareto_df)
    pareto_df = pareto_df.sort_values(["on_frontier", "mean_cost_s"], ascending=[False, True])
    pareto_df.to_csv(R.RES / "cascade_orders_pareto.csv", index=False)

    # ---- sanity checks against the existing run ------------------------------------
    summ = json.load(open(R.RES / "summary.json"))
    ref_op = {d["variant"]: d for d in summ["cascade_operating_points_pooled"]}
    ref_fold = summ["cascade_operating_points_per_fold"]
    ref_sweep = pd.read_csv(R.RES / "cascade_sweep.csv")
    ref_main = pd.read_csv(R.RES / "main_results.csv").set_index("system")
    checks = []

    def add(check, ours, ref, tol=TOL):
        diff = float(np.max(np.abs(np.asarray(ours, dtype=float) - np.asarray(ref, dtype=float))))
        checks.append(dict(check=check, ours=float(np.max(np.abs(ours))) if np.ndim(ours) else float(ours),
                           reference=float(np.max(np.abs(ref))) if np.ndim(ref) else float(ref),
                           max_abs_diff=diff, passed=bool(diff <= tol)))

    for variant, name in (("A", "A0+V0"), ("B", "A0")):
        o = op_df[op_df.stage1 == name].iloc[0]
        ref = ref_op[variant]
        add(f"variant {variant} ({name}) op escalation_rate", o.escalation_rate, ref["escalation_rate"])
        add(f"variant {variant} ({name}) op accuracy", o.accuracy, ref["accuracy"])
        add(f"variant {variant} ({name}) op macro_f1", o.macro_f1, ref["macro_f1"])
        add(f"variant {variant} ({name}) op mean_cost_s", o.mean_cost_s, ref["mean_cost_s"])
        ref_taus = np.array([f["tau"] for f in ref_fold[variant]])
        ours_taus = np.array([float(x) for x in o.tau_per_fold.split(";")])
        add(f"variant {variant} ({name}) per-fold tau", ours_taus, ref_taus)
        sw = sweep_df[sweep_df.stage1 == name].reset_index(drop=True)
        rs = ref_sweep[ref_sweep.variant == variant].reset_index(drop=True)
        add(f"variant {variant} ({name}) sweep accuracy (51 points)", sw.accuracy.values, rs.accuracy.values)
        add(f"variant {variant} ({name}) sweep macro_f1", sw.macro_f1.values, rs.macro_f1.values)
        add(f"variant {variant} ({name}) sweep mean_cost_s", sw.mean_cost_s.values, rs.mean_cost_s.values)
        add(f"variant {variant} ({name}) sweep tau_global", sw.tau_global.values, rs.tau_global.values)
        add(f"variant {variant} ({name}) op CI low (accuracy)", o.acc_ci_low,
            ref_main.loc[f"Cascade {variant} (op)", "acc_ci_low"])
        add(f"variant {variant} ({name}) op CI high (accuracy)", o.acc_ci_high,
            ref_main.loc[f"Cascade {variant} (op)", "acc_ci_high"])
    add("F-all accuracy", accF, ref_main.loc["F-all (LF)", "accuracy"])
    add("F-all macro_f1", f1_F, ref_main.loc["F-all (LF)", "macro_f1"])
    add("F-all CI low", acc_ci_F[0], ref_main.loc["F-all (LF)", "acc_ci_low"])
    add("F-all CI high", acc_ci_F[1], ref_main.loc["F-all (LF)", "acc_ci_high"])
    add("stage-1 cost A0+V0 equals F-cheap cost", stage1_cost(("A0", "V0"), comp), C_CHEAP)
    add("stage-1 cost A0 equals variant B stage-1 cost", stage1_cost(("A0",), comp),
        COST["Variant B stage 1 (audio + A0)"])
    san_df = pd.DataFrame(checks)
    san_df.to_csv(R.RES / "cascade_orders_sanity.csv", index=False)

    # ---- figure -----------------------------------------------------------------------
    R.set_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    colours = {"A0+V0": R.PAL["blue"], "A1": R.PAL["orange"], "A0+A1": R.PAL["green"],
               "A1+V0": R.PAL["purple"], "A0": R.PAL["vermil"]}
    for name in PLOT_SETS:
        sw = sweep_df[sweep_df.stage1 == name]
        ax.plot(sw.mean_cost_s, sw.accuracy, "-", lw=1, color=colours[name], label=f"S = {name}")
    for name in PLOT_SETS:
        o = op_df[op_df.stage1 == name].iloc[0]
        ax.scatter(o.mean_cost_s, o.accuracy, marker="*", s=60, color=colours[name], edgecolor="black",
                   linewidth=0.4, zorder=4)
    ax.scatter(C_ALL, accF, marker="s", s=18, color=R.PAL["black"], zorder=5, label="F-all (LF)")
    ax.set_xlabel("Mean CPU seconds per clip")
    ax.set_ylabel("Accuracy")
    ax.legend(frameon=False, loc="lower right", fontsize=6)
    R.save_fig(fig, "cascade_orders.pdf")

    # ---- console report -----------------------------------------------------------------
    pd.set_option("display.width", 250)
    print("\nOperating points (pooled test, per-fold tau from training OOF):")
    print(op_df[["stage1", "stage1_cost_s", "mean_cost_s", "saving_vs_F_all", "escalation_rate", "accuracy",
                 "acc_ci_low", "acc_ci_high", "macro_f1", "mean_acc_diff_vs_F_all", "p_value_vs_F_all",
                 "p_holm_vs_F_all", "r_star_per_fold"]].round(4).to_string(index=False))
    print("\nIso-accuracy (test sweep, descriptive):")
    print(iso_df.round(4).to_string(index=False))
    print("\nPareto frontier:")
    print(pareto_df[pareto_df.on_frontier][["kind", "name", "mean_cost_s", "accuracy", "macro_f1"]]
          .round(4).to_string(index=False))
    print("\nSanity checks:")
    print(san_df.to_string(index=False))
    if not san_df.passed.all():
        raise SystemExit("sanity checks FAILED")


if __name__ == "__main__":
    main()
