"""Diagnostic for the speaker-adaptive (SPK) monitor on S5 streams.

For every degraded channel, compares the standardised SPK score on clean clips, on degraded clips
inside a session whose enrolment was clean ("mid-session"), and on degraded clips inside a session
that was enrolled under degradation ("enrolled degraded"), with the CONTENT and QD scores of the
same degraded clips. Output: results/ext_x3_spk_diag.csv.
"""
import sys
import numpy as np
import pandas as pd
import run_extension_exp as X

corpora = sys.argv[1:] or ["ravdess"]
rows = []
for name in corpora:
    pk = X.make_pack(name)
    for k in range(len(pk.folds_full)):
        ctx = X.build_ctx(pk, k)
        for st in X.build_streams(pk, k, X.N_SEEDS if hasattr(X, "N_SEEDS") else 20):
            if st["scn"] != "S5":
                continue
            deg = X.deg_truth(st)
            for ch in X.CH:
                if not deg[ch].any():
                    continue
                zs = X.spk_stream_z(ctx, st, ch, "clean")
                zc = X.gather(ctx.zc["clean"][("CONTENT", ch)], st["mc"][ch], st["clips"])
                zq = X.gather(ctx.zc["clean"][("QD", ch)], st["mc"][ch], st["clips"])
                for a, b in st["sess"]:
                    for t in range(a + X.SPK_ENROL, b):
                        grp = "clean" if not deg[ch][t] else ("enrolled_degraded" if deg[ch][a] else "mid_session")
                        rows.append((name, st["cond"], ch, grp, zs[t], zc[t], zq[t]))
df = pd.DataFrame(rows, columns=["corpus", "condition", "channel", "group", "z_spk", "z_content", "z_qd"])
out = df.groupby(["corpus", "condition", "channel", "group"]).agg(
    n=("z_spk", "size"), z_spk=("z_spk", "mean"), z_content=("z_content", "mean"), z_qd=("z_qd", "mean")).reset_index()
out.to_csv("../results/ext_x3_spk_diag.csv", index=False, float_format="%.3f")
print(out.to_string())
