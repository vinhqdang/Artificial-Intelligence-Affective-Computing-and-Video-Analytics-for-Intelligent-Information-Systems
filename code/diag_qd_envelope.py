"""Per-clip flag rate of the clip-level monitors under each degradation (CREMA-D and CMU-MOSEI).

For every outer fold, the CONTENT and QD monitors are fitted on the clean training clips (as in X3) and
applied to the test clips under each condition; a clip is flagged when its distance exceeds the P99
threshold. The table shows how far each degradation leaves the clean reference envelope of the corpus.
Output: results/ext_qd_envelope.csv.
"""
import sys
import numpy as np
import pandas as pd
import run_extension_exp as X
import run_qag as Q

corpora = sys.argv[1:] or ["cremad", "mosei"]
rows = []
for name in corpora:
    pk = X.make_pack(name)
    bank = X.load_bank(pk)
    for k, (tr, te) in enumerate(pk.folds_full):
        for (kind, ch), d in bank.items():
            m = Q.monitor_fold(d["clean"], tr, pk.actor)
            for cond, Xc in d.items():
                lq = np.log(Q.monitor_d(m["model"], Xc[te]) / m["theta"]["p99"])
                rows.append(dict(corpus=name, monitor=kind, channel=ch, condition=cond, fold=k + 1,
                                 flag_rate=float(np.mean(lq > 0))))
df = pd.DataFrame(rows)
out = df.groupby(["corpus", "monitor", "channel", "condition"]).flag_rate.agg(["mean", "std"]).reset_index()
out.columns = ["corpus", "monitor", "channel", "condition", "flag_rate_mean", "flag_rate_sd"]
out.to_csv("../results/ext_qd_envelope.csv", index=False, float_format="%.3f")
print(out.to_string())
