"""POST-HOC check A2 (not pre-registered): does the cluster-bootstrap p-value mean what it says for persistent features?
Null = Gaussian AR(1) noise with the same lag-1 autocorrelation as each feature, same sample days and labels. 2,000 sims per variable.
Reports the empirical one-sided p of the observed AUC under that null and the false-positive rate of the bootstrap p<=0.05 rule is NOT
re-run here (AUC null only)."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import stage1_run as s

rng = np.random.default_rng(s.SEED + 1)
cal, close, y_dn, y_up, F, sign, BASE, E, es = s.build()
rows = []
for vid in F.columns:
    grp = vid[:2]
    w0, w1 = ((pd.Timestamp("2020-09-01"), s.H7_END) if grp == "H7" else (cal[0], s.LAST_LABEL))
    msk = (cal >= w0) & (cal <= w1)
    x = (sign[vid] * F[vid])[msk]; y = y_dn[msk]
    ok = x.notna() & y.notna(); x, y = x[ok].values, y[ok].values
    a_obs = s.auc(x, y)
    xs = (x - x.mean()) / x.std(); phi = float(np.corrcoef(xs[:-1], xs[1:])[0, 1])
    n = len(x); sims = np.empty(2000)
    for k in range(2000):
        e = rng.standard_normal(n); z = np.empty(n); z[0] = e[0]
        for t in range(1, n):
            z[t] = phi * z[t - 1] + np.sqrt(1 - phi ** 2) * e[t]
        sims[k] = s.auc(z, y)
    rows.append(dict(var=vid, n=n, lag1_autocorr=round(phi, 3), auc_obs=round(a_obs, 4), null_sd=round(sims.std(), 4),
                     null_q95=round(np.quantile(sims, 0.95), 4), p_ar1_null=round((1 + (sims >= a_obs).sum()) / 2001, 4)))
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(os.path.join(s.OUT, "stage1_calibration_ar1.csv"), index=False)
