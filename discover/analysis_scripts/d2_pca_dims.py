"""d2 — which PCA n_components {5,10,20} the inner CV actually picked for the
openSMILE_pca cells (per fold, best model per split from c2_paper_table.csv).
Informs the k-grid cap for the upcoming SelectKBest sweep.

Run from REPO ROOT: PYTHONPATH=discover/analysis_scripts python3 discover/analysis_scripts/d2_pca_dims.py
Output: a1_out/d2_pca_dims.csv (split, model, test_group, n_components, acc)
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import warnings
warnings.filterwarnings("ignore")
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ab_common as ab
import c2_paper_tables as c2

# best openSMILE_pca model per split (rank-1 in c2_paper_table.csv)
BEST = {"All": "NB", "D": "LogReg", "T": "NB"}

from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

his = c2.build_all()
osm = c2.osm_cols(his)
rows = []
for sp, model in BEST.items():
    sub = c2.sub_for(his, sp)
    X = sub[osm].values
    y = sub["y"].values
    g = sub["group_name"].values
    clf, grid = ab.PANEL[model]
    pgrid = {f"m__{k}": v for k, v in grid.items()}
    pgrid["p__n_components"] = [5, 10, 20]
    steps = [("s", StandardScaler()), ("p", PCA(random_state=ab.RS)), ("m", clf)]
    gkf = GroupKFold(n_splits=pd.Series(g).nunique())
    for tr, te in gkf.split(X, y, groups=g):
        if pd.Series(y[tr]).nunique() < 2:
            continue
        inner = GroupKFold(n_splits=int(pd.Series(g[tr]).nunique()))
        gs = GridSearchCV(Pipeline(steps), pgrid, cv=inner, n_jobs=1, error_score=np.nan)
        gs.fit(X[tr], y[tr], groups=g[tr])
        tg = str(pd.Series(g[te]).iloc[0])
        acc = float((gs.best_estimator_.predict(X[te]) == y[te]).mean())
        nc = int(gs.best_params_["p__n_components"])
        rows.append((sp, model, tg, nc, acc))
        print(f"{sp} {model} {tg}: n_components={nc} acc={acc:.3f}", flush=True)

out = pd.DataFrame(rows, columns=["split", "model", "test_group", "n_components", "acc"])
out.to_csv("scratch/te_adoption/a1_out/d2_pca_dims.csv", index=False)
print("\npicked n_components frequency per split:")
print(out.groupby(["split", "n_components"]).size().unstack(fill_value=0))
print("\nmean acc by split (sanity vs c2: All .598 / D .668 / T .540):")
print(out.groupby("split").acc.mean().round(3))
