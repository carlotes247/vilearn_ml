"""d4 — descriptive univariate ranking of the 88 eGeMAPS openSMILE features
per split (full-data f_classif on the cached c2 feature matrix). Complements
the openSMILE_sel k-sweep: the sweep says HOW MANY features help (unbiased,
in-CV); this ranking + the per-fold picks in c2_selection.csv say WHICH.

Run from repo root: PYTHONPATH=discover/analysis_scripts python3 discover/analysis_scripts/d4_opensmile_ranking.py
Output: a1_out/d4_opensmile_ranking.csv (split, rank, feature, F, p)
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import warnings
warnings.filterwarnings("ignore")
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c2_paper_tables as c2
from sklearn.feature_selection import f_classif

his = c2.build_all()
osm = c2.osm_cols(his)
rows = []
for sp in ("All", "D", "T"):
    sub = c2.sub_for(his, sp)
    F, p = f_classif(sub[osm].values, sub["y"].values)
    rk = pd.DataFrame({"feature": osm, "F": F, "p": p}).sort_values("F", ascending=False)
    for i, (_, r) in enumerate(rk.iterrows(), 1):
        rows.append((sp, i, r.feature, round(r.F, 3), round(r.p, 5)))
    print(f"\n== {sp} top 10 (f_classif, full data — descriptive only) ==")
    print(rk.head(10).to_string(index=False))

pd.DataFrame(rows, columns=["split", "rank", "feature", "F", "p"]).to_csv(
    "scratch/te_adoption/a1_out/d4_opensmile_ranking.csv", index=False)
print("\nwrote a1_out/d4_opensmile_ranking.csv")
