"""d3 — restore the manually-appended c2 cells after the 2026-07-10 accidental
driver wipe (importing c2_paper_tables from d2_pca_dims executed the
module-level driver, which rewrote c2_paper_table.csv + c2_foldscores.csv with
only the 14 SETS cells). Protocol is deterministic (seed 42), so reruns
reproduce the original numbers exactly.

Steps:
  1. purge any half-restored rows, then rerun 6 extra detectors x 3 splits via
     worker subprocess calls (appends to c2_paper_table.csv + c2_foldscores.csv)
  2. purge stale c2_selection.csv rows for the _sel cells first (reruns re-append
     identical picks; avoids duplicates)
  3. recompute nan perm_p for rank-1 QDA rows with reg_param=0.1 (the 2026-07-07
     convention for singular-covariance cells) and patch the CSV in place

Run from repo root AFTER the driver rebuild finishes AND after the __main__
guard was added to c2_paper_tables.py (asserted below):
  python3 discover/analysis_scripts/d3_restore_extras.py
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import warnings
warnings.filterwarnings("ignore")
import subprocess
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = "scratch/te_adoption/a1_out"  # gitignored workbench; final CSVs are copied to discover/paper_results
RESULTS = f"{OUT}/c2_paper_table.csv"
FOLDS = f"{OUT}/c2_foldscores.csv"
SELECTION = f"{OUT}/c2_selection.csv"

src = open(f"{HERE}/c2_paper_tables.py").read()
assert '__name__ == "__main__"' in src, \
    "add the __main__ guard to c2_paper_tables.py first (import would re-wipe!)"

EXTRAS = ["linguistic+AIxVR", "linguistic_core", "linguistic_sel",
          "linguistic+GazexSpeaking_sel", "audio+linguistic_sel",
          "audio+linguistic+GazexSpeaking_sel"]
SPLITS = ("All", "D", "T")

# 1+2: purge, then rerun
for path in (RESULTS, FOLDS):
    df = pd.read_csv(path)
    n0 = len(df)
    df = df[~df.detector.isin(EXTRAS)]
    if len(df) != n0:
        df.to_csv(path, index=False)
        print(f"purged {n0 - len(df)} stale rows from {path}")
if os.path.exists(SELECTION):
    sel = pd.read_csv(SELECTION, header=None,
                      names=["split", "detector", "model", "test_group", "k", "picked"])
    n0 = len(sel)
    sel = sel[~sel.detector.isin(EXTRAS)]
    if len(sel) != n0:
        sel.to_csv(SELECTION, index=False, header=False)
        print(f"purged {n0 - len(sel)} stale rows from {SELECTION}")

env = dict(os.environ, PYTHONPATH=HERE)


def run_cell(job):
    sp, det = job
    p = subprocess.run([sys.executable, f"{HERE}/c2_paper_tables.py", sp, det],
                       env=env, capture_output=True, text=True)
    print((p.stdout.strip() or f"FAIL {sp} {det}: {p.stderr[-300:]}"), flush=True)


from concurrent.futures import ThreadPoolExecutor
jobs = [(sp, det) for sp in SPLITS for det in EXTRAS]
with ThreadPoolExecutor(max_workers=min(12, os.cpu_count() or 4)) as ex:
    list(ex.map(run_cell, jobs))

# 3: nan perm_p on rank-1 QDA rows -> recompute with reg_param=0.1
# The patch REWRITES the whole CSV -> must not race the driver's atomic appends.
# Wait for a bare-args driver (`c2_paper_tables.py` with no cell args) to finish.
import time
while subprocess.run(["pgrep", "-f", r"c2_paper_tables\.py$"],
                     capture_output=True).returncode == 0:
    print("waiting for driver to finish before CSV rewrite ...", flush=True)
    time.sleep(20)

sys.path.insert(0, HERE)
import ab_common as ab
import c2_paper_tables as c2
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GroupKFold, permutation_test_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(RESULTS)
todo = df[(df["rank"] == 1) & (df.model == "QDA") & df.perm_p.isna()
          & (df.detector != "baseline_majority")]
if len(todo):
    his = c2.build_all()
    OSM = c2.osm_cols(his)
    for i, r in todo.iterrows():
        base_det = r.detector[:-4] if r.detector.endswith("_sel") else r.detector
        sub = c2.sub_for(his, r.split)
        feats = c2.feats_for(base_det, r.split, OSM)
        steps = [("s", StandardScaler())]
        if r.detector.endswith("_sel"):
            steps.append(("k", SelectKBest(f_classif, k=min(5, len(feats)))))
        steps.append(("m", QuadraticDiscriminantAnalysis(reg_param=0.1)))
        X, y, g = sub[feats].values, sub["y"].values, sub["group_name"].values
        _, _, pp = permutation_test_score(
            Pipeline(steps), X, y, groups=g,
            cv=GroupKFold(n_splits=pd.Series(g).nunique()),
            scoring="accuracy", n_permutations=300, random_state=ab.RS, n_jobs=1)
        df.loc[i, "perm_p"] = round(float(pp), 4)
        print(f"perm patched (QDA reg=0.1) {r.split} {r.detector}: {pp:.4f}", flush=True)
    df.to_csv(RESULTS, index=False)
else:
    print("no nan perm_p QDA rows to patch")

print("done — verify against slide tables / conversation snapshot")
