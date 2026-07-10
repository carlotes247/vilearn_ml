"""D1 — sanity check the linguistic 0.788: drop-one + solo-feature ablation.

Per split (All/D/T) and variant (full / drop_<feat> / solo_<feat>), same canonical
protocol as c2 (nested LOGO over 4-model panel, cached c2_features.csv). No perm
(acc deltas only). Also prints standardized LogReg coefficients (full fit) per split
for direction of effect.

Driver:  PYTHONPATH=discover/analysis_scripts python3 discover/analysis_scripts/d1_linguistic_dropone.py
Worker:  ... d1_linguistic_dropone.py <split> <variant>     (from REPO ROOT)
Output:  a1_out/d1_linguistic_dropone.csv
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import warnings
warnings.filterwarnings("ignore")
import sys
import subprocess
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = f"scratch/te_adoption/a1_out/d1_linguistic_dropone.csv"
sys.path.insert(0, HERE)
import ab_common as ab

LING = ["segment_count", "word_count", "avg_word_length", "question_ratio",
        "speech_ratio", "mean_segment_duration_s", "unfinished_ratio"]


def feats_for(variant):
    if variant == "full":
        return LING
    kind, feat = variant.split("_", 1)
    if kind == "drop":
        return [f for f in LING if f != feat]
    return [feat]  # solo


def load_sub(sp):
    his = pd.read_csv(f"scratch/te_adoption/a1_out/c2_features.csv")
    return his if sp == "All" else his[his["grp"] == sp]


if len(sys.argv) == 3:  # worker
    sp, variant = sys.argv[1], sys.argv[2]
    sub = load_sub(sp)
    feats = feats_for(variant)
    per_model = {}
    for model in ab.PANEL:
        try:
            accs = ab.perfold_accs(sub, feats, model)
        except Exception as e:
            print(f"skip {sp} {variant} {model}: {type(e).__name__}", flush=True)
            continue
        per_model[model] = accs
    best = max(per_model, key=lambda m: np.mean(per_model[m]))
    accs = np.array(per_model[best])
    with open(OUT, "a") as f:
        f.write(f"{sp},{variant},{len(feats)},{best},{accs.mean():.3f},{accs.std(ddof=1):.3f}\n")
    print(f"OK {sp} {variant} {best} {accs.mean():.3f}", flush=True)
    sys.exit(0)

# driver
VARIANTS = ["full"] + [f"drop_{f}" for f in LING] + [f"solo_{f}" for f in LING]
with open(OUT, "w") as f:
    f.write("split,variant,n_feat,best_model,acc_mean,acc_std\n")
env = dict(os.environ, PYTHONPATH=HERE)
for sp in ("All", "D", "T"):
    for variant in VARIANTS:
        p = subprocess.run([sys.executable, f"{HERE}/d1_linguistic_dropone.py", sp, variant],
                           env=env, capture_output=True, text=True)
        print((p.stdout.strip() or f"FAIL {sp} {variant}: {p.stderr[-200:]}"), flush=True)

# standardized LogReg coefficients, full fit (direction only, not accuracy)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
print("\nstandardized LogReg coefs (full-data fit, direction only):")
for sp in ("All", "D", "T"):
    sub = load_sub(sp)
    X = StandardScaler().fit_transform(sub[LING].values)
    lr = LogisticRegression(solver="liblinear", random_state=ab.RS).fit(X, sub["y"].values)
    order = np.argsort(-np.abs(lr.coef_[0]))
    print(f"  {sp}: " + ", ".join(f"{LING[i]}={lr.coef_[0][i]:+.2f}" for i in order))

tab = pd.read_csv(OUT)
full = {sp: tab[(tab.split == sp) & (tab.variant == "full")].acc_mean.iloc[0] for sp in ("All", "D", "T")}
tab["delta_vs_full"] = tab.apply(lambda r: round(r.acc_mean - full[r.split], 3), axis=1)
tab.to_csv(OUT, index=False)
print(f"\nwrote {OUT}")
