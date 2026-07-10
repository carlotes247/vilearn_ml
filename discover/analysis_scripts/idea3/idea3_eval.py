"""Idea3 step 3: evaluate LLM rubric scores (think + nothink) and the German BERT baseline.
All group-level windows, LOSO by group, nested CV + label permutation. No med-drop:
labels binarized at 0.5 (the clean-mean construction), predictions te_continuous>0.5.

Sources:
  (a) LLM zero-shot   : te_continuous>0.5 vs label, permutation p          (think & nothink)
  (b) LLM markers     : 4 axes -> nested LOSO panel + permutation           (think & nothink)
  (c) BERT panel      : gbert frozen embeddings -> PCA -> nested LOSO panel  (generic text)
Writes idea3_results.csv. Prints audio/gaze prior context from a1_out for comparison.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, GroupKFold, permutation_test_score
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

AXES = ["task_relevance", "content_depth", "reasoning_present", "connecting_ideas"]
RS = 42
rng = np.random.default_rng(RS)
PANEL = {"QDA": (QuadraticDiscriminantAnalysis(), {"reg_param": [0.1, 0.2, 0.3, 0.4, 0.5]}),
         "SVM": (SVC(random_state=RS), {"kernel": ["linear", "rbf"], "C": [0.1, 1], "gamma": [0.1, 0.01, 0.001]}),
         "LogReg": (LogisticRegression(solver="liblinear", random_state=RS), {"C": [0.01, 0.1, 1, 10]}),
         "NB": (GaussianNB(), {"var_smoothing": np.logspace(0, -9, num=50)})}


def grp(gn):
    return np.where(pd.Series(gn).str.contains("dyad"), "D", "T")


def zeroshot(sub):
    pred = (sub["te_continuous"].values > 0.5).astype(int)
    y = sub["y"].values
    acc = (pred == y).mean()
    null = [(pred == rng.permutation(y)).mean() for _ in range(2000)]
    p = (sum(n >= acc for n in null) + 1) / 2001
    return acc, p


def panel_eval(X, y, g, pre_steps, panel_keys):
    """Nested LOSO panel: inner GridSearchCV per outer fold, pick best model by mean acc,
    then a permutation test on that model. pre_steps = list of (name, transformer) before clf."""
    gkf = GroupKFold(n_splits=pd.Series(g).nunique())
    res = {}
    for n in panel_keys:
        clf, grid = PANEL[n]
        accs = []
        for tr, te in gkf.split(X, y, groups=g):
            if pd.Series(y[tr]).nunique() < 2:
                continue
            inner = GroupKFold(n_splits=int(pd.Series(g[tr]).nunique()))
            pipe = Pipeline(pre_steps + [("m", clf)])
            gs = GridSearchCV(pipe, {f"m__{k}": v for k, v in grid.items()},
                              cv=inner, n_jobs=1, error_score=np.nan)
            gs.fit(X[tr], y[tr], groups=g[tr])
            accs.append((gs.best_estimator_.predict(X[te]) == y[te]).mean())
        res[n] = float(np.mean(accs)) if accs else float("nan")
    best = max(res, key=lambda k: (res[k] if res[k] == res[k] else -1))
    clf, _ = PANEL[best]
    pipe = Pipeline(pre_steps + [("m", clf)])
    _, _, pp = permutation_test_score(pipe, X, y, groups=g,
                                      cv=GroupKFold(n_splits=pd.Series(g).nunique()),
                                      scoring="accuracy", n_permutations=300,
                                      random_state=RS, n_jobs=1)
    return best, res[best], pp


rows = []

# (a)+(b) LLM think & nothink
for mode in ["think", "nothink"]:
    f = f"scratch/ideas/idea3_window_scores_{mode}.csv"
    if not os.path.exists(f):
        print(f"SKIP {mode}: {f} missing", flush=True)
        continue
    df = pd.read_csv(f)
    df = df[df["te_continuous"].notna()].copy()
    df["grp"] = grp(df.group_name)
    for sp in ["all", "D", "T"]:
        sub = df if sp == "all" else df[df["grp"] == sp]
        if sub.group_name.nunique() < 2 or sub.y.nunique() < 2:
            print(f"SKIP {mode}/{sp}: groups={sub.group_name.nunique()} classes={sub.y.nunique()}", flush=True)
            continue
        maj = max(sub.y.mean(), 1 - sub.y.mean())
        za, zp = zeroshot(sub)
        rows.append({"source": mode, "mode": "zeroshot", "split": sp, "n": len(sub),
                     "majority": round(maj, 3), "acc": round(za, 3), "p": round(zp, 4), "best_model": "-"})
        print(rows[-1], flush=True)
        fb, fa, fp = panel_eval(sub[AXES].values, sub["y"].values, sub["group_name"].values,
                                [("s", StandardScaler())], list(PANEL))
        rows.append({"source": mode, "mode": "features", "split": sp, "n": len(sub),
                     "majority": round(maj, 3), "acc": round(fa, 3), "p": round(fp, 4), "best_model": fb})
        print(rows[-1], flush=True)

# (c) BERT panel — 768-d -> PCA(20) -> linear panel (QDA dropped: singular at high dim)
bf = "scratch/ideas/idea3_bert_embeddings.npy"
if os.path.exists(bf):
    E = np.load(bf)
    idx = pd.read_csv("scratch/ideas/idea3_bert_index.csv")
    idx["grp"] = grp(idx.group_name)
    for sp in ["all", "D", "T"]:
        m = np.ones(len(idx), bool) if sp == "all" else (idx["grp"] == sp).values
        X, y, g = E[m], idx.y.values[m], idx.group_name.values[m]
        maj = max(y.mean(), 1 - y.mean())
        ncomp = min(20, m.sum() - 2)
        pre = [("s", StandardScaler()), ("pca", PCA(n_components=ncomp, random_state=RS))]
        bb, ba, bp = panel_eval(X, y, g, pre, ["LogReg", "SVM", "NB"])
        rows.append({"source": "bert", "mode": "features", "split": sp, "n": int(m.sum()),
                     "majority": round(maj, 3), "acc": round(ba, 3), "p": round(bp, 4), "best_model": bb})
        print(rows[-1], flush=True)
else:
    print(f"SKIP bert: {bf} missing", flush=True)

pd.DataFrame(rows).to_csv("scratch/ideas/idea3_results.csv", index=False)
print("\nwrote idea3_results.csv")

# prior context (no recompute)
ctx = "scratch/te_adoption/a1_out/comprehensive_table.csv"
if os.path.exists(ctx):
    c = pd.read_csv(ctx)
    print("\n=== prior/audio context (a1_out, group-level clean nested) ===")
    print(c[c.detector.isin(["dummy(majority)", "GazexSpeaking", "audio", "audio+GazexSpeaking"])]
          [["split", "detector", "acc", "perm_p"]].to_string(index=False))
