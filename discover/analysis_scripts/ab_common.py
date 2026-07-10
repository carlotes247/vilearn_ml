"""Shared feature builder + nested-CV evaluator for the 2026-07-01 A/B todos.

build_features() -> his 185 floorlevel windows with:
  gaze cols (from HIS_CSV), y (clean-mean TE >0.5), group_name/group_type/grp/sec,
  vad (arousal/dominance/valence windowed onto his grid), 4 LLM rubric axes (think),
  and `rate` (native annotation rate 60Hz/90Hz via duration match, per merge script).

No CLI side effects on import (unlike comprehensive_comparison.py).
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import warnings
warnings.filterwarnings("ignore")
import glob
from pathlib import Path
import numpy as np
import pandas as pd

os.makedirs("scratch/te_adoption/a1_out", exist_ok=True)
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, GroupKFold, permutation_test_score
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import a1_lib

RS = 42
AUDIO = ["arousal", "dominance", "valence"]
GXS = ["BPM", "blink_durations", "G_OnSpeaker", "No_G_OnSpeaker", "G_SI", "No_G_SI"]
AIXVR = {"All": ["MG", "BPM"], "D": ["MG", "1d_DG"], "T": ["BPM"]}
AXES = ["task_relevance", "content_depth", "reasoning_present", "connecting_ideas"]

PANEL = {"QDA": (QuadraticDiscriminantAnalysis(), {"reg_param": [0.1, 0.2, 0.3, 0.4, 0.5]}),
         "SVM": (SVC(random_state=RS), {"kernel": ["linear", "rbf"], "C": [0.1, 1], "gamma": [0.1, 0.01, 0.001]}),
         "LogReg": (LogisticRegression(solver="liblinear", random_state=RS), {"C": [0.01, 0.1, 1, 10]}),
         "NB": (GaussianNB(), {"var_smoothing": np.logspace(0, -9, num=50)})}


def _detect_rate(g, n_helen):
    """Native annotation rate of the second annotator via duration match (mirrors
    merge_vilearn_features.load_carlos_task_engagement). Returns '60Hz' or '90Hz'."""
    dur = n_helen / 90.0
    cands = [(f"data/discover/recording_{g}/task_engagement.group.carlosgonzalez.csv", 90.0),
             (f"data/discover/recording_{g}/task_engagement60Hz.group.carlosgonzalez.csv", 60.0),
             (f"data/annotations/recording_{g}/task engagement.group.carlosgonzalez.annotation~", 90.0),
             (f"data/annotations/recording_{g}/task engagement60Hz.group.carlosgonzalez.annotation~", 60.0)]
    for path, named in cands:
        p = Path(path)
        if not p.exists():
            continue
        if p.suffix == ".csv":
            raw = pd.read_csv(p)
            v = pd.to_numeric(raw["score"], errors="coerce").to_numpy()
        else:
            raw = pd.read_csv(p, sep=";", names=["score", "conf"])
            v = pd.to_numeric(raw["score"], errors="coerce").to_numpy()
        for freq in (named, 60.0, 90.0):
            if abs(len(v) / freq - dur) < 2.0:
                return f"{int(freq)}Hz"
    return "?"


def build_features(llm_mode="think"):
    his = a1_lib.load_data()  # 185 windows, y, gaze cols, group_name/type
    FLOOR = pd.read_csv("data/group_names_with_time_floorlevel.csv", sep=";")
    RT = pd.read_csv("data/recording_times_group_info.csv")
    rec = dict(zip(RT["long_name"], pd.to_datetime(RT["start_recording"])))
    bounds = {r["Group_Name"]: (pd.to_datetime(r["TS_Start_Interaction"]) - rec[r["Group_Name_Long"]]).total_seconds() * 1000.0
              for _, r in FLOOR.iterrows() if r["Group_Name_Long"] in rec}
    HIS = pd.read_csv(a1_lib.HIS_CSV)
    av = {}
    for g, gg in his.groupby("group_name"):
        fr = pd.read_csv(f"data/discover/merged/recording_{g}.csv", usecols=["time_ms"] + AUDIO)
        s0 = bounds[g]
        for idx in gg.index:
            w0 = s0 + HIS.loc[idx, "seconds_interaction_window"] * 1000.0
            m = (fr.time_ms >= w0) & (fr.time_ms < w0 + 60_000.0)
            av[idx] = fr.loc[m, AUDIO].mean().values
    his[AUDIO] = pd.DataFrame(av, index=AUDIO).T
    his["sec"] = HIS.loc[his.index, "seconds_interaction_window"].values
    llm = pd.read_csv(f"discover/paper_results/idea3_window_scores_{llm_mode}.csv")[["group_name", "sec"] + AXES]
    his = his.merge(llm, on=["group_name", "sec"], how="left", validate="one_to_one")
    assert his[AXES].notna().all().all(), "LLM axes join failed"
    assert his[AUDIO].notna().all().all(), "NaN audio"
    his["grp"] = np.where(his.group_type == "dyad", "D", "T")
    # native rate per group (duration match on second-annotator file)
    nh = {g: len(pd.read_csv(f"data/discover/recording_{g}/task_engagement.group.helenrisack.csv"))
          for g in his.group_name.unique() if Path(f"data/discover/recording_{g}/task_engagement.group.helenrisack.csv").exists()}
    rate = {g: _detect_rate(g, nh.get(g, 5400)) for g in his.group_name.unique()}
    his["rate"] = his.group_name.map(rate)
    return his


def nested(X, y, g):
    """Nested LOGO model-selection over PANEL; returns (best_model, acc)."""
    def one(clf, grid):
        gkf = GroupKFold(n_splits=pd.Series(g).nunique()); a = []
        for tr, te in gkf.split(X, y, groups=g):
            if pd.Series(y[tr]).nunique() < 2:
                continue
            inner = GroupKFold(n_splits=int(pd.Series(g[tr]).nunique()))
            gs = GridSearchCV(Pipeline([("s", StandardScaler()), ("m", clf)]),
                              {f"m__{k}": v for k, v in grid.items()}, cv=inner, n_jobs=1, error_score=np.nan)
            gs.fit(X[tr], y[tr], groups=g[tr]); a.append((gs.best_estimator_.predict(X[te]) == y[te]).mean())
        return float(np.mean(a))
    res = {n: one(c, gr) for n, (c, gr) in PANEL.items()}
    best = max(res, key=lambda k: res[k])
    return best, res[best]


def perfold_accs(sub, feats, best_model):
    """Per-fold LOGO accuracies for a FIXED model (inner GridSearch on its grid).
    Mirrors add_ttest.py — used for the one-sample t-test vs majority."""
    X = sub[feats].values; y = sub["y"].values; g = sub["group_name"].values
    clf, grid = PANEL[best_model]
    gkf = GroupKFold(n_splits=pd.Series(g).nunique()); accs = []
    for tr, te in gkf.split(X, y, groups=g):
        if pd.Series(y[tr]).nunique() < 2:
            continue
        inner = GroupKFold(n_splits=int(pd.Series(g[tr]).nunique()))
        gs = GridSearchCV(Pipeline([("s", StandardScaler()), ("m", clf)]),
                          {f"m__{k}": v for k, v in grid.items()}, cv=inner, n_jobs=1, error_score=np.nan)
        gs.fit(X[tr], y[tr], groups=g[tr]); accs.append((gs.best_estimator_.predict(X[te]) == y[te]).mean())
    return accs


def perm_p(sub, feats, clf):
    X = sub[feats].values; y = sub["y"].values; g = sub["group_name"].values
    _, _, pp = permutation_test_score(Pipeline([("s", StandardScaler()), ("m", clf)]), X, y, groups=g,
                                      cv=GroupKFold(n_splits=pd.Series(g).nunique()), scoring="accuracy",
                                      n_permutations=300, random_state=RS, n_jobs=1)
    return pp


def evaluate(sub, feats):
    X = sub[feats].values; y = sub["y"].values; g = sub["group_name"].values
    best, acc = nested(X, y, g)
    clf = PANEL[best][0]
    return best, acc, perm_p(sub, feats, clf)
