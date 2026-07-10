"""A1 shared lib: thread-capped env, data loading, model panel, single-model runner.

Used by a1_worker.py (one model per subprocess — avoids the in-sequence libgomp
segfault that kills RandomForest/MLP after many prior GridSearchCV calls).
"""
import os
# cap BLAS/OpenMP threads BEFORE numpy/sklearn import (segfault avoidance)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

RS = 42
FEATS = ["BPM", "blink_durations", "G_OnSpeaker", "No_G_OnSpeaker", "G_SI", "No_G_SI"]
OUT = "scratch/te_adoption/a1_out"
HIS_CSV = "Recordings/SavedData/v2_no_low_sampled/60s_TE_correlation_2026-05-19.csv"


def load_data():
    """his 185 floorlevel windows with OUR clean-mean TE label at his (group, sec) grid."""
    HIS = pd.read_csv(HIS_CSV)
    FLOOR = pd.read_csv("data/group_names_with_time_floorlevel.csv", sep=";")
    RT = pd.read_csv("data/recording_times_group_info.csv")
    rec_start = dict(zip(RT["long_name"], pd.to_datetime(RT["start_recording"])))
    floor_groups = set(FLOOR.Group_Name)
    bounds = {}
    for _, r in FLOOR.iterrows():
        ln = r["Group_Name_Long"]
        if ln in rec_start:
            bounds[r["Group_Name"]] = (pd.to_datetime(r["TS_Start_Interaction"]) - rec_start[ln]).total_seconds() * 1000.0
    his = HIS[HIS.group_name.isin(floor_groups)].copy()
    labels = {}
    for g, gg in his.groupby("group_name"):
        fr = pd.read_csv(f"data/discover/merged/recording_{g}.csv", usecols=["time_ms", "task_engagement"])
        s_ms = bounds[g]
        for idx, hr in gg.iterrows():
            w0 = s_ms + hr["seconds_interaction_window"] * 1000.0
            m = (fr.time_ms >= w0) & (fr.time_ms < w0 + 60_000.0)
            labels[idx] = fr.loc[m, "task_engagement"].mean()
    his["our_te"] = pd.Series(labels)
    his["y"] = (his["our_te"] > 0.5).astype(int)
    assert his["our_te"].notna().all(), "NaN label - empty window"
    return his


def panel():
    """EXACT copy of derive.make_classifier_panel_nested (seeded). RF/NeuralNet/AdaBoost
    given n_jobs=1 where supported."""
    return {
        "Baseline Most Frequent": (DummyClassifier(strategy="most_frequent"), {}),
        "Baseline Prior": (DummyClassifier(strategy="prior"), {}),
        "Baseline Stratified": (DummyClassifier(strategy="stratified", random_state=RS), {}),
        "Baseline Uniform": (DummyClassifier(strategy="uniform", random_state=RS), {}),
        "kNN": (KNeighborsClassifier(), {"n_neighbors": np.arange(1, 100, 1), "weights": ["uniform", "distance"]}),
        "Logistic Regression": (LogisticRegression(solver="liblinear", random_state=RS),
                                {"C": [0.01, 0.1, 1, 10], "penalty": ["l1", "l2"], "max_iter": [100, 200], "tol": [1e-4, 1e-3]}),
        "Linear SVM l1": (LinearSVC(dual="auto", random_state=RS),
                          {"penalty": ["l1"], "loss": ["squared_hinge"], "C": [0.01, 0.1, 1, 5, 10, 100], "max_iter": [5000, 10000, 50000]}),
        "Linear SVM l2": (LinearSVC(dual="auto", random_state=RS),
                          {"penalty": ["l2"], "loss": ["hinge", "squared_hinge"], "C": [0.01, 0.1, 1, 5, 10, 100], "max_iter": [5000, 10000, 50000]}),
        "SVM linear or rbf": (SVC(random_state=RS),
                              {"kernel": ["linear", "rbf"], "C": [0.1, 1], "gamma": [0.1, 0.01, 0.001], "degree": [0, 1, 2, 4]}),
        "Decision Tree": (DecisionTreeClassifier(random_state=RS),
                          {"max_depth": [10, 20, 30, None], "min_samples_split": [2, 5, 10], "min_samples_leaf": [1, 2, 4]}),
        "Random Forest": (RandomForestClassifier(random_state=RS, n_jobs=1),
                          {"n_estimators": [100, 200], "max_depth": [None, 10, 20], "min_samples_split": [2, 5], "min_samples_leaf": [1, 2], "bootstrap": [True, False]}),
        "Neural Net": (MLPClassifier(max_iter=1000, random_state=RS),
                       {"hidden_layer_sizes": [(10, 30, 10), (20,)], "activation": ["tanh", "relu"], "solver": ["sgd", "adam"], "alpha": [0.0001, 0.05, 1], "learning_rate": ["constant", "adaptive"]}),
        "AdaBoost": (AdaBoostClassifier(random_state=RS),
                     {"n_estimators": [10, 50, 100, 500], "learning_rate": [0.0001, 0.001, 0.01, 0.1, 1.0, 10]}),
        "Naive Bayes": (GaussianNB(), {"var_smoothing": np.logspace(0, -9, num=100)}),
        "QDA": (QuadraticDiscriminantAnalysis(), {"reg_param": [0.1, 0.2, 0.3, 0.4, 0.5]}),
    }


BASELINES = {"Baseline Most Frequent", "Baseline Prior", "Baseline Stratified", "Baseline Uniform"}


def run_one(his, split, model_name, scaled):
    """Nested LOSO/LOGO for one model on his features; per-group accuracy rows."""
    df = {"all": his, "D": his[his.group_type == "dyad"], "T": his[his.group_type == "triad"]}[split]
    X = df[FEATS].reset_index(drop=True)
    y = df["y"].reset_index(drop=True)
    groups = df["group_name"].reset_index(drop=True)
    clf, grid = panel()[model_name]
    gkf = GroupKFold(n_splits=groups.nunique())
    rows = []
    for tr, te in gkf.split(X, y, groups=groups):
        Xtr, Xte = X.iloc[tr].reset_index(drop=True), X.iloc[te].reset_index(drop=True)
        ytr, yte = y.iloc[tr].reset_index(drop=True), y.iloc[te].reset_index(drop=True)
        gtr = groups.iloc[tr].reset_index(drop=True)
        held = groups.iloc[te].iloc[0]
        if ytr.nunique() < 2:
            continue
        steps = ([("scaler", StandardScaler())] if scaled else []) + [("model", clf)]
        pipe = Pipeline(steps)
        bp = ""
        if grid:
            inner = GroupKFold(n_splits=int(gtr.nunique()))
            gs = GridSearchCV(pipe, {f"model__{k}": v for k, v in grid.items()},
                              cv=inner, n_jobs=1, error_score=np.nan)
            gs.fit(Xtr, ytr, groups=gtr)
            fitted = gs.best_estimator_
            bp = json.dumps({k.removeprefix("model__"): v for k, v in gs.best_params_.items()}, default=str)
        else:
            fitted = pipe.fit(Xtr, ytr)
        acc = float((fitted.predict(Xte) == yte).mean())
        rows.append({"split": split, "model": model_name, "scaled": scaled,
                     "group": held, "accuracy": acc, "best_params": bp})
    return rows
