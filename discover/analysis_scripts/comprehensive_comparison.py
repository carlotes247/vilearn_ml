"""Definitive comparison vs 3 priors (dummy/AIxVR/GazexSpeaking) + audio + fusions.
Group-level, clean labels, nested CV + permutation. Crash-isolated: one subprocess
per (split, detector) to dodge the in-process libgomp segfault.

Driver:  python3 discover/analysis_scripts/comprehensive_comparison.py
Worker:  python3 discover/analysis_scripts/comprehensive_comparison.py <split> <detector>
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import warnings
warnings.filterwarnings("ignore")
import sys
import subprocess
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUTCSV = f"scratch/te_adoption/a1_out/comprehensive_table.csv"
sys.path.insert(0, HERE)
import a1_lib

AUDIO = ["arousal", "dominance", "valence"]
GXS = ["BPM", "blink_durations", "G_OnSpeaker", "No_G_OnSpeaker", "G_SI", "No_G_SI"]
AIXVR = {"All": ["MG", "BPM"], "D": ["MG", "1d_DG"], "T": ["BPM"]}


def feats_for(det, sp):
    return {"AIxVR": AIXVR[sp], "GazexSpeaking": GXS, "audio": AUDIO,
            "audio+AIxVR": AIXVR[sp] + AUDIO, "audio+GazexSpeaking": GXS + AUDIO}[det]


def build():
    his = a1_lib.load_data()
    FLOOR = pd.read_csv("data/group_names_with_time_floorlevel.csv", sep=";")
    RT = pd.read_csv("data/recording_times_group_info.csv")
    rec = dict(zip(RT["long_name"], pd.to_datetime(RT["start_recording"])))
    bounds = {r["Group_Name"]: (pd.to_datetime(r["TS_Start_Interaction"]) - rec[r["Group_Name_Long"]]).total_seconds() * 1000.0
              for _, r in FLOOR.iterrows() if r["Group_Name_Long"] in rec}
    HIS = pd.read_csv(a1_lib.HIS_CSV)
    av = {}
    for g, gg in his.groupby("group_name"):
        fr = pd.read_csv(f"data/discover/merged/recording_{g}.csv", usecols=["time_ms", "arousal", "dominance", "valence"])
        s0 = bounds[g]
        for idx in gg.index:
            w0 = s0 + HIS.loc[idx, "seconds_interaction_window"] * 1000.0
            m = (fr.time_ms >= w0) & (fr.time_ms < w0 + 60_000.0)
            av[idx] = fr.loc[m, ["arousal", "dominance", "valence"]].mean().values
    his[["arousal", "dominance", "valence"]] = pd.DataFrame(av, index=["arousal", "dominance", "valence"]).T
    his["grp"] = np.where(his.group_type == "dyad", "D", "T")
    return his


if len(sys.argv) == 3:  # worker
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV, GroupKFold, permutation_test_score
    from sklearn.naive_bayes import GaussianNB
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    RS = 42
    PANEL = {"QDA": (QuadraticDiscriminantAnalysis(), {"reg_param": [0.1, 0.2, 0.3, 0.4, 0.5]}),
             "SVM": (SVC(random_state=RS), {"kernel": ["linear", "rbf"], "C": [0.1, 1], "gamma": [0.1, 0.01, 0.001]}),
             "LogReg": (LogisticRegression(solver="liblinear", random_state=RS), {"C": [0.01, 0.1, 1, 10]}),
             "NB": (GaussianNB(), {"var_smoothing": np.logspace(0, -9, num=50)})}
    sp, det = sys.argv[1], sys.argv[2]
    his = build()
    sub = his if sp == "All" else his[his["grp"] == sp]
    feats = feats_for(det, sp)
    X = sub[feats].values; y = sub["y"].values; g = sub["group_name"].values
    maj = max(y.mean(), 1 - y.mean())

    def nested(clf, grid):
        gkf = GroupKFold(n_splits=pd.Series(g).nunique()); a = []
        for tr, te in gkf.split(X, y, groups=g):
            if pd.Series(y[tr]).nunique() < 2:
                continue
            inner = GroupKFold(n_splits=int(pd.Series(g[tr]).nunique()))
            gs = GridSearchCV(Pipeline([("s", StandardScaler()), ("m", clf)]),
                              {f"m__{k}": v for k, v in grid.items()}, cv=inner, n_jobs=1, error_score=np.nan)
            gs.fit(X[tr], y[tr], groups=g[tr]); a.append((gs.best_estimator_.predict(X[te]) == y[te]).mean())
        return float(np.mean(a))
    res = {n: nested(c, gr) for n, (c, gr) in PANEL.items()}
    best = max(res, key=lambda k: res[k]); c, gr = PANEL[best]
    _, _, pp = permutation_test_score(Pipeline([("s", StandardScaler()), ("m", c)]), X, y, groups=g,
                                      cv=GroupKFold(n_splits=pd.Series(g).nunique()), scoring="accuracy",
                                      n_permutations=300, random_state=RS, n_jobs=1)
    with open(OUTCSV, "a") as f:
        f.write(f"{sp},{det},{best},{res[best]:.3f},{pp:.4f},{maj:.3f}\n")
    print(f"OK {sp} {det} {best} {res[best]:.3f} p={pp:.4f}")
    sys.exit(0)

# driver
with open(OUTCSV, "w") as f:
    f.write("split,detector,best_model,acc,perm_p,majority\n")
his = build()
for sp in ("All", "D", "T"):
    sub = his if sp == "All" else his[his["grp"] == sp]
    maj = max(sub.y.mean(), 1 - sub.y.mean())
    with open(OUTCSV, "a") as f:
        f.write(f"{sp},dummy(majority),-,{maj:.3f},-,{maj:.3f}\n")
env = dict(os.environ, PYTHONPATH=HERE)
for sp in ("All", "D", "T"):
    for det in ("AIxVR", "GazexSpeaking", "audio", "audio+AIxVR", "audio+GazexSpeaking"):
        p = subprocess.run([sys.executable, f"{HERE}/comprehensive_comparison.py", sp, det],
                           env=env, capture_output=True, text=True)
        print((p.stdout or p.stderr[-150:]).strip(), flush=True)
print("done")
