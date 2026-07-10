"""Add a one-sample t-test (vs majority) p-value column to comprehensive_table.csv,
alongside the existing permutation p — method not decided, show both. Re-runs the
best model per cell (group-level nested), one-sample t-test on per-fold accs vs majority.
Per-split subprocess to avoid the libgomp segfault.
Driver: python3 add_ttest.py    Worker: python3 add_ttest.py <split>
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
TAB = f"scratch/te_adoption/a1_out/comprehensive_table.csv"
TTMP = f"scratch/te_adoption/a1_out/ttest_tmp.csv"
sys.path.insert(0, HERE)
import a1_lib

AUDIO = ["arousal", "dominance", "valence"]
GXS = ["BPM", "blink_durations", "G_OnSpeaker", "No_G_OnSpeaker", "G_SI", "No_G_SI"]
AIXVR = {"All": ["MG", "BPM"], "D": ["MG", "1d_DG"], "T": ["BPM"]}
FEATS = lambda det, sp: {"AIxVR": AIXVR[sp], "GazexSpeaking": GXS, "audio": AUDIO,
                         "audio+AIxVR": AIXVR[sp] + AUDIO, "audio+GazexSpeaking": GXS + AUDIO}[det]


def build():
    his = a1_lib.load_data()
    FLOOR = pd.read_csv("data/group_names_with_time_floorlevel.csv", sep=";")
    RT = pd.read_csv("data/recording_times_group_info.csv")
    rec = dict(zip(RT["long_name"], pd.to_datetime(RT["start_recording"])))
    bd = {r["Group_Name"]: (pd.to_datetime(r["TS_Start_Interaction"]) - rec[r["Group_Name_Long"]]).total_seconds() * 1000.0
          for _, r in FLOOR.iterrows() if r["Group_Name_Long"] in rec}
    HIS = pd.read_csv(a1_lib.HIS_CSV)
    av = {}
    for g, gg in his.groupby("group_name"):
        fr = pd.read_csv(f"data/discover/merged/recording_{g}.csv", usecols=["time_ms", "arousal", "dominance", "valence"])
        for idx in gg.index:
            w0 = bd[g] + HIS.loc[idx, "seconds_interaction_window"] * 1000.0
            m = (fr.time_ms >= w0) & (fr.time_ms < w0 + 60_000.0)
            av[idx] = fr.loc[m, ["arousal", "dominance", "valence"]].mean().values
    his[["arousal", "dominance", "valence"]] = pd.DataFrame(av, index=["arousal", "dominance", "valence"]).T
    his["grp"] = np.where(his.group_type == "dyad", "D", "T")
    return his


if len(sys.argv) == 2:  # worker for one split
    from scipy.stats import ttest_1samp
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV, GroupKFold
    from sklearn.naive_bayes import GaussianNB
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    RS = 42
    M = {"QDA": (QuadraticDiscriminantAnalysis(), {"reg_param": [0.1, 0.2, 0.3, 0.4, 0.5]}),
         "SVM": (SVC(random_state=RS), {"kernel": ["linear", "rbf"], "C": [0.1, 1], "gamma": [0.1, 0.01, 0.001]}),
         "LogReg": (LogisticRegression(solver="liblinear", random_state=RS), {"C": [0.01, 0.1, 1, 10]}),
         "NB": (GaussianNB(), {"var_smoothing": np.logspace(0, -9, num=50)})}
    sp = sys.argv[1]
    tab = pd.read_csv(TAB)
    his = build()
    sub = his if sp == "All" else his[his["grp"] == sp]
    for _, row in tab[(tab.split == sp) & (tab.detector != "dummy(majority)")].iterrows():
        feats = FEATS(row.detector, sp)
        X = sub[feats].values; y = sub["y"].values; g = sub["group_name"].values
        maj = max(y.mean(), 1 - y.mean())
        clf, grid = M[row.best_model]
        gkf = GroupKFold(n_splits=pd.Series(g).nunique()); accs = []
        for tr, te in gkf.split(X, y, groups=g):
            if pd.Series(y[tr]).nunique() < 2:
                continue
            inner = GroupKFold(n_splits=int(pd.Series(g[tr]).nunique()))
            gs = GridSearchCV(Pipeline([("s", StandardScaler()), ("m", clf)]),
                              {f"m__{k}": v for k, v in grid.items()}, cv=inner, n_jobs=1, error_score=np.nan)
            gs.fit(X[tr], y[tr], groups=g[tr]); accs.append((gs.best_estimator_.predict(X[te]) == y[te]).mean())
        _, p = ttest_1samp(accs, maj)
        with open(TTMP, "a") as f:
            f.write(f"{sp},{row.detector},{p:.4f}\n")
        print(f"OK {sp} {row.detector} ttest_p={p:.4f}", flush=True)
    sys.exit(0)

# driver
open(TTMP, "w").close()
env = dict(os.environ, PYTHONPATH=HERE)
for sp in ("All", "D", "T"):
    p = subprocess.run([sys.executable, f"{HERE}/add_ttest.py", sp], env=env, capture_output=True, text=True)
    print((p.stdout or p.stderr[-200:]).strip(), flush=True)
tt = pd.read_csv(TTMP, names=["split", "detector", "p_ttest_majority"])
tab = pd.read_csv(TAB).merge(tt, on=["split", "detector"], how="left")
tab.to_csv(TAB, index=False)
print("\nmerged t-test column into comprehensive_table.csv")
print(tab.to_string(index=False))
