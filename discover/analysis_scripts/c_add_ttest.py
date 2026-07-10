"""C' — add one-sample t-test-vs-majority p to b1 + b2 tables (meeting: report both).

Reuses each row's already-selected best_model; refits per fold to get per-fold
accuracies, then ttest_1samp(accs, majority). Adds `p_ttest_maj` column in place.
Rebuilds the feature set per row from ab_common (so labels match exactly).

Run from repo root:  PYTHONPATH=discover/analysis_scripts python3 discover/analysis_scripts/c_add_ttest.py
"""
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
import ab_common as ab

his = ab.build_features()
VAD, AXES, GXS = ab.AUDIO, ab.AXES, ab.GXS


def featset(name, sp):
    aix = ab.AIXVR[sp]
    return {
        # b1 names
        "gaze_GxS": GXS, "gaze_AIxVR": aix, "llm": AXES, "audio": VAD,
        "GxS+llm": GXS + AXES, "AIxVR+llm": aix + AXES, "audio+llm": VAD + AXES,
        "GxS+vad": GXS + VAD, "AIxVR+vad": aix + VAD,
        "GxS+vad+llm": GXS + VAD + AXES, "AIxVR+vad+llm": aix + VAD + AXES,
        # b2 names
        "GxS": GXS,
    }[name]


def sub_for(split_or_subset):
    m = {"All": his, "D": his[his.grp == "D"], "T": his[his.grp == "T"],
         "60Hz": his[his.rate == "60Hz"], "90Hz": his[his.rate == "90Hz"],
         "60Hz-T": his[(his.rate == "60Hz") & (his.grp == "T")],
         "90Hz-D": his[(his.rate == "90Hz") & (his.grp == "D")]}
    return m[split_or_subset]


def augment(path, split_col, feat_col):
    tab = pd.read_csv(path)
    ps = []
    for _, r in tab.iterrows():
        sp = r[split_col]
        # split key for AIxVR feature lookup: b2 subsets use All-style [MG,BPM] except -T/-D
        aix_key = {"D": "D", "T": "T", "All": "All",
                   "60Hz": "All", "90Hz": "All", "60Hz-T": "T", "90Hz-D": "D"}[sp]
        aix = ab.AIXVR[aix_key]
        feats = {"gaze_GxS": GXS, "gaze_AIxVR": aix, "llm": AXES, "audio": VAD, "GxS": GXS,
                 "GxS+llm": GXS + AXES, "AIxVR+llm": aix + AXES, "audio+llm": VAD + AXES,
                 "GxS+vad": GXS + VAD, "AIxVR+vad": aix + VAD,
                 "GxS+vad+llm": GXS + VAD + AXES, "AIxVR+vad+llm": aix + VAD + AXES}[r[feat_col]]
        sub = sub_for(sp)
        accs = ab.perfold_accs(sub, feats, r["best_model"])
        maj = max(sub.y.mean(), 1 - sub.y.mean())
        _, p = ttest_1samp(accs, maj)
        ps.append(round(float(p), 4))
        print(f"OK {sp:8s} {r[feat_col]:14s} ttest_p={p:.4f}", flush=True)
    tab["p_ttest_maj"] = ps
    tab.to_csv(path, index=False)
    print(f"\nwrote {path}\n{tab.to_string(index=False)}\n")


augment("scratch/te_adoption/a1_out/b1_fusion_priors_llm.csv", "split", "features")
augment("scratch/te_adoption/a1_out/b2_rate_split.csv", "subset", "detector")
