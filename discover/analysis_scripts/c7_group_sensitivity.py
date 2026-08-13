"""C7 — does the headline survive the group-exclusion decisions?

The analysed set of 20 groups is the result of several filters that cannot be
reconstructed from a single criterion (see paper_abstract_2026-08-13.md §12). This
script reruns the paper's feature sets on four group sets, identical protocol
throughout (nested LOGO model selection over the 4-model panel, seed 42, 300-permutation
test, one-sample t-test vs the majority floor, Cohen's d, plus prediction-level metrics
from the out-of-fold predictions):

  g20  the analysed set                                     185 windows  (validation vs c2)
  g21  + dyad_09, the group excluded as a TE outlier        195 windows  (answers R1 Q3)
  g25  every group with usable data                         243 windows
  g16  g20 minus the four groups with Cronbach alpha < .60  153 windows

Feature matrices come from c6_build_groupset.build(), which reproduces
discover/paper_results/c2_features.csv bit-exactly on g20.

Driver:  python3 discover/analysis_scripts/c7_group_sensitivity.py
Worker:  PYTHONPATH=discover/analysis_scripts python3 discover/analysis_scripts/c7_group_sensitivity.py <groupset> <split> <detector>
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
OUT = "scratch/te_adoption/a1_out"
RESULTS = f"{OUT}/c7_sensitivity.csv"
PRF = f"{OUT}/c7_sensitivity_prf.csv"

sys.path.insert(0, HERE)
import ab_common as ab
import c2_paper_tables as c2
import c3_prf_confusion as c3
import c6_build_groupset as c6

SETS = ("AIxVR", "AIED", "GazexSpeaking", "linguistic",
        "linguistic+AIED", "linguistic+GazexSpeaking")
SPLITS = ("All", "D", "T")
GROUPSETS = ("g20", "g21", "g25", "g16")

HDR = ("groupset,split,detector,n,n_groups,n_feat,rank,model,acc_mean,acc_std,majority,"
       "perm_p,t_maj,p_ttest_maj,cohen_d_maj\n")


def feats_for(det, sp):
    return {"AIxVR": ab.AIXVR[sp], "AIED": c2.AIED, "GazexSpeaking": ab.GXS,
            "linguistic": c6.LING,
            "linguistic+AIED": c6.LING + c2.AIED,
            "linguistic+GazexSpeaking": c6.LING + ab.GXS}[det]


if __name__ == "__main__" and len(sys.argv) == 4:  # ---------------- worker ----------------
    from scipy.stats import ttest_1samp
    gs, sp, det = sys.argv[1], sys.argv[2], sys.argv[3]
    his = c6.build(c6.group_sets()[gs])
    sub = c2.sub_for(his, sp)
    feats = feats_for(det, sp)
    maj = max(sub.y.mean(), 1 - sub.y.mean())
    ng = sub.group_name.nunique()

    per_model = {}
    for model in ab.PANEL:
        try:
            per_model[model] = c2.perfold(sub, feats, model)
        except Exception as e:
            print(f"skip {gs} {sp} {det} {model}: {type(e).__name__}", flush=True)
    if not per_model:
        sys.exit(f"FAIL {gs} {sp} {det}: all models failed")
    ranked = sorted(per_model, key=lambda m: -np.mean([a for _, a in per_model[m]]))
    try:
        pp = c2.perm_p_c2(sub, feats, ranked[0])
    except Exception as e:
        print(f"perm failed {gs} {sp} {det}: {type(e).__name__}", flush=True)
        pp = np.nan
    rows = []
    for rank, model in enumerate(ranked[:2], 1):
        accs = np.array([a for _, a in per_model[model]])
        t, p = ttest_1samp(accs, maj)
        sd = accs.std(ddof=1)
        d = (accs.mean() - maj) / sd if sd > 0 else np.nan
        rows.append(f"{gs},{sp},{det},{len(sub)},{ng},{len(feats)},{rank},{model},"
                    f"{accs.mean():.3f},{sd:.3f},{maj:.3f},"
                    f"{(f'{pp:.4f}' if rank == 1 else '')},{t:.3f},{p:.4f},{d:.3f}")
    with open(RESULTS, "a") as f:
        f.write("\n".join(rows) + "\n")

    m = c3.metrics(c3.oof_predictions(sub, feats, ranked[0]))
    with open(PRF, "a") as f:
        f.write(f"{gs}," + c3.fmt(sp, det, len(sub), len(feats), ranked[0], maj, m) + "\n")
    print(f"OK {gs} {sp} {det} best={ranked[0]} acc={np.mean([a for _, a in per_model[ranked[0]]]):.3f} "
          f"macroF1={m['macro_f1']:.3f} perm={pp:.4f}", flush=True)
    sys.exit(0)

# ---------------- driver ----------------
if __name__ == "__main__":
    from concurrent.futures import ThreadPoolExecutor
    os.makedirs(OUT, exist_ok=True)
    with open(RESULTS, "w") as f:
        f.write(HDR)
    with open(PRF, "w") as f:
        f.write("groupset," + c3.HDR)
    env = dict(os.environ, PYTHONPATH=HERE)

    def run_cell(job):
        gs, sp, det = job
        p = subprocess.run([sys.executable, f"{HERE}/c7_group_sensitivity.py", gs, sp, det],
                           env=env, capture_output=True, text=True)
        print((p.stdout.strip() or f"FAIL {gs} {sp} {det}: {p.stderr[-400:]}"), flush=True)

    jobs = [(gs, sp, det) for gs in GROUPSETS for sp in SPLITS for det in SETS]
    print(f"{len(jobs)} cells", flush=True)
    with ThreadPoolExecutor(max_workers=min(12, os.cpu_count() or 4)) as ex:
        list(ex.map(run_cell, jobs))
    print("done —", RESULTS, flush=True)
