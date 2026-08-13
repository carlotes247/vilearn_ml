"""C4 — acoustic (openSMILE) fusions without the v/a/d affect estimate.

Context (2026-08-13): the IUI paper keeps v/a/d and the LLM rubric in reserve for
the follow-up journal article, so the 88 eGeMAPS functionals are the paper's ONLY
acoustic modality. c2 never fused openSMILE with anything except v/a/d, so the
"multimodal (eye-tracking + acoustic + transcript)" claim had no acoustic fusion
cell behind it. This script fills the gap:

    linguistic+openSMILE, openSMILE+GazexSpeaking, linguistic+openSMILE+GazexSpeaking

Identical protocol to c2_paper_tables.py (185 windows, clean 2-annotator labels,
nested LOGO model selection, seed 42, 300-permutation test + one-sample t-test vs
majority floor, Cohen's d vs that floor) plus the c3 prediction-level metrics.
Writes to SEPARATE files so the canonical c2/c3 snapshots stay untouched.

Driver:  python3 discover/analysis_scripts/c4_acoustic_fusions.py
Worker:  PYTHONPATH=discover/analysis_scripts python3 discover/analysis_scripts/c4_acoustic_fusions.py <split> <detector>
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
PR = "discover/paper_results"
OUT = "scratch/te_adoption/a1_out"
RESULTS = f"{OUT}/c4_acoustic_fusions.csv"
PRF = f"{OUT}/c4_acoustic_fusions_prf.csv"

sys.path.insert(0, HERE)
import ab_common as ab
import c2_paper_tables as c2
import c3_prf_confusion as c3

SETS = ("linguistic+openSMILE", "openSMILE+GazexSpeaking",
        "linguistic+openSMILE+GazexSpeaking")
SPLITS = ("All", "D", "T")

if __name__ == "__main__" and len(sys.argv) == 3:  # ---------------- worker ----------------
    from scipy.stats import ttest_1samp
    sp, det = sys.argv[1], sys.argv[2]
    his = pd.read_csv(f"{PR}/c2_features.csv")
    sub = c2.sub_for(his, sp)
    feats = c2.feats_for(det, sp, c2.osm_cols(his))
    maj = max(sub.y.mean(), 1 - sub.y.mean())

    per_model = {}
    for model in ab.PANEL:
        try:
            per_model[model] = c2.perfold(sub, feats, model)
        except Exception as e:  # QDA singular covariance on high-dim sets
            print(f"skip {sp} {det} {model}: {type(e).__name__}", flush=True)
    if not per_model:
        sys.exit(f"FAIL {sp} {det}: all models failed")
    ranked = sorted(per_model, key=lambda m: -np.mean([a for _, a in per_model[m]]))
    try:
        pp = c2.perm_p_c2(sub, feats, ranked[0])
    except Exception as e:
        print(f"perm failed {sp} {det}: {type(e).__name__}", flush=True)
        pp = np.nan
    rows = []
    for rank, model in enumerate(ranked[:2], 1):
        accs = np.array([a for _, a in per_model[model]])
        t, p = ttest_1samp(accs, maj)
        sd = accs.std(ddof=1)
        d = (accs.mean() - maj) / sd if sd > 0 else np.nan
        rows.append(f"{sp},{det},{len(sub)},{len(feats)},{rank},{model},"
                    f"{accs.mean():.3f},{sd:.3f},{maj:.3f},"
                    f"{(f'{pp:.4f}' if rank == 1 else '')},{t:.3f},{p:.4f},{d:.3f}")
    with open(RESULTS, "a") as f:
        f.write("\n".join(rows) + "\n")

    m = c3.metrics(c3.oof_predictions(sub, feats, ranked[0]))
    with open(PRF, "a") as f:
        f.write(c3.fmt(sp, det, len(sub), len(feats), ranked[0], maj, m) + "\n")
    print(f"OK {sp} {det} best={ranked[0]} {np.mean([a for _, a in per_model[ranked[0]]]):.3f} "
          f"perm_p={pp:.4f} macroF1={m['macro_f1']:.3f}", flush=True)
    sys.exit(0)

# ---------------- driver ----------------
if __name__ == "__main__":
    from concurrent.futures import ThreadPoolExecutor
    os.makedirs(OUT, exist_ok=True)
    with open(RESULTS, "w") as f:
        f.write("split,detector,n,n_feat,rank,model,acc_mean,acc_std,majority,"
                "perm_p,t_maj,p_ttest_maj,cohen_d_maj\n")
    with open(PRF, "w") as f:
        f.write(c3.HDR)
    env = dict(os.environ, PYTHONPATH=HERE)

    def run_cell(job):
        sp, det = job
        p = subprocess.run([sys.executable, f"{HERE}/c4_acoustic_fusions.py", sp, det],
                           env=env, capture_output=True, text=True)
        print((p.stdout.strip() or f"FAIL {sp} {det}: {p.stderr[-400:]}"), flush=True)

    jobs = [(sp, det) for sp in SPLITS for det in SETS]
    with ThreadPoolExecutor(max_workers=min(9, os.cpu_count() or 4)) as ex:
        list(ex.map(run_cell, jobs))
    print("done —", RESULTS, flush=True)
