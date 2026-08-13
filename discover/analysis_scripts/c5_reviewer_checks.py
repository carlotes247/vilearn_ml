"""C5 — two analyses that answer the two live reviewer objections from the last
rejection round (see discover/paper_results/paper_abstract_2026-08-13.md §11).

A) CIRCULARITY / SHARED-SHORTCUT CHECK.
   R1: annotators saw gaze rays while labelling, and gaze is also model input, so a
   gaze detector may be reading the annotation channel back out. If a single
   annotation shortcut drove both, the gaze-only and transcript-only detectors would
   make the SAME predictions on the SAME windows. We test that directly on the
   out-of-fold predictions: agreement, Cohen's kappa between the two detectors,
   the 2x2 error overlap, McNemar, and the phi coefficient between their error
   indicators. Independent errors => two distinct signals, not one shared readout.

B) SMALL-SAMPLE UNCERTAINTY.
   R2/R3: 20 groups is small. We report group-level bootstrap CIs (resample GROUPS
   with replacement, so the dependency structure is respected) on pooled macro-F1
   and accuracy for the headline cells, plus the fraction of resamples in which the
   fusion beats the majority floor and beats the transcript-only tier.

Reads discover/paper_results/c3_predictions.csv only — no retraining.

Run:  python3 discover/analysis_scripts/c5_reviewer_checks.py
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score

PR = "discover/paper_results"
OUT = "scratch/te_adoption/a1_out"
RS = 42
N_BOOT = 10_000

PAIRS = [("GazexSpeaking", "linguistic"),
         ("GazexSpeaking", "linguistic+GazexSpeaking"),
         ("linguistic", "linguistic+GazexSpeaking")]
HEADLINE = ["baseline_majority", "GazexSpeaking", "linguistic",
            "linguistic+GazexSpeaking", "linguistic+AIED"]
SPLITS = ["All", "D", "T"]


def get(preds, sp, det):
    d = preds[(preds.split == sp) & (preds.detector == det)]
    return d.set_index("test_group")[["y_true", "y_pred"]]


def error_overlap(preds, sp, a, b):
    """2x2 table of which detector is wrong on which window + independence stats."""
    from scipy.stats import chi2_contingency
    A, B = get(preds, sp, a), get(preds, sp, b)
    assert len(A) == len(B) and (A.y_true.values == B.y_true.values).all()
    ea = (A.y_pred.values != A.y_true.values).astype(int)
    eb = (B.y_pred.values != B.y_true.values).astype(int)
    n11 = int(((ea == 1) & (eb == 1)).sum())   # both wrong
    n10 = int(((ea == 1) & (eb == 0)).sum())   # only a wrong
    n01 = int(((ea == 0) & (eb == 1)).sum())   # only b wrong
    n00 = int(((ea == 0) & (eb == 0)).sum())   # both right
    # McNemar (exact binomial, discordant pairs)
    from scipy.stats import binomtest
    mc = binomtest(n10, n10 + n01, 0.5).pvalue if (n10 + n01) else np.nan
    phi = np.corrcoef(ea, eb)[0, 1] if ea.std() and eb.std() else np.nan
    tbl = np.array([[n00, n01], [n10, n11]])
    try:
        chi_p = chi2_contingency(tbl)[1] if tbl.min() >= 0 and tbl.sum() else np.nan
    except ValueError:
        chi_p = np.nan
    return dict(split=sp, det_a=a, det_b=b, n=len(A),
                acc_a=1 - ea.mean(), acc_b=1 - eb.mean(),
                pred_agreement=(A.y_pred.values == B.y_pred.values).mean(),
                kappa_preds=cohen_kappa_score(A.y_pred.values, B.y_pred.values),
                both_right=n00, only_a_wrong=n10, only_b_wrong=n01, both_wrong=n11,
                phi_errors=phi, p_errors_indep=chi_p, p_mcnemar=mc)


def boot_ci(preds, sp, det, rng):
    d = get(preds, sp, det).reset_index()
    groups = d.test_group.unique()
    by = {g: d[d.test_group == g] for g in groups}
    accs, f1s = [], []
    for _ in range(N_BOOT):
        pick = rng.choice(groups, size=len(groups), replace=True)
        s = pd.concat([by[g] for g in pick])
        accs.append((s.y_true.values == s.y_pred.values).mean())
        f1s.append(f1_score(s.y_true, s.y_pred, average="macro", zero_division=0))
    return np.array(accs), np.array(f1s)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    preds = pd.read_csv(f"{PR}/c3_predictions.csv")

    # ---- A) shared-shortcut check -------------------------------------------
    rows = [error_overlap(preds, sp, a, b) for sp in SPLITS for a, b in PAIRS]
    A = pd.DataFrame(rows)
    A.to_csv(f"{OUT}/c5_error_independence.csv", index=False)
    print("=== A) do gaze-only and transcript-only detectors err on the same windows? ===")
    print(A.round(3).to_string(index=False), "\n")

    # ---- B) group-level bootstrap -------------------------------------------
    rng = np.random.default_rng(RS)
    out = []
    cache = {}
    for sp in SPLITS:
        for det in HEADLINE:
            if not len(get(preds, sp, det)):
                continue
            acc, f1 = boot_ci(preds, sp, det, rng)
            cache[(sp, det)] = (acc, f1)
            out.append(dict(split=sp, detector=det, n_boot=N_BOOT,
                            acc=acc.mean(), acc_lo=np.percentile(acc, 2.5),
                            acc_hi=np.percentile(acc, 97.5),
                            macro_f1=f1.mean(), f1_lo=np.percentile(f1, 2.5),
                            f1_hi=np.percentile(f1, 97.5)))
    B = pd.DataFrame(out)
    # paired bootstrap contrasts on the SAME resamples would need shared indices;
    # rerun the contrast pairs with a fresh, shared stream instead
    contrasts = []
    for sp in SPLITS:
        for a, b in [("linguistic+GazexSpeaking", "baseline_majority"),
                     ("linguistic+GazexSpeaking", "linguistic"),
                     ("linguistic", "baseline_majority")]:
            da, db = get(preds, sp, a).reset_index(), get(preds, sp, b).reset_index()
            groups = da.test_group.unique()
            ga = {g: da[da.test_group == g] for g in groups}
            gb = {g: db[db.test_group == g] for g in groups}
            r = np.random.default_rng(RS)
            diffs = []
            for _ in range(N_BOOT):
                pick = r.choice(groups, size=len(groups), replace=True)
                sa = pd.concat([ga[g] for g in pick])
                sb = pd.concat([gb[g] for g in pick])
                diffs.append(f1_score(sa.y_true, sa.y_pred, average="macro", zero_division=0)
                             - f1_score(sb.y_true, sb.y_pred, average="macro", zero_division=0))
            diffs = np.array(diffs)
            contrasts.append(dict(split=sp, a=a, b=b, d_macro_f1=diffs.mean(),
                                  lo=np.percentile(diffs, 2.5), hi=np.percentile(diffs, 97.5),
                                  p_a_better=(diffs > 0).mean()))
    C = pd.DataFrame(contrasts)
    B.to_csv(f"{OUT}/c5_bootstrap_ci.csv", index=False)
    C.to_csv(f"{OUT}/c5_bootstrap_contrasts.csv", index=False)
    print("=== B) group-level bootstrap (resample groups, 10k) ===")
    print(B.round(3).to_string(index=False), "\n")
    print("=== B2) paired bootstrap contrasts (macro-F1) ===")
    print(C.round(3).to_string(index=False))
