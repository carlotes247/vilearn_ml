"""C3 — precision / recall / F1 / confusion matrices for every paper cell.

Same canonical protocol as c2_paper_tables.py (185 floorlevel 60 s group-windows,
clean 2-annotator-mean TE binarised at 0.5, nested LOGO model selection, seed 42),
but instead of only per-fold accuracies we KEEP THE OUT-OF-FOLD PREDICTIONS. Every
window is predicted exactly once (by the model trained without its group), so the
pooled confusion matrix is a proper leave-one-group-out prediction matrix.

For each (split, detector) we rerun the rank-1 model recorded in
`discover/paper_results/c2_paper_table.csv`.

Reported per cell:
  - acc_foldmean (reproduces c2_paper_table acc_mean) and acc_pooled
  - confusion matrix TN/FP/FN/TP with class 1 = High TE, class 0 = Low TE
  - per-class precision / recall (= class accuracy) / F1
  - macro-F1, weighted-F1, balanced accuracy, MCC
  - macro-F1 averaged over folds (+ SD) for consistency with the accuracy convention

Feature matrices are read from the TRACKED snapshots in discover/paper_results/
(c2_features.csv, c2_emb_features.csv) — no parquets needed.

Driver:  python3 discover/analysis_scripts/c3_prf_confusion.py
Worker:  PYTHONPATH=discover/analysis_scripts python3 discover/analysis_scripts/c3_prf_confusion.py <split> <detector>
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
RESULTS = f"{OUT}/c3_prf.csv"
PREDS = f"{OUT}/c3_predictions.csv"

sys.path.insert(0, HERE)
import ab_common as ab
import c2_paper_tables as c2  # import-safe (has __main__ guards)

HDR = ("split,detector,n,n_feat,model,acc_foldmean,acc_pooled,majority,"
       "TN,FP,FN,TP,prec_low,rec_low,f1_low,prec_high,rec_high,f1_high,"
       "macro_f1,weighted_f1,balanced_acc,mcc,macro_f1_foldmean,macro_f1_foldsd\n")


def load_features(det):
    """Tracked feature snapshot; embedding sets get the emb matrix merged in."""
    his = pd.read_csv(f"{PR}/c2_features.csv")
    base = det[:-4] if det.endswith("_sel") else det
    emo, semb = [], []
    if base in ("emow2v_pca", "sentemb_pca"):
        emb = pd.read_csv(f"{PR}/c2_emb_features.csv")
        emo = [c for c in emb.columns if c.startswith("emo_")]
        semb = [c for c in emb.columns if c.startswith("semb_")]
        his = his.merge(emb, on=["group_name", "sec"], validate="one_to_one")
    return his, emo, semb


def oof_predictions(sub, feats, model, pca=False, sel=False):
    """Out-of-fold LOGO predictions for one model (inner GridSearch on its grid).
    Mirrors c2_paper_tables.perfold but returns (test_group, y_true, y_pred)."""
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.model_selection import GridSearchCV, GroupKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    X = sub[feats].values
    y = sub["y"].values
    g = sub["group_name"].values
    clf, grid = ab.PANEL[model]
    steps = [("s", StandardScaler())]
    if pca:
        steps.append(("p", PCA(random_state=ab.RS)))
    if sel:
        steps.append(("k", SelectKBest(f_classif)))
    steps.append(("m", clf))
    pgrid = {f"m__{k}": v for k, v in grid.items()}
    if pca:
        pgrid["p__n_components"] = [5, 10, 20]
    if sel:
        n = len(feats)
        if n <= 20:
            pgrid["k__k"] = list(range(1, n + 1))
        else:
            pgrid["k__k"] = sorted({k for k in (1, 2, 3, 4, 6, 8, 11, 16, 22, 32, 45, 64, n) if k <= n})
    gkf = GroupKFold(n_splits=pd.Series(g).nunique())
    rows = []
    for tr, te in gkf.split(X, y, groups=g):
        if pd.Series(y[tr]).nunique() < 2:
            continue
        inner = GroupKFold(n_splits=int(pd.Series(g[tr]).nunique()))
        gs = GridSearchCV(Pipeline(steps), pgrid, cv=inner, n_jobs=1, error_score=np.nan)
        gs.fit(X[tr], y[tr], groups=g[tr])
        pred = gs.best_estimator_.predict(X[te])
        tg = str(pd.Series(g[te]).iloc[0])
        for yt, yp in zip(y[te], pred):
            rows.append((tg, int(yt), int(yp)))
    return rows


def majority_predictions(sub):
    """Per-fold majority-class baseline: predict the training-majority label."""
    from sklearn.model_selection import GroupKFold
    y = sub["y"].values
    g = sub["group_name"].values
    rows = []
    gkf = GroupKFold(n_splits=pd.Series(g).nunique())
    for tr, te in gkf.split(y.reshape(-1, 1), y, groups=g):
        m = 1 if y[tr].mean() >= 0.5 else 0
        tg = str(pd.Series(g[te]).iloc[0])
        for yt in y[te]:
            rows.append((tg, int(yt), m))
    return rows


def metrics(rows):
    """Pooled + per-fold metrics from (group, y_true, y_pred) triples."""
    from sklearn.metrics import (balanced_accuracy_score, confusion_matrix,
                                 f1_score, matthews_corrcoef, precision_recall_fscore_support)
    df = pd.DataFrame(rows, columns=["group", "y", "p"])
    y, p = df.y.values, df.p.values
    cm = confusion_matrix(y, p, labels=[0, 1])
    (tn, fp), (fn, tp) = cm
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, labels=[0, 1], zero_division=0)
    fold_f1 = df.groupby("group").apply(
        lambda d: f1_score(d.y, d.p, labels=[0, 1], average="macro", zero_division=0))
    return dict(
        acc_pooled=(y == p).mean(),
        acc_foldmean=df.groupby("group").apply(lambda d: (d.y == d.p).mean()).mean(),
        TN=tn, FP=fp, FN=fn, TP=tp,
        prec_low=pr[0], rec_low=rc[0], f1_low=f1[0],
        prec_high=pr[1], rec_high=rc[1], f1_high=f1[1],
        macro_f1=f1_score(y, p, average="macro", zero_division=0),
        weighted_f1=f1_score(y, p, average="weighted", zero_division=0),
        balanced_acc=balanced_accuracy_score(y, p),
        mcc=matthews_corrcoef(y, p) if len(set(p)) > 1 else 0.0,
        macro_f1_foldmean=fold_f1.mean(), macro_f1_foldsd=fold_f1.std(ddof=1),
    )


def fmt(sp, det, n, n_feat, model, maj, m):
    return (f"{sp},{det},{n},{n_feat},{model},{m['acc_foldmean']:.3f},{m['acc_pooled']:.3f},{maj:.3f},"
            f"{m['TN']},{m['FP']},{m['FN']},{m['TP']},"
            f"{m['prec_low']:.3f},{m['rec_low']:.3f},{m['f1_low']:.3f},"
            f"{m['prec_high']:.3f},{m['rec_high']:.3f},{m['f1_high']:.3f},"
            f"{m['macro_f1']:.3f},{m['weighted_f1']:.3f},{m['balanced_acc']:.3f},{m['mcc']:.3f},"
            f"{m['macro_f1_foldmean']:.3f},{m['macro_f1_foldsd']:.3f}")


if __name__ == "__main__" and len(sys.argv) == 3:  # ---------------- worker ----------------
    sp, det = sys.argv[1], sys.argv[2]
    tbl = pd.read_csv(f"{PR}/c2_paper_table.csv")
    row = tbl[(tbl.split == sp) & (tbl.detector == det) & (tbl["rank"] == 1)].iloc[0]
    model = row.model
    his, emo, semb = load_features(det)
    sub = c2.sub_for(his, sp)
    maj = max(sub.y.mean(), 1 - sub.y.mean())

    if det == "baseline_majority":
        rows, n_feat = majority_predictions(sub), 0
    else:
        base = det[:-4] if det.endswith("_sel") else det
        feats = c2.feats_for(base, sp, c2.osm_cols(his), emo, semb)
        rows = oof_predictions(sub, feats, model, pca=base.endswith("_pca"), sel=det.endswith("_sel"))
        n_feat = len(feats)
    m = metrics(rows)
    with open(RESULTS, "a") as f:
        f.write(fmt(sp, det, len(sub), n_feat, model, maj, m) + "\n")
    with open(PREDS, "a") as f:
        for tg, yt, yp in rows:
            f.write(f"{sp},{det},{model},{tg},{yt},{yp}\n")
    print(f"OK {sp} {det} {model} acc={m['acc_foldmean']:.3f} "
          f"(c2 {row.acc_mean:.3f}) macroF1={m['macro_f1']:.3f}", flush=True)
    sys.exit(0)

# ---------------- driver ----------------
if __name__ == "__main__":
    from concurrent.futures import ThreadPoolExecutor
    os.makedirs(OUT, exist_ok=True)
    tbl = pd.read_csv(f"{PR}/c2_paper_table.csv")
    jobs = [(r.split, r.detector) for _, r in tbl[tbl["rank"] == 1].iterrows()]
    with open(RESULTS, "w") as f:
        f.write(HDR)
    with open(PREDS, "w") as f:
        f.write("split,detector,model,test_group,y_true,y_pred\n")
    env = dict(os.environ, PYTHONPATH=HERE)

    def run_cell(job):
        sp, det = job
        p = subprocess.run([sys.executable, f"{HERE}/c3_prf_confusion.py", sp, det],
                           env=env, capture_output=True, text=True)
        print((p.stdout.strip() or f"FAIL {sp} {det}: {p.stderr[-400:]}"), flush=True)

    with ThreadPoolExecutor(max_workers=min(12, os.cpu_count() or 4)) as ex:
        list(ex.map(run_cell, jobs))
    print("done —", RESULTS, flush=True)
