"""Ad-hoc numbers used in the presentation that are NOT emitted by the main
classifier panel in derive_vilearn_stats.py:

  - 1 Hz frame group-TE accuracy (the panel excludes frame-1Hz: SVM-rbf is
    intractable on ~27k rows, so we compute it here with fast models only)
  - 60 s window group-TE accuracy: base vs base+embeddings (streams_pca)

Reuses _prepare_xy / transform_with_block_pca from derive_vilearn_stats.py.
Reads the per-granularity row CSVs written by the main script under
data/discover/derived/compare/. Run from the repo root:

    python discover/presentation_scripts/compute_extra_panel_numbers.py
"""
import importlib.util
import warnings

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
COMPARE = "data/discover/derived/compare"

spec = importlib.util.spec_from_file_location("d", "discover/derive_vilearn_stats.py")
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)


def loso_group_te_accuracy(df: pd.DataFrame, include_streams: bool, use_pca: bool) -> dict:
    """Leave-one-group-out mean accuracy per model for group task engagement (>0.5)."""
    x, y, g = d._prepare_xy(df, "task_engagement", include_streams)
    yb = (y > 0.5).astype(int)
    gkf = GroupKFold(n_splits=g.nunique())
    models = {
        "QDA": QuadraticDiscriminantAnalysis(reg_param=0.1),
        "LogReg": LogisticRegression(max_iter=2000),
        "NB": GaussianNB(),
        "RF": RandomForestClassifier(n_estimators=200, random_state=42),
    }
    accs = {m: [] for m in models}
    for tr, te in gkf.split(x, yb, groups=g):
        xtr, xte = x.iloc[tr].reset_index(drop=True), x.iloc[te].reset_index(drop=True)
        ytr, yte = yb.iloc[tr], yb.iloc[te]
        xte = xte.reindex(columns=xtr.columns, fill_value=0)
        if ytr.nunique() < 2:
            continue
        if use_pca:
            xtr, xte, _ = d.transform_with_block_pca(xtr, xte)
        for m, clf in models.items():
            try:
                pred = Pipeline([("s", StandardScaler()), ("m", clf)]).fit(xtr, ytr).predict(xte)
                accs[m].append(accuracy_score(yte, pred))
            except Exception:
                pass
    return {m: round(float(np.mean(v)), 3) for m, v in accs.items() if v}


if __name__ == "__main__":
    frame = pd.read_csv(f"{COMPARE}/frame_rows_streams_1hz.csv")
    window = pd.read_csv(f"{COMPARE}/window_rows_streams_60s.csv")
    print("1 Hz frame  group TE acc:", loso_group_te_accuracy(frame, False, False))
    print("60 s base   group TE acc:", loso_group_te_accuracy(window, False, False))
    print("60 s +embed group TE acc:", loso_group_te_accuracy(window, True, True))
    # segment number (0.59) comes from classifier_panel_metrics.csv (acc/pooled);
    # the main panel already covers segment + 60 s window.
