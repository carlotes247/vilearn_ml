import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import pearsonr, shapiro, spearmanr, ttest_rel
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from statsmodels.stats.multitest import multipletests

"""
Derive segment-level and session-role-level statistics for ViLearn and run simple models.

Inputs:
- Merged frame-level files: data/discover/merged/<session>.csv
- Transcript annotations: data/discover/<session>/transcript.<role>.helenrisack.csv

Outputs:
- data/discover/derived/segments.csv + .parquet
- data/discover/derived/session_role_stats.csv + .parquet
- data/discover/derived/model_metrics.json
- data/discover/derived/model_coefficients.csv
"""

# ------------------------
# CONFIG
# ------------------------
MERGED_ROOT = Path("data/discover/merged")
ANNOTATION_ROOT = Path("data/discover")
OUTPUT_ROOT = Path("data/discover/derived")
COMPARE_OUTPUT_ROOT = OUTPUT_ROOT / "compare"
SESSION_SET_FILES = [Path("discover/vilearn_more.set")]
ROLES = ["p_blue", "p_green", "p_red"]
GENERATE_PARQUET = False
USE_PARQUET_INPUT = True
STREAM_PREFIXES = ("opensmile_", "emow2v_", "sentiment_emb_")
FRAME_DOWNSAMPLE_STRIDE = 25
WRITE_LEGACY_OUTPUTS = False
PCA_EXPLAINED_VARIANCE = 0.95
PCA_MAX_COMPONENTS = 128


def load_sessions_from_set_files() -> list[str]:
    sessions: list[str] = []
    for set_file in SESSION_SET_FILES:
        if not set_file.exists():
            continue
        entries = [line.strip() for line in set_file.read_text().splitlines()]
        sessions.extend([e for e in entries if e])
    return sorted(set(sessions))


def tokenize_words(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text, flags=re.UNICODE)


def seg_mean_std(series: pd.Series) -> tuple[float, float]:
    if len(series) == 0:
        return np.nan, np.nan
    return float(series.mean()), float(series.std(ddof=0))


def build_segment_rows(
    session: str, role: str, frame_df: pd.DataFrame, tr_df: pd.DataFrame, include_streams: bool
) -> list[dict]:
    rows: list[dict] = []
    tr = tr_df.copy()
    tr["from"] = pd.to_numeric(tr["from"], errors="coerce")
    tr["to"] = pd.to_numeric(tr["to"], errors="coerce")
    tr = tr.dropna(subset=["from", "to"]).reset_index(drop=True)

    sentiment_col = f"sentiment_{role}"
    engagement_col = f"engagement_{role}"
    feature_cols = [sentiment_col, engagement_col, "task_engagement", "arousal", "dominance", "valence"]
    stream_cols = [c for c in frame_df.columns if c.startswith(STREAM_PREFIXES)] if include_streams else []

    for i, seg in tr.iterrows():
        start_ms = float(seg["from"])
        end_ms = float(seg["to"])
        text = str(seg.get("name", "") if pd.notna(seg.get("name", "")) else "")
        duration_s = max((end_ms - start_ms) / 1000.0, 0.0)

        words = tokenize_words(text)
        word_count = len(words)
        avg_word_len = float(np.mean([len(w) for w in words])) if words else 0.0
        words_per_second = float(word_count / duration_s) if duration_s > 0 else np.nan
        is_question = int(text.strip().endswith("?"))
        is_statement = int(word_count > 0 and not is_question)

        # right-open transcript intervals: [from, to)
        window = frame_df[(frame_df["time_ms"] >= start_ms) & (frame_df["time_ms"] < end_ms)]

        out = {
            "session": session,
            "role": role,
            "segment_idx": int(i),
            "from_ms": start_ms,
            "to_ms": end_ms,
            "segment_duration_s": duration_s,
            "text": text,
            "word_count": word_count,
            "avg_word_length": avg_word_len,
            "words_per_second": words_per_second,
            "question": is_question,
            "statement": is_statement,
        }

        for col in feature_cols:
            if col not in frame_df.columns:
                out[f"{col}_mean"] = np.nan
                out[f"{col}_std"] = np.nan
                continue
            m, s = seg_mean_std(window[col].dropna())
            out[f"{col}_mean"] = m
            out[f"{col}_std"] = s

        # Stream features are high-dimensional; keep segment means only.
        for col in stream_cols:
            out[f"{col}_mean"] = float(window[col].mean()) if col in window.columns else np.nan

        rows.append(out)

    return rows


def build_session_role_stats(segments_df: pd.DataFrame) -> pd.DataFrame:
    if segments_df.empty:
        return pd.DataFrame()

    feature_mean_cols = [c for c in segments_df.columns if c.endswith("_mean")]
    feature_std_cols = [c for c in segments_df.columns if c.endswith("_std")]
    cols_to_aggregate = feature_mean_cols + feature_std_cols

    grouped = []
    for (session, role), g in segments_df.groupby(["session", "role"], dropna=False):
        total_duration = float(g["segment_duration_s"].sum())
        total_words = int(g["word_count"].sum())
        row = {
            "session": session,
            "role": role,
            "segment_count": int(len(g)),
            "total_segment_duration_s": total_duration,
            "total_words": total_words,
            "words_per_second_overall": float(total_words / total_duration) if total_duration > 0 else np.nan,
            "avg_words_per_segment": float(g["word_count"].mean()),
            "avg_word_length": float(g["avg_word_length"].mean()),
            "question_ratio": float(g["question"].mean()),
            "statement_ratio": float(g["statement"].mean()),
        }

        for col in cols_to_aggregate:
            row[f"{col}_avg"] = float(g[col].mean()) if col in g.columns else np.nan
            row[f"{col}_spread"] = float(g[col].std(ddof=0)) if col in g.columns else np.nan

        grouped.append(row)

    return pd.DataFrame(grouped)


def transform_with_block_pca(
    x_train: pd.DataFrame, x_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    train_blocks = []
    test_blocks = []
    block_info: list[dict] = []

    stream_blocks = [
        ("opensmile", [c for c in x_train.columns if c.startswith("opensmile_")]),
        ("emow2v", [c for c in x_train.columns if c.startswith("emow2v_")]),
        ("sentiment_emb", [c for c in x_train.columns if c.startswith("sentiment_emb_")]),
    ]
    stream_cols = {c for _, cols in stream_blocks for c in cols}
    base_cols = [c for c in x_train.columns if c not in stream_cols]
    if base_cols:
        train_blocks.append(x_train[base_cols].reset_index(drop=True))
        test_blocks.append(x_test[base_cols].reset_index(drop=True))
        block_info.append({"block": "base", "input_dims": len(base_cols), "output_dims": len(base_cols)})

    for block_name, cols in stream_blocks:
        if not cols:
            continue
        tr = x_train[cols].to_numpy(dtype=float)
        te = x_test[cols].to_numpy(dtype=float)
        scaler = StandardScaler()
        tr_s = scaler.fit_transform(tr)
        te_s = scaler.transform(te)

        max_comp = min(PCA_MAX_COMPONENTS, tr_s.shape[0] - 1, tr_s.shape[1])
        if max_comp < 2:
            # Degenerate case: keep one standardized dimension.
            tr_df = pd.DataFrame(tr_s[:, :1], columns=[f"{block_name}_pc001"])
            te_df = pd.DataFrame(te_s[:, :1], columns=[f"{block_name}_pc001"])
            out_dims = 1
        else:
            pca = PCA(n_components=min(PCA_EXPLAINED_VARIANCE, max_comp), svd_solver="full")
            tr_p = pca.fit_transform(tr_s)
            te_p = pca.transform(te_s)
            out_dims = tr_p.shape[1]
            pc_cols = [f"{block_name}_pc{i + 1:03d}" for i in range(out_dims)]
            tr_df = pd.DataFrame(tr_p, columns=pc_cols)
            te_df = pd.DataFrame(te_p, columns=pc_cols)

        train_blocks.append(tr_df.reset_index(drop=True))
        test_blocks.append(te_df.reset_index(drop=True))
        block_info.append({"block": block_name, "input_dims": len(cols), "output_dims": out_dims})

    x_train_out = pd.concat(train_blocks, axis=1) if train_blocks else pd.DataFrame(index=x_train.index)
    x_test_out = pd.concat(test_blocks, axis=1) if test_blocks else pd.DataFrame(index=x_test.index)
    return x_train_out, x_test_out, block_info


def run_models(
    data_df: pd.DataFrame, target_col: str, name: str, use_pca: bool = False, include_streams: bool = True
) -> tuple[dict, pd.DataFrame]:
    prepared = _prepare_xy(data_df, target_col, include_streams)
    if prepared is None:
        return {"target": target_col, "status": "insufficient_rows", "rows": int(len(data_df))}, pd.DataFrame()
    x, y, _groups = prepared

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    pca_info: list[dict] = []
    if use_pca:
        x_train, x_test, pca_info = transform_with_block_pca(x_train, x_test)

    # Linear regression
    # Use ridge regularization to keep linear estimates stable in high-dimensional settings.
    lin = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=10.0))])
    lin.fit(x_train, y_train)
    y_pred = lin.predict(x_test)
    y_test_arr = np.asarray(y_test)
    lin_corr_pearson, lin_corr_pearson_p = pearsonr(y_pred, y_test_arr)
    lin_corr_spearman, lin_corr_spearman_p = spearmanr(y_pred, y_test_arr)
    lin_metrics = {
        "r2": float(r2_score(y_test, y_pred)),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "corr_pred_true_pearson": float(lin_corr_pearson),
        "corr_pred_true_pearson_p": float(lin_corr_pearson_p),
        "corr_pred_true_spearman": float(lin_corr_spearman),
        "corr_pred_true_spearman_p": float(lin_corr_spearman_p),
    }
    lin_coefs = pd.DataFrame(
        {
            "analysis": name,
            "target": target_col,
            "model": "linear_pca" if use_pca else "linear",
            "feature": x_train.columns,
            "coef": lin.named_steps["model"].coef_,
        }
    )

    # statsmodels OLS (for significance tests)
    sm_lin_summary = ""
    sm_lin_table = pd.DataFrame()
    if (not use_pca) and x.shape[1] <= 80:
        try:
            x_sm = sm.add_constant(x, has_constant="add").astype(float)
            ols = sm.OLS(y, x_sm).fit()
            sm_lin_summary = ols.summary().as_text()
            sm_lin_table = (
                pd.DataFrame(
                    {
                        "analysis": name,
                        "target": target_col,
                        "model": "linear_statsmodels",
                        "feature": ols.params.index,
                        "coef": ols.params.values,
                        "std_err": ols.bse.values,
                        "t": ols.tvalues.values,
                        "p_value": ols.pvalues.values,
                    }
                )
            )
            lin_metrics["f_pvalue"] = float(ols.f_pvalue) if ols.f_pvalue is not None else np.nan
            lin_metrics["adj_r2"] = float(ols.rsquared_adj)
        except Exception as e:
            sm_lin_summary = f"statsmodels OLS failed: {e}"
    else:
        if use_pca:
            sm_lin_summary = "statsmodels OLS skipped in PCA mode (component-space inference not directly interpretable)."
        else:
            sm_lin_summary = f"statsmodels OLS skipped: high-dimensional design matrix with {x.shape[1]} predictors"

    # Logistic regression (median split)
    thr = float(y.median())
    yb_train = (y_train > thr).astype(int)
    yb_test = (y_test > thr).astype(int)
    if yb_train.nunique() < 2 or yb_test.nunique() < 2:
        return {
            "target": target_col,
            "status": "single_class_for_logistic",
            "rows": int(len(y)),
            "linear": lin_metrics,
            "linear_statsmodels_summary": sm_lin_summary,
        }, pd.concat([lin_coefs, sm_lin_table], ignore_index=True)

    logi = Pipeline(
        [("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=2000, solver="lbfgs"))]
    )
    logi.fit(x_train, yb_train)
    yb_pred = logi.predict(x_test)
    yb_prob = logi.predict_proba(x_test)[:, 1]
    yb_test_arr = np.asarray(yb_test)
    log_corr_pearson, log_corr_pearson_p = pearsonr(yb_prob, yb_test_arr)
    log_corr_spearman, log_corr_spearman_p = spearmanr(yb_prob, yb_test_arr)
    logi_metrics = {
        "threshold_median": thr,
        "accuracy": float(accuracy_score(yb_test, yb_pred)),
        "roc_auc": float(roc_auc_score(yb_test, yb_prob)),
        "corr_pred_true_pearson": float(log_corr_pearson),
        "corr_pred_true_pearson_p": float(log_corr_pearson_p),
        "corr_pred_true_spearman": float(log_corr_spearman),
        "corr_pred_true_spearman_p": float(log_corr_spearman_p),
    }
    logi_coefs = pd.DataFrame(
        {
            "analysis": name,
            "target": target_col,
            "model": "logistic_pca" if use_pca else "logistic",
            "feature": x_train.columns,
            "coef": logi.named_steps["model"].coef_[0],
        }
    )

    # statsmodels Logit (for significance tests)
    sm_logi_summary = ""
    sm_logi_table = pd.DataFrame()
    if (not use_pca) and x.shape[1] <= 80:
        try:
            x_logit = sm.add_constant(x, has_constant="add").astype(float)
            y_bin_full = (y > thr).astype(int)
            logit_model = sm.Logit(y_bin_full, x_logit).fit(disp=False)
            sm_logi_summary = logit_model.summary().as_text()
            sm_logi_table = pd.DataFrame(
                {
                    "analysis": name,
                    "target": target_col,
                    "model": "logistic_statsmodels",
                    "feature": logit_model.params.index,
                    "coef": logit_model.params.values,
                    "std_err": logit_model.bse.values,
                    "z": logit_model.tvalues.values,
                    "p_value": logit_model.pvalues.values,
                }
            )
            logi_metrics["llr_pvalue"] = float(logit_model.llr_pvalue) if logit_model.llr_pvalue is not None else np.nan
            logi_metrics["pseudo_r2"] = float(logit_model.prsquared) if logit_model.prsquared is not None else np.nan
        except Exception as e:
            sm_logi_summary = f"statsmodels Logit failed: {e}"
    else:
        if use_pca:
            sm_logi_summary = "statsmodels Logit skipped in PCA mode (component-space inference not directly interpretable)."
        else:
            sm_logi_summary = f"statsmodels Logit skipped: high-dimensional design matrix with {x.shape[1]} predictors"

    metrics = {
        "target": target_col,
        "status": "ok",
        "rows": int(len(y)),
        "linear": lin_metrics,
        "logistic": logi_metrics,
        "linear_statsmodels_summary": sm_lin_summary,
        "logistic_statsmodels_summary": sm_logi_summary,
    }
    if pca_info:
        metrics["pca"] = pca_info
    metrics["include_streams"] = include_streams
    coef_tables = [lin_coefs, logi_coefs]
    if not sm_lin_table.empty:
        coef_tables.append(sm_lin_table)
    if not sm_logi_table.empty:
        coef_tables.append(sm_logi_table)
    return metrics, pd.concat(coef_tables, ignore_index=True)


def _prepare_xy(
    data_df: pd.DataFrame, target_col: str, include_streams: bool
) -> tuple[pd.DataFrame, pd.Series, pd.Series] | None:
    """Shared feature prep for run_models and run_models_cv. Returns (X, y, groups) or None."""
    predictors = [
        "segment_duration_s", "word_count", "avg_word_length", "words_per_second",
        "question", "statement", "sentiment_role", "speaking_role",
        "sentiment_p_blue", "sentiment_p_green", "sentiment_p_red",
        "arousal", "dominance", "valence",
        "sentiment_p_blue_mean", "sentiment_p_green_mean", "sentiment_p_red_mean",
        "arousal_mean", "dominance_mean", "valence_mean",
    ]
    use = data_df.copy()
    use = use.rename(columns={
        "sentiment_p_blue_mean": "sentiment_p_blue_mean",
        "sentiment_p_green_mean": "sentiment_p_green_mean",
        "sentiment_p_red_mean": "sentiment_p_red_mean",
        "arousal_mean": "arousal_mean",
        "dominance_mean": "dominance_mean",
        "valence_mean": "valence_mean",
        "task_engagement_mean": "task_engagement_mean",
    })
    stream_predictors = [
        c for c in use.columns
        if c.startswith(STREAM_PREFIXES) and (
            c.endswith("_mean") or c.startswith("opensmile_")
            or c.startswith("emow2v_") or c.startswith("sentiment_emb_")
        )
    ]
    predictor_cols = [c for c in predictors if c in use.columns and c != target_col]
    if include_streams:
        predictor_cols += stream_predictors
    predictor_cols = [c for c in predictor_cols if "engagement" not in c.lower()]
    keep_cols = ["session", "role", target_col] + predictor_cols
    use = use[[c for c in keep_cols if c in use.columns]].dropna(subset=[target_col]).copy()
    if len(use) < 20:
        return None
    groups = use["session"] if "session" in use.columns else pd.Series(["all"] * len(use))
    x = use.drop(columns=[c for c in ["session", target_col] if c in use.columns])
    if "role" in x.columns:
        x = pd.get_dummies(x, columns=["role"], drop_first=True)
    y = use[target_col].astype(float)
    x = x.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    return x, y, groups


def run_models_cv(
    data_df: pd.DataFrame,
    target_col: str,
    name: str,
    use_pca: bool = False,
    include_streams: bool = True,
) -> tuple[dict, pd.DataFrame]:
    """GroupKFold CV (leave-one-session-out). Returns cv_metrics dict and fold_scores DataFrame."""
    prepared = _prepare_xy(data_df, target_col, include_streams)
    if prepared is None:
        return {"target": target_col, "status": "insufficient_rows"}, pd.DataFrame()
    x, y, groups = prepared

    unique_sessions = groups.unique()
    n_splits = len(unique_sessions)
    if n_splits < 2:
        return {"target": target_col, "status": "insufficient_sessions", "sessions": int(n_splits)}, pd.DataFrame()

    gkf = GroupKFold(n_splits=n_splits)
    thr = float(y.median())

    fold_records: list[dict] = []
    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(x, y, groups=groups)):
        x_tr, x_te = x.iloc[train_idx].reset_index(drop=True), x.iloc[test_idx].reset_index(drop=True)
        y_tr, y_te = y.iloc[train_idx].reset_index(drop=True), y.iloc[test_idx].reset_index(drop=True)
        test_session = groups.iloc[test_idx].iloc[0]

        # Align columns after get_dummies (train may miss a role column)
        x_te = x_te.reindex(columns=x_tr.columns, fill_value=0)

        pca_info: list[dict] = []
        if use_pca:
            x_tr, x_te, pca_info = transform_with_block_pca(x_tr, x_te)

        # Ridge linear
        lin = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=10.0))])
        lin.fit(x_tr, y_tr)
        y_pred = lin.predict(x_te)
        lin_r2 = float(r2_score(y_te, y_pred))

        # Baseline linear (predict training mean)
        dummy_reg = DummyRegressor(strategy="mean")
        dummy_reg.fit(x_tr, y_tr)
        y_baseline = dummy_reg.predict(x_te)
        baseline_r2 = float(r2_score(y_te, y_baseline))

        fold_records.append({
            "fold": fold_idx, "session": test_session, "analysis": name,
            "target": target_col, "model": "linear_pca" if use_pca else "linear",
            "metric": "r2", "score": lin_r2, "baseline_score": baseline_r2,
        })

        # Logistic (median split)
        yb_tr = (y_tr > thr).astype(int)
        yb_te = (y_te > thr).astype(int)
        if yb_tr.nunique() < 2 or yb_te.nunique() < 2:
            fold_records.append({
                "fold": fold_idx, "session": test_session, "analysis": name,
                "target": target_col, "model": "logistic_pca" if use_pca else "logistic",
                "metric": "auc", "score": np.nan, "baseline_score": np.nan,
            })
            continue

        logi = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=2000, solver="lbfgs"))])
        logi.fit(x_tr, yb_tr)
        yb_prob = logi.predict_proba(x_te)[:, 1]
        logi_auc = float(roc_auc_score(yb_te, yb_prob))

        # Baseline logistic (stratified random = AUC ≈ 0.5)
        dummy_clf = DummyClassifier(strategy="stratified", random_state=42)
        dummy_clf.fit(x_tr, yb_tr)
        yb_base_prob = dummy_clf.predict_proba(x_te)[:, 1]
        baseline_auc = float(roc_auc_score(yb_te, yb_base_prob)) if yb_te.nunique() > 1 else 0.5

        fold_records.append({
            "fold": fold_idx, "session": test_session, "analysis": name,
            "target": target_col, "model": "logistic_pca" if use_pca else "logistic",
            "metric": "auc", "score": logi_auc, "baseline_score": baseline_auc,
        })

    fold_df = pd.DataFrame(fold_records)

    cv_metrics: dict = {"target": target_col, "status": "ok", "n_folds": n_splits}
    for metric_key, model_label in [("r2", "linear_pca" if use_pca else "linear"),
                                     ("auc", "logistic_pca" if use_pca else "logistic")]:
        sub = fold_df[(fold_df["metric"] == metric_key) & (fold_df["model"] == model_label)]["score"].dropna()
        cv_metrics[f"{model_label}_{metric_key}_mean"] = float(sub.mean()) if len(sub) else np.nan
        cv_metrics[f"{model_label}_{metric_key}_std"] = float(sub.std(ddof=1)) if len(sub) > 1 else np.nan

    return cv_metrics, fold_df


def run_ttest_vs_baseline(fold_df: pd.DataFrame) -> pd.DataFrame:
    """Paired t-test with Bonferroni correction, one row per (analysis, target, model, metric)."""
    results = []
    for (analysis, target, model, metric), grp in fold_df.groupby(["analysis", "target", "model", "metric"]):
        grp = grp.dropna(subset=["score", "baseline_score"])
        n = len(grp)
        if n < 3:
            continue
        scores = grp["score"].values
        baseline = grp["baseline_score"].values

        _, p_normal_model = shapiro(scores)
        _, p_normal_base = shapiro(baseline)
        _, p_val = ttest_rel(scores, baseline)

        results.append({
            "analysis": analysis, "target": target, "model": model, "metric": metric,
            "n_folds": n,
            "mean_model": float(scores.mean()), "std_model": float(scores.std(ddof=1)),
            "mean_baseline": float(baseline.mean()), "std_baseline": float(baseline.std(ddof=1)),
            "t_stat": float(ttest_rel(scores, baseline).statistic),
            "p_value": float(p_val),
            "normal_model_p": float(p_normal_model),
            "normal_baseline_p": float(p_normal_base),
        })

    if not results:
        return pd.DataFrame()

    ttest_df = pd.DataFrame(results)
    reject, p_adj, _, _ = multipletests(ttest_df["p_value"], alpha=0.05, method="bonferroni")
    ttest_df["p_value_adj"] = p_adj
    ttest_df["significant"] = reject
    return ttest_df


def build_frame_role_rows(session: str, frame_df: pd.DataFrame, role: str, stride: int) -> pd.DataFrame:
    sampled = frame_df.iloc[::stride].copy()
    engagement_col = f"engagement_{role}"
    sentiment_col = f"sentiment_{role}"
    speaking_col = f"speaking_{role}"
    sentiment_emb_cols = [c for c in sampled.columns if c.startswith(f"sentiment_emb_{role}_")]
    rename_emb = {c: c.replace(f"sentiment_emb_{role}_", "sentiment_emb_") for c in sentiment_emb_cols}
    opensmile_cols = [c for c in sampled.columns if c.startswith("opensmile_")]
    emow2v_cols = [c for c in sampled.columns if c.startswith("emow2v_")]

    keep = ["session", "time_ms", "task_engagement", "arousal", "dominance", "valence"] + opensmile_cols + emow2v_cols + sentiment_emb_cols
    if sentiment_col in sampled.columns:
        keep.append(sentiment_col)
    if speaking_col in sampled.columns:
        keep.append(speaking_col)
    if engagement_col in sampled.columns:
        keep.append(engagement_col)
    use = sampled[[c for c in keep if c in sampled.columns]].copy()
    if engagement_col not in use.columns:
        return pd.DataFrame()
    use = use.rename(columns=rename_emb)
    if sentiment_col in use.columns:
        use = use.rename(columns={sentiment_col: "sentiment_role"})
    if speaking_col in use.columns:
        use = use.rename(columns={speaking_col: "speaking_role"})
    use = use.rename(columns={engagement_col: "engagement_target"})
    use["role"] = role
    return use


def build_model_outputs(
    input_df: pd.DataFrame, use_pca: bool = False, include_streams: bool = True
) -> tuple[dict, list[pd.DataFrame], str, pd.DataFrame, pd.DataFrame]:
    """Returns (model_metrics, coef_frames, summary_text, fold_scores_df, ttest_df)."""
    model_metrics = {}
    coef_frames = []
    all_fold_scores: list[pd.DataFrame] = []

    m1, c1 = run_models(
        input_df, "engagement_target", "individual_engagement", use_pca=use_pca, include_streams=include_streams
    )
    model_metrics["individual_engagement"] = m1
    if not c1.empty:
        coef_frames.append(c1)

    m2, c2 = run_models(
        input_df, "task_engagement", "group_task_engagement", use_pca=use_pca, include_streams=include_streams
    )
    model_metrics["group_task_engagement"] = m2
    if not c2.empty:
        coef_frames.append(c2)

    cv1, fold1 = run_models_cv(
        input_df, "engagement_target", "individual_engagement", use_pca=use_pca, include_streams=include_streams
    )
    model_metrics["individual_engagement"]["cv"] = cv1
    if not fold1.empty:
        all_fold_scores.append(fold1)

    cv2, fold2 = run_models_cv(
        input_df, "task_engagement", "group_task_engagement", use_pca=use_pca, include_streams=include_streams
    )
    model_metrics["group_task_engagement"]["cv"] = cv2
    if not fold2.empty:
        all_fold_scores.append(fold2)

    fold_scores_df = pd.concat(all_fold_scores, ignore_index=True) if all_fold_scores else pd.DataFrame()
    ttest_df = run_ttest_vs_baseline(fold_scores_df) if not fold_scores_df.empty else pd.DataFrame()

    summary_chunks = []
    for k in ["individual_engagement", "group_task_engagement"]:
        m = model_metrics.get(k, {})
        lin_m = m.get("linear", {})
        log_m = m.get("logistic", {})
        cv_m = m.get("cv", {})
        summary_chunks.append(f"=== {k} ===")
        summary_chunks.append(
            "Linear pred-vs-true corr: "
            f"pearson={lin_m.get('corr_pred_true_pearson', np.nan):.4f} (p={lin_m.get('corr_pred_true_pearson_p', np.nan):.3g}), "
            f"spearman={lin_m.get('corr_pred_true_spearman', np.nan):.4f} (p={lin_m.get('corr_pred_true_spearman_p', np.nan):.3g})"
        )
        lin_cv_key = "linear_pca_r2" if use_pca else "linear_r2"
        summary_chunks.append(
            f"Linear CV (LOSO): R²={cv_m.get(f'{lin_cv_key}_mean', np.nan):.4f} ± {cv_m.get(f'{lin_cv_key}_std', np.nan):.4f}"
        )
        summary_chunks.append("Linear (statsmodels)")
        summary_chunks.append(m.get("linear_statsmodels_summary", "n/a"))
        summary_chunks.append("")
        summary_chunks.append(
            "Logistic pred-vs-true corr: "
            f"pearson={log_m.get('corr_pred_true_pearson', np.nan):.4f} (p={log_m.get('corr_pred_true_pearson_p', np.nan):.3g}), "
            f"spearman={log_m.get('corr_pred_true_spearman', np.nan):.4f} (p={log_m.get('corr_pred_true_spearman_p', np.nan):.3g})"
        )
        logi_cv_key = "logistic_pca_auc" if use_pca else "logistic_auc"
        summary_chunks.append(
            f"Logistic CV (LOSO): AUC={cv_m.get(f'{logi_cv_key}_mean', np.nan):.4f} ± {cv_m.get(f'{logi_cv_key}_std', np.nan):.4f}"
        )
        summary_chunks.append("Logistic (statsmodels)")
        summary_chunks.append(m.get("logistic_statsmodels_summary", "n/a"))
        summary_chunks.append("")

    if not ttest_df.empty:
        summary_chunks.append("=== T-Tests vs Baseline (Bonferroni corrected) ===")
        for _, row in ttest_df.iterrows():
            sig = "* SIGNIFICANT" if row["significant"] else "ns"
            summary_chunks.append(
                f"{row['analysis']} {row['target']} [{row['model']} {row['metric']}]: "
                f"model={row['mean_model']:.4f}±{row['std_model']:.4f} vs "
                f"baseline={row['mean_baseline']:.4f}±{row['std_baseline']:.4f}, "
                f"p={row['p_value']:.4f} (adj={row['p_value_adj']:.4f}) {sig}"
            )

    return model_metrics, coef_frames, "\n".join(summary_chunks), fold_scores_df, ttest_df


def write_model_artifacts(
    root: Path,
    tag: str,
    model_metrics: dict,
    coef_frames: list[pd.DataFrame],
    summary_text: str,
    fold_scores_df: pd.DataFrame | None = None,
    ttest_df: pd.DataFrame | None = None,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    summaries_path = root / f"model_summaries_{tag}.txt"
    summaries_path.write_text(summary_text)

    json_metrics = {}
    for k, v in model_metrics.items():
        vv = dict(v)
        vv.pop("linear_statsmodels_summary", None)
        vv.pop("logistic_statsmodels_summary", None)
        json_metrics[k] = vv
    (root / f"model_metrics_{tag}.json").write_text(json.dumps(json_metrics, indent=2))

    coef_path = root / f"model_coefficients_{tag}.csv"
    if coef_frames:
        pd.concat(coef_frames, ignore_index=True).to_csv(coef_path, index=False)
    else:
        pd.DataFrame(columns=["analysis", "target", "model", "feature", "coef"]).to_csv(coef_path, index=False)

    if fold_scores_df is not None and not fold_scores_df.empty:
        fold_scores_df.to_csv(root / f"cv_fold_scores_{tag}.csv", index=False)
    if ttest_df is not None and not ttest_df.empty:
        ttest_df.to_csv(root / f"cv_ttest_{tag}.csv", index=False)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    COMPARE_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    sessions = load_sessions_from_set_files()
    if not sessions:
        sessions = sorted([p.stem for p in MERGED_ROOT.glob("recording_*.csv")])

    segment_rows: list[dict] = []
    frame_rows: list[pd.DataFrame] = []
    for ses in sessions:
        merged_csv = MERGED_ROOT / f"{ses}.csv"
        merged_parquet = MERGED_ROOT / f"{ses}.parquet"
        if USE_PARQUET_INPUT and merged_parquet.exists():
            frame_df = pd.read_parquet(merged_parquet)
        elif merged_csv.exists():
            frame_df = pd.read_csv(merged_csv)
        else:
            continue
        for role in ROLES:
            tr_path = ANNOTATION_ROOT / ses / f"transcript.{role}.helenrisack.csv"
            if not tr_path.exists():
                continue
            tr_df = pd.read_csv(tr_path)
            segment_rows.extend(build_segment_rows(ses, role, frame_df, tr_df, include_streams=True))
            frame_role = build_frame_role_rows(ses, frame_df, role, FRAME_DOWNSAMPLE_STRIDE)
            if not frame_role.empty:
                frame_rows.append(frame_role)

    segments_df = pd.DataFrame(segment_rows)
    segments_compare_csv = COMPARE_OUTPUT_ROOT / "segments_streams.csv"
    segments_df.to_csv(segments_compare_csv, index=False)
    segments_compare_parquet = COMPARE_OUTPUT_ROOT / "segments_streams.parquet"
    if GENERATE_PARQUET:
        segments_df.to_parquet(segments_compare_parquet, index=False)

    session_role_df = build_session_role_stats(segments_df)
    session_role_compare_csv = COMPARE_OUTPUT_ROOT / "session_role_stats_streams.csv"
    session_role_df.to_csv(session_role_compare_csv, index=False)
    session_role_compare_parquet = COMPARE_OUTPUT_ROOT / "session_role_stats_streams.parquet"
    if GENERATE_PARQUET:
        session_role_df.to_parquet(session_role_compare_parquet, index=False)

    # Segment-level stream analysis
    segment_model_df = segments_df.copy()
    segment_model_df["engagement_target"] = np.nan
    for role in ROLES:
        col = f"engagement_{role}_mean"
        idx = segment_model_df["role"] == role
        if col in segment_model_df.columns:
            segment_model_df.loc[idx, "engagement_target"] = segment_model_df.loc[idx, col]
    segment_model_df["task_engagement"] = segment_model_df.get("task_engagement_mean", np.nan)
    seg_metrics, seg_coefs, seg_summary, seg_folds, seg_ttest = build_model_outputs(segment_model_df)
    write_model_artifacts(COMPARE_OUTPUT_ROOT, "segment_streams", seg_metrics, seg_coefs, seg_summary, seg_folds, seg_ttest)
    seg_pca_metrics, seg_pca_coefs, seg_pca_summary, seg_pca_folds, seg_pca_ttest = build_model_outputs(segment_model_df, use_pca=True)
    write_model_artifacts(COMPARE_OUTPUT_ROOT, "segment_streams_pca", seg_pca_metrics, seg_pca_coefs, seg_pca_summary, seg_pca_folds, seg_pca_ttest)
    seg_base_pca_metrics, seg_base_pca_coefs, seg_base_pca_summary, seg_base_pca_folds, seg_base_pca_ttest = build_model_outputs(
        segment_model_df, use_pca=True, include_streams=False
    )
    write_model_artifacts(
        COMPARE_OUTPUT_ROOT, "segment_base_only_pca", seg_base_pca_metrics, seg_base_pca_coefs, seg_base_pca_summary, seg_base_pca_folds, seg_base_pca_ttest
    )

    # Frame-level downsampled stream analysis
    frame_df_all = pd.concat(frame_rows, ignore_index=True) if frame_rows else pd.DataFrame()
    frame_df_all.to_csv(COMPARE_OUTPUT_ROOT / "frame_rows_streams_1hz.csv", index=False)
    if GENERATE_PARQUET and not frame_df_all.empty:
        frame_df_all.to_parquet(COMPARE_OUTPUT_ROOT / "frame_rows_streams_1hz.parquet", index=False)
    frame_metrics, frame_coefs, frame_summary, frame_folds, frame_ttest = build_model_outputs(frame_df_all)
    write_model_artifacts(COMPARE_OUTPUT_ROOT, "frame_streams_1hz", frame_metrics, frame_coefs, frame_summary, frame_folds, frame_ttest)
    frame_pca_metrics, frame_pca_coefs, frame_pca_summary, frame_pca_folds, frame_pca_ttest = build_model_outputs(frame_df_all, use_pca=True)
    write_model_artifacts(
        COMPARE_OUTPUT_ROOT, "frame_streams_1hz_pca", frame_pca_metrics, frame_pca_coefs, frame_pca_summary, frame_pca_folds, frame_pca_ttest
    )
    frame_base_pca_metrics, frame_base_pca_coefs, frame_base_pca_summary, frame_base_pca_folds, frame_base_pca_ttest = build_model_outputs(
        frame_df_all, use_pca=True, include_streams=False
    )
    write_model_artifacts(
        COMPARE_OUTPUT_ROOT,
        "frame_base_only_1hz_pca",
        frame_base_pca_metrics,
        frame_base_pca_coefs,
        frame_base_pca_summary,
        frame_base_pca_folds,
        frame_base_pca_ttest,
    )

    # Compact ablation table for stream utility checks.
    def get_metric(d: dict, analysis: str, target: str, key: str) -> float:
        return float(d.get(analysis, {}).get(target, {}).get(key, np.nan))

    ablation_rows = [
        {
            "analysis": "segment",
            "target": "individual_engagement",
            "model": "linear_r2",
            "base_only_pca": get_metric(seg_base_pca_metrics, "individual_engagement", "linear", "r2"),
            "streams_pca": get_metric(seg_pca_metrics, "individual_engagement", "linear", "r2"),
        },
        {
            "analysis": "segment",
            "target": "individual_engagement",
            "model": "logistic_auc",
            "base_only_pca": get_metric(seg_base_pca_metrics, "individual_engagement", "logistic", "roc_auc"),
            "streams_pca": get_metric(seg_pca_metrics, "individual_engagement", "logistic", "roc_auc"),
        },
        {
            "analysis": "segment",
            "target": "group_task_engagement",
            "model": "linear_r2",
            "base_only_pca": get_metric(seg_base_pca_metrics, "group_task_engagement", "linear", "r2"),
            "streams_pca": get_metric(seg_pca_metrics, "group_task_engagement", "linear", "r2"),
        },
        {
            "analysis": "segment",
            "target": "group_task_engagement",
            "model": "logistic_auc",
            "base_only_pca": get_metric(seg_base_pca_metrics, "group_task_engagement", "logistic", "roc_auc"),
            "streams_pca": get_metric(seg_pca_metrics, "group_task_engagement", "logistic", "roc_auc"),
        },
        {
            "analysis": "frame_1hz",
            "target": "individual_engagement",
            "model": "linear_r2",
            "base_only_pca": get_metric(frame_base_pca_metrics, "individual_engagement", "linear", "r2"),
            "streams_pca": get_metric(frame_pca_metrics, "individual_engagement", "linear", "r2"),
        },
        {
            "analysis": "frame_1hz",
            "target": "individual_engagement",
            "model": "logistic_auc",
            "base_only_pca": get_metric(frame_base_pca_metrics, "individual_engagement", "logistic", "roc_auc"),
            "streams_pca": get_metric(frame_pca_metrics, "individual_engagement", "logistic", "roc_auc"),
        },
        {
            "analysis": "frame_1hz",
            "target": "group_task_engagement",
            "model": "linear_r2",
            "base_only_pca": get_metric(frame_base_pca_metrics, "group_task_engagement", "linear", "r2"),
            "streams_pca": get_metric(frame_pca_metrics, "group_task_engagement", "linear", "r2"),
        },
        {
            "analysis": "frame_1hz",
            "target": "group_task_engagement",
            "model": "logistic_auc",
            "base_only_pca": get_metric(frame_base_pca_metrics, "group_task_engagement", "logistic", "roc_auc"),
            "streams_pca": get_metric(frame_pca_metrics, "group_task_engagement", "logistic", "roc_auc"),
        },
    ]
    ablation_df = pd.DataFrame(ablation_rows)
    ablation_df["delta_streams_minus_base"] = ablation_df["streams_pca"] - ablation_df["base_only_pca"]
    ablation_df.to_csv(COMPARE_OUTPUT_ROOT / "ablation_streams_vs_base_pca.csv", index=False)

    # Optional legacy outputs (kept off by default to avoid overwriting previous baseline files)
    if WRITE_LEGACY_OUTPUTS:
        segments_df.to_csv(OUTPUT_ROOT / "segments.csv", index=False)
        if GENERATE_PARQUET:
            segments_df.to_parquet(OUTPUT_ROOT / "segments.parquet", index=False)
        session_role_df.to_csv(OUTPUT_ROOT / "session_role_stats.csv", index=False)
        if GENERATE_PARQUET:
            session_role_df.to_parquet(OUTPUT_ROOT / "session_role_stats.parquet", index=False)
        write_model_artifacts(OUTPUT_ROOT, "", seg_metrics, seg_coefs, seg_summary, seg_folds, seg_ttest)

    print(f"Derived segments: {len(segments_df)}")
    print(f"Derived session-role rows: {len(session_role_df)}")
    print(f"Frame rows (downsampled): {len(frame_df_all)}")
    if GENERATE_PARQUET:
        print(
            "Wrote comparison outputs under "
            f"{COMPARE_OUTPUT_ROOT} (segment + frame, csv/json/txt and parquet where configured)"
        )
    else:
        print(f"Wrote comparison outputs under {COMPARE_OUTPUT_ROOT} (segment + frame, csv/json/txt)")


if __name__ == "__main__":
    main()
