import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
SESSION_SET_FILES = [Path("discover/vilearn_more.set")]
ROLES = ["p_blue", "p_green", "p_red"]
GENERATE_PARQUET = False


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


def build_segment_rows(session: str, role: str, frame_df: pd.DataFrame, tr_df: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    tr = tr_df.copy()
    tr["from"] = pd.to_numeric(tr["from"], errors="coerce")
    tr["to"] = pd.to_numeric(tr["to"], errors="coerce")
    tr = tr.dropna(subset=["from", "to"]).reset_index(drop=True)

    sentiment_col = f"sentiment_{role}"
    engagement_col = f"engagement_{role}"
    feature_cols = [sentiment_col, engagement_col, "task_engagement", "arousal", "dominance", "valence"]

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


def run_models(segments_df: pd.DataFrame, target_col: str, name: str) -> tuple[dict, pd.DataFrame]:
    # Leakage-safe explanatory set (task-agnostic):
    # Do not use engagement or task_engagement as predictors for either engagement target.
    predictors = [
        "segment_duration_s",
        "word_count",
        "avg_word_length",
        "words_per_second",
        "question",
        "statement",
        "sentiment_p_blue_mean",
        "sentiment_p_green_mean",
        "sentiment_p_red_mean",
        "arousal_mean",
        "dominance_mean",
        "valence_mean",
    ]

    use = segments_df.copy()
    use = use.rename(
        columns={
            "sentiment_p_blue_mean": "sentiment_p_blue_mean",
            "sentiment_p_green_mean": "sentiment_p_green_mean",
            "sentiment_p_red_mean": "sentiment_p_red_mean",
            "arousal_mean": "arousal_mean",
            "dominance_mean": "dominance_mean",
            "valence_mean": "valence_mean",
            "task_engagement_mean": "task_engagement_mean",
        }
    )

    predictor_cols = [c for c in predictors if c in use.columns and c != target_col]
    # Safety guard against leakage: never use any engagement variable as predictor.
    predictor_cols = [c for c in predictor_cols if "engagement" not in c.lower()]
    keep_cols = ["role", target_col] + predictor_cols
    use = use[keep_cols].dropna(subset=[target_col]).copy()
    if len(use) < 20:
        return {"target": target_col, "status": "insufficient_rows", "rows": int(len(use))}, pd.DataFrame()

    x = use.drop(columns=[target_col])
    x = pd.get_dummies(x, columns=["role"], drop_first=True)
    y = use[target_col].astype(float)

    x = x.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    x = x.astype(float)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Linear regression
    lin = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
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
            "model": "linear",
            "feature": x.columns,
            "coef": lin.named_steps["model"].coef_,
        }
    )

    # statsmodels OLS (for significance tests)
    sm_lin_summary = ""
    sm_lin_table = pd.DataFrame()
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

    # Logistic regression (median split)
    thr = float(y.median())
    y_bin = (y > thr).astype(int)
    if y_bin.nunique() < 2:
        return {
            "target": target_col,
            "status": "single_class_for_logistic",
            "rows": int(len(use)),
            "linear": lin_metrics,
            "linear_statsmodels_summary": sm_lin_summary,
        }, pd.concat([lin_coefs, sm_lin_table], ignore_index=True)

    xb_train, xb_test, yb_train, yb_test = train_test_split(x, y_bin, test_size=0.2, random_state=42)
    logi = Pipeline(
        [("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=2000, solver="lbfgs"))]
    )
    logi.fit(xb_train, yb_train)
    yb_pred = logi.predict(xb_test)
    yb_prob = logi.predict_proba(xb_test)[:, 1]
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
            "model": "logistic",
            "feature": x.columns,
            "coef": logi.named_steps["model"].coef_[0],
        }
    )

    # statsmodels Logit (for significance tests)
    sm_logi_summary = ""
    sm_logi_table = pd.DataFrame()
    try:
        x_logit = sm.add_constant(x, has_constant="add").astype(float)
        logit_model = sm.Logit(y_bin, x_logit).fit(disp=False)
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

    metrics = {
        "target": target_col,
        "status": "ok",
        "rows": int(len(use)),
        "linear": lin_metrics,
        "logistic": logi_metrics,
        "linear_statsmodels_summary": sm_lin_summary,
        "logistic_statsmodels_summary": sm_logi_summary,
    }
    coef_tables = [lin_coefs, logi_coefs]
    if not sm_lin_table.empty:
        coef_tables.append(sm_lin_table)
    if not sm_logi_table.empty:
        coef_tables.append(sm_logi_table)
    return metrics, pd.concat(coef_tables, ignore_index=True)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    sessions = load_sessions_from_set_files()
    if not sessions:
        sessions = sorted([p.stem for p in MERGED_ROOT.glob("recording_*.csv")])

    segment_rows: list[dict] = []
    for ses in sessions:
        merged_path = MERGED_ROOT / f"{ses}.csv"
        if not merged_path.exists():
            continue
        frame_df = pd.read_csv(merged_path)
        for role in ROLES:
            tr_path = ANNOTATION_ROOT / ses / f"transcript.{role}.helenrisack.csv"
            if not tr_path.exists():
                continue
            tr_df = pd.read_csv(tr_path)
            segment_rows.extend(build_segment_rows(ses, role, frame_df, tr_df))

    segments_df = pd.DataFrame(segment_rows)
    segments_csv = OUTPUT_ROOT / "segments.csv"
    segments_df.to_csv(segments_csv, index=False)
    segments_parquet = OUTPUT_ROOT / "segments.parquet"
    if GENERATE_PARQUET:
        segments_df.to_parquet(segments_parquet, index=False)

    session_role_df = build_session_role_stats(segments_df)
    session_role_csv = OUTPUT_ROOT / "session_role_stats.csv"
    session_role_df.to_csv(session_role_csv, index=False)
    session_role_parquet = OUTPUT_ROOT / "session_role_stats.parquet"
    if GENERATE_PARQUET:
        session_role_df.to_parquet(session_role_parquet, index=False)

    # Modeling targets
    model_metrics = {}
    coef_frames = []

    # Individual engagement target, role-specific
    # Build unified per-role target column for modeling
    segments_model = segments_df.copy()
    segments_model["engagement_target"] = np.nan
    for role in ROLES:
        col = f"engagement_{role}_mean"
        idx = segments_model["role"] == role
        if col in segments_model.columns:
            segments_model.loc[idx, "engagement_target"] = segments_model.loc[idx, col]

    m1, c1 = run_models(segments_model, "engagement_target", "individual_engagement")
    model_metrics["individual_engagement"] = m1
    if not c1.empty:
        coef_frames.append(c1)

    # Group task engagement target
    m2, c2 = run_models(segments_df, "task_engagement_mean", "group_task_engagement")
    model_metrics["group_task_engagement"] = m2
    if not c2.empty:
        coef_frames.append(c2)

    # Write full summaries to text for easier inspection.
    summaries_path = OUTPUT_ROOT / "model_summaries.txt"
    with summaries_path.open("w") as fh:
        for k in ["individual_engagement", "group_task_engagement"]:
            fh.write(f"=== {k} ===\n")
            m = model_metrics.get(k, {})
            lin_m = m.get("linear", {})
            log_m = m.get("logistic", {})
            fh.write(
                "Linear pred-vs-true corr: "
                f"pearson={lin_m.get('corr_pred_true_pearson', np.nan):.4f} (p={lin_m.get('corr_pred_true_pearson_p', np.nan):.3g}), "
                f"spearman={lin_m.get('corr_pred_true_spearman', np.nan):.4f} (p={lin_m.get('corr_pred_true_spearman_p', np.nan):.3g})\n"
            )
            fh.write("Linear (statsmodels)\n")
            fh.write(m.get("linear_statsmodels_summary", "n/a"))
            fh.write("\n\n")
            fh.write(
                "Logistic pred-vs-true corr: "
                f"pearson={log_m.get('corr_pred_true_pearson', np.nan):.4f} (p={log_m.get('corr_pred_true_pearson_p', np.nan):.3g}), "
                f"spearman={log_m.get('corr_pred_true_spearman', np.nan):.4f} (p={log_m.get('corr_pred_true_spearman_p', np.nan):.3g})\n"
            )
            fh.write("Logistic (statsmodels)\n")
            fh.write(m.get("logistic_statsmodels_summary", "n/a"))
            fh.write("\n\n")

    # Keep JSON compact and machine-readable without huge summary text blocks.
    json_metrics = {}
    for k, v in model_metrics.items():
        vv = dict(v)
        vv.pop("linear_statsmodels_summary", None)
        vv.pop("logistic_statsmodels_summary", None)
        json_metrics[k] = vv

    metrics_path = OUTPUT_ROOT / "model_metrics.json"
    metrics_path.write_text(json.dumps(json_metrics, indent=2))

    coef_path = OUTPUT_ROOT / "model_coefficients.csv"
    if coef_frames:
        pd.concat(coef_frames, ignore_index=True).to_csv(coef_path, index=False)
    else:
        pd.DataFrame(columns=["analysis", "target", "model", "feature", "coef"]).to_csv(coef_path, index=False)

    print(f"Derived segments: {len(segments_df)}")
    print(f"Derived session-role rows: {len(session_role_df)}")
    if GENERATE_PARQUET:
        print(f"Wrote: {segments_csv}, {segments_parquet}, {session_role_csv}, {session_role_parquet}, {metrics_path}, {coef_path}, {summaries_path}")
    else:
        print(f"Wrote: {segments_csv}, {session_role_csv}, {metrics_path}, {coef_path}, {summaries_path}")


if __name__ == "__main__":
    main()
