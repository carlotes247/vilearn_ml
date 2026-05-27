# ViLearn Engagement Model Comparison

**Date:** 2026-05-27
**Script:** `discover/derive_vilearn_stats.py`
**Sessions:** `discover/vilearn_more.set`
**Targets:** `individual_engagement` (per-role), `task_engagement` (group-level)

---

## Feature Sets

**Base features** (hand-crafted, segment-level):
`segment_duration_s`, `word_count`, `avg_word_length`, `words_per_second`, `question`, `statement`, `sentiment_p_{role}_mean`, `arousal_mean`, `dominance_mean`, `valence_mean`, `role`

**Embedding streams** (added on top of base):
- `opensmile_*` — acoustic/prosodic features (openSMILE)
- `emow2v_*` — emotion word2vec embeddings
- `sentiment_emb_*` — per-role sentence-level sentiment embeddings

**PCA** (applied per stream block separately, 95% variance, max 128 components):
Reduces opensmile/emow2v/sentiment_emb blocks independently before concatenating with base features. Base features pass through unchanged.

**Granularity:**
- *Segment-level* — one row per speech segment per role
- *Frame-level (1 Hz)* — downsampled from raw frames (stride=25), one row per second per role

---

## Results

### Segment-level

| Configuration | indiv. linear R² | indiv. linear r | indiv. AUC | indiv. logistic r | group linear R² | group linear r | group AUC | group logistic r |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Baseline (yesterday)** — base only | 0.114 | 0.322 | ~0.70 | 0.356 | 0.094 | 0.318 | ~0.68 | 0.304 |
| Base only + PCA | 0.099 | 0.326 | 0.705 | 0.356 | 0.100 | 0.318 | 0.678 | 0.304 |
| Base + streams (no PCA)* | — | 0.413 | — | 0.297 | — | 0.391 | — | 0.344 |
| **Base + streams + PCA** | **0.164** | **0.435** | **0.716** | **0.374** | **0.256** | **0.517** | **0.764** | **0.458** |

*3430 predictors without PCA — R²/AUC not available (sklearn skips statsmodels at this dim); r from pred-vs-true pearsonr.

Delta (base+PCA → streams+PCA):

| target | Δ linear R² | Δ linear r | Δ logistic AUC | Δ logistic r |
|---|---:|---:|---:|---:|
| individual_engagement | +0.065 | +0.109 | +0.012 | +0.018 |
| group_task_engagement | +0.156 | +0.199 | +0.086 | +0.154 |

### Frame-level (1 Hz)

| Configuration | indiv. linear R² | indiv. linear r | indiv. AUC | indiv. logistic r | group linear R² | group linear r | group AUC | group logistic r |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base only + PCA | 0.162 | 0.402 | 0.699 | 0.352 | 0.083 | 0.289 | 0.633 | 0.227 |
| Base + streams (no PCA) | 0.286 | **0.564** | 0.747 | **0.464** | 0.245 | **0.557** | 0.745 | **0.503** |
| **Base + streams + PCA** | **0.286** | 0.536 | **0.747** | 0.433 | **0.245** | 0.496 | **0.745** | 0.426 |

Delta (base+PCA → streams+PCA):

| target | Δ linear R² | Δ linear r | Δ logistic AUC | Δ logistic r |
|---|---:|---:|---:|---:|
| individual_engagement | +0.125 | +0.134 | +0.049 | +0.081 |
| group_task_engagement | +0.162 | +0.207 | +0.111 | +0.199 |

---

## Key Findings

1. **Embedding streams drive most of the gain** — especially for group task engagement (segment R² nearly triples: 0.10 → 0.26).

2. **Frame-level granularity helps further** — individual engagement linear r: 0.32 → 0.56 (frame + streams, no PCA).

3. **PCA hurts at frame level** — raw 1887-dim streams generalize better than compressed version at 1 Hz. Likely because frame-level has much larger N (hundreds of thousands of rows), so high-dim regularized models don't overfit.

4. **PCA helps at segment level** — segment N is small (~4000 rows), 3430 predictors → Ridge/LogReg still overfit without compression.

5. **Group engagement easier to model than individual** — streams+PCA segment: group R²=0.256 vs. individual R²=0.164. Likely because task_engagement aggregates across roles (less noisy label).

6. **Logistic (AUC) lags linear (R²) gains** — median-split binary classification is noisier than regression, especially for individual engagement.

---

## Caveats

- Models use Ridge (α=10) and L2 LogReg — regularized but not tuned. R²/AUC are test-set (20% holdout, random_state=42), not cross-validated.
- Durbin-Watson ~0.4–0.6 in baseline OLS → strong autocorrelation (segments from same session are correlated). R² inflated relative to independent samples.
- Frame-level p=0 entries are floating-point underflow, not literal zero.
- Statsmodels inference skipped for stream runs (high dim / PCA mode). Coefficient-level significance unavailable for stream models.
