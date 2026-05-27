# ViLearn Engagement Model Comparison

**Updated:** 2026-05-27 (run v2 — dyad_02 repaired, GroupKFold LOSO CV + t-tests added)
**Script:** `discover/derive_vilearn_stats.py`
**Sessions:** `discover/vilearn_more.set` (25 sessions: 11 dyads, 14 triads)
**Targets:** `individual_engagement` (per-role), `task_engagement` (group-level)

---

## Feature Sets

**Base features** (hand-crafted, segment-level):
`segment_duration_s`, `word_count`, `avg_word_length`, `words_per_second`, `question`, `statement`, `sentiment_p_{role}_mean`, `arousal_mean`, `dominance_mean`, `valence_mean`, `role`

**Embedding streams** (added on top of base):
- `opensmile_*` — acoustic/prosodic features (openSMILE eGeMAPSv02, 88 dims)
- `emow2v_*` — emotion word2vec embeddings (1024 dims)
- `sentiment_emb_*` — per-role sentence-level sentiment embeddings (1536 dims)

**PCA** (applied per stream block separately, 95% variance, max 128 components):
Reduces opensmile/emow2v/sentiment_emb blocks independently; base features pass through unchanged.

**Granularity:**
- *Segment-level* — one row per speech segment per role (~4274 rows)
- *Frame-level (1 Hz)* — downsampled from raw frames (stride=25), ~46493 rows

**Evaluation:**
- *Holdout*: single 80/20 random split (random_state=42) — rows mixed across sessions
- *LOSO CV*: leave-one-session-out GroupKFold (24 folds) — honest cross-session generalization

---

## Holdout Results (80/20 split — optimistic, sessions not isolated)

### Segment-level

| Configuration | indiv. R² | indiv. r | indiv. AUC | group R² | group r | group AUC |
|---|---:|---:|---:|---:|---:|---:|
| **v1 baseline** (base only, no PCA) | 0.114 | 0.322 | ~0.70 | 0.094 | 0.318 | ~0.68 |
| Base only + PCA | 0.099 | 0.326 | 0.708 | 0.099 | 0.317 | 0.677 |
| Base + streams (no PCA) | — | 0.392 | — | — | 0.285 | — |
| **Base + streams + PCA** | **0.163** | **0.444** | **0.723** | **0.266** | **0.527** | **0.768** |

### Frame-level (1 Hz)

| Configuration | indiv. R² | indiv. r | indiv. AUC | group R² | group r | group AUC |
|---|---:|---:|---:|---:|---:|---:|
| Base only + PCA | 0.175 | 0.419 | 0.706 | 0.087 | 0.295 | 0.632 |
| **Base + streams (no PCA)** | **0.295** | **0.572** | 0.749 | **0.247** | **0.553** | 0.744 |
| Base + streams + PCA | 0.295 | 0.543 | **0.749** | 0.247 | 0.497 | **0.744** |

---

## LOSO CV Results (leave-one-session-out — honest cross-session estimate)

### Key result: Linear R² is negative in all configurations

LOSO R² < 0 means the model predicts engagement **worse than predicting the session mean**. Models learn session-level characteristics, not universal engagement predictors. Regression does not generalize across sessions.

| Configuration | indiv. CV R² | indiv. CV AUC | group CV R² | group CV AUC |
|---|---:|---:|---:|---:|
| Segment base+PCA | -0.531 ± 1.455 | **0.676 ± 0.118** | -0.728 ± 2.628 | 0.620 ± 0.073 |
| Segment streams (no PCA) | -2.124 ± 2.265 | 0.596 ± 0.076 | -2.742 ± 4.559 | 0.620 ± 0.067 |
| Segment streams+PCA | -0.836 ± 1.542 | 0.631 ± 0.084 | -0.633 ± 2.108 | **0.668 ± 0.097** |
| Frame base+PCA | -0.316 ± 1.077 | **0.691 ± 0.133** | -0.842 ± 2.796 | 0.605 ± 0.057 |
| Frame streams (no PCA) | -0.611 ± 1.666 | 0.663 ± 0.115 | -0.681 ± 1.888 | 0.666 ± 0.067 |
| **Frame streams+PCA** | **-0.246 ± 0.869** | **0.701 ± 0.131** | **-0.512 ± 1.842** | **0.685 ± 0.084** |

---

## T-Tests vs Baseline (paired, Bonferroni corrected, α=0.05)

Baseline = DummyRegressor(mean) for R², DummyClassifier(stratified) for AUC.

### Logistic AUC — significant across ALL configurations

| Configuration | target | CV AUC | baseline AUC | p (adj) |
|---|---|---:|---:|---:|
| Segment base+PCA | individual | 0.676 | 0.480 | <0.0001 ✓ |
| Segment base+PCA | group | 0.620 | 0.493 | <0.0001 ✓ |
| Segment streams+PCA | individual | 0.631 | 0.480 | <0.0001 ✓ |
| Segment streams+PCA | group | 0.668 | 0.493 | <0.0001 ✓ |
| Frame base+PCA | individual | 0.691 | 0.498 | <0.0001 ✓ |
| Frame base+PCA | group | 0.605 | 0.499 | <0.0001 ✓ |
| Frame streams (no PCA) | individual | 0.663 | 0.498 | <0.0001 ✓ |
| Frame streams (no PCA) | group | 0.666 | 0.499 | <0.0001 ✓ |
| Frame streams+PCA | individual | 0.701 | 0.498 | <0.0001 ✓ |
| Frame streams+PCA | group | 0.685 | 0.499 | <0.0001 ✓ |

### Linear R² — NOT significant (all ns after Bonferroni)

All linear R² comparisons are non-significant after Bonferroni correction, except:
- Frame base+PCA individual: p_adj=0.0022 ✓ (but R²=-0.316, still negative)

Linear regression in LOSO does not beat predicting the session mean.

---

## Comparison: v1 (old) vs v2 (this run)

Changes from v1:
- `recording_dyad_02` repaired (opensmile + emow2v + arousal/dominance/valence were missing due to corrupt audio)
- N: 4058 → 4274 segments, frame rows ~46K
- LOSO CV and t-tests added

| Metric | v1 (old) | v2 (new) | Δ |
|---|---:|---:|---:|
| Segment streams+PCA indiv R² (holdout) | 0.164 | 0.163 | -0.001 |
| Segment streams+PCA group R² (holdout) | 0.256 | 0.266 | +0.010 |
| Frame streams indiv r (holdout) | 0.564 | 0.572 | +0.008 |
| Frame streams group r (holdout) | 0.557 | 0.553 | -0.004 |

dyad_02 repair had negligible effect on holdout metrics (expected — 1 of 25 sessions).

---

## Key Findings

1. **Holdout R² is misleading** — positive R² (up to 0.29) reflects session-level memorization. LOSO R² is negative for all configs. Models cannot predict engagement for unseen sessions from regression alone.

2. **Classification generalizes** — LOSO AUC 0.60–0.70 significant vs baseline across all configs (Bonferroni p < 0.0001). Above/below median engagement is learnable cross-session.

3. **Best LOSO classifier: frame+streams+PCA** — individual AUC=0.701, group AUC=0.685. Frame granularity + PCA compression gives most stable cross-session AUC.

4. **Streams help group engagement more than individual** — LOSO AUC delta (base→streams+PCA): group +0.048 vs individual -0.045 at segment level. For frame level: group +0.080 vs individual +0.010.

5. **Session variance dominates** — negative LOSO R² + high LOSO R² std (up to ±4.5) indicates engagement level varies more between sessions than within-session predictors explain.

6. **dyad_02 repair confirmed** — streams fully populated post audio re-extraction. Negligible effect on aggregate metrics (1/25 sessions).

---

## Caveats

- LOSO R² std is high (up to ±4.5) — 24 folds with small per-fold test sets (1 session). AUC is more stable.
- Ridge α=10, LogReg L2 — regularized but not tuned. Hyperparameter search per fold would likely improve LOSO performance.
- Logistic baseline uses stratified random (AUC ≈ 0.50); linear baseline uses mean prediction (R² = 0 in-sample, can go negative out-of-sample).
- Statsmodels inference unavailable for stream/PCA models (high-dim). Coefficient significance only available for base-only non-PCA segment model.
- Autocorrelation in frame-level data (frames from same session/role are correlated) inflates holdout metrics further.
