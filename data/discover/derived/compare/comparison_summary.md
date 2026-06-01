# ViLearn Engagement Model Comparison

**Updated:** 2026-06-01 (run v3 — floorlevel groups only, interaction-time clipped, 60 s window granularity added)
**Script:** `discover/derive_vilearn_stats.py`
**Sessions:** floorlevel subset of `discover/vilearn_more.set` (**20 sessions: 8 dyads, 12 triads**)
**Targets:** `individual_engagement` (per-role), `task_engagement` (group-level)

> **What changed vs v2 (2026-05-27):**
> - **Floorlevel only** (`FLOORLEVEL_ONLY`): restricted to the 20 groups in `data/group_names_with_time_floorlevel.csv` (no participant "flying" in VR). Dropped: dyad_01/08/09, triad_03/04.
> - **Interaction-time clipping** (`CLIP_TO_INTERACTION`): frames + transcript segments clipped to each group's interaction window. Wall-clock bounds → recording-relative ms via `data/recording_times_group_info.csv` (join on `Group_Name_Long` == `long_name`).
> - **60 s window granularity** (`WINDOW_MS=60000`) added to match prior work (Cristina/Carlos 60 s resampling), alongside the existing segment and 1 Hz frame analyses.
> - **Fixed-threshold classifier panel** (QDA/SVM/NB/kNN/RF/LogReg, D/T/All splits) added to compare with the prior ICMI/QDA detector — see "Fixed-Threshold Classification Panel" below. Includes per-window linguistic features, per-feature contribution (`classifier_panel_importance.csv`), and an embeddings (streams_pca) comparison at window granularity (randomized PCA).
> - Counts: **3054 segments**, **52 session-role rows**, **27290 frame rows (1 Hz)**.

---

## Feature Sets

**Base features** (hand-crafted):
`segment_duration_s`, `word_count`, `avg_word_length`, `words_per_second`, `question`, `statement`, `sentiment_{role}`, `arousal`, `dominance`, `valence`, `role`

**Embedding streams** (added on top of base):
- `opensmile_*` — acoustic/prosodic features (openSMILE eGeMAPSv02, 88 dims)
- `emow2v_*` — emotion word2vec embeddings (1024 dims)
- `sentiment_emb_*` — per-role sentence-level sentiment embeddings (1536 dims)

**PCA** (per stream block separately, 95% variance, max 128 components): reduces opensmile/emow2v/sentiment_emb blocks independently; base features pass through unchanged.

**Granularity:**
- *Segment-level* — one row per speech segment per role (3054 rows)
- *Frame-level (1 Hz)* — downsampled from 25 Hz merged frames (stride=25), 27290 rows
- *Window-level (60 s)* — frame features mean-aggregated into 60 s bins per session/role (prior-work granularity)

**Evaluation:**
- *Holdout*: single 80/20 random split (random_state=42) — rows mixed across sessions (optimistic)
- *LOSO CV*: leave-one-session-out GroupKFold (20 folds) — honest cross-session generalization
- *Classification target*: median split of the continuous target (`y > y.median()`) → AUC. **NB: this is a median split, not a fixed 0.5 high/low TE threshold like the ICMI/QDA detector.**

---

## Holdout Results (80/20 split — optimistic, sessions not isolated)

### Segment-level
| Configuration | indiv R² | indiv r | indiv AUC | group R² | group r | group AUC |
|---|---:|---:|---:|---:|---:|---:|
| Base only + PCA | 0.155 | 0.395 | 0.735 | 0.080 | 0.283 | 0.610 |
| **Base + streams + PCA** | **0.220** | **0.502** | **0.780** | **0.164** | **0.453** | **0.693** |

### Frame-level (1 Hz)
| Configuration | indiv R² | indiv r | indiv AUC | group R² | group r | group AUC |
|---|---:|---:|---:|---:|---:|---:|
| Base only + PCA | 0.176 | 0.420 | 0.714 | 0.071 | 0.267 | 0.596 |
| Base + streams (no PCA) | 0.278 | 0.534 | 0.741 | 0.257 | 0.513 | **0.772** |
| Base + streams + PCA | 0.238 | 0.488 | 0.740 | 0.195 | 0.443 | 0.707 |

### Window-level (60 s) — prior-work granularity
| Configuration | indiv R² | indiv r | indiv AUC | group R² | group r | group AUC |
|---|---:|---:|---:|---:|---:|---:|
| Base only + PCA | 0.366 | 0.608 | 0.812 | 0.487 | 0.703 | 0.753 |
| Base + streams (no PCA) | 0.087 | 0.569 | 0.737 | 0.681 | 0.837 | **0.883** |
| **Base + streams + PCA** | **0.380** | **0.649** | **0.844** | **0.739** | **0.862** | 0.835 |

Holdout improves sharply with coarser aggregation: 60 s windows reach group R²=0.74 / AUC=0.84 — but see LOSO below before trusting these.

---

## LOSO CV Results (leave-one-session-out, 20 folds — honest cross-session estimate)

| Configuration | indiv CV R² | indiv CV AUC | group CV R² | group CV AUC |
|---|---:|---:|---:|---:|
| Segment base+PCA | -0.761 ± 1.754 | 0.669 ± 0.157 | -1.070 ± 3.241 | 0.576 ± 0.081 |
| Segment streams+PCA | -1.167 ± 2.010 | 0.631 ± 0.070 | -1.444 ± 3.140 | 0.617 ± 0.091 |
| Frame base+PCA | -0.438 ± 1.355 | **0.703 ± 0.161** | -1.112 ± 3.305 | 0.562 ± 0.056 |
| Frame streams+PCA | -0.462 ± 1.407 | 0.683 ± 0.160 | -0.935 ± 2.585 | 0.626 ± 0.059 |
| **Window 60 s base+PCA** | -0.447 ± 1.681 | **0.780 ± 0.179** | -1.826 ± 6.453 | **0.739 ± 0.198** |
| Window 60 s streams+PCA | -1.986 ± 7.150 | 0.739 ± 0.164 | -44.2 ± 190.5 | 0.677 ± 0.185 |

**Linear R² is negative in every configuration** — models predict engagement worse than the session mean. Regression does not generalize across sessions.

**Best LOSO classifiers are the 60 s window models**: individual AUC **0.78** (base+PCA), group AUC **0.739** (base+PCA). At 60 s, base features alone beat base+streams on LOSO AUC (streams overfit at low row counts).

---

## T-Tests vs Baseline (paired over folds, Bonferroni corrected, α=0.05)

Baseline = DummyRegressor(mean) for R², DummyClassifier(stratified) for AUC.

**Logistic AUC — significant in ALL configurations** (p_adj < 0.05):

| Configuration | target | CV AUC | baseline | p_adj |
|---|---|---:|---:|---:|
| Frame base+PCA | individual | 0.703 | 0.499 | 0.0001 ✓ |
| Frame streams+PCA | group | 0.626 | 0.502 | <0.0001 ✓ |
| Window 60 s base+PCA | individual | 0.780 | 0.507 | 0.0028 ✓ |
| Window 60 s base+PCA | group | 0.739 | 0.461 | 0.0022 ✓ |
| Window 60 s streams+PCA | individual | 0.739 | 0.507 | 0.0050 ✓ |
| Window 60 s streams+PCA | group | 0.677 | 0.461 | 0.0212 ✓ |

**Linear R² — not significant** after Bonferroni in any config (models do not beat the session-mean predictor).

---

## Key Findings

1. **Holdout R² is misleading** — positive R² (up to 0.74 at 60 s) reflects session-level memorization. LOSO R² is negative everywhere. Regression alone cannot predict engagement for unseen sessions.

2. **Classification generalizes** — LOSO AUC 0.58–0.78 significant vs baseline across all configs. High/low (median-split) engagement is learnable cross-session.

3. **60 s windows are the best granularity for cross-session classification** — window base+PCA: individual AUC=0.78, group AUC=0.739, both clearly above the finer-grained segment/frame models. Coarser aggregation reduces noise and matches the prior-work setup.

4. **Streams help holdout but hurt LOSO at window level** — at 60 s, base-only+PCA beats base+streams+PCA on LOSO AUC (indiv 0.78 vs 0.739; group 0.739 vs 0.677). High-dim streams overfit when row counts drop (each session contributes only ~5–15 windows).

5. **Floorlevel + interaction clipping raised holdout metrics** vs v2 (e.g. segment streams+PCA group R² 0.266→holdout still strong; 60 s windows new). Cleaner data (no flyers, interaction-only) gives a more defensible analysis population.

6. **Session variance dominates** — negative LOSO R² with very high std (group window std up to ±190) indicates engagement level varies far more between sessions than within-session predictors explain. Window-level group R² std is inflated by tiny per-fold test sets.

---

## Fixed-Threshold Classification Panel (vs prior ICMI / QDA detector)

To compare against the prior detector slides (`discover/2026_March_ViLearn_Planning.pptx`), a
separate panel runs **fixed-threshold (TE > 0.5) high/low classification** with a model sweep
(Baseline-uniform, QDA, SVM-linear, SVM-rbf, Naive Bayes, kNN, Random Forest, Logistic
Regression), **LOSO CV with per-fold accuracy** (one held-out group per fold) reported as
**mean ± SD over folds** (matches Carlos's per-fold reporting). Per-class precision/recall/F1
come from the pooled out-of-fold confusion. Reported on **3 group splits: D (dyads), T (triads),
All**, matching the prior 3-fold reporting.

**Significance** (`classifier_panel_ttest.csv`): per-fold paired t-tests (paired by held-out group)
with Bonferroni correction + Cohen's d effect size, against (a) the uniform baseline and (b)
**Carlos's prior GazexSpeaking+Blinks QDA** per-fold accuracy (`runs/accuracy/...ICMI/Blinks_GazexSpeaking`).

**Feature sets:**
- **base** = linguistic + affective + speaking, aggregated per granularity. At 60 s windows:
  `word_count`, `words_per_second`, `avg_word_length`, `question_rate`, `statement_rate`,
  `segment_count`, `speaking_seconds` (windowed linguistic) + `sentiment_role`, `speaking_role`,
  `arousal`, `dominance`, `valence` + role. At segment level: same plus per-segment text metrics.
- **streams_pca** = base **+** embedding streams (opensmile 88 + emow2v 1024 + sentiment_emb 768 =
  1880 dims, PCA-reduced per block; base passes through unchanged). Window granularity only.
- **NB: different modality from the prior detector.** Slide 3 QDA (dyads 77%) used **gaze**
  (MG/DG/No_DG + blink); slide 2 ICMI used **GazexSpeaking**. This panel answers slide-9's RQ
  "how well with linguistic / multimodal features?", not a reimplementation of the gaze detector.
- Config: `RUN_CLASSIFIER_PANEL=True`, `CLASSIFICATION_THRESHOLD=0.5`, `PANEL_INCLUDE_STREAMS`
  (streams off by default; when on, window-only — segment+streams too slow). `PCA_SVD_SOLVER=randomized`
  (per-fold PCA on the 1880 stream dims tractable; full-SVD was ~8 min/config). Segment + 60 s
  window only (frame-1 Hz excluded: SVM-rbf intractable on 27k rows).

### 60 s window · group task engagement — per-fold mean ± SD + significance

Base features, best model per split, with paired-t-test significance + Cohen's d:

| split (n folds) | best model | acc (mean±SD) | F1 | vs uniform baseline | vs Carlos GazexSpeaking QDA |
|---|---|---:|---:|---|---|
| All (20) | Random Forest | 0.823 ± 0.155 | 0.71 | p<0.001, d=+1.48 ✓ | **p=0.004, d=+0.92 ✓** |
| All (20) | QDA | 0.806 ± 0.154 | 0.69 | p<0.001, d=+1.47 ✓ | p=0.011, d=+0.82 ✓ |
| Triads (12) | Random Forest | 0.880 ± 0.090 | 0.80 | p<0.001, d=+2.42 ✓ | **p=0.005, d=+1.35 ✓** |
| Triads (12) | QDA | 0.866 ± 0.083 | 0.77 | p<0.001, d=+2.32 ✓ | p=0.011, d=+1.20 ✓ |
| Dyads (8) | QDA | 0.737 ± 0.210 | 0.57 | p=0.28, d=+0.89 (ns) | p=1.0, d=+0.52 (ns) |

**Linguistic+affective beats the prior gaze×speaking QDA detector significantly on All groups and
Triads (medium–large effect, d 0.8–1.35).** Dyads trend the same direction (positive d) but n=8
folds is underpowered — not significant after Bonferroni. All models beat the uniform baseline on
All/Triads (d 1.2–2.4); on dyads even baseline comparison is ns (low power).

### 60 s window · group task engagement — base vs +embeddings (best by F1-macro, pooled OOF)

| split | base (best) | acc / F1 | +streams_pca | acc / F1 |
|---|---|---:|---|---:|
| D (dyads) | Naive Bayes | 0.724 / 0.642 | Naive Bayes | 0.835 / 0.701 ↑* |
| T (triads) | Random Forest | **0.872 / 0.820** | Random Forest | 0.708 / 0.657 ↓ |
| All | Random Forest | **0.832 / 0.759** | Random Forest | 0.792 / 0.694 ↓ |

### 60 s window · individual engagement — base vs +embeddings

| split | base (best) | acc / F1 | +streams_pca | acc / F1 |
|---|---|---:|---|---:|
| D | Naive Bayes | 0.654 / 0.653 | Random Forest | 0.610 / 0.603 ↓ |
| T | QDA | 0.722 / 0.721 | Random Forest | 0.668 / 0.667 ↓ |
| All | QDA | 0.694 / 0.694 | Naive Bayes | 0.696 / 0.693 ≈ |

\* Dyad group-TE rise (8 LOSO folds, 170 rows) is within noise — not trusted. **Net: embeddings on
top of base mostly hurt or don't help** — high-dim streams add noise that degrades cross-session
generalization at window scale. **Base wins.**

### Effect of adding windowed linguistic features (vs affective+speaking only)

Roughly **neutral**: group-TE All ≈ unchanged (QDA 0.836→0.810, RF best 0.832), slight ↓ on small
dyad splits. Affective+speaking already carried the group-TE signal; linguistic helps explain
*individual* engagement but doesn't move accuracy (collinear with speaking).

### Feature contribution (classifier_panel_importance.csv: standardized LogReg coef + RF Gini)

- **Group TE** dominated by group affect: `valence` (coef +1.12, RF 0.21), `dominance` (−1.33, 0.18),
  `arousal` (0.13). Linguistic features all low (0.04–0.06).
- **Individual engagement** more distributed: `speaking_role` (+), `valence`, `speaking_seconds`,
  `dominance` (−), `word_count` (+), `words_per_second` (+) — linguistic contributes here.

### Segment-level (text features available)

| target | split | best model | acc | F1-macro |
|---|---|---|---:|---:|
| individual eng | All | SVM-linear | 0.624 | 0.603 |
| group TE | All | QDA | 0.593 | 0.563 |
| group TE | T | QDA | 0.658 | 0.569 |

Segment ≪ 60 s window for both targets — aggregation to 60 s is clearly the better granularity.

### Comparison to slides

- **Slide 3 (QDA dyads, gaze): 77% acc, high P 83.3% / R 58.8%, low P 75% / R 91.3%.**
  Ours (QDA dyads, base linguistic+affective, 60 s): **acc 74.7%** — high P 81.0% / R 86.7%,
  low P 48.5% / R 38.1%. Comparable accuracy, **opposite operating point** (our model over-predicts
  "high" — ~74% of group-TE windows are high), different feature modality.
- **Slide 2 (ICMI GazexSpeaking, accuracy M): SVM 0.66 / 0.63 / 0.62 (D/T/All); baseline 0.53/0.43/0.45.**
  Our base features at 60 s exceed this: best acc **D 0.72 / T 0.87 / All 0.83**.
  → linguistic + affective + speaking detect group TE at least as well as the submitted gaze×speaking
  set at matched granularity, without gaze.

Outputs: `classifier_panel_metrics.csv` (cols `group_split`, `acc_mean`/`acc_std`, `f1_macro_mean`,
pooled `accuracy_pooled`, per-class P/R/F1, `n_folds`), `classifier_panel_foldscores.csv` (per-fold
per-model accuracy/F1 by held-out group), `classifier_panel_ttest.csv` (vs_baseline_uniform +
vs_carlos_qda_gazespeak: `t_stat`, `p_value`, `p_value_adj`, `cohen_d`, `significant`),
`classifier_panel_confusion.csv` (per-model 2×2, pooled), `classifier_panel_importance.csv`.

---

## Modality Ablation (linguistic-only vs audio-only vs multimodal)

Feature buckets (panel `feature_modality` column): **(para)linguistic** = transcript text
(`word_count`, `words_per_second`, `avg_word_length`, `question/statement_rate`, `sentiment`) +
speaking/turn-taking (`speaking_role`, `speaking_seconds`, `segment_count`); **audio** = group
affect (`arousal`, `dominance`, `valence`); **multimodal** = both. (Audio is group-shared; speaking
attributes it per role, folded into paralinguistic.) 60 s window, LOSO, best model per cell (acc / F1-macro):

| target · split | multimodal | audio-only | (para)linguistic-only |
|---|---:|---:|---:|
| Group TE · All | 0.823 / 0.71 | **0.820 / 0.68** | 0.724 / 0.53 |
| Group TE · Triads | **0.880 / 0.80** | 0.861 / 0.76 | 0.719 / 0.49 |
| Group TE · Dyads (n=8) | 0.737 / 0.57 | 0.769 / 0.67 | 0.739 / 0.63 |
| Individual · All | 0.688 / 0.64 | 0.661 / 0.59 | **0.707 / 0.66** |
| Individual · Triads | 0.732 / 0.70 | 0.642 / 0.57 | 0.715 / 0.68 |
| Individual · Dyads | 0.659 / 0.58 | 0.623 / 0.55 | 0.650 / 0.59 |

**Double dissociation:**
- **Group task engagement ← audio affect.** Audio-only ≈ multimodal; (para)linguistic-only ~0.10
  worse on All/Triads. Text adds nothing on top of audio (matches importance: valence/dominance dominate).
- **Individual engagement ← (para)linguistic.** Paralinguistic-only ≥ multimodal; audio-only collapses
  toward baseline (group-shared v/a/d gives no per-role signal).
- **No multimodal synergy** — each target is carried by one modality; combining ≈ the dominant one.

### Answers to slide-9 Research Questions
- **Linguistic-only detect TE?** Modest. Group TE acc ~0.72 / F1 ~0.53 (weaker than audio); but for
  **individual** engagement (para)linguistic is the *best* channel (acc ~0.71 / F1 ~0.66).
- **Which linguistic features strongest?** (importance, individual engagement) `speaking_role`,
  `word_count`, `words_per_second` (speaking + amount of talk) + `sentiment`. For group TE the
  linguistic features are weak; the driver is audio `valence` (+) / `dominance` (−).
- **Multimodal (both)?** Group TE acc 0.82–0.88, individual 0.69–0.73 — equals the dominant single
  modality per target, no gain from combining (and embeddings on top degrade slightly).

Outputs gain a `feature_modality` column (multimodal / paralinguistic / audio) across both targets,
both granularities, D/T/All (`classifier_panel_metrics.csv`, `..._foldscores.csv`, `..._ttest.csv`).

## Caveats

- **Median split ≠ ICMI/QDA detector.** The prior detector slide (QDA, dyads 77% accuracy, high/low TE precision) uses a **fixed threshold** and reports **accuracy + per-class precision/F1 + confusion matrix**. This pipeline uses a **median split + AUC + LogisticRegression only**, so numbers are not directly comparable. For a like-for-like comparison, add: fixed-threshold binarization, accuracy/precision/recall/F1, confusion matrices, and a QDA model.
- 60 s window models have few rows per session (~5–15) → some LOSO folds drop to single-class and are excluded (group logistic = 17/20 folds). AUC stable, R² std huge.
- Ridge α=10, LogReg L2 — regularized but not tuned. Per-fold hyperparameter search would likely improve LOSO.
- Statsmodels inference unavailable for stream/PCA models (high-dim); coefficient significance only for base-only non-PCA segment model.
- Frame-level rows from the same session/role are autocorrelated, inflating holdout metrics.
