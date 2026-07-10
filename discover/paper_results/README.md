# Paper results — canonical CSVs (IUI revision + journal)

Tracked snapshots of the analysis outputs the paper tables/slides are built
from. Source of truth: `scratch/te_adoption/a1_out/` (gitignored workbench);
generator scripts in `scratch/te_adoption/` (`c2_paper_tables.py` driver +
worker, `d1`–`d4`). Protocol everywhere: 185 floorlevel 60 s group-windows,
clean 2-annotator-mean TE binarized at 0.5, nested LOGO model selection
(seed 42, deterministic), permutation test (300 perms) + one-sample t-test vs
majority floor, Cohen's d vs majority floor.

Snapshot date: 2026-07-10.

## IUI (pre-LLM) scope

- `c2_paper_table.csv` — headline cells: top-2 models per (split × feature set),
  acc±SD, majority floor, perm_p, t, p_ttest, Cohen's d. Feeds deck slides
  12–14 (modality ablation), 18/20 (detector vs priors), 22 (extended sets).
- `c2_foldscores.csv` — per-fold accuracies for ALL panel models per cell; any
  statistic above is recomputable from this.
- `c2_feature_lists.md` — exact feature lists per set (paper appendix).
- `c2_selection.csv` — per-fold SelectKBest picks for the `_sel` robustness
  cells (split, detector, model, test_group, k, features '|'-joined).
  openSMILE_sel uses a log-spaced k grid (1,2,3,4,6,8,11,16,22,32,45,64,88);
  low-dim sets use the full 1..n range.
- `d1_linguistic_dropone.csv` — linguistic drop-one ablation + coefficients
  (deck slide "What Drives the Linguistic Detector?").
- `d2_pca_dims.csv` — which PCA n_components {5,10,20} the inner CV picked per
  fold for openSMILE_pca (Overall: 5 in 20/20 folds = grid floor).
- `d4_opensmile_ranking.csv` — descriptive full-data f_classif ranking of the
  88 eGeMAPS features per split (variability functionals dominate).
- `comprehensive_table.csv` — earlier 3-priors + fusion table (same protocol);
  superseded by c2_paper_table.csv where they overlap, kept for cross-checking.
- `c2_features.csv` — the WINDOWED FEATURE MATRIX itself (185 windows × 123
  cols): prior gaze features on the original floorlevel grid, prior TE +
  clean 2-annotator TE (`our_te`) + binarized target `y`, audio v/a/d, LLM
  rubric scores, linguistic stats, 88 eGeMAPS. `group_name` = fold key for
  leave-one-group-out. Training is directly reproducible from this file alone
  (no parquets needed).
- `c2_emb_features.csv` — embedding features on the same grid (emow2v 1024 +
  sentiment-embedding 512, window-aggregated) for the embedding cells.

## Journal (LLM-era) scope

- `b1_fusion_priors_llm.csv` — gaze/audio × LLM-marker fusions per split.
- `b2_rate_split.csv` — 60 Hz vs 90 Hz annotation-rate robustness (confounded
  with group type; appendix).

- `idea3_window_scores_think.csv` / `idea3_window_scores_nothink.csv` — LLM
  rubric scores per window (numeric only, no transcript text); inputs to the
  LLM columns in c2_features.csv and to b1. `idea3_results.csv` — LLM/BERT
  detector evaluation summary.

## Regeneration

Scripts live in `discover/analysis_scripts/` (see its README). From repo root
(discover conda env):

    python3 discover/analysis_scripts/c2_paper_tables.py            # full driver, parallel
    PYTHONPATH=discover/analysis_scripts python3 discover/analysis_scripts/c2_paper_tables.py <split> <set>   # one cell
    python3 discover/analysis_scripts/d3_restore_extras.py          # extra cells not in the driver SETS
    PYTHONPATH=discover/analysis_scripts python3 discover/analysis_scripts/d2_pca_dims.py
    PYTHONPATH=discover/analysis_scripts python3 discover/analysis_scripts/d4_opensmile_ranking.py

outputs land in `scratch/te_adoption/a1_out/` (gitignored workbench,
auto-created); copy the refreshed files here and commit.
