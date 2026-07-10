# Analysis scripts — canonical TE-detection pipeline (IUI revision + journal)

Everything that windows features onto the 185-window floorlevel grid, builds
the paper tables, and reproduces the label-bug findings. Moved here from the
gitignored scratch workbench 2026-07-10 so the pipeline is in the repo.

**Conventions** (all scripts):
- run from REPO ROOT, e.g. `PYTHONPATH=discover/analysis_scripts python3 discover/analysis_scripts/<script>.py`
- intermediate outputs go to `scratch/te_adoption/a1_out/` (gitignored
  workbench, auto-created); final CSVs are copied to `discover/paper_results/`
  and committed there
- everything is deterministic (seed 42) — reruns reproduce committed numbers
- protocol: 185 floorlevel 60 s group-windows, clean 2-annotator-mean TE
  binarized at 0.5, nested leave-one-group-out model selection, permutation
  test (300) + one-sample t-test vs majority floor

## Core

- `ab_common.py` — shared: `build_features()` (windowing onto the grid, prior
  gaze / audio / LLM feature loading), model panel + grids, nested CV, perm_p.
- `a1_lib.py` — grid/label plumbing used by ab_common and idea3.
- `c2_paper_tables.py` — THE table generator. Driver (no args, parallel
  12-worker pool) rebuilds all cells; worker (`<split> <set>`) one cell;
  `featurelists` regenerates the appendix doc. Windowing for openSMILE +
  linguistic lives in `build_all()`, embeddings in `build_emb()`.
  NB: module-level import is safe (`__main__`-guarded); thread-parallelism
  inside one process segfaults (libgomp) — parallelism is subprocess-level only.
- `d3_restore_extras.py` — reruns the detectors not in the driver SETS
  (linguistic+AIxVR, linguistic_core, `_sel` variants).

## Experiments

- `d1_linguistic_dropone.py` — linguistic drop-one + coefficients.
- `d2_pca_dims.py` — which PCA n_components the inner CV picks (openSMILE).
- `d4_opensmile_ranking.py` — univariate f_classif ranking of the 88 eGeMAPS.
- `b1_fusion_priors_llm.py`, `b2_rate_split.py`, `c_add_ttest.py` — LLM-fusion
  and annotation-rate cells (journal scope).
- `comprehensive_comparison.py`, `add_ttest.py` — earlier 3-priors+fusion
  table (superseded by c2 where they overlap).

## Label-bug reproduction (see discover/TE_LABEL_DISCREPANCIES.md)

- `verify_stretch_bug.py` — simulates the 60→90 Hz time-stretch and matches it
  against the saved averages.
- `sanity_check_clean_mean.py` — clean 2-annotator mean vs prior TE file.
- `validate_origin_alignment.py` — window-origin alignment check (shift k=0).

## idea3/ — LLM rubric pipeline (journal)

`idea3_build_text.py` (window transcripts) → `idea3_score.py`
(MODE=think|nothink, local ollama qwen) → `idea3_bert.py` (gbert embeddings)
→ `idea3_eval.py`. Text/embedding intermediates stay in `scratch/ideas/`
(transcript content is not committed); the numeric window scores are tracked
at `discover/paper_results/idea3_window_scores_{think,nothink}.csv`.
