# Presentation scripts (DISCOVER results deck)

Helper scripts for `discover/2026_March_ViLearn_Planning.pptx` and the numbers in it.
Run from the repo root.

## Where the numbers come from
- **Main source**: `discover/derive_vilearn_stats.py` writes all panel CSVs to
  `data/discover/derived/compare/` (`classifier_panel_metrics.csv`,
  `classifier_panel_ttest.csv`, `classifier_panel_foldscores.csv`,
  `classifier_panel_importance.csv`, `comparison_summary.md`, etc.). The slide
  tables are read from these.
- **`compute_extra_panel_numbers.py`**: the few numbers the panel does NOT emit —
  1 Hz frame accuracy (frame-1Hz is excluded from the panel: SVM-rbf intractable
  on ~27k rows) and the 60 s base-vs-embeddings comparison. Reuses
  `_prepare_xy` / `transform_with_block_pca` from the main script.

## Slide builders (ONE-SHOT, order matters — NOT idempotent)
Applied once to produce the current deck. Re-running on the already-edited deck
duplicates content; only re-run in order on a fresh deck.
1. `01_inject_results_slides.py` — appends the 6 results slides (backs up to `*_backup.pptx`).
2. `02_update_slide11_significance.py` — rebuilds slide 11 with the significance table.
3. `03_edit_tables_and_rq.py` — adds 1 Hz row + feature/embedding tables, reframes the RQ slide.

Note: python-pptx `slide.shapes.title` returns a fresh wrapper each call, so
identity checks (`sh is title`) don't work — match by `placeholder_format.type`
or element. Some result slides' title+body both report placeholder idx 0; target
the body by `type != TITLE`.
