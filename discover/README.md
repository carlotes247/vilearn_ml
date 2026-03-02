# ViLearn DISCOVER Job Submission

This folder contains the batch submission script and supporting files for running extraction jobs on the ViLearn dataset in DISCOVER.

## Files

- `batch_submit_jobs.py` — submits extraction jobs to DISCOVER/NOVA.
- `export_vilearn_annotations.py` — exports annotations from NOVA into `data/discover/<session>/`.
- `merge_vilearn_features.py` — merges annotations into one CSV per session in `data/discover/merged/`.
- `derive_vilearn_stats.py` — derives segment/session-role statistics and fits linear/logistic models for engagement targets.
- `opensmile_egemapsv02_functionals_dims.txt` — openSMILE eGeMAPSv02 functional dimension names (88 dims).
- `vilearn_more.set` — list of session names (one per line). Source of truth for what gets processed.
- `.env.example` — template for DB connection credentials.
- `.env` — your local credentials (not committed).
- `requirements.txt` — minimal Python requirements for the submission script.

## Prerequisites

- Python 3.9+
- `pip install -r discover/requirements.txt`
- A running DISCOVER/NOVA endpoint reachable at `URL` in `batch_submit_jobs.py`.
- Valid DB credentials in `discover/.env`.

## Setup

1. Copy the env template and fill in your credentials:

```bash
cp discover/.env.example discover/.env
```

2. Verify the dataset config in `discover/batch_submit_jobs.py`:

- `DATASET` should match the NOVA dataset name.
- `DATASET_PATH` should point to the dataset root.
- `SET_FILES` should list one or more `.set` files.

3. Ensure `discover/vilearn_more.set` contains **one session name per line**. This is the only source of sessions to process.

## Roles

Sessions include participant roles by naming convention:

- Dyads: `p_blue`, `p_green`
- Triads: `p_blue`, `p_green`, `p_red`
- Group-level streams: `group`

The script submits jobs per role. If a session name contains `triad`, the `p_red` role is included for participant jobs. Otherwise it is skipped.

## Jobs

Jobs are defined in `JOBS` in `batch_submit_jobs.py`. Each job specifies:

- `trainerFilePath`
- input/output `data` JSON (supports `{role}`, `{CG}`, `{HR}` placeholders)
- `options`

Current jobs include:

- **opensmile** on group audio
- **sentiment** on participant transcripts
- **emow2v** on group audio

## Submitting Jobs

Run:

```bash
python discover/batch_submit_jobs.py
```

Job IDs are generated as:

- `vl_{trainer}_{role}` for non‑chunked jobs
- `vl_{trainer}_{role}_{k:02}` for chunked jobs

## Export + Merge

1. Export annotations from NOVA:

```bash
python discover/export_vilearn_annotations.py
```

2. Merge into one CSV per session:

```bash
python discover/merge_vilearn_features.py
```

`merge_vilearn_features.py` also supports optional Parquet generation:

- `GENERATE_PARQUET = False` (default): only CSV is written
- `GENERATE_PARQUET = True`: Parquet includes stream features
  - labeled openSMILE columns (`opensmile_<feature_name>`)
  - synthetic emow2v columns (`emow2v_0000...`)
  - synthetic sentiment embedding columns (`sentiment_emb_<role>_0000...`)

For quick tests, `TEST_SESSIONS` can be set to a small subset. Set `TEST_SESSIONS = []` for full runs.

3. Derive higher-level stats and run baseline explanatory models:

```bash
python discover/derive_vilearn_stats.py
```

Outputs in `data/discover/derived/`:

- `segments.csv` (and optional `segments.parquet` when enabled in script):
  - one row per transcript segment (`from,to` right-open interval)
  - text metrics: `word_count`, `avg_word_length`, `words_per_second`, `question`, `statement`
  - feature window stats per segment: mean + std for sentiment/engagement/task_engagement/arousal/dominance/valence
- `session_role_stats.csv` (and optional `session_role_stats.parquet` when enabled in script): aggregated per `(session, role)`
- `model_metrics.json`: metrics for
  - linear regression and logistic regression on individual engagement
  - linear regression and logistic regression on group task engagement
- `model_coefficients.csv`: feature coefficients for all fitted models

## Notes

- The `.set` file is authoritative; folder discovery is disabled.
- Update annotator names via `CG` and `HR` at the top of the script.
- openSMILE may emit `Segment too short, filling with NaN` warnings (make window bigger which includes possible tradeoffs). This can introduce NaNs in the output stream. Downstream processing should account for missing values (e.g., imputation or filtering).
- `recording_dyad_02` has a corrupt `group.audio.wav` (malformed WAV `fmt` chunk). Expect missing arousal/dominance/valence annotations for that session until the audio is re-exported or fixed.
