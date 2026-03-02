# ViLearn DISCOVER Job Submission

This folder contains the batch submission script and supporting files for running extraction jobs on the ViLearn dataset in DISCOVER.

## Files

- `batch_submit_jobs.py` — submits extraction jobs to DISCOVER/NOVA.
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

## Notes

- The `.set` file is authoritative; folder discovery is disabled.
- Update annotator names via `CG` and `HR` at the top of the script.
- openSMILE may emit `Segment too short, filling with NaN` warnings (make window bigger which includes possible tradeoffs). This can introduce NaNs in the output stream. Downstream processing should account for missing values (e.g., imputation or filtering).
