import math
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd

"""
Merge ViLearn annotations into per-session dataframes.

- Uses transcript intervals to align/group features to the speaking role.
- Downsamples 90 Hz engagement/task_engagement to 40 ms bins.
- Outputs one CSV per session with role-specific columns.
"""

# ------------------------
# CONFIG
# ------------------------
EXPORT_ROOT = Path("data/discover")
OUTPUT_ROOT = Path("data/discover/merged")
STREAM_ROOT = Path("/mnt/datasets/nova/data/vilearn_more")
OPENSMILE_DIMS_FILE = Path("discover/opensmile_egemapsv02_functionals_dims.txt")
SESSION_SET_FILES = [Path("discover/vilearn_more.set")]
GENERATE_PARQUET = False  # set True to include stream features (opensmile, emow2v, sentiment embeddings) in Parquet output

ROLES = ["p_blue", "p_green", "p_red"]

STEP_MS_40 = 40.0
STEP_MS_90 = 1000.0 / 90.0

# TEMP: limit to one dyad + one triad for testing (set to [] for full run)
TEST_SESSIONS = ["recording_dyad_01", "recording_triad_01"]
TEST_SESSIONS = []  # uncomment for full run

# ------------------------
# HELPERS
# ------------------------

def load_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    return df


def load_sessions_from_set_files() -> list[str]:
    sessions: list[str] = []
    for set_file in SESSION_SET_FILES:
        if not set_file.exists():
            continue
        entries = [line.strip() for line in set_file.read_text().splitlines()]
        sessions.extend([e for e in entries if e])
    return sorted(set(sessions))


def load_opensmile_dim_names() -> list[str]:
    if not OPENSMILE_DIMS_FILE.exists():
        return []
    names: list[str] = []
    for line in OPENSMILE_DIMS_FILE.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split("\t", 1)
        names.append(parts[1] if len(parts) == 2 else parts[0])
    return names


def read_stream_matrix(stream_header_path: Path) -> np.ndarray | None:
    data_path = stream_header_path.with_suffix(stream_header_path.suffix + "~")
    if not stream_header_path.exists() or not data_path.exists():
        return None
    root = ET.parse(stream_header_path).getroot()
    info = root.find("info")
    if info is None:
        return None
    dim = int(info.get("dim"))
    # Discover writes FLOAT binary streams as little-endian float32
    raw = np.fromfile(data_path, dtype="<f4")
    rows = raw.size // dim
    if rows == 0:
        return None
    if rows * dim != raw.size:
        raw = raw[: rows * dim]
    return raw.reshape(rows, dim)


def stream_df_for_session(session: str, role: str, stream_name: str, col_names: list[str]) -> pd.DataFrame | None:
    stream_path = STREAM_ROOT / session / f"{role}.{stream_name}.stream"
    matrix = read_stream_matrix(stream_path)
    if matrix is None:
        return None
    if not col_names or len(col_names) != matrix.shape[1]:
        col_names = [f"{stream_name}_{i:04d}" for i in range(matrix.shape[1])]
    df = pd.DataFrame(matrix, columns=col_names)
    df["time_ms"] = np.arange(len(df), dtype=float) * STEP_MS_40
    return df


def add_time_index(df: pd.DataFrame, step_ms: float) -> pd.DataFrame:
    out = df.copy()
    out["time_ms"] = np.arange(len(out), dtype=float) * step_ms
    return out


def downsample_to_40ms(df: pd.DataFrame, step_ms: float, prefix: str) -> pd.DataFrame:
    """Downsample by averaging into 40 ms bins."""
    tmp = add_time_index(df, step_ms)
    tmp["bin"] = (tmp["time_ms"] // STEP_MS_40).astype(int)
    # Drop confidence columns: not meaningful here
    # - model streams (sentiment/arousal/dominance/valence): conf == score
    # - engagement/task_engagement: conf is human-provided (often 1.0)
    cols = [c for c in tmp.columns if c not in {"time_ms", "bin", "conf"}]
    agg = tmp.groupby("bin")[cols].mean().reset_index()
    agg["time_ms"] = agg["bin"] * STEP_MS_40
    agg = agg.drop(columns=["bin"])
    agg = agg.rename(columns={"score": prefix})
    agg = agg.add_prefix(f"{prefix}.")
    agg = agg.rename(columns={f"{prefix}.time_ms": "time_ms", f"{prefix}.{prefix}": prefix})
    return agg


def attach_transcript(base: pd.DataFrame, tr: pd.DataFrame, role: str) -> pd.DataFrame:
    """Assign transcript text/conf to 40 ms bins using interval coverage."""
    base = base.copy()
    base[f"text_{role}"] = None
    # text_conf is not model-derived (human corrected) and not needed downstream

    # Ensure numeric
    tr = tr.copy()
    tr["from"] = pd.to_numeric(tr["from"], errors="coerce")
    tr["to"] = pd.to_numeric(tr["to"], errors="coerce")

    n = len(base)
    for _, row in tr.iterrows():
        if math.isnan(row["from"]) or math.isnan(row["to"]):
            continue
        start_idx = int(max(0, math.floor(row["from"] / STEP_MS_40)))
        end_idx = int(min(n - 1, math.floor(row["to"] / STEP_MS_40)))
        if end_idx < start_idx:
            continue
        base.loc[start_idx:end_idx, f"text_{role}"] = row.get("name")

    base[f"speaking_{role}"] = base[f"text_{role}"].notna()
    return base


# ------------------------
# MAIN
# ------------------------

def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    opensmile_names = [f"opensmile_{x}" for x in load_opensmile_dim_names()]

    sessions = load_sessions_from_set_files()
    if not sessions:
        # fallback if set file is missing
        sessions = sorted([p.name for p in EXPORT_ROOT.iterdir() if p.is_dir() and p.name.startswith("recording_")])
    if TEST_SESSIONS:
        sessions = [s for s in sessions if s in TEST_SESSIONS]
    print(f"Merging {len(sessions)} sessions...")

    for ses in sessions:
        ses_dir = EXPORT_ROOT / ses

        # Group-level 40 ms annotations
        arousal = load_csv(ses_dir / "arousal.group.carlosgonzalez.csv")
        dominance = load_csv(ses_dir / "dominance.group.carlosgonzalez.csv")
        valence = load_csv(ses_dir / "valence.group.carlosgonzalez.csv")

        # Group-level 90 Hz annotation (downsample)
        task_eng = load_csv(ses_dir / "task_engagement.group.helenrisack.csv")
        task_eng_40 = None if task_eng is None else downsample_to_40ms(task_eng, STEP_MS_90, "task_engagement")

        # Determine base length (40 ms grid) from group features
        candidates = []
        if arousal is not None:
            candidates.append(len(arousal))
        if dominance is not None:
            candidates.append(len(dominance))
        if valence is not None:
            candidates.append(len(valence))

        if candidates:
            n = max(candidates)
        else:
            # fallback: first available transcript length
            any_tr = None
            for role in ROLES:
                tr_path = ses_dir / f"transcript.{role}.helenrisack.csv"
                if tr_path.exists():
                    any_tr = pd.read_csv(tr_path)
                    break
            if any_tr is None:
                continue
            max_to = float(any_tr["to"].max())
            n = int(math.ceil(max_to / STEP_MS_40)) + 1

        session_df = pd.DataFrame({"time_ms": np.arange(n, dtype=float) * STEP_MS_40})

        # Attach group annotations
        if arousal is not None:
            ar = arousal.copy()
            if "conf" in ar.columns:
                # conf mirrors score for arousal
                ar = ar.drop(columns=["conf"])
            ar = ar.rename(columns={"score": "arousal"})
            session_df = session_df.join(ar.reindex(session_df.index))
        if dominance is not None:
            dom = dominance.copy()
            if "conf" in dom.columns:
                # conf mirrors score for dominance
                dom = dom.drop(columns=["conf"])
            dom = dom.rename(columns={"score": "dominance"})
            session_df = session_df.join(dom.reindex(session_df.index))
        if valence is not None:
            val = valence.copy()
            if "conf" in val.columns:
                # conf mirrors score for valence
                val = val.drop(columns=["conf"])
            val = val.rename(columns={"score": "valence"})
            session_df = session_df.join(val.reindex(session_df.index))

        if task_eng_40 is not None:
            session_df = session_df.merge(task_eng_40, on="time_ms", how="left")

        for role in ROLES:
            if role == "p_red" and "triad" not in ses.lower():
                continue

            tr = load_csv(ses_dir / f"transcript.{role}.helenrisack.csv")
            if tr is None:
                continue

            sentiment = load_csv(ses_dir / f"sentiment.{role}.carlosgonzalez.csv")
            engagement = load_csv(ses_dir / f"engagement.{role}.helenrisack.csv")
            engagement_40 = None if engagement is None else downsample_to_40ms(engagement, STEP_MS_90, "engagement")

            base = pd.DataFrame({"time_ms": session_df["time_ms"]})
            base = attach_transcript(base, tr, role)

            # Attach 40 ms per-role and group annotations by index
            if sentiment is not None:
                sent = sentiment.copy().reset_index(drop=True)
                if "conf" in sent.columns:
                    # conf mirrors score for sentiment
                    sent = sent.drop(columns=["conf"])
                sent = sent.rename(columns={"score": f"sentiment_{role}"})
                base = base.join(sent.reindex(base.index))

            # Attach downsampled 90 Hz annotations via time_ms
            if engagement_40 is not None:
                base = base.merge(engagement_40, on="time_ms", how="left")
                if "engagement" in base.columns:
                    base = base.rename(columns={"engagement": f"engagement_{role}"})

            session_df = session_df.merge(base, on="time_ms", how="left")

        session_df.insert(0, "session", ses)

        out_csv = OUTPUT_ROOT / f"{ses}.csv"
        session_df.to_csv(out_csv, index=False)

        if GENERATE_PARQUET:
            # Parquet includes stream features (opensmile, emow2v, sentiment embeddings).
            session_parquet_df = session_df.copy()
            opensmile_df = stream_df_for_session(ses, "group", "opensmile", opensmile_names)
            if opensmile_df is not None:
                session_parquet_df = session_parquet_df.merge(opensmile_df, on="time_ms", how="left")

            emow2v_df = stream_df_for_session(ses, "group", "emow2v", [])
            if emow2v_df is not None:
                emow2v_df = emow2v_df.rename(columns={c: f"emow2v_{c.split('_')[-1]}" for c in emow2v_df.columns if c != "time_ms"})
                session_parquet_df = session_parquet_df.merge(emow2v_df, on="time_ms", how="left")

            for role in ROLES:
                if role == "p_red" and "triad" not in ses.lower():
                    continue
                sent_stream_df = stream_df_for_session(ses, role, "sentiment", [])
                if sent_stream_df is None:
                    continue
                sent_stream_df = sent_stream_df.rename(columns={c: f"sentiment_emb_{role}_{c.split('_')[-1]}" for c in sent_stream_df.columns if c != "time_ms"})
                session_parquet_df = session_parquet_df.merge(sent_stream_df, on="time_ms", how="left")

            out_parquet = OUTPUT_ROOT / f"{ses}.parquet"
            session_parquet_df.to_parquet(out_parquet, index=False)

    print("Done")


if __name__ == "__main__":
    main()
