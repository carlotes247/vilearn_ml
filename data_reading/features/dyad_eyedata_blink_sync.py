import os
import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime

from vrcft_eyedata_loader import VrcftEyeDataLoader
from blink_collision import BlinkCollisionsList

def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

#region variables
dyad_txt_base_dir = os.path.join(_project_root(), "data")
dyad_txt_files = [
    "DYAD_2026_10_22_Eng_Case_1.txt", "DYAD_2026_10_22_Eng_Case_2.txt",
    "DYAD_2026_10_22_Eng_Case_3.txt", "DYAD_2026_10_22_Eng_Case_4.txt",
    "DYAD_2026_10_22_Eng_Case_5.txt", "DYAD_2026_10_22_Eng_Case_6.txt",
    "DYAD_2026_10_22_Eng_Case_7.txt",
]
dyad_sync_results = pd.DataFrame(columns=[
    "dyad_name",
    "p1_eyedata_file", "p2_eyedata_file",
    "p1_blink_onsets", "p2_blink_onsets",
    "count_sync_blinks", "percent_sync_blinks",
    "avg_blinks_async_ms",
    "p1_mean_blink_duration_ms", "p2_mean_blink_duration_ms",
    "session_duration_seconds", "p1_blink_rate_per_min", "p2_blink_rate_per_min",
    "p1_gaze_mismatch_frames", "p1_gaze_mismatch_percent",
    "p2_gaze_mismatch_frames", "p2_gaze_mismatch_percent",
])

# blink detection params (same conventions as the previous pipeline)
blink_openness_threshold = 0.8
min_blink_ms = 100.0
max_blink_ms = 500.0
# collision window in +/− seconds around the reference onset
collision_window_seconds = 1.5
# minimum euclidean distance between the left and right gaze vectors for a
# frame to count as a "different" gaze vector (gaze coordinate units)
gaze_mismatch_threshold = 0.0
#endregion

def load_dyads() -> list[tuple[str, str, str]]:
    """
    Reads each dyad txt file and returns a list of (dyad_name, p1_path, p2_path).
    Each txt file contains two per-participant eyedata CSV paths (line 1 = P1, line 2 = P2),
    relative to the project root.
    """
    dyads = []
    for txt_name in dyad_txt_files:
        txt_path = os.path.join(dyad_txt_base_dir, txt_name)
        if not os.path.exists(txt_path):
            print(f"Warning: dyad file {txt_name} not found at {txt_path}")
            continue
        with open(txt_path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
        if len(lines) < 2:
            print(f"Warning: {txt_name} does not contain two participant paths; skipping.")
            continue
        dyad_name = os.path.splitext(os.path.basename(txt_name))[0]
        dyads.append((dyad_name, lines[0], lines[1]))
    return dyads

def process_dyad(dyad_name, p1_evd, p2_evd):
    """
    Detects valid blinks for both participants of a dyad and computes the synced
    blink collisions (closest onset hit within +/− collision_window_seconds).
    """
    onsets_p1 = p1_evd.blink_onsets
    onsets_p2 = p2_evd.blink_onsets

    hits = BlinkCollisionsList("P1", "P2")
    # the participant with fewer valid blinks is the reference (same as the previous
    # pipeline in GroupCSVDataLoader.get_blink_onset_collisions)
    if len(onsets_p2) >= len(onsets_p1):
        ref_onsets, adv_onsets = onsets_p1, onsets_p2
    else:
        ref_onsets, adv_onsets = onsets_p2, onsets_p1
    for ref_ts in ref_onsets:
        within = [c for c in adv_onsets if abs((c - ref_ts).total_seconds()) <= collision_window_seconds]
        if not within:
            continue
        closest = min(within, key=lambda t: abs((t - ref_ts).total_seconds()))
        if closest == ref_ts:
            continue
        delta_ms = (closest - ref_ts).total_seconds() * 1000.0
        hits.add_collision(ref_ts, closest, delta_ms)
    return hits

def calculate_sync_blink_asynchrony_bins(hits: BlinkCollisionsList) -> dict:
    """
    Counts synced blinks per 250ms time-lag bin (same binning as BlinkStats:
    bins keyed by lower bound ms from -1500 up to 1250, upper = lower + 250).
    """
    bins = {-1500: 0, -1250: 0, -1000: 0, -750: 0, -500: 0, -250: 0,
            0: 0, 250: 0, 500: 0, 750: 0, 1000: 0, 1250: 0}
    for collision in hits.collisions:
        bin_timelag = collision.delta_ms / 250
        if bin_timelag < 0:
            bin_multiplier = int(bin_timelag - 1)
        else:
            bin_multiplier = int(bin_timelag)
        if bin_multiplier == 6: bin_multiplier = 5    # 1250-1500 bin upper bound
        if bin_multiplier == -7: bin_multiplier = -6  # -1500..-1250 bin lower bound
        bins[bin_multiplier * 250] += 1
    return bins

def calculate_gaze_mismatch_stats(evd: VrcftEyeDataLoader,
                                  gaze_diff_threshold: float = gaze_mismatch_threshold) -> tuple[int, float]:
    """
    Counts the frames in the loaded dataset where the left eye gaze vector
    differs from the right eye gaze vector, i.e. the euclidean distance
    between (LeftGazeX, LeftGazeY) and (RightGazeX, RightGazeY) exceeds
    ``gaze_diff_threshold``.
    Returns (mismatch_frame_count, mismatch_fraction) where the fraction is
    the count over all frames with valid gaze values. Returns (0, np.nan) if
    the data is not loaded or the gaze columns are missing.
    """
    df = evd.raw_data
    if df is None or len(df) == 0:
        return 0, np.nan
    gaze_cols = ["LeftGazeX", "LeftGazeY", "RightGazeX", "RightGazeY"]
    missing = [c for c in gaze_cols if c not in df.columns]
    if missing:
        print(f"Warning: {evd.participant_name} ({os.path.basename(evd.file_path)}) "
              f"is missing gaze columns {missing}; cannot calculate gaze mismatch.")
        return 0, np.nan
    dx = pd.to_numeric(df["LeftGazeX"], errors="coerce") - pd.to_numeric(df["RightGazeX"], errors="coerce")
    dy = pd.to_numeric(df["LeftGazeY"], errors="coerce") - pd.to_numeric(df["RightGazeY"], errors="coerce")
    dist = np.sqrt(dx**2 + dy**2)
    valid = dist.notna().to_numpy()
    if not valid.any():
        return 0, np.nan
    mismatch_count = int(((dist > gaze_diff_threshold).to_numpy() & valid).sum())
    return mismatch_count, float(mismatch_count / valid.sum())

def calculate_gaze_mismatch_percent(evd: VrcftEyeDataLoader,
                                    gaze_diff_threshold: float = gaze_mismatch_threshold) -> float:
    """
    Fraction (0-1) of frames in the loaded dataset where the left eye gaze
    vector differs from the right eye gaze vector (see
    calculate_gaze_mismatch_stats). Returns np.nan if the data is not loaded
    or the gaze columns are missing.
    """
    return calculate_gaze_mismatch_stats(evd, gaze_diff_threshold)[1]

def plot_dyad_blink_asynchrony_boxplot(results_rows, collisions_per_dyad, out_png: str):
    """
    Boxplot of the synced blinks' time lag per dyad (signed ms; negative = the
    dyad partner blinked first). Mirrors BlinkStats.calculate_blink_asynchrony_distribution_boxplot.
    """
    data, names = [], []
    for dyad_name, hits in collisions_per_dyad.items():
        if hits.collisions:
            data.append([c.delta_ms for c in hits.collisions])
            names.append(dyad_name)
    if not data:
        print("No synced blinks found; skipping the asynchrony boxplot.")
        return
    fig, ax = plt.subplots()
    ax.boxplot(data, tick_labels=names)
    ax.set_ylabel("Blink asynchrony (ms)")
    ax.set_title("Synced blinks asynchrony per dyad (2026 eyedata)")
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved boxplot: {out_png}")

def compute_dyad_sync_stats(dyads):
    """
    For each dyad loads the eyedata, detects valid blinks, computes the
    percentage of synced blinks and the average blink asynchrony (ms).
    Appends results to dyad_sync_results and returns it.
    """
    rows = []
    collisions_per_dyad = {}
    for dyad_name, p1_rel, p2_rel in dyads:
        p1 = VrcftEyeDataLoader(p1_rel, "P1")
        p2 = VrcftEyeDataLoader(p2_rel, "P2")
        # load restricted to the interaction window (UTC boundaries, from
        # 2026_retreat_dyads_interaction_times.csv) so blinks/rates below only
        # reflect the window; participant labels p1/p2 are derived from the
        # loader's participant_name ("P1"/"P2")
        try:
            loaded_p1 = p1.load_data(filter_interaction_window=True, dyad_name=dyad_name)
            loaded_p2 = p2.load_data(filter_interaction_window=True, dyad_name=dyad_name)
        except (ValueError, KeyError) as e:
            print(f"Skipping dyad {dyad_name}: interaction window lookup failed ({e}).")
            continue
        loaded = loaded_p1 and loaded_p2
        if not loaded:
            print(f"Skipping dyad {dyad_name} because eyedata could not be loaded.")
            continue
        p1.detect_blinks(blink_openness_threshold, min_blink_ms, max_blink_ms)
        p2.detect_blinks(blink_openness_threshold, min_blink_ms, max_blink_ms)

        hits = process_dyad(dyad_name, p1, p2)
        collisions_per_dyad[dyad_name] = hits
        total_onsets = len(p1.blink_onsets) + len(p2.blink_onsets)
        n_synced = len(hits.collisions)
        percent_sync = (n_synced * 2) / total_onsets if total_onsets > 0 else np.nan
        avg_async = float(np.mean([abs(c.delta_ms) for c in hits.collisions])) if n_synced > 0 else np.nan

        session_s = float(np.mean([p1.session_duration_seconds, p2.session_duration_seconds]))
        p1_rate = len(p1.blink_onsets) / (session_s / 60.0) if session_s > 0 else np.nan
        p2_rate = len(p2.blink_onsets) / (session_s / 60.0) if session_s > 0 else np.nan

        p1_gaze_frames, p1_gaze_percent = calculate_gaze_mismatch_stats(p1)
        p2_gaze_frames, p2_gaze_percent = calculate_gaze_mismatch_stats(p2)

        rows.append({
            "dyad_name": dyad_name,
            "p1_eyedata_file": p1.file_path, "p2_eyedata_file": p2.file_path,
            "p1_blink_onsets": len(p1.blink_onsets), "p2_blink_onsets": len(p2.blink_onsets),
            "count_sync_blinks": n_synced, "percent_sync_blinks": percent_sync,
            "avg_blinks_async_ms": avg_async,
            "p1_mean_blink_duration_ms": p1.mean_blink_duration_ms(),
            "p2_mean_blink_duration_ms": p2.mean_blink_duration_ms(),
            "session_duration_seconds": session_s,
            "p1_blink_rate_per_min": p1_rate, "p2_blink_rate_per_min": p2_rate,
            "p1_gaze_mismatch_frames": p1_gaze_frames, "p1_gaze_mismatch_percent": p1_gaze_percent,
            "p2_gaze_mismatch_frames": p2_gaze_frames, "p2_gaze_mismatch_percent": p2_gaze_percent,
        })

    global dyad_sync_results
    dyad_sync_results = pd.concat([dyad_sync_results, pd.DataFrame(rows)], ignore_index=True)
    return dyad_sync_results, collisions_per_dyad

if __name__ == "__main__":
    found = load_dyads()
    print(f"Found {len(found)} dyads to process.")
    results, collisions_per_dyad = compute_dyad_sync_stats(found)
    print("\n=== Dyad blink synchrony summary (2026 retreat eyedata) ===")
    print(results.to_string(index=False))
    out_dir = os.path.join(_project_root(), "data", "2026_dyad_blink_sync")
    os.makedirs(out_dir, exist_ok=True)
    results.to_csv(os.path.join(out_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_dyad_blink_sync_all_dyads.csv"), index=False)
    # calculate averages of relevant columns
    avg_count_sync_blink = results['count_sync_blinks'].mean()
    avg_percent_sync_blinks= results['percent_sync_blinks'].mean()
    avg_blink_duration_ms = np.average([results['p1_mean_blink_duration_ms'].mean(), results['p2_mean_blink_duration_ms'].mean()])
    avg_blink_rate_per_min = np.average([results['p1_blink_rate_per_min'].mean(), results['p2_blink_rate_per_min'].mean()])
    avg_session_duration_seconds = results['session_duration_seconds'].mean()
    avg_gaze_mismatch_frames = np.average([results['p1_gaze_mismatch_frames'].mean(), results['p2_gaze_mismatch_frames'].mean()])
    avg_gaze_mismatch_percent = np.average([results['p1_gaze_mismatch_percent'].mean(), results['p2_gaze_mismatch_percent'].mean()])
    sum_sessions_mins = results['session_duration_seconds'].sum() / 60
    summary = pd.DataFrame([{
        'avg_count_sync_blink':       avg_count_sync_blink,
        'avg_percent_sync_blinks':    avg_percent_sync_blinks,
        'avg_blink_duration_ms':      avg_blink_duration_ms,
        'avg_blink_rate_per_min':     avg_blink_rate_per_min,
        'avg_session_duration_sec':   avg_session_duration_seconds,
        'sum_sessions_mins':          sum_sessions_mins,
        'avg_gaze_mismatch_frames':   avg_gaze_mismatch_frames,
        'avg_gaze_mismatch_percent':  avg_gaze_mismatch_percent,
    }])
    summary.to_csv(os.path.join(out_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_dyads_retreat_blinks_features_summary.csv"), index=False)

    # per-dyad synced blink collision dataframes (ref_TS, adv_TS, delta_ms)
    for dyad_name, hits in collisions_per_dyad.items():
        if hits.collisions:
            hits.get_collisions_df().to_csv(os.path.join(out_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_dyad_{dyad_name}_collisions.csv"), index=False)
            
    # per-dyad synced blink count in each 250ms time-lag bin (same bins as BlinkStats)
    if collisions_per_dyad:
        bins_rows = {d: calculate_sync_blink_asynchrony_bins(h) for d, h in collisions_per_dyad.items()}
        pd.DataFrame(bins_rows).T.to_csv(os.path.join(out_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_dyad_blink_async_250ms_bins.csv"))

    plot_dyad_blink_asynchrony_boxplot(results, collisions_per_dyad,
                                       os.path.join(out_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_dyad_blink_async_boxplot_all_dyads.png"))
    print(f"\nSaved results in: {out_dir}")
