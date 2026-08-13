"""C6 builder — feature matrix for an ARBITRARY group set (not just the analysed 20).

`a1_lib.load_data()` hard-filters to `data/group_names_with_time_floorlevel.csv` (20
groups / 185 windows). The gaze feature file actually covers all 25 groups / 243
windows, and `data/group_names_with_time_subsetsFullVERSION.csv` has their interaction
timestamps, so the excluded groups can be brought back in. This module rebuilds the
gaze + label + linguistic matrix for any subset of the 25.

Audio (v/a/d) and the LLM rubric are deliberately NOT built: both are reserved for the
follow-up journal paper, and the LLM scores only exist for the 185-window grid.

`build(groups)` is validated against the committed `discover/paper_results/c2_features.csv`
on the 20-group set — run this file directly to execute that check.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import glob
import re
import numpy as np
import pandas as pd

HIS_CSV = "Recordings/SavedData/v2_no_low_sampled/60s_TE_correlation_2026-05-19.csv"
FULL = "data/group_names_with_time_subsetsFullVERSION.csv"
FLOOR = "data/group_names_with_time_floorlevel.csv"

LING = ["segment_count", "word_count", "avg_word_length", "question_ratio",
        "speech_ratio", "mean_segment_duration_s", "unfinished_ratio"]
GAZE = ["MG", "1d_DG", "BPM", "blink_durations",
        "G_OnSpeaker", "No_G_OnSpeaker", "G_SI", "No_G_SI"]

ALPHA_LOW = ["dyad_10", "triad_05", "triad_11", "triad_14"]  # per-group Cronbach alpha < .60


def group_sets():
    g20 = sorted(pd.read_csv(FLOOR, sep=";").Group_Name)
    g25 = sorted(pd.read_csv(FULL, sep=";").Group_Name)
    return {
        "g20": g20,                                              # the analysed set
        "g21": sorted(set(g20) | {"dyad_09"}),                   # + the excluded outlier (R1 Q3)
        "g25": g25,                                              # every group with usable data
        "g16": sorted(set(g20) - set(ALPHA_LOW)),                # drop annotator-agreement < .60
    }


def _bounds():
    full = pd.read_csv(FULL, sep=";")
    rt = pd.read_csv("data/recording_times_group_info.csv")
    rec = dict(zip(rt["long_name"], pd.to_datetime(rt["start_recording"])))
    return {r["Group_Name"]: (pd.to_datetime(r["TS_Start_Interaction"]) - rec[r["Group_Name_Long"]]).total_seconds() * 1000.0
            for _, r in full.iterrows() if r["Group_Name_Long"] in rec}


def _segments(g):
    """(from_ms, to_ms, text, speaker) for every ASR segment of a group."""
    segs = []
    for path in glob.glob(f"data/discover/recording_{g}/transcript.*.helenrisack.csv"):
        m = re.search(r"transcript\.(.+)\.helenrisack\.csv$", os.path.basename(path))
        spk = m.group(1) if m else path
        tr = pd.read_csv(path)
        for _, s in tr.iterrows():
            segs.append((float(s["from"]), float(s["to"]), str(s["name"]).strip(), spk))
    return sorted(segs)


def _unfinished_flags(segs):
    """A segment is 'unfinished' if it trails off ('…'/'...'), or has no terminal
    punctuation and is not continued by the SAME speaker in the IMMEDIATELY NEXT
    segment starting within 2 s. (The "immediately next" reading is what reproduces
    the committed c2_features.csv exactly — a later same-speaker turn after somebody
    else has spoken does not count as a continuation.)"""
    flags = []
    for i, (f, t, txt, spk) in enumerate(segs):
        s = txt.rstrip()
        if s.endswith("…") or s.endswith("..."):
            flags.append(True)
            continue
        if s[-1:] in (".", "!", "?"):
            flags.append(False)
            continue
        cont = (i + 1 < len(segs) and segs[i + 1][3] == spk
                and (segs[i + 1][0] - t) < 2000.0)
        flags.append(not cont)
    return flags


def build(groups):
    """Gaze features + clean 2-annotator TE label + 7 linguistic stats, per 60 s window."""
    his = pd.read_csv(HIS_CSV)
    his = his[his.group_name.isin(groups)].reset_index(drop=True)
    bounds = _bounds()

    labels = {}
    for g, gg in his.groupby("group_name"):
        fr = pd.read_csv(f"data/discover/merged/recording_{g}.csv",
                         usecols=["time_ms", "task_engagement"])
        s0 = bounds[g]
        for idx, hr in gg.iterrows():
            w0 = s0 + hr["seconds_interaction_window"] * 1000.0
            m = (fr.time_ms >= w0) & (fr.time_ms < w0 + 60_000.0)
            labels[idx] = fr.loc[m, "task_engagement"].mean()
    his["our_te"] = pd.Series(labels)
    assert his["our_te"].notna().all(), "NaN label - empty window"
    his["y"] = (his["our_te"] > 0.5).astype(int)
    his["sec"] = his["seconds_interaction_window"]
    his["grp"] = np.where(his.group_type == "dyad", "D", "T")

    for c in LING:
        his[c] = 0.0
    for g, gg in his.groupby("group_name"):
        segs = _segments(g)
        unf = _unfinished_flags(segs)
        s0 = bounds[g]
        for idx in gg.index:
            w0 = s0 + his.loc[idx, "sec"] * 1000.0
            w1 = w0 + 60_000.0
            win = [(max(f, w0), min(t, w1), txt, u)
                   for (f, t, txt, _), u in zip(segs, unf) if t > w0 and f < w1 and txt]
            if not win:
                continue
            words = [w for _, _, txt, _ in win for w in txt.split()]
            durs = [(t - f) / 1000.0 for f, t, _, _ in win]
            his.loc[idx, "segment_count"] = len(win)
            his.loc[idx, "word_count"] = len(words)
            his.loc[idx, "avg_word_length"] = float(np.mean([len(w) for w in words])) if words else 0.0
            his.loc[idx, "question_ratio"] = float(np.mean([("?" in txt) for _, _, txt, _ in win]))
            his.loc[idx, "speech_ratio"] = float(np.sum(durs)) / 60.0
            his.loc[idx, "mean_segment_duration_s"] = float(np.mean(durs))
            his.loc[idx, "unfinished_ratio"] = float(np.mean([u for _, _, _, u in win]))
    return his


if __name__ == "__main__":  # validation against the committed 20-group matrix
    ref = pd.read_csv("discover/paper_results/c2_features.csv")
    got = build(group_sets()["g20"])
    key = ["group_name", "sec"]
    m = ref[key + LING + GAZE + ["y", "our_te"]].merge(
        got[key + LING + GAZE + ["y", "our_te"]], on=key, suffixes=("_ref", "_new"), validate="one_to_one")
    print(f"rows ref={len(ref)} new={len(got)} matched={len(m)}")
    bad = 0
    for c in LING + GAZE + ["y", "our_te"]:
        d = (m[f"{c}_ref"] - m[f"{c}_new"]).abs()
        n = int((d > 1e-6).sum())
        flag = "  <-- MISMATCH" if n else ""
        print(f"  {c:26s} max|diff|={d.max():.3e}  n_diff={n}{flag}")
        bad += n
    print("VALIDATION", "PASSED" if bad == 0 else f"FAILED ({bad} cells)")
    print("unfinished_ratio base rate:", round(got.unfinished_ratio.mean(), 4))
