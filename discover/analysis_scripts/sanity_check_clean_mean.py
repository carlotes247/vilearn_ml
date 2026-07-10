"""Sanity check: clean 2-annotator TE mean vs Carlos's 60s TE column.

Rebuilds TE = NaN-skipping mean of helen + carlos raw annotations on helen's
90 Hz grid (carlos 60 Hz files linearly interpolated), slices to the
interaction window (group_durations_all_commas.csv offsets), averages into
60 s windows, and correlates against 60s_TE_correlation_2026-05-19.csv.
"""
import os
import numpy as np
import pandas as pd

ANNO = "data/annotations"
INFO = pd.read_csv("data/group_durations_all_commas.csv")
HIS = pd.read_csv("Recordings/SavedData/v2_no_low_sampled/60s_TE_correlation_2026-05-19.csv")
FLOOR = pd.read_csv("data/group_names_with_time_floorlevel.csv", sep=";")


def load_raw(path):
    df = pd.read_csv(path, sep=";", names=["score", "conf"])
    return pd.to_numeric(df["score"], errors="coerce").to_numpy()


def carlos_on_helen_grid(ses, n_helen):
    """Pick the carlos file whose duration matches helen's; resample to 90 Hz grid."""
    dur_h = n_helen / 90.0
    cands = [
        (f"{ANNO}/{ses}/task engagement.group.carlosgonzalez.annotation~", 90),
        (f"{ANNO}/{ses}/task engagement60Hz.group.carlosgonzalez.annotation~", 60),
    ]
    for path, freq in cands:
        if not os.path.exists(path):
            continue
        v = load_raw(path)
        for f in (freq, 60, 90):  # actual rate may differ from filename (dyad_08)
            if abs(len(v) / f - dur_h) < 2.0:
                t_src = np.arange(len(v)) / f
                t_dst = np.arange(n_helen) / 90.0
                mask = ~np.isnan(v)
                if mask.sum() < 2:
                    return np.full(n_helen, np.nan)
                out = np.interp(t_dst, t_src[mask], v[mask])
                out[(t_dst < t_src[mask][0]) | (t_dst > t_src[mask][-1])] = np.nan
                return out
    raise FileNotFoundError(f"no duration-matching carlos file for {ses}")


rows = []
for g in sorted(HIS.group_name.unique()):
    ses = f"recording_{g}"
    helen = pd.read_csv(f"data/discover/{ses}/task_engagement.group.helenrisack.csv")["score"].to_numpy()
    carlos = carlos_on_helen_grid(ses, len(helen))
    mean_te = np.nanmean(np.vstack([helen, carlos]), axis=0)  # clean mean: NaN skipped, not zeroed

    info = INFO[INFO["name"] == g]
    start = float(info["offset_recording_interaction_start"].iloc[0])
    end = start + float(info["duration_interaction"].iloc[0])
    i0, i1 = int(round(start * 90)), int(round(end * 90))
    seg = mean_te[i0:i1]

    his_g = HIS[HIS.group_name == g]
    for _, r in his_g.iterrows():
        w0 = int(round(r.seconds_interaction_window * 90))
        w1 = w0 + 60 * 90
        if w0 >= len(seg):
            continue
        ours = np.nanmean(seg[w0:min(w1, len(seg))])
        rows.append({"group_name": g, "sec": r.seconds_interaction_window,
                     "ours_clean": ours, "his_te": r.TE})

df = pd.DataFrame(rows).dropna()
df["floor"] = df.group_name.isin(FLOOR.Group_Name)
d = df[df.floor]
print(f"windows joined: {len(d)} (his floorlevel total 185)")
print(f"pooled corr ours_clean vs his_te: {d.ours_clean.corr(d.his_te):.4f}")
print(f"mean ours {d.ours_clean.mean():.3f}  his {d.his_te.mean():.3f}")
hi_o, hi_h = d.ours_clean > 0.5, d.his_te > 0.5
gt = d.group_name.str.extract(r"(dyad|triad)")[0]
print(f"high% ours {100*hi_o.mean():.1f} (D {100*hi_o[gt=='dyad'].mean():.1f} / T {100*hi_o[gt=='triad'].mean():.1f})")
print(f"high% his  {100*hi_h.mean():.1f} (D {100*hi_h[gt=='dyad'].mean():.1f} / T {100*hi_h[gt=='triad'].mean():.1f})")
print(f"label agreement (binarized): {100*(hi_o==hi_h).mean():.1f}%")
print("\nper-group corr:")
pg = d.groupby("group_name").apply(lambda x: x.ours_clean.corr(x.his_te), include_groups=False).round(3)
print(pg.to_string())
