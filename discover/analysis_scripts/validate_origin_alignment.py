"""A1 origin-alignment validation.

Replicates derive's interaction-bound logic, windows our frame-level clean-mean
task_engagement onto Carlos's (group, seconds_interaction_window) grid, and
correlates against his CSV TE. On the 90 Hz groups (his averaging is correct
there) corr should be ~1.0 -> confirms our frame origin == his interaction-second
origin, so [sec, sec+60) picks the right frames. 60 Hz groups expected lower
(his time-stretch bug), not an origin problem.
"""
import glob
import os
import numpy as np
import pandas as pd

FLOOR = pd.read_csv("data/group_names_with_time_floorlevel.csv", sep=";")
RT = pd.read_csv("data/recording_times_group_info.csv")
HIS = pd.read_csv("Recordings/SavedData/v2_no_low_sampled/60s_TE_correlation_2026-05-19.csv")
MERGED = "data/discover/merged"

rec_start = dict(zip(RT["long_name"], pd.to_datetime(RT["start_recording"])))

# detect 60Hz groups by presence of a task_engagement60Hz export
hz60 = set()
for p in glob.glob("data/discover/recording_*/task_engagement60Hz.*.csv"):
    g = os.path.basename(os.path.dirname(p)).replace("recording_", "")
    hz60.add(g)

bounds = {}
for _, r in FLOOR.iterrows():
    ln = r["Group_Name_Long"]
    if ln not in rec_start:
        print(f"  !! no rec_start for {r['Group_Name']} ({ln})")
        continue
    s = (pd.to_datetime(r["TS_Start_Interaction"]) - rec_start[ln]).total_seconds() * 1000.0
    e = (pd.to_datetime(r["TS_End_Interaction"]) - rec_start[ln]).total_seconds() * 1000.0
    bounds[r["Group_Name"]] = (s, e)

rows = []
for g, (s_ms, e_ms) in bounds.items():
    fpath = f"{MERGED}/recording_{g}.csv"
    if not os.path.exists(fpath):
        print(f"  !! no merged file {fpath}")
        continue
    fr = pd.read_csv(fpath, usecols=["time_ms", "task_engagement"])
    his_g = HIS[HIS.group_name == g]
    for _, hr in his_g.iterrows():
        sec = hr["seconds_interaction_window"]
        w0 = s_ms + sec * 1000.0
        w1 = w0 + 60_000.0
        m = (fr.time_ms >= w0) & (fr.time_ms < w1)
        our_te = fr.loc[m, "task_engagement"].mean()
        rows.append({
            "group": g, "sec": sec, "n_frames": int(m.sum()),
            "our_te": our_te, "his_te": hr["TE"],
            "hz": "60" if g in hz60 else "90",
        })

# ---- shift test: bug-independent origin check ----
# For each group, re-window our TE with the his-grid shifted by k windows and
# correlate vs his TE. If OUR origin is correct, shift k=0 maximizes corr for
# the clean groups. His time-STRETCH bug is not a shift, so bugged groups won't
# be rescued by any integer k (expected).
def shift_corr(g, s_ms, fr, his_g, k):
    vals_o, vals_h = [], []
    for _, hr in his_g.iterrows():
        sec = hr["seconds_interaction_window"] + k * 60.0
        w0 = s_ms + sec * 1000.0
        m = (fr.time_ms >= w0) & (fr.time_ms < w0 + 60_000.0)
        if m.sum() == 0:
            continue
        vals_o.append(fr.loc[m, "task_engagement"].mean())
        vals_h.append(hr["TE"])
    if len(vals_o) < 3:
        return np.nan
    return np.corrcoef(vals_o, vals_h)[0, 1]


print("--- shift test (corr at window-offset k; * = argmax) ---")
shift_rows = []
for g, (s_ms, e_ms) in bounds.items():
    fpath = f"{MERGED}/recording_{g}.csv"
    if not os.path.exists(fpath):
        continue
    fr = pd.read_csv(fpath, usecols=["time_ms", "task_engagement"])
    his_g = HIS[HIS.group_name == g]
    cs = {k: shift_corr(g, s_ms, fr, his_g, k) for k in (-2, -1, 0, 1, 2)}
    best = max(cs, key=lambda k: (cs[k] if not np.isnan(cs[k]) else -9))
    shift_rows.append({"group": g, "best_k": best, "c0": cs[0]})
    mark = lambda k: "*" if k == best else " "
    print("  %-10s " % g + "  ".join(f"k={k:+d}:{cs[k]:+.2f}{mark(k)}" for k in (-2, -1, 0, 1, 2)))

SH = pd.DataFrame(shift_rows)
clean = SH[SH.c0 >= 0.7]
print(f"\nclean groups (c0>=0.7): {len(clean)} -> {list(clean.group)}")
print(f"  of these, best_k==0: {(clean.best_k == 0).sum()}/{len(clean)}")
print(f"all groups best_k==0: {(SH.best_k == 0).sum()}/{len(SH)}")

R = pd.DataFrame(rows)
print(f"\nwindows mapped: {len(R)} (his floorlevel total 185)")
print(f"empty windows (0 frames): {(R.n_frames == 0).sum()}")
print(f"our_te NaN: {R.our_te.isna().sum()}")

print("\n--- per-group corr (our_te vs his_te) ---")
for g, gg in R.groupby("group"):
    v = gg.dropna(subset=["our_te", "his_te"])
    c = v.our_te.corr(v.his_te) if len(v) > 2 else np.nan
    lag = ""
    # quick shift check: corr if our windows shifted +/-1 vs his
    print(f"  {g:10s} hz={gg.hz.iloc[0]} n={len(gg):2d} corr={c:+.3f}")

print("\n--- corr by rate ---")
for hz, gg in R.groupby("hz"):
    v = gg.dropna(subset=["our_te", "his_te"])
    print(f"  {hz}Hz groups: pooled corr={v.our_te.corr(v.his_te):+.3f}  n={len(v)}")

# binary agreement at 0.5
R2 = R.dropna(subset=["our_te", "his_te"])
agree = ((R2.our_te > 0.5) == (R2.his_te > 0.5)).mean()
print(f"\nbinary(>0.5) agreement all: {agree:.3f}")
for hz, gg in R2.groupby("hz"):
    a = ((gg.our_te > 0.5) == (gg.his_te > 0.5)).mean()
    print(f"  {hz}Hz binary agreement: {a:.3f}  high_ours={ (gg.our_te>0.5).mean():.3f} high_his={(gg.his_te>0.5).mean():.3f}")
