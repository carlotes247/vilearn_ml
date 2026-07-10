"""Verify the 1.5x helen time-stretch bug in Carlos's TE for 60 Hz groups.

For a 60 Hz group: simulate his processing two ways (helen treated as 60 Hz =
the bug; helen treated correctly as 90 Hz) and correlate each against his saved
task_engagement90Hz_avg_all.csv. Whichever matches reveals what his file holds.
"""
import numpy as np
import pandas as pd

GROUPS = ["dyad_10", "triad_05", "triad_11", "dyad_03"]  # last one is 90Hz control
ANNO = "data/annotations"


def load_raw(path):
    df = pd.read_csv(path, sep=";", names=["task_eng", "conf"])
    v = pd.to_numeric(df["task_eng"], errors="coerce").fillna(0).to_numpy()  # his fillna(0)
    return v


def interp_to_90(v, assumed_freq):
    sec = np.arange(len(v)) / assumed_freq
    new = np.arange(sec.min(), sec.max(), 1 / 90)
    return np.interp(new, sec, v)


for g in GROUPS:
    base = f"{ANNO}/recording_{g}"
    try:
        helen = load_raw(f"{base}/group.task engagement.helenrisack.annotation~")
    except FileNotFoundError:
        helen = load_raw(f"{base}/task engagement60Hz.group.helenrisack.annotation~")
    try:
        carlos = load_raw(f"{base}/task engagement60Hz.group.carlosgonzalez.annotation~")
        c90 = interp_to_90(carlos, 60)
    except FileNotFoundError:
        c90 = load_raw(f"{base}/task engagement.group.carlosgonzalez.annotation~")

    saved = pd.read_csv(f"{base}/task_engagement90Hz_avg_all.csv")["task_eng"].to_numpy()

    def avg(a, b):
        n = min(len(a), len(b))
        return (a[:n] + b[:n]) / 2

    sims = {}
    if g == "dyad_03":  # 90Hz group: no interpolation in his path
        sims["correct(90Hz)"] = avg(helen, c90)
    else:
        sims["bug(helen as 60Hz)"] = avg(interp_to_90(helen, 60), c90)
        sims["correct(helen 90Hz)"] = avg(helen, c90)

    out = [f"{g}: saved n={len(saved)}"]
    for name, sim in sims.items():
        n = min(len(sim), len(saved))
        c = np.corrcoef(sim[:n], saved[:n])[0, 1]
        out.append(f"{name}: n={len(sim)}, corr={c:.4f}")
    print("  |  ".join(out))
