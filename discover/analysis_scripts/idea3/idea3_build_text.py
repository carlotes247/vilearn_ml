"""Idea3 step 1: build per-group-window transcript text on Carlos's 185-window grid
(same grid as the clean labels), tagged by speaker. Output window_text.csv.
Transcript files: data/discover/recording_<g>/transcript.<role>.helenrisack.csv
columns from,to,name(text),conf ; from/to are recording-relative ms (same as audio windowing).
"""
import os
import sys
import glob
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import a1_lib

his = a1_lib.load_data()  # group_name, group_type, seconds_interaction_window, our_te, y
FLOOR = pd.read_csv("data/group_names_with_time_floorlevel.csv", sep=";")
RT = pd.read_csv("data/recording_times_group_info.csv")
rec = dict(zip(RT["long_name"], pd.to_datetime(RT["start_recording"])))
bounds = {r["Group_Name"]: (pd.to_datetime(r["TS_Start_Interaction"]) - rec[r["Group_Name_Long"]]).total_seconds() * 1000.0
          for _, r in FLOOR.iterrows() if r["Group_Name_Long"] in rec}
HIS = pd.read_csv(a1_lib.HIS_CSV)
ROLE_TAG = {"p_blue": "blue", "p_green": "green", "p_red": "red"}

rows = []
for g, gg in his.groupby("group_name"):
    s0 = bounds[g]
    segs = []  # (from, to, tag, text)
    for path in glob.glob(f"data/discover/recording_{g}/transcript.*.helenrisack.csv"):
        role = path.split("transcript.")[1].split(".")[0]
        tag = ROLE_TAG.get(role, role)
        tr = pd.read_csv(path)
        for _, s in tr.iterrows():
            segs.append((float(s["from"]), float(s["to"]), tag, str(s["name"]).strip()))
    for idx in gg.index:
        sec = HIS.loc[idx, "seconds_interaction_window"]
        w0 = s0 + sec * 1000.0
        w1 = w0 + 60_000.0
        win = sorted([s for s in segs if s[1] > w0 and s[0] < w1], key=lambda x: x[0])
        text = "\n".join(f"[{tag}] {txt}" for _, _, tag, txt in win if txt)
        rows.append({"group_name": g, "group_type": gg.loc[idx, "group_type"],
                     "sec": sec, "our_te": gg.loc[idx, "our_te"], "y": gg.loc[idx, "y"],
                     "n_segments": len(win), "text": text})

out = pd.DataFrame(rows)
out.to_csv("scratch/ideas/idea3_window_text.csv", index=False)
print(f"wrote idea3_window_text.csv: {len(out)} windows, {(out.n_segments == 0).sum()} empty (silence)")
print(out[["group_name", "sec", "n_segments", "y"]].head(6).to_string())
