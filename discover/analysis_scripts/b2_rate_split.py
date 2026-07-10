"""B2 — all-60Hz vs all-90Hz experiment (native annotation rate).

Ties to the time-stretch bug: does the split of groups by native rate change
detector accuracy under OUR clean labels? Per rate-subset, nested-CV + permutation
for the key detectors. CONFOUND (printed): 60Hz is mostly triads, 90Hz mostly
dyads — so rate ~ group-size here; report cell counts and don't over-read All-level.

Run from repo root:  PYTHONPATH=discover/analysis_scripts python3 discover/analysis_scripts/b2_rate_split.py
"""
import pandas as pd
import ab_common as ab

his = ab.build_features()
VAD, AXES, GXS = ab.AUDIO, ab.AXES, ab.GXS

print("group composition by rate x type:")
comp = his.groupby("group_name").agg(rate=("rate", "first"), grp=("grp", "first")).reset_index()
print(comp.groupby(["rate", "grp"]).size().to_string())
print("windows by rate x type:")
print(his.groupby(["rate", "grp"]).size().to_string())
print(f"\nCONFOUND: 60Hz={sorted(comp[comp.rate=='60Hz'].grp.value_counts().to_dict().items())} "
      f"90Hz={sorted(comp[comp.rate=='90Hz'].grp.value_counts().to_dict().items())}\n")

DETECTORS = {"audio": VAD, "GxS": GXS, "llm": AXES, "GxS+llm": GXS + AXES,
             "audio+llm": VAD + AXES}

rows = []
# rate subsets at All level, plus the only viable within-type cells (>=4 groups)
subsets = {"60Hz": his[his.rate == "60Hz"], "90Hz": his[his.rate == "90Hz"],
           "60Hz-T": his[(his.rate == "60Hz") & (his.grp == "T")],
           "90Hz-D": his[(his.rate == "90Hz") & (his.grp == "D")]}
for label, sub in subsets.items():
    ng = sub.group_name.nunique()
    if ng < 4:
        print(f"skip {label}: only {ng} groups")
        continue
    maj = max(sub.y.mean(), 1 - sub.y.mean())
    for name, feats in DETECTORS.items():
        b, a, p = ab.evaluate(sub, feats)
        rows.append({"subset": label, "n_groups": ng, "n_win": len(sub), "majority": round(maj, 3),
                     "detector": name, "acc": round(a, 3), "perm_p": round(p, 4), "best_model": b})
        print(rows[-1], flush=True)

out = pd.DataFrame(rows)
out.to_csv("scratch/te_adoption/a1_out/b2_rate_split.csv", index=False)
print("\nwrote a1_out/b2_rate_split.csv")
piv = out.pivot_table(index="subset", columns="detector", values="acc", aggfunc="first")
print(piv.to_string())
