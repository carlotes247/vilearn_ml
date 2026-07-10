"""B1 — fusion table with BOTH gaze priors × LLM (adds AIxVR×LLM + audio×LLM).

Per split, nested-CV + permutation for:
  gaze_GxS, gaze_AIxVR, llm, audio,
  GxS+llm, AIxVR+llm, audio+llm,
  GxS+vad, AIxVR+vad,
  GxS+vad+llm, AIxVR+vad+llm
Lets each split pick its actual best prior instead of forcing GazexSpeaking
everywhere (slide 26). Writes a1_out/b1_fusion_priors_llm.csv (acc at 3 dp).

Run from repo root:  PYTHONPATH=discover/analysis_scripts python3 discover/analysis_scripts/b1_fusion_priors_llm.py
"""
import pandas as pd
import ab_common as ab

his = ab.build_features()
VAD, AXES, GXS = ab.AUDIO, ab.AXES, ab.GXS

rows = []
for sp in ("All", "D", "T"):
    sub = his if sp == "All" else his[his.grp == sp]
    aix = ab.AIXVR[sp]
    sets = {
        "gaze_GxS": GXS,
        "gaze_AIxVR": aix,
        "llm": AXES,
        "audio": VAD,
        "GxS+llm": GXS + AXES,
        "AIxVR+llm": aix + AXES,
        "audio+llm": VAD + AXES,
        "GxS+vad": GXS + VAD,
        "AIxVR+vad": aix + VAD,
        "GxS+vad+llm": GXS + VAD + AXES,
        "AIxVR+vad+llm": aix + VAD + AXES,
    }
    for name, feats in sets.items():
        b, a, p = ab.evaluate(sub, feats)
        rows.append({"split": sp, "features": name, "n": len(sub), "n_feat": len(feats),
                     "acc": round(a, 3), "perm_p": round(p, 4), "best_model": b})
        print(rows[-1], flush=True)

out = pd.DataFrame(rows)
out.to_csv("scratch/te_adoption/a1_out/b1_fusion_priors_llm.csv", index=False)
print("\nwrote a1_out/b1_fusion_priors_llm.csv")
piv = out.pivot_table(index="split", columns="features", values="acc", aggfunc="first")
cols = ["gaze_GxS", "gaze_AIxVR", "llm", "audio", "GxS+llm", "AIxVR+llm", "audio+llm",
        "GxS+vad", "AIxVR+vad", "GxS+vad+llm", "AIxVR+vad+llm"]
print(piv[cols].to_string())
print("\nBest feature-set per split (acc):")
for sp in ("All", "D", "T"):
    s = out[out.split == sp].sort_values("acc", ascending=False).iloc[0]
    print(f"  {sp}: {s.features} = {s.acc:.3f} ({s.best_model}, perm p={s.perm_p})")
