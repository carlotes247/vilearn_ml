"""Idea3 step 2: score each window's transcript against the TE rubric with a LOCAL LLM.
Backend: ollama native /api/chat (qwen3.6:27b). Two modes, HF-recommended sampling each:
  - think   : reasoning ON  (temp 1.0, top_p 0.95, top_k 20, presence 0.0)
  - nothink : reasoning OFF (temp 0.7, top_p 0.80, top_k 20, presence 1.5)
seed=42 both for reproducibility. Output idea3_window_scores_<mode>.csv with 4 marker
axes + continuous te + predicted level. LOCAL ONLY (GDPR) — no cloud backend.

Usage:
  MODE=think   OLLAMA_MODEL=qwen3.6:27b python3 discover/analysis_scripts/idea3/idea3_score.py
  MODE=nothink OLLAMA_MODEL=qwen3.6:27b python3 discover/analysis_scripts/idea3/idea3_score.py
  LIMIT=5 MODE=think ... python3 discover/analysis_scripts/idea3/idea3_score.py   # smoke test
"""
import json
import os
import re
import urllib.request
import pandas as pd

MODE = os.environ.get("MODE", "think")
MODEL = os.environ.get("OLLAMA_MODEL", "qwen3.6:27b")
LIMIT = int(os.environ.get("LIMIT", "0"))  # 0 = all

# HF Qwen3.6 sampling recommendations differ per mode. seed=42 pins reproducibility.
PARAMS = {
    "think":   {"think": True,  "temperature": 1.0, "top_p": 0.95, "top_k": 20,
                "min_p": 0.0, "presence_penalty": 0.0, "repeat_penalty": 1.0},
    "nothink": {"think": False, "temperature": 0.7, "top_p": 0.80, "top_k": 20,
                "min_p": 0.0, "presence_penalty": 1.5, "repeat_penalty": 1.0},
}[MODE]

RUBRIC = """You rate the TASK ENGAGEMENT (TE) of a group's discussion in a 60-second window of a
collaborative VR learning task (German transcript, speakers tagged [blue]/[green]/[red]).

Coding schema (from the study):
- LOW TE: silence / no verbal or cognitive interaction; OR interaction WITHOUT content
  engagement (discussing non-task-related / off-topic things).
- MEDIUM TE: the group discusses the task on a SUPERFICIAL level — repeating the task,
  how to start, or stating own opinions/experiences WITHOUT reasoning, depth, or connecting
  to new ideas. e.g. "I think XY said...", "I like it, I'd try it out."
- HIGH TE: the group discusses the task on an ELABORATE level — including prior knowledge,
  making connections, hypotheticals, thinking out of the box. e.g. "It doesn't make sense to
  use VR this way, but if we set it up differently...", "Maybe not in normal classes, but in
  project days...".

Rate these axes from 0.0 to 1.0:
- task_relevance: how on-task the talk is (0 = silence/off-topic, 1 = fully on task)
- content_depth: superficial repetition/opinion (low) vs elaborate reasoning (high)
- reasoning_present: presence of justification/reasoning ("because", weighing options)
- connecting_ideas: linking to prior knowledge / other concepts / hypotheticals
Then give te_continuous (0.0-1.0 overall task engagement) and predicted_level (low/medium/high).
Reply with ONLY a JSON object, no prose:
{"task_relevance":_, "content_depth":_, "reasoning_present":_, "connecting_ideas":_,
 "te_continuous":_, "predicted_level":"low|medium|high"}"""


def parse_json(s):
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL)  # strip reasoning if it leaks
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    return json.loads(m.group(0)) if m else None


def score(text):
    opts = {k: v for k, v in PARAMS.items() if k != "think"}
    opts["seed"] = 42
    body = {"model": MODEL, "format": "json", "stream": False,
            "think": PARAMS["think"], "options": opts,
            "messages": [{"role": "system", "content": RUBRIC},
                         {"role": "user", "content": "Transcript:\n" + (text or "(silence — no speech)")}]}
    req = urllib.request.Request("http://localhost:11434/api/chat",
                                 data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=600).read())
    return parse_json(r["message"]["content"])


df = pd.read_csv("scratch/ideas/idea3_window_text.csv")
if LIMIT:
    df = df.head(LIMIT)
rows = []
for i, r in df.iterrows():
    try:
        s = score("" if r.n_segments == 0 else r.text) or {}
    except Exception as e:
        s = {"error": str(e)[:80]}
    s.update({"group_name": r.group_name, "sec": r.sec, "y": r.y, "our_te": r.our_te})
    rows.append(s)
    dst = f"scratch/ideas/idea3_window_scores_{MODE}.csv"
    if i % 20 == 0:
        print(f"{i}/{len(df)} [{MODE}]", flush=True)
        pd.DataFrame(rows).to_csv(dst, index=False)   # incremental flush (crash-safe)
out = pd.DataFrame(rows)
out.to_csv(dst, index=False)
n_ok = out.get("te_continuous", pd.Series(dtype=float)).notna().sum()
print(f"wrote {dst} ({MODE}/{MODEL}); {n_ok}/{len(out)} scored")
