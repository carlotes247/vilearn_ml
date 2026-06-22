"""Append an LLM-rubric slide: operationalize the study's OWN TE coding schema
(Low/Med/High markers) as a local-LLM prompt (Qwen3.6-27B), per 60s group-window,
German transcripts. Two deliverables: (a) zero-shot classifier (training-free),
(b) 4 rubric-axis marker features -> nested LOSO panel. Plus a generic German-BERT
(gbert) frozen-embedding baseline for contrast. Compares vs the audio/gaze priors.
Idempotent. Picks the 'think' LLM pass if present, else 'nothink'."""
import pandas as pd
from pptx import Presentation
from pptx.util import Inches, Pt

SRC = "discover/2026_March_ViLearn_Planning.pptx"
TITLE = "Operationalizing the TE Rubric with a Local LLM (zero-shot + features)"
prs = Presentation(SRC)
if any(any(sh.has_text_frame and sh.text_frame.text.strip() == TITLE for sh in s.shapes) for s in prs.slides):
    print("slide already present"); raise SystemExit

res = pd.read_csv("scratch/ideas/idea3_results.csv")
has_think = (res.source == "think").any()
ctx = pd.read_csv("scratch/te_adoption/a1_out/comprehensive_table.csv")


def _acc(src, mode, sp):
    r = res[(res.source == src) & (res["mode"] == mode) & (res.split == sp)]
    return f"{r.acc.iloc[0]:.2f}" if len(r) else "—"


def llm(mode, sp):
    # show "think / nothink" when both passes exist, else whichever is present
    if has_think:
        return f"{_acc('think', mode, sp)} / {_acc('nothink', mode, sp)}"
    return _acc("nothink", mode, sp)


def bert(sp):
    r = res[(res.source == "bert") & (res.split == sp)]
    return f"{r.acc.iloc[0]:.2f}" if len(r) else "—"


def prior(det, sp):
    r = ctx[(ctx.detector == det) & (ctx.split == sp)]
    return f"{r.acc.iloc[0]:.2f}" if len(r) else "—"


SPLITS = [("all", "All"), ("D", "Dyads"), ("T", "Triads")]
A1 = {"all": "All", "D": "D", "T": "T"}

s = prs.slides.add_slide(prs.slide_layouts[7])
s.shapes.title.text = TITLE
ll = "LLM {} (think/nothink)" if has_think else "LLM {}"
header = ["Split", "Majority", ll.format("zero-shot"), ll.format("markers"), "BERT (text)",
          "Prior gaze", "Audio", "Audio+Gaze"]
rows = []
for sp, lab in SPLITS:
    a = A1[sp]
    rows.append([lab, prior("dummy(majority)", a), llm("zeroshot", sp), llm("features", sp),
                 bert(sp), prior("GazexSpeaking", a), prior("audio", a),
                 prior("audio+GazexSpeaking", a)])
widths = [1.0, 1.0, 2.0, 1.9, 1.2, 1.3, 1.0, 1.3]
tbl = s.shapes.add_table(len(rows) + 1, len(header), Inches(0.4), Inches(1.4),
                         Inches(sum(widths)), Inches(0.45 * (len(rows) + 1))).table
for j, wd in enumerate(widths):
    tbl.columns[j].width = Inches(wd)
for j, h in enumerate(header):
    c = tbl.cell(0, j); c.text = h
    c.text_frame.paragraphs[0].runs[0].font.size = Pt(11); c.text_frame.paragraphs[0].runs[0].font.bold = True
for i, row in enumerate(rows, 1):
    for j, v in enumerate(row):
        c = tbl.cell(i, j); c.text = str(v)
        c.text_frame.paragraphs[0].runs[0].font.size = Pt(11)

tb = s.shapes.add_textbox(Inches(0.4), Inches(3.7), Inches(12.4), Inches(3.2))
tb.text_frame.word_wrap = True
bullets = [
    "Idea: the authors' TE coding schema (Low/Med/High markers) IS a set of linguistic criteria — operationalize it directly with an LLM instead of generic embeddings. Local Qwen3.6-27B (GDPR: on-prem, German transcripts), 60 s group-windows, clean 2-annotator labels.",
    "(a) Zero-shot — rubric + transcript -> TE level, NO training, NO gaze: ~0.74–0.76 All, significant every split (permutation p ≈ .0005). Matches the best multimodal fusion (0.75 All), beats audio alone (0.66). Think and non-think near-identical here.",
    "(b) Marker features — 4 rubric axes (task_relevance, content_depth, reasoning, connecting_ideas) -> same nested LOSO panel: thinking mode All 0.81 / Triads 0.84 — the highest of ANY method/modality here, and above non-think (0.76 / 0.82).",
    "Generic German BERT (gbert, frozen embeddings, no rubric) ties it (0.74–0.77) — the signal lives in the text; the rubric makes it training-free AND interpretable (per-axis scores).",
    "All group-level, nested LOSO + permutation. Two LLM passes reported (think / non-think); thinking helps the marker features, not zero-shot. NB: Qwen3.6 thinking-EFFORT levels (low/med) are inert on ollama — only on/off differ.",
]
for i, t in enumerate(bullets):
    p = tb.text_frame.paragraphs[0] if i == 0 else tb.text_frame.add_paragraph()
    p.text = t
    for run in p.runs:
        run.font.size = Pt(11)

prs.save(SRC)
print(f"LLM-rubric slide added (think+nothink={has_think}); {len(prs.slides)} slides")
