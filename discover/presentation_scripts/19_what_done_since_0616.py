"""Add a 'What Was Done Since the 16 June Meeting' opener slide (placed at index 1,
right after the title). Summarizes everything committed after the 14:00-15:00 Tue
06-16 meeting: methodology-review section, baseline redo, 3-prior+fusion, openSMILE,
and this week's idea3 LLM-rubric detector + gaze×LLM fusion. Idempotent. Reordering
the slide list is orphan-free (no part removal) so no repack needed."""
from pptx import Presentation
from pptx.util import Inches, Pt

SRC = "discover/2026_March_ViLearn_Planning.pptx"
TITLE = "What Was Done Since the 16 June Meeting"
prs = Presentation(SRC)
if any(any(sh.has_text_frame and sh.text_frame.text.strip() == TITLE for sh in s.shapes) for s in prs.slides):
    print("slide already present"); raise SystemExit

s = prs.slides.add_slide(prs.slide_layouts[7])
s.shapes.title.text = TITLE
tb = s.shapes.add_textbox(Inches(0.5), Inches(1.4), Inches(12.3), Inches(5.4))
tb.text_frame.word_wrap = True
groups = [
    ("NEW RESULT — LLM operationalizes the TE rubric (see results slides):", [
        "Took the study's own coding schema (Low/Med/High markers) and made it an LLM prompt — local Qwen3.6-27B, German transcripts, 60 s group-windows, clean labels. Zero-shot (NO training, NO gaze) ties the best multimodal fusion (~0.75 All).",
        "Marker features (4 rubric axes): thinking mode All 0.81 / Triads 0.84 — highest single channel. German BERT baseline ties → the signal is in the text.",
        "BEST CONFIG = Gaze×Speaking + LLM fusion: All 0.86 / Dyads 0.87 (perm p≈.003) — beats every prior, every modality, and the old gaze+vad fusion; makes Dyads gaze significant.",
    ]),
    ("Methodology & comparison:", [
        "Methodology-review section: significance flip-flop (permutation vs majority t-test), group-level vs long-format, open decisions — significance method still PENDING team decision.",
        "Our detector vs 3 consistent priors (dummy / AIxVR / GazexSpeaking) + audio fusion, both significance tests shown; slide 2 redone with the true majority baseline.",
        "openSMILE raw acoustics (88 eGeMAPS) vs v/a/d affect → affect wins; raw acoustics don't beat the 3-dim estimate.",
    ]),
    ("Pipeline:", [
        "Group-TE decoupled to per-group rows (Q2); idea1 stale labels retired (now points to the clean fusion table).",
    ]),
]
first = True
for head, items in groups:
    p = tb.text_frame.paragraphs[0] if first else tb.text_frame.add_paragraph()
    first = False
    p.text = head
    for run in p.runs:
        run.font.size = Pt(13); run.font.bold = True
    for it in items:
        q = tb.text_frame.add_paragraph(); q.text = "• " + it; q.level = 1
        for run in q.runs:
            run.font.size = Pt(11)

# move the just-added (last) slide to index 1 — orphan-free reorder, no repack
xs = prs.slides._sldIdLst
new = list(xs)[-1]
xs.remove(new); xs.insert(1, new)
prs.save(SRC)
print(f"'What Was Done' slide added at index 1; {len(prs.slides)} slides")
