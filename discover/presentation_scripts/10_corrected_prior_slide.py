"""Insert a 'Prior Baseline — Corrected for the Label Bug' slide right after the
'What Changed Since Last Week' opener. Reconciles three numbers for the prior
gaze QDA so adjacent slides don't look contradictory:
  published (slides 2-3, bugged) -> label-fixed same protocol (unscaled) ->
  our nested CV (scaled; the column the results slide compares against).
Idempotent. Run from repo root (after 06-09)."""
from pptx import Presentation
from pptx.util import Inches, Pt

SRC = "discover/2026_March_ViLearn_Planning.pptx"
TITLE = "Prior Baseline — Corrected for the Label Bug"
prs = Presentation(SRC)

for s in prs.slides:
    for sh in s.shapes:
        if sh.has_text_frame and sh.text_frame.text.strip() == TITLE:
            print("slide already present — nothing to do")
            raise SystemExit

slide = prs.slides.add_slide(prs.slide_layouts[7])  # Title Only
slide.shapes.title.text = TITLE

header = ["Prior gaze QDA", "Published (slides 2–3)", "Label-fixed (same protocol)", "Our nested CV (results)"]
rows = [
    ["All (20)", "0.60", "0.67", "0.73"],
    ["Dyads (8)", "0.63", "0.69", "0.77"],
    ["Triads (12)", "0.55", "0.59", "0.65"],
]
widths = [2.6, 2.9, 3.4, 3.0]
nrows, ncols = len(rows) + 1, len(header)
tbl = slide.shapes.add_table(nrows, ncols, Inches(0.4), Inches(1.5),
                             Inches(sum(widths)), Inches(0.45 * nrows)).table
for j, w in enumerate(widths):
    tbl.columns[j].width = Inches(w)
for j, h in enumerate(header):
    c = tbl.cell(0, j); c.text = h
    r = c.text_frame.paragraphs[0].runs[0]
    r.font.size = Pt(12); r.font.bold = True
for i, row in enumerate(rows, 1):
    for j, val in enumerate(row):
        c = tbl.cell(i, j); c.text = str(val)
        run = c.text_frame.paragraphs[0].runs[0]
        run.font.size = Pt(12)
        if j == 0:
            run.font.bold = True

tb = slide.shapes.add_textbox(Inches(0.4), Inches(4.1), Inches(12.2), Inches(2.8))
tf = tb.text_frame; tf.word_wrap = True
bullets = [
    "Same QDA, same gaze features — only the labels are harmonized (60 Hz time-stretch removed, 2-annotator mean).",
    "Label fix alone (col 2): published 0.60 → 0.67 All. The bug depressed published numbers ~0.05–0.08 (13/20 groups at 60 Hz; the 7 pure-90 Hz groups replicate exactly).",
    "Col 3 = our harmonized nested-CV protocol (scaled) — the column the results slide uses for the fair head-to-head vs our detector.",
    "None of these reach significance vs chance after Bonferroni (d up to 1.18 but n=8–20 underpowered).",
]
for i, text in enumerate(bullets):
    p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
    p.text = text
    for r in p.runs:
        r.font.size = Pt(13)

# move new slide to right after the 'What Changed' opener
target = None
for i, s in enumerate(prs.slides):
    if any(sh.has_text_frame and sh.text_frame.text.strip() == "What Changed Since Last Week"
           for sh in s.shapes):
        target = i + 1
        break
xml = prs.slides._sldIdLst
ids = list(xml)
new_el = ids[-1]
xml.remove(new_el)
xml.insert(target if target is not None else len(ids) - 1, new_el)

prs.save(SRC)
print(f"inserted '{TITLE}' at position {target};", len(prs.slides), "slides")
