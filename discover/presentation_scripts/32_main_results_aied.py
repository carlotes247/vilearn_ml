"""Slide 15 (Main Results) — add the linguistic+AIED fusion rows (2026-07-10).

Slide predates the AIED baseline: Overall showed ling+GxS 0.786 as the combined
row while ling+AIED scores 0.794 (statistical tie, paired p=.83, higher d);
Triads ling+AIED is the d-strongest fusion (1.03). One row per split inserted
after the existing combined row (XML tr copy — preserves manual bolding);
tie/story bullet appended add-once. Cells from c2_paper_table.csv.

Run from repo root:  python3 discover/presentation_scripts/32_main_results_aied.py
"""
import os
import shutil
from copy import deepcopy
import pandas as pd
from pptx import Presentation
from pptx.util import Pt

SRC = "scratch/2026_March_ViLearn_Planning.pptx"
BAK = "scratch/2026_March_ViLearn_Planning_pre-s15aied_2026-07-10.pptx"
if not os.path.exists(BAK):
    shutil.copyfile(SRC, BAK)
    print(f"backup -> {BAK}")

prs = Presentation(SRC)
c2 = pd.read_csv("scratch/te_adoption/a1_out/c2_paper_table.csv")


def r1(sp, det):
    return c2[(c2.split == sp) & (c2.detector == det) & (c2["rank"] == 1)].iloc[0]


def pfmt(p):
    p = float(p)
    return "<.001" if p < 0.001 else f"{p:.3f}"


def find_slide(prefix):
    for s in prs.slides:
        for sh in s.shapes:
            if sh.has_text_frame and sh.text_frame.text.strip().startswith(prefix):
                return s
    return None


def insert_row_after(table, anchor_idx, texts):
    tbl = table._tbl
    src = tbl.tr_lst[anchor_idx]
    new = deepcopy(src)
    src.addnext(new)
    row_idx = anchor_idx + 1
    tmpl = table.cell(anchor_idx, 0).text_frame.paragraphs[0].runs
    size = tmpl[0].font.size if tmpl and tmpl[0].font.size else Pt(9)
    for j, t in enumerate(texts):
        cell = table.cell(row_idx, j)
        cell.text = str(t)
        for run in cell.text_frame.paragraphs[0].runs:
            run.font.size = size
            run.font.bold = False


s = find_slide("Main Results — Group-Level TE Detection")
if s is None:
    raise SystemExit("Main Results slide not found")
tbl = next(sh.table for sh in s.shapes if sh.has_table)

if any("linguistic+AIED" in tbl.cell(i, 1).text for i in range(len(tbl.rows))):
    print("AIED rows already present, skip table")
else:
    # anchor = the existing combined row of each split (col0=split label on it)
    ANCHORS = {"All": ("Overall", "combined: linguistic+GxS"),
               "D": ("Dyads", "combined: linguistic+GxS"),
               "T": ("Triads", "combined: audio+linguistic")}
    for sp in ("All", "D", "T"):
        lab, anchor_txt = ANCHORS[sp]
        ai = next((i for i in range(len(tbl.rows))
                   if tbl.cell(i, 0).text.strip() == lab
                   and tbl.cell(i, 1).text.strip() == anchor_txt), None)
        if ai is None:
            print(f"WARN: anchor '{anchor_txt}' ({lab}) not found")
            continue
        r = r1(sp, "linguistic+AIED")
        insert_row_after(tbl, ai, [lab, "combined: linguistic+AIED", r.model,
                                   f"{r.acc_mean:.3f} ± {r.acc_std:.3f}",
                                   f"{r.cohen_d_maj:.2f}", pfmt(r.perm_p),
                                   pfmt(r.p_ttest_maj)])
        print(f"{lab}: linguistic+AIED row inserted")

MARK = "paired p=.83"
tb = next((sh for sh in s.shapes if sh.has_text_frame
           and "Both significance tests" in sh.text_frame.text), None)
if tb is not None and MARK not in tb.text_frame.text:
    p = tb.text_frame.add_paragraph()
    p.text = ("AIED fusions added 2026-07-10: Overall linguistic+AIED 0.794 and "
              "linguistic+GxS 0.786 are statistically tied (paired p=.83) — the text "
              "backbone carries the fusion, the exact gaze set is secondary. Triads "
              "headline stays audio+linguistic (gaze-free); linguistic+AIED is the "
              "d-strongest triad fusion (1.03) at comparable accuracy.")
    for run in p.runs:
        run.font.size = Pt(10)
    print("tie bullet appended")
else:
    print("tie bullet already present / textbox not found")

prs.save(SRC)
print(f"saved {SRC}")
