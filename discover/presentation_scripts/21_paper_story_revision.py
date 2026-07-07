"""D2 — paper-story revision of the PRESENTED scratch deck (2026-07-01).

Targets scratch/2026_March_ViLearn_Planning.pptx (the corrected+presented deck; a
backup _presented_2026-07-01.pptx exists). Idempotent.

Changes:
  (1) Slide 21 'Detector vs 3 Priors + Fusion': acc column -> 3 decimals (kills the
      audio vs audio+AIxVR duplication look). Numbers from comprehensive_table.csv.
  (2) Slide 26 'Best Config: Gaze×Speaking × LLM Rubric': rebuild table with BOTH gaze
      priors × LLM + audio×LLM + a Best column; Triads winner flips to audio+LLM 0.864.
      Retitled. Data from b1_fusion_priors_llm.csv. Absorbs new slide N1.
  (3) Append N2 hardware-tiered recommendation, N3 Triads mechanism, N4 60/90 robustness.

Run:  python3 discover/presentation_scripts/21_paper_story_revision.py
"""
import pandas as pd
from pptx import Presentation
from pptx.util import Inches, Pt

SRC = "scratch/2026_March_ViLearn_Planning.pptx"
prs = Presentation(SRC)

comp = pd.read_csv("scratch/te_adoption/a1_out/comprehensive_table.csv")
b1 = pd.read_csv("scratch/te_adoption/a1_out/b1_fusion_priors_llm.csv")
b2 = pd.read_csv("scratch/te_adoption/a1_out/b2_rate_split.csv")


def find_slide(prefix):
    for s in prs.slides:
        for sh in s.shapes:
            if sh.has_text_frame and sh.text_frame.text.strip().startswith(prefix):
                return s
    return None


def set_cell(cell, text, size=Pt(9), bold=False):
    cell.text = str(text)
    r = cell.text_frame.paragraphs[0].runs[0]
    r.font.size = size
    r.font.bold = bold


def pfmt(p):
    p = float(p)
    return "<.001" if p < 0.001 else f"{p:.3f}"


def acc1(df, split, feat):
    return float(df[(df.split == split) & (df.features == feat)].acc.iloc[0])


def both(df, split, feat):
    r = df[(df.split == split) & (df.features == feat)].iloc[0]
    return f"{pfmt(r.perm_p)} / {pfmt(r.p_ttest_maj)}"


# ---------- (1) slide 21 acc -> 3dp ----------
s21 = find_slide("Detector vs 3 Priors")
if s21:
    tbl = next(sh.table for sh in s21.shapes if sh.has_table)
    look = {(r.split, r.detector): r.acc for _, r in comp.iterrows()}
    n = 0
    for i in range(1, len(tbl.rows)):
        sp = tbl.cell(i, 0).text.strip()
        det = tbl.cell(i, 1).text.strip()
        if det.startswith("dummy"):
            continue
        key = (sp, det)
        if key in look:
            sz = tbl.cell(i, 2).text_frame.paragraphs[0].runs[0].font.size or Pt(9)
            set_cell(tbl.cell(i, 2), f"{look[key]:.3f}", size=sz)
            n += 1
    print(f"slide 21: rewrote {n} acc cells to 3dp")
else:
    print("slide 21 NOT found — skipped")


# ---------- (2) slide 26 rebuild ----------
NEW26_TITLE = "Best Detector per Interaction Size (gaze / audio × LLM fusion)"
s26 = find_slide("Best Config: Gaze")
if s26 is None:
    s26 = find_slide("Best Detector per Interaction Size")  # idempotent re-run
if s26:
    title_sh = s26.shapes[0]
    for sh in list(s26.shapes)[1:]:  # drop old table + bullets, keep title
        sh._element.getparent().remove(sh._element)
    title_sh.text_frame.text = NEW26_TITLE
    header = ["Split", "Gaze+LLM", "AIxVR+LLM", "Audio+LLM", "LLM", "Best (perm / t-test)"]
    feats = ["GxS+llm", "AIxVR+llm", "audio+llm", "llm"]
    winner = {"All": ("GxS+llm", "Gaze+LLM"), "D": ("GxS+llm", "Gaze+LLM"), "T": ("audio+llm", "Audio+LLM")}
    lab = {"All": "Overall", "D": "Dyads", "T": "Triads"}
    widths = [1.3, 1.7, 1.7, 1.7, 1.2, 3.0]
    rows = []
    for sp in ["All", "D", "T"]:
        wf, wname = winner[sp]
        best = f"{wname} {acc1(b1, sp, wf):.3f}  ({both(b1, sp, wf)})"
        rows.append([lab[sp]] + [f"{acc1(b1, sp, f):.3f}" for f in feats] + [best])
    tbl = s26.shapes.add_table(len(rows) + 1, len(header), Inches(0.4), Inches(1.3),
                               Inches(sum(widths)), Inches(0.5 * (len(rows) + 1))).table
    for j, wd in enumerate(widths):
        tbl.columns[j].width = Inches(wd)
    for j, h in enumerate(header):
        set_cell(tbl.cell(0, j), h, size=Pt(12), bold=True)
    for i, (sp, row) in enumerate(zip(["All", "D", "T"], rows), 1):
        wname = winner[sp][1]
        for j, v in enumerate(row):
            is_win = (header[j] == wname)
            set_cell(tbl.cell(i, j), v, size=Pt(12), bold=is_win or j in (0, 5))
    bullets = [
        "Per-split best detector (nested LOSO + permutation + one-sample t-test, clean 2-annotator labels, 185 group-windows):",
        "  • Overall & Dyads → Gaze(GazexSpeaking)+LLM: 0.860 / 0.869 (both perm≈.003, t-test<.001). Gaze and rubric-text are complementary.",
        "  • Triads → Audio+LLM 0.864 (gaze-free) BEATS every gaze fusion (Gaze+LLM 0.826). Even LLM alone (0.836) tops gaze fusion here.",
        "WHY: in dyads the gaze target is unambiguous (one partner) so gaze-on-speaker is informative; in triads gaze diffuses (2 targets, mutual gaze rare) and participants focus on speaking, not looking — audio/text carry the signal. See mechanism slide.",
        "AIxVR gaze prior is weak throughout (AIxVR+LLM only competitive on Triads, where any gaze ≈ noise). Triple gaze+vad+LLM in appendix.",
    ]
    tb = s26.shapes.add_textbox(Inches(0.4), Inches(3.5), Inches(12.5), Inches(3.4))
    tb.text_frame.word_wrap = True
    for i, t in enumerate(bullets):
        p = tb.text_frame.paragraphs[0] if i == 0 else tb.text_frame.add_paragraph()
        p.text = t
        for run in p.runs:
            run.font.size = Pt(11)
    print(f"slide 26 rebuilt -> '{NEW26_TITLE}'")
else:
    print("slide 26 NOT found — skipped")


# ---------- append helper ----------
def add_table_slide(title, header, rows, widths, bold_col=None, bold_rows_fn=None,
                    bullets=None, note=None, ncap=Pt(12), rcap=Pt(11)):
    if find_slide(title):
        print(f"'{title[:40]}...' already present — skip")
        return
    s = prs.slides.add_slide(prs.slide_layouts[7])
    s.shapes.title.text = title
    tbl = s.shapes.add_table(len(rows) + 1, len(header), Inches(0.4), Inches(1.3),
                             Inches(sum(widths)), Inches(0.5 * (len(rows) + 1))).table
    for j, wd in enumerate(widths):
        tbl.columns[j].width = Inches(wd)
    for j, h in enumerate(header):
        set_cell(tbl.cell(0, j), h, size=ncap, bold=True)
    for i, row in enumerate(rows, 1):
        bset = bold_rows_fn(i - 1) if bold_rows_fn else set()
        for j, v in enumerate(row):
            set_cell(tbl.cell(i, j), v, size=rcap, bold=(j in bset) or (bold_col == j))
    y = 1.3 + 0.5 * (len(rows) + 1) + 0.3
    if bullets:
        tb = s.shapes.add_textbox(Inches(0.4), Inches(y), Inches(12.5), Inches(6.8 - y))
        tb.text_frame.word_wrap = True
        for i, t in enumerate(bullets):
            p = tb.text_frame.paragraphs[0] if i == 0 else tb.text_frame.add_paragraph()
            p.text = t
            for run in p.runs:
                run.font.size = Pt(11)
    if note:
        nb = s.shapes.add_textbox(Inches(0.4), Inches(6.7), Inches(12.5), Inches(0.6))
        nb.text_frame.word_wrap = True
        nb.text_frame.paragraphs[0].text = note
        nb.text_frame.paragraphs[0].runs[0].font.size = Pt(9)
        nb.text_frame.paragraphs[0].runs[0].font.italic = True
    print(f"appended '{title[:45]}'")


# ---------- N2 hardware-tiered ----------
def g(sp, f):
    return f"{acc1(b1, sp, f):.3f}"


add_table_slide(
    "Deployment Recommendation by Available Hardware",
    ["Hardware available", "Detector", "Overall", "Dyads", "Triads"],
    [["Eye-tracker (+ mic)", "Gaze+LLM", g("All", "GxS+llm"), g("D", "GxS+llm"), g("T", "GxS+llm")],
     ["Microphone only", "Audio+LLM", g("All", "audio+llm"), g("D", "audio+llm"), g("T", "audio+llm")],
     ["Transcript only", "LLM rubric", g("All", "llm"), g("D", "llm"), g("T", "llm")]],
    [3.0, 2.0, 1.6, 1.6, 1.6],
    bold_rows_fn=lambda i: {2, 3, 4} if i == 0 else ({2, 3} if i == 2 else {4}),
    bullets=[
        "Practical takeaway — best deployable detector scales with sensing, not lab gear:",
        "  • With an eye-tracker → Gaze+LLM: best Overall (0.860) and Dyads (0.869).",
        "  • Microphone only (no eye-tracker) → Audio+LLM: best Triads (0.864), and within ~0.05 of the eye-tracker rig Overall.",
        "  • Nothing but a transcript → the training-free LLM rubric alone already reaches 0.81 Overall / 0.84 Triads.",
        "All configs use the same local LLM rubric (Qwen3.6-27B, zero-shot); only the sensor-derived channel changes. Nested LOSO + permutation + t-test, clean labels.",
    ],
    note="Bold = recommended detector for that hardware tier. Accuracies from b1_fusion_priors_llm.csv (group-level, 185 windows).")


# ---------- N3 Triads mechanism (text) ----------
if not find_slide("Why Gaze Helps Dyads but Not Triads"):
    s = prs.slides.add_slide(prs.slide_layouts[7])
    s.shapes.title.text = "Why Gaze Helps Dyads but Not Triads"
    tb = s.shapes.add_textbox(Inches(0.5), Inches(1.5), Inches(12.3), Inches(5.0))
    tb.text_frame.word_wrap = True
    pts = [
        "Empirical pattern (fusion table): gaze is decisive for DYADS, useless for TRIADS.",
        "  • Dyads: Gaze+LLM 0.869  >  Audio+LLM 0.733  >  LLM 0.753 — gaze adds a lot.",
        "  • Triads: Audio+LLM 0.864  >  LLM 0.836  >  Gaze+LLM 0.826 — gaze adds nothing (even hurts vs text alone).",
        "  • Gaze-alone: Dyads 0.83 (sig) vs Triads 0.70 (weakest prior).",
        "",
        "Interpretation — the gaze-on-speaker signal degrades with group size:",
        "  • In a DYAD the gaze target is unambiguous: there is exactly one other person, so 'looking at the speaker' is a clean, informative cue.",
        "  • In a TRIAD gaze diffuses across two possible targets; mutual gaze is rarer and turn-taking is faster. Participants engage by TALKING, not by looking — so eye-gaze carries little task-engagement signal.",
        "  • Speech affect (v/a/d) and rubric-text, by contrast, are unaffected by group size → Audio+LLM wins triads with no eye-tracker at all.",
        "",
        "Consequence for deployment: eye-tracking pays off only for pairs; for larger groups a microphone + LLM rubric is both cheaper and more accurate.",
    ]
    for i, t in enumerate(pts):
        p = tb.text_frame.paragraphs[0] if i == 0 else tb.text_frame.add_paragraph()
        p.text = t
        for run in p.runs:
            run.font.size = Pt(13)
    print("appended 'Why Gaze Helps Dyads but Not Triads'")
else:
    print("N3 mechanism slide already present — skip")


# ---------- N4 60/90 robustness (appendix) ----------
def b2acc(sub, det):
    r = b2[(b2.subset == sub) & (b2.detector == det)]
    return f"{float(r.acc.iloc[0]):.3f}" if len(r) else "—"


add_table_slide(
    "Appendix — Robustness to Annotation Rate (60 Hz vs 90 Hz)",
    ["Subset (n groups)", "audio", "GxS", "LLM", "GxS+LLM", "audio+LLM"],
    [[f"{sub} ({int(b2[b2.subset==sub].n_groups.iloc[0])})"] +
     [b2acc(sub, d) for d in ["audio", "GxS", "llm", "GxS+llm", "audio+llm"]]
     for sub in ["60Hz", "90Hz", "60Hz-T", "90Hz-D"]],
    [3.0, 1.5, 1.5, 1.5, 1.7, 1.9],
    bold_col=3,  # LLM column — the invariant one
    bullets=[
        "Split groups by native annotation rate (12× 60 Hz / 8× 90 Hz) to check for a residual of the time-stretch label bug.",
        "CONFOUND: 60 Hz groups are mostly triads (10/12), 90 Hz mostly dyads (6/8) — rate cannot be separated from group size here, so read cautiously.",
        "The LLM rubric detector is RATE- AND TYPE-INVARIANT: 0.81 / 0.80 / 0.82 / 0.76 across all four cells (bold column). The text signal is untouched by annotation rate.",
        "Gaze (GxS) tracks group TYPE, not rate (strong on 90 Hz≈dyads 0.81, weak on 60 Hz≈triads 0.60); audio is weakest and fails on dyads (90 Hz-D 0.58, ns).",
        "Conclusion: under clean 2-annotator labels there is no evidence of a rate artifact; the apparent 60/90 gap in gaze is the dyad/triad confound.",
    ],
    note="Group-level nested LOSO; per-cell permutation + t-test in b2_rate_split.csv.")


prs.save(SRC)
print(f"\nsaved {SRC}: {len(prs.slides)} slides")
