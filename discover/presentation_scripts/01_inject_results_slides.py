import shutil
from pptx import Presentation
from pptx.util import Inches, Pt

SRC = "discover/2026_March_ViLearn_Planning.pptx"
shutil.copy(SRC, SRC.replace(".pptx", "_backup.pptx"))

prs = Presentation(SRC)
BULLET = prs.slide_layouts[1]   # Title and Content
TITLE_ONLY = prs.slide_layouts[5]


def add_bullets(title, bullets):
    s = prs.slides.add_slide(BULLET)
    s.shapes.title.text = title
    body = s.placeholders[1].text_frame
    body.word_wrap = True
    for i, (text, lvl) in enumerate(bullets):
        p = body.paragraphs[0] if i == 0 else body.add_paragraph()
        p.text = text
        p.level = lvl
        for r in p.runs:
            r.font.size = Pt(18 if lvl == 0 else 16)
    return s


def add_table(title, header, rows, note=None, col_w=None):
    s = prs.slides.add_slide(TITLE_ONLY)
    s.shapes.title.text = title
    nrows, ncols = len(rows) + 1, len(header)
    left, top, width = Inches(0.5), Inches(1.6), Inches(12.19 - 1.0)
    height = Inches(0.35 * nrows)
    tbl = s.shapes.add_table(nrows, ncols, left, top, width, height).table
    if col_w:
        for j, w in enumerate(col_w):
            tbl.columns[j].width = Inches(w)
    for j, h in enumerate(header):
        c = tbl.cell(0, j)
        c.text = h
        c.text_frame.paragraphs[0].runs[0].font.size = Pt(13)
        c.text_frame.paragraphs[0].runs[0].font.bold = True
    for i, row in enumerate(rows, 1):
        for j, val in enumerate(row):
            c = tbl.cell(i, j)
            c.text = str(val)
            c.text_frame.paragraphs[0].runs[0].font.size = Pt(12)
    if note:
        tb = s.shapes.add_textbox(left, Inches(1.6) + height + Inches(0.15), width, Inches(0.6))
        p = tb.text_frame.paragraphs[0]
        p.text = note
        p.runs[0].font.size = Pt(12)
        p.runs[0].font.italic = True
    return s


# Slide 10 — Setup
add_bullets("Linguistic TE Detector — Setup (DISCOVER)", [
    ("Data: 20 floorlevel groups (8 dyads, 12 triads), interaction-time only", 0),
    ("Unit: 60 s windows (best granularity, see next); target: task engagement high/low at 0.5", 0),
    ("Features (no gaze):", 0),
    ("Linguistic: word_count, words_per_second, avg_word_length, question/statement_rate, segment_count, speaking_seconds", 1),
    ("Affective: sentiment, arousal, dominance, valence; plus speaking activity", 1),
    ("Eval: leave-one-session-out CV; model sweep (QDA, SVM, NB, kNN, RF, LogReg) vs uniform baseline", 0),
    ("Reported on dyads (D) / triads (T) / all, matching prior 3-fold reporting", 0),
])

# Slide 11 — Group TE results vs ICMI
add_table("Group Task-Engagement Detection — Results vs ICMI",
          ["Split", "Our best model", "Acc", "F1-macro", "ICMI SVM Acc", "Baseline Acc"],
          [["Dyads", "Naive Bayes", "0.72", "0.64", "0.66", "0.53"],
           ["Triads", "Random Forest", "0.87", "0.82", "0.63", "0.43"],
           ["All", "Random Forest", "0.83", "0.76", "0.62", "0.45"]],
          note="Linguistic + affective >= submitted GazexSpeaking, without gaze. "
               "vs current QDA gaze detector (dyads 77%): our QDA dyads 74.7% - comparable, different modality.",
          col_w=[1.3, 2.6, 1.0, 1.4, 1.8, 1.9])

# Slide 12 — Granularity
add_table("Why 60 s Windows (Granularity)",
          ["Granularity", "Group TE (All): Acc", "F1-macro"],
          [["Transcript segment", "0.59", "0.56"],
           ["60 s window", "0.83", "0.76"]],
          note="Aggregating to 60 s removes per-utterance noise -> large gain. 1 Hz frames add nothing. "
               "60 s also matches the prior-work window size.",
          col_w=[4.0, 3.5, 3.0])

# Slide 13 — Feature contribution
add_bullets("What Drives Task Engagement? (feature contribution)", [
    ("Group TE: group affect dominates", 0),
    ("valence (positive -> high TE), dominance (negative), arousal", 1),
    ("Individual engagement: more distributed", 0),
    ("own speaking activity + word_count / words_per_second + affect", 1),
    ("Method: standardized logistic-regression coefficients + Random-Forest importance (LOSO)", 0),
    ("Linguistic features explain individual engagement more than group TE", 0),
])

# Slide 14 — Embeddings
add_bullets("Do Audio / Text Embeddings Help?", [
    ("Tested base features + openSMILE + emoW2V + sentiment embeddings (1880 dims -> PCA)", 0),
    ("No improvement - embeddings on top of base features hurt cross-session F1", 0),
    ("High-dimensional streams add noise that degrades leave-one-session-out generalization", 1),
    ("Conclusion: keep the detector light - simple interpretable features win", 0),
])

# Slide 15 — RQ answers & next steps
add_bullets("Answers to Research Questions & Next Steps", [
    ("Linguistic-only detects TE? Yes - group TE F1 0.76 (All), beats gaze x speaking baseline", 0),
    ("Strongest features? speaking + word rate (individual); group affect (group TE)", 0),
    ("Multimodal (linguistic + embeddings)? No gain yet", 0),
    ("Open: individual engagement still hard (~0.70); class imbalance (74% high); per-segment text RQ; tune threshold for balanced recall", 0),
    ("1-year vision: real-time per-window TE detection from transcript + audio affect, generalizing across sessions, feeding adaptive VR feedback", 0),
])

prs.save(SRC)
print("saved", SRC, "now", len(prs.slides), "slides")
