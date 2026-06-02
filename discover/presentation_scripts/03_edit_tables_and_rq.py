from pptx import Presentation
from pptx.util import Inches, Pt

SRC = "discover/2026_March_ViLearn_Planning.pptx"
prs = Presentation(SRC)


def add_table(slide, header, rows, top, widths, fontsz=12):
    nrows, ncols = len(rows) + 1, len(header)
    left, width = Inches(0.5), Inches(sum(widths))
    height = Inches(0.34 * nrows)
    tbl = slide.shapes.add_table(nrows, ncols, left, Inches(top), width, height).table
    for j, w in enumerate(widths):
        tbl.columns[j].width = Inches(w)
    for j, h in enumerate(header):
        c = tbl.cell(0, j); c.text = h
        r = c.text_frame.paragraphs[0].runs[0]; r.font.size = Pt(fontsz); r.font.bold = True
    for i, row in enumerate(rows, 1):
        for j, val in enumerate(row):
            c = tbl.cell(i, j); c.text = str(val)
            c.text_frame.paragraphs[0].runs[0].font.size = Pt(fontsz)
    return tbl


def add_note(slide, text, top):
    tb = slide.shapes.add_textbox(Inches(0.5), Inches(top), Inches(11.0), Inches(1.0))
    tb.text_frame.word_wrap = True
    p = tb.text_frame.paragraphs[0]; p.text = text
    p.runs[0].font.size = Pt(11); p.runs[0].font.italic = True


# --- Slide 12 (idx 11): Granularity — rebuild table to add 1 Hz row ---
s = prs.slides[11]
for sh in list(s.shapes):
    if sh.has_table:
        sh._element.getparent().remove(sh._element)
add_table(s,
          ["Granularity", "Group TE acc (best model)", "Note"],
          [["Transcript segment", "0.59", "per-utterance, noisy"],
           ["1 Hz frame", "0.70", "downsampled frames"],
           ["60 s window", "0.82", "best — matches prior work"]],
          top=1.6, widths=[3.5, 3.8, 3.6])
add_note(s, "Best model per granularity (group TE, LOSO). Coarser aggregation -> less noise; 60 s wins. "
            "Features differ slightly by scale (segment & 60 s include linguistic; 1 Hz frame = affective+speaking).", top=3.5)

# --- Slide 13 (idx 12): Feature contribution — add corroborating table ---
s = prs.slides[12]
add_table(s,
          ["Feature (Group TE)", "LogReg coef", "RF importance"],
          [["valence", "+1.12", "0.21"],
           ["dominance", "-1.33", "0.18"],
           ["arousal", "-0.05", "0.13"],
           ["speaking_role", "-0.09", "0.06"]],
          top=4.1, widths=[3.6, 2.4, 2.6])
add_note(s, "Group TE driven by audio affect (valence +, dominance -). "
            "Individual engagement: speaking_role (+), valence, word_count (+), words_per_second (+).", top=6.2)

# --- Slide 14 (idx 13): Embeddings — add base-vs-embeddings table ---
s = prs.slides[13]
add_table(s,
          ["Model (Group TE, All)", "Base acc", "+Embeddings acc", "Delta"],
          [["Random Forest", "0.82", "0.77", "-0.05"],
           ["Logistic Regression", "0.82", "0.72", "-0.10"],
           ["Naive Bayes", "0.75", "0.77", "+0.03"]],
          top=4.1, widths=[3.6, 2.2, 2.7, 1.8])
add_note(s, "Embeddings (openSMILE + emoW2V + sentiment, PCA) added on top of base features. "
            "Net slight degradation - best base 0.82 > best +embeddings 0.77. Added dimensions = noise.", top=6.2)

# --- Slide 15 (idx 14): RQ answers — reframe (multimodal, not linguistic-only) ---
s = prs.slides[14]
body = None
for ph in s.placeholders:
    if ph.placeholder_format.idx == 1:
        body = ph
if body is not None:
    tf = body.text_frame
    tf.clear()
    bullets = [
        ("Our features = linguistic (text) + audio-derived affect (valence/arousal/dominance) + speaking (VAD) -> a MULTIMODAL detector, not linguistic-only", 0),
        ("Multimodal (text + audio affect): group TE acc 0.83 / F1 0.76 (All); significantly beats prior gaze x speaking QDA (d = 0.9-1.35)", 0),
        ("Strongest signal: audio affect (valence +, dominance -) for group TE; speaking + word-rate for individual", 0),
        ("Linguistic-only RQ NOT yet isolated - needs an ablation with text-only features (word counts, rates, sentiment), dropping audio affect + speaking (future work)", 0),
        ("Embeddings (openSMILE/emoW2V/sentiment) on top: no gain, slight degradation", 0),
        ("Open: individual engagement still ~0.70; dyads underpowered (n=8); class imbalance (74% high)", 0),
        ("1-year: real-time multimodal per-window TE detector feeding adaptive VR feedback", 0),
    ]
    for i, (text, lvl) in enumerate(bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = text; p.level = lvl
        for r in p.runs:
            r.font.size = Pt(15)

prs.save(SRC)
print("edited slides 12,13,14,15;", len(prs.slides), "slides")
