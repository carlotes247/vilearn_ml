"""Group/individual-TE disentangling deck patch (2026-07-07 decision).

Decision: individual-TE track ARCHIVED (single-annotator labels); IUI = group TE only.
All affected slides rebuilt on the canonical group-level protocol (c2 tables).

Targets scratch/2026_March_ViLearn_Planning.pptx (backup first). Idempotent-ish:
rebuild slides are reconstructed on every run (safe); footnotes are add-once.

  Slide 13  Modality Ablation      -> rebuilt from c2 (audio/linguistic/gaze/fusions)
  Slide 15  old per-role results   -> rebuilt as "Main Results — Group-Level"
  Slide 16  Why 60s Windows        -> unit footnote only (choice slide, kept)
  Slide 17  What Drives TE         -> rebuilt from d1 drop-one + LogReg coefs
  Slide 18  Embeddings             -> rebuilt from emow2v_pca / sentemb_pca cells

Run from repo root:  python3 discover/presentation_scripts/23_group_level_slides.py
"""
import numpy as np
import pandas as pd
from pptx import Presentation
from pptx.util import Inches, Pt

SRC = "scratch/2026_March_ViLearn_Planning.pptx"
prs = Presentation(SRC)
c2 = pd.read_csv("scratch/te_adoption/a1_out/c2_paper_table.csv")
d1 = pd.read_csv("scratch/te_adoption/a1_out/d1_linguistic_dropone.csv")

LAB = {"All": "Overall", "D": "Dyads", "T": "Triads"}
LING = ["segment_count", "word_count", "words_per_second", "avg_word_length",
        "question_rate", "speaking_seconds", "mean_segment_duration_s"]


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
    if pd.isna(p) or p == "":
        return "—"
    p = float(p)
    return "<.001" if p < 0.001 else f"{p:.3f}"


def r1(sp, det):
    m = c2[(c2.split == sp) & (c2.detector == det) & (c2["rank"] == 1)]
    return m.iloc[0] if len(m) else None


def row_for(sp, det, name):
    r = r1(sp, det)
    if det == "baseline_majority":
        return [LAB[sp], name, "—", f"{r.acc_mean:.3f} ± {r.acc_std:.3f}", "—", "—", "—"]
    return [LAB[sp], name, r.model, f"{r.acc_mean:.3f} ± {r.acc_std:.3f}",
            f"{r.cohen_d_maj:.2f}", pfmt(r.perm_p), pfmt(r.p_ttest_maj)]


HEADER = ["Split", "Feature set", "Best model", "Acc ± SD", "d", "p perm", "p t-test"]
WIDTHS = [1.0, 2.6, 1.2, 1.9, 0.8, 1.0, 1.0]


def rebuild(slide, title, rows, bullets, note=None, row_font=Pt(9), bullet_font=Pt(10)):
    title_sh = next((sh for sh in slide.shapes if sh.has_text_frame), None)
    keep = title_sh._element if title_sh is not None else None
    for sh in list(slide.shapes):
        if sh._element is not keep:
            sh._element.getparent().remove(sh._element)
    if title_sh is None:
        title_sh = slide.shapes.add_textbox(Inches(0.4), Inches(0.3), Inches(12.5), Inches(0.7))
    title_sh.text_frame.text = title
    for run in title_sh.text_frame.paragraphs[0].runs:
        run.font.size = Pt(24)
        run.font.bold = True
    tbl = slide.shapes.add_table(len(rows) + 1, len(HEADER), Inches(0.4), Inches(1.1),
                                 Inches(sum(WIDTHS)), Inches(0.3 * (len(rows) + 1))).table
    for j, wd in enumerate(WIDTHS):
        tbl.columns[j].width = Inches(wd)
    for j, h in enumerate(HEADER):
        set_cell(tbl.cell(0, j), h, size=Pt(10), bold=True)
    for i, row in enumerate(rows, 1):
        peers = [r for r in rows if r[0] == row[0] and r[2] != "—"]
        best = row[2] != "—" and peers and row == max(peers, key=lambda r: float(r[3].split(" ±")[0]))
        for j, v in enumerate(row):
            set_cell(tbl.cell(i, j), v, size=row_font, bold=best and j in (1, 3))
    y = 1.1 + 0.3 * (len(rows) + 1) + 0.2
    tb = slide.shapes.add_textbox(Inches(0.4), Inches(y), Inches(12.5), Inches(6.6 - y))
    tb.text_frame.word_wrap = True
    for i, t in enumerate(bullets):
        p = tb.text_frame.paragraphs[0] if i == 0 else tb.text_frame.add_paragraph()
        p.text = t
        for run in p.runs:
            run.font.size = bullet_font
    if note:
        nb = slide.shapes.add_textbox(Inches(0.4), Inches(6.9), Inches(12.5), Inches(0.5))
        nb.text_frame.word_wrap = True
        nb.text_frame.paragraphs[0].text = note
        nb.text_frame.paragraphs[0].runs[0].font.size = Pt(8)
        nb.text_frame.paragraphs[0].runs[0].font.italic = True


PROTOCOL_NOTE = ("Group-level, 185 floorlevel windows, clean 2-annotator labels, nested LOSO; "
                 "d = Cohen's d vs majority floor; p perm = 300 permutations; p t-test = one-sample t vs majority. "
                 "Source: c2_paper_table.csv.")

# ---------- slide 13: modality ablation, group level ----------
s13 = find_slide("Modality Ablation")
if s13:
    sets13 = [("audio", "audio (affect v/a/d)"), ("linguistic", "linguistic (transcript stats)"),
              ("GazexSpeaking", "gaze (GazexSpeaking)"), ("audio+linguistic", "audio+linguistic"),
              ("audio+linguistic+GazexSpeaking", "multimodal (audio+ling+gaze)")]
    rows = []
    for sp in ("All", "D", "T"):
        rows.append(row_for(sp, "baseline_majority", "majority baseline"))
        rows += [row_for(sp, det, nm) for det, nm in sets13 if r1(sp, det) is not None]
    rebuild(s13, "Modality Ablation — Group-Level Windows (60 s, nested LOSO)", rows, [
        "UNIT FIX: earlier version broadcast the group label onto per-role rows — that diluted text features "
        "(a silent listener contributed a zero-word row to an engaged group). Every modality is now aggregated "
        "over the whole group window: same unit as the target construct.",
        "Individual-TE prediction track ARCHIVED (labels are single-annotator; group TE has the clean "
        "2-annotator mean). This paper = group TE only.",
        "Linguistic is the strongest single modality Overall + Triads; gaze leads Dyads — see fusion rows and "
        "slides 21/23 for the full grid.",
    ], note=PROTOCOL_NOTE, row_font=Pt(8), bullet_font=Pt(10))
    print("slide 13 rebuilt (group-level modality ablation)")
else:
    print("slide 13 NOT found")

# ---------- slide 15: main results, group level ----------
s15 = find_slide("Per-fold LOSO accuracy")
if s15 is None:
    s15 = find_slide("Main Results — Group-Level")  # idempotent re-run
if s15:
    singles = ["AIxVR", "GazexSpeaking", "audio", "linguistic"]
    fusions = ["audio+AIxVR", "audio+GazexSpeaking", "audio+linguistic",
               "linguistic+GazexSpeaking", "audio+linguistic+GazexSpeaking"]
    rows = []
    for sp in ("All", "D", "T"):
        bs = max((r1(sp, d) for d in singles if r1(sp, d) is not None), key=lambda r: r.acc_mean)
        bf = max((r1(sp, d) for d in fusions if r1(sp, d) is not None), key=lambda r: r.acc_mean)
        rows.append(row_for(sp, "baseline_majority", "majority baseline"))
        rows.append(row_for(sp, "GazexSpeaking", "prior gaze (GxS)"))
        rows.append(row_for(sp, bs.detector, f"best single: {bs.detector}"))
        rows.append(row_for(sp, bf.detector, f"best fusion: {bf.detector}"))
    rebuild(s15, "Main Results — Group-Level TE Detection (185 windows, clean labels)", rows, [
        "Both significance tests reported per the 2026-07-01 decision: permutation (shuffled-label null) and "
        "one-sample t-test vs the majority floor; effect size = Cohen's d vs that floor.",
        "Prior gaze (GazexSpeaking, features from prior work) re-evaluated under the identical protocol — "
        "fair comparison, clean labels.",
        "Full feature-set grid with all sets and second-best models: slides 21 & 23 + c2_paper_table.csv.",
    ], note=PROTOCOL_NOTE, row_font=Pt(9))
    print("slide 15 rebuilt (main results, group level)")
else:
    print("slide 15 NOT found")

# ---------- slide 16: unit footnote ----------
s16 = find_slide("Why 60 s Windows")
if s16:
    NOTE = "Granularity comparison computed on per-role rows (historic)"
    if not any(sh.has_text_frame and NOTE in sh.text_frame.text for sh in s16.shapes):
        nb = s16.shapes.add_textbox(Inches(0.4), Inches(6.9), Inches(12.5), Inches(0.5))
        nb.text_frame.word_wrap = True
        nb.text_frame.paragraphs[0].text = (
            f"{NOTE} — justifies the 60 s window choice only; all result slides use group-level windows.")
        nb.text_frame.paragraphs[0].runs[0].font.size = Pt(9)
        nb.text_frame.paragraphs[0].runs[0].font.italic = True
        print("slide 16 footnote added")
    else:
        print("slide 16 footnote already present")
else:
    print("slide 16 NOT found")

# ---------- slide 17: what drives TE (linguistic drop-one) ----------
s17 = find_slide("What Drives Task Engagement")
if s17:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    feats_df = pd.read_csv("scratch/te_adoption/a1_out/c2_features.csv")
    coefs = {}
    for sp in ("All", "D", "T"):
        sub = feats_df if sp == "All" else feats_df[feats_df.grp == sp]
        X = StandardScaler().fit_transform(sub[LING].values)
        lr = LogisticRegression(solver="liblinear", random_state=42).fit(X, sub["y"].values)
        coefs[sp] = dict(zip(LING, lr.coef_[0]))
    title_sh = next((sh for sh in s17.shapes if sh.has_text_frame), None)
    keep = title_sh._element if title_sh is not None else None
    for sh in list(s17.shapes):
        if sh._element is not keep:
            sh._element.getparent().remove(sh._element)
    if title_sh is None:
        title_sh = s17.shapes.add_textbox(Inches(0.4), Inches(0.3), Inches(12.5), Inches(0.7))
    title_sh.text_frame.text = "What Drives the Linguistic Detector? (drop-one + coefficients)"
    for run in title_sh.text_frame.paragraphs[0].runs:
        run.font.size = Pt(24)
        run.font.bold = True
    hdr = ["Feature", "solo acc (All)", "drop-one Δ (All)", "coef All", "coef D", "coef T"]
    wds = [2.6, 1.6, 1.7, 1.1, 1.1, 1.1]
    tbl = s17.shapes.add_table(len(LING) + 1, len(hdr), Inches(0.4), Inches(1.1),
                               Inches(sum(wds)), Inches(0.3 * (len(LING) + 1))).table
    for j, wd in enumerate(wds):
        tbl.columns[j].width = Inches(wd)
    for j, h in enumerate(hdr):
        set_cell(tbl.cell(0, j), h, size=Pt(10), bold=True)
    for i, f in enumerate(LING, 1):
        solo = d1[(d1.split == "All") & (d1.variant == f"solo_{f}")].iloc[0]
        drop = d1[(d1.split == "All") & (d1.variant == f"drop_{f}")].iloc[0]
        vals = [f, f"{solo.acc_mean:.3f}", f"{drop.delta_vs_full:+.3f}",
                *(f"{coefs[sp][f]:+.2f}" for sp in ("All", "D", "T"))]
        for j, v in enumerate(vals):
            set_cell(tbl.cell(i, j), v, size=Pt(9), bold=(f in ("word_count", "question_rate") and j == 0))
    full_all = d1[(d1.split == "All") & (d1.variant == "full")].iloc[0]
    bullets = [
        f"Signal is DISTRIBUTED, not one feature: full set {full_all.acc_mean:.3f}; best solo word_count 0.753 "
        "(talk amount alone already beats the gaze prior 0.734); no drop-one collapses the score.",
        "question_rate is the second pillar with a NEGATIVE sign — more questions → LOW task engagement "
        "(reads as confusion / clarification-seeking); dropping it costs the most (−0.047 Overall).",
        "word_count ≡ words_per_second (fixed 60 s window) — the paper set drops words_per_second (6 features).",
        "Dyads (n=8 folds): the full set overfits — solo word_count 0.743 beats full 0.689; report the simpler "
        "set or flag it.",
    ]
    tb = s17.shapes.add_textbox(Inches(0.4), Inches(1.1 + 0.3 * (len(LING) + 1) + 0.25),
                                Inches(12.5), Inches(2.6))
    tb.text_frame.word_wrap = True
    for i, t in enumerate(bullets):
        p = tb.text_frame.paragraphs[0] if i == 0 else tb.text_frame.add_paragraph()
        p.text = t
        for run in p.runs:
            run.font.size = Pt(10)
    print("slide 17 rebuilt (linguistic drivers)")
else:
    print("slide 17 NOT found")

# ---------- slide 18: embeddings, group level ----------
s18 = find_slide("Do Audio / Text Embeddings Help")
if s18:
    pairs = [("audio", "handcrafted: affect v/a/d"), ("emow2v_pca", "audio embedding (emow2v, PCA)"),
             ("linguistic", "handcrafted: linguistic stats"), ("sentemb_pca", "text embedding (sentiment emb, PCA)")]
    rows = []
    for sp in ("All", "D", "T"):
        rows.append(row_for(sp, "baseline_majority", "majority baseline"))
        rows += [row_for(sp, det, nm) for det, nm in pairs if r1(sp, det) is not None]
    emo_all, lin_all = r1("All", "emow2v_pca"), r1("All", "linguistic")
    semb_all, vad_all = r1("All", "sentemb_pca"), r1("All", "audio")
    verdict_audio = "beats" if emo_all is not None and emo_all.acc_mean > vad_all.acc_mean else "loses to"
    verdict_text = "beats" if semb_all is not None and semb_all.acc_mean > lin_all.acc_mean else "loses to"
    rebuild(s18, "Do Audio / Text Embeddings Help? (group-level rerun)", rows, [
        "Group-level rerun of the embeddings question (earlier per-role numbers superseded): stream embeddings "
        "windowed onto the same 185 windows; PCA inside the nested pipeline (n_components tuned 5/10/20).",
        f"Audio: emow2v embedding {verdict_audio} handcrafted affect v/a/d "
        f"({(emo_all.acc_mean if emo_all is not None else float('nan')):.3f} vs {vad_all.acc_mean:.3f} Overall).",
        f"Text: sentiment embedding {verdict_text} handcrafted linguistic stats "
        f"({(semb_all.acc_mean if semb_all is not None else float('nan')):.3f} vs {lin_all.acc_mean:.3f} Overall).",
        "Interpretable handcrafted features remain the pre-LLM recommendation.",
    ], note=PROTOCOL_NOTE, row_font=Pt(8))
    print("slide 18 rebuilt (embeddings, group level)")
else:
    print("slide 18 NOT found")

prs.save(SRC)
print(f"\nsaved {SRC}: {len(prs.slides)} slides")
