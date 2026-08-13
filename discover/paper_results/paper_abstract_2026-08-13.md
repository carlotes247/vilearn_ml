# IUI paper — abstract rewrite + underselling audit (2026-08-13)

Scope assumed here (saveli, 2026-08-13): **v/a/d affect and the LLM rubric are held
back for the follow-up journal article.** The paper's three modalities are therefore
eye-tracking (AIxVR / AIED / GazexSpeaking), acoustics (88 eGeMAPS functionals) and
transcript (7 surface statistics). Every number below is from the committed
`c2_paper_table.csv`, `c3_prf.csv` and `c4_acoustic_fusions.csv` under the canonical
protocol: 185 floor-level 60 s group windows, clean 2-annotator-mean TE binarised at
0.5, nested leave-one-group-out model selection (seed 42), 300-permutation test +
one-sample t-test vs the majority floor, Cohen's d vs that floor.

**How to read this document.** §1 is the deliverable due 2026-08-14 — the abstract and
the findings that go in it. Everything else (§2–§11) is material for the paper body and
the next revision cycle; none of it needs to be settled tonight.

---

## 1. Proposed abstract

Headline is **macro-F1-ranked**, not accuracy-ranked (saveli, 2026-08-13). That makes
`linguistic+GazexSpeaking` the best detector in **all three splits** — one feature set
across every table — with `linguistic` alone as the best-per-effort tier.

> Social VR classrooms hide the cues a teacher uses to notice that a group has stopped
> working on its task. We present the first multimodal detectors of group task
> engagement (TE) for free-flow Social VR discussions, and we identify which signal
> carries TE at which group size. From 178 minutes of conversation by 52 pedagogy
> students (8 dyads, 12 triads), two annotators rated group TE continuously; we detect
> it in 185 60-second windows from eye-tracking, acoustic and transcript features.
> Every feature set, ours as well as the two published gaze sets re-evaluated on the
> same data, runs through one protocol: nested leave-one-group-out cross-validation, a
> majority-class floor, a 300-permutation null and a one-sample t-test, reported with
> Cohen's d. Fusing seven surface transcript statistics with prior-work gaze×speaking
> features detects TE at macro-F1 .786 overall (QDA, 78.6% accuracy, d=1.33), .824 in
> dyads (logistic regression, 84.2%, d=2.81) and .773 in triads (naive Bayes, 75.8%,
> d=0.72), significant under both tests in every split, while the best published gaze
> set alone reaches .736/.795/.715 and the AIxVR set stays dominated even inside its
> own fusion. Which modality earns its place flips with group size: in triads the
> transcript statistics alone already match the fusion (macro-F1 .773) and need nothing
> but a microphone, whereas in dyads gaze lifts them from .651 to .824 (+.170 macro-F1, bootstrap
> 95% CI [.047, .279]). Class-level
> results show why fusion matters for intervention: the gaze-only dyad detector
> recalls every high-TE window but only 56% of the low-TE ones, the class an
> intervention has to catch, and the fusion raises that to 72%. Raw acoustic
> functionals never clear the floor under both tests and cost 6 to 8 points when fused in.
> Group TE is therefore detectable in real time and, in larger groups, from consumer
> hardware without eye tracking.

~255 words. Every claim in it is in `c2_paper_table.csv` / `c3_prf.csv` /
`c4_acoustic_fusions.csv`.

### Which undersells are woven in

| # | Underselling (see §3) | In the abstract? |
|---|---|---|
| 1 | Effect sizes | yes — d per split, plus "reported with Cohen's d" |
| 2 | Fair re-evaluation of prior work | yes — "the two published gaze sets, re-evaluated on the same data" |
| 3 | Protocol rigour | yes — nested LOGO + majority floor + permutation null + t-test, one clause |
| 4 | Microphone-only detector | yes — "need nothing but a microphone" + closing sentence |
| 5 | Group-size dissociation, quantified | yes — .651 → .824 in dyads vs a tie in triads |
| 6 | AIxVR dominated everywhere | yes — one clause ("dominated even inside its own fusion") |
| 7 | Class-level behaviour | yes — the 56% → 72% low-TE recall sentence |
| 8–11 | selection robustness, coefficients, unit fix | **no — paper body only** (§10 below) |

### What changed and why

| Current draft | Problem | Replacement |
|---|---|---|
| "This paper explores ML detectors…" | Explores = no claim | "We present the first multimodal detectors… and we show which signal carries TE at which group size" |
| 6 sentences of Social VR / SSRL background before any result | Reader reaches the contribution at sentence 7 | 1 problem sentence, then straight to data and results |
| "TE was detected with an accuracy of 84% / 78% / 77%" | Bare accuracies, no baseline, no model, no significance, no effect size, and accuracy hides that the detector is class-imbalanced | macro-F1 + accuracy + classifier + Cohen's d + both significance tests + the published gaze set's own score under the same protocol |
| "Our results show promise as a first exploratory step" (×3, also in intro and conclusion) | Tells the reviewer to lower expectations | Cut. Replaced by the deployment claim the numbers support |
| "We discuss how our classifier behave in different classification metrics" | Announces a discussion instead of stating the finding | The finding itself: gaze-only misses 44% of low-TE windows, fusion misses 28% |
| "future work should investigate how to account for different group sizes" | Frames the dissociation as an unsolved weakness | The dissociation **is** the result: +16 points from gaze in dyads, +1 in triads |
| No mention of protocol | Reviewer assumes group leakage | Nested LOGO + majority floor + permutation null + t-test, named in one sentence |

---

## 2. Errors and internal contradictions in the current PDF

1. **Introduction contradicts the abstract.** p.2: "our results show that TE can be
   detected with an accuracy of 77% in dyads." Dyads is 84.2%; 77% is the overall
   number. Same paper, two pages apart.
2. **Overall accuracy is understated.** Abstract says 77%; the best overall cell is
   `linguistic+AIED` at **79.4%** (SVM, d=1.81), with `linguistic+GazexSpeaking` at
   78.6% statistically tied (paired p=.83).
3. **Section 5 describes a pipeline we do not run.** It lists 8 sklearn algorithms
   (kNN, Decision Tree, Random Forest, Neural Network, AdaBoost, …) and a
   **uniform DummyClassifier** baseline. The canonical protocol is a 4-model panel
   (QDA / SVM / LogReg / GaussianNB) with inner-CV grid search against a
   **majority-class floor**. Deck slide 19 shows exactly why the uniform dummy has to
   go — it is a stochastic, gameable floor, and significance verdicts flip depending on
   which baseline is used. Leaving it in is the single most likely rejection trigger.
4. **Threshold mismatch.** Section 5 says "median split of .507"; the labels are
   binarised at **0.5** on the clean 2-annotator mean.
5. **Table 3 class distribution is wrong.** Paper: 58/42 dyads, 48/52 triads, 52/48
   all. Actual majority floors: **59.5%** dyads, **53.8%** triads, **51.9%** overall
   (triads are majority *low* TE, the table has the direction inverted).
6. **"Relevant-subset features" is presented as our design choice.** It is the AIxVR
   prior-work feature set, and it is the *worst* set we test (overall 0.625; in triads
   it collapses to a single feature, BPM, 0.581). It belongs in the paper as a
   re-evaluated prior baseline, not as our method.
7. **Participant counts don't reconcile:** 108 volunteered (§3.1) vs 154 students
   collected (§4) vs 52 analysed.
8. **"a high interrater agreement of .772"** — Cronbach's α of .772 is acceptable, not
   high. Reviewers who know the scale will flag the adjective.
9. §4.3, §4.4, §4.5, §6 and §7 are `#TODO` placeholders. §6 currently contains Carlos
   asking whether the ablation "is not as important though?".
10. **Window count is never stated.** n = **185** windows over 20 groups (4–13 per
    group) is the unit of every result and never appears.

---

## 3. Undersellings — results we have and the paper does not claim

1. **Effect sizes.** For the headline `linguistic+GazexSpeaking` detector: d = 2.81
   (dyads), 1.33 (overall), 0.72 (triads) — large to huge, and absent from the
   abstract. The dyad fusion is the only dyad detector significant under *both* tests.
   (Accuracy-ranked cells run higher still: `linguistic+AIED` d = 1.81 overall.)
2. **Fair re-evaluation of prior work.** Prior work [13] reported linear correlations.
   We re-run its exact feature sets under our protocol on the same data — that is a
   methodological contribution in its own right, and it is what licenses the
   "+6 / +8 points over prior gaze" claims. Currently unstated.
3. **Protocol rigour.** Nested LOGO (no group leakage), majority floor, permutation
   null *and* t-test, effect sizes throughout. Most engagement-detection papers report
   one number against a dummy. This is a selling point, currently invisible.
4. **A microphone-only detector.** Seven surface transcript statistics — segments,
   words, average word length, question ratio, speech ratio, mean segment duration,
   unfinished ratio — reach **0.771 overall / 0.766 triads**, beating *every* prior
   gaze feature set overall and in triads, with no eye tracker, no sentiment model and
   no semantic model. That is the strongest deployment claim in the paper and it is
   not made anywhere.
5. **The group-size dissociation, quantified.** Gaze fusion adds **+16.2** points in
   dyads (0.680 → 0.842) and **+1.0** in triads (0.766 → 0.776). The conclusion states
   the direction qualitatively; the numbers make it a finding. It also *refines* prior
   work's "blinks for triads" claim: in triads the useful signal is talk, not eyes.
6. **AIxVR is dominated everywhere**, including inside its own fusion
   (linguistic+AIxVR 0.741 < linguistic alone 0.771). Direct, quantified comparison
   against the previous paper, unreported.
7. **Class-level behaviour** (the paper's own §7 TODO: "We do not currently have this
   calculated!"). Now computed — see §4. The headline: gaze alone in dyads has perfect
   high-TE recall (1.000) and 0.562 low-TE recall. A system that never notices
   disengagement is useless for intervention; the transcript fusion lifts low-TE recall
   to 0.719. This is the argument §7 is trying to make and cannot currently support.
8. **Acoustics tested properly and rejected.** 88 eGeMAPS functionals: 0.576 / 0.658 /
   0.567, never significant under both tests. PCA (n tuned in-CV) and SelectKBest
   (k tuned 1–88 in-CV) do not rescue them, and fusing them *costs* 6–8 points
   (transcript 0.771 → 0.689 overall). Reported as a finding, this saves other groups
   the same dead end.
9. **Feature-selection robustness.** SelectKBest with k tuned in the inner CV
   (`c2_selection.csv`) answers the inevitable "why a fixed feature set?" reviewer
   question: selection helps nowhere robustly and destabilises fusions at n=8.
10. **Interpretable coefficients.** Drop-one ablation shows the signal is distributed
    (best solo feature: words/window 0.753), and **question ratio predicts TE
    negatively** (coef −0.56 overall, −0.64 triads) — more questions means
    clarification-seeking, not engagement. A concrete, discussable mechanism.
11. **The unit fix.** Every modality is aggregated over the whole 60 s group window,
    matching the unit of the group-level target. The earlier per-role formulation
    diluted text features. Worth one methods sentence; it is why these numbers differ
    from the ICMI submission.

---

## 4. Precision / recall / F1 / confusion — headline cells

Class 1 = High TE, class 0 = Low TE. Pooled over the out-of-fold LOGO predictions
(each window predicted exactly once, by a model that never saw its group).
`acc` = mean over folds, reproduces `c2_paper_table.csv` exactly for all 74 cells.

### Overall (n=185, 20 groups, majority floor .519)

| Feature set | Model | acc | TN | FP | FN | TP | P/R/F1 high | P/R/F1 low | macro-F1 | bal.acc | MCC |
|---|---|---|---|---|---|---|---|---|---|---|---|
| majority baseline | — | .448 | 2 | 87 | 10 | 86 | .497/.896/.639 | .167/.022/.040 | .340 | .459 | −.166 |
| AIxVR (prior) | NB | .625 | 59 | 30 | 42 | 54 | .643/.562/.600 | .584/.663/.621 | .611 | .613 | .226 |
| AIED (prior) | QDA | .626 | 60 | 29 | 39 | 57 | .663/.594/.626 | .606/.674/.638 | .632 | .634 | .268 |
| GazexSpeaking (prior) | QDA | .734 | 51 | 38 | 9 | 87 | .696/.906/.787 | .850/.573/.685 | .736 | .740 | .512 |
| openSMILE (88) | NB | .576 | 23 | 66 | 8 | 88 | .571/.917/.704 | .742/.258/.383 | .544 | .588 | .234 |
| linguistic (7) | SVM | .771 | 60 | 29 | 14 | 82 | .739/.854/.792 | .811/.674/.736 | .764 | .764 | .539 |
| **linguistic+AIED** | **SVM** | **.794** | 62 | 27 | 14 | 82 | .752/.854/.800 | .816/.697/.752 | .776 | .775 | .559 |
| linguistic+GazexSpeaking | QDA | .786 | 61 | 28 | 11 | 85 | .752/.885/.813 | .847/.685/.758 | **.786** | **.785** | **.585** |
| linguistic+AIxVR | NB | .741 | 52 | 37 | 9 | 87 | .702/.906/.791 | .852/.584/.693 | .742 | .745 | .521 |

### Dyads (n=79, 8 groups, majority floor .595)

| Feature set | Model | acc | TN | FP | FN | TP | P/R/F1 high | P/R/F1 low | macro-F1 | bal.acc | MCC |
|---|---|---|---|---|---|---|---|---|---|---|---|
| majority baseline | — | .560 | 0 | 32 | 0 | 47 | .595/1.000/.746 | .000/.000/.000 | .373 | .500 | .000 |
| AIxVR (prior) | LogReg | .672 | 14 | 18 | 7 | 40 | .690/.851/.762 | .667/.438/.528 | .645 | .644 | .321 |
| AIED (prior) | NB | .749 | 20 | 12 | 8 | 39 | .765/.830/.796 | .714/.625/.667 | .731 | .727 | .467 |
| GazexSpeaking (prior) | SVM | .830 | 18 | 14 | 0 | 47 | .770/**1.000**/.870 | 1.000/**.562**/.720 | .795 | .781 | .658 |
| openSMILE (88) | LogReg | .658 | 19 | 13 | 14 | 33 | .717/.702/.710 | .576/.594/.585 | .647 | .648 | .295 |
| linguistic (7) | LogReg | .680 | 17 | 15 | 11 | 36 | .706/.766/.735 | .607/.531/.567 | .651 | .649 | .305 |
| **linguistic+GazexSpeaking** | **LogReg** | **.842** | 23 | 9 | 4 | 43 | .827/.915/.869 | .852/**.719**/.780 | **.824** | **.817** | .656 |
| linguistic+AIED | LogReg | .815 | 25 | 7 | 8 | 39 | .848/.830/.839 | .758/.781/.769 | .804 | .806 | .608 |

### Triads (n=106, 12 groups, majority floor .538)

| Feature set | Model | acc | TN | FP | FN | TP | P/R/F1 high | P/R/F1 low | macro-F1 | bal.acc | MCC |
|---|---|---|---|---|---|---|---|---|---|---|---|
| majority baseline | — | .572 | 57 | 0 | 49 | 0 | .000/.000/.000 | .538/1.000/.699 | .350 | .500 | .000 |
| AIxVR (prior, 1 feat) | LogReg | .581 | 39 | 18 | 27 | 22 | .550/.449/.494 | .591/.684/.634 | .564 | .567 | .137 |
| AIED (prior) | QDA | .628 | 45 | 12 | 31 | 18 | .600/.367/.456 | .592/.789/.677 | .566 | .578 | .174 |
| GazexSpeaking (prior) | NB | .695 | 34 | 23 | 7 | 42 | .646/.857/.737 | .829/.596/.694 | .715 | .727 | .464 |
| openSMILE (88) | NB | .567 | 21 | 36 | 8 | 41 | .532/.837/.651 | .724/.368/.488 | .570 | .603 | .229 |
| linguistic (7) | NB | .766 | 39 | 18 | 6 | 43 | .705/.878/.782 | .867/.684/.765 | **.773** | **.781** | **.567** |
| **linguistic+AIED** | **QDA** | **.776** | 44 | 13 | 13 | 36 | .735/.735/.735 | .772/.772/.772 | .753 | .753 | .507 |
| linguistic+GazexSpeaking | NB | .758 | 38 | 19 | 5 | 44 | .698/.898/.786 | .884/.667/.760 | .773 | .782 | .573 |
| linguistic+AIxVR | NB | .758 | 38 | 19 | 6 | 43 | .694/.878/.775 | .864/.667/.752 | .764 | .772 | .551 |

Full grid (all 75 cells, including `_sel`, `_pca`, `_core` and the reserved v/a/d and
embedding cells): `c3_prf.csv`. Per-window out-of-fold predictions:
`c3_predictions.csv`.

### Reading these for the paper

- **The majority floor is not a neutral opponent.** In triads it predicts "low TE"
  for every window (F1 high = .000); in dyads "high TE" for every window (F1 low =
  .000). Any detector with a non-degenerate confusion matrix is already doing
  something the floor cannot. Say so — it is stronger than an accuracy delta.
- **Gaze detects presence of engagement, transcript detects its absence.** Every
  gaze-only cell has high-TE recall ≫ low-TE recall (dyads 1.000 vs .562; overall
  .906 vs .573). Fusing in transcript features moves the errors off the low-TE class,
  which is the class an SSRL intervention has to catch.
- **Rank by macro-F1, not accuracy** (decided 2026-08-13). Accuracy rewards a detector
  for riding the majority class; the floors here are .519 / .595 / .538 and the classes
  are the thing an intervention has to tell apart. Under macro-F1 the ranking is also
  simpler and more consistent:

  | Feature set | All | Dyads | Triads |
  |---|---|---|---|
  | **linguistic+GazexSpeaking** | **.786** | **.824** | **.773** |
  | linguistic+AIED | .776 | .804 | .753 |
  | linguistic | .764 | .651 | .773 |
  | GazexSpeaking (prior) | .736 | .795 | .715 |

  `linguistic+GazexSpeaking` is best in every split — one feature set for every table,
  significant under both tests everywhere. Accuracy-ranking would instead have picked
  `linguistic+AIED` in All and Triads, i.e. a different gaze set per split, purchased
  with a worse class balance (Triads: `ling+AIED` .776 acc but macro-F1 .753 and MCC
  .507, against `ling+GxS` .758 acc / .773 / .573). The retired accuracy-ranked
  headline was 79.4 / 84.2 / 77.6.

- **Two tiers, and say so.** Best = `linguistic+GazexSpeaking`. Best-per-effort =
  `linguistic` alone: it **ties the fusion in triads** (macro-F1 .773 both) and needs
  only a microphone, but collapses in dyads (.651). That is the hardware argument, and
  it is a result, not a caveat.

---

## 5. New cells run today: acoustic fusions (`c4_acoustic_fusions.csv`)

With v/a/d reserved for the journal paper, openSMILE became the paper's only acoustic
modality — and c2 had never fused it with anything except v/a/d, so the
"eye-tracking + acoustic + transcript" claim in the title had no acoustic fusion behind
it. Those cells now exist (same protocol):

| Split | Feature set | #f | Model | acc ± SD | d | perm p | t-test p | macro-F1 | MCC |
|---|---|---|---|---|---|---|---|---|---|
| All | linguistic+openSMILE | 95 | LogReg | .689 ± .258 | 0.66 | .003 | .008 | .694 | .394 |
| All | openSMILE+GazexSpeaking | 94 | NB | .635 ± .245 | 0.47 | .003 | .048 | .613 | .335 |
| All | linguistic+openSMILE+GazexSpeaking | 101 | LogReg | .679 ± .271 | 0.59 | .003 | .016 | .680 | .361 |
| D | linguistic+openSMILE | 95 | NB | .601 ± .228 | 0.03 | .302 | .945 | .498 | .150 |
| D | openSMILE+GazexSpeaking | 94 | LogReg | .724 ± .155 | 0.83 | .522 | .051 | .708 | .417 |
| D | linguistic+openSMILE+GazexSpeaking | 101 | NB | .626 ± .221 | 0.14 | .316 | .703 | .546 | .229 |
| T | linguistic+openSMILE | 95 | LogReg | .704 ± .221 | 0.76 | .003 | .024 | .717 | .438 |
| T | openSMILE+GazexSpeaking | 94 | NB | .635 ± .296 | 0.33 | .007 | .278 | .656 | .361 |
| T | linguistic+openSMILE+GazexSpeaking | 101 | NB | .690 ± .309 | 0.49 | .003 | .117 | .714 | .472 |

**Every acoustic fusion is worse than the same set without acoustics.** Adding the 88
functionals to the transcript set costs 8.2 points overall (.771 → .689), 7.9 in dyads
(.680 → .601) and 6.2 in triads (.766 → .704); adding them to gaze costs 9.9 in dyads
(.830 → .724). This closes the acoustic story cleanly: raw acoustics do not carry group
TE at this scale, alone or in fusion.

**Title implication:** "…from Eye-Tracking, Acoustic and Transcript Data…" is now
accurate only in the sense that acoustics were *tested*. Either keep the title and
report acoustics honestly as a tested-and-rejected modality (defensible — the negative
result is useful), or drop "Acoustic" from the title. Keeping it and staying silent on
the result is the one option that will not survive review.

---

## 6. Reproduction

```bash
# per-class metrics + confusion matrices for every rank-1 cell in c2_paper_table.csv
python3 discover/analysis_scripts/c3_prf_confusion.py
# acoustic fusions without v/a/d
python3 discover/analysis_scripts/c4_acoustic_fusions.py
```

Both write to the gitignored workbench `scratch/te_adoption/a1_out/`; refreshed CSVs
are copied into `discover/paper_results/` and committed. `c3` reruns the rank-1 model
of each cell and keeps the out-of-fold predictions; its fold-mean accuracies reproduce
`c2_paper_table.csv` to three decimals in all 74 non-baseline cells.

---

## 7. 178 minutes vs 185 windows — no contradiction, but the paper must explain it

Windows are non-overlapping 60 s slices on the floor-level interaction grid, and each
group gets `ceil(duration / 60 s)` of them, so the **last window of every group is
partial**.

| | groups | windows | window span | reported conversation | unfilled tail |
|---|---|---|---|---|---|
| Dyads | 8 | 79 | 79.0 min | 77.88 min | 1.12 min |
| Triads | 12 | 106 | 106.0 min | 100.11 min | 5.89 min |
| **Total** | **20** | **185** | **185.0 min** | **177.99 min** | **7.01 min** |

Per-group window counts run 4–13 (`triad_11` = 4, `triad_01` = 13), mean 9.25. The
average final window holds ~39 s of the 60 s, i.e. ~21 s short. Nothing is padded or
imputed: window features are computed over whatever frames fall inside the interval, so
a short tail window simply aggregates less data.

Two sentences for §5 cover it:

> Each group interaction was divided into non-overlapping 60 s windows from the start of
> the interaction, yielding 4–13 windows per group (M=9.25) and 185 windows in total (79
> dyadic, 106 triadic). The final window of each group is shorter than 60 s (21 s short
> on average); features are computed over the frames it contains rather than padded.

This also removes an inconsistency a reviewer would find on their own: 185 min of
windows against a stated 177.99 min of conversation.

---

## 8. Should we push "no eye tracker needed"?

**Yes, but as a group-size-conditional claim, not a blanket one.** The blanket version
is false in dyads and the data says so loudly:

| | transcript only | + gaze×speaking | gaze buys |
|---|---|---|---|
| Overall | .764 | .786 | +.022 |
| **Dyads** | **.651** | **.824** | **+.173** |
| **Triads** | **.773** | **.773** | **±.000** |

(macro-F1; the accuracy view agrees: dyads .680 → .842, triads .766 → .758.)

So the defensible claim is:

> In triads, group task engagement is detected as well from the transcript alone as
> from the full multimodal fusion — a consumer headset with a microphone suffices. In
> dyads, eye tracking is not optional: it is what makes the detector work at all
> (macro-F1 .651 → .824).

This is stronger than "cheap hardware suffices" because it is a **design rule**, not a
cost note: it tells a deployment which sensor to buy for which classroom configuration,
and it is the operational form of the paper's central finding. It also lands better
against the prior work, which claimed eye-tracking predictors for *both* group sizes.

Two honesty guards:
- The transcript features need ASR plus diarization of the shared group audio. Say
  "microphone plus on-device ASR", not "microphone".
- The dyad transcript-only number (.651, acc .680) is **not significant** under either
  test — that is exactly why the claim must be conditional.

---

## 9. Title

The current title states the procedure ("Detecting X from A, B and C in D") and no
finding. It also now over-promises on the acoustic modality. Candidates, best first:

1. **Gaze for Dyads, Talk for Triads: Detecting Group Task Engagement in Social VR
   Classrooms** — direct riff on the prior paper's *"Gaze for Dyads & Blinks for Triads"*
   [13], which makes the relationship explicit and states that we **correct its triad
   half**: the triad signal is talk, not blinks. Reviewers who know [13] get the
   contribution from the title alone.
2. **Group Size Decides the Modality: Multimodal Task-Engagement Detection in Social VR
   Classrooms** — same finding, no dependence on the reader knowing [13].
3. **What the Microphone Already Knows: Detecting Group Task Engagement in Social VR
   Without Eye Tracking** — leads with the deployment result; risks overstating, since
   the claim only holds for triads.
4. **Talk Carries Triads, Gaze Carries Dyads: Multimodal Detection of Task Engagement in
   Social VR Small Groups** — variant of 1 without the borrowed phrasing.

Recommendation: **1**. It is a finding, it is short, it positions the paper against the
work it extends, and it survives the loss of the acoustic modality.

---

## 10. Split: what goes in the abstract now vs the paper later

### Now (abstract, due 2026-08-14)

- Two-tier headline: `linguistic+GazexSpeaking` best in all three splits by macro-F1;
  `linguistic` alone as the microphone-only tier that ties it in triads.
- Effect sizes and both significance tests, named in one clause.
- Prior gaze sets re-evaluated under the identical protocol, with their numbers.
- The group-size dissociation, quantified.
- The low-TE recall gap (56% → 72%) as the intervention argument.
- Acoustics tested and rejected, stated as a result.
- New title.

### Later (paper body, before submission)

**Must fix — deprecated content carried over from the ICMI/AIxVR rejections.** These are
not stale numbers, they are descriptions of a pipeline that no longer exists, and every
one of them is independently rejection-worthy:

1. §5: delete the 8-algorithm panel (kNN, Decision Tree, Random Forest, Neural Network,
   AdaBoost) — the canonical protocol is QDA / SVM / LogReg / GaussianNB with inner-CV
   grid search.
2. §5: delete the **uniform DummyClassifier** baseline. It is stochastic and gameable;
   the floor is the majority class. Deck slide 19 documents verdicts flipping between
   the two — a reviewer finding that on their own is fatal.
3. §5: "relevant-subset features … we computed a different subset for dyads and triads"
   is the **AIxVR prior set**, not our design, and it is the worst set we test (All
   .625; in triads it reduces to a single feature, BPM, .581). Reframe as a
   re-evaluated prior baseline.
4. §5: threshold is **0.5** on the clean two-annotator mean, not the .507 median split.
5. Table 3: regenerate. Real floors are .595 dyads / .538 triads / .519 overall, and the
   triad row currently has the majority class inverted.
6. §1 vs abstract: the "77% in dyads" sentence in the introduction contradicts the
   abstract's dyad number. Delete.
7. §3.1 vs §4: 108 volunteered vs 154 collected vs 52 analysed — reconcile.
8. §4.1: Cronbach's α = .772 is *acceptable*, not "high".
9. Add: 185 windows, 4–13 per group, final window partial (§7 above).
10. Add: nested LOGO means no group ever appears in its own training set — state it
    explicitly, or reviewers will assume leakage.
11. Add: the unit fix — every modality is aggregated over the whole 60 s group window,
    matching the unit of the group-level target. This is why these numbers differ from
    the ICMI submission.

**Results the body should carry that the abstract cannot hold (§3 items 8–11):**

12. §6: the full modality ablation (all base sets × 3 splits) with macro-F1 alongside
    accuracy — `c3_prf.csv` has every cell.
13. §6/§7: confusion matrices for the three headline detectors plus the majority floor.
    The floor's matrices are degenerate (triads predicts low TE for all 106 windows,
    dyads high TE for all 79) — that single observation justifies the whole
    majority-floor + permutation apparatus.
14. §6: the feature-selection robustness analysis (SelectKBest, k tuned in the inner CV,
    `c2_selection.csv`) — pre-empts "why a fixed feature set?". Answer: selection helps
    nowhere robustly and destabilises fusions at n=8.
15. §6/§7: the drop-one ablation and coefficient signs (`d1_linguistic_dropone.csv`) —
    the signal is distributed (best solo feature: words/window, .753), and question
    ratio predicts TE **negatively** (−.56 overall, −.64 triads): more questions means
    clarification-seeking, not engagement. This is the mechanism §7 is missing.
16. §6: acoustics — the 88 eGeMAPS functionals alone, under PCA, under in-CV selection,
    and in fusion (`c4_acoustic_fusions.csv`). All fail. What little signal exists is
    *variability*, not level (selection converges on mfcc1V / spectral-flux / loudness
    stddevNorm, k≈4).
17. §8: the dyad transcript-only result is not significant under either test; the
    hardware claim is conditional on group size.

---

## 11. The rejection reviews — what is already answered, what still needs work

Title decided (saveli, 2026-08-13): **"Gaze for Dyads, Talk for Triads: Detecting Group
Task Engagement in Social VR Classrooms."**

Three of the five substantive rejection reasons are **dead** — killed by work already
committed, not by rewording. Two needed new analysis, run today as
`c5_reviewer_checks.py` (`c5_error_independence.csv`, `c5_bootstrap_ci.csv`,
`c5_bootstrap_contrasts.csv`). Two remain genuinely open.

| Objection | Who | Status | Evidence |
|---|---|---|---|
| "Triad models do not beat the baseline" | Coordinator, R3 | **dead** | Triads `linguistic` macro-F1 .773, acc .766, perm p=.003, t-test p=.008 — significant under **both** tests. `linguistic+GxS` likewise (.773 / .758 / .003 / .030). |
| "Not fully multimodal; limited to eye-tracking with limited speech" | R3 (main weakness) | **dead** | Three modalities with full per-split ablation and every pairwise fusion; embeddings (emoW2V, sentiment) and 88 eGeMAPS functionals additionally tested and reported. |
| "Positive results limited to dyads" | Coordinator | **dead** | Every split now has a detector significant under both tests; the *dissociation* is the finding, not a coverage gap. |
| "Circularity: annotators saw gaze rays, gaze is model input" | R1 C4, Coordinator | **answered** (§11.1) | Gaze-free detector works; gaze detector degrades with group size; error structures separate. |
| "Small dataset, may not generalise" | R2, R3, Coordinator | **quantified** (§11.2) | Group-level bootstrap CIs; every headline detector beats the floor in 100% of 10,000 resamples. |
| "Same dataset as [13] → secondary classification analysis" | R1 C1, Coordinator | **reframeable** (§11.3) | Same recordings, **different labels**, two new modalities, a corrected prior baseline. |
| "Binary high/low TE is simplistic" | R2 | open (§11.4) | — |
| "Participants co-located" | R3 | open — limitation only | — |
| "Participant counts inconsistent" | R1 C3 | fix in body | 156 invited / 108 participated / 52 analysed; the 154 is a typo (per rebuttal). Add the flow table. |
| "Do results hold with the outlier dyad included?" | R1 Q3 | **runnable now** (§11.5) | `data/group_names_with_time_subsetsFullVERSION.csv` has all 25 groups' interaction timestamps, and the gaze feature file already covers all 25. Nothing is missing. |

### 11.1 Circularity — the strongest available answer

R1's concern: annotators saw gaze rays while labelling, so gaze features may be reading
the annotation channel back out. Five pieces of evidence, in descending strength:

1. **A gaze-free detector works.** `linguistic` uses seven surface transcript
   statistics and no eye-tracking at all: macro-F1 .764 overall and .773 in triads,
   significant under both tests. TE labels cannot be a pure gaze readout if a detector
   that never sees gaze recovers them.
2. **The gaze detector degrades with group size, and the annotation setup does not.**
   Annotators saw the same gaze rays in dyads and in triads. A label-readout channel
   would not care how many people are in the room; a *behavioural* gaze–engagement
   relation does. GazexSpeaking drops from macro-F1 .795 (dyads) to .715 (triads), and
   in triads it fails the t-test (p=.090). This is the pattern a real effect makes and
   a readout does not.
3. **In dyads — where the concern bites hardest, because that is where gaze works — the
   gaze-only and transcript-only detectors are close to statistically independent.**
   They agree on only 69.6% of windows (Cohen's κ = .278 between their predictions),
   and their per-window *errors* are uncorrelated (φ = .239, χ² p = .070): 6 windows
   only gaze gets wrong, 18 only transcript gets wrong. One shared annotation shortcut
   would produce one shared error pattern. (Overall and triads the errors do correlate,
   φ = .50 / .61 — expected, since both detectors track the same construct and the easy
   windows are easy for both. Report all three honestly and lead with dyads.)
4. **The coding schema is verbal.** Table 1 defines TE by silence, off-task talk,
   surface-level task talk and elaborated task talk with prior knowledge and connected
   ideas. There is no gaze criterion in it, and the rebuttal already states that
   annotators used the verbal information.
5. **And the mirror-image objection does not hold either.** If the labels are verbally
   grounded, are the transcript features circular instead? No: the schema codes
   *content* (depth, reasoning, connecting ideas), while the seven features are
   *surface* statistics — segment and word counts, average word length, question ratio,
   speech ratio, mean segment duration, unfinished ratio. No semantic model, no
   sentiment model, no topic. Talk-amount is not what was coded.

Point 2 is the one to put in the paper: it converts a design flaw into a testable
prediction that the data passes. It also explains R1's Q2 (eye movements were **not**
rendered on the avatars, so participants read head direction, not gaze) — with one
partner the head-direction target is unambiguous, with two it is diffuse. The reviewer's
own observation becomes the mechanism behind the paper's central finding.

### 11.2 Small sample — quantify it instead of apologising

Group-level bootstrap: resample the 20 (8 / 12) **groups** with replacement so the
within-group dependency is preserved; 10,000 resamples; pooled macro-F1.

| Split | Detector | macro-F1 [95% CI] | vs majority floor |
|---|---|---|---|
| All | majority floor | .336 [.273, .394] | — |
| All | linguistic | .762 [.679, .842] | +.425 [.317, .529], better in **100%** of resamples |
| All | linguistic+GazexSpeaking | .783 [.696, .857] | +.446 [.340, .545], **100%** |
| Dyads | majority floor | .368 [.266, .437] | — |
| Dyads | linguistic | .645 [.537, .765] | +.275 [.141, .433], **100%** |
| Dyads | linguistic+GazexSpeaking | .814 [.719, .886] | +.445 [.288, .592], **99.99%** |
| Triads | majority floor | .349 [.279, .409] | — |
| Triads | linguistic | .770 [.640, .876] | +.420 [.267, .557], **100%** |
| Triads | linguistic+GazexSpeaking | .768 [.622, .896] | +.419 [.243, .585], **100%** |

The paired contrasts are the more interesting result, because they certify the paper's
central claim directly:

| Split | fusion − transcript-only (macro-F1) | 95% CI | fusion better in |
|---|---|---|---|
| **Dyads** | **+.170** | **[+.047, +.279]** | **99.6%** of resamples |
| Triads | −.001 | [−.060, +.054] | 45.4% |
| All | +.021 | [−.020, +.062] | 83.5% |

Gaze's contribution is certified in dyads with a CI that excludes zero, is a tight null
in triads (CI ±.06 around zero — the microphone-only claim is not a failure to detect a
difference, the interval is narrow enough to call it a tie), and is not established
overall. Report it exactly that way. It is a stronger answer to "n is small" than any
amount of hedging, and it makes the title's claim a measured one.

### 11.3 "Same dataset → secondary analysis" — the reframe

Same recordings, but:

- **Different labels.** The prior work's labels carry a 60 Hz annotation time-stretch
  bug (confirmed by Carlos, `TE_LABEL_DISCREPANCIES.md`); this paper uses the clean
  two-annotator mean. The prior published numbers *understated their own method* — we
  re-run their feature sets on corrected labels and report the improvement.
- **Two modalities that do not exist in [13]** (acoustic, transcript), plus their
  fusions.
- **A different protocol**: nested LOGO model selection against a majority floor with a
  permutation null, versus linear correlation.
- **A correction of the prior finding.** [13] reports blinks as the triad predictor.
  Under this protocol the triad signal is talk: `AIED` (which contains the blink
  features) reaches macro-F1 .566 in triads, against .773 for transcript statistics.
  That is a substantive disagreement with the work we extend, not a re-analysis of it.

Position the paper as *"the transfer question [13] raised, answered — and its triad half
corrected"*, and R1's framing objection resolves itself.

### 11.4 Binary classification — the annotation makes the argument

R2 called binary high/low simplistic. The current draft invites that by justifying
binary as "a simpler problem to tackle as a first exploratory step" — convenience. The
real justification is in the annotation design and is much stronger (saveli,
2026-08-13):

- The construct is **ternary by design**: the coding schema (Table 1) defines three TE
  levels, low / medium / high.
- It was **annotated on a continuous scale** so that the annotators would not be pushed
  onto boundaries the schema does not specify.
- **The annotators never used the full spectrum.** Each settled into a different band of
  the slider for the same coded level, so the continuous values are not comparable
  across annotators as magnitudes — only as ordering within an annotator.
- Consolidating to binary is therefore what makes the two annotators **commensurable**.
  It is a measurement decision forced by the annotation behaviour, not a simplification
  of the research question.

Write it that way and R2's objection turns into a methods paragraph. In hindsight the
scale should have been discrete from the start — that belongs in Limitations as a
concrete, actionable lesson rather than a generic "more data needed".

Optional extra if there is time: report Spearman ρ between the detector's decision
function and the continuous two-annotator mean (`our_te`, already in
`c2_features.csv`) — no retraining, just keep `decision_function` instead of `predict`.

### 11.5 The outlier dyad (R1 Q3) — not blocked, and the exclusion needs re-checking

Everything needed is in the repo:

- `data/group_names_with_time_subsetsFullVERSION.csv` — interaction timestamps for **25**
  groups (the floor-level file holds only the 20 analysed ones).
- `Recordings/SavedData/v2_no_low_sampled/60s_TE_correlation_2026-05-19.csv` — the gaze
  feature file already covers **all 25 groups / 243 windows**; `a1_lib.load_data()`
  filters it down to 185.
- The five extra groups (`dyad_01`, `dyad_08`, `dyad_09`, `triad_03`, `triad_04`) all
  have recording times, both annotators' TE, transcripts, and merged CSV + parquet.

This also reconciles the exclusion accounting exactly: 43 groups − 18 unusable
recordings = 25 (the gaze file) − 4 avatar-height groups − 1 outlier dyad = 20.

**The outlier is `dyad_09`, and under the corrected labels it is no longer an outlier.**
Group-mean TE on the clean two-annotator mean, against the 20 included groups
(M=.456, SD=.163):

| Group | mean TE | z vs included | in analysis? |
|---|---|---|---|
| triad_08 | .078 | **−2.32** | **yes** |
| dyad_10 | .132 | **−1.99** | **yes** |
| dyad_09 | .155 | −1.85 | **no — excluded as the outlier** |
| dyad_08 | .403 | −0.33 | no (avatar height) |
| triad_03 | .450 | −0.04 | no (avatar height) |
| dyad_01 | .509 | +0.33 | no (avatar height) |
| triad_04 | .554 | +0.60 | no (avatar height) |

The exclusion was decided on the old labels, which carried the 60 Hz time-stretch bug.
On the corrected labels the excluded dyad is **less** extreme than two groups that were
kept. As written, the "> 2 SD from the mean" criterion in §4 no longer describes what
was done — a reviewer who recomputes it will find that immediately.

Two options, in order of preference:

1. **Re-include `dyad_09` and report the 21-group / 195-window results** as the headline
   or as a robustness row. One rerun of `c2`/`c3` with the floor-level group list
   swapped for the FullVERSION list. This answers R1 Q3 with data instead of prose and
   removes a criterion that no longer holds.
2. Keep the exclusion but restate the criterion in terms of the labels it was actually
   applied to, and report the re-included result as a robustness check.

Either way the current §4 sentence has to change.

---

## 12. Exclusion-chain audit — the inclusion rule is not reconstructible

Triggered by saveli, 2026-08-13: "first time I hear about excluding sessions for being an
outlier; I was always told floor level + interaction times." Source ledger:
`data/groups_info_as_of_2025_06_30.csv` (28 rows; an export of `data/groups_info.xlsx`,
sheet `Sheet1`). The chain is

    43 recorded (paper §4)  →  28 in the triage ledger  →  25 named in NOVA
                            →  25 in vilearn_more.set / the gaze feature file
                            →  20 analysed

and **at least four different filters act between 25 and 20, applied inconsistently.**

| group | in analysis | valid blinks | valid gaze | P flying | Cronbach α | annot. freq |
|---|---|---|---|---|---|---|
| dyad_01 | **no** | 1 | 0 | — | **.967** | 90:90 |
| dyad_02 | yes | 1 | **0** | — | .775 | 90:90 |
| dyad_03 | yes | 1 | 1 | — | .706 | 90:90 |
| dyad_04 | yes | 1 | 1 | — | .892 | 90:90 |
| dyad_05 | yes | 1 | 1 | — | .848 | 90:90 |
| dyad_06 | yes | 1 | 1 | — | .837 | 90:90 |
| dyad_07 | yes | 1 | 1 | — | .848 | 90:90 |
| dyad_08 | **no** | 1 | 0 | — | .807 | 60:90 |
| dyad_09 | **no** | 1 | 0 | 0 | .829 | 60:90 |
| dyad_10 | yes | 1 | **0** | — | **.551** | 60:90 |
| dyad_11 | yes | — | — | — | .842 | 60:90 |
| triad_01 | yes | 1 | 1 | — | .901 | 90:90 |
| triad_02 | yes | 1 | **0** | — | .748 | 90:90 |
| triad_03 | **no** | 1 | 0 | — | — | 90:90 |
| triad_04 | **no** | **1** | **1** | 1 | **.889** | 90:90 |
| triad_05 | yes | 1 | 1 | 0 | **.458** | 60:90 |
| triad_06 | yes | 1 | 1 | — | .852 | 60:90 |
| triad_07 | yes | 1 | 1 | 0 | .913 | 60:90 |
| triad_08 | yes | 1 | 1 | 0 | — | 60:90? |
| triad_09 | yes | 1 | 1 | — | .875 | 60:90 |
| triad_10 | yes | — | — | — | .908 | 60:90 |
| triad_11 | yes | — | — | — | **.530** | 60:90 |
| triad_12 | yes | — | — | — | .855 | 60:90 |
| triad_13 | yes | — | — | — | .820 | 60:90 |
| triad_14 | yes | — | — | — | **.517** | 60:90 |

### What the table shows

1. **`Valid for gaze?` does not decide inclusion.** `dyad_02`, `dyad_10` and `triad_02`
   are flagged **0** and are in the gaze-based analysis; `dyad_01`, `dyad_08`,
   `dyad_09`, `triad_03` carry the same 0 and are out. The flag is either not the
   criterion, or it was overridden four times without a record.
2. **`triad_04` is excluded although both validity flags are 1 and its α is .889.** The
   only reason on file is a free-text note ("The Participant in red is stuck in the
   floor the whole time. Seems like they look at each other despite that") — the note
   itself argues the data is usable.
3. **`P Flying` is populated for four groups only** (`triad_04`=1; `dyad_09`,
   `triad_05`, `triad_07`, `triad_08`=0), and two of the zeros are *in* the analysis
   while `dyad_09`'s zero is *out*. The paper's "four more groups were removed as
   technical errors changed the virtual height of one participant" cannot be read off
   this column.
4. **`dyad_01`, the group with the highest agreement in the whole dataset (α = .967), is
   excluded.**
5. **The outlier removal (§11.5) targets `dyad_09`, and does not survive the label
   correction** — it is less extreme than two groups that were kept.

### The agreement number in the paper is a mean, and it hides the range

The 19 included groups with an α on file average **exactly .772** — the value §4.1
reports as "a high interrater agreement of .772". It is the **mean of per-group
Cronbach's alphas**, range **.458 – .913**, and four analysed groups sit below .60:

| group | α |
|---|---|
| triad_05 | .458 |
| triad_14 | .517 |
| triad_11 | .530 |
| dyad_10 | .551 |

Reporting a single pooled ".772 = high" for a set that includes an α of .458 will not
survive a reviewer who asks for the distribution — and R1 already asked for a
participant/group flow table, which is where this would surface. Report the mean **with
the range and n**, and note that four groups fall below conventional thresholds.

### Two more undocumented per-group interventions (not exclusions, but non-uniform)

6. **`triad_08`'s interaction window was manually extended**: "[added more minutes at the
   end as they are still discussing, even though prob low cog]". One group therefore has
   a different window definition from the other 19. It is also the group with the lowest
   mean TE (.078) and one of the two that make `dyad_09`'s outlier removal inconsistent.
7. **Six of the twenty analysed groups carry an open provenance question** in the
   ledger: `dyad_11`, `triad_10`, `triad_11` — "We don't seem to have a group_features
   file for this group?? How was this included in the blink calculations?" (`triad_12`,
   `triad_13`, `triad_14` share the missing-flags pattern). Their blink features exist in
   the gaze file regardless.
8. **Annotator naming.** Resolved (saveli, 2026-08-13): the ledger's
   "Annotation Freq - **Laura**:Helen" column is correct — **Laura's annotation was
   saved under Carlos's NOVA user**, so `*.carlosgonzalez.csv` files hold Laura's
   track. The two annotators are Laura and Helen; the paper must name them (or their
   roles) accordingly and not imply Carlos annotated. The column itself is the
   60 Hz/90 Hz split that `b2_rate_split.csv` already treats as confounded with group
   type.

### What to do

- **Report the flow.** R1 C3 asked for a participant/group flow table; build it from this
  ledger, with the reason per dropped group. This is required either way.
- **State one criterion and apply it.** As it stands the 25 → 20 step is not
  reconstructible from any column in the file. Either (a) re-include the groups that no
  defensible criterion excludes — `triad_04` (both flags valid) and `dyad_09` (outlier
  status gone after the label fix) — or (b) define the criterion in prose and accept
  that three included groups violate it.
- **Run the sensitivity analysis.** Both variants are a one-line swap of the group list
  (`group_names_with_time_floorlevel.csv` → `group_names_with_time_subsetsFullVERSION.csv`):
  all 25 groups / 243 windows, and the 20-group set minus the four α<.60 groups. If the
  headline survives both, the exclusion debate stops mattering and the paper gains a
  robustness section instead of a weakness.
- **Fix §4.1's agreement sentence** to mean + range + n.

None of this changes tonight's abstract — the 20-group numbers stand as the reported
analysis. It changes what the paper has to disclose, and it is much better found now
than by a reviewer.

---

## 13. Exclusion sensitivity — the experiment (run 2026-08-13)

All 25 groups are fully annotated: both annotators' TE tracks (Helen, and **Laura under
Carlos's NOVA user**), 2–3 role transcripts, merged CSV + parquet, and gaze features
already present in `60s_TE_correlation_2026-05-19.csv`. **243 windows, zero NaN labels.**
Nothing is missing in NOVA — the analysis was never limited to 21 sessions by data
availability.

`c6_build_groupset.py` rebuilds the matrix for any subset and reproduces the committed
`c2_features.csv` **bit-exactly** on the analysed 20 (0 differing cells across all 7
linguistic + 8 gaze columns, `y` and `our_te`). `c7_group_sensitivity.py` then reran 6
feature sets × 3 splits × 4 group sets = **72 cells**, same protocol; the 18 `g20` cells
reproduce `c2_paper_table.csv` exactly.

| set | groups | windows | definition |
|---|---|---|---|
| g20 | 20 | 185 | the analysed set (validation) |
| g21 | 21 | 195 | + `dyad_09`, the group dropped as a TE outlier |
| g25 | 25 | 243 | every group with usable data |
| g16 | 16 | 153 | g20 minus the four groups with per-group α < .60 |

### Headline (macro-F1; ** = significant under both tests)

| Split | Detector | g20 | g21 | g25 | g16 |
|---|---|---|---|---|---|
| **All** | GazexSpeaking | .736** | .753** | .755** | .767** |
| | linguistic | .764** | .779** | .760** | .789** |
| | linguistic+AIED | .776** | .779** | .767** | .755** |
| | **linguistic+GazexSpeaking** | **.786**\*\* | **.788**\*\* | **.769**\*\* | **.797**\*\* |
| **Dyads** | GazexSpeaking | .795 | .815** | .716 | .741** |
| | linguistic | .651 | .722** | .737** | .525 |
| | linguistic+AIED | .804** | .797** | .754** | .759** |
| | **linguistic+GazexSpeaking** | **.824**\*\* | .762** | **.771**\*\* | .732** |
| **Triads** | GazexSpeaking | .715 | .715 | .670 | .826** |
| | linguistic | .773** | .773** | .775** | .864** |
| | linguistic+AIED | .753** | .753** | .767** | .815** |
| | **linguistic+GazexSpeaking** | **.773**\*\* | **.773**\*\* | .749** | **.889**\*\* |

### What survives

1. **`linguistic+GazexSpeaking` is significant under both tests in all 12 cells** — every
   split × every group set. The headline detector never depends on an exclusion
   decision. That is the single most useful sentence this experiment produces.
2. **The dissociation holds in all four group sets.** Gaze's contribution over
   transcript-only (macro-F1) is positive in dyads everywhere and ≈ zero in triads
   everywhere:

   | | g20 | g21 | g25 | g16 |
   |---|---|---|---|---|
   | Dyads | +.173 | +.040 | +.034 | +.207 |
   | Triads | ±.000 | ±.000 | −.026 | +.025 |

   The *direction* is robust; the *magnitude* in dyads swings between +.03 and +.21
   depending on which dyads are in. State the direction as the finding and the
   magnitude with a range.
3. **One claim does not survive and must not be made:** on g25, gaze *alone* in dyads
   (.716) is **worse** than transcript alone (.737). "Gaze for dyads" is a statement
   about what gaze *adds*, never about gaze standing on its own.

### R1 Q3 answered: including the excluded dyad makes dyads *stronger*

`g21` vs `g20`:

- **Overall is unchanged** (.786 → .788 macro-F1) and Triads is identical by
  construction.
- **GazexSpeaking gains significance in dyads** — perm p .083 → **.017**, so it now
  clears both tests, which it never did in the reported analysis. macro-F1 .795 → .815.
- **The transcript detector becomes significant in dyads** for the first time (.651 →
  .722, perm .027, t-test .008).
- The dyad fusion drops (.824 → .762) and the ranking inside dyads reshuffles.

Mechanism: `dyad_09` is the lowest-TE dyad, so adding it moves the dyad majority floor
from .595 to .528. A harder floor with a better-balanced target is exactly what makes
the tests more informative. The honest summary for the rebuttal is: *"we reran with the
excluded group included; the conclusions hold, and both dyad detectors become
significant under both tests."*

### Annotation quality is the biggest single lever

`g16` — dropping only the four groups where the two annotators agree least (α < .60) —
lifts Triads from macro-F1 .773 to **.889** (acc .895, d = 2.92) and Overall from .786 to
.797, on 32 fewer windows. GazexSpeaking in triads goes from non-significant (.715) to
.826\*\*. Dyads get slightly worse, but that set is down to 7 groups.

That is a result in its own right and worth a paragraph: **detector performance tracks
inter-annotator agreement.** Where the annotators agree on what task engagement looks
like, a seven-feature transcript detector reaches .864 macro-F1 in triads. Where they do
not, the ceiling is the label noise, not the model. It reframes "small dataset" as "label
quality", which is both truer and more actionable.

### Recommendation

**Report `g21` (21 groups / 195 windows) as the main analysis**, with g20, g25 and g16 as
a robustness table.

- It removes the one exclusion that cannot be defended — the outlier criterion does not
  hold on the corrected labels (§11.5) — while keeping the four exclusions that have a
  stated technical cause (avatar mispositioned → gaze geometry invalid).
- It answers R1 Q3 by construction rather than by rebuttal prose.
- Overall and Triads headlines are unchanged; Dyads gains two newly significant
  detectors.
- Cost: the dyad fusion headline moves from .842 acc / .824 macro-F1 to .781 / .762.

If a single number matters more than defensibility, keep g20 and add g21/g25/g16 as the
robustness table — but then §4 must state the exclusion criteria in a form that actually
matches the data, which §12 shows it currently does not.

Outputs: `c7_sensitivity.csv` (acc ± SD, majority, perm p, t, p, Cohen's d, top-2 models
per cell) and `c7_sensitivity_prf.csv` (confusion matrix, per-class P/R/F1, macro-F1,
balanced accuracy, MCC).
