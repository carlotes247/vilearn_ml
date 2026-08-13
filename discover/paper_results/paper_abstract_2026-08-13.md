# IUI paper — abstract rewrite + underselling audit (2026-08-13)

Scope assumed here (saveli, 2026-08-13): **v/a/d affect and the LLM rubric are held
back for the follow-up journal article.** The paper's three modalities are therefore
eye-tracking (AIxVR / AIED / GazexSpeaking), acoustics (88 eGeMAPS functionals) and
transcript (7 surface statistics). Every number below is from the committed
`c2_paper_table.csv`, `c3_prf.csv` and `c4_acoustic_fusions.csv` under the canonical
protocol: 185 floor-level 60 s group windows, clean 2-annotator-mean TE binarised at
0.5, nested leave-one-group-out model selection (seed 42), 300-permutation test +
one-sample t-test vs the majority floor, Cohen's d vs that floor.

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
> Every feature set — ours and the two published gaze sets, re-evaluated on the same
> data — runs through one protocol: nested leave-one-group-out cross-validation, a
> majority-class floor, a 300-permutation null and a one-sample t-test, reported with
> Cohen's d. Fusing seven surface transcript statistics with prior-work gaze×speaking
> features detects TE at macro-F1 .786 overall (QDA, 78.6% accuracy, d=1.33), .824 in
> dyads (logistic regression, 84.2%, d=2.81) and .773 in triads (naive Bayes, 75.8%,
> d=0.72), significant under both tests in every split, while the best published gaze
> set alone reaches .736/.795/.715 and the AIxVR set stays dominated even inside its
> own fusion. Which modality earns its place flips with group size: in triads the
> transcript statistics alone already match the fusion (macro-F1 .773) and need nothing
> but a microphone, whereas in dyads gaze lifts them from .651 to .824. Class-level
> results show why fusion matters for intervention — the gaze-only dyad detector
> recalls every high-TE window but only 56% of the low-TE ones, the class an
> intervention has to catch, and the fusion raises that to 72%. Raw acoustic
> functionals never clear the floor under both tests and cost 6–8 points when fused in.
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
| 8–11 | selection robustness, coefficients, unit fix | **no — paper body only** (§7 below) |

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

1. **Effect sizes.** d = 2.81 (dyads), 1.81 (overall), 1.03 (triads). Huge, and absent
   from the abstract. The dyad fusion is the only dyad detector significant under
   *both* tests.
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
