# Task-Engagement (TE) label discrepancies — pointers for review

While reproducing the TE labels under our protocol we hit a few discrepancies in the
engagement-label construction. Listing them here with exact code pointers so they can be
checked directly. Framing is technical, not a verdict — happy to be wrong on any of these.

Scope: the 60 s floorlevel TE labels and the QDA baselines trained on them
(`Recordings/SavedData/v2_no_low_sampled/60s_TE_correlation_2026-05-19.csv`).

---

## 1. Time-stretch bug — 90 Hz annotations processed as 60 Hz  ⭐ (the impactful one)

**File:** `preprocessing/engagement/engagement_processor.py`

- **Line 63:** `if os.path.isfile("{self.data_path}/{self.filename_2_90Hz}"):`
  Missing `f` prefix — line 59 has `f"..."` for annotator 1, line 63 does **not** for
  annotator 2. So this checks a literal string `"{self.data_path}/..."`, which never
  exists → the 90 Hz file for annotator 2 (carlosgonzalez) is never detected →
  `is_file_2_90Hz` stays `False` and `filename_2` falls back to the 60 Hz name.
- **Lines 67–68:** `if self.is_file_1_90Hz and self.is_file_2_90Hz: self.freq = 90`
  Because the annotator-2 check at line 63 always fails, `freq` stays **60** even for
  groups where both 90 Hz files actually exist.
- **Line 112:** `ts_secs = [x * (1/self.freq) for x in range(len(df_avg))]`
  Timestamps are then built at `freq = 60`, while annotator 1's data (helen, selected as
  the 90 Hz file at line 60) is genuinely 90 Hz → her track is **stretched ×1.5** in time.

**Effect:** for the 13/20 floorlevel groups annotated under the 60 Hz scheme where helen's
90 Hz file is present, the merged TE is temporally misaligned. The 2026-05-19 QDA baselines
for those groups were trained on time-corrupted labels.

**How to verify:** for a 60 Hz group, simulate the processing two ways — annotator 1 treated
as 60 Hz (the bug) vs correctly as 90 Hz — and correlate each against the saved
`task_engagement90Hz_avg_all.csv`. The bug simulation matches the saved file (corr ≈ 0.75–0.87);
the correct alignment does not (≈ 0). A pure 90 Hz control group (e.g. dyad_03) replicates
exactly (corr = 1.0000), confirming the issue is specific to the 60 Hz path.

---

## 2. `fillna(0)` before averaging the two annotators

**File:** `preprocessing/engagement/engagement_processor.py`

- **Line 108–109:**
  ```python
  df_merged = df_merged.fillna(0)
  df_avg = pd.DataFrame(df_merged.mean(axis=1), columns=['task_eng'])
  ```
  Missing values are filled with **0** (lowest engagement) before the mean, rather than
  NaN-skipped. Where one annotator has gaps, the average is pulled toward 0 — a single
  present annotator at 0.8 averages to 0.4, not 0.8.

**Suggested:** `df_merged.mean(axis=1, skipna=True)` without the `fillna(0)`.

---

## 3. Label-construction note (ours, not yours)

For completeness: our earlier pipeline used **helen-only** TE (≈74 % high), whereas the
correct construction is the **mean of both annotators** as done in this file (≈52 % high).
This was a mismatch on our side — we have since adopted the two-annotator mean. The old
"significantly beats prior" deltas were an artifact of that label skew and are retracted.

---

## 4. Minor — independent t-test used on paired fold scores (low impact)

**File:** `stats_tests/anova_ml_runs.py`, ~line 78:
```python
# _, p_val = ttest_rel(model_scores, base_scores)
_, p_val = ttest_ind(model_scores, base_scores, equal_var=False)
```
The model and baseline scores are paired by fold (same folds, `df_paired`), but the active
line is `ttest_ind` (Welch, independent) with `ttest_rel` commented out. An independent test
on paired data ignores the fold pairing → underpowered (loses the variance-reduction from
pairing). Suggested: use `ttest_rel(model_scores, base_scores)`. Low impact, but it affects
the reported p-values.

---

## Context — known differences (NOT bugs)

These are reconciliations, not errors — listed so the numbers line up.

- **Window count: 185 (yours) vs 197 (our derive).** Your floorlevel 60 s set =
  **185** group-windows (20 floorlevel groups). Our independent derive pipeline yields
  **197** — the difference is tail-window trimming (we keep partial/tail windows slightly
  differently). **The TE/model comparison is run on your exact 185-window grid**, so it
  stays apples-to-apples; the 197 only appears in our standalone panel. Tests paired by group.

- **Our clean two-annotator mean reproduces your distribution, but not exactly.** It matches
  your class balance (~52 % high) and correlates with your saved values on the **90 Hz
  groups** (corr ≈ 0.70–0.98). It does **not** match exactly because your construction
  differs: `fillna(0)` before the mean (§2) pulls averages down, and the time-stretch (§1)
  decorrelates the 13 60 Hz groups. So the residual gap is explained by §1 + §2 — the clean
  mean is the corrected version, not a different labeling choice.

---

*Reproduction scripts (local, not committed): `verify_stretch_bug.py`,
`sanity_check_clean_mean.py`, `validate_origin_alignment.py` — available on request.*
