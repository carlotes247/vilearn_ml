# Classifiers & feature sets (canonical group-level protocol, 185 windows)

## Classifier panel (identical for every cell)

Nested leave-one-group-out model selection: outer LOGO fold -> inner LOGO
GridSearchCV over the panel below; pipeline = StandardScaler -> [PCA or
SelectKBest where the set says so] -> model. Reported: best (rank-1) and
second-best (rank-2) model by mean outer-fold accuracy. seed=42.

| model | hyperparameter grid |
|---|---|
| QDA (QuadraticDiscriminantAnalysis) | `{'reg_param': [0.1, 0.2, 0.3, 0.4, 0.5]}` |
| SVM (SVC) | `{'kernel': ['linear', 'rbf'], 'C': [0.1, 1], 'gamma': [0.1, 0.01, 0.001]}` |
| LogReg (LogisticRegression) | `{'C': [0.01, 0.1, 1, 10]}` |
| NB (GaussianNB) | `{'var_smoothing': 'np.logspace(0,-9,50)'}` |

QDA is skipped on high-dimensional sets (singular class covariance); the
permutation test (300 perms) runs the best model with default hyperparams.

## Feature sets

- **AIxVR** gaze (per split): All=['MG', 'BPM'] / D=['MG', '1d_DG'] / T=['BPM']
  — mirrors the prior pipeline's `features_label="AIxVR"` runs
  (training/vilearn_windowed_train_ML.py, '### AIxVR FEATURES ###' block,
  'according to AIxVR paper'). NB the 4-feature gaze set
  ['MG','1d_DG','BPM','blink_durations'] is that script's separate **AIED**
  baseline ('### AIED FEATURES (BLINKS + GAZE) ###', features_label="AIED"),
  same 4 features for all splits — a different prior, not AIxVR.
- **AIED** gaze (same all splits): ['MG', '1d_DG', 'BPM', 'blink_durations'] — the prior 4-feature baseline above,
  reported as its own detector (+ audio/linguistic fusions).
- **GazexSpeaking** gaze (same all splits): ['BPM', 'blink_durations', 'G_OnSpeaker', 'No_G_OnSpeaker', 'G_SI', 'No_G_SI']
- **audio** (estimated affect, v/a/d): ['arousal', 'dominance', 'valence']
- **linguistic** — 7 surface stats per group-window (no model-derived sentiment):

  | feature (paper name) | column / former name | definition (per 60 s group-window) |
  |---|---|---|
  | segments / window | `segment_count` | # ASR segments (all speakers) |
  | words / window | `word_count` (absorbs former `words_per_second` = wc/60) | total words |
  | avg word length | `avg_word_length` | mean characters per word |
  | question ratio | `question_ratio` (former `question_rate`) | fraction of segments containing '?' |
  | speech ratio | `speech_ratio` (former `speaking_seconds`/60) | summed speech time / 60 s |
  | mean segment duration | `mean_segment_duration_s` | mean segment length, s |
  | unfinished ratio | `unfinished_ratio` (NEW) | segments ending '…' or unterminated and not same-speaker-continued <2 s |

  Naming footnote: earlier per-role pipelines computed these PER SEGMENT/ROLE; the move to
  group-window aggregation turned counts into per-window totals and rates into ratios —
  hence the renames. `words_per_second` was dropped: with a fixed 60 s window it is exactly
  `word_count`/60 (identical after standardization). Segments are raw Whisper ASR chunks,
  NOT sentences (median 1.7 s; 66% end in terminal punctuation).

- **linguistic_core** (consensus/deployment subset, see selection appendix): ['word_count', 'question_ratio', 'speech_ratio', 'mean_segment_duration_s']
- **openSMILE**: 88 eGeMAPS functionals (opensmile_* in merged parquets), window means
- **openSMILE_pca / emow2v_pca / sentemb_pca**: PCA in nested pipeline, n_components tuned {5,10,20} inner-CV
- **emow2v** audio embedding (1024-d group stream), **sentemb** text sentiment embedding (512-d, role-mean)
- fusions = concatenation of the above; `_sel` variants = SelectKBest(f_classif), k tuned 1..n inner-CV
