"""C2 — paper tables for the IUI revision (Carlos's asks, 2026-07-06 meeting).

Per (split x feature-set), canonical group-level protocol (his 185 floorlevel
windows, clean 2-annotator-mean labels, nested LOGO model selection):
  - per-fold accuracies for ALL 4 panel models  -> a1_out/c2_foldscores.csv
  - top-2 models: acc mean +- std, one-sample t-test vs majority (t, p, Cohen's d)
  - permutation p for the best model (300 perms, default-hyperparam pipeline —
    same convention as ab_common.perm_p / comprehensive_comparison)
  - majority baseline as its own per-fold row (train-majority -> test fold)
  -> a1_out/c2_paper_table.csv  (+ c2_feature_lists.md for the paper appendix)

NEW feature sets vs comprehensive_table: linguistic (surface transcript stats
windowed onto his grid) + openSMILE 88 eGeMAPS functionals (windowed from the
merged parquets) + openSMILE_pca (PCA inside the nested pipeline, n_components
tuned in the inner CV — no leakage) + audio+openSMILE.

Driver:  PYTHONPATH=discover/analysis_scripts python3 discover/analysis_scripts/c2_paper_tables.py
Worker:  ... c2_paper_tables.py <split> <set>       (run from REPO ROOT)
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import warnings
warnings.filterwarnings("ignore")
import glob
import sys
import subprocess
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = "scratch/te_adoption/a1_out"  # gitignored workbench; final CSVs are copied to discover/paper_results
CACHE = f"{OUT}/c2_features.csv"
RESULTS = f"{OUT}/c2_paper_table.csv"
FOLDS = f"{OUT}/c2_foldscores.csv"

sys.path.insert(0, HERE)
import ab_common as ab

# 2026-07-07 final paper set (segments = raw ASR chunks; words_per_second dropped
# as wc/60 duplicate; ratios renamed; NEW unfinished_ratio = trail-off/abandoned
# segments: ends '...' OR no .!? terminal and not same-speaker-continued <2s)
LING = ["segment_count", "word_count", "avg_word_length", "question_ratio",
        "speech_ratio", "mean_segment_duration_s", "unfinished_ratio"]
# consensus core from the _sel experiment (universally picked features). NB: its
# accuracy is a DESCRIPTIVE post-hoc number (set chosen from all-fold selection
# frequencies) — cite _sel for the unbiased lean-set estimate.
LING_CORE = ["word_count", "question_ratio", "speech_ratio", "mean_segment_duration_s"]
SPLITS = ("All", "D", "T")
SETS = ("AIxVR", "GazexSpeaking", "audio", "linguistic", "openSMILE",
        "openSMILE_pca", "audio+AIxVR", "audio+GazexSpeaking", "audio+openSMILE",
        # 2026-07-07 additions (slide 13/18 group-level rebuild, indiv-TE cut)
        "audio+linguistic", "linguistic+GazexSpeaking", "audio+linguistic+GazexSpeaking",
        "emow2v_pca", "sentemb_pca")
EMB_CACHE = f"{OUT}/c2_emb_features.csv"


def osm_cols(df):
    return [c for c in df.columns if c.startswith("opensmile_")]


def feats_for(det, sp, osm, emo=(), semb=()):
    aix = ab.AIXVR[sp]
    return {"AIxVR": aix, "GazexSpeaking": ab.GXS, "audio": ab.AUDIO,
            "linguistic": LING, "openSMILE": osm, "openSMILE_pca": osm,
            "audio+AIxVR": ab.AUDIO + aix, "audio+GazexSpeaking": ab.AUDIO + ab.GXS,
            "audio+openSMILE": ab.AUDIO + osm,
            "linguistic_core": LING_CORE,
            "audio+linguistic": ab.AUDIO + LING,
            "linguistic+GazexSpeaking": LING + ab.GXS,
            "linguistic+AIxVR": LING + aix,
            "audio+linguistic+GazexSpeaking": ab.AUDIO + LING + ab.GXS,
            "emow2v_pca": list(emo), "sentemb_pca": list(semb)}[det]


def build_emb():
    """Window emow2v (1024, group stream) + sentiment emb (512 per role -> role-nanmean)
    onto his (group, sec) grid. Cached to EMB_CACHE (keyed group_name+sec)."""
    if os.path.exists(EMB_CACHE):
        return pd.read_csv(EMB_CACHE)
    his = ab.build_features()[["group_name", "sec"]]
    bounds = _window_bounds()
    first = pd.read_parquet(f"data/discover/merged/recording_{his.group_name.iloc[0]}.parquet")
    EMO = sorted(c for c in first.columns if c.startswith("emow2v_"))
    SEMB_ALL = sorted(c for c in first.columns if c.startswith("sentiment_emb_"))
    ndim = 512
    roles = sorted({c.split("_")[3] for c in SEMB_ALL})  # sentiment_emb_p_<role>_<k>
    out = {}
    for g, gg in his.groupby("group_name"):
        fr = pd.read_parquet(f"data/discover/merged/recording_{g}.parquet",
                             columns=["time_ms"] + EMO + SEMB_ALL)
        s0 = bounds[g]
        for idx in gg.index:
            w0 = s0 + his.loc[idx, "sec"] * 1000.0
            m = (fr.time_ms >= w0) & (fr.time_ms < w0 + 60_000.0)
            emo = fr.loc[m, EMO].mean().values
            per_role = np.stack([fr.loc[m, [f"sentiment_emb_p_{r}_{k:04d}" for k in range(ndim)]].mean().values
                                 for r in roles if f"sentiment_emb_p_{r}_0000" in fr.columns])
            semb = np.nanmean(per_role, axis=0)
            out[idx] = np.concatenate([emo, semb])
        print(f"emb windowed {g}", flush=True)
    cols = [f"emo_{i:04d}" for i in range(len(EMO))] + [f"semb_{k:04d}" for k in range(ndim)]
    emb = pd.DataFrame(out, index=cols).T.sort_index()
    emb[["group_name", "sec"]] = his[["group_name", "sec"]]
    nnan = int(emb[cols].isna().sum().sum())
    if nnan:
        print(f"emb: {nnan} NaN cells -> column-mean fill", flush=True)
        emb[cols] = emb[cols].fillna(emb[cols].mean())
    emb.to_csv(EMB_CACHE, index=False)
    print(f"cached {EMB_CACHE}: {emb.shape}", flush=True)
    return emb


def _window_bounds():
    FLOOR = pd.read_csv("data/group_names_with_time_floorlevel.csv", sep=";")
    RT = pd.read_csv("data/recording_times_group_info.csv")
    rec = dict(zip(RT["long_name"], pd.to_datetime(RT["start_recording"])))
    return {r["Group_Name"]: (pd.to_datetime(r["TS_Start_Interaction"]) - rec[r["Group_Name_Long"]]).total_seconds() * 1000.0
            for _, r in FLOOR.iterrows() if r["Group_Name_Long"] in rec}


def build_all():
    """ab_common features + openSMILE (parquet frames -> window mean) + linguistic
    (raw transcripts -> window aggregates). Cached to CACHE."""
    if os.path.exists(CACHE):
        return pd.read_csv(CACHE)
    his = ab.build_features()
    bounds = _window_bounds()

    # openSMILE: window the 88 frame-level functionals onto his (group, sec) grid
    first = pd.read_parquet(f"data/discover/merged/recording_{his.group_name.iloc[0]}.parquet")
    OSM = osm_cols(first)
    vals = {}
    for g, gg in his.groupby("group_name"):
        fr = pd.read_parquet(f"data/discover/merged/recording_{g}.parquet",
                             columns=["time_ms"] + OSM)
        s0 = bounds[g]
        for idx in gg.index:
            w0 = s0 + his.loc[idx, "sec"] * 1000.0
            m = (fr.time_ms >= w0) & (fr.time_ms < w0 + 60_000.0)
            vals[idx] = fr.loc[m, OSM].mean().values
    his[OSM] = pd.DataFrame(vals, index=OSM).T
    nnan = int(his[OSM].isna().sum().sum())
    if nnan:
        print(f"openSMILE: {nnan} NaN cells -> column-mean fill", flush=True)
        his[OSM] = his[OSM].fillna(his[OSM].mean())

    # linguistic: surface stats from raw transcripts, same windowing as idea3_build_text
    for c in LING:
        his[c] = 0.0
    for g, gg in his.groupby("group_name"):
        segs = []
        for path in glob.glob(f"data/discover/recording_{g}/transcript.*.helenrisack.csv"):
            tr = pd.read_csv(path)
            for _, s in tr.iterrows():
                segs.append((float(s["from"]), float(s["to"]), str(s["name"]).strip()))
        s0 = bounds[g]
        for idx in gg.index:
            w0 = s0 + his.loc[idx, "sec"] * 1000.0
            w1 = w0 + 60_000.0
            win = [(max(f, w0), min(t, w1), txt) for f, t, txt in segs if t > w0 and f < w1 and txt]
            if not win:
                continue
            words = [w for _, _, txt in win for w in txt.split()]
            durs = [(t - f) / 1000.0 for f, t, _ in win]
            his.loc[idx, "segment_count"] = len(win)
            his.loc[idx, "word_count"] = len(words)
            his.loc[idx, "words_per_second"] = len(words) / 60.0
            his.loc[idx, "avg_word_length"] = float(np.mean([len(w) for w in words])) if words else 0.0
            his.loc[idx, "question_rate"] = float(np.mean([("?" in txt) for _, _, txt in win]))
            his.loc[idx, "speaking_seconds"] = float(np.sum(durs))
            his.loc[idx, "mean_segment_duration_s"] = float(np.mean(durs))
    his.to_csv(CACHE, index=False)
    print(f"cached {CACHE}: {len(his)} rows, {len(his.columns)} cols", flush=True)
    return his


def perfold(sub, feats, model, pca=False, sel=False, sel_log=None):
    """Per-fold LOGO accs for one model (inner GridSearch); mirrors ab.perfold_accs
    but returns (test_group, acc) pairs. pca: tuned PCA step. sel: tuned SelectKBest
    (f_classif, k full-range 1..n_feat) — per-split feature selection INSIDE the CV;
    selected features per fold appended to sel_log (list) when given."""
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.model_selection import GridSearchCV, GroupKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    X = sub[feats].values
    y = sub["y"].values
    g = sub["group_name"].values
    clf, grid = ab.PANEL[model]
    steps = [("s", StandardScaler())]
    if pca:
        steps.append(("p", PCA(random_state=ab.RS)))
    if sel:
        steps.append(("k", SelectKBest(f_classif)))
    steps.append(("m", clf))
    pgrid = {f"m__{k}": v for k, v in grid.items()}
    if pca:
        pgrid["p__n_components"] = [5, 10, 20]
    if sel:
        n = len(feats)
        if n <= 20:
            pgrid["k__k"] = list(range(1, n + 1))
        else:  # high-dim (openSMILE 88): log-spaced grid, dense low end — acc(k)
            # is too noisy for binary search; coarse coverage + inner-CV pick
            pgrid["k__k"] = sorted({k for k in (1, 2, 3, 4, 6, 8, 11, 16, 22, 32, 45, 64, n) if k <= n})
    gkf = GroupKFold(n_splits=pd.Series(g).nunique())
    out = []
    for tr, te in gkf.split(X, y, groups=g):
        if pd.Series(y[tr]).nunique() < 2:
            continue
        inner = GroupKFold(n_splits=int(pd.Series(g[tr]).nunique()))
        gs = GridSearchCV(Pipeline(steps), pgrid, cv=inner, n_jobs=1, error_score=np.nan)
        gs.fit(X[tr], y[tr], groups=g[tr])
        tg = str(pd.Series(g[te]).iloc[0])
        out.append((tg, float((gs.best_estimator_.predict(X[te]) == y[te]).mean())))
        if sel and sel_log is not None:
            mask = gs.best_estimator_.named_steps["k"].get_support()
            sel_log.append((model, tg, int(mask.sum()), "|".join(f for f, m in zip(feats, mask) if m)))
    return out


def perm_p_c2(sub, feats, model, pca=False, sel=False):
    """Permutation p, best model, default hyperparams (ab.perm_p convention);
    PCA variant fixes n_components=10; sel variant fixes SelectKBest k=min(5, n)."""
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.model_selection import GroupKFold, permutation_test_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    X = sub[feats].values
    y = sub["y"].values
    g = sub["group_name"].values
    clf = ab.PANEL[model][0]
    steps = [("s", StandardScaler())]
    if pca:
        steps.append(("p", PCA(n_components=10, random_state=ab.RS)))
    if sel:
        steps.append(("k", SelectKBest(f_classif, k=min(5, len(feats)))))
    steps.append(("m", clf))
    _, _, pp = permutation_test_score(Pipeline(steps), X, y, groups=g,
                                      cv=GroupKFold(n_splits=pd.Series(g).nunique()),
                                      scoring="accuracy", n_permutations=300,
                                      random_state=ab.RS, n_jobs=1)
    return float(pp)


def sub_for(his, sp):
    return his if sp == "All" else his[his["grp"] == sp]


def write_feature_lists():
    with open(f"{OUT}/c2_feature_lists.md", "w") as f:
        f.write("# Classifiers & feature sets (canonical group-level protocol, 185 windows)\n\n")
        f.write("## Classifier panel (identical for every cell)\n\n")
        f.write("Nested leave-one-group-out model selection: outer LOGO fold -> inner LOGO\n")
        f.write("GridSearchCV over the panel below; pipeline = StandardScaler -> [PCA or\n")
        f.write("SelectKBest where the set says so] -> model. Reported: best (rank-1) and\n")
        f.write("second-best (rank-2) model by mean outer-fold accuracy. seed=42.\n\n")
        f.write("| model | hyperparameter grid |\n|---|---|\n")
        for name, (clf, grid) in ab.PANEL.items():
            g = {k: ("np.logspace(0,-9,50)" if k == "var_smoothing" else v) for k, v in grid.items()}
            f.write(f"| {name} ({type(clf).__name__}) | `{g}` |\n")
        f.write("\nQDA is skipped on high-dimensional sets (singular class covariance); the\n")
        f.write("permutation test (300 perms) runs the best model with default hyperparams.\n\n")
        f.write("## Feature sets\n\n")
        f.write(f"- **AIxVR** gaze (per split): All={ab.AIXVR['All']} / D={ab.AIXVR['D']} / T={ab.AIXVR['T']}\n")
        f.write("  — mirrors the prior pipeline's `features_label=\"AIxVR\"` runs\n")
        f.write("  (training/vilearn_windowed_train_ML.py, '### AIxVR FEATURES ###' block,\n")
        f.write("  'according to AIxVR paper'). NB the 4-feature gaze set\n")
        f.write("  ['MG','1d_DG','BPM','blink_durations'] is that script's separate **AIED**\n")
        f.write("  baseline ('### AIED FEATURES (BLINKS + GAZE) ###', features_label=\"AIED\"),\n")
        f.write("  same 4 features for all splits — a different prior, not AIxVR.\n")
        f.write(f"- **GazexSpeaking** gaze (same all splits): {ab.GXS}\n")
        f.write(f"- **audio** (estimated affect, v/a/d): {ab.AUDIO}\n")
        f.write("- **linguistic** — 7 surface stats per group-window (no model-derived sentiment):\n\n")
        f.write("  | feature (paper name) | column / former name | definition (per 60 s group-window) |\n")
        f.write("  |---|---|---|\n")
        f.write("  | segments / window | `segment_count` | # ASR segments (all speakers) |\n")
        f.write("  | words / window | `word_count` (absorbs former `words_per_second` = wc/60) | total words |\n")
        f.write("  | avg word length | `avg_word_length` | mean characters per word |\n")
        f.write("  | question ratio | `question_ratio` (former `question_rate`) | fraction of segments containing '?' |\n")
        f.write("  | speech ratio | `speech_ratio` (former `speaking_seconds`/60) | summed speech time / 60 s |\n")
        f.write("  | mean segment duration | `mean_segment_duration_s` | mean segment length, s |\n")
        f.write("  | unfinished ratio | `unfinished_ratio` (NEW) | segments ending '…' or unterminated and not same-speaker-continued <2 s |\n\n")
        f.write("  Naming footnote: earlier per-role pipelines computed these PER SEGMENT/ROLE; the move to\n")
        f.write("  group-window aggregation turned counts into per-window totals and rates into ratios —\n")
        f.write("  hence the renames. `words_per_second` was dropped: with a fixed 60 s window it is exactly\n")
        f.write("  `word_count`/60 (identical after standardization). Segments are raw Whisper ASR chunks,\n")
        f.write("  NOT sentences (median 1.7 s; 66% end in terminal punctuation).\n\n")
        f.write("- **linguistic_core** (consensus/deployment subset, see selection appendix): "
                f"{LING_CORE}\n")
        f.write("- **openSMILE**: 88 eGeMAPS functionals (opensmile_* in merged parquets), window means\n")
        f.write("- **openSMILE_pca / emow2v_pca / sentemb_pca**: PCA in nested pipeline, n_components tuned {5,10,20} inner-CV\n")
        f.write("- **emow2v** audio embedding (1024-d group stream), **sentemb** text sentiment embedding (512-d, role-mean)\n")
        f.write("- fusions = concatenation of the above; `_sel` variants = SelectKBest(f_classif), k tuned 1..n inner-CV\n")


if __name__ == "__main__" and len(sys.argv) == 2 and sys.argv[1] == "featurelists":  # standalone regen
    write_feature_lists()
    print(f"wrote {OUT}/c2_feature_lists.md")
    sys.exit(0)

if __name__ == "__main__" and len(sys.argv) == 2:  # guard: stray single arg must NOT fall into the driver
    sys.exit(f"unknown arg {sys.argv[1]!r}; worker needs <split> <detector>, driver takes none")

if __name__ == "__main__" and len(sys.argv) == 3:  # ---------------- worker ----------------
    from scipy.stats import ttest_1samp
    sp, det = sys.argv[1], sys.argv[2]
    his = build_all()
    OSM = osm_cols(his)
    emo, semb = [], []
    base_det = det[:-4] if det.endswith("_sel") else det
    if base_det in ("emow2v_pca", "sentemb_pca"):
        emb = build_emb()
        emo = [c for c in emb.columns if c.startswith("emo_")]
        semb = [c for c in emb.columns if c.startswith("semb_")]
        his = his.merge(emb, on=["group_name", "sec"], validate="one_to_one")
    sub = sub_for(his, sp)
    feats = feats_for(base_det, sp, OSM, emo, semb)
    pca = base_det.endswith("_pca")
    sel = det.endswith("_sel")
    maj = max(sub.y.mean(), 1 - sub.y.mean())

    per_model = {}
    sel_log = [] if sel else None
    for model in ab.PANEL:
        try:
            pairs = perfold(sub, feats, model, pca=pca, sel=sel, sel_log=sel_log)
        except Exception as e:  # QDA singular covariance on high-dim sets etc.
            print(f"skip {sp} {det} {model}: {type(e).__name__}", flush=True)
            continue
        per_model[model] = pairs
    if sel and sel_log:
        with open(f"{OUT}/c2_selection.csv", "a") as f:
            for model, tg, k, picked in sel_log:
                f.write(f"{sp},{det},{model},{tg},{k},{picked}\n")
    # per-fold accs for ALL models (was lost in the sel-experiment edit — the
    # sel branch above used to write only the LAST model's pairs)
    with open(FOLDS, "a") as f:
        for model, mp in per_model.items():
            for i, (grp, acc) in enumerate(mp):
                f.write(f"{sp},{det},{model},{i},{grp},{acc:.6f}\n")
    if not per_model:
        print(f"FAIL {sp} {det}: all models failed", flush=True)
        sys.exit(1)
    ranked = sorted(per_model, key=lambda m: -np.mean([a for _, a in per_model[m]]))
    try:
        pp = perm_p_c2(sub, feats, ranked[0], pca=pca, sel=sel)
    except Exception as e:
        print(f"perm failed {sp} {det} {ranked[0]}: {type(e).__name__}", flush=True)
        pp = np.nan
    rows = []
    for rank, model in enumerate(ranked[:2], 1):
        accs = np.array([a for _, a in per_model[model]])
        t, p = ttest_1samp(accs, maj)
        sd = accs.std(ddof=1)
        d = (accs.mean() - maj) / sd if sd > 0 else np.nan
        rows.append(f"{sp},{det},{len(sub)},{len(feats)},{rank},{model},"
                    f"{accs.mean():.3f},{sd:.3f},{maj:.3f},"
                    f"{(f'{pp:.4f}' if rank == 1 else '')},{t:.3f},{p:.4f},{d:.3f}")
    with open(RESULTS, "a") as f:
        f.write("\n".join(rows) + "\n")
    print(f"OK {sp} {det} best={ranked[0]} "
          f"{np.mean([a for _, a in per_model[ranked[0]]]):.3f} perm_p={pp:.4f}", flush=True)
    sys.exit(0)

# ---------------- driver ----------------
# GUARDED: a bare `import c2_paper_tables` used to run this block and WIPE
# RESULTS/FOLDS (happened 2026-07-10 via d2_pca_dims.py). Cells now run as a
# parallel pool of single-threaded subprocesses (thread-parallelism inside one
# process segfaults libgomp; process isolation is the safe axis).
if __name__ == "__main__":
    from concurrent.futures import ThreadPoolExecutor
    from sklearn.model_selection import GroupKFold

    his = build_all()
    OSM = osm_cols(his)
    with open(RESULTS, "w") as f:
        f.write("split,detector,n,n_feat,rank,model,acc_mean,acc_std,majority,"
                "perm_p,t_maj,p_ttest_maj,cohen_d_maj\n")
    with open(FOLDS, "w") as f:
        f.write("split,detector,model,fold,test_group,accuracy\n")

    # per-fold majority baseline (train-majority class -> test fold)
    for sp in SPLITS:
        sub = sub_for(his, sp)
        y = sub["y"].values
        g = sub["group_name"].values
        accs = []
        gkf = GroupKFold(n_splits=pd.Series(g).nunique())
        for tr, te in gkf.split(y.reshape(-1, 1), y, groups=g):
            m = 1.0 if y[tr].mean() >= 0.5 else 0.0
            accs.append((y[te] == m).mean())
            with open(FOLDS, "a") as f:
                f.write(f"{sp},baseline_majority,majority,{len(accs)-1},{pd.Series(g[te]).iloc[0]},{accs[-1]:.6f}\n")
        accs = np.array(accs)
        maj = max(y.mean(), 1 - y.mean())
        with open(RESULTS, "a") as f:
            f.write(f"{sp},baseline_majority,{len(sub)},0,1,majority,"
                    f"{accs.mean():.3f},{accs.std(ddof=1):.3f},{maj:.3f},,,,\n")
        print(f"baseline {sp}: {accs.mean():.3f} +- {accs.std(ddof=1):.3f} (floor {maj:.3f})", flush=True)

    env = dict(os.environ, PYTHONPATH=HERE)

    def run_cell(job):
        sp, det = job
        p = subprocess.run([sys.executable, f"{HERE}/c2_paper_tables.py", sp, det],
                           env=env, capture_output=True, text=True)
        print((p.stdout.strip() or f"FAIL {sp} {det}: {p.stderr[-300:]}"), flush=True)

    jobs = [(sp, det) for sp in SPLITS for det in SETS]
    with ThreadPoolExecutor(max_workers=min(12, os.cpu_count() or 4)) as ex:
        list(ex.map(run_cell, jobs))

    write_feature_lists()
    print("done — see", RESULTS, flush=True)
