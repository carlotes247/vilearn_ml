import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import ttest_rel, wilcoxon
import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.stats.multicomp as mc
from statsmodels.stats.multitest import multipletests
import os
import datetime

def run_stat_test(df_in: pd.DataFrame, models: list[str] ):
    # discriminate for wanted models
    df = df_in.loc[df_in['Model'].isin(models_selected)]
    # remove avg and std rows
    df = df[~df["Fold"].isin(["avg", "std"])]    
    print("df configured")

    # check for normality with saphiro test
    # apparently we need to run a saphiro test per model. Doing that
    results_saphiro = []
    for model, df_model in df.groupby("Model"):
        scores = df_model["Score"].values
        W, p = stats.shapiro(scores)
        results_saphiro.append({
        'Model': model,
        'W': W,
        'p-value': p,
        'n': len(scores)
        })

    normality_df = pd.DataFrame(results_saphiro)
    normality_df['Normal (α=0.05)'] = normality_df['p-value'] > 0.05
    # print(normality_df)
    if (normality_df['Normal (α=0.05)'].mean()*100 < 70):
        raise Exception("normality not met!")

    # We assume normality at this point
    # We run a simple t-test for the moment

    # Get baseline for multiple paired t-test
    baseline_name = "Baseline Uniform Strategy"
    baseline_scores = df[df['Model'] == baseline_name][['Fold', 'Score']].rename(
        columns={'Score': 'BaselineScore'}
    )
    # Merge on the fold number
    df_paired = df.merge(baseline_scores, on='Fold', how='inner')

    # run paired t-tests
    tests = []          # will hold dicts with model, p‑value, etc.
    for model, grp in df_paired.groupby('Model'):
        if model == baseline_name:
            continue  # skip the baseline itself

        # The two paired samples
        model_scores = grp['Score'].values
        base_scores  = grp['BaselineScore'].values

        # Paired t‑test (parametric)
        _, p_val = ttest_rel(model_scores, base_scores)

        tests.append({
            'Model': model,
            'n_folds': len(grp),       # number of matched folds
            'p_value': p_val
        })

    # Create a DataFrame and adjust the p‑values
    tests_df = pd.DataFrame(tests)

    # Bonferroni correction (you can use 'fdr_bh', 'holm', etc.)
    reject, p_adj, _, _ = multipletests(tests_df['p_value'],
                                        alpha=0.05,
                                        method='bonferroni')

    tests_df['p_value_adj'] = p_adj
    tests_df['significant'] = reject

    #  Show / export the result
    print(tests_df[['Model', 'n_folds', 'p_value', 'p_value_adj', 'significant']])

    return

    # perform the one-way anova on models
    # ANOVA: 5models [4models and the baseline] x 3dataset [dyads, triads, all] x  2 features [all, or AIxVR features]
    
    # # three way ANOVA with interactions across all conditions
    # formula = """
    # Score ~ C(Model) + C(Group) + C(Features) +
    #         C(Model):C(Group) +
    #         C(Model):C(Features) +
    #         C(Group):C(Features) +
    #         C(Model):C(Group):C(Features)
    # """
    formula_interactions_single = """
    Score ~ C(Model) + C(Group) + C(Features) + 
    C(Model):C(Group) + C(Model):C(Features) + C(Group):C(Features)
    """
    # simpler three way ANOVA without interactions
    formula_single = "Score ~ C(Model) + C(Group) + C(Features)"
    
    # formula taking into account the version as one more factor
    formula_version = "Score ~ C(Model) + C(Group) + C(Features) + C(Version)"
    formula_interactions_version = """
    Score ~ C(Model) + C(Group) + C(Features) + C(Version) + 
    C(Model):C(Group) + C(Model):C(Features) + C(Model):C(Version) 
    + C(Group):C(Features) + C(Group):C(Version) + 
    C(Features):C(Version)
    """
    df_anova = df
    formula = formula_interactions_single
    model = ols(formula, data=df_anova).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)  # or typ=3
    print(anova_table)
    anova_table.to_csv(f"anova_results{datetime.datetime.today().date()}.csv", index=True)
    
    # posthoc test (Groups was significant)
    comp = mc.MultiComparison(df_anova["Score"], df_anova["Group"])
    posthoc_res = comp.tukeyhsd(alpha=0.05)
    print(posthoc_res.summary())
    # Convert results to a DataFrame (data starts from the second row of the table)
    # tukey_df = pd.DataFrame(data=tukey_results._results_table.data[1:],
    #                     columns=tukey_results._results_table.data[0])
    # tukey_df.to_csv("tukey_posthoc.csv", index=False)
    pass

if __name__ == "__main__":
    # config flags
    binary:bool = False
    # data_path
    data_folder:str = "runs/accuracy"
    data_subfolder:str = "2026_05_19_binary_60s/Blinks_GazexSpeaking"
    # files three way
    # simple, one file processed with relevant models from excel
    # file_three_way:str ="Processed_Results_Accuracy_three_way_60s.csv"
    # file_binary_30s:str = "Processed_Results_Accuracy_binary_30s_avg.csv"
    file_dyads:str = "results_ML_train_SIMPLE_dyads_f_Blinks_GazexSpeaking_60s_binary_all_groups_avg_separated_with_group_name__2026-05-19.csv"
    file_triads:str = "results_ML_train_SIMPLE_triads_f_Blinks_GazexSpeaking_60s_binary_all_groups_avg_separated_with_group_name__2026-05-19.csv"
    file_all:str = "results_ML_train_SIMPLE_all_groups_f_Blinks_GazexSpeaking_60s_binary_all_groups_avg_separated_with_group_name__2026-05-19.csv"
    file_path_dyads:str = os.path.join(os.getcwd(), data_folder, data_subfolder, file_dyads)
    file_path_triads:str = os.path.join(os.getcwd(), data_folder, data_subfolder, file_triads)
    file_path_all:str = os.path.join(os.getcwd(), data_folder, data_subfolder, file_all)
    df_data_dyads: pd.DataFrame = pd.read_csv(file_path_dyads)
    df_data_triads: pd.DataFrame = pd.read_csv(file_path_triads)
    df_data_all: pd.DataFrame = pd.read_csv(file_path_all)    
    # get models to compare
    models_selected: list[str] = [
        "AdaBoost",
        "Baseline Uniform Strategy",
        "Decision Tree",
        "Linear SVM l1",
        "Linear SVM l2",
        "Logistic Regression",
        "Naive Bayes",
        "Nearest Neighbors",
        "Neural Net",
        "QDA",
        "Random Forest",
        "SVM linear or rbf",
    ]
    print("==========================")
    print("Dyads T-Tests")
    print("==========================")
    run_stat_test(df_data_dyads, models_selected)
    print("==========================")
    print("Triads T-Tests")
    print("==========================")
    run_stat_test(df_data_triads, models_selected)
    print("==========================")
    print("All groups T-Tests")
    print("==========================")
    run_stat_test(df_data_all, models_selected)
    