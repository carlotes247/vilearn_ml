import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.stats.multicomp as mc
import os

if __name__ == "__main__":
    # config flags
    binary:bool = False
    # data_path
    data_folder:str = "runs/accuracy"
    # files three way
    # simple, one file processed with relevant models from excel
    # TODO: do a run for all groups all features simple-scalar because we are missing baselines
    file_three_way:str ="Processed_Results_Accuracy_three_way_60s.csv"
    file_path:str = os.path.join(os.getcwd(), data_folder, file_three_way)
    df_three_way: pd.DataFrame = pd.read_csv(file_path)
    df_three_way.drop(columns=['Time'], inplace=True)
    # get models to compare
    # SVM L1, L2, linear? (maybe keep one rather than the other? here we weren’t sure, we need to think about it- also keep simple or scaler?); 
    # nearest neighbours , 
    # naive bayes, 
    # QDA, 
    # decision trees (performs well in triads)
    # remove neural net from the analysis as it can get too complex? or maybe keep it if mentioned that it came from the scikitlearn package? Or remove it? it performed ok, almost 70
    # also the baseline
    models_selected: list[str] = [
        "Baseline Most Frequent Strategy",
        "Nearest Neighbors", # beats baseline in triads aixvr scalar
        "Neural Net", # beats baseline in triads aixvr simple
        "Linear SVM l1", # beats baseline in triads & dyads all features
        # "Decision Tree", 
        # "Naive Bayes",
        "QDA" # beats baseline in triads aixvr simple
        ]
    # beat the baseline in simple version 3-way:
        # only triads aixvr (qda, neural net)
    # beat the baseline in scaler version 3-way:
        # triads all (linear svm l1)
        # triads aixvr (knn)
        # dyads all (linear svm l1)

    # discriminate for wanted models
    df_subset = df_three_way.loc[df_three_way['Model'].isin(models_selected)]
    # select simple
    df_simple = df_subset.loc[df_subset['Version'] == 'Simple_Manual']
    df_simple = df_simple.drop(columns=['Version'])
    # select scaler
    df_scaler = df_subset.loc[df_subset['Version'] == 'Scaler_Manual']
    df_scaler = df_scaler.drop(columns=['Version'])
    print("DFs created")

    # perform the one-way anova on models
    # ANOVA: 5models [4models and the baseline] x 3dataset [dyads, triads, all] x  2 features [all, or AIxVR features]
    
    # # three way ANOVA with interactions across all conditions
    # formula = """
    # Score ~ C(Model) + C(Groups) + C(Features) +
    #         C(Model):C(Groups) +
    #         C(Model):C(Features) +
    #         C(Groups):C(Features) +
    #         C(Model):C(Groups):C(Features)
    # """
    formula_interactions_single = """
    Score ~ C(Model) + C(Groups) + C(Features) + 
    C(Model):C(Groups) + C(Model):C(Features) + C(Groups):C(Features)
    """
    # simpler three way ANOVA without interactions
    formula_single = "Score ~ C(Model) + C(Groups) + C(Features)"
    
    # formula taking into account the version as one more factor
    formula_version = "Score ~ C(Model) + C(Groups) + C(Features) + C(Version)"
    formula_interactions_version = """
    Score ~ C(Model) + C(Groups) + C(Features) + C(Version) + 
    C(Model):C(Groups) + C(Model):C(Features) + C(Model):C(Version) 
    + C(Groups):C(Features) + C(Groups):C(Version) + 
    C(Features):C(Version)
    """
    df_anova = df_subset
    formula = formula_version
    model = ols(formula, data=df_anova).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)  # or typ=3
    print(anova_table)
    anova_table.to_csv("anova_results.csv", index=True)
    
    # posthoc test (Groups was significant)
    comp = mc.MultiComparison(df_anova["Score"], df_anova["Groups"])
    posthoc_res = comp.tukeyhsd(alpha=0.05)
    print(posthoc_res.summary())
    tukey_df = posthoc_res.summary_frame()
    tukey_df.to_csv("tukey_posthoc.csv", index=False)
    