import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.stats.multicomp as mc
import os
import datetime

if __name__ == "__main__":
    # config flags
    binary:bool = False
    # data_path
    data_folder:str = "runs/accuracy"
    # files three way
    # simple, one file processed with relevant models from excel
    # TODO: do a run for all groups all features simple-scalar because we are missing baselines
    file_three_way:str ="Processed_Results_Accuracy_three_way_60s.csv"
    file_binary_30s:str = "Processed_Results_Accuracy_binary_30s_avg.csv"
    file:str = file_binary_30s
    file_path:str = os.path.join(os.getcwd(), data_folder, file)
    df_data: pd.DataFrame = pd.read_csv(file_path)
    if "Time" in df_data.columns:
        df_data.drop(columns=['Time'], inplace=True)
    if "CV" in df_data.columns:
        df_data.drop(columns="CV", inplace=True)
    if "Fold" in df_data.columns:
        df_data.drop(columns="Fold", inplace=True)
    # get models to compare
    # SVM L1, L2, linear? (maybe keep one rather than the other? here we weren’t sure, we need to think about it- also keep simple or scaler?); 
    # nearest neighbours , 
    # naive bayes, 
    # QDA, 
    # decision trees (performs well in triads)
    # remove neural net from the analysis as it can get too complex? or maybe keep it if mentioned that it came from the scikitlearn package? Or remove it? it performed ok, almost 70
    # also the baseline
    models_selected_60s_three_way: list[str] = [
        "Baseline Most Frequent Strategy",
        "Nearest Neighbors", # beats baseline in triads aixvr scalar
        "Neural Net", # beats baseline in triads aixvr simple
        "Linear SVM l1", # beats baseline in triads & dyads all features
        # "Decision Tree", 
        # "Naive Bayes",
        "QDA" # beats baseline in triads aixvr simple
        ]
    models_selected_30s_binary: list[str] = [
        "Baseline Most Frequent Strategy",
        "Adaboost",
        "Linear SVM l1",
        "Nearest Neighbors",
        "Naive Bayes",
        "Decision Tree"
    ]
    models_selected = models_selected_30s_binary
    # beat the baseline in simple 60s version 3-way:
        # only triads aixvr f (qda, neural net)
    # beat the baseline in scaler 60s version 3-way:
        # triads all f (linear svm l1)
        # triads aixvr f (knn)
        # dyads all f (linear svm l1)
    # beat the baseline in simple 30s version 3-way:
        # all groups all f (QDA, NN > 0.01, linear svm l1 > 0.005)
        # all groups aixvr f (NN, SVM linear or rbf, knn, QDA, Naive Bayes)
        # dyads all f (linear svm l1)
        # triads all f (linear svm l2, NN, linear svm l1)
        # triads aixvr f (adaboost, linear svm l1, naive bayes, QDA, knn)
    # beat the baseline in scaler 30s version 3-way:
        # dyads all f (knn)
        # triads all f (knn)
        # triads aixvr f (QDA, Naive Bayes, NN, linear svm l1, knn)


    # discriminate for wanted models
    df_subset = df_data.loc[df_data['Model'].isin(models_selected)]
    if "Sampling" in df_subset.columns.to_list():
        df_subset = df_subset.drop(columns=['Sampling'])
    # select simple
    df_simple = df_subset.loc[df_subset['Version'] == 'Simple']
    df_simple = df_simple.drop(columns=['Version'])
    # select scaler
    df_scaler = df_subset.loc[df_subset['Version'] == 'Scaler']
    df_scaler = df_scaler.drop(columns=['Version'])
    print("DFs created")

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
    df_anova = df_simple
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
    