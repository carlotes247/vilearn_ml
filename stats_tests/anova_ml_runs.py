import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import os

if __name__ == "__main__":
    # config flags
    binary:bool = False
    # data_path
    data_folder:str = "runs/accuracy"
    # files three way
    # simple, one file processed with relevant models from excel
    # TODO: do a run for all groups all features simple-scalar because we are missing baselines
    file_three_way:str ="Processed_Results_Accuracy_three_way.csv"
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
        "Nearest Neighbors",
        "Neural Net",
        "Linear SVM l1",
        "Decision Tree",
        "Naive Bayes",
        "QDA"]
    # beat the baseline in simple version 3-way:
        # only triads aixvr (qda, neural net)
        # unkown all groups, all aivr
    # beat the baseline in scaler version 3-way:
        # triads all (linear svm l1)
        # triads aivr (knn)
        # dyads all (linear svm l1)
        # unknown all groups, all aivr

    # discriminate for wanted models
    df_subset = df_three_way.loc[df_three_way['Model'].isin(models_selected)]
    # select simple
    df_simple = df_subset.loc[df_subset['Version'] == 'Simple_Manual']
    df_simple.drop(columns=['Version'], inplace=True)
    # select scaler
    df_scaler = df_subset.loc[df_subset['Version'] == 'Scaler_Manual']
    df_scaler.drop(columns=['Version'], inplace=True)
    print("DFs created")

    # perform the one-way anova on models
    # ANOVA: 5models [4models and the baseline] x 3dataset [dyads, triads, all] x  2 features [all, or AIxVR features]
    # three way ANOVA with interactions across all conditions
    # formula = """
    # Score ~ C(Model) + C(Groups) + C(Features) +
    #         C(Model):C(Groups) +
    #         C(Model):C(Features) +
    #         C(Groups):C(Features) +
    #         C(Model):C(Groups):C(Features)
    # """
    # simpler three way ANOVA without interactions
    formula = "Score ~ C(Model) + C(Groups) + C(Features)"
    
    model = ols(formula, data=df_simple).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)  # or typ=3
    print(anova_table)
    # posthoc test
    