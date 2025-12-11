import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# Class with classifiers to try
from vilearn_ML_models import VilearnMLModels
# classifiers imports
from sklearn import neighbors
from sklearn import naive_bayes
from sklearn import neural_network
from sklearn import svm
from sklearn import tree
from sklearn import model_selection
from sklearn import gaussian_process
from sklearn import ensemble
from sklearn import discriminant_analysis 
# pipeline and scaler imports for classifiers
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn import metrics
import os
import copy
# Added this try catch because on some machines it cannot find folders from working directory 
try:
    from data_reading.vilearn_windowed_data_loader_ML import VilearnWindowedDataLoaderML
except ImportError:
    import sys
    sys.path.append(os.getcwd())
    from data_reading.vilearn_windowed_data_loader_ML import VilearnWindowedDataLoaderML
# for plotting tables
import matplotlib.pyplot as plt
import pandas as pd
# for calculating how long it takes per fold
import time

class VilearnMLTrain:
    # class vars
    # config flags
    nested_cv: bool = True
    nested_cv_manual: bool = True
    binary_clf: bool = False
    # df to plot results
    results_df: pd.DataFrame = pd.DataFrame()
    results_list: list[dict] = []

    # load data
    data_loader: VilearnWindowedDataLoaderML = VilearnWindowedDataLoaderML(bins_binary=binary_clf, print_folds=False, debug_all_folds=False)
    # models
    ml_models: VilearnMLModels = VilearnMLModels()

    # TODO: move logic into class instance
    def __init__(self) -> None:
        pass

    def nested_cv(self) -> None:
        pass

    def main(self) -> None:
        pass



if __name__ == '__main__':
    # config flags
    nested_cv: bool = True
    nested_cv_manual: bool = True
    binary_clf: bool = False
    # df to plot results
    results_df: pd.DataFrame = pd.DataFrame()
    results_list: list[dict] = []

    # load data
    data_loader: VilearnWindowedDataLoaderML = VilearnWindowedDataLoaderML(bins_binary=binary_clf, print_folds=False, debug_all_folds=False)
    # load models
    models: VilearnMLModels = VilearnMLModels()

    # Cross validation for all models
    for model_name, model_dict in models.param_grids_models.items():        
        model = model_dict['estimator']
        param_grid = model_dict['params']
        print(f"Cross val score {model_name}")
        # print(f"Estimator: {model}")
        # print(f"Params: {param_grid}")

        # create alternative model with scaler to see what scores better
        # the scaler can be useful to standardize features. It depends on how the features approximate the std normal distribution of the data (e.g. Gaussian with 0 mean and unit variance).
        # more info on scalers: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
        model_scaler = Pipeline(
            steps=[("scaler", StandardScaler()), ("clf", model)]
        )
        param_grid_scaler = {f'clf__{k}': v for k, v in param_grid.items()}


        # We ensure to do a nested CV
        # inner cv, outer cv NEEDED for nested CV
        inner_cv_simple: model_selection.GroupKFold = copy.deepcopy(data_loader.group_kfold)
        inner_cv_simple.n_splits = inner_cv_simple.get_n_splits() - 1
        outer_cv_simple = copy.deepcopy(data_loader.group_kfold)
        # this grid search is declared here for the 'automatic' nested cv
        grid_search_cv_simple = model_selection.GridSearchCV(estimator=model,
                                                                            param_grid=param_grid,
                                                                            cv=inner_cv_simple, 
                                                                            verbose=1)            
        if nested_cv:
            # Nested CV Manual
            if nested_cv_manual:
                y_true_all = []
                y_pred_all = []
                outer_scores = []
                i = 1
                start_outer_cv = time.process_time()
                # Outer CV loop
                for train_idx, test_idx in outer_cv_simple.split(data_loader.X, data_loader.y, groups=data_loader.groups):
                    X_train, X_test = data_loader.X.loc[train_idx], data_loader.X.loc[test_idx]
                    y_train, y_test = data_loader.y.loc[train_idx], data_loader.y.loc[test_idx]
                    group_out_name: str = data_loader.df_data.loc[test_idx]['group_name'].iloc[0]
                    print(f"Outer Simple CV Fold {i}. Leave out fold is: {group_out_name}") 
                    start_inner_cv = time.process_time()                   
                    # Inner CV grid search
                    grid_search_cv_inner = model_selection.GridSearchCV(estimator=model,
                                                                    param_grid=param_grid,
                                                                    cv=inner_cv_simple, 
                                                                    verbose=1)
                    # Given the n-1 training data, run cv search function on that and not whole data (as one would usually do in a regular cv search. but this is nested)
                    grid_search_cv_inner.fit(X_train, y_train, groups=data_loader.groups[train_idx])
                    # Select best model and evaluate on unseen data, our testing fold not included in the CV search
                    best_model = grid_search_cv_inner.best_estimator_
                    y_pred = best_model.predict(X_test)
                    # Collect predictions for confusion matrix
                    y_true_all.extend(y_test)
                    y_pred_all.extend(y_pred)
                    # Collect score for this outer fold
                    fold_acc = metrics.accuracy_score(y_test, y_pred)
                    outer_scores.append(fold_acc)                    
                    end_inner_cv = time.process_time()
                    print(f"Outer Simple CV Fold {i} took {end_inner_cv-start_inner_cv} secs.")
                    i = i+1
                end_outer_cv = time.process_time()
                print(f"Outer Simple CV completed! Took {end_outer_cv-start_outer_cv} seconds")
                # Nested CV score (mean of outer fold scores)
                nested_cv_score = np.mean(outer_scores)
                print(f"Manual Nested CV Accuracy Simple: {nested_cv_score:.4f}")
                # Print confusion matrix and score once all loops are done                
                conf_matrix = metrics.confusion_matrix(y_true_all, y_pred_all)
                results_list.append({'Model': model_name, 
                                                'Score':nested_cv_score, 
                                                'CV': 'Nested', 
                                                'Version': 'Simple_Manual',
                                                'Conf_Matrix': conf_matrix,
                                                'Time': end_outer_cv-start_outer_cv})
                # print("Confusion Matrix:\n", conf_matrix)
            # Nested CV Automatic
            else:                
                nested_score_simple = model_selection.cross_val_score(estimator=model,        
                                                        X=data_loader.X, y=data_loader.y,
                                                        cv=outer_cv_simple, 
                                                        groups=data_loader.groups, verbose=1)
                print(f"Avg nested acc SIMPLE: {nested_score_simple.mean()}")
                results_list.append({'Model': model_name, 
                                                'Score':nested_score_simple.mean(), 
                                                'CV': 'Nested', 
                                                'Version': 'Simple'})                    
        else:
            # DEBUGGING NON_NESTED PARAMETER SEARCH AND SCORING (THIS IS NOT WHAT WE SHOULD DO ACCORDING TO CRISTINA CONATI)
            fit_worked: bool = False
            try:
                grid_search_cv_simple.fit(X=data_loader.X, y=data_loader.y, groups=data_loader.groups)
                fit_worked = True
            except Exception as err:
                print(f"Unexpected {err=}, {type(err)=}")
            if fit_worked:
                print(f"Avg non_nested acc SIMPLE: {grid_search_cv_simple.best_score_}")
                results_list.append({'Model': model_name, 
                                               'Score':grid_search_cv_simple.best_score_, 
                                               'CV': 'Non_Nested', 
                                               'Version': 'Simple'})

            y_pred = model_selection.cross_val_predict(estimator=best_knn, 
                                               X=data_loader.X, y=data_loader.y, 
                                               cv=data_loader.group_kfold, 
                                               groups=data_loader.groups)                        

        # SCALER version
        inner_cv_scaler: model_selection.GroupKFold = copy.deepcopy(data_loader.group_kfold)
        inner_cv_scaler.n_splits = inner_cv_scaler.get_n_splits() - 1
        outer_cv_scaler: model_selection.GroupKFold = copy.deepcopy(data_loader.group_kfold)
        # this grid search is declared here for the 'automatic' nested cv
        grid_search_cv_scaler = model_selection.GridSearchCV(
            estimator=model_scaler,
            param_grid=param_grid_scaler,
            cv=inner_cv_scaler, 
            verbose=1)
        if nested_cv:
            # Nested CV Manual SCALER
            if nested_cv_manual:
                y_true_all = []
                y_pred_all = []
                outer_scores = []
                i = 1
                start_outer_cv = time.process_time()
                # Outer CV loop
                for train_idx, test_idx in outer_cv_scaler.split(data_loader.X, data_loader.y, groups=data_loader.groups):
                    X_train, X_test = data_loader.X.loc[train_idx], data_loader.X.loc[test_idx]
                    y_train, y_test = data_loader.y.loc[train_idx], data_loader.y.loc[test_idx]                    
                    group_out_name: str = data_loader.df_data.loc[test_idx]['group_name'].iloc[0]
                    print(f"Outer Scaler CV Fold {i}. Leave out fold is: {group_out_name}")                    
                    start_inner_cv = time.process_time()                   
                    # Inner CV grid search
                    grid_search_cv_inner = model_selection.GridSearchCV(estimator=model,
                                                                    param_grid=param_grid,
                                                                    cv=inner_cv_scaler, 
                                                                    verbose=1)
                    grid_search_cv_inner.fit(X_train, y_train, groups=data_loader.groups[train_idx])
                    best_model = grid_search_cv_inner.best_estimator_
                    y_pred = best_model.predict(X_test)
                    # Collect predictions for confusion matrix
                    y_true_all.extend(y_test)
                    y_pred_all.extend(y_pred)
                    # Collect score for this outer fold
                    fold_acc = metrics.accuracy_score(y_test, y_pred)
                    outer_scores.append(fold_acc)                    
                    end_inner_cv = time.process_time()
                    print(f"Outer Scaler CV Fold {i} took {end_inner_cv-start_inner_cv} secs.")
                    i = i+1
                end_outer_cv = time.process_time()
                print(f"Outer Scaler CV completed! Took {end_outer_cv-start_outer_cv} seconds")
                # Nested CV score (mean of outer fold scores)
                nested_cv_score = np.mean(outer_scores)
                print(f"Manual Nested CV Accuracy Scaler: {nested_cv_score:.4f}")
                conf_matrix = metrics.confusion_matrix(y_true_all, y_pred_all)
                results_list.append({'Model': model_name, 
                                                'Score':nested_cv_score, 
                                                'CV': 'Nested', 
                                                'Version': 'Scaler_Manual',
                                                'Conf_Matrix': conf_matrix,
                                                'Time': end_outer_cv-start_outer_cv})
                # Print confusion matrix and score once all loops are done                                
                # print("Confusion Matrix:\n", conf_matrix)
            # Nested CV Automatic
            else:
                nested_score_scaler = model_selection.cross_val_score(estimator=model_scaler,        
                                                        X=data_loader.X, y=data_loader.y,
                                                        cv=outer_cv_scaler, 
                                                        groups=data_loader.groups, verbose=1)
                print(f"Avg nested acc SCALER: {nested_score_scaler.mean()}")
                results_list.append({'Model': model_name, 
                                                'Score':nested_score_scaler.mean(), 
                                                'CV': 'Nested', 
                                                'Version': 'Scaler'})
        else:
            # DEBUGGING NON_NESTED PARAMETER SEARCH AND SCORING (THIS IS NOT WHAT WE SHOULD DO ACCORDING TO CRISTINA CONATI)
            fit_worked: bool = False
            try:
                grid_search_cv_scaler.fit(X=data_loader.X, y=data_loader.y, groups=data_loader.groups)
                fit_worked = True
            except Exception as err:
                print(f"Unexpected {err=}, {type(err)=}")
            if fit_worked:
                print(f"Avg non_nested acc SCALER: {grid_search_cv_scaler.best_score_}")
                results_list.append({'Model': model_name, 
                                               'Score':grid_search_cv_scaler.best_score_, 
                                               'CV': 'Non_Nested', 
                                               'Version': 'Scaler'})
    
    # output results as html
    results_df = pd.DataFrame(results_list)
    print(results_df.to_string())
    models_used = [model_name for model_name in results_df['Model']]
    models_used = set(models_used)
    models_suffix = ""
    for model_name in models_used:
        models_suffix = f"{models_suffix}_{model_name}"
    #results_df.to_html(f'results_ML_train_manual{models_suffix}.html')
    results_df.to_csv(f'results_ML_train_manual{models_suffix}.csv')

    print("done")