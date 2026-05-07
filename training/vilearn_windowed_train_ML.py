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
from datetime import datetime
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
    results_df: pd.DataFrame
    results_list: list[dict] 

    # load data
    data_loader: VilearnWindowedDataLoaderML 
    # models
    ml_models: VilearnMLModels 

    def __init__(self, nested_cv:bool, auto_cv:bool, binary_clf:bool, data_loader: VilearnWindowedDataLoaderML, ml_models: VilearnMLModels) -> None:
        self.nested_cv = nested_cv
        self.nested_cv_manual = not auto_cv
        self.binary_clf = binary_clf
        self.data_loader = data_loader
        self.ml_models = ml_models
        self.results_df = pd.DataFrame()
        self.results_list = []

    def run_nested_cv(self, models: VilearnMLModels, data_loader: VilearnWindowedDataLoaderML, nested_cv_manual: bool, eval_label: str = "", group_label:str = "", features_label:str = "", sampling_label:str = "", non_nested_comparison: bool = False, separate_avg_groups:bool = False, debug: bool = False) -> list[dict]:
        # Cross validation for all models
        results_to_return: list[dict] = []
        for model_name, model_dict in models.param_grids_models.items():        
            model = model_dict['estimator']
            param_grid = model_dict['params']
            if (debug):
                print(f"Cross val score {model_name}")
                # print(f"Estimator: {model}")
                # print(f"Params: {param_grid}")

            # We ensure to do a nested CV
            # inner cv, outer cv NEEDED for nested CV
            inner_cv: model_selection.GroupKFold = copy.deepcopy(data_loader.group_kfold)
            inner_cv.n_splits = inner_cv.get_n_splits() - 1
            outer_cv = copy.deepcopy(data_loader.group_kfold)
        
            # Nested CV Manual
            if nested_cv_manual:
                results =  self.__run_nested_cv_manual(model=model, model_name=model_name,
                                                        data_loader=data_loader, param_grid=param_grid, 
                                                        inner_cv=inner_cv, outer_cv=outer_cv, 
                                                        model_version_label=eval_label, 
                                                        group_label=group_label, features_label=features_label, 
                                                        sampling_label=sampling_label, 
                                                        separate_avg_groups=separate_avg_groups, debug=debug)
                results_to_return.extend(results)
            # Nested CV Automatic
            else:                
                results = self.__run_nested_cv_automatic(model=model, model_name=model_name,
                                                            data_loader=data_loader, param_grid=param_grid,
                                                            inner_cv=inner_cv, outer_cv=outer_cv, 
                                                            model_version_label=eval_label)
                results_to_return.extend(results)
            # Non_nested CV for comparison
            if non_nested_comparison:
                results = self.__run_non_nested_cv(model=model, model_name=model_name,
                                                            data_loader=data_loader, param_grid=param_grid,
                                                            inner_cv=inner_cv,  
                                                            model_version_label=eval_label)
                results_to_return.extend(results)
        return results_to_return
                             
                                 
    def __run_nested_cv_manual(self, model, model_name:str, data_loader:VilearnWindowedDataLoaderML, param_grid, inner_cv, outer_cv, model_version_label: str, group_label:str, features_label:str, sampling_label:str, separate_avg_groups:bool = False, debug: bool = False) -> list[dict]:        
        results_to_return: list[dict] = [] 
        y_true_all = []
        y_pred_all = []
        outer_scores = []
        outer_scores_dyads = []
        outer_scores_triads = []
        i = 1
        start_outer_cv = time.process_time()
        verbose = 1 if debug else 0
        # Outer CV loop
        for train_idx, test_idx in outer_cv.split(data_loader.X, data_loader.y, groups=data_loader.groups):
            X_train, X_test = data_loader.X.loc[train_idx], data_loader.X.loc[test_idx]
            y_train, y_test = data_loader.y.loc[train_idx], data_loader.y.loc[test_idx]
            group_out_name: str = data_loader.df_data.loc[test_idx]['group_name'].iloc[0]
            if(debug):
                print(f"Outer {model_version_label} CV Fold {i}. Leave out fold is: {group_out_name}") 
            start_inner_cv = time.process_time()                   
            # Inner CV grid search
            grid_search_cv_inner = model_selection.GridSearchCV(estimator=model,
                                                            param_grid=param_grid,
                                                            cv=inner_cv, 
                                                            verbose=verbose, n_jobs=-1)
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
            # conf matrix outer fold
            conf_matrix = metrics.confusion_matrix(y_test, y_pred)
            # save all the information of each outer fold model
            results_to_return.append({'Model': model_name,
                                        'Fold': f"{i}", 
                                        'Score':fold_acc, 
                                        'CV': 'Nested', 
                                        'Version': f'{model_version_label}_Manual',
                                        'Group': group_label,
                                        'Features': features_label,
                                        'Group_Name': group_out_name,
                                        'Sampling': sampling_label,
                                        'Conf_Matrix': conf_matrix,
                                        'Time': end_inner_cv-start_inner_cv})
            # only if we want a difference between test score on dyad or triad test set
            if separate_avg_groups:
                if 'dyad' in group_out_name:
                    outer_scores_dyads.append(fold_acc)
                    results_to_return[-1]['Score_on_dyads'] = outer_scores_dyads[-1]
                    results_to_return[-1]['Score_on_triads'] = 0     
                elif 'triad' in group_out_name:
                    outer_scores_triads.append(fold_acc)  
                    results_to_return[-1]['Score_on_dyads'] = 0
                    results_to_return[-1]['Score_on_triads'] = outer_scores_triads[-1]                                         
            if (debug):
                print(f"Outer {model_version_label} CV Fold {i} took {end_inner_cv-start_inner_cv} secs.")
            i = i+1
        end_outer_cv = time.process_time()
        if (debug):
            print(f"Outer {model_version_label} CV completed! Took {end_outer_cv-start_outer_cv} seconds")
        # Nested CV score (mean of outer fold scores)
        nested_cv_score = np.mean(outer_scores) 
        nested_cv_std= np.std(outer_scores)
        if (debug):
            print(f"Manual Nested CV Accuracy {model_version_label}: {nested_cv_score:.4f}")
        # Print confusion matrix and score once all loops are done                
        conf_matrix = metrics.confusion_matrix(y_true_all, y_pred_all)
        # Average all outer folds
        results_to_return.append({'Model': model_name, 
                                    'Fold': 'avg',
                                    'Score':nested_cv_score, 
                                    'CV': 'Nested', 
                                    'Version': f'{model_version_label}_Manual',
                                    'Group': group_label,
                                    'Features': features_label,
                                    'Group_Name': 'avg',
                                    'Sampling': sampling_label,
                                    'Conf_Matrix': conf_matrix,
                                    'Time': end_outer_cv-start_outer_cv})
        # only if we want a difference between test score on dyad or triad test set
        if separate_avg_groups:
            results_to_return[-1]['Score_on_dyads'] = np.mean(outer_scores_dyads)
            results_to_return[-1]['Score_on_triads'] = np.mean(outer_scores_triads)     
        # Std all outer folds
        results_to_return.append({'Model': model_name, 
                                    'Fold': 'std',
                                    'Score': nested_cv_std, 
                                    'CV': 'Nested', 
                                    'Version': f'{model_version_label}_Manual',
                                    'Group': group_label,
                                    'Features': features_label,
                                    'Group_Name': 'std',
                                    'Sampling': sampling_label,
                                    'Conf_Matrix': conf_matrix,
                                    'Time': end_outer_cv-start_outer_cv})
        # only if we want a difference between test score on dyad or triad test set
        if separate_avg_groups:
            results_to_return[-1]['Score_on_dyads'] = np.std(outer_scores_dyads)
            results_to_return[-1]['Score_on_triads'] = np.std(outer_scores_triads)     
        # print("Confusion Matrix:\n", conf_matrix)
        return results_to_return

    # TODO: correct this automatic nested cv based on the scikitlearn example, it does not seem correct
    def __run_nested_cv_automatic(self, model, model_name:str, data_loader:VilearnWindowedDataLoaderML, param_grid, inner_cv, outer_cv, model_version_label: str, debug:bool = False) -> list[dict]:
        results_to_return: list[dict] = []
        verbose = 1 if debug else 0
        # this grid search is declared here for the 'automatic'nested cv
        grid_search_cv = model_selection.GridSearchCV(estimator=model,
                                                            param_grid=param_grid,
                                                            cv=inner_cv, 
                                                            verbose=verbose, n_jobs=-1)   

        nested_score_simple = model_selection.cross_val_score(estimator=model,        
                                                        X=data_loader.X, y=data_loader.y,
                                                        cv=outer_cv, 
                                                        groups=data_loader.groups, verbose=verbose)
        if(debug):
            print(f"Avg automatic nested cv acc {model_version_label}: {nested_score_simple.mean()}")
        results_to_return.append({'Model': model_name, 
                                            'Score':nested_score_simple.mean(), 
                                            'CV': 'Nested', 
                                            'Version': f'{model_version_label}_Auto'})  
        return results_to_return

    def __run_non_nested_cv(self, model, model_name: str, inner_cv, data_loader: VilearnWindowedDataLoaderML,param_grid, model_version_label: str, debug:bool = False) -> list[dict]:
        # DEBUGGING NON_NESTED PARAMETER SEARCH AND SCORING (THIS IS NOT WHAT WE SHOULD DO ACCORDING TO CRISTINA CONATI)
        fit_worked: bool = False
        results_to_return:list[dict] = []
        verbose = 1 if debug else 0
        # this grid search is declared here for the 'automatic' cv
        grid_search_cv = model_selection.GridSearchCV(estimator=model,
                                                                param_grid=param_grid,
                                                                cv=inner_cv, 
                                                                verbose=verbose, n_jobs=-1)   

        try:
            grid_search_cv.fit(X=data_loader.X, y=data_loader.y, groups=data_loader.groups)
            fit_worked = True
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
        if fit_worked:
            print(f"Avg non_nested acc {model_version_label}: {grid_search_cv.best_score_}")
            results_to_return.append({'Model': model_name, 
                                        'Score':grid_search_cv.best_score_, 
                                        'CV': 'Non_Nested', 
                                        'Version': model_version_label})  
        return results_to_return  

    def save_results(self, results_list, model_version_label: str, debug: bool = False) -> pd.DataFrame:
        results_df = pd.DataFrame(results_list)
        if (debug):
            print(results_df.to_string())
        models_used = [model_name for model_name in results_df['Model']]
        models_used = set(models_used)
        models_suffix = ""
        # for the moment model suffix is empty because it makes the string too long and an error is raised
        # for model_name in models_used: 
        #     models_suffix = f"{models_suffix}_{model_name}"
        #results_df.to_html(f'results_ML_train_manual{models_suffix}.html')
        results_df.to_csv(f'results_ML_train_{model_version_label}_{models_suffix}_{datetime.now().date()}.csv')
        return results_df 

    def train_and_evaluate(self, eval_label: str = "", group_label:str = "", features_label:str = "", sampling_label:str = "", separate_avg_groups:bool = False, debug=False) -> None:
        results = []
        if self.nested_cv:
            results = self.run_nested_cv(models=self.ml_models, data_loader=self.data_loader, 
                                        nested_cv_manual=True, eval_label=eval_label,
                                        group_label=group_label, features_label=features_label,
                                        sampling_label=sampling_label, 
                                        separate_avg_groups=separate_avg_groups, debug=debug)
            self.results_list.extend(results)
        self.results_df = self.save_results(self.results_list, model_version_label=eval_label)



if __name__ == '__main__':
    # config flags
    debug: bool = False
    nested_cv: bool = True
    nested_cv_manual: bool = True
    binary_clf: bool = True
    all_groups: bool = True
    dyads: bool = True
    triads: bool = True
    simple: bool = True
    scaler: bool = True    
    all_features : bool = False # to train models with all features
    aied_feautures:bool = True # to train models with aied features (blinks + gaze)
    aixvr_features: bool = True # to train models with aixvr features (gaze for dyads, blinks for triads)
    blink_speaking_x_gaze_features: bool = False # to train models with blinks and gaze x speaking features
    blinks_only: bool = False # to train models with blinks only
    separate_avg_groups: bool = True
    data_file: str = "" # leave empty for the original 60s file from the AixVR paper
    sampling: int = 60
    if sampling == 30:
        data_file = "30s_TE_correlation_2025-12-27_edited.csv"
    elif sampling == 60:
        data_file = "60s_TE_correlation_2026-04-09.csv"
    # suffix run
    suffix_run: str = f"{sampling}s_binary" if binary_clf else f"{sampling}s_three_way"
    # extra suffix opportunity
    suffix_run = f"{suffix_run}_all_groups_avg_separated_with_group_name"
    # load data
    # all groups, all features
    data_loader: VilearnWindowedDataLoaderML = VilearnWindowedDataLoaderML(bins_binary=binary_clf, 
                                                                           print_folds=False, debug_all_folds=False, 
                                                                           file_with_TE=data_file, sep=",")
    # dyads, all features
    data_loader_dyads: VilearnWindowedDataLoaderML = VilearnWindowedDataLoaderML(bins_binary=binary_clf, dyads_only=True, 
                                                                                 print_folds=False, debug_all_folds=False, 
                                                                                 file_with_TE=data_file, sep=",")
    # triads, all features
    data_loader_triads: VilearnWindowedDataLoaderML = VilearnWindowedDataLoaderML(bins_binary=binary_clf, triads_only=True, 
                                                                                  print_folds=False, debug_all_folds=False, 
                                                                                  file_with_TE=data_file, sep=",")
    # load models
    models: VilearnMLModels = VilearnMLModels()
    models_scaler: VilearnMLModels = VilearnMLModels(scaler=True)

    # all logic encapsulated in class
    ### ALL FEATURES ####
    # Simple models, all groups
    if all_groups and simple and all_features:
        vilearn_train_simple: VilearnMLTrain = VilearnMLTrain(nested_cv=nested_cv, 
                                                    auto_cv=(not nested_cv_manual),
                                                        binary_clf=binary_clf,
                                                        data_loader=data_loader, ml_models=models)
        vilearn_train_simple.train_and_evaluate(eval_label=f"SIMPLE_all_groups_all_features_{suffix_run}",
                                                group_label="All", features_label="All", sampling_label=f"{sampling}", 
                                                separate_avg_groups=separate_avg_groups, debug=debug)
    # simple model dyads, all features
    if dyads and simple and all_features:
        vilearn_train_simple_dyads_all_features: VilearnMLTrain = VilearnMLTrain(nested_cv=nested_cv, 
                                                                    auto_cv=(not nested_cv_manual),
                                                                        binary_clf=binary_clf,
                                                                        data_loader=data_loader_dyads, ml_models=models)
        vilearn_train_simple_dyads_all_features.train_and_evaluate(eval_label=f"SIMPLE_dyads_all_features_{suffix_run}",
                                                group_label="Dyads", features_label="All", sampling_label=f"{sampling}", debug=debug)
    # simple model triads, all features
    if triads and simple and all_features:
        vilearn_train_simple_triads_all_features: VilearnMLTrain = VilearnMLTrain(nested_cv=nested_cv, 
                                                                    auto_cv=(not nested_cv_manual),
                                                                        binary_clf=binary_clf,
                                                                        data_loader=data_loader_triads, ml_models=models)
        vilearn_train_simple_triads_all_features.train_and_evaluate(eval_label=f"SIMPLE_triads_all_features_{suffix_run}",
                                                group_label="Triads", features_label="All", sampling_label=f"{sampling}", debug=debug)
    # Scaler models, all groups
    if all_groups and scaler and all_features:
        vilearn_train_scaler: VilearnMLTrain = VilearnMLTrain(nested_cv=nested_cv, 
                                                    auto_cv=(not nested_cv_manual),
                                                        binary_clf=binary_clf,
                                                        data_loader=data_loader, ml_models=models_scaler)
        vilearn_train_scaler.train_and_evaluate(eval_label=f"SCALER_all_groups_all_features_{suffix_run}",
                                                group_label="All", features_label="All", sampling_label=f"{sampling}", 
                                                separate_avg_groups=separate_avg_groups, debug=debug)
    # Scaler models, dyads all features  
    if dyads and scaler and all_features:  
        vilearn_train_scaler_dyads_all_features: VilearnMLTrain = VilearnMLTrain(nested_cv=nested_cv, 
                                                    auto_cv=(not nested_cv_manual),
                                                        binary_clf=binary_clf,
                                                        data_loader=data_loader_triads, ml_models=models_scaler)
        vilearn_train_scaler_dyads_all_features.train_and_evaluate(eval_label=f"SCALER_dyads_all_features_{suffix_run}",
                                                group_label="Dyads", features_label="All", sampling_label=f"{sampling}", debug=debug)
    # Scaler models, triads all features
    if triads and scaler and all_features:
        vilearn_train_scaler_triads_all_features: VilearnMLTrain = VilearnMLTrain(nested_cv=nested_cv, 
                                                    auto_cv=(not nested_cv_manual),
                                                        binary_clf=binary_clf,
                                                        data_loader=data_loader_triads, ml_models=models_scaler)
        vilearn_train_scaler_triads_all_features.train_and_evaluate(eval_label=f"SCALER_triads_all_features_{suffix_run}",
                                                group_label="Triads", features_label="All", sampling_label=f"{sampling}", debug=debug)
    
    ### AIED FEATURES (BLINKS + GAZE) ###
    # Simple models, all groups, 
    if all_groups and simple and aied_feautures:
        data_loader.select_features(['MG','1d_DG','BPM','blink_durations'])
        vilearn_train_simple: VilearnMLTrain = VilearnMLTrain(nested_cv=nested_cv, 
                                                    auto_cv=(not nested_cv_manual),
                                                        binary_clf=binary_clf,
                                                        data_loader=data_loader, ml_models=models)
        vilearn_train_simple.train_and_evaluate(eval_label=f"SIMPLE_all_groups_f_MG_1DG_BPM_BDM_{suffix_run}",
                                                group_label="All", features_label="AIED", sampling_label=f"{sampling}", 
                                                separate_avg_groups=separate_avg_groups, debug=debug)
    # simple model dyads, 
    if dyads and simple and aied_feautures:
        data_loader_dyads.select_features(['MG','1d_DG','BPM','blink_durations'])
        vilearn_train_simple: VilearnMLTrain = VilearnMLTrain(nested_cv=nested_cv, 
                                                    auto_cv=(not nested_cv_manual),
                                                        binary_clf=binary_clf,
                                                        data_loader=data_loader_dyads, ml_models=models)
        vilearn_train_simple.train_and_evaluate(eval_label=f"SIMPLE_dyads_f_MG_1DG_BPM_BDM_{suffix_run}",
                                                group_label="Dyads", features_label="AIED", sampling_label=f"{sampling}", debug=debug)
    # simple model triads, 
    if triads and simple and aied_feautures:
        data_loader_triads.select_features(['MG','1d_DG','BPM','blink_durations'])
        vilearn_train_simple: VilearnMLTrain = VilearnMLTrain(nested_cv=nested_cv, 
                                                    auto_cv=(not nested_cv_manual),
                                                        binary_clf=binary_clf,
                                                        data_loader=data_loader_triads, ml_models=models)
        vilearn_train_simple.train_and_evaluate(eval_label=f"SIMPLE_triads_f_MG_1DG_BPM_BDM_{suffix_run}",
                                                group_label="Triads", features_label="AIED", sampling_label=f"{sampling}", debug=debug)
    # Scaler models, all groups 
    if all_groups and scaler and aied_feautures:
        data_loader.select_features(['MG','1d_DG','BPM','blink_durations'])
        vilearn_train_scaler: VilearnMLTrain = VilearnMLTrain(nested_cv=nested_cv, 
                                                    auto_cv=(not nested_cv_manual),
                                                        binary_clf=binary_clf,
                                                        data_loader=data_loader, ml_models=models_scaler)
        vilearn_train_scaler.train_and_evaluate(eval_label=f"SCALER_all_groups_f_MG_1DG_BPM_BDM_{suffix_run}",
                                                group_label="All", features_label="AIED", sampling_label=f"{sampling}", 
                                                separate_avg_groups=separate_avg_groups, debug=debug)
    # Scaler models, dyads  
    if dyads and scaler and aied_feautures:
        data_loader_dyads.select_features(['MG','1d_DG','BPM','blink_durations'])
        vilearn_train_scaler: VilearnMLTrain = VilearnMLTrain(nested_cv=nested_cv, 
                                                    auto_cv=(not nested_cv_manual),
                                                        binary_clf=binary_clf,
                                                        data_loader=data_loader_dyads, ml_models=models_scaler)
        vilearn_train_scaler.train_and_evaluate(eval_label=f"SCALER_dyads_f_MG_1DG_BPM_BDM_{suffix_run}",
                                                group_label="Dyads", features_label="AIED", sampling_label=f"{sampling}", debug=debug)
    # Scaler models, triads 
    if triads and scaler and aied_feautures:
        data_loader_triads.select_features(['MG','1d_DG','BPM','blink_durations'])
        vilearn_train_scaler: VilearnMLTrain = VilearnMLTrain(nested_cv=nested_cv, 
                                                    auto_cv=(not nested_cv_manual),
                                                        binary_clf=binary_clf,
                                                        data_loader=data_loader_triads, ml_models=models_scaler)
        vilearn_train_scaler.train_and_evaluate(eval_label=f"SCALER_triads_f_MG_1DG_BPM_BDM_{suffix_run}",
                                                group_label="Triads", features_label="AIED", sampling_label=f"{sampling}", debug=debug)

    ### AIxVR FEATURES ###
    # Select features for dyads and triads according to AIxVR paper for both simple and scaler models
    # Simple models, all groups, AIxVR paper features (blink rate, MG)
    if all_groups and simple and aixvr_features:
        data_loader.select_features(['MG','BPM'])
        vilearn_train_simple: VilearnMLTrain = VilearnMLTrain(nested_cv=nested_cv, 
                                                    auto_cv=(not nested_cv_manual),
                                                        binary_clf=binary_clf,
                                                        data_loader=data_loader, ml_models=models)
        vilearn_train_simple.train_and_evaluate(eval_label=f"SIMPLE_all_groups_f_MG_BPM_{suffix_run}",
                                                group_label="All", features_label="AIxVR", sampling_label=f"{sampling}", 
                                                separate_avg_groups=separate_avg_groups, debug=debug)
    # simple model dyads, features (1DG, MG)
    if dyads and simple and aixvr_features:
        data_loader_dyads.select_features(['MG','1d_DG'])
        vilearn_train_simple: VilearnMLTrain = VilearnMLTrain(nested_cv=nested_cv, 
                                                    auto_cv=(not nested_cv_manual),
                                                        binary_clf=binary_clf,
                                                        data_loader=data_loader_dyads, ml_models=models)
        vilearn_train_simple.train_and_evaluate(eval_label=f"SIMPLE_dyads_f_MG_1DG_{suffix_run}",
                                                group_label="Dyads", features_label="AIxVR", sampling_label=f"{sampling}", debug=debug)
    # simple model triads, features (BPM)
    if triads and simple and aixvr_features:
        data_loader_triads.select_features(['BPM'])
        vilearn_train_simple: VilearnMLTrain = VilearnMLTrain(nested_cv=nested_cv, 
                                                    auto_cv=(not nested_cv_manual),
                                                        binary_clf=binary_clf,
                                                        data_loader=data_loader_triads, ml_models=models)
        vilearn_train_simple.train_and_evaluate(eval_label=f"SIMPLE_triads_f_BPM_{suffix_run}",
                                                group_label="Triads", features_label="AIxVR", sampling_label=f"{sampling}", debug=debug)
    # Scaler models, all groups AIxVR paper features (blink rate, MG)
    if all_groups and scaler and aixvr_features:
        data_loader.select_features(['MG','BPM'])
        vilearn_train_scaler: VilearnMLTrain = VilearnMLTrain(nested_cv=nested_cv, 
                                                    auto_cv=(not nested_cv_manual),
                                                        binary_clf=binary_clf,
                                                        data_loader=data_loader, ml_models=models_scaler)
        vilearn_train_scaler.train_and_evaluate(eval_label=f"SCALER_all_groups_f_MG_BPM_{suffix_run}",
                                                group_label="All", features_label="AIxVR", sampling_label=f"{sampling}", 
                                                separate_avg_groups=separate_avg_groups, debug=debug)
    # Scaler models, dyads features (1DG, MG)
    if dyads and scaler and aixvr_features:
        data_loader_dyads.select_features(['MG','1d_DG'])
        vilearn_train_scaler: VilearnMLTrain = VilearnMLTrain(nested_cv=nested_cv, 
                                                    auto_cv=(not nested_cv_manual),
                                                        binary_clf=binary_clf,
                                                        data_loader=data_loader_dyads, ml_models=models_scaler)
        vilearn_train_scaler.train_and_evaluate(eval_label=f"SCALER_dyads_f_MG_1DG_{suffix_run}",
                                                group_label="Dyads", features_label="AIxVR", sampling_label=f"{sampling}", debug=debug)
    # Scaler models, triads features (BPM)
    if triads and scaler and aixvr_features:
        data_loader_triads.select_features(['BPM'])
        vilearn_train_scaler: VilearnMLTrain = VilearnMLTrain(nested_cv=nested_cv, 
                                                    auto_cv=(not nested_cv_manual),
                                                        binary_clf=binary_clf,
                                                        data_loader=data_loader_triads, ml_models=models_scaler)
        vilearn_train_scaler.train_and_evaluate(eval_label=f"SCALER_triads_f_BPM_{suffix_run}",
                                                group_label="Triads", features_label="AIxVR", sampling_label=f"{sampling}", debug=debug)

    ### BLINKS + SPEAKINGxGAZE ICMI FEATURES ###
    # Select features for gaze x speaking ICMI iteration (substitute gaze only features with GazexSpeaking)
    # Simple models, all groups, Blinks features + GazexSpeaking Features
    if all_groups and simple and blink_speaking_x_gaze_features:
        data_loader.select_features(['BPM','blink_durations',"G_OnSpeaker", "No_G_OnSpeaker", "G_SI", "No_G_SI"])
        vilearn_train_simple: VilearnMLTrain = VilearnMLTrain(nested_cv=nested_cv, 
                                                    auto_cv=(not nested_cv_manual),
                                                        binary_clf=binary_clf,
                                                        data_loader=data_loader, ml_models=models)
        vilearn_train_simple.train_and_evaluate(eval_label=f"SIMPLE_all_groups_f_Blinks_GazexSpeaking_{suffix_run}",
                                                group_label="All", features_label="Blinks_GazexSpeaking", sampling_label=f"{sampling}", 
                                                separate_avg_groups=separate_avg_groups, debug=debug)
    # simple model dyads, Blinks features + GazexSpeaking Features
    if dyads and simple and blink_speaking_x_gaze_features:
        data_loader_dyads.select_features(['BPM','blink_durations',"G_OnSpeaker", "No_G_OnSpeaker", "G_SI", "No_G_SI"])
        vilearn_train_simple: VilearnMLTrain = VilearnMLTrain(nested_cv=nested_cv, 
                                                    auto_cv=(not nested_cv_manual),
                                                        binary_clf=binary_clf,
                                                        data_loader=data_loader_dyads, ml_models=models)
        vilearn_train_simple.train_and_evaluate(eval_label=f"SIMPLE_dyads_f_Blinks_GazexSpeaking_{suffix_run}",
                                                group_label="Dyads", features_label="Blinks_GazexSpeaking", sampling_label=f"{sampling}", debug=debug)
    # simple model triads, Blinks features + GazexSpeaking Features
    if triads and simple and blink_speaking_x_gaze_features:
        data_loader_triads.select_features(['BPM','blink_durations',"G_OnSpeaker", "No_G_OnSpeaker", "G_SI", "No_G_SI"])
        vilearn_train_simple: VilearnMLTrain = VilearnMLTrain(nested_cv=nested_cv, 
                                                    auto_cv=(not nested_cv_manual),
                                                        binary_clf=binary_clf,
                                                        data_loader=data_loader_triads, ml_models=models)
        vilearn_train_simple.train_and_evaluate(eval_label=f"SIMPLE_triads_f_Blinks_GazexSpeaking_{suffix_run}",
                                                group_label="Triads", features_label="Blinks_GazexSpeaking", sampling_label=f"{sampling}", debug=debug)
    # Scaler models, all groups, Blinks features + GazexSpeaking Features
    if all_groups and scaler and blink_speaking_x_gaze_features:
        data_loader.select_features(['BPM','blink_durations',"G_OnSpeaker", "No_G_OnSpeaker", "G_SI", "No_G_SI"])
        vilearn_train_scaler: VilearnMLTrain = VilearnMLTrain(nested_cv=nested_cv, 
                                                    auto_cv=(not nested_cv_manual),
                                                        binary_clf=binary_clf,
                                                        data_loader=data_loader, ml_models=models_scaler)
        vilearn_train_scaler.train_and_evaluate(eval_label=f"SCALER_all_groups_f_Blinks_GazexSpeaking_{suffix_run}",
                                                group_label="All", features_label="Blinks_GazexSpeaking", sampling_label=f"{sampling}", 
                                                separate_avg_groups=separate_avg_groups, debug=debug)
    # Scaler models, dyads, Blinks features + GazexSpeaking Features
    if dyads and scaler and blink_speaking_x_gaze_features:
        data_loader_dyads.select_features(['BPM','blink_durations',"G_OnSpeaker", "No_G_OnSpeaker", "G_SI", "No_G_SI"])
        vilearn_train_scaler: VilearnMLTrain = VilearnMLTrain(nested_cv=nested_cv, 
                                                    auto_cv=(not nested_cv_manual),
                                                        binary_clf=binary_clf,
                                                        data_loader=data_loader_dyads, ml_models=models_scaler)
        vilearn_train_scaler.train_and_evaluate(eval_label=f"SCALER_dyads_f_Blinks_GazexSpeaking_{suffix_run}",
                                                group_label="Dyads", features_label="Blinks_GazexSpeaking", sampling_label=f"{sampling}", debug=debug)
    # Scaler models, triads, Blinks features + GazexSpeaking Features
    if triads and scaler and blink_speaking_x_gaze_features:
        data_loader_triads.select_features(['BPM','blink_durations',"G_OnSpeaker", "No_G_OnSpeaker", "G_SI", "No_G_SI"])
        vilearn_train_scaler: VilearnMLTrain = VilearnMLTrain(nested_cv=nested_cv, 
                                                    auto_cv=(not nested_cv_manual),
                                                        binary_clf=binary_clf,
                                                        data_loader=data_loader_triads, ml_models=models_scaler)
        vilearn_train_scaler.train_and_evaluate(eval_label=f"SCALER_triads_f_Blinks_GazexSpeaking_{suffix_run}",
                                                group_label="Triads", features_label="Blinks_GazexSpeaking", sampling_label=f"{sampling}", debug=debug)

    ### BLINKS ONLY FEATURES ###
    # Select features for blinks only ICMI iteration
    # Simple models, all groups, Blinks features + GazexSpeaking Features
    if all_groups and simple and blinks_only:
        data_loader.select_features(['BPM','blink_durations'])
        vilearn_train_simple: VilearnMLTrain = VilearnMLTrain(nested_cv=nested_cv, 
                                                    auto_cv=(not nested_cv_manual),
                                                        binary_clf=binary_clf,
                                                        data_loader=data_loader, ml_models=models)
        vilearn_train_simple.train_and_evaluate(eval_label=f"SIMPLE_all_groups_f_Blinks_{suffix_run}",
                                                group_label="All", features_label="Blinks", sampling_label=f"{sampling}", 
                                                separate_avg_groups=separate_avg_groups, debug=debug)
    # simple model dyads, Blinks features + GazexSpeaking Features
    if dyads and simple and blinks_only:
        data_loader_dyads.select_features(['BPM','blink_durations'])
        vilearn_train_simple: VilearnMLTrain = VilearnMLTrain(nested_cv=nested_cv, 
                                                    auto_cv=(not nested_cv_manual),
                                                        binary_clf=binary_clf,
                                                        data_loader=data_loader_dyads, ml_models=models)
        vilearn_train_simple.train_and_evaluate(eval_label=f"SIMPLE_dyads_f_Blinks_{suffix_run}",
                                                group_label="Dyads", features_label="Blinks", sampling_label=f"{sampling}", debug=debug)
    # simple model triads, Blinks features + GazexSpeaking Features
    if triads and simple and blinks_only:
        data_loader_triads.select_features(['BPM','blink_durations'])
        vilearn_train_simple: VilearnMLTrain = VilearnMLTrain(nested_cv=nested_cv, 
                                                    auto_cv=(not nested_cv_manual),
                                                        binary_clf=binary_clf,
                                                        data_loader=data_loader_triads, ml_models=models)
        vilearn_train_simple.train_and_evaluate(eval_label=f"SIMPLE_triads_f_Blinks_{suffix_run}",
                                                group_label="Triads", features_label="Blinks", sampling_label=f"{sampling}", debug=debug)
    # Scaler models, all groups, Blinks features + GazexSpeaking Features
    if all_groups and scaler and blinks_only:
        data_loader.select_features(['BPM','blink_durations'])
        vilearn_train_scaler: VilearnMLTrain = VilearnMLTrain(nested_cv=nested_cv, 
                                                    auto_cv=(not nested_cv_manual),
                                                        binary_clf=binary_clf,
                                                        data_loader=data_loader, ml_models=models_scaler)
        vilearn_train_scaler.train_and_evaluate(eval_label=f"SCALER_all_groups_f_Blinks_{suffix_run}",
                                                group_label="All", features_label="Blinks", sampling_label=f"{sampling}", 
                                                separate_avg_groups=separate_avg_groups, debug=debug)
    # Scaler models, dyads, Blinks features + GazexSpeaking Features
    if dyads and scaler and blinks_only:
        data_loader_dyads.select_features(['BPM','blink_durations'])
        vilearn_train_scaler: VilearnMLTrain = VilearnMLTrain(nested_cv=nested_cv, 
                                                    auto_cv=(not nested_cv_manual),
                                                        binary_clf=binary_clf,
                                                        data_loader=data_loader_dyads, ml_models=models_scaler)
        vilearn_train_scaler.train_and_evaluate(eval_label=f"SCALER_dyads_f_Blinks_{suffix_run}",
                                                group_label="Dyads", features_label="Blinks", sampling_label=f"{sampling}", debug=debug)
    # Scaler models, triads, Blinks features + GazexSpeaking Features
    if triads and scaler and blinks_only:
        data_loader_triads.select_features(['BPM','blink_durations'])
        vilearn_train_scaler: VilearnMLTrain = VilearnMLTrain(nested_cv=nested_cv, 
                                                    auto_cv=(not nested_cv_manual),
                                                        binary_clf=binary_clf,
                                                        data_loader=data_loader_triads, ml_models=models_scaler)
        vilearn_train_scaler.train_and_evaluate(eval_label=f"SCALER_triads_f_Blinks_{suffix_run}",
                                                group_label="Triads", features_label="Blinks", sampling_label=f"{sampling}", debug=debug)


    # TODO: run svm poly separately because it takes too long

    print("done")