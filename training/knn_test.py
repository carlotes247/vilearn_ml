import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import neighbors
from sklearn import model_selection
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn import metrics
import os

# This script is a test for Knn regression

if __name__ == '__main__':
    # config flags
    use_file_TE: bool = True
    separate_groups: bool = False
    print_folds: bool = False
    debug_all_folds: bool = False
    # bining vars
    # three classes
    # bins = [-0.1, 0.3, 0.7, 1]
    # labels = [0, 1, 2]
    # labels_text = ['low', 'middle', 'high']
    # two classes
    bins = [-0.1, 0.5, 1]
    labels = [0, 1]
    labels_text = ['low', 'high']
    # dataframes
    df_X: pd.DataFrame = pd.DataFrame()
    df_y: pd.DataFrame = pd.DataFrame()
    X: pd.DataFrame = pd.DataFrame()
    y: pd.Series = pd.Series()
    X_train: pd.DataFrame = pd.DataFrame()
    y_train: pd.Series = pd.Series()
    X_test: pd.DataFrame = pd.DataFrame()
    y_test: pd.Series = pd.Series()
    # k fold vars
    group_names: np.ndarray
    groups: pd.Series = pd.Series()
    group_kfold: model_selection.GroupKFold = model_selection.GroupKFold()
    # paths
    working_dir = os.getcwd()
    data_folder = 'Recordings/SavedData/v2_no_low_sampled'
    data_file = 'all_features_60s_resampled.csv'
    data_file_with_TE = '60s_TE_correlation.csv'
    sep = ";" if use_file_TE else ","
    full_data_path = ""
    
    # Defining knn models
    # knn with scaling
    knn_scaler = Pipeline(
    steps=[("scaler", StandardScaler()), ("knn", neighbors.KNeighborsClassifier())]
    )
    #knn without scaling
    knn_simple = neighbors.KNeighborsClassifier()
    
    # load vilearn windowed data
    if use_file_TE:
        full_data_path = os.path.join(working_dir, data_folder, data_file_with_TE)
    else:
        full_data_path = os.path.join(working_dir, data_folder, data_file)
    df_data = pd.read_csv(full_data_path, sep=";")
    # logic for TE file used for R correlation analysis (long dataframe)
    if use_file_TE:
        # binning TE
        df_y = df_data.loc[:, ['TE']].apply(pd.cut, bins=bins, labels=labels_text)
        df_data['TE'] = df_y
        df_X = df_data.loc[:, df_data.columns != 'TE']
        # separating into dyads and triads
        dyads_df = df_data.loc[df_data['group_name'].str.contains("dyad")]
        triads_df = df_data.loc[df_data['group_name'].str.contains("triad")]
    # logic for wide dataframe with all gaze configurations (missing TE and blink metrics)
    else:
        # names for x, y cols
        dyad_cols = [col for col in df_data.columns if 'dyad' in col]
        triad_cols = [col for col in df_data.columns if 'triad' in col]
        dyad_X_cols = [col for col in dyad_cols if not 'target' in col]
        triad_X_cols = [col for col in triad_cols if not 'target' in col]
        dyad_target_cols = [col for col in dyad_cols if 'target' in col]
        triad_target_cols = [col for col in triad_cols if 'target' in col]
    
        dyads_y = df_data[dyad_target_cols]
        dyads_y_categorical= df_data[dyad_target_cols].apply(pd.cut, bins=bins, labels=labels_text)
        triads_y_categorical= df_data[triad_target_cols].apply(pd.cut, bins=bins, labels=labels_text)
    
        print(dyads_y_categorical.stack().value_counts())
        print(triads_y_categorical.stack().value_counts())

    # splitting into train, test set
    if separate_groups:
        # dyads
        dyads_X_train, dyads_X_test, dyads_y_train, dyads_y_test = model_selection.train_test_split(df_X, df_y)
    else:
        # since we want whole groups and not individual windows, we use a proxy var to calculate the split
        group_names = df_X['group_name'].unique()
        # TODO: probably suffle groups eaech iteration?
        X_train_names, X_test_names = model_selection.train_test_split(group_names)
        df_train = df_data.loc[df_data['group_name'].isin(X_train_names)]
        df_test = df_data.loc[df_data['group_name'].isin(X_test_names)]
        X_train = df_train.drop(columns=['seconds_interaction_window','group_type','group_formation','group_name' ,'TE'])
        y_train = df_train.loc[:, 'TE']
        X_test = df_test.drop(columns=['seconds_interaction_window','group_type','group_formation','group_name' ,'TE'])
        y_test = df_test.loc[:, 'TE']
        
        # Now trying to use GroupkFold to do the split to see if it works with our dataset (this is independent from the prior code, I am testing things out)        
        X = df_data.drop(columns=['seconds_interaction_window','group_type','group_formation','group_name' ,'TE'])
        y = df_data.loc[:, 'TE']
        groups = df_data['group_name']
        group_kfold = model_selection.GroupKFold(n_splits=len(group_names))
        n_splits = group_kfold.get_n_splits(X, y, groups)
        if print_folds:
                print(f"Num folds: {n_splits}")
                print(group_kfold)
        # debug code to understand if all folds are formed correctly
        if debug_all_folds:
            acc_list_simple_uniform = []
            acc_list_scaler_uniform = []
            acc_list_simple_distance = []
            acc_list_scaler_distance = []
            # loop through each fold
            for i, (train_index, test_index) in enumerate(group_kfold.split(X, y, groups)):                                    
                # debug info
                if print_folds:
                    print(f"Fold {i}:")            
                    print(f"  TRAIN: groups={groups[train_index].unique()}, count={len(groups[test_index].unique())}")
                    print("")
                    print(f"  TEST: groups={groups[test_index].unique()}, count={len(groups[test_index].unique())}")
                    if len(groups[test_index].unique()) > 1:
                        print("THIS FOLD HAS MORE THAN ONE GROUP!!!")
                    print("============================")    
                # manual training KNN
                for weights in ("uniform", "distance"):
                    knn_scaler.set_params(knn__weights=weights).fit(X.iloc[train_index], y.iloc[train_index])
                    knn_simple.__weights = weights
                    knn_simple.fit(X.iloc[train_index], y.iloc[train_index])
                    # evaluate
                    y_pred_simple = knn_simple.predict(X.iloc[test_index])
                    y_pred_scaler = knn_scaler.predict(X.iloc[test_index])
                    acc_simple = metrics.accuracy_score(y.iloc[test_index], y_pred_simple)   
                    acc_scaler = metrics.accuracy_score(y.iloc[test_index], y_pred_scaler)
                    if weights == "uniform":
                        acc_list_simple_uniform.append(acc_simple)
                        acc_list_scaler_uniform.append(acc_scaler) 
                    else:
                        acc_list_simple_distance.append(acc_simple)
                        acc_list_scaler_distance.append(acc_scaler) 
                    print("========SIMPLE========")
                    print(f"KNN accuracy {weights} simple: {acc_simple}")
                    print(metrics.classification_report(y.iloc[test_index], y_pred_simple, zero_division=0))
                    print("========SCALER=========")
                    print(f"KNN accuracy {weights} scaler: {acc_scaler}")
                    print(metrics.classification_report(y.iloc[test_index], y_pred_scaler, zero_division=0))
                    print("=======================")

                # calculate avg accuracies over all folds
                print(f"Avg accuracy KNN simple uniform: {np.average(acc_list_simple_uniform)}")
                print(f"Avg accuracy KNN scaler uniform: {np.average(acc_list_scaler_uniform)}")
                print(f"Avg accuracy KNN simple distance: {np.average(acc_list_simple_distance)}")
                print(f"Avg accuracy KNN scaler distance: {np.average(acc_list_scaler_distance)}")


    # Hyper parameter tuning
    # kf=model_selection.KFold(n_splits=5,shuffle=True,random_state=42)
    neighbours_candidates= np.arange(2, 100, 1)
    # Define parameter grid
    param_grid = {
        'n_neighbors': neighbours_candidates,
        'weights': ['uniform', 'distance']
    }

    param_grid_scaler = {
        'knn__n_neighbors': neighbours_candidates,
        'knn__weights': ['uniform', 'distance']
    }


    # Setup GridSearchCV with GroupKFold
    knn_cv_simple = model_selection.GridSearchCV(
        estimator=knn_simple,
        param_grid=param_grid,
        cv=group_kfold, 
        verbose=1)
    knn_cv_scaler = model_selection.GridSearchCV(
        estimator=knn_scaler,
        param_grid=param_grid_scaler,
        cv=group_kfold, 
        verbose=1
    )
    print("Searching best parameter for model")
    knn_cv_simple.fit(X, y, groups=groups)
    knn_cv_scaler.fit(X,y, groups=groups)
    # knn_cv_scaler.fit(X, y, groups=groups)
    # 7. Results
    print(f"Best KNN_SIMPLE neighbor paramenter found: {knn_cv_simple.best_params_}")
    print("Best CV KNN_SIMPLE score:", knn_cv_simple.best_score_)
    print(f"Best KNN_SCALER neighbor paramenter found: {knn_cv_scaler.best_params_}")
    print("Best CV KNN_SCALER score:", knn_cv_scaler.best_score_)


    # 8. Repeat evaluation
    # override models with best ones found
    knn_simple = knn_cv_simple.best_estimator_
    knn_scaler = knn_cv_scaler.best_estimator_

    # print(f"Best KNN_SCALER neighbor paramenter found: {knn_cv_scaler.best_params_}")
    # Setting best parameter found
    # knn_simple.set_params(n_neighbors=knn_cv_simple.best_params_)
    # knn_scaler.set_params(n_neighbors=knn_cv_scaler.best_params_)

    print("Cross val score knn_SIMPLE")
    accuracies = model_selection.cross_val_score(knn_simple, X, y, cv=group_kfold, groups=groups, verbose=1)
    #print(accuracies)
    print(f"Avg acc: {np.average(accuracies)}")
    print("Cross val score knn_SCALER")
    accuracies = model_selection.cross_val_score(knn_scaler, X, y, cv=group_kfold, groups=groups, verbose=1)
    #print(accuracies)
    print(f"Avg acc: {np.average(accuracies)}")
    print("done")

    # for weights in ("uniform", "distance"):
    #     knn_scaler.set_params(knn__weights=weights).fit(X_train, y_train)
    #     knn_simple.__weights = weights
    #     knn_simple.fit(X_train,y_train)
    #     # evaluate
    #     y_pred_simple = knn_simple.predict(X_test)
    #     y_pred_scaler = knn_scaler.predict(X_test)
    #     acc_simple = metrics.accuracy_score(y_test, y_pred_simple)   
    #     acc_scaler = metrics.accuracy_score(y_test, y_pred_scaler)   
    #     print("========SIMPLE========")
    #     print(f"KNN accuracy {weights} simple: {acc_simple}")
    #     print(metrics.classification_report(y_test, y_pred_simple, zero_division=0))
    #     print("========SCALER=========")
    #     print(f"KNN accuracy {weights} scaler: {acc_scaler}")
    #     print(metrics.classification_report(y_test, y_pred_scaler, zero_division=0))
    #     print("=======================")


