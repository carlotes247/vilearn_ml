import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import neighbors
from sklearn import model_selection
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import DecisionBoundaryDisplay
import os

# This script is a test for Knn regression

if __name__ == '__main__':
    # config flags
    use_file_TE: bool = True
    separate_groups: bool = False
    # bining vars
    bins = [-0.1, 0.3, 0.7, 1]
    labels = [0, 1, 2]
    labels_text = ['low', 'middle', 'high']
    # dataframes
    df_X: pd.DataFrame = pd.DataFrame()
    df_y: pd.DataFrame = pd.DataFrame()
    # paths
    working_dir = os.getcwd()
    data_folder = 'Recordings/SavedData/v2_no_low_sampled'
    data_file = 'all_features_60s_resampled.csv'
    data_file_with_TE = '60s_TE_correlation.csv'
    sep = ";" if use_file_TE else ","
    full_data_path = ""
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
        # TODO: crossvalidation loop where we leave one group out and train on the others
        X_train_names, X_test_names = model_selection.train_test_split(group_names)
        df_train = df_data.loc[df_data['group_name'].isin(X_train_names)]
        df_test = df_data.loc[df_data['group_name'].isin(X_test_names)]
        X_train = df_train.drop(columns=['seconds_interaction_window','group_type','group_formation','group_name' ,'TE'])
        y_train = df_train.loc[:, 'TE']
        X_test = df_test.drop(columns=['seconds_interaction_window','group_type','group_formation','group_name' ,'TE'])
        y_test = df_test.loc[:, 'TE']
    # knn with scaling
    clf = Pipeline(
    steps=[("scaler", StandardScaler()), ("knn", neighbors.KNeighborsClassifier(n_neighbors=11))]
    )
    #knn without scaling
    knn_simple = neighbors.KNeighborsClassifier(n_neighbors=11)
    
    # for ax, weights in zip(axs, ("uniform", "distance")):
    # clf.set_params(knn__weights=weights).fit(X_train, y_train)
    # disp = DecisionBoundaryDisplay.from_estimator(
    #     clf,
    #     X_test,
    #     response_method="predict",
    #     plot_method="pcolormesh",
    #     xlabel=iris.feature_names[0],
    #     ylabel=iris.feature_names[1],
    #     shading="auto",
    #     alpha=0.5,
    #     ax=ax,
    # )
    # scatter = disp.ax_.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, edgecolors="k")
    # disp.ax_.legend(
    #     scatter.legend_elements()[0],
    #     iris.target_names,
    #     loc="lower left",
    #     title="Classes",
    # )
    # _ = disp.ax_.set_title(
    #     f"3-Class classification\n(k={clf[-1].n_neighbors}, weights={weights!r})"
    # )


    print('hello')