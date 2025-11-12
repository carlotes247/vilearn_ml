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
    # load vilearn windowed data
    working_dir = os.getcwd()
    data_folder = 'Recordings/SavedData/v2_no_low_sampled'
    data_file = 'all_features_60s_resampled.csv'
    full_data_path = os.path.join(working_dir, data_folder, data_file)
    data = pd.read_csv(full_data_path)
    # names for x, y cols
    dyad_cols = [col for col in data.columns if 'dyad' in col]
    triad_cols = [col for col in data.columns if 'triad' in col]
    dyad_X_cols = [col for col in dyad_cols if not 'target' in col]
    triad_X_cols = [col for col in triad_cols if not 'target' in col]
    dyad_target_cols = [col for col in dyad_cols if 'target' in col]
    triad_target_cols = [col for col in triad_cols if 'target' in col]
    bins = [-0.1, 0.3, 0.7, 1]
    labels = [0, 1, 2]
    labels_text = ['low', 'middle', 'high']
    dyads_y = data[dyad_target_cols]
    dyads_y_categorical= data[dyad_target_cols].apply(pd.cut, bins=bins, labels=labels_text)
    triads_y_categorical= data[triad_target_cols].apply(pd.cut, bins=bins, labels=labels_text)
    
    print(dyads_y_categorical.stack().value_counts())
    print(triads_y_categorical.stack().value_counts())

    # plt.bar(dyads_y_categorical.stack().value_counts(), height=1)
    # plt.show()
    print("binning")
    # splitting into train, test set
    # dyads
    dyads_X_train, dyads_X_test, dyads_y_train, dyads_y_test = model_selection.train_test_split(data[dyad_X_cols], data[dyad_target_cols])
    
    # knn with scaling
    clf = Pipeline(
    steps=[("scaler", StandardScaler()), ("knn", neighbors.KNeighborsClassifier(n_neighbors=11))]
    )
    #knn without scaling

    
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