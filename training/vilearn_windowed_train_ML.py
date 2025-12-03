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
# Added this try catch because on some machines it cannot find folders from working directory 
try:
    from data_reading.vilearn_windowed_data_loader_ML import VilearnWindowedDataLoaderML
except ImportError:
    import sys
    sys.path.append(os.getcwd())
    from data_reading.vilearn_windowed_data_loader_ML import VilearnWindowedDataLoaderML

if __name__ == '__main__':
    # load data
    data_loader: VilearnWindowedDataLoaderML = VilearnWindowedDataLoaderML(bins_binary=False, print_folds=False, debug_all_folds=False)
    
    # Defining knn models
    # knn with scaling
    knn_scaler = Pipeline(
    steps=[("scaler", StandardScaler()), ("knn", neighbors.KNeighborsClassifier())]
    )
    #knn without scaling
    knn_simple = neighbors.KNeighborsClassifier()
    
    # Hyper parameter tuning
    # kf=model_selection.KFold(n_splits=5,shuffle=True,random_state=42)
    neighbours_candidates= np.arange(2, 60, 1)
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
        cv=data_loader.group_kfold, 
        verbose=1)
    knn_cv_scaler = model_selection.GridSearchCV(
        estimator=knn_scaler,
        param_grid=param_grid_scaler,
        cv=data_loader.group_kfold, 
        verbose=1
    )
    print("Searching best parameter for model")
    knn_cv_simple.fit(data_loader.X, data_loader.y, groups=data_loader.groups)
    knn_cv_scaler.fit(data_loader.X,data_loader.y, groups=data_loader.groups)
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
    accuracies_knn_simple = model_selection.cross_val_score(knn_simple,
                                                             data_loader.X, data_loader.y,
                                                             cv=data_loader.group_kfold, 
                                                             groups=data_loader.groups, verbose=1)
    avg_acc_knn_simple = np.average(accuracies_knn_simple)
    #print(accuracies)
    print(f"Avg acc: {avg_acc_knn_simple}")
    print("Cross val score knn_SCALER")
    accuracies_knn_scaler = model_selection.cross_val_score(knn_scaler,
                                                            data_loader.X, data_loader.y, 
                                                            cv=data_loader.group_kfold, 
                                                            groups=data_loader.groups, verbose=1)
    avg_acc_knn_scaler =np.average(accuracies_knn_scaler)
    #print(accuracies)
    print(f"Avg acc: {avg_acc_knn_scaler}")

    # Pick best model
    best_knn = None
    best_knn_name = ""
    best_knn_score = 0
    if avg_acc_knn_simple > avg_acc_knn_scaler:
        best_knn = knn_cv_simple.best_estimator_
        best_knn_name = "KNN_SIMPLE"
        best_knn_score = avg_acc_knn_simple
    else:
        best_knn = knn_cv_scaler.best_estimator_
        best_knn_name = "KNN_SCALER"
        best_knn_score = avg_acc_knn_scaler

    y_pred = model_selection.cross_val_predict(estimator=best_knn, 
                                               X=data_loader.X, y=data_loader.y, 
                                               cv=data_loader.group_kfold, 
                                               groups=data_loader.groups)
    conf_mat = metrics.confusion_matrix(data_loader.y, y_pred)
    conf_mat = conf_mat/len(data_loader.y)*100

    # Plot non-normalized confusion matrix
    np.set_printoptions(precision=2)
    titles_options = [
        ("Confusion matrix, without normalization", None),
        ("Normalized confusion matrix", "true"),
    ]
    for title, normalize in titles_options:
        disp = metrics.ConfusionMatrixDisplay(
            #best_knn,
            #X_test,
            #y_test,
            confusion_matrix=conf_mat,
            display_labels=data_loader.labels_text_three,
            #cmap=plt.cm.Blues,
            #normalize=normalize,
        )
        #disp.ax_.set_title(title)
        disp.plot()
        plt.show()

    print("done")