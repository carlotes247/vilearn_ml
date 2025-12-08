import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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

if __name__ == '__main__':
    # load data
    data_loader: VilearnWindowedDataLoaderML = VilearnWindowedDataLoaderML(bins_binary=False, print_folds=False, debug_all_folds=False)
    
    # Defining all the classifiers to try
    names = [
        "Nearest Neighbors",
        "Linear SVM",
        "RBF SVM",
        "Decision Tree",
        "Random Forest",
        "Neural Net",
        "AdaBoost",
        "Naive Bayes",
        "QDA"
    ]

    # TODO: Explore other models that I don't understand:
    # Gaussian Process
    # TODO: Improve the parameter tuning of:
    # Decision Tree, Random Forest Tree, Adaboost (seems more or less ok?)
    # TODO: Explore other parameter search options:
    # Random search, bayesion optimization, Neural Net (Hidden Layer Sizes)
    # See: https://www.geeksforgeeks.org/machine-learning/how-to-tune-a-decision-tree-in-hyperparameter-tuning/
    # See: https://www.geeksforgeeks.org/machine-learning/random-forest-hyperparameter-tuning-in-python/
    param_grids_models = {
        "Nearest Neighbors": {
            'estimator': neighbors.KNeighborsClassifier(),
            'params': {
                'n_neighbors': np.arange(2, 60, 1),
                'weights': ['uniform', 'distance']
            }
        },
        "Linear SVM": {
            'estimator': svm.LinearSVC(dual=False),
            'params': {
                'penalty': ['l1', 'l2'],
                'loss': ['hinge', 'squared_hinge'],
                'C': [0.1, 1, 5, 10, 100]
            }
        },
        "SVM": {
            'estimator': svm.SVC(),
            'params': {
                'kernel': ['linear', 'rbf', 'poly'],
                'C': [0.1, 1, 5, 10, 100],
                'gamma': [0.1,0.01,0.001,1,10],
                'degree':[0,1,2,3,4,5,6]
            }
        },
        "Decision Tree": {
            'estimator': tree.DecisionTreeClassifier(),
            'params': {
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        },
        "Random Forest": {
            'estimator': ensemble.RandomForestClassifier(),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'bootstrap': [True, False]
            }
        },
        "Neural Net": {
            'estimator': neural_network.MLPClassifier(max_iter=1000),
            'params': {
                'hidden_layer_sizes': [(10,30,10),(20,)],
                'activation': ['tanh', 'relu'],
                'solver': ['sgd', 'adam'],
                'alpha': [0.0001, 0.05, 1],
                'learning_rate': ['constant','adaptive'], 
            }
        },
        "AdaBoost": {
            'estimator': ensemble.AdaBoostClassifier(),
            'params': {
                'n_estimators': [10, 50, 100, 500],
                'learning_rate': [0.0001, 0.001, 0.01, 0.1, 1.0, 10]
            }
        },
        "Naive Bayes": {
            'estimator': naive_bayes.GaussianNB(),
            'params': {
                'var_smoothing': np.logspace(0,-9, num=100)
            }
        },
        "QDA": {
            'estimator': discriminant_analysis.QuadraticDiscriminantAnalysis(),
            'params': {
                'reg_param': [0.1, 0.2, 0.3, 0.4, 0.5]
            }
        }
    }

    classifiers = [
        neighbors.KNeighborsClassifier(3),
        svm.SVC(kernel="linear", C=0.025, random_state=42),
        svm.SVC(gamma=2, C=1, random_state=42),
        gaussian_process.GaussianProcessClassifier(1.0 * gaussian_process.kernels.RBF(1.0), random_state=42),
        tree.DecisionTreeClassifier(max_depth=5, random_state=42),
        ensemble.RandomForestClassifier(
            max_depth=5, n_estimators=10, max_features=1, random_state=42
        ),
        neural_network.MLPClassifier(max_iter=1000),
        ensemble.AdaBoostClassifier(random_state=42),
        naive_bayes.GaussianNB(),
        discriminant_analysis.QuadraticDiscriminantAnalysis(),
    ]

    # Cross validation for all models
    for model_name, model_dict in param_grids_models.items():        
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

        # We ensure to do a nested CV
        # inner cv, outer cv NEEDED for nested CV
        inner_cv_simple = copy.deepcopy(data_loader.group_kfold)
        outer_cv_simple = copy.deepcopy(data_loader.group_kfold)
        inner_cv_scaler = copy.deepcopy(data_loader.group_kfold)
        outer_cv_scaler = copy.deepcopy(data_loader.group_kfold)
        grid_search_cv_simple = model_selection.GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=inner_cv_simple, 
            verbose=1)
        # grid_search_cv_scaler = model_selection.GridSearchCV(
        #     estimator=model_scaler,
        #     param_grid=param_grid,
        #     cv=inner_cv_scaler, 
        #     verbose=1)
        nested_score_simple = model_selection.cross_val_score(estimator=model,        
                                                X=data_loader.X, y=data_loader.y,
                                                cv=outer_cv_simple, 
                                                groups=data_loader.groups, verbose=1)
        # nested_score_scaler = model_selection.cross_val_score(estimator=model_scaler,        
        #                                         X=data_loader.X, y=data_loader.y,
        #                                         cv=outer_cv_scaler, 
        #                                         groups=data_loader.groups, verbose=1)
        print(f"Avg acc SIMPLE: {nested_score_simple.mean()}")
        # print(f"Avg acc SCALER: {nested_score_scaler.mean()}")


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