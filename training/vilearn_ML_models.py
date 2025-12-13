# classifiers imports
from sklearn import dummy
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
import numpy as np

class VilearnMLModels:
    # class vars
    param_grid = None
    def __init__(self, scaler: bool = False) -> None:
        # Defining all the classifiers to try
        # TODO: Explore other models that I don't understand:
        # Gaussian Process
        # TODO: Improve the parameter tuning of:
        # Decision Tree, Random Forest Tree, Adaboost (seems more or less ok?)
        # TODO: Explore other parameter search options:
        # Random search, bayesion optimization, Neural Net (Hidden Layer Sizes)
        # See: https://www.geeksforgeeks.org/machine-learning/how-to-tune-a-decision-tree-in-hyperparameter-tuning/
        # See: https://www.geeksforgeeks.org/machine-learning/random-forest-hyperparameter-tuning-in-python/
        self.param_grids_models = {
            "Baseline Most Frequent Strategy": {
                'estimator': dummy.DummyClassifier(strategy='most_frequent'),
                'params': {}
            },
            "Baseline Prior Strategy": {
                'estimator': dummy.DummyClassifier(strategy='prior'),
                'params': {}
            },
            "Baseline Stratified Strategy": {
                'estimator': dummy.DummyClassifier(strategy='stratified'),
                'params': {}
            },
            "Baseline Uniform Strategy": {
                'estimator': dummy.DummyClassifier(strategy='uniform'),
                'params': {}
            },
            "Nearest Neighbors Test": {
                'estimator': neighbors.KNeighborsClassifier(),
                'params': {
                    'n_neighbors': np.arange(1, 3, 1),
                    'weights': ['uniform', 'distance']
                }
            },
            "Nearest Neighbors": {
                'estimator': neighbors.KNeighborsClassifier(),
                'params': {
                    'n_neighbors': np.arange(2, 70, 1),
                    'weights': ['uniform', 'distance']
                }
            },
            "Linear SVM l1": {
                'estimator': svm.LinearSVC(dual="auto"),
                'params': {
                    'penalty': ['l1'],
                    'loss': ['squared_hinge'],
                    'C': [0.01, 0.1, 1, 5, 10, 100],
                    'max_iter': [5000, 10000, 50000]
                }
            },
            "Linear SVM l2": {
                'estimator': svm.LinearSVC(dual="auto"),
                'params': {
                    'penalty': ['l2'],
                    'loss': ['hinge', 'squared_hinge'],
                    'C': [0.01, 0.1, 1, 5, 10, 100],
                    'max_iter': [5000, 10000, 50000]
                }
            },
            "SVM": {
                'estimator': svm.SVC(),
                'params': {
                    'kernel': ['linear', 'rbf', 'poly'],
                    'C': [0.1, 1],
                    'gamma': [0.1,0.01,0.001],
                    'degree':[0,1,2,4]
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

        # Adding pipeline elements if requested
        if (scaler):
            self.param_grids_models = self.__add_scalers(self.param_grids_models)
        

    def __add_scalers(self, param_grid_models) -> list[dict]:
        # create alternative model with scaler to see what scores better
        # the scaler can be useful to standardize features. It depends on how the features approximate the std normal distribution of the data (e.g. Gaussian with 0 mean and unit variance).
        # more info on scalers: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
        for model_name, model_dict in param_grid_models.items(): 
            model = model_dict['estimator']
            param_grid = model_dict['params']
            model_scaler = Pipeline(
                steps=[("scaler", StandardScaler()), ("clf", model)]
            )
            param_grid_scaler = {f'clf__{k}': v for k, v in param_grid.items()}
            param_grid_models[model_name]['estimator'] = model_scaler
            param_grid_models[model_name]['params'] = param_grid_scaler
        return param_grid_models

        