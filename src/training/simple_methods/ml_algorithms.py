from functools import partial

from skelm import ELMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ml_alrogithms = {
    'DT': partial(DecisionTreeClassifier),
    'RF': partial(RandomForestClassifier, n_jobs=-1),
    'ELM': ELMClassifier,
    'knn': partial(KNeighborsClassifier, n_jobs=-1),
    'MLP': partial(MLPClassifier, max_iter=300, learning_rate_init=0.005),
    'SVC': partial(SVC, probability=True),
}

hyperparams = {
    'ELM': {
        'n_neurons': [10, 50, 100, 200],
    },
    'RF': {
            'n_estimators': [10, 100, 500, 1000],
            'max_features': ['sqrt', 'log2'],
            'class_weight': ['balanced', None],
        },
    'SVC': {
        'C': [ 0.1, 1, 10, 100],
        'gamma': 'scale',
        'kernel': ['rbf', 'linear', 'sigmoid'],
        'class_weight': ['balanced', None],
    },
    'knn': {
        'n_neighbors': [3, 5, 11, 19],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    },
    'MLP': {
        'hidden_layer_sizes': [(100,), (300,), (100,100), (300,100)],
        'activation': ['tanh', 'relu'],
        'solver': ['lbfgs', 'adam'],
        'alpha': [0.0001, 0.05],
    },
    'DT': {
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random'],
        'class_weight': ['balanced', None],
    },
}
