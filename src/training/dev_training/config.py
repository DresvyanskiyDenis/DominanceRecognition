from functools import partial

from skelm import ELMClassifier
from sklearn.svm import SVC

ml_alrogithms = {
    'ELM': ELMClassifier,
    'SVC': partial(SVC, probability=True),
}

hyperparams = {
    'ELM': {
        'n_neurons': [200],
    },
}
