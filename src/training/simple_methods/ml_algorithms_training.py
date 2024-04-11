import glob
import itertools
import os
import sys

from tqdm import tqdm

# dynamically append the path to the project to the system path
path_to_project = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))+os.sep
sys.path.append(path_to_project)
sys.path.append(path_to_project.replace('DominanceRecognition', 'datatools'))
from functools import partial
from typing import Dict, Union, List

import click
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, recall_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from src.training.simple_methods.data_utils import load_and_prepare_DOME_and_ELEA_features_and_labels
from src.training.simple_methods.ml_algorithms import ml_alrogithms, hyperparams


@click.group()
def cli():
    pass


dict_data_type = Dict[str, Dict[str, Union[np.ndarray, pd.DataFrame, float, int]]]


def calculate_statistics_for_every_participant(participants_with_features: dict_data_type) \
        -> dict_data_type:
    """ Calculates various statistics (defined inside function) for every participant in the provided dictionary using
    the provided dataframes (audio and visual features).

    :param participants_with_features: Dict[str, Dict[str, Union[pd.DataFrame, float, int]]]
        Dictionary with participant ids as keys and dataframes with features as values.
        It has the following format:
        {'participant_id': {'audio_features': pd.DataFrame, 'visual_features': pd.DataFrame,
        'least_dominant': float}, 'most_dominant': float}
        }
    :return: Dict[str, Dict[str, Union[np.ndarray, float, int]]]
        Dictionary with participant ids as keys and dictionaries with statistics as values.
        It has the following format:
        {'participant_id': {'audio_features': np.ndarray, 'visual_features': np.ndarray,
        'least_dominant': float, 'most_dominant': float,
        }, where np.ndarray contains statistics for the corresponding features.
    """
    statistics_to_calculate = [np.mean, np.std, kurtosis, skew]
    result = {}
    for participant_id, data in participants_with_features.items():
        audio_df = data['audio_features']
        visual_df = data['visual_features']
        # drop nan values
        audio_df.dropna(inplace=True)
        visual_df.dropna(inplace=True)
        audio_statistics = []
        visual_statistics = []
        # calculate audio statistics. If the dataframe is empty, fill with zeros
        if audio_df.shape[0] == 0:
            num_features = (len(audio_df.columns) - 4) * len(statistics_to_calculate)
            audio_statistics.append(np.zeros((num_features,)))
        else:
            data_to_use = audio_df.drop(
                columns=['filename', 'participant_id', 'start_timestep', 'end_timestep']).values.astype('float')
            for stat in statistics_to_calculate:
                audio_statistics.append(stat(data_to_use, axis=0))
        # calculate visual statistics, If they are empty, fill with zeros
        if visual_df.shape[0] == 0:
            num_features = (len(visual_df.columns) - 6) * len(statistics_to_calculate)
            visual_statistics.append(np.zeros((num_features,)))
        else:
            data_to_use = visual_df.drop(
                columns=['video_name', 'participant_id', 'frame_number', 'timestep', 'filename',
                         'found_face']).values.astype('float')
            for stat in statistics_to_calculate:
                visual_statistics.append(stat(data_to_use, axis=0))
        # add the result to the dictionary
        result[participant_id] = {'audio_features': np.concatenate(audio_statistics),
                                  'visual_features': np.concatenate(visual_statistics),
                                  'label_least_dominant': data['label_least_dominant'],
                                  'label_most_dominant': data['label_most_dominant']}
    return result


def normalize_features(participants_with_features: dict_data_type) -> dict_data_type:
    """ Normalizes the features for every participant. To do so, combines audio and visual features from all participants
    (separately) and fit the normalizer. Afterwords, this normalizer will be applied to the features of every participant.

    :param participants_with_features: Dict[str, Dict[str, Union[np.ndarray, pd.DataFrame, float, int]]]
        Dictionary with participant ids as keys and dictionaries with statistics as values.
        It has the following format:
        {'participant_id': {'audio_features': np.ndarray, 'visual_features': np.ndarray,
        'least_dominant': float, 'most_dominant': float,
        }, where np.ndarray contains statistics for the corresponding features.
    :return: Dict[str, Dict[str, Union[np.ndarray, pd.DataFrame, float, int]]]
        Dictionary with participant ids as keys and dictionaries with statistics as values.
        It has the following format:
        {'participant_id': {'audio_features': np.ndarray, 'visual_features': np.ndarray,
        'least_dominant': float, 'most_dominant': float,
        }, where np.ndarray contains normalized statistics for the corresponding features.
    """
    audio_combined = np.concatenate(
        [data['audio_features'].reshape((1, -1)) for data in participants_with_features.values()], axis=0)
    visual_combined = np.concatenate(
        [data['visual_features'].reshape((1, -1)) for data in participants_with_features.values()], axis=0)
    # create normalizers
    audio_normalizer = MinMaxScaler(feature_range=(-1, 1))
    visual_normalizer = MinMaxScaler(feature_range=(-1, 1))
    # fit normalizers
    audio_normalizer = audio_normalizer.fit(audio_combined)
    visual_normalizer = visual_normalizer.fit(visual_combined)
    # normalize features
    result = {}
    for participant_id, data in participants_with_features.items():
        result[participant_id] = {
            'audio_features': audio_normalizer.transform(data['audio_features'].reshape((1, -1))).squeeze(),
            'visual_features': visual_normalizer.transform(data['visual_features'].reshape((1, -1))).squeeze(),
            'label_least_dominant': data['label_least_dominant'], 'label_most_dominant': data['label_most_dominant']}
    return result


def apply_PCA_to_features(participants_with_features: dict_data_type) -> dict_data_type:
    """ Applies PCA to the features of every participant. To do so, combines audio and visual features from all participants
    (separately) and fit the PCA. Afterwords, this PCA will be applied to the features of every participant.

    :param participants_with_features: Dict[str, Dict[str, Union[np.ndarray, pd.DataFrame, float, int]]]
        Dictionary with participant ids as keys and dictionaries with statistics as values.
        It has the following format:
        {'participant_id': {'audio_features': np.ndarray, 'visual_features': np.ndarray,
        'least_dominant': float, 'most_dominant': float,
        }, where np.ndarray contains statistics for the corresponding features.
    :return: Dict[str, Dict[str, Union[np.ndarray, pd.DataFrame, float, int]]]
        Dictionary with participant ids as keys and dictionaries with statistics as values.
        It has the following format:
        {'participant_id': {'audio_features': np.ndarray, 'visual_features': np.ndarray,
        'least_dominant': float, 'most_dominant': float,
        }, where np.ndarray contains PCA-transformed statistics for the corresponding features.
    """
    audio_combined = np.concatenate([data['audio_features'].reshape((1, -1)) for data in participants_with_features.values()], axis=0)
    visual_combined = np.concatenate([data['visual_features'].reshape((1, -1)) for data in participants_with_features.values()], axis=0)
    # create PCA
    audio_pca = PCA(n_components=0.95)
    visual_pca = PCA(n_components=0.95)
    # fit PCA
    audio_pca = audio_pca.fit(audio_combined)
    visual_pca = visual_pca.fit(visual_combined)
    # apply PCA
    result = {}
    for participant_id, data in participants_with_features.items():
        result[participant_id] = {'audio_features': audio_pca.transform(data['audio_features'].reshape((1, -1))).squeeze(),
                                  'visual_features': visual_pca.transform(data['visual_features'].reshape((1, -1))).squeeze(),
                                  'label_least_dominant': data['label_least_dominant'],
                                  'label_most_dominant': data['label_most_dominant']}
    return result

def train_classifier(ml_model, hyperparams:dict, train_features: np.ndarray, labels: List[np.ndarray], dev_features: np.ndarray)->List[np.ndarray]:
    # create ML model
    models = [ml_model(**hyperparams) for _ in range(len(labels))]
    # train the model
    for i, model in enumerate(models):
        model.fit(train_features, labels[i])
    # predict the labels
    predictions = [model.predict_proba(dev_features) for model in models]
    return predictions


def leave_one_instance_out_predictions(ml_model, hyperparameters, elea_labels: dict_data_type, dome_labels: dict_data_type):
    # features are already normalized and PCA-transformed
    # create dict idx_to_ids
    combined_dict = {**elea_labels, **dome_labels}
    idx_to_ids = {idx: participant_id for idx, participant_id in enumerate(combined_dict.keys())}
    # create predictions dictionary participant_id->'least_dominant'->int, 'most_dominant'->int
    predictions = {participant_id: {'least_dominant': None, 'most_dominant': None} for participant_id in
                   idx_to_ids.values()}
    combined_features = np.concatenate([np.concatenate([combined_dict[value]['visual_features'].reshape((1, -1)),
                                                        combined_dict[value]['audio_features'].reshape((1, -1))],
                                                       axis=1)
                                        for value in idx_to_ids.values()], axis=0)
    combined_labels_least = np.array([combined_dict[value]['label_least_dominant'] for value in idx_to_ids.values()])
    combined_labels_most = np.array([combined_dict[value]['label_most_dominant'] for value in idx_to_ids.values()])
    # leave-one-out cross-validation
    for idx in idx_to_ids.keys():
        least_dom, most_dom = train_classifier(ml_model, hyperparameters,
                                        train_features=combined_features[np.arange(len(idx_to_ids)) != idx],
                                        labels=[combined_labels_least[np.arange(len(idx_to_ids)) != idx],
                                                combined_labels_most[np.arange(len(idx_to_ids)) != idx]],
                                        dev_features=combined_features[idx].reshape((1, -1)))
        predictions[idx_to_ids[idx]]['least_dominant'] = least_dom # we get probability of being least dominant
        predictions[idx_to_ids[idx]]['least_dominant_label'] = combined_labels_least[idx] # inject the label into predictions
        predictions[idx_to_ids[idx]]['most_dominant'] = most_dom # we get probability of being most dominant
        predictions[idx_to_ids[idx]]['most_dominant_label'] = combined_labels_most[idx] # inject the label into predictions

    return predictions

def calculate_least_and_most_dominant_accuracies_from_probabilities(predictions:dict):
    # divide predictions and labels into DOME and ELEA
    dome_predictions = {key: value for key, value in predictions.items() if 'IS' in key}
    elea_predictions = {key: value for key, value in predictions.items() if 'g' in key}
    # DOME
    # get unique names of sessions
    dome_session_names = set(['_'.join([key.split('_')[0], key.split('_')[-2], key.split('_')[-1]]) for key in dome_predictions.keys()])
    # divide participants on groups according to the session
    groups = []
    for session in dome_session_names:
        session_name = session.split('_')[0]
        session_part = session.split('_')[-2] + '_' + session.split('_')[-1]
        group = [key for key in dome_predictions.keys() if session_name in key and session_part in key]
        group = [dome_predictions[key] for key in group]
        groups.append(group)
    # calculate accuracy for every group
    right_or_not_least = []
    right_or_not_most = []
    for group in groups:
        label_least = np.argmax([item['least_dominant_label'] for item in group])
        label_most = np.argmax([item['most_dominant_label'] for item in group])
        # get the most probable label
        least_dominant = np.argmax([item['least_dominant'].squeeze()[1] for item in group])
        most_dominant = np.argmax([item['most_dominant'].squeeze()[1] for item in group])
        # check if the most probable label is the same as the true label
        right_or_not_least.append(label_least == least_dominant)
        right_or_not_most.append(label_most == most_dominant)
    # calculate accuracy
    accuracy_least_dome = np.mean(right_or_not_least)
    accuracy_most_dome = np.mean(right_or_not_most)
    # ELEA
    # get unique names of sessions
    elea_session_names = set([key.split('_')[0] for key in elea_predictions.keys()])
    # divide participants on groups according to the session
    groups = []
    for session in elea_session_names:
        group = [key for key in elea_predictions.keys() if session in key]
        group = [elea_predictions[key] for key in group]
        groups.append(group)
    # calculate accuracy for every group
    right_or_not_least = []
    right_or_not_most = []
    for group in groups:
        label_least = np.argmax([item['least_dominant_label'] for item in group])
        label_most = np.argmax([item['most_dominant_label'] for item in group])
        # get the most probable label
        least_dominant = np.argmax([item['least_dominant'].squeeze()[1] for item in group])
        most_dominant = np.argmax([item['most_dominant'].squeeze()[1] for item in group])
        # check if the most probable label is the same as the true label
        right_or_not_least.append(label_least == least_dominant)
        right_or_not_most.append(label_most == most_dominant)
    # calculate accuracy
    accuracy_least_elea = np.mean(right_or_not_least)
    accuracy_most_elea = np.mean(right_or_not_most)

    return {
        'dome_least_acc': accuracy_least_dome,
        'dome_most_acc': accuracy_most_dome,
        'elea_least_acc': accuracy_least_elea,
        'elea_most_acc': accuracy_most_elea
    }













@cli.command("main")
@click.option('--normalization', default=None, help="To apply the normalization to data or not", type=bool)
@click.option('--pca', default=None, help="To apply PCA to data or not", type=bool)
@click.option('--audio_features', default=None, help="type_of_audio_features", type=str)
@click.option('--output_path', default=None, help="Output folder for logging", type=str)
def main(normalization: bool, pca: bool, audio_features:str, output_path: str):
    training_params: dict = {
        'normalization': normalization,
        'pca': pca,
        'output_path': output_path,
        'svc_metaparams': {
            'C': [0.01, 0.1, 1, 10, 100],
            'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
            'degree': [2, 3, 4],
            'gamma': ['scale', 'auto'],
        },
        'elea_visual_paths': glob.glob("/work/home/dsu/Datasets/ELEA/extracted_features/*.csv"),
        'elea_audio_paths': glob.glob(f"/work/home/dsu/Datasets/ELEA/extracted_audio_features/*/*{audio_features}*"),
        'elea_labels_paths': glob.glob(
            "/work/home/dsu/Datasets/ELEA/preprocessed_labels/ELEA_external_annotations/Annotator*.csv"),
        'dome_visual_paths': glob.glob("/work/home/dsu/Datasets/DOME/extracted_features/*.csv"),
        'dome_audio_paths': glob.glob(f"/work/home/dsu/Datasets/DOME/extracted_audio_features/*/*{audio_features}*"),
        'dome_labels_paths': "/work/home/dsu/Datasets/DOME/Annotations/dome_annotations_M1.csv",
    }

    # create output folder if needed
    if not os.path.exists(training_params['output_path']):
        os.makedirs(training_params['output_path'])
    # create logger
    logger = pd.DataFrame(
        columns=['ml_algorithm', 'hyperparameters', 'dome_least_acc', 'dome_most_acc', 'elea_least_acc', 'elea_most_acc'])
    logger.to_csv(os.path.join(output_path, f'ml_algorithms_results_{audio_features}_norm_{normalization}_pca_{pca}.csv'))
    # load the labels
    dome_data, elea_data = load_and_prepare_DOME_and_ELEA_features_and_labels(training_params['elea_visual_paths'],
                                                                              training_params['elea_audio_paths'],
                                                                              training_params['elea_labels_paths'],
                                                                              training_params['dome_visual_paths'],
                                                                              training_params['dome_audio_paths'],
                                                                              training_params['dome_labels_paths'])
    # calculate statistics for every participant
    dome_data = calculate_statistics_for_every_participant(dome_data)
    elea_data = calculate_statistics_for_every_participant(elea_data)
    # normalize features if needed
    if normalization:
        dome_data = normalize_features(dome_data)
        elea_data = normalize_features(elea_data)
    # apply PCA if needed
    if pca:
        # concatenate dome and elea data
        mixed_data = {**dome_data, **elea_data}
        mixed_data = apply_PCA_to_features(mixed_data)
        # split the data back to dome and elea
        dome_data = {key: value for key, value in mixed_data.items() if 'IS' in key}
        elea_data = {key: value for key, value in mixed_data.items() if 'g' in key}
    # grid search for the best parameters
    for ml_algorithm in ml_alrogithms.keys():
        all_hyperparameters = hyperparams[ml_algorithm]
        keys, values = zip(*all_hyperparameters.items())
        combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        # test every combination
        for combination in tqdm(combinations, desc=f"Testing {ml_algorithm}"):
            ml_alg = ml_alrogithms[ml_algorithm]
            current_predictions = leave_one_instance_out_predictions(ml_alg, combination,
                                               elea_labels=elea_data, dome_labels=dome_data)
            accuracies = calculate_least_and_most_dominant_accuracies_from_probabilities(current_predictions)
            print(f"Algorithm: {ml_algorithm}, combination: {combination}")
            print(accuracies)
            print('----------------------------------------------------------------------')
            row = pd.DataFrame({'normalization': normalization, 'pca': pca,
                                        'ml_algorithm': ml_algorithm, 'hyperparameters': str(combination),
                                        'dome_least_acc': accuracies['dome_least_acc'],
                                        'dome_most_acc': accuracies['dome_most_acc'],
                                        'elea_least_acc': accuracies['elea_least_acc'],
                                        'elea_most_acc': accuracies['elea_most_acc']},
                                       index=[0])
            logger = pd.concat([logger, row], ignore_index=True)
            logger.to_csv(os.path.join(output_path, f'ml_algorithms_results_{audio_features}_norm_{normalization}_pca_{pca}.csv'))



if __name__ == "__main__":
    cli()
