import glob
import os
import sys
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
    audio_combined = np.concatenate([data['audio_features'] for data in participants_with_features.values()], axis=0)
    visual_combined = np.concatenate([data['visual_features'] for data in participants_with_features.values()], axis=0)
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


def train_svm(metaparams: dict, features: np.ndarray, labels: List[np.ndarray], dev_features: np.ndarray) -> List[
    np.ndarray]:
    """ Trains the SVM model for every label types in the provided list of labels.
    In this case, we send least_dominant and most_dominant labels separately in the list of labels.
    """
    # create the model
    models = [SVC(**metaparams, class_weight='balanced') for _ in range(len(labels))]
    # fit the mode
    for i, model in enumerate(models):
        model.fit(features, labels[i])
    # predict the labels
    predictions = [model.predict(dev_features) for model in models]
    return predictions


def leave_one_instace_out_svm(elea_labels: dict_data_type, dome_labels: dict_data_type, metaparams_svc: dict,
                              dev_dataset: str):
    # features are already normalized and PCA-transformed
    metrics = {
        'val_accuracy': accuracy_score,
        'val_recall': partial(recall_score, average='macro'),
    }
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
        least_dom, most_dom = train_svm(metaparams_svc,
                                        features=combined_features[np.arange(len(idx_to_ids)) != idx],
                                        labels=[combined_labels_least[np.arange(len(idx_to_ids)) != idx],
                                                combined_labels_most[np.arange(len(idx_to_ids)) != idx]],
                                        dev_features=combined_features[idx].reshape((1, -1)))
        predictions[idx_to_ids[idx]]['least_dominant'] = least_dom
        predictions[idx_to_ids[idx]]['most_dominant'] = most_dom
    # calculate metrics
    # if dev dataset is dome, we need to calculate metrics for dome only. THeir participants always have 'IS' in ids
    # elea always has 'g' in ids. If all is selected, we need to calculate metrics for all participants
    particles_in_ids = {'dome': 'IS', 'elea': 'g', 'all': ''}
    least_dom = np.array([predictions[value]['least_dominant'] for value in idx_to_ids.values() if
                          particles_in_ids[dev_dataset] in value])
    most_dom = np.array([predictions[value]['most_dominant'] for value in idx_to_ids.values() if
                         particles_in_ids[dev_dataset] in value])
    least_dom_true = np.array([combined_dict[value]['label_least_dominant'] for value in idx_to_ids.values() if
                               particles_in_ids[dev_dataset] in value])
    most_dom_true = np.array([combined_dict[value]['label_most_dominant'] for value in idx_to_ids.values() if
                              particles_in_ids[dev_dataset] in value])

    # calculate metrics for least dominant, including the accuracy for detecting the least dominant (when prediction and true label both equal 1)
    metrics_values_least = {metric_name: metric(least_dom_true, least_dom) for metric_name, metric in metrics.items()}
    metrics_values_least.update(
        {'least_dom_detection': np.sum((least_dom_true == 1) & (least_dom == 1)) / np.sum(least_dom_true == 1)})

    # calculate metrics for most dominant, including the accuracy for detecting the most dominant (when prediction and true label both equal 1)
    metrics_values_most = {metric_name: metric(most_dom_true, most_dom) for metric_name, metric in metrics.items()}
    metrics_values_most.update(
        {'most_dom_detection': np.sum((most_dom_true == 1) & (most_dom == 1)) / np.sum(most_dom_true == 1)})
    return metrics_values_least, metrics_values_most


@cli.command("main")
@click.option('--normalization', default=None, help="To apply the normalization to data or not", type=bool)
@click.option('--pca', default=None, help="To apply PCA to data or not", type=bool)
@click.option('--dataset', default=None,
              help="Which dataset will be used for leave-one-instace-out cross-validation. Can be: dome, elea, all",
              type=str)
@click.option('--output_path', default=None, help="Output folder for logging", type=str)
def main(normalization: bool, pca: bool, dataset: str, output_path: str):
    training_params: dict = {
        'normalization': normalization,
        'pca': pca,
        'dataset': dataset,
        'output_path': output_path,
        'svc_metaparams': {
            'C': [0.01, 0.1, 1, 10, 100],
            'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
            'degree': [2, 3, 4],
            'gamma': ['scale', 'auto'],
        },
        'elea_visual_paths': glob.glob("/work/home/dsu/Datasets/ELEA/extracted_features/*.csv"),
        'elea_audio_paths': glob.glob("/work/home/dsu/Datasets/ELEA/extracted_audio_features/*/*16*"),
        'elea_labels_paths': glob.glob(
            "/work/home/dsu/Datasets/ELEA/preprocessed_labels/ELEA_external_annotations/Annotator*.csv"),
        'dome_visual_paths': glob.glob("/work/home/dsu/Datasets/DOME/extracted_features/*.csv"),
        'dome_audio_paths': glob.glob("/work/home/dsu/Datasets/DOME/extracted_audio_features/*/*16*"),
        'dome_labels_paths': "/work/home/dsu/Datasets/DOME/Annotations/dome_annotations_M1.csv",
    }

    # create output folder if needed
    if not os.path.exists(training_params['output_path']):
        os.makedirs(training_params['output_path'])
    # create logger
    logger = pd.DataFrame(
        columns=['dataset', 'normalization', 'pca', 'metaparams', 'least_dominant_accuracy', 'least_dominant_recall',
                 'least_dominant_detection_rate',
                 'most_dominant_accuracy', 'most_dominant_recall', 'most_dominant_detection_rate'])
    logger.to_csv(os.path.join(output_path, f'svc_results_{dataset}_norm_{normalization}_pca_{pca}.csv'))
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
        dome_data = apply_PCA_to_features(dome_data)
        elea_data = apply_PCA_to_features(elea_data)
    # grid search for the best parameters
    for C in training_params['svc_metaparams']['C']:
        for kernel in training_params['svc_metaparams']['kernel']:
            for gamma in training_params['svc_metaparams']['gamma']:
                for degree in training_params['svc_metaparams']['degree']:
                    metaparams_svc = {'C': C, 'kernel': kernel, 'degree': degree, 'gamma': gamma}
                    metrics_values_least, metrics_values_most = leave_one_instace_out_svm(elea_data, dome_data,
                                                                                          metaparams_svc, dataset)
                    print(f'Metaparameters: {metaparams_svc}')
                    print(f'Least dominant: {metrics_values_least}')
                    print(f'Most dominant: {metrics_values_most}')
                    # write results to the logger
                    row = pd.DataFrame({'dataset': dataset, 'normalization': normalization, 'pca': pca,
                                        'metaparams': str(metaparams_svc),
                                        'least_dominant_accuracy': metrics_values_least['val_accuracy'],
                                        'least_dominant_recall': metrics_values_least['val_recall'],
                                        'least_dominant_detection_rate': metrics_values_least['least_dom_detection'],
                                        'most_dominant_accuracy': metrics_values_most['val_accuracy'],
                                        'most_dominant_recall': metrics_values_most['val_recall'],
                                        'most_dominant_detection_rate': metrics_values_most['most_dom_detection']},
                                       index=[0])
                    logger = pd.concat([logger, row], ignore_index=True)
    # save the results
    logger.to_csv(os.path.join(output_path, f'svc_results_{dataset}_norm_{normalization}_pca_{pca}.csv'))


if __name__ == "__main__":
    cli()
