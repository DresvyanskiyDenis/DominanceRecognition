import glob
import sys
import os
from typing import Dict, Union

import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearnex import patch_sklearn
from sklearnex.svm import SVC

patch_sklearn()

from src.training.simple_methods.data_utils import load_and_prepare_DOME_and_ELEA_features_and_labels


dict_data_type = Dict[str, Dict[str, Union[np.ndarray, pd.DataFrame, float, int]]]


def calculate_statistics_for_every_participant(participants_with_features:dict_data_type)\
        ->dict_data_type:
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
        if audio_df.shape[0]==0:
            num_features = (len(audio_df.columns)-4)*len(statistics_to_calculate)
            audio_statistics.append(np.zeros((num_features,)))
        else:
            data_to_use = audio_df.drop(columns=['filename', 'participant_id', 'start_timestep', 'end_timestep']).values.astype('float')
            for stat in statistics_to_calculate:
                audio_statistics.append(stat(data_to_use, axis=0))
        # calculate visual statistics, If they are empty, fill with zeros
        if visual_df.shape[0]==0:
            num_features = (len(visual_df.columns)-6)*len(statistics_to_calculate)
            visual_statistics.append(np.zeros((num_features,)))
        else:
            data_to_use = visual_df.drop(columns=['video_name', 'participant_id', 'frame_number', 'timestep', 'filename', 'found_face']).values.astype('float')
            for stat in statistics_to_calculate:
                visual_statistics.append(stat(data_to_use, axis=0))
        # add the result to the dictionary
        result[participant_id] = {'audio_features': np.concatenate(audio_statistics), 'visual_features': np.concatenate(visual_statistics),
                                    'label_least_dominant': data['label_least_dominant'], 'label_most_dominant': data['label_most_dominant']}
    return result



def normalize_features(participants_with_features:dict_data_type)->dict_data_type:
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
    audio_combined = np.concatenate([data['audio_features'].reshape((1,-1)) for data in participants_with_features.values()], axis=0)
    visual_combined = np.concatenate([data['visual_features'].reshape((1,-1)) for data in participants_with_features.values()], axis=0)
    # create normalizers
    audio_normalizer = MinMaxScaler(feature_range=(-1,1))
    visual_normalizer = MinMaxScaler(feature_range=(-1,1))
    # fit normalizers
    audio_normalizer = audio_normalizer.fit(audio_combined)
    visual_normalizer = visual_normalizer.fit(visual_combined)
    # normalize features
    result = {}
    for participant_id, data in participants_with_features.items():
        result[participant_id] = {'audio_features': audio_normalizer.transform(data['audio_features'].reshape((1,-1))).squeeze(),
                                  'visual_features': visual_normalizer.transform(data['visual_features'].reshape((1,-1))).squeeze(),
                                  'label_least_dominant': data['label_least_dominant'], 'label_most_dominant': data['label_most_dominant']}
    return result

def apply_PCA_to_features(participants_with_features:dict_data_type)->dict_data_type:
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
        result[participant_id] = {'audio_features': audio_pca.transform(data['audio_features']),
                                  'visual_features': visual_pca.transform(data['visual_features']),
                                  'label_least_dominant': data['label_least_dominant'], 'label_most_dominant': data['label_most_dominant']}
    return result


def train_SVM(elea_labels:dict_data_type, dome_labels:dict_data_type, metaparams_svc:dict):
    # features are already normalized and PCA-transformed
    metrics = {
        'val_accuracy': accuracy_score,
        'val_recall': recall_score,
        'val_precision': precision_score,
        'val_f1': f1_score,
    }
    # we make a grid search for the best parameters, making cross-validation at the same time
    # cross-validation is made in the leave-one-instance-out manner
    # while both datasets are combined for training and cross-validation, we also need to get the metric values on
    # separate datasets
    combined_dataset = {**elea_labels, **dome_labels}
    combined_audio_features = np.concatenate([data['audio_features'].reshape((1,-1)) for data in combined_dataset.values()], axis=0)
    combined_visual_features = np.concatenate([data['visual_features'].reshape((1,-1)) for data in combined_dataset.values()], axis=0)
    combined_features = np.concatenate([combined_audio_features, combined_visual_features], axis=1)
    combined_labels_least_dom = np.array([data['label_least_dominant'] for data in combined_dataset.values()])
    combined_labels_most_dom = np.array([data['label_most_dominant'] for data in combined_dataset.values()])
    # create lists for storing predictions and true labels
    predictions_least = []
    predictions_most = []
    true_labels_least = []
    true_labels_most = []
    for participant_idx in range(len(combined_dataset)):
        # leave-one-instance-out cross-validation
        # get the training and validation sets
        training_indices = np.arange(len(combined_dataset))
        training_indices = np.delete(training_indices, participant_idx)
        validation_indices = np.array([participant_idx])
        # get the training and validation sets
        training_features = combined_features[training_indices]
        validation_features = combined_features[validation_indices]
        training_labels_least_dom = combined_labels_least_dom[training_indices]
        validation_labels_least_dom = combined_labels_least_dom[validation_indices]
        training_labels_most_dom = combined_labels_most_dom[training_indices]
        validation_labels_most_dom = combined_labels_most_dom[validation_indices]
        # train the model
        svc_least_dom = SVC(C=metaparams_svc['C'], kernel=metaparams_svc['kernel'], degree=metaparams_svc['degree'],
                    gamma=metaparams_svc['gamma'])
        svc_least_dom.fit(training_features, training_labels_least_dom)
        svc_most_dom = SVC(C=metaparams_svc['C'], kernel=metaparams_svc['kernel'], degree=metaparams_svc['degree'],
                    gamma=metaparams_svc['gamma'])
        svc_most_dom.fit(training_features, training_labels_most_dom)
        # get the predictions
        predictions_least_dom = svc_least_dom.predict(validation_features)
        predictions_most_dom = svc_most_dom.predict(validation_features)
        # store the predictions and true labels
        predictions_least.append(predictions_least_dom)
        predictions_most.append(predictions_most_dom)
        true_labels_least.append(validation_labels_least_dom)
        true_labels_most.append(validation_labels_most_dom)
    # calculate the metrics
    metrics_values_least = {metric_name: metric_function(true_labels_least, predictions_least) for metric_name, metric_function in metrics.items()}
    metrics_values_most = {metric_name: metric_function(true_labels_most, predictions_most) for metric_name, metric_function in metrics.items()}
    return metrics_values_least, metrics_values_most



def visualize_features_t_SNE():
    pass


def main():
    elea_visual_paths = glob.glob("/work/home/dsu/Datasets/ELEA/extracted_features/*.csv")
    elea_audio_paths = glob.glob("/work/home/dsu/Datasets/ELEA/extracted_audio_features/*/*16*")
    elea_labels_paths = glob.glob(
        "/work/home/dsu/Datasets/ELEA/preprocessed_labels/ELEA_external_annotations/Annotator*.csv")
    dome_visual_paths = glob.glob("/work/home/dsu/Datasets/DOME/extracted_features/*.csv")
    dome_audio_paths = glob.glob("/work/home/dsu/Datasets/DOME/extracted_audio_features/*/*16*")
    dome_labels_paths = "/work/home/dsu/Datasets/DOME/Annotations/dome_annotations_M1.csv"
    # load the labels
    dome_labels, elea_labels = load_and_prepare_DOME_and_ELEA_features_and_labels(elea_visual_paths, elea_audio_paths,
                                                                                  elea_labels_paths,
                                                                                  dome_visual_paths, dome_audio_paths,
                                                                                  dome_labels_paths)
    # calculate statistics for every participant
    dome_statistics = calculate_statistics_for_every_participant(dome_labels)
    elea_statistics = calculate_statistics_for_every_participant(elea_labels)
    # normalize features
    dome_statistics_normalized = normalize_features(dome_statistics)
    elea_statistics_normalized = normalize_features(elea_statistics)
    # define metaparameters for the SVM
    all_metaparams_svc = {
        'C':[0.01, 0.1, 1, 10, 100],
        'kernel':['linear', 'poly', 'rbf', 'sigmoid'],
        'degree':[2, 3, 4],
        'gamma':['scale', 'auto'],
    }
    # grid search for the best parameters
    for C in all_metaparams_svc['C']:
        for kernel in all_metaparams_svc['kernel']:
            for degree in all_metaparams_svc['degree']:
                for gamma in all_metaparams_svc['gamma']:
                    metaparams_svc = {'C':C, 'kernel':kernel, 'degree':degree, 'gamma':gamma}
                    metrics_values_least, metrics_values_most = train_SVM(elea_statistics_normalized, dome_statistics_normalized, metaparams_svc)
                    print(f'Metaparameters: {metaparams_svc}')
                    print(f'Least dominant: {metrics_values_least}')
                    print(f'Most dominant: {metrics_values_most}')



if __name__ == "__main__":
    main()