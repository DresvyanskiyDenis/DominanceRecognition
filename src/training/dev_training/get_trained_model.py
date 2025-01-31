import glob
import os
import itertools
from typing import Union, List

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from src.training.dev_training.config import ml_alrogithms, hyperparams
from src.training.simple_methods.complex_approach_training import calculate_statistics_for_every_participant, \
    transform_features_to_interpersonal, leave_one_instance_out_predictions, \
    calculate_least_and_most_dominant_accuracies_from_probabilities, dict_data_type
from src.training.simple_methods.data_utils import load_and_prepare_DOME_and_ELEA_features_and_labels

def normalize_features(participants_with_features: dict_data_type) -> Union[dict_data_type, MinMaxScaler, MinMaxScaler]:
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
    return result, audio_normalizer, visual_normalizer


def apply_PCA_to_features(participants_with_features: dict_data_type) -> Union[dict_data_type, PCA, PCA]:
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
    return result, audio_pca, visual_pca

def train_classifier(ml_model, hyperparameters, elea_labels: dict_data_type, dome_labels: dict_data_type):
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

    # create model
    model_most = ml_model(**hyperparameters)
    model_least = ml_model(**hyperparameters)
    train_features = combined_features
    labels_least = combined_labels_least
    labels_most = combined_labels_most
    # train models
    model_most.fit(train_features, labels_most)
    model_least.fit(train_features, labels_least)
    # return
    return model_most, model_least






def main(statistics, normalization: bool, pca: bool, audio_features:str, output_path: str):
    training_params: dict = {
        'statistics': statistics.split(','),
        'normalization': normalization,
        'pca': pca,
        'output_path': output_path,
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
    dome_data = calculate_statistics_for_every_participant(dome_data, training_params['statistics'])
    elea_data = calculate_statistics_for_every_participant(elea_data, training_params['statistics'])
    # apply PCA if needed
    if pca:
        # concatenate dome and elea data
        mixed_data = {**dome_data, **elea_data}
        mixed_data, audio_pca_transformer, visual_pca_transformer = apply_PCA_to_features(mixed_data)
        # split the data back to dome and elea
        dome_data = {key: value for key, value in mixed_data.items() if 'IS' in key}
        elea_data = {key: value for key, value in mixed_data.items() if 'g' in key}
    # normalize features if needed
    if normalization:
        dome_data, _, __ = normalize_features(dome_data)
        elea_data, audio_normalizer, visual_normalizer = normalize_features(elea_data)
    # transform features to interpersonal
    dome_data = transform_features_to_interpersonal(dome_data)
    elea_data = transform_features_to_interpersonal(elea_data)
    # train specified model on full data ans save it afterwards
    ml_model = ml_alrogithms['ELM']
    model_hyperparams = hyperparams['ELM']
    model_most, model_least = train_classifier(ml_model, model_hyperparams, elea_data, dome_data)
    # save models, normalizers and pca transformers using pickle. Path is output_path
    # names - model_most_ELM, model_least_ELM, audio_normalizer, visual_normalizer, audio_pca_transformer, visual_pca_transformer
    import pickle
    with open(os.path.join(output_path, 'model_most_ELM.pkl'), 'wb') as f:
        pickle.dump(model_most, f)
    with open(os.path.join(output_path, 'model_least_ELM.pkl'), 'wb') as f:
        pickle.dump(model_least, f)
    if normalization:
        with open(os.path.join(output_path, 'audio_normalizer.pkl'), 'wb') as f:
            pickle.dump(audio_normalizer, f)
        with open(os.path.join(output_path, 'visual_normalizer.pkl'), 'wb') as f:
            pickle.dump(visual_normalizer, f)
    if pca:
        with open(os.path.join(output_path, 'audio_pca_transformer.pkl'), 'wb') as f:
            pickle.dump(audio_pca_transformer, f)
        with open(os.path.join(output_path, 'visual_pca_transformer.pkl'), 'wb') as f:
            pickle.dump(visual_pca_transformer, f)







if __name__ == "__main__":
    main(statistics='mean,std,max,min',
            normalization=True,
            pca=True,
            audio_features='AudioSpectrogram*8.0',
            output_path='/work/home/dsu/Datasets/dev_models')