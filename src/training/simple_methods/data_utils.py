import glob
from typing import List, Dict, Union

import pandas as pd
import numpy as np

from src.preprocessing.DOME.labels_processing.labels_processing import get_most_least_dominant_participants_DOME
from src.preprocessing.ELEA.labels_processing.labels_preprocessing import get_most_least_dominant_participants_ELEA


def load_all_visual_features_dome(paths:List[str])->pd.DataFrame:
    """ Loads all visual features for the DOME dataset from specified paths and merges them into a single dataframe
    along the columns and taking into account the columns  'participant_id', 'frame_number', 'timestep'

    :param paths: List[str]
        List of paths to the files
    :return: pd.DataFrame
        Dataframe with the features
    """
    dataframes = [pd.read_csv(path) for path in paths]
    result_dataframe = dataframes[0]
    for df in dataframes[1:]:
        result_dataframe = pd.merge(result_dataframe, df, on=['participant_id', 'frame_number', 'timestep'])
        # drop the duplicated columns with y suffix and rename x suffix to the original name
        result_dataframe.drop(columns=[col for col in result_dataframe.columns if '_y' in col], inplace=True)
        result_dataframe.rename(columns=lambda x: x.replace('_x', ''), inplace=True)
    # drop 'found_pose' column
    result_dataframe.drop(columns=['found_pose'], inplace=True)
    return result_dataframe


def load_all_audio_features_dome(paths:List[str])->pd.DataFrame:
    """ Loads all audio features for the DOME dataset from specified paths and merges them into a single dataframe
    along the columns and taking into account the columns 'participant_id', 'start_timestep', 'end_timestep'

    :param paths: List[str]
        List of paths to the files
    :return: pd.DataFrame
        Dataframe with the features
    """ # not merge, but concatenate
    dataframes = [pd.read_csv(path) for path in paths]
    result_dataframe = dataframes[0]
    for df in dataframes[1:]:
        result_dataframe = pd.concat([result_dataframe, df], axis=0, ignore_index=True)
    result_dataframe.drop(columns=['features'], inplace=True)
    result_dataframe.reset_index(drop=True, inplace=True)
    return result_dataframe


def load_all_visual_features_elea(paths:List[str])->pd.DataFrame:
    """ Loads all visual features for the ELEA dataset from specified paths and merges them into a single dataframe
    along the columns and taking into account the columns  'participant_id', 'frame_number', 'timestep'

    :param paths: List[str]
        List of paths to the files
    :return: pd.DataFrame
        Dataframe with the features
    """
    dataframes = [pd.read_csv(path) for path in paths]
    result_dataframe = dataframes[0]
    for df in dataframes[1:]:
        result_dataframe = pd.merge(result_dataframe, df, on=['participant_id', 'frame_number', 'timestep'])
        # drop the duplicated columns with y suffix and rename x suffix to the original name
        result_dataframe.drop(columns=[col for col in result_dataframe.columns if '_y' in col], inplace=True)
        result_dataframe.rename(columns=lambda x: x.replace('_x', ''), inplace=True)
    # drop 'found_pose' column
    result_dataframe.drop(columns=['found_pose'], inplace=True)
    return result_dataframe

def load_all_audio_features_elea(paths:List[str])->pd.DataFrame:
    """ Loads all audio features for the ELEA dataset from specified paths and merges them into a single dataframe
    along the columns and taking into account the columns 'participant_id', 'start_timestep', 'end_timestep'

    :param paths: List[str]
        List of paths to the files
    :return: pd.DataFrame
        Dataframe with the features
    """
    dataframes = [pd.read_csv(path) for path in paths]
    result_dataframe = dataframes[0]
    for df in dataframes[1:]:
        result_dataframe = pd.concat([result_dataframe, df], axis=0, ignore_index=True)
    result_dataframe.drop(columns=['features'], inplace=True)
    result_dataframe.reset_index(drop=True, inplace=True)
    return result_dataframe


def get_participants_to_features_dict(visual_features:pd.DataFrame, audio_features:pd.DataFrame)->\
        Dict[str, Dict[str, pd.DataFrame]]:
    """ Restructures the provided feature dataframes in a way that it creates a Dict[{participant_id}->Dict[{feature_type}->pd.DataFrame]]
    for either DOME or ELEA datasets. This dictionary is more convenient to work with in the following steps of the pipeline.

    :param visual_features: pd.DataFrame
        Dataframe with visual features
    :param audio_features: pd.DataFrame
        Dataframe with audio features
    :return: Dict[str, Dict[str, pd.DataFrame]]
        Dictionary with the restructured dataframes (see description above).
    """
    participants_to_features = {}
    # get unique participant ids from both dome and elea datasets
    participant_ids = np.unique(np.concatenate([visual_features['participant_id'].unique(), audio_features['participant_id'].unique()]))
    for participant_id in participant_ids:
        participant_features = {}
        # get the visual features for the participant
        visual = visual_features[visual_features['participant_id'] == participant_id]
        # get the audio features for the participant
        audio = audio_features[audio_features['participant_id'] == participant_id]
        # add the features to the dictionary
        participant_features['visual_features'] = visual
        participant_features['audio_features'] = audio
        # add the participant to the dictionary
        participants_to_features[participant_id] = participant_features
    return participants_to_features

def unite_labels_and_features_elea(participants_to_features:Dict[str, Dict[str, pd.DataFrame]], labels:pd.DataFrame)->\
        Dict[str, Dict[str, Union[pd.DataFrame, int]]]:
    """ Unites the labels and the features for ELEA dataset into a single dictionary with the following structure:
    Dict[{participant_id}->Dict[{feature_type}->pd.DataFrame, 'label_most_dominant'->int, 'label_least_dominant'->int]]

    :param participants_to_features: Dict[str, Dict[str, pd.DataFrame]]
        Dictionary with the restructured dataframes (see get_participants_to_features_dict function).
    :param labels: pd.DataFrame
        Dataframe with the labels. The columns are ['group', 'most_dominant', 'least_dominant']
    :return: Dict[str, Dict[str, Union[pd.DataFrame, int]]]
        Dictionary with the keys being the participant ids and the values being the dictionaries:
        {'visual_features'->pd.DataFrame, 'audio_features'->pd.DataFrame, 'label_most_dominant'->int, 'label_least_dominant'->int}
        1 means True, 0 means False (for most and least dominant labels)
    """
    result = {}
    for participant_id, features in participants_to_features.items():
        # extract group number and participant letter from the participant id
        group = int(participant_id.split('_')[0][1:])
        p_letter = participant_id.split('_')[-1]
        # get the labels for the participant
        most_dominant_in_group = labels[labels['group'] == group]['most_dominant'].values[0]
        least_dominant_in_group = labels[labels['group'] == group]['least_dominant'].values[0]
        # assign labels to the participant (e.g. if the participants letter is the same as the most dominant in the group, then
        # the participant is most dominant, and vise versa)
        label_most_dominant = 1 if p_letter == most_dominant_in_group else 0
        label_least_dominant = 1 if p_letter == least_dominant_in_group else 0
        # create new dictionary with the features and the labels
        participant_dict = {'visual_features': features['visual_features'], 'audio_features': features['audio_features'],
                            'label_most_dominant': label_most_dominant, 'label_least_dominant': label_least_dominant}
        # add to the result dictionary
        result[participant_id] = participant_dict
    return result

def unite_labels_and_features_dome(participants_to_features:Dict[str, Dict[str, pd.DataFrame]], labels:pd.DataFrame)->\
        Dict[str, Dict[str, Union[pd.DataFrame, int]]]:
    """ Unites the labels and the features for DOME dataset into a single dictionary with the following structure:
    Dict[{participant_id}->Dict[{feature_type}->pd.DataFrame, 'label_most_dominant'->int, 'label_least_dominant'->int]]
    Important to note - DOME is annotated not only in terms of sessions, but also in terms of time. This means that
    the same participant can have different labels for different time periods. Therefore, we need to separate the features
    using the time periods from the labels.

    :param participants_to_features: Dict[str, Dict[str, pd.DataFrame]]
        Dictionary with the restructured dataframes (see get_participants_to_features_dict function).
    :param labels: pd.DataFrame
        Dataframe with the labels. The columns are ['session_id', 'start_sec', 'end_sec', 'most_dominant', 'least_dominant']
    :return: Dict[str, Dict[str, Union[pd.DataFrame, int]]]
        Dictionary with the keys being the participant ids and the values being the dictionaries:
        {'visual_features'->pd.DataFrame, 'audio_features'->pd.DataFrame, 'label_most_dominant'->int, 'label_least_dominant'->int}
        1 means True, 0 means False (for most and least dominant labels)
    """
    result = {}
    # go over participants_to_features dictionary
    for participant_id, features in participants_to_features.items():
        # extract session_id and the participant number from the participant id
        session_id = participant_id.split('_')[0]
        participant_number = int(participant_id.split('_')[-1])
        # get the labels for the participant
        session_labels = labels[labels['session_id'] == session_id]
        # go over session labels as these are dataframe with several rows and columns are:
        # ['session_id', 'start_sec', 'end_sec', 'most_dominant', 'least_dominant']
        for idx in range(len(session_labels)):
            # get the start and end seconds
            start_sec = session_labels.iloc[idx]['start_sec']
            end_sec = session_labels.iloc[idx]['end_sec']
            # get the idx of the most dominant and least dominant participants
            most_dominant_idx = session_labels.iloc[idx]['most_dominant']
            least_dominant_idx = session_labels.iloc[idx]['least_dominant']
            # transform those to the labels for the participant. If the participant number is the same as the most dominant
            # then the participant is most dominant, and vise versa.
            label_most_dominant = 1 if participant_number == most_dominant_idx else 0
            label_least_dominant = 1 if participant_number == least_dominant_idx else 0
            # extract features according to the start and end seconds
            visual = features['visual_features'][(features['visual_features']['timestep'] >= start_sec) &
                                                (features['visual_features']['timestep'] <= end_sec)]
            audio = features['audio_features'][(features['audio_features']['start_timestep'] >= start_sec) &
                                                (features['audio_features']['end_timestep'] <= end_sec)]
            # create new participant_id that will include old participant_id and the number of "part" (idx)
            new_participant_id = f"{participant_id}_part_{idx}"
            # create new dictionary with the features and the labels
            participant_dict = {'visual_features': visual, 'audio_features': audio,
                                'label_most_dominant': label_most_dominant, 'label_least_dominant': label_least_dominant}
            # add to the result dictionary
            result[new_participant_id] = participant_dict
    return result


def load_and_prepare_DOME_and_ELEA_features_and_labels(elea_visual_paths:List[str], elea_audio_paths:List[str], elea_labels_paths:List[str],
                                                       dome_visual_paths:List[str], dome_audio_paths:List[str], dome_labels_paths:str):
    """ Main function that loads and prepares the features and labels for both DOME and ELEA datasets.

    :param elea_visual_paths: List[str]
        List of paths to the ELEA visual features
    :param elea_audio_paths: List[str]
        List of paths to the ELEA audio features
    :param elea_labels_paths: List[str]
        List of paths to the ELEA labels
    :param dome_visual_paths: List[str]
        List of paths to the DOME visual features
    :param dome_audio_paths: List[str]
        List of paths to the DOME audio features
    :param dome_labels_paths: str
        Path to the DOME labels
    :return: Dict[str, Dict[str, Union[pd.DataFrame, int]]], Dict[str, Dict[str, Union[pd.DataFrame, int]]]
        Two dictionaries with the features and labels for ELEA and DOME datasets
    """
    # load the labels
    dome_labels = get_most_least_dominant_participants_DOME(dome_labels_paths)
    elea_labels = get_most_least_dominant_participants_ELEA(elea_labels_paths)
    # load all audio features
    audio_df_dome = load_all_audio_features_dome(dome_audio_paths)
    audio_df_elea = load_all_audio_features_elea(elea_audio_paths)
    # load all visual features
    visual_df_dome = load_all_visual_features_dome(dome_visual_paths)
    visual_df_elea = load_all_visual_features_elea(elea_visual_paths)
    # restructure the dataframes
    participants_to_features_dome = get_participants_to_features_dict(visual_df_dome, audio_df_dome)
    participants_to_features_elea = get_participants_to_features_dict(visual_df_elea, audio_df_elea)
    # combine the labels with the features
    result_dome = unite_labels_and_features_dome(participants_to_features_dome, dome_labels)
    result_elea = unite_labels_and_features_elea(participants_to_features_elea, elea_labels)
    return result_dome, result_elea








def main():
    elea_visual_paths = glob.glob("/work/home/dsu/Datasets/ELEA/extracted_features/*.csv")
    elea_audio_paths = glob.glob("/work/home/dsu/Datasets/ELEA/extracted_audio_features/*/*16*")
    elea_labels_paths = glob.glob("/work/home/dsu/Datasets/ELEA/preprocessed_labels/ELEA_external_annotations/Annotator*.csv")
    dome_visual_paths = glob.glob("/work/home/dsu/Datasets/DOME/extracted_features/*.csv")
    dome_audio_paths = glob.glob("/work/home/dsu/Datasets/DOME/extracted_audio_features/*/*16*")
    dome_labels_paths = "/work/home/dsu/Datasets/DOME/Annotations/dome_annotations_M1.csv"
    # load the labels
    dome_labels, elea_labels = load_and_prepare_DOME_and_ELEA_features_and_labels(elea_visual_paths, elea_audio_paths, elea_labels_paths,
                                                       dome_visual_paths, dome_audio_paths, dome_labels_paths)




if __name__ == "__main__":
    main()