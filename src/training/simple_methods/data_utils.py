import glob
from typing import List, Dict

import pandas as pd
import numpy as np



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
        result_dataframe = pd.merge(result_dataframe, df, on=['participant_id', 'start_timestep', 'end_timestep'])
        # drop the duplicated columns with y suffix and rename x suffix to the original name
        result_dataframe.drop(columns=[col for col in result_dataframe.columns if '_y' in col], inplace=True)
        result_dataframe.rename(columns=lambda x: x.replace('_x', ''), inplace=True)
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
        result_dataframe = pd.merge(result_dataframe, df, on=['participant_id', 'start_timestep', 'end_timestep'])
        # drop the duplicated columns with y suffix and rename x suffix to the original name
        result_dataframe.drop(columns=[col for col in result_dataframe.columns if '_y' in col], inplace=True)
        result_dataframe.rename(columns=lambda x: x.replace('_x', ''), inplace=True)
    return result_dataframe


def get_participants_to_features_dict(visual_df_dome:pd.DataFrame, audio_df_dome:pd.DataFrame,
                                      visual_df_elea:pd.DataFrame, audio_df_elea:pd.DataFrame)->\
        Dict[str, Dict[str, pd.DataFrame]]:
    """ Restructures the provided dataframes in a way that it creates a Dict[{participant_id}->Dict[{feature_type}->pd.DataFrame]]
    for the DOME and ELEA datasets. This dictionary is more convenient to work with in the following steps of the pipeline.

    :param visual_df_dome: pd.DataFrame
        Dataframe with the visual features for the DOME dataset
    :param audio_df_dome: pd.DataFrame
        Dataframe with the audio features for the DOME dataset
    :param visual_df_elea: pd.DataFrame
        Dataframe with the visual features for the ELEA dataset
    :param audio_df_elea: pd.DataFrame
        Dataframe with the audio features for the ELEA dataset
    :return: Dict[str, Dict[str, pd.DataFrame]]
        Dictionary with the restructured dataframes (see description above).
    """
    participants_to_features = {}
    # get unique participant ids from both dome and elea datasets
    participant_ids = np.unique(np.concatenate([visual_df_dome['participant_id'].unique(), visual_df_elea['participant_id'].unique()]))
    for participant_id in participant_ids:
        participant_features = {}
        # get the visual features for the participant
        visual_dome = visual_df_dome[visual_df_dome['participant_id'] == participant_id]
        visual_elea = visual_df_elea[visual_df_elea['participant_id'] == participant_id]
        # combine the visual features
        participant_features['visual'] = pd.concat([visual_dome, visual_elea], axis=0)
        # get the audio features for the participant
        audio_dome = audio_df_dome[audio_df_dome['participant_id'] == participant_id]
        audio_elea = audio_df_elea[audio_df_elea['participant_id'] == participant_id]
        # combine the audio features
        participant_features['audio'] = pd.concat([audio_dome, audio_elea], axis=0)
        # add the participant to the dictionary
        participants_to_features[participant_id] = participant_features
    return participants_to_features





def main():
    elea_visual_paths = glob.glob("/work/home/dsu/Datasets/ELEA/extracted_features/*.csv")
    elea_audio_paths = glob.glob("/work/home/dsu/Datasets/ELEA/extracted_audio_features/*/*16*")
    dome_visual_paths = glob.glob("/work/home/dsu/Datasets/DOME/extracted_features/*.csv")
    dome_audio_paths = glob.glob("/work/home/dsu/Datasets/DOME/extracted_audio_features/*/*16*")
    # load all audio features
    audio_df_dome = load_all_audio_features_dome(dome_audio_paths)
    audio_df_elea = load_all_audio_features_elea(elea_audio_paths)
    # load all visual features
    visual_df_dome = load_all_visual_features_dome(dome_visual_paths)
    visual_df_elea = load_all_visual_features_elea(elea_visual_paths)
    # restructure the dataframes
    participants_to_features = get_participants_to_features_dict(visual_df_dome, audio_df_dome, visual_df_elea, audio_df_elea)
    print(participants_to_features.keys())
    a = 1+2.


if __name__ == "__main__":
    main()