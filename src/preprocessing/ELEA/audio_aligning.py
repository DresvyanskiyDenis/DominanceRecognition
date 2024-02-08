import glob
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import wave
import os

from scipy.io import wavfile
import scipy.signal as sps
from tqdm import tqdm


def resample_audio(audio:np.ndarray, old_frame_rate:int, new_frame_rate:int)->np.ndarray:
    if old_frame_rate == new_frame_rate:
        return audio
    # Resample data
    number_of_samples = round(len(audio) * float(new_frame_rate) / old_frame_rate)
    resampled = sps.resample(audio, number_of_samples)
    # convert it to int16
    resampled = resampled.astype(np.int16)
    return resampled

def get_audio_segmentations(path_to_segmentation_files:str)->Dict[str, Dict[str, List[Tuple[int,int]]]]:
    """ Creates a Dictionary out of segmentation files provided by ELEA authors.
    The created dictionary has the following structure: {audio_name: {participant_id: [start, end]}}

    :param path_to_audipath_to_segmentation_filesos: str
        Path to the folder with the segmentation files. The are in the csv format.
    :return: Dict[str, Dict[str, List[Tuple[int,int]]]]
        Dictionary with the following structure: {audio_name: {participant_id: [start, end]}}
    """
    # generate all paths to the segmentation files
    segmentation_files = glob.glob(os.path.join(path_to_segmentation_files, '*.csv'))
    # create result dictionary
    result = {}
    # go over all the files
    for file in segmentation_files:
        # extract the audio name
        audio_name = os.path.basename(file)
        group_number = audio_name.split('.')[0].split('_')[-1]
        # get rid of G letter in group number
        group_number = int(group_number[1:])
        # infer the header position by checking which line has the word 'start' and 'end'
        with open(file, 'r') as f:
            lines = f.readlines()
            header = 0
            for index, line in enumerate(lines):
                if 'start' in line and 'end' in line:
                    header = index
                    break
        # read the file
        df = pd.read_csv(file, header=header)
        # get rid of \' signs in the column names
        df.columns = [col.replace('\'', '') for col in df.columns]
        # create dictionary for the audio
        result[f'g{group_number}'] = {}
        # go over all the rows
        for index, row in df.iterrows():
            # extract the participant ID
            participant_id = row['person']
            # transform it by adding the group so that the pattern will be g{group_number}_{participant_id}
            participant_id = f'g{group_number}_{participant_id}'
            # extract the start and end
            start = row['start']
            end = row['end']
            # save to the dictionary
            if participant_id not in result[f'g{group_number}']:
                result[f'g{group_number}'][participant_id] = []
            result[f'g{group_number}'][participant_id].append((start, end))
    return result





def generate_audio_with_only_one_participant(path_to_source_audio:str, segmentations:List[Tuple[int, int]])->np.ndarray:
    """ Reads source_audio, resamples it to 16KHz, then generates silent audio of duration equalled to source audio,
    and adds the speech of the participant to the silent audio. The speech is extracted using the segmentations list.
    segmentations are given in seconds.

    :param path_to_source_audio: str
        Path to the source audio file, from which the participant speech will be extracted.
    :param segmentations: List[Tuple[int, int]]
        List of tuples, where each tuple is a start and end of the speech segment in seconds.
    :return: np.ndarray
        The audio with the speech of the participant only. 16 Khz.
    """
    # read the source audio using scipy.io
    sr, data = wavfile.read(path_to_source_audio)
    # resample to 16KHz
    data = resample_audio(data, sr, 16000)
    # create silent audio
    silent = np.zeros_like(data)
    # go over all the segmentations
    for start, end in segmentations:
        # transform to samples
        start = int(start * 16000)
        end = int(end * 16000)
        # add the speech to the silent audio
        silent[start:end] = data[start:end]
    return silent


def get_audio_delays(path_to_file:str)->Dict[str, int]:
    """ Reads the file provided by the authors of ELEA and extracts the audio delays (positive or negative) for every
    participant. The first column is the group number, the second column - the delay in milliseconds.
    If it is negative, it means that audio starts before the video, so that we need to cut the audio to align it with the video.
    If it is positive, it means that audio starts after the video, so that we need to add silence to the beginning of the audio.

    :param path_to_file: str
        Path to the file with the audio delays.
    :return: Dict[str, int]
        Dictionary with the following structure: {group_number: delay}
    """
    # read the file
    df = pd.read_csv(path_to_file, header=None, sep=' ')
    # transform the group number to string
    df[0] = 'g'+df[0].astype(str)
    # create dictionary
    result = {}
    # go over all the rows
    for index, row in df.iterrows():
        group_number = row[0]
        delay = row[1]/1000. # transform to seconds
        result[group_number] = delay
    # for group 19, 13 there is no annotation. I suppose that the delay is 0
    result['g19'] = 0
    result['g13'] = 0
    return result


def remove_delay_from_audio(audio:np.ndarray, delay:float)->np.ndarray:
    """ Removes the delay from the audio. If the delay is negative, it means that audio starts before the video,
    so that we need to cut the audio to align it with the video. If the delay is positive, it means that audio starts
    after the video, so that we need to add silence to the beginning of the audio.

    :param audio: np.ndarray
        The audio to remove the delay from.
    :param delay: float
        The delay in seconds.
    :return: np.ndarray
        The audio with the delay removed.
    """
    if delay < 0:
        delay = -delay
        return audio[int(delay*16000):]
    else:
        delay = int(delay*16000)
        result = np.concatenate((np.zeros(delay), audio)).astype(np.int16)
        return result



def align_audio_with_video(path_to_audio:str, path_to_delays:str, path_to_segmentations:str)->Dict[str, np.ndarray]:
    """ General function that aligns the audio with the video. The following steps are done:
            0. Get audio delays and segmentations onto participans for each audio
            1. Read the audio
            2. Separate the audio into participants
            3. Remove the delays from every participant audio (by adding silence or cutting the audio)
            4. Return Dict with the following structure: {participant_id: audio}

    :param path_to_audio: str
        Source audio with all four/three participants in it.
    :return: Dict[str, np.ndarray]
        Dictionary with the following structure: {participant_id: audio}
    """
    # get audio delays
    delays = get_audio_delays(path_to_delays)
    # get audio segmentations
    segmentations = get_audio_segmentations(path_to_segmentations)
    # read the audio
    sr, data = wavfile.read(path_to_audio)
    # extract groupname from the audio
    group_number = os.path.basename(path_to_audio).split('.')[0].split('_')[-1][5:]
    # extract segmentations for given audio
    current_audio_segmentations = segmentations[f'g{group_number}']
    # create result dictionary
    result = {}
    # go over all the segmentations
    for participant_id, segmentations in current_audio_segmentations.items():
        # extract the audio for the participant
        participant_audio = generate_audio_with_only_one_participant(path_to_audio, segmentations)
        # remove the delay
        participant_audio = remove_delay_from_audio(participant_audio, delays[participant_id.split('_')[0]])
        # save to the result dictionary
        result[participant_id] = participant_audio

    return result



def preprocess_all_audios(path_to_audios:str, path_to_delays:str, path_to_segmentations:str, output_path:str)->None:
    """ Preprocesses all audios in the given folder (for ELEA). For every audio, the following steps will be done:
            1. Get audio delays and segmentations onto participans for each audio
            2. Read the audio
            3. Separate the audio into participants
            4. Remove the delays from every participant audio (by adding silence or cutting the audio)
            5. Save the audio to the output folder.

    :param path_to_audios: str
        Path to the folder with all the audios.
    :param path_to_delays: str
        Path to the file with the audio delays. (see get_audio_delays function)
    :param path_to_segmentations: str
        Path to the folder with the segmentation files. (see get_audio_segmentations function)
    :param output_path: str
        Path to the output folder.
    :return: None
    """
    # check if the output folder exists
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    # get all the audio files
    audio_files = glob.glob(os.path.join(path_to_audios, '*.wav'))
    # take only groups starting from 12. The template of the files is group{number}.wav
    audio_files = [file for file in audio_files if int(os.path.basename(file).split('.')[0][5:]) >= 12]
    # go over all the audio files
    for audio_file in tqdm(audio_files, desc="Preprocessing audios..."):
        # align the audio with the video
        result = align_audio_with_video(audio_file, path_to_delays, path_to_segmentations)
        # save the result to the output folder
        for participant_id, audio in result.items():
            output_file = os.path.join(output_path, f"{participant_id}_{os.path.basename(audio_file)}")
            wavfile.write(output_file, 16000, audio)




def main():
    path_to_audios = "/work/home/dsu/Datasets/ELEA/elea/audio/Groups1-40_wav/"
    path_to_delays = "/work/home/dsu/Datasets/ELEA/elea/video/audiodelayMS.txt"
    path_to_segmentations = "/work/home/dsu/Datasets/ELEA/elea/audio/SpeakingSegmentation/"
    output_path = "/work/home/dsu/Datasets/ELEA/preprocessed/aligned_audio/"
    preprocess_all_audios(path_to_audios, path_to_delays, path_to_segmentations, output_path)


if __name__ == '__main__':
    main()