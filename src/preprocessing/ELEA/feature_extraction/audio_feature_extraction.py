import glob
import os
import sys
sys.path.append("/work/home/dsu/PhD/scripts/DominanceRecognition/")
sys.path.append("/work/home/dsu/PhD/scripts/datatools/")

import numpy as np
import pandas as pd
from tqdm import tqdm

from feature_extraction.pytorch_based.embeddings_extraction_audio_torch import AudioEmbeddingsExtractor



def initialize_audio_feature_extractor(extraction_model:str)-> object:
    """ Initializes the audio feature extractor.

    :param extraction_model: str
        The type of the extractor. Can be 'wav2vec', 'HuBERT', or 'AudioSpectrogramTransformer'.
    :return: object
        The initialized extractor. This is the class from datatools.feature_extraction.embeddings_extraction_audio_torch file.
    """
    extractor = AudioEmbeddingsExtractor(extractor_type= extraction_model, frame_rate=16000)
    return extractor


def extract_embeddings_audio_file(extractor:AudioEmbeddingsExtractor, audio_file:str, chunk_size:float)->pd.DataFrame:
    """ Extracts embeddings from the provided audio file. Before extraction, the audio file will be resampled to
    16000 Hz, converted to mono, and cut into chunks of the specified chunk_size (it is all done inside the
    extract_features_audio method of the AudioEmbeddingsExtractor class).

    :param extractor: AudioEmbeddingsExtractor
        The extractor class from datatools.feature_extraction.embeddings_extraction_audio_torch file.
    :param audio_file: str
        THe path to the audio file.
    :param chunk_size: float
        The size of the chunk in seconds.
    :return: pd.DataFrame
        The metadata containing the extracted embeddings and the corresponding timestamps.
        THe columns are: ['filename', 'participant_id', 'start_timestep', 'end_timestep', 'features'] +
        ['embedding_{}'.format(i) for i in range(embeddings_size)]
    """
    # create metadata
    embeddings_size = extractor.get_embeddings_size()
    metadata = pd.DataFrame(columns=['filename', 'participant_id', 'start_timestep', 'end_timestep', 'features'] +
                                    ['embedding_{}'.format(i) for i in range(embeddings_size)])
    # extract features
    features = extractor.extract_features_audio(audio_file, chunk_size=chunk_size)  # list of np.arrays
    features = np.array(features)
    # save features
    for idx, features_chunk in enumerate(features):
        # infer from the filename the group and participant if, forming the full participant id as g{group_num}_{participant_id}
        # the files have the following pattern: g{group_num}_{participant_id}_group{group_num}.wav
        participant_id = '_'.join(os.path.basename(audio_file).split('_')[0:2])
        new_row = pd.DataFrame.from_dict({'filename': [os.path.basename(audio_file)],
                                          'participant_id': [participant_id],
                                          'start_timestep': [idx*chunk_size],
                                          'end_timestep': [(idx+1)*chunk_size],
                                          **{'embedding_{}'.format(i): [features_chunk[i]] for i in range(embeddings_size)}
                                          })
        metadata = pd.concat([metadata, new_row], ignore_index=True)
    return metadata


def extract_features(path_to_dataset:str, extractor_type:str, chunk_size:float, output_path:str)->None:
    """ Extracts features from the ELEA dataset using the provided extractor type and chunk size. The features
    for each participant will be saved in a separate file in the output_path directory.
    THe pattern is as follows: output_path/*participant_id*/*participant_id*_features_*extractor_type*_*chunk_size*.csv

    :param path_to_dataset: str
        The path to the audiofiles (aligned, preprocessed). wav files
    :param extractor_type: str
        The type of the extractor. Can be 'wav2vec', 'HuBERT', or 'AudioSpectrogramTransformer'.
    :param chunk_size: float
        The size of the chunks the audio will be cut into in seconds.
    :param output_path:str
        The path to the output directory.
    :return: None
    """
    # create output directory
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # initialize extractor
    extractor = initialize_audio_feature_extractor(extractor_type)
    # get all filenames
    audio_filenames = glob.glob(os.path.join(path_to_dataset, '*.wav'))
    # extract features
    for audio_file in tqdm(audio_filenames,
                           desc=f'Extracting features with {extractor_type} and chunk size {chunk_size}...'):
        # we add 1 to the participant_id because the id of participants starts from 0 instead of 1 as for videos
        participant_id = '_'.join(os.path.basename(audio_file).split('_')[0:2])
        embeddings = extract_embeddings_audio_file(extractor, audio_file, chunk_size)
        # create output directory for the participant
        participant_output_path = output_path + '/' + participant_id
        if not os.path.exists(participant_output_path):
            os.makedirs(participant_output_path)
        # save features
        embeddings.to_csv(participant_output_path + f'/{participant_id}_features_{extractor_type}_{chunk_size}.csv',
                          index=False)
    return None


def main():
    path_to_dataset = "/work/home/dsu/Datasets/ELEA/preprocessed/aligned_audio/"
    extractor_types = ['wav2vec', 'HuBERT', 'AudioSpectrogramTransformer']
    chunk_sizes = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 60.0]
    output_path = "/work/home/dsu/Datasets/ELEA/extracted_audio_features/"
    for extractor_type in extractor_types:
        for chunk_size in chunk_sizes:
            extract_features(path_to_dataset, extractor_type, chunk_size, output_path)


if __name__ == "__main__":
    main()