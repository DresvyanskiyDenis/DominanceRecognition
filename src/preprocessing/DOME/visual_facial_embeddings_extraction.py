import os
from functools import partial

import numpy as np
import pandas as pd
import torch
from PIL import Image
from deepface import DeepFace
from tqdm import tqdm



def prepare_embeddings_extractor():
    # preprocessing functions for Engagement EfficientNet model
    class deepface_based_embeddings_extractor:
        def __init__(self, model_name:str):
            # super method
            super().__init__()
            self.model_name = model_name
        def extract_embeddings(self, img:np.ndarray)->np.ndarray:
            # TODO: check it
            embeddings =  DeepFace.represent(img, model_name=self.model_name, enforce_detection=False)
            embeddings = embeddings[0]['embedding']
            embeddings = np.array(embeddings)
            return embeddings
    extractor = deepface_based_embeddings_extractor(model_name='Facenet512')
    return extractor




def extract_affective_embeddings_all_videos(path_to_data:str, path_to_metafile:str, output_path:str)->pd.DataFrame:
    """ Extracts affective embeddings using DeepFace lib and FaceNet512 model for all provided frames.

    :param path_to_data: str
        Path to the folder with frames (general).
    :param path_to_metafile: str
        Path to the metafile with information about frames. The loaded dataframe will have the following columns:
        ['video_name', 'frame_number', 'timestep', 'filename', 'found_face']
    :param output_path: str
        Path for the result dataframe to be saved. Should end with .csv
    :return: pd.DataFrame
    """
    if not os.path.exists(output_path.split('.')[0]):
        os.makedirs(output_path.split('.')[0])
    # read metafile
    metafile = pd.read_csv(path_to_metafile)
    num_embeddings = 512
    # create result_file dataframe
    columns = ['video_name', 'participant_id', 'frame_number', 'timestep', 'filename', 'found_face'] + \
              [f'facial_embedding_{i}' for i in range(num_embeddings)]
    result_file = pd.DataFrame(columns=columns)
    # create file by writing to csv
    result_file.to_csv(output_path, index=False)
    # load embeddings extractor
    extractor = prepare_embeddings_extractor()
    # go over rows in the metafile and extract embeddings for every row (except for non-recognized faces)
    for idx, row in tqdm(metafile.iterrows(), total=len(metafile)):
        if row['found_face'] == False:
            extracted_embeddings = {f'facial_embedding_{i}': np.NaN for i in range(num_embeddings)}
        else:
            # generate participant id and full path to the frame to read it
            participant_id = row['participant_id']
            filename_full_path = os.path.join(path_to_data, participant_id, row['filename'])
            # load image
            img = np.array(Image.open(filename_full_path))
            # extract features
            extracted_embeddings = extractor.extract_embeddings(img)
            extracted_embeddings = extracted_embeddings.squeeze()
            extracted_embeddings = {f'facial_embedding_{i}': extracted_embeddings[i] for i in range(num_embeddings)}
        # add row to the result dataframe
        new_row = {'video_name': row['video_name'], 'participant_id': row['participant_id'],
                   'frame_number': row['frame_number'], 'timestep': row['timestep'],
                      'filename': row['filename'], 'found_face': row['found_face'],
                      **extracted_embeddings}
        result_file = pd.concat([result_file, pd.DataFrame(new_row, index=[0])], ignore_index=True)
        # dump result dataframe every 5000 rows
        if idx % 5000 == 0:
            result_file.reset_index(inplace=True, drop=True)
            result_file.to_csv(output_path, index=False, mode='a', header=False)
            result_file = pd.DataFrame(columns=columns)
    # save result dataframe
    result_file.reset_index(inplace=True, drop=True)
    result_file.to_csv(output_path, index=False, mode='a', header=False)






def main():
    path_to_data = "/work/home/dsu/Datasets/DOME/extracted_faces/"
    path_to_metafile = "/work/home/dsu/Datasets/DOME/extracted_faces/metadata_all.csv"
    output_path = "/work/home/dsu/Datasets/DOME/extracted_features/Facial_embeddings_all.csv"
    extract_affective_embeddings_all_videos(path_to_data, path_to_metafile, output_path)

if __name__ == '__main__':
    main()