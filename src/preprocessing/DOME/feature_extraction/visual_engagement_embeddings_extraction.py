import os
from functools import partial
import sys
sys.path.append("/work/home/dsu/PhD/scripts/DominanceRecognition/")
sys.path.append("/work/home/dsu/PhD/scripts/datatools/")

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from torchvision.io import read_image

from feature_extraction.pytorch_based.embeddings_extraction_torch import EmbeddingsExtractor
from pytorch_utils.models.CNN_models import Modified_EfficientNet_B1
from pytorch_utils.models.input_preprocessing import resize_image_saving_aspect_ratio, EfficientNet_image_preprocessor


def prepare_embeddings_extractor():
    # preprocessing functions for Engagement EfficientNet model
    preprocessing_functions = [partial(resize_image_saving_aspect_ratio, expected_size=240),
                                   EfficientNet_image_preprocessor()]
    # prepare embeddings extractor
    weights_path = "/work/home/dsu/PhD/Model_weights/deep-capybara-42.pth"
    backbone_model = Modified_EfficientNet_B1(embeddings_layer_neurons=256, num_classes=3, num_regression_neurons=None)
    backbone_model.load_state_dict(torch.load(weights_path))
    # cut off last two layers responsible for classification and regression
    backbone_model.classifier = torch.nn.Identity()
    # freeze model
    for param in backbone_model.parameters():
        param.requires_grad = False
    backbone_model.eval()
    # create embeddings extractor class
    extractor = EmbeddingsExtractor(model=backbone_model, device=None,
                                    preprocessing_functions=preprocessing_functions, output_shape=256)
    return extractor




def extract_affective_embeddings_all_videos(path_to_data:str, path_to_metafile:str, output_path:str)->pd.DataFrame:
    """ Extracts affective embeddings using Engagement Recognition EfficientNet-B1 model for all provided frames.

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
    # create result_file dataframe
    columns = ['video_name', 'participant_id', 'frame_number', 'timestep', 'filename', 'found_face'] + \
              [f'eng_embedding_{i}' for i in range(256)]
    result_file = pd.DataFrame(columns=columns)
    # create file by writing to csv
    result_file.to_csv(output_path, index=False)
    # load embeddings extractor
    extractor = prepare_embeddings_extractor()
    # go over rows in the metafile and extract embeddings for every row (except for non-recognized faces)
    for idx, row in tqdm(metafile.iterrows(), total=len(metafile)):
        if row['found_face'] == False:
            extracted_embeddings = {f'eng_embedding_{i}': np.NaN for i in range(256)}
        else:
            # generate participant id and full path to the frame to read it
            participant_id = row['participant_id']
            filename_full_path = os.path.join(path_to_data, participant_id, row['filename'])
            # load image
            # img = np.array(Image.open(filename_full_path))
            img = read_image(filename_full_path)
            # extract features
            extracted_embeddings = extractor.extract_embeddings(img)
            extracted_embeddings = extracted_embeddings.squeeze()
            extracted_embeddings = {f'eng_embedding_{i}': extracted_embeddings[i] for i in range(256)}
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
    output_path = "/work/home/dsu/Datasets/DOME/extracted_features/Engagement_embeddings_all.csv"
    extract_affective_embeddings_all_videos(path_to_data, path_to_metafile, output_path)

if __name__ == '__main__':
    main()