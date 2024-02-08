import os
from functools import partial
import sys
sys.path.append("/work/home/dsu/PhD/scripts/DominanceRecognition/")
sys.path.append("/work/home/dsu/PhD/scripts/datatools/")
sys.path.append("/work/home/dsu/PhD/scripts/simple-HRNet-master/")

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from torchvision.io import read_image
import torchvision.transforms as T

from feature_extraction.pytorch_based.embeddings_extraction_torch import EmbeddingsExtractor
from pytorch_utils.models.CNN_models import Modified_EfficientNet_B1
from pytorch_utils.models.input_preprocessing import resize_image_saving_aspect_ratio, EfficientNet_image_preprocessor

from pytorch_utils.data_preprocessing import convert_image_to_float_and_scale
from pytorch_utils.models.Pose_estimation.HRNet import Modified_HRNet
from deepface import DeepFace


class deepface_based_embeddings_extractor:
    def __init__(self, model_name: str):
        # super method
        super().__init__()
        self.model_name = model_name

    def extract_embeddings(self, img: np.ndarray) -> np.ndarray:
        embeddings = DeepFace.represent(img, model_name=self.model_name, enforce_detection=False)
        embeddings = embeddings[0]['embedding']
        embeddings = np.array(embeddings)
        return embeddings




def prepare_embeddings_extractor(extractor_type:str):
    if extractor_type not in ['engagement', 'affective', 'kinesics', 'facial']:
        raise ValueError(f"Unknown extractor type: {extractor_type}")

    preprocessing_types = {
        'engagement': [partial(resize_image_saving_aspect_ratio, expected_size=240),
                                   EfficientNet_image_preprocessor()],
        'affective': [partial(resize_image_saving_aspect_ratio, expected_size=240),
                               EfficientNet_image_preprocessor()],
        'kinesics': [partial(resize_image_saving_aspect_ratio, expected_size=256),
                                   convert_image_to_float_and_scale,
                                   T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                   ],  # From HRNet
        'facial': None,
    }

    weight_paths = {
        'engagement': "/work/home/dsu/PhD/Model_weights/deep-capybara-42.pth",
        'affective': "/work/home/dsu/PhD/Model_weights/radiant_fog_160.pth",
        'kinesics': "/work/home/dsu/PhD/Model_weights/Engagement/kinesics_base_model.pth",
        'facial': None,
    }

    # define preprocessing functions
    preprocessing_functions = preprocessing_types[extractor_type]
    # prepare embeddings extractor
    weights_path = weight_paths[extractor_type]
    backbone_model = None
    if extractor_type == 'engagement':
        backbone_model = Modified_EfficientNet_B1(embeddings_layer_neurons=256, num_classes=3, num_regression_neurons=None)
    elif extractor_type == 'affective':
        backbone_model = Modified_EfficientNet_B1(embeddings_layer_neurons=256, num_classes=8, num_regression_neurons=2)
    elif extractor_type == 'kinesics':
        backbone_model = Modified_HRNet(pretrained=True,
                               path_to_weights="/work/home/dsu/PhD/scripts/simple-HRNet-master/pose_hrnet_w32_256x192.pth",
                               embeddings_layer_neurons=256, num_classes=3,
                               num_regression_neurons=None,
                               consider_only_upper_body=True)
    elif extractor_type == 'facial':
        extractor = deepface_based_embeddings_extractor(model_name='Facenet512')
        return extractor

    backbone_model.load_state_dict(torch.load(weights_path))
    # cut off last layers
    if extractor_type in ['engagement', 'affective']:
        backbone_model = torch.nn.Sequential(*list(backbone_model.children())[:-2])
    elif extractor_type == 'kinesics':
        backbone_model.classifier = torch.nn.Identity()
    # freeze model
    for param in backbone_model.parameters():
        param.requires_grad = False
    backbone_model.eval()
    # create embeddings extractor class
    extractor = EmbeddingsExtractor(model=backbone_model, device=None,
                                    preprocessing_functions=preprocessing_functions, output_shape=256)
    return extractor



def extract_kinesics_embeddings_all_videos(path_to_data:str, path_to_metafile:str, output_path:str,
                                           extractor_type:str)->pd.DataFrame:
    """ Extracts embeddings (of provided type) using pre-trained model (see prepare_embeddings_extractor)
    for all provided frames.

    :param path_to_data: str
        Path to the folder with frames (general).
    :param path_to_metafile: str
        Path to the metafile with information about frames. The loaded dataframe will have the following columns:
        ['video_name', 'frame_number', 'timestep', 'filename', found_column]
    :param output_path: str
        Path for the result dataframe to be saved. Should end with .csv
    :param extractor_type: str
        Type of the extractor. Should be one of ['engagement', 'affective', 'kinesics', 'facial']
    :return: pd.DataFrame
    """
    num_embeddings = 512 if extractor_type in ['facial'] else 256
    found_column = 'found_face' if extractor_type in ['facial', 'engagement', 'affective'] else 'found_pose'
    if not os.path.exists(output_path.split('.')[0]):
        os.makedirs(output_path.split('.')[0])
    # read metafile
    metafile = pd.read_csv(path_to_metafile)
    # create result_file dataframe
    columns = ['video_name', 'participant_id', 'frame_number', 'timestep', 'filename', found_column] +\
              [f'{extractor_type}_embedding_{i}' for i in range(num_embeddings)]
    result_file = pd.DataFrame(columns=columns)
    # create file by writing to csv
    result_file.to_csv(output_path, index=False)
    # load embeddings extractor
    extractor = prepare_embeddings_extractor(extractor_type=extractor_type)
    # go over rows in the metafile and extract embeddings for every row (except for non-recognized faces)
    for idx, row in tqdm(metafile.iterrows(), total=len(metafile)):
        if row[found_column] == False:
            extracted_embeddings = {f'{extractor_type}_embedding_{i}': np.NaN for i in range(num_embeddings)}
        else:
            # generate participant id and full path to the frame to read it
            participant_id = row['participant_id']
            filename_full_path = os.path.join(path_to_data, participant_id, row['filename'])
            # load image
            img = np.array(Image.open(filename_full_path)) if extractor_type=='facial' else read_image(filename_full_path)
            # extract features
            extracted_embeddings = extractor.extract_embeddings(img)
            extracted_embeddings = extracted_embeddings.squeeze()
            extracted_embeddings = {f'{extractor_type}_embedding_{i}': extracted_embeddings[i] for i in range(num_embeddings)}
        # add row to the result dataframe
        new_row = {'video_name': row['video_name'], 'participant_id': row['participant_id'],
                   'frame_number': row['frame_number'], 'timestep': row['timestep'],
                      'filename': row['filename'], found_column: row[found_column],
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
    path_to_data_faces = "/work/home/dsu/Datasets/DOME/extracted_faces/"
    path_to_data_poses = "/work/home/dsu/Datasets/DOME/extracted_poses/"
    path_to_metafile_faces = "/work/home/dsu/Datasets/DOME/extracted_faces/metadata_all.csv"
    path_to_metafile_poses = "/work/home/dsu/Datasets/DOME/extracted_poses/metadata_all.csv"
    extractor_types = ['kinesics', 'facial', 'engagement', 'affective', ]
    for extractor_type in extractor_types:
        output_path = f"/work/home/dsu/Datasets/DOME/extracted_features/{extractor_type}_embeddings_all.csv"
        path_to_metafile = path_to_metafile_faces if extractor_type in ['facial', 'engagement', 'affective'] else path_to_metafile_poses
        path_to_data = path_to_data_faces if extractor_type in ['facial', 'engagement', 'affective'] else path_to_data_poses
        extract_kinesics_embeddings_all_videos(path_to_data, path_to_metafile, output_path, extractor_type)

if __name__ == "__main__":
    main()