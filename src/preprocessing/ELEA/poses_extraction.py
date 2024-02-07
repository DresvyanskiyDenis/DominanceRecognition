import glob
import os
import sys

from feature_extraction.pytorch_based.pose_recognition_utils import get_pose_bbox, crop_frame_to_pose

sys.path.append("/work/home/dsu/PhD/scripts/DominanceRecognition/")
sys.path.append("/work/home/dsu/PhD/scripts/datatools/")
sys.path.append("/work/home/dsu/PhD/scripts/simple-HRNet-master/")

import torch

from SimpleHRNet import SimpleHRNet
from typing import Dict, Union, Tuple

from src.preprocessing.ELEA.participants_ids import prepare_id_splitting


import numpy as np
import pandas as pd
import cv2
from PIL import Image
from tqdm import tqdm

def initialize_pose_extractor():
    pose_detector = SimpleHRNet(c=48, nof_joints=17, multiperson=True,
                                yolo_version='v3',
                                yolo_model_def=os.path.join("/work/home/dsu/PhD/scripts/simple-HRNet-master/",
                                                            "models_/detectors/yolo/config/yolov3.cfg"),
                                yolo_class_path=os.path.join("/work/home/dsu/PhD/scripts/simple-HRNet-master/",
                                                             "models_/detectors/yolo/data/coco.names"),
                                yolo_weights_path=os.path.join("/work/home/dsu/PhD/scripts/simple-HRNet-master/",
                                                               "models_/detectors/yolo/weights/yolov3.weights"),
                                checkpoint_path="/work/home/dsu/PhD/scripts/simple-HRNet-master/pose_hrnet_w48_384x288.pth",
                                return_heatmaps=False, return_bounding_boxes=True, max_batch_size=1,
                                device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    return pose_detector


def extract_pose_from_frame(frame:np.ndarray, pose_extractor:torch.nn.Module)->Union[None, np.ndarray]:
    pose_bbox = get_pose_bbox(frame, pose_extractor)
    if pose_bbox is None:
        return None
    else:
        pose = crop_frame_to_pose(frame, pose_bbox, limits=[30, 30])
        return pose



def extract_poses_single_video(path_to_video:str, output_path:str, participants_ids:Dict[str, str], final_fps:int,
                                 pose_extractor:torch.nn.Module)->Tuple[pd.DataFrame, pd.DataFrame]:
    """ Extracts poses from a single video and saves them to the output folder. In participants_ids the left and right
    participants are defined (because every frame will be split into two images, one for each participant). For every
    participant, a metadata file is created.

    :param path_to_video: str
        Path to the video.
    :param output_path: str
        Path to the output folder. In this folder, subfolders for the left and right participants (named as they IDs) will be created.
    :param participants_ids: Dict[str, str]
        Dictionary with the IDs of the left and right participants. The pattern is {left: ID, right: ID}. Where ID has
        the pattern g{group_number}_{participant_id}
    :param final_fps: int
        The final fps of the video. The extracted data will be resampled to this fps.
    :param pose_extractor: torch.nn.Module
        The pose extractor model. The loaded HRNet model from datatools.pytorch_utils.models.Pose_estimation.HRNet.py.
    :return: Tuple[pd.DataFrame, pd.DataFrame]
        The metadata for the left and right participants. Columns are
        ['video_name', 'participant_id', 'frame_number', 'timestep', 'filename', 'found_pose']
    """
    # Create the output folder
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, participants_ids['left']), exist_ok=True)
    os.makedirs(os.path.join(output_path, participants_ids['right']), exist_ok=True)
    # create the metadata files
    metadata_l = pd.DataFrame(columns=['video_name', 'participant_id', 'frame_number', 'timestep', 'filename', 'found_pose'])
    metadata_r = pd.DataFrame(columns=['video_name', 'participant_id', 'frame_number', 'timestep', 'filename', 'found_pose'])
    # load video file
    video = cv2.VideoCapture(path_to_video)
    # get FPS
    FPS = video.get(cv2.CAP_PROP_FPS)
    FPS_in_seconds = 1. / FPS
    # calculate every which frame should be taken
    every_n_frame = int(round(FPS / final_fps))
    # go through all frames
    counter = 0
    with tqdm(total=video.get(cv2.CAP_PROP_FRAME_COUNT)) as pbar:
        pbar.set_description(f"Processing video {path_to_video.split(os.sep)[-1]}...")
        while video.isOpened():
            ret, frame = video.read()
            pbar.update(1)
            if ret:
                if counter % every_n_frame == 0:
                    # convert BGR to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # calculate timestamp
                    timestamp = counter * FPS_in_seconds
                    # round it to 2 digits to make it readable
                    timestamp = round(timestamp, 2)
                    # divide the frame into two parts
                    left_frame = frame[:, :frame.shape[1]//2]
                    right_frame = frame[:, frame.shape[1]//2:]
                    # extract the pose from the left and right frames
                    left_pose = extract_pose_from_frame(left_frame, pose_extractor)
                    right_pose = extract_pose_from_frame(right_frame, pose_extractor)
                    # pre-form rows for the metadata of l and r participants
                    row_l = {'video_name': os.path.basename(path_to_video), 'participant_id': participants_ids['left'],
                                'frame_number': counter, 'timestep': timestamp, 'filename': f"{path_to_video.split(os.sep)[-1]}_l_{timestamp}.png"}
                    row_r = {'video_name': os.path.basename(path_to_video), 'participant_id': participants_ids['right'],
                                'frame_number': counter, 'timestep': timestamp, 'filename': f"{path_to_video.split(os.sep)[-1]}_r_{timestamp}.png"}
                    # process the left pose
                    if left_pose is not None:
                        # save the face
                        Image.fromarray(left_pose).save(os.path.join(output_path, participants_ids['left'], row_l['filename']))
                        row_l['found_pose'] = True
                    else:
                        row_l['found_pose'] = False
                    # process the right pose
                    if right_pose is not None:
                        # save the face
                        Image.fromarray(right_pose).save(os.path.join(output_path, participants_ids['right'], row_r['filename']))
                        row_r['found_pose'] = True
                    else:
                        row_r['found_pose'] = False
                    # append the rows to the metadata
                    metadata_l = pd.concat([metadata_l, pd.DataFrame([row_l])], ignore_index=True)
                    metadata_r = pd.concat([metadata_r, pd.DataFrame([row_r])], ignore_index=True)
                counter += 1
            else:
                break
    return metadata_l, metadata_r


def extract_poses_all_videos(path_to_videos:str, output_path:str, final_fps:int):
    """ Extracts poses from all videos in the folder. TO do so, it firstly loads the pose extractor model and then
    the dictionary with the participants IDs. Those IDs will be used to identify the left and right participants in
    every video and save extracted frames in different folders. THere are also participants with "noParticipant" ID.
    They will be filtered out as they are not present.

    :param path_to_videos: str
        Path to the folder with videos.
    :param output_path: str
        Path to the output folder. In this folder, subfolders for every participant with respective ID will be created.
    :param final_fps: int
        The final fps of the video. The extracted data will be resampled to this fps.
    :return: None
    """
    # create the output folder
    os.makedirs(output_path, exist_ok=True)
    # load the pose extractor model
    pose_extractor = initialize_pose_extractor()
    # load the participants IDs
    participants_ids = prepare_id_splitting(path_to_videos)
    # create metadata file
    metadata_all = pd.DataFrame(columns=['video_name', 'participant_id', 'frame_number', 'timestep', 'filename', 'found_pose'])
    # get full paths to the videos
    videos = glob.glob(os.path.join(path_to_videos, "*.avi"))
    # go through all videos
    for video in tqdm(videos, desc="Processing videos..."):
        # get participant IDs
        participant_ids = participants_ids[os.path.basename(video)]
        # if there is noParticipant in the items, replace noParticipant with 'garbage' subfolder name so that
        # all such noParticipants from different videos will be saved to the same folder that will be deleted then
        # moreover, 'garbage' participant_id will be sorted out from the metadata_all
        for key, value in participant_ids.items():
            if value == 'noParticipant':
                participant_ids[key] = 'garbage'
        # extract the poses from the video
        metadata_l, metadata_r = extract_poses_single_video(video, output_path, final_fps, pose_extractor, participant_ids)












def main():
    pass

if __name__ == "__main__":
    main()