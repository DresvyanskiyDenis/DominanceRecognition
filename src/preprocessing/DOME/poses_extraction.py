import glob
import os
from typing import List, Callable, Dict, Union, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from SimpleHRNet import SimpleHRNet
from feature_extraction.pytorch_based.pose_recognition_utils import get_pose_bbox, crop_frame_to_pose

def extract_poses_all_videos(path_to_data:str, output_path:str, final_fps:int)->None:
    """ Extracts the poses from all videos in the path_to_data and saves them to the output_path.
    The videos contain two persons sitting next to each other (left and right). Their positions are approximately half-half in the
    width of the video. The function extracts the poses of both persons and saves them to the output_path.
    The output pattern is: output_path/{participant_id}/{participant_id}_{timestep}.png

    :param path_to_data: str
        The path to the folder with videos to extract the poses from. The videos will be found in the subfolders
        with the following pattern: path_to_data/*/video/*L.* and path_to_data/*/video/*R.*
    :param output_path: str
        The path to save the extracted poses. The output pattern is: output_path/{participant_id}/{participant_id}_{timestep}.png
    :param final_fps: int
        The final FPS of the extracted poses.
    :return: None
    """

    # allocations of the participants on the left-right videos are as follows (manually annotated by me):
    allocation= {
        # IS1000a
        'IS1000a.L': {'L':'IS1000a_1', 'R':'IS1000a_3'},
        'IS1000a.R': {'L':'IS1000a_4', 'R':'IS1000a_2'},
        # IS1001a
        'IS1001a.L': {'L':'IS1001a_1', 'R':'IS1001a_3'},
        'IS1001a.R': {'L':'IS1001a_4', 'R':'IS1001a_2'},
        # IS1001b
        'IS1001b.L': {'L':'IS1001b_1', 'R':'IS1001b_3'},
        'IS1001b.R': {'L':'IS1001b_4', 'R':'IS1001b_2'},
        # IS1001c
        'IS1001c.L': {'L':'IS1001c_1', 'R':'IS1001c_3'},
        'IS1001c.R': {'L':'IS1001c_4', 'R':'IS1001c_2'},
        # IS1001d
        'IS1003b.L': {'L':'IS1003b_1', 'R':'IS1003b_3'},
        'IS1003b.R': {'L':'IS1003b_4', 'R':'IS1003b_2'},
        # IS1003c
        'IS1003d.L': {'L':'IS1003d_1', 'R':'IS1003d_3'},
        'IS1003d.R': {'L':'IS1003d_4', 'R':'IS1003d_2'},
        # IS1006b
        'IS1006b.L': {'L':'IS1006b_1', 'R':'IS1006b_3'},
        'IS1006b.R': {'L':'IS1006b_4', 'R':'IS1006b_2'},
        # IS1008a
        'IS1008a.L': {'L':'IS1008a_1', 'R':'IS1008a_3'},
        'IS1008a.R': {'L':'IS1008a_4', 'R':'IS1008a_2'},
        # IS1008b
        'IS1008b.L': {'L':'IS1008b_1', 'R':'IS1008b_3'},
        'IS1008b.R': {'L':'IS1008b_4', 'R':'IS1008b_2'},
        # IS1008c
        'IS1008c.L': {'L':'IS1008c_1', 'R':'IS1008c_3'},
        'IS1008c.R': {'L':'IS1008c_4', 'R':'IS1008c_2'},
        # IS1008d
        'IS1008d.L': {'L':'IS1008d_1', 'R':'IS1008d_3'},
        'IS1008d.R': {'L':'IS1008d_4', 'R':'IS1008d_2'},
    }
    # generate metadata
    metadata = pd.DataFrame(columns=['video_name', 'filename', 'participant_id', 'frame_number', 'timestep', 'found_pose'])
    # load the pose detector
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
    # generate video paths
    video_filenames = glob.glob(os.path.join(path_to_data, "**", "video", "*L.*")) + glob.glob(os.path.join(path_to_data, "**", "video", "*R.*"))
    # go through all videos
    for video_filename in tqdm(video_filenames, desc="Processing videos..."):
        video_name = '.'.join(os.path.basename(video_filename).split('.')[:-1])
        left_participant_id = allocation[video_name]['L']
        right_participant_id = allocation[video_name]['R']
        metadata_l, metadata_r = extract_poses_single_video(video_filename, pose_detector, output_path, final_fps,
                                                            left_participant_id, right_participant_id)
        metadata = pd.concat([metadata, metadata_l, metadata_r], ignore_index=True)
    # save the metadata
    metadata.reset_index(drop=True, inplace=True)
    metadata.to_csv(os.path.join(output_path, "metadata_all.csv"), index=False)




def extract_pose_from_frame(frame:np.ndarray, detector:torch.nn.Module, row:Dict[str,object]):
    pose_bbox = get_pose_bbox(frame, detector)
    if pose_bbox is None:
        row.update({'found_pose': False})
        return None, row
    else:
        pose = crop_frame_to_pose(frame, pose_bbox, limits=[30,30])
        row.update({'found_pose': True})
        return pose, row




def extract_poses_single_video(path_to_video:str, pose_detector:torch.nn.Module, output_path:str,
                               final_fps:int, left_participant_id:str, right_participant_id:str)->Tuple[pd.DataFrame, pd.DataFrame]:
    """ Extracts the poses from the video and saves them to the output_path. Returns the metafile with all information about extracted poses.
    The video contains two persons sitting next to each other (left and right). Their positions are approximately half-half in the
    width of the video. The function extracts the poses of both persons and saves them to the output_path.

    :param path_to_video: str
        path to the video to extract the poses from.
    :param pose_detector: torch.nn.Module
        The loaded HRNet model from datatools.pytorch_utils.models.Pose_estimation.HRNet.py.
    :param output_path: str
        The path to save the extracted poses.
    :param final_fps: int
        The final FPS of the extracted poses.
    :param left_participant_id: str
        The ID of the participant sitting on the left side of the video.
    :param right_participant_id: str
        The ID of the participant sitting on the right side of the video.
    :return: pd.DataFrame
        The metafile with all information about extracted poses.
    """
    # create a folder to save the extracted poses
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(os.path.join(output_path, left_participant_id)):
        os.makedirs(os.path.join(output_path, left_participant_id))
    if not os.path.exists(os.path.join(output_path, right_participant_id)):
        os.makedirs(os.path.join(output_path, right_participant_id))
    # create metafile to save the information about the extracted poses
    metadata_l = pd.DataFrame(columns=['video_name', 'filename', 'participant_id', 'frame_number', 'timestep', 'found_pose'])
    metadata_r = pd.DataFrame(columns=['video_name', 'filename', 'participant_id', 'frame_number', 'timestep', 'found_pose'])
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
                    # pre-form rows for metadata
                    l_row = { 'video_name': os.path.basename(path_to_video),
                        'filename': f"{left_participant_id}_{timestamp}.png",
                        'participant_id': left_participant_id, 'frame_number': counter, 'timestep': timestamp}

                    r_row = {'video_name': os.path.basename(path_to_video),
                        'filename': f"{right_participant_id}_{timestamp}.png",
                        'participant_id': right_participant_id, 'frame_number': counter, 'timestep': timestamp}
                    # extraction of poses
                    left_pose_frame, l_row = extract_pose_from_frame(left_frame, pose_detector, l_row)
                    right_pose_frame, r_row = extract_pose_from_frame(right_frame, pose_detector, r_row)
                    # save the extracted poses
                    if left_pose_frame is not None:
                        Image.fromarray(left_pose_frame).save(os.path.join(output_path, left_participant_id, f"{left_participant_id}_{timestamp}.png"))
                    if right_pose_frame is not None:
                        Image.fromarray(right_pose_frame).save(os.path.join(output_path, right_participant_id, f"{right_participant_id}_{timestamp}.png"))
                    # save the metadata
                    metadata_l = pd.concat([metadata_l, pd.DataFrame([l_row])], ignore_index=True)
                    metadata_r = pd.concat([metadata_r, pd.DataFrame([r_row])], ignore_index=True)
                counter += 1
            else:
                break
    # return the metadata
    return metadata_l, metadata_r



def main():
    path_to_data = "/work/home/dsu/Datasets/DOME/amicorpus/"
    output_path = "/work/home/dsu/Datasets/DOME/extracted_poses/"
    final_fps = 5
    extract_poses_all_videos(path_to_data, output_path, final_fps)


if __name__ == '__main__':
    main()