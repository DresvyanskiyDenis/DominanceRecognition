import glob
import os
import sys
from typing import Dict, Union, Tuple

from feature_extraction.tf_based.deep_face_utils import recognize_faces, extract_face_according_bbox
from src.preprocessing.ELEA.participants_ids import prepare_id_splitting

sys.path.append("/work/home/dsu/PhD/scripts/DominanceRecognition/")
sys.path.append("/work/home/dsu/PhD/scripts/datatools/")
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from tqdm import tqdm


def recognize_face_from_frame(frame:np.ndarray, face_detection_model:str='retinaface')->Union[None, np.ndarray]:
    # recognize bboxes for the frame
    bboxes = recognize_faces(frame, face_detection_model)
    if bboxes == None:
        return None
    # take the first bbox
    bbox = bboxes[0]
    # extract the face
    face = extract_face_according_bbox(frame, bbox)
    return face




def extract_faces_single_video(path_to_video:str, output_path:str, participants_ids:Dict[str, str], final_fps:int,
                               face_detection_model:str='retinaface')->Tuple[pd.DataFrame, pd.DataFrame]:
    """ Extracts faces from a single video and saves them to the output folder.

    :param path_to_video: str
        Path to the video.
    :param output_path: str
        Path to the foutput folder.
    :param participants_ids: Dict[str, str]
        Dictionary with the following structure: {left: ID, right: ID}. ID is g{group_number}_{participant_id}
    :param final_fps: int
        The final FPS of the video.
    :param face_detection_model: str
        Face detection model to use. Can be 'opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe','yolov8','yunet', or 'fastmtcnn'.
    :return: Tuple[pd.DataFrame, pd.DataFrame]
        Two DataFrame for the left and right person
        with the following columns: ['video_name', 'participant_id', 'frame_number', 'timestep', 'filename', 'found_face']
    """
    # create output folder
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    # create subfolders for the left and right participants
    left_output = os.path.join(output_path, participants_ids['left'])
    right_output = os.path.join(output_path, participants_ids['right'])
    os.makedirs(left_output, exist_ok=True)
    os.makedirs(right_output, exist_ok=True)
    # create metadata of the extracted frames
    metadata_l = pd.DataFrame(columns=['video_name', 'participant_id', 'frame_number', 'timestep', 'filename', 'found_face'])
    metadata_r = pd.DataFrame(columns=['video_name', 'participant_id', 'frame_number', 'timestep', 'filename', 'found_face'])
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
                    # pre-form the rows for metafiles
                    row_l = {'video_name': os.path.basename(path_to_video), 'participant_id': participants_ids['left'],
                             'frame_number': counter, 'timestep': timestamp, 'filename': f'{participants_ids["left"]}_{timestamp}.png'}
                    row_r = {'video_name': os.path.basename(path_to_video), 'participant_id': participants_ids['right'],
                                'frame_number': counter, 'timestep': timestamp, 'filename': f'{participants_ids["right"]}_{timestamp}.png'}
                    # recognize faces
                    left_face_bboxes = recognize_face_from_frame(left_frame, face_detection_model)
                    right_face_bboxes = recognize_face_from_frame(right_frame, face_detection_model)
                    # process left results
                    if left_face_bboxes is not None:
                        # save the face
                        Image.fromarray(left_face_bboxes).save(os.path.join(left_output, row_l['filename']))
                        row_l['found_face'] = True
                    else:
                        row_l['found_face'] = False
                    # process right results
                    if right_face_bboxes is not None:
                        # save the face
                        Image.fromarray(right_face_bboxes).save(os.path.join(right_output, row_r['filename']))
                        row_r['found_face'] = True
                    else:
                        row_r['found_face'] = False
                    # append to metadata
                    metadata_l = pd.concat([metadata_l, pd.DataFrame([row_l])], ignore_index=True)
                    metadata_r = pd.concat([metadata_r, pd.DataFrame([row_r])], ignore_index=True)
                counter += 1
            else:
                break
    return metadata_l, metadata_r



def extract_faces_all_videos(path_to_videos:str, output_path:str, final_fps:int,
                             face_detection_model:str='retinaface'):
    """ Extracts faces from all videos and saves them to the output folder. To do so, the function uses provided
    path to the folder, where there are all videos and the final FPS. The function also uses the participants_ids
    dictionary, which is created by the prepare_id_splitting function. The function generates dictionary with the
    following structure: {video_name: {left: ID, right: ID}}. ID is g{group_number}_{participant_id}
    All faces will be extracted to different subfolders with the participant IDs as the folder names.


    :param path_to_videos: str
        Path to the folder with all the videos.
    :param output_path: str
        Path to the output folder. There, many subfolders will be created, each for the participant.
    :param final_fps: int
        The final FPS of the extracted data.
    :param face_detection_model: str
        Face detection model to use. Can be 'opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe','yolov8','yunet', or 'fastmtcnn'.
    :return: None
    """
    # create output folder
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    # get the participants IDs
    participants_ids = prepare_id_splitting(path_to_videos) # pattern: {video_name: {left: ID, right: ID}}
    # generate video paths
    videos = glob.glob(f'{path_to_videos}/*.avi')
    # generate metadata
    metadata_all = pd.DataFrame(columns=['video_name', 'participant_id', 'frame_number', 'timestep', 'filename', 'found_face'])
    # go through all videos
    for video in tqdm(videos, desc='Processing videos...'):
        if os.path.basename(video) != 'group36_1.avi':
            continue
        # get participant id
        participants = participants_ids[os.path.basename(video)]
        # if there is noParticipant in the items, replace noParticipant with 'garbage' subfolder name so that
        # all such nopParticipants from different videos will be saved to the same folder that will be deleted then
        # moreover, 'garbage' participant_id will be sorted out from the metadata_all
        for key, value in participants.items():
            if 'noParticipant' in value:
                participants[key] = 'garbage'
        # extract faces
        metadata_l, metadata_r = extract_faces_single_video(video, output_path, participants, final_fps, face_detection_model)
        # append to metadata
        metadata_all = pd.concat([metadata_all, metadata_l, metadata_r], ignore_index=True)
    # sort out the 'garbage' participant_id
    metadata_all = metadata_all[metadata_all['participant_id'] != 'garbage']
    # save metadata
    metadata_all.to_csv(os.path.join(output_path, 'metadata.csv'), index=False)












def main():
    path_to_data = '/work/home/dsu/Datasets/ELEA/elea/video/'
    output_path = '/work/home/dsu/Datasets/ELEA/preprocessed/faces/'
    final_fps = 5
    face_detection_model = 'retinaface'
    extract_faces_all_videos(path_to_data, output_path, final_fps, face_detection_model)



if __name__ == '__main__':
    main()