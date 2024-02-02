import glob
import os
import sys
sys.path.append("/work/home/dsu/PhD/scripts/DominanceRecognition/")
sys.path.append("/work/home/dsu/PhD/scripts/datatools/")
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from tqdm import tqdm

from feature_extraction.tf_based.deep_face_utils import recognize_faces, extract_face_according_bbox, verify_two_images


def extract_reference_facial_frames_every_participant(path_to_data:str, output_path:str, detector:str='retinaface'):
    """ Extracts reference facial frames for every participant in the DOME dataset. To do so, we go over
    every video file with the '*Closeup*.avi' pattern in all folders in the path_to_data folder.
    Then, the frame from the 300th second (5th minute) is taken, the face is detected and saved as a reference.
    If the face is not detected, the frame is skipped and the next one in 5 seconds is taken.

    :param path_to_data: str
        Path to the folder with the video data of DOME dataset.
    :param detector: str
        Detector to use. Can be 'opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe','yolov8','yunet', or 'fastmtcnn'.
    :param output_path: str
        Path to the output folder, where the reference frames will be saved.
    :return: None
    """
    # manually chosen timesteps for the reference frame extraction for every participant
    # video names are : IS1000a  IS1001a  IS1001b  IS1001c  IS1003b  IS1003d  IS1006b  IS1008a  IS1008b  IS1008c  IS1008d
    # every video has 4 participants, so we have 4 reference frames for every video
    needed_timesteps = {
        'IS1000a_1': 13694, 'IS1000a_2': 20212, 'IS1000a_3': 2703, 'IS1000a_4': 20193,
        'IS1001a_1': 300*25, 'IS1001a_2': 150*25, 'IS1001a_3': 21337, 'IS1001a_4': 150*25,
        'IS1001b_1': 900*25, 'IS1001b_2': 900*25, 'IS1001b_3': 1312, 'IS1001b_4': 900*25,
        'IS1001c_1': 32366, 'IS1001c_2': 900*25, 'IS1001c_3': 900*25, 'IS1001c_4': 900*25,
        'IS1003b_1': 15094, 'IS1003b_2': 18996, 'IS1003b_3': 29383, 'IS1003b_4': 900*25,
        'IS1003d_1': 18630, 'IS1003d_2': 5238, 'IS1003d_3': 900*25, 'IS1003d_4': 27056,
        'IS1006b_1': 9376, 'IS1006b_2': 16162, 'IS1006b_3': 900*25, 'IS1006b_4': 900*25,
        'IS1008a_1': 900*25, 'IS1008a_2': 900*25, 'IS1008a_3': 900*25, 'IS1008a_4': 900*25,
        'IS1008b_1': 900*25, 'IS1008b_2': 900*25, 'IS1008b_3': 900*25, 'IS1008b_4': 900*25,
        'IS1008c_1': 1189, 'IS1008c_2': 14167, 'IS1008c_3': 900*25, 'IS1008c_4': 16252,
        'IS1008d_1': 10296, 'IS1008d_2': 900*25, 'IS1008d_3': 900*25, 'IS1008d_4': 900*25,
    }
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    # get absolute paths to all closeup videos
    paths_to_videos = glob.glob(os.path.join(path_to_data, '**', 'video', '*Closeup*.avi'), recursive=True)
    #                                                  session_name     participant_id_in_session
    # go over all videos
    for video_filename in (pbar :=tqdm(paths_to_videos)):
        pbar.set_description(f"Processing reference frames...")
        # construct participant_id
        participant_id = video_filename.split(os.sep)[-3] + '_' + video_filename.split(os.sep)[-1].split('.')[-2][-1]
        session_name = video_filename.split(os.sep)[-3]
        # load video
        video = cv2.VideoCapture(video_filename)
        # get FPS
        FPS = video.get(cv2.CAP_PROP_FPS)
        # get frame number of the 900th second
        frame_number = needed_timesteps[participant_id]
        # set counter
        counter = 0
        # go over all frames
        while video.isOpened():
            ret, frame = video.read()
            if ret:
                if counter == frame_number:
                    # convert BGR to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = np.array(frame)
                    # recognize the face
                    bbox = recognize_faces(img=frame, detector=detector)
                    # if not recognized, skip the frame and take the next one in 15 seconds
                    if bbox is None:
                        frame_number += int(15 * FPS)
                        continue
                    # extract face
                    face = extract_face_according_bbox(frame, bbox[0]) # take the first face
                    # save the face
                    Image.fromarray(face).save(os.path.join(output_path, participant_id + '.png'))
                    break
                counter += 1
            else:
                break


def extract_faces_closeup_video(path_to_video:str, output_path:str, final_fps:int, path_to_face_reference:str,
                                face_detection_model:str='retinaface',
                                face_verification_model:str='ArcFace') -> pd.DataFrame:
    """ Extracts faces from a video and saves them as images. Face_detection_model will work only for the
    first encountered face in the video as the facial embeddings will be saved.

    :param path_to_video: str
        Path to the video file
    :param output_path: str
        Path to the output folder
    :param final_fps: int
        Final fps of the video. The video will be resampled to this fps.
    :param path_to_face_reference: str
        Path to the file with face reference of the person in the video.
    :param face_detection_model: str
        Face detection model to use. Can be 'opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe','yolov8','yunet', or 'fastmtcnn'.
    :param face_verification_model: str
        Face verification model to use. Can be 'VGG-Face', 'Facenet', 'Facenet512', 'OpenFace', 'ArcFace', 'Dlib', or 'SFace'.
    :return: pd.DataFrame
        Metadata of the extracted frames. Columns: ['video_name', 'frame_number', 'timestep', 'filename', 'found_face']
    """
    # create output folder
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    # create metadata of the extracted frames
    metadata = pd.DataFrame(columns=['video_name', 'frame_number', 'timestep', 'filename', 'found_face'])
    participant_id = path_to_video.split(os.sep)[-1].split('.')[-2][-1]
    session_name = path_to_video.split(os.sep)[-1].split('.')[0]
    # load video file
    video = cv2.VideoCapture(path_to_video)
    # get FPS
    FPS = video.get(cv2.CAP_PROP_FPS)
    FPS_in_seconds = 1. / FPS
    # calculate every which frame should be taken
    every_n_frame = int(round(FPS / final_fps))
    # facial embeddings of the person
    face_reference = np.array(Image.open(path_to_face_reference))
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
                    # get bboxes for every face in the frame
                    bboxes = recognize_faces(img=frame, detector=face_detection_model)
                    if bboxes is None:
                        found_face = False
                        save_filename = None
                    else:
                        # extract all faces
                        faces = [extract_face_according_bbox(frame, bbox) for bbox in bboxes]
                        # compare facial embeddings of all faces with the reference. The function returns the distance
                        # between the embeddings. The smaller the distance, the more similar the faces are.
                        verification = [verify_two_images(face, face_reference, face_verification_model, return_distance=True) for face in faces] # list of tuples (verified, distance)
                        # take the most similar face
                        face = faces[np.argmin([dist for _, dist in verification])]
                        # save the face
                        save_filename = '%s_%s_%.2f.png' % (session_name, participant_id, timestamp) # session_participantid_timestamp.png
                        Image.fromarray(face).save(os.path.join(output_path, save_filename))
                        found_face = True
                    # add metadata (append does not exist anymore)
                    row = {'video_name': session_name, 'frame_number': counter, 'timestep': timestamp,
                              'filename': save_filename, 'found_face': found_face}
                    metadata = pd.concat([metadata, pd.DataFrame(row, index=[0])], ignore_index=True)
                counter += 1
            else:
                break
    # save metadata
    metadata.reset_index(drop=True, inplace=True)
    metadata.to_csv(os.path.join(output_path, 'metadata.csv'), index=False)
    return metadata


def extract_faces_closeup_videos_all(path_to_folder:str, output_path:str, final_fps:int, face_detection_model:str, face_verification_model:str):
    """ Extracts faces from all videos in the folder and saves them as images. Before extraction of all frames,
    the reference faces for every participant will be extracted as well and saved to output_path/reference_faces.
    The pattern filename of the reference face is session_participantId.png.

    :param path_to_folder: str
       Path to the folder with the videos.
    :param output_path: str
       Path to the output folder, where the extracted faces will be saved. Each participant will be saved to
       output_path/session_participantId folder.
    :param final_fps: int
       Final fps of the video. The video will be resampled to this fps by taking every n-th frame.
    :param face_detection_model: str
       Face detection model to use. Can be 'opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe','yolov8','yunet', or 'fastmtcnn'.
    :param face_verification_model: str
       Face verification model to use (will be used to verify the extracted face with reference).
       Can be 'VGG-Face', 'Facenet', 'Facenet512', 'OpenFace', 'ArcFace', 'Dlib', or 'SFace'.
    :return: None
    """
    # create output folder
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    # create folder for reference faces
    if not os.path.exists(os.path.join(output_path, 'reference_faces')):
        os.makedirs(os.path.join(output_path, 'reference_faces'), exist_ok=True)
    # extract reference faces for every participant
    extract_reference_facial_frames_every_participant(path_to_data=path_to_folder,
                                                       output_path=os.path.join(output_path, 'reference_faces'),
                                                       detector=face_detection_model)
    # create metadata of the extracted frames
    metadata_all = pd.DataFrame(columns=['video_name', 'frame_number', 'timestep', 'filename', 'found_face'])
    # get list of filenames in the folder
    videos = glob.glob(os.path.join(path_to_folder, '**', 'video', '*Closeup*.avi'))
    # go over all filenames:
    for video in tqdm(videos, desc="Extracting faces from videos..."):
        participant_id = video.split(os.sep)[-3] + '_' + video.split(os.sep)[-1].split('.')[-2][-1] # session+participantId
        path_to_reference_face = os.path.join(output_path, 'reference_faces', f'{participant_id}.png')
        output_path_current_video = os.path.join(output_path, participant_id)
        # extract faces from the video
        video_metadata = extract_faces_closeup_video(path_to_video=video, output_path=output_path_current_video, final_fps=final_fps,
                                    path_to_face_reference=path_to_reference_face,
                                    face_detection_model=face_detection_model, face_verification_model=face_verification_model)
        # concat metadata_all with metadata of the current video
        metadata_all = pd.concat([metadata_all, video_metadata], ignore_index=True)
        del video_metadata
    # save metadata_all
    metadata_all.to_csv(os.path.join(output_path, 'metadata_all.csv'), index=False)


def main():
    path_to_dataset = "/work/home/dsu/Datasets/DOME/amicorpus/"
    path_to_extracted_faces = "/work/home/dsu/Datasets/DOME/extracted_faces/"
    extract_faces_closeup_videos_all(path_to_folder=path_to_dataset,
                                     output_path=path_to_extracted_faces,
                                     final_fps=5,
                                     face_detection_model='retinaface',
                                     face_verification_model='ArcFace')



if __name__ == '__main__':
    main()