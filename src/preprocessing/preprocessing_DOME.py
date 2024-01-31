import os

import pandas as pd
import cv2

from feature_extraction.pytorch_based.face_recognition_utils import recognize_one_face_bbox, extract_face_according_bbox


"""

THe procedure of extraction of facial frames in one video

    # create metadata with the following columns = ['video_name', 'frame_number', 'timestep', 'filename', 'found_face']
    # calculate the new FPS
    # go over frames in video:
        - if this is the right frame (according to the new FPS):
            - BGR2RGB
            - calculate timestep
            - get bboxes for every face in the frame
            - if there are some faces (bbox is not None):
                - save reference facial embeddings if it is not saved yet
                - extract all faces
                - compare facial embeddings of all faces with the reference
                - take the most similar face
            - put False in found_face column otherwise

"""





def extract_faces_closeup_video(path_to_video:str, output_path:str, final_fps:int,
                                face_detection_model:object):
    """ Extracts faces from a video and saves them as images. Face_detection_model will work only for the
    first encountered face in the video as the facial embeddings will be saved.

    :param path_to_video: str
        Path to the video file
    :param output_path: str
        Path to the output folder
    :param final_fps: int
        Final fps of the video. The video will be resampled to this fps.
    :param face_detection_model: object
        Face detection model. Based on RetinafaceDetector class.
    :return: None
    """
    # create output folder
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    # create metadata of the extracted frames
    metadata = pd.DataFrame(columns=['video_name', 'frame_number', 'timestep', 'filename', 'found_face'])
    # load video file
    video = cv2.VideoCapture(path_to_video)
    # get FPS
    FPS = video.get(cv2.CAP_PROP_FPS)
    FPS_in_seconds = 1. / FPS
    # calculate every which frame should be taken
    every_n_frame = int(round(FPS / final_fps))
    # facial embeddings of the person
    facial_embeddings_reference = None
    # go through all frames
    counter = 0
    while video.isOpened():
        ret, frame = video.read()
        if ret:
            if counter % every_n_frame == 0:
                # convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # calculate timestamp
                timestamp = counter * FPS_in_seconds
                # round it to 2 digits to make it readable
                timestamp = round(timestamp, 2)
                # recognize the face
                bbox =
                # if not recognized, note it as NaN
                if bbox is None:
                    output_filename = np.NaN
                    detected = False
                else:
                    # extract face
                    face = extract_face_according_bbox(frame, bbox)
                    # check if we have the facial reference embeddings already
                    if facial_embeddings_reference is None:






"""
Procedure of extraction of faces in every video in the folder

    # generate metadata_all file with the following columns: ['video_name', 'frame_number', 'timestep', 'filename', 'found_face']
    # get list of filenames in the folder
    # go over all filenames:
        - extract faces from the video
        - concat metadata_all with metadata of the current video
    # save metadata_all

"""


def extract_faces_closeup_videos_in_folder(path_to_folder:str, output_path:str, final_fps:int,
                                             face_detection_model:object):
     pass


"""
Procedure of extraction of affective embeddings for every video in the folder
    
        # load affective extractor model
        # create metadata_affective_embeddings file with the following columns: ['video_name', 'frame_number', 'timestep', 'filename', 'found_face'] + ['af_embedding_' + str(i) for i in range(256)]
        


"""















def main():
    pass

if __name__ == '__main__':
    main()