# The part_id matrix below contains the information for how to relate
# the recorded videos with the participant ids. Each row refers to
# meetings from 1 to 40. Each column refers to the videos _1 and _2
# and the participant on the left and the participant on the right
# in the following order: 1L 1R 2L 2R and mapped to participan IDs
# as follows: K:1, L:2, M:3, N:4
# 0 if there is no participants in that position.
#
# EXAMPLE:
# For group 23, group23_1.avi contains the recording of 2 participants.
# The one on the left has ID K and the one on the right has ID L.
# group23_2.avi contains the recording of a single participant (left side
# is empty) and the one on the right has ID M.

import glob
import os.path
import re
from typing import Dict, List

part_id = {
1:[0, 0, 0, 0], # template: video_number: [video_1_left, video_1_right, video_2_left, video_2_right],
2:[0, 2, 1, 3], # the values in the list are the participant ids in the following order: 1:K, 2:L, 3:M, 4:N
3:[2, 1, 0, 3],
4:[0, 3, 2, 1],
5:[2, 1, 0, 3],
6:[3, 1, 4, 2],
7:[1, 2, 4, 3],
8:[0, 3, 1, 2],
9:[4, 2, 1, 3],
10:[4, 2, 3, 1],
11:[3, 1, 4, 2],
12:[3, 1, 2, 4],
13:[0, 0, 0, 0],
14:[4, 1, 2, 3],
15:[3, 2, 1, 0],
16:[2, 3, 4, 1],
17:[2, 3, 1, 0],
18:[2, 1, 4, 3],
19:[1, 4, 2, 3],
20:[2, 3, 4, 1],
21:[1, 2, 3, 4],
22:[3, 2, 4, 1],
23:[1, 2, 0, 3],
24:[4, 1, 2, 3],
25:[4, 3, 1, 2],
26:[4, 1, 3, 2],
27:[1, 2, 3, 4],
28:[2, 3, 4, 1],
29:[2, 3, 1, 4],
30:[1, 4, 2, 3],
31:[3, 1, 2, 4],
32:[0, 2, 1, 3],
33:[2, 1, 4, 3],
34:[2, 3, 4, 1],
35:[2, 0, 3, 1],
36:[0, 3, 2, 1],
37:[1, 3, 2, 4],
38:[3, 2, 1, 4],
39:[2, 4, 3, 1],
40:[3, 2, 1, 4],
}

value_to_id = {1:'K', 2:'L', 3:'M', 4:'N'}

def prepare_id_splitting(path_to_videos:str, id_references:Dict[int, List[int]], value_to_id:Dict[int, str])->Dict[str, Dict[str, str]]:
    """ Prepares the splitting of the videos into participants and their IDs to extract faces/poses from the frames.
        To do so, the following is done:
        1. paths to the videos are found by the pattern: {path_to_videos}/*.avi
        2. for every video:
            2.1. Identify the group number and the 'video_type' (1 or 2) from the video name. The video name has
                the following pattern: group{group_number}_{video_type}.avi
            2.2. Extract the participants IDs from the id_references dictionary. This dictionary has the following
                structure: {group_number: [participant_1_left, participant_1_right, participant_2_left, participant_2_right]}
                The participants are identified by the following values in the list: 1:K, 2:L, 3:M, 4:N, 0: no participant
                So, for example, file group12_1.avi has the following values: [3, 1, 2, 4] (we look onto two first numbers as the video type is 1)
                                                                               _  _
                thus, the participants are: 3 on the left and 1 on the right, with IDs M and K, respectively.
            2.3. save it to result Dictionary with the following structure: {video_name: {left: ID, right: ID}}
            ID is g{group_number}_{participant_id} (e.g. g12_M)

    :param path_to_videos:
        Path to the folder with all the videos.
    :param id_references: Dict[int, List[int]]
        Dictionary with the references to the participants IDs. The keys are the group numbers and the values are lists
        of 4 integers, which are the IDs of the participants in the following order: 1L 1R 2L 2R
    :return: Dict[str, Dict[str, str]]
        Dictionary with the following structure: {video_name: {left: ID, right: ID}}. ID is g{group_number}_{participant_id}
    """
    # find all the videos
    videos = glob.glob(f'{path_to_videos}/*.avi')
    # create result dictionary
    result = {}
    # go over videos and extract participants IDs
    for video in videos:
        # extract group number and video type
        group_number, video_type = re.findall(r'group(\d+)_(\d).avi', video)[0]
        video_type = int(video_type)
        group_number = int(group_number)
        # extract participants IDs
        participants_ids = id_references[group_number]
        left_id = participants_ids[video_type*2-2]
        left_id = value_to_id[left_id]
        right_id = participants_ids[video_type*2-1]
        right_id = value_to_id[right_id]
        # save to result dictionary
        result[os.path.basename(video)] = {'left': f'g{group_number}_{left_id}', 'right': f'g{group_number}_{right_id}'}
    return result



def main():
    path_to_videos = "/work/home/dsu/Datasets/ELEA/elea/video/"
    result = prepare_id_splitting(path_to_videos, part_id, value_to_id)
    print(result)



if __name__ == '__main__':
    main()