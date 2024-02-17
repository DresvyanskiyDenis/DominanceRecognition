import numpy as np
import pandas as pd


def get_annotations_file(path_to_file:str)->pd.DataFrame:
    """ Reads the annotations file, transforms it to the convenient format and returns the dataframe.

    :param path_to_file: str
        Path to the annotations file
    :return: pd.DataFrame
        Dataframe with columns ['session_id', 'start_sec', 'end_sec', 'a_1_p_1', 'a_1_p_2', 'a_1_p_3', 'a_1_p_4',
        'a_2_p_1', 'a_2_p_2', 'a_2_p_3', 'a_2_p_4',
        'a_3_p_1', 'a_3_p_2', 'a_3_p_3', 'a_3_p_4'], where a_i_p_j is the annotation of the i-th annotator for the
                                                        j-th participant.
    """
    df = pd.read_csv(path_to_file, delimiter=';')
    # change column names. It was: [name;start (sec);end (sec);A11;A12;A13;A14;A21;A22;A23;A24;A31;A32;A33;A34]
    # it should be: ['session_id', 'start_sec', 'end_sec', 'a_1_p_1', 'a_1_p_2', 'a_1_p_3', 'a_1_p_4',
    #                 'a_2_p_1', 'a_2_p_2', 'a_2_p_3', 'a_2_p_4',
    #                 'a_3_p_1', 'a_3_p_2', 'a_3_p_3', 'a_3_p_4']
    df.rename(columns={'name': 'session_id', 'start (sec)': 'start_sec', 'end (sec)': 'end_sec',
                       'A11': 'a_1_p_1', 'A12': 'a_1_p_2', 'A13': 'a_1_p_3', 'A14': 'a_1_p_4',
                       'A21': 'a_2_p_1', 'A22': 'a_2_p_2', 'A23': 'a_2_p_3', 'A24': 'a_2_p_4',
                       'A31': 'a_3_p_1', 'A32': 'a_3_p_2', 'A33': 'a_3_p_3', 'A34': 'a_3_p_4'}, inplace=True)
    return df


def transform_to_most_and_least_dominance_persons(df:pd.DataFrame)->pd.DataFrame:
    """ Transforms the dataframe got from get_annotations_file to the dataframe with the most and the least dominant
    persons for each session id and corresponding start and end seconds.
    As there are three annotators, the most and the least dominant persons are chosen by the majority vote.
    However, sometimes there is no majority, in this case the most dominant person is np.NaN.
    1 - the most dominant person, 4 - the least dominant person.

    :param df: pd.DataFrame
        Dataframe with columns ['session_id', 'start_sec', 'end_sec', 'a_1_p_1', 'a_1_p_2', 'a_1_p_3', 'a_1_p_4',
                                'a_2_p_1', 'a_2_p_2', 'a_2_p_3', 'a_2_p_4',
                                'a_3_p_1', 'a_3_p_2', 'a_3_p_3', 'a_3_p_4']
    :return: pf.DataFrame
        Dataframe with columns ['session_id', 'start_sec', 'end_sec', 'most_dominant', 'least_dominant']
    """
    result = pd.DataFrame(columns=['session_id', 'start_sec', 'end_sec', 'most_dominant', 'least_dominant'])
    # go over all rows of df
    for idx in range(len(df)):
        session_id = df.iloc[idx]['session_id']
        start_sec = df.iloc[idx]['start_sec']
        end_sec = df.iloc[idx]['end_sec']
        # get annotations for all participants
        annotations = df.iloc[idx][['a_1_p_1', 'a_1_p_2', 'a_1_p_3', 'a_1_p_4',
                                    'a_2_p_1', 'a_2_p_2', 'a_2_p_3', 'a_2_p_4',
                                    'a_3_p_1', 'a_3_p_2', 'a_3_p_3', 'a_3_p_4']]
        # get the most dominant person. To do so, find columns that equal to 1 first
        most_dominant = annotations[annotations == 1]
        # extract participant ids from the column names
        most_dominant = [int(column.split('_')[-1]) for column in most_dominant.index]
        # if there is a number that appears more than once in most_dominant list, then it means that two or more annotators chose the same
        # participant as the most dominant. If all numbers in most_dominant list are different from each other, then there is no majority
        # and the most dominant person is np.NaN
        if len(set(most_dominant)) == 3:
            most_dominant = np.NaN
        else:
            most_dominant = max(set(most_dominant), key=most_dominant.count)

        # get the least dominant person. To do so, find columns that equal to 4 first
        least_dominant = annotations[annotations == 4]
        # extract participant ids from the column names
        least_dominant = [int(column.split('_')[-1]) for column in least_dominant.index]
        # if there is a number that appears more than once in least_dominant list, then it means that two or more annotators chose the same
        # participant as the least dominant. If all numbers in least_dominant list are different from each other, then there is no majority
        # and the least dominant person is np.NaN
        if len(set(least_dominant)) == 3:
            least_dominant = np.NaN
        else:
            least_dominant = max(set(least_dominant), key=least_dominant.count)
        # add to result dataframe
        new_row = {'session_id': session_id, 'start_sec': start_sec, 'end_sec': end_sec, 'most_dominant': most_dominant,
                   'least_dominant': least_dominant}
        result = pd.concat([result, pd.DataFrame(new_row, index=[0])], ignore_index=True)
    result.reset_index(inplace=True, drop=True)
    return result


def get_most_least_dominant_participants_DOME(path_to_file:str)->pd.DataFrame:
    """ Loads, combines and transforms the annotations from multiple annotators into a single dataframe with the most
    and least dominant participants.

    :param path_to_file: str
        Path to the annotations file
    :return: pd.DataFrame
        Dataframe with columns ['session_id', 'start_sec', 'end_sec', 'most_dominant', 'least_dominant']
    """
    df = get_annotations_file(path_to_file)
    result = transform_to_most_and_least_dominance_persons(df)
    return result





def main():
    path_to_file = "/work/home/dsu/Datasets/DOME/Annotations/dome_annotations_M1.csv"
    m_l_dominant = get_most_least_dominant_participants_DOME(path_to_file)
    print(m_l_dominant)



if __name__== "__main__":
    main()