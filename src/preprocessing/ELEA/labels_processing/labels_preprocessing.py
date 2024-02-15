import glob
from typing import List

import pandas as pd
import numpy as np
import os

import scipy


def combine_annotations_from_annotators(paths:List[str])->pd.DataFrame:
    """ Reads annotations from multiple annotators and combines them into a single dataframe, averaging the
    annotations. All files should have csv format.

    :param paths: List[str]
        List of paths to the annotations files
    :return: pd.DataFrame
        Dataframe with columns ['group', 'PLead_1', 'PLead_2', 'PLead_3', 'PLead_4', 'PDom_1', 'PDom_2', 'PDom_3', 'PDom_4']
    """
    dataframes = [pd.read_csv(path) for path in paths]
    # average annotations
    result_dataframe = pd.DataFrame(columns=dataframes[0].columns)
    # change group number to group
    result_dataframe.rename(columns={'group_number': 'group'}, inplace=True)
    columns_to_sum = [column for column in dataframes[0].columns if column != 'group_number']
    for idx in range(len(dataframes[0])):
        group = dataframes[0].iloc[idx]['group_number']
        sum_element = dataframes[0].iloc[idx][columns_to_sum]
        for df in dataframes[1:]:
            sum_element += df.iloc[idx][columns_to_sum]
        # average
        sum_element = sum_element / len(dataframes)
        # add to result dataframe
        new_row = {'group': group, **sum_element}
        result_dataframe = pd.concat([result_dataframe, pd.DataFrame(new_row, index=[0])], ignore_index=True)
    result_dataframe.reset_index(inplace=True, drop=True)
    return result_dataframe








def main():
    # average annotations from three annotators
    paths = glob.glob("/work/home/dsu/Datasets/ELEA/preprocessed_labels/ELEA_external_annotations/*.csv")
    result = combine_annotations_from_annotators(paths)
    print(result)
    result.to_csv("/work/home/dsu/Datasets/ELEA/preprocessed_labels/ELEA_external_annotations/averaged_annotations.csv", index=False)

if __name__ == "__main__":
    main()