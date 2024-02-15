import pandas as pd
import numpy as np
import os

import scipy


def main():
    path_to_file = "/work/home/dsu/Datasets/ELEA/elea/questionnaires/matlabFiles/externalAnnotations_structure.mat"
    mat = scipy.io.loadmat(path_to_file)
    print(mat)
    mat['external_annotationsAVG'][11]

if __name__ == "__main__":
    main()