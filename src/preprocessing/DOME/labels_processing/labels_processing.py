import pandas as pd


def main():
    path_to_file = "/work/home/dsu/Datasets/DOME/Annotations/dome_annotations_M1.csv"
    df = pd.read_csv(path_to_file)
    print(df.head())

if __name__== "__main__":
    main()