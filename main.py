import pandas as pd
import gudhi as gd
from collections import defaultdict
from sklearn.cluster import DBSCAN


def read_dataset(filename):
    fin = pd.read_csv(filename).fillna(0)
    return fin


'''
A way of converting one dimensional data into a point cloud data cooridinates for Topological data analysis
'''


def pcd(row):
    info_cols = ["ID", "HISTOGRAM_MIDDLE_NORM", "CYCLE_PHASE_ESTIMATED"]
    protein_cols = row.drop(labels=info_cols)
    point_cloud = [(protein_cols.iloc[i], protein_cols.iloc[i + 1]) for i in range(len(protein_cols) - 1)]
    info = tuple(row[info_cols])
    return info, point_cloud


'''
Generate a point cloud dataset for each row of the dataset and make a dictionary of it
'''


def pcd_dict_df(dataset):
    pcd_dict = defaultdict()
    for index, row_info in dataset.iterrows():
        info, point_cloud = pcd(row_info)
        pcd_dict[info] = point_cloud
    return pcd_dict


'''
Create the list of persistence diagram for clustering
'''


def persistence_analysis(point_clouds: dict):
    persistence_diagrams = [gd.RipsComplex(points=pc).create_simplex_tree(max_dimension=2).persistence() for pc in
                            point_clouds.values()]
    return persistence_diagrams


if __name__ == '__main__':
    test_in = read_dataset("test_set (2).csv")
    train_in = read_dataset("train_set (2).csv")
    test_pcd = pcd_dict_df(test_in)
    train_pcd = pcd_dict_df(train_in)
    combined_pcd = test_pcd | train_pcd
    persistence = persistence_analysis(combined_pcd)
