import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import neighbors
import os

# This script is a test for Knn regression

if __name__ == '__main__':
    # load vilearn windowed data
    working_dir = os.getcwd()
    data_folder = 'Recordings/SavedData/v2_no_low_sampled'
    data_file = 'all_features_60s_resampled.csv'
    full_data_path = os.path.join(working_dir, data_folder, data_file)
    data = pd.read_csv(full_data_path)
    dyad_cols = [col for col in data.columns if 'dyad' in col]
    triad_cols = [col for col in data.columns if 'triad' in col]
    target_cols = [col for col in data.columns if 'target' in col]
    print('hello')