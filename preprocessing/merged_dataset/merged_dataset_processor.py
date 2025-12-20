import pandas as pd
import os
class MergedDatasetProcessor():
    df: pd.DataFrame
    working_dir: str
    data_folder: str 
    file_path: str
    def __init__(self, filename: str, data_folder_path: str = "") -> None:
        # load whole dataframe
        self.working_dir = os.getcwd()
        self.data_folder = 'Recordings/SavedData/v2_no_low_sampled' if data_folder_path == "" else data_folder_path
        self.file_path = filename
        full_data_path = os.path.join(self.working_dir, self.data_folder, self.file_path)
        self.df = pd.read_csv(full_data_path)
        # modify from wide to long
        # return dataset
        print("done")


if __name__ == "__main__":
    process_30s: bool = True
    file_name:str = ""
    # 30s
    if process_30s:
        file_name = "all_features_30s_resampled2025-12-20.csv"
    # 60s
    else:
        file_name = 'all_features_60s_resampled.csv'
        # data_file_with_TE = '60s_TE_correlation.csv'

    datasetCtrl: MergedDatasetProcessor = MergedDatasetProcessor(filename=file_name)
