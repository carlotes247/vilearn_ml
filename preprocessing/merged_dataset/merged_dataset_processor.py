import pandas as pd
import os
class MergedDatasetProcessor():
    df_wide: pd.DataFrame
    working_dir: str
    data_folder: str 
    file_path: str
    group_names_file:str 
    def __init__(self, filename: str, data_folder_path: str = "") -> None:
        # load whole dataframe
        self.working_dir = os.getcwd()
        self.data_folder = 'Recordings/SavedData/v2_no_low_sampled' if data_folder_path == "" else data_folder_path
        self.file_path = filename
        full_data_path = os.path.join(self.working_dir, self.data_folder, self.file_path)
        self.df_wide = pd.read_csv(full_data_path)
        group_names_df = pd.read_csv(os.path.join(self.working_dir, 'data', 'group_names_with_time_subsetsFullVERSION.csv'), sep=';')
        group_names = group_names_df['Group_Name'].to_list()
        # modify from wide to long
        dyad_features_names = ["MG_P1P2","1d_DG_P1","1d_DG_P2","0_D1"]
        triad_features_names= ["", ""]
        df_long: pd.DataFrame = pd.DataFrame()
        for group_name in group_names:
            group_cols = self.df_wide.columns.str.contains(group_name)
            df_group = self.df_wide[self.df_wide.columns[group_cols]]
            feature_names = dyad_features_names if "dyad" in group_name else triad_features_names
            feature_cols = [col for col in df_group.columns if any(term in col for term in feature_names)]
            df_group = df_group[feature_cols]
            df_group.dropna(inplace=True)
            df_group.insert(0, 'seconds_interaction_window', self.df_wide.iloc[df_group.index]['seconds_interaction_window'])
            print("pio")
        # for dyads
        # dyad_01_MG_P1P2,dyad_01_1d_DG_P1,dyad_01_1d_DG_P2,dyad_01_0_D1

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
