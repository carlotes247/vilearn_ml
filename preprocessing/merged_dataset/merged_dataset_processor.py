import pandas as pd
import os

# TO USE THIS FILE YOU NEED TO PRODUCE FOUR DIFFERENT FILES
# 1. gaze file from plotting/plotting_eye_data.py by calling eye_plotter.create_line_plot(sampling)
# 2. blink durations file from data_reading/features/blink_stats.py by calling calculate_blink_duration_for_each_minute(sampling), this is called as part of another method in line 220
# 3. blink rate file from data_reading/features/blink_stats.py by calling calculate_blink_rate(sampling) which is called as part of class_obj.get_groups_blink_rate(sampling)
# 4: task engagement (TE) file from engagement_manager, but we can use the 90Hz file in data/annotations. It is not resampled to a particular sample rate, this is done in this processor class
class MergedDatasetProcessor():
    df_wide_gaze: pd.DataFrame
    working_dir: str
    data_folder_gaze: str 
    data_folder_blinks: str
    data_folder_TE: str
    file_path_gaze: str
    file_path_blink_duration_dyads:str
    file_path_blink_duration_triads:str
    file_path_blink_rate:str
    file_path_TE:str
    group_names_file:str 
    def __init__(self, sampling:int, filename_gaze:str, filename_blink_duration_dyads:str, filename_blink_duration_triads:str, filename_blink_rate:str, filename_TE:str, data_folder_gaze:str = "", data_folder_blinks:str = "", data_folder_TE:str = "") -> None:
        # load dataframes
        self.working_dir = os.getcwd()
        # gaze (wide df)
        self.data_folder_gaze = 'Recordings/SavedData/v2_no_low_sampled' if data_folder_gaze == "" else data_folder_gaze
        self.file_path_gaze = filename_gaze
        full_path_gaze = os.path.join(self.working_dir, self.data_folder_gaze, self.file_path_gaze)
        self.df_wide_gaze = pd.read_csv(full_path_gaze)        
        # blink duration (split in dyads and triads), blink rate (all groups)
        self.data_folder_blinks = "data" if data_folder_blinks == "" else data_folder_blinks
        self.file_path_blink_duration_dyads = filename_blink_duration_dyads
        self.file_path_blink_duration_triads = filename_blink_duration_triads
        self.file_path_blink_rate = filename_blink_rate
        full_path_blink_duration_dyads = os.path.join(self.working_dir, self.data_folder_blinks, self.file_path_blink_duration_dyads)
        full_path_blink_duration_triads = os.path.join(self.working_dir, self.data_folder_blinks, self.file_path_blink_duration_triads)
        full_path_blink_rates = os.path.join(self.working_dir, self.data_folder_blinks, self.file_path_blink_rate)
        self.df_blink_duration_dyads = pd.read_csv(full_path_blink_duration_dyads)
        self.df_blink_duration_triads = pd.read_csv(full_path_blink_duration_triads)
        self.df_blink_rates = pd.read_csv(full_path_blink_rates)        
        # TE (not sampled)
        self.data_folder_TE = 'data/annotations/' if data_folder_TE == "" else data_folder_TE
        self.file_path_TE = "all_groups_interaction_task_eng90Hz.csv" if filename_TE == "" else filename_TE
        full_path_TE = os.path.join(self.working_dir, self.data_folder_TE, self.file_path_TE)        
        self.df_TE = pd.read_csv(full_path_TE)
        # group_names
        group_names_df = pd.read_csv(os.path.join(self.working_dir, 'data', 'group_names_with_time_subsetsFullVERSION.csv'), sep=';')
        group_names = group_names_df['Group_Name'].to_list()

        # modify gaze from wide to long
        dyad_features_names = ["MG_P1P2","1d_DG_P1","1d_DG_P2","0_D1"]
        triad_features_names= ["", ""]
        df_long: pd.DataFrame = pd.DataFrame()
        for group_name in group_names:            
            group_cols = self.df_wide_gaze.columns.str.contains(group_name)
            df_group = self.df_wide_gaze[self.df_wide_gaze.columns[group_cols]]
            feature_names = dyad_features_names if "dyad" in group_name else triad_features_names
            feature_cols = [col for col in df_group.columns if any(term in col for term in feature_names)]
            cols = self.df_wide_gaze.columns[group_cols]
            df_group = df_group[feature_cols]
            df_group.dropna(inplace=True)
            if "dyad" in group_name:
                df_group = df_group.set_axis(['MG', '1d_DG_P1', '1d_DG_P2', '0_D1'], axis=1)
                df_group['1d_DG'] = df_group['1d_DG_P1'] + df_group['1d_DG_P2']
                df_group.drop(['1d_DG_P1', '1d_DG_P2'], axis=1, inplace=True)
            if "triad" in group_name:
                df_group['MG'] = df_group[f"{group_name}_MG_D1"] + df_group[f"{group_name}_MG_D0"]
                df_group['0_D1'] = df_group[f"{group_name}_0_D1"]
                df_group['1d_DG'] = df_group[f"{group_name}_3_D1"] + df_group[f"{group_name}_2_D1_different"] + df_group[f"{group_name}_2_D1_same"] + df_group[f"{group_name}_1_D1"]
                df_group.drop(cols, axis=1, inplace=True)
            df_group.insert(0, 'seconds_interaction_window', self.df_wide_gaze.iloc[df_group.index]['seconds_interaction_window'])
            df_group['group_name'] = group_name
            df_long = pd.concat([df_long, df_group])
            print("pio")
        # for dyads
        # dyad_01_MG_P1P2,dyad_01_1d_DG_P1,dyad_01_1d_DG_P2,dyad_01_0_D1

        # return dataset
        print("done")


if __name__ == "__main__":
    process_30s: bool = True
    sampling: int = 30
    file_name_gaze:str = ""
    file_name_blink_duration_dyads:str = ""
    file_name_blink_duration_triads:str = ""
    file_name_blink_rate:str =""
    file_name_TE:str = ""
    # 30s
    if process_30s and sampling == 30:
        file_name_gaze = f"all_features_{sampling}s_resampled2025-12-20.csv"
        file_name_blink_duration_dyads = f"blink_duration_per_minute_dyads_{sampling}s_2025-12-22.csv"
        file_name_blink_duration_triads = f"blink_duration_per_minute_triads_{sampling}s_2025-12-22.csv"
        file_name_blink_rate = f"blink_per_minute_all_groups_{sampling}s_2025-12-22.csv"
        file_name_TE = "all_groups_interaction_task_eng90Hz.csv"
    # 60s
    else:
        file_name_gaze = 'all_features_60s_resampled.csv'
        # data_file_with_TE = '60s_TE_correlation.csv'

    datasetCtrl: MergedDatasetProcessor = MergedDatasetProcessor(sampling=sampling, filename_gaze=file_name_gaze,
                                                                 filename_blink_duration_dyads=file_name_blink_duration_dyads, 
                                                                 filename_blink_duration_triads=file_name_blink_duration_triads,
                                                                 filename_blink_rate=file_name_blink_rate,
                                                                 filename_TE=file_name_TE)
