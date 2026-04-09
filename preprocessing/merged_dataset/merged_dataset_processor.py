import pandas as pd
import os
import datetime

# TO USE THIS FILE YOU NEED TO PRODUCE SEVERAL DIFFERENT FILES
# 1. gaze file from plotting/plotting_eye_data.py by calling eye_plotter.create_line_plot(sampling)
# 2. blink durations file from data_reading/features/blink_stats.py by calling calculate_blink_duration_for_each_minute(sampling), this is called as part of another method in line 220
# 3. blink rate file from data_reading/features/blink_stats.py by calling calculate_blink_rate(sampling) which is called as part of class_obj.get_groups_blink_rate(sampling)
# 4: task engagement (TE) file from engagement_manager, but we can use the 90Hz file in data/annotations. It is not resampled to a particular sample rate, this is done in this processor class
# 5: speaking-silence file from discover server
class MergedDatasetProcessor():
    df_processed: pd.DataFrame
    df_wide_gaze: pd.DataFrame
    df_long_gaze: pd.DataFrame
    df_speaking: pd.DataFrame
    df_TE: pd.DataFrame
    working_dir: str
    data_folder_gaze: str 
    data_folder_blinks: str
    data_folder_TE: str
    data_folder_speaking: str
    file_path_gaze: str
    file_path_blink_duration_dyads:str
    file_path_blink_duration_triads:str
    file_path_blink_rate:str
    file_path_TE:str
    file_path_speaking: str
    group_names_file:str 

    def __init__(self, sampling:int, filename_gaze:str, filename_blink_duration_dyads:str, filename_blink_duration_triads:str, filename_blink_rate:str, filename_speaking_gaze:str, filename_TE:str, data_folder_gaze:str = "", data_folder_blinks:str = "", data_folder_TE:str = ""):
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
        # Speaking
        self.data_folder_speaking = 'data/discover/merged'
        self.file_path_speaking = filename_speaking_gaze
        full_path_speaking = os.path.join(self.working_dir, self.data_folder_speaking, self.file_path_speaking)  
        self.df_speaking = pd.read_csv(full_path_speaking)     
        # TE (not sampled)
        self.data_folder_TE = 'data/annotations/' if data_folder_TE == "" else data_folder_TE
        self.file_path_TE = "all_groups_interaction_task_eng90Hz.csv" if filename_TE == "" else filename_TE
        full_path_TE = os.path.join(self.working_dir, self.data_folder_TE, self.file_path_TE)        
        self.df_TE = pd.read_csv(full_path_TE)     
        # group_names
        group_names_df = pd.read_csv(os.path.join(self.working_dir, 'data', 'group_names_with_time_subsetsFullVERSION.csv'), sep=';')
        group_names = group_names_df['Group_Name'].to_list()

        # modify gaze df from wide to long
        self.df_long_gaze = self.__gaze_df_wide_to_long(self.df_wide_gaze, group_names)
        # merge blink rates into dataset (potential inconsistency with the rates calculated in blink_stats.py, but I think it's a result of how the are calculated there, taking into account the whole group interaction time and diving by sampling rate)
        df_merged:pd.DataFrame = self.__merge_df_blink_rate_into_df_long_gaze(self.df_blink_rates, self.df_long_gaze, group_names)
        # merge blink durations into dataset
        df_merged = self.__merge_df_blink_durations_into_df_long_gaze(self.df_blink_duration_dyads, self.df_blink_duration_triads,
                                                                    df_merged, group_names)
        # merge speaking x gaze into dataset
        df_merged = self.__merge_df_speaking_gaze_into_df_long_gaze(self.df_speaking, df_merged, group_names)
        # add TE column
        df_merged = self.__merge_resample_df_TE_into_df_long_gaze(sampling, self.df_TE, df_merged, group_names)
        # reorder columns to match the original 60s file that Cristina made by hand
        df_merged = df_merged.rename(columns={'seconds':'seconds_interaction_window'})
        df_merged = df_merged[["seconds_interaction_window","group_type","group_name","TE","MG","1d_DG","BPM","blink_durations", "G_OnSpeaker", "No_G_OnSpeaker", "G_SI", "No_G_SI"]]
        self.df_processed = df_merged

    def __gaze_df_wide_to_long(self, df_wide_gaze, group_names) -> pd.DataFrame:
        dyad_features_names = ["MG_P1P2","1d_DG_P1","1d_DG_P2","0_D1"]
        triad_features_names= ["", ""]
        df_long: pd.DataFrame = pd.DataFrame()
        for group_name in group_names:            
            group_cols = df_wide_gaze.columns.str.contains(group_name)
            df_group = df_wide_gaze[df_wide_gaze.columns[group_cols]]
            feature_names = dyad_features_names if "dyad" in group_name else triad_features_names
            feature_cols = [col for col in df_group.columns if any(term in col for term in feature_names)]
            cols = df_wide_gaze.columns[group_cols]
            df_group = df_group[feature_cols]
            df_group.dropna(inplace=True)
            if "dyad" in group_name:
                df_group = df_group.set_axis(['MG', '1d_DG_P1', '1d_DG_P2', '0_D1'], axis=1)
                df_group['1d_DG'] = df_group['1d_DG_P1'] + df_group['1d_DG_P2']
                df_group.drop(['1d_DG_P1', '1d_DG_P2'], axis=1, inplace=True)
                df_group['group_type'] = 'dyad'
            if "triad" in group_name:
                df_group['MG'] = df_group[f"{group_name}_MG_D1"] + df_group[f"{group_name}_MG_D0"]
                df_group['0_D1'] = df_group[f"{group_name}_0_D1"]
                df_group['1d_DG'] = df_group[f"{group_name}_3_D1"] + df_group[f"{group_name}_2_D1_different"] + df_group[f"{group_name}_2_D1_same"] + df_group[f"{group_name}_1_D1"]
                df_group.drop(cols, axis=1, inplace=True)
                df_group['group_type'] = 'triad'
            df_group.insert(0, 'seconds_interaction_window', df_wide_gaze.iloc[df_group.index]['seconds_interaction_window'])
            df_group['group_name'] = group_name
            df_long = pd.concat([df_long, df_group])
        df_long = df_long.rename(columns={"seconds_interaction_window":"seconds"})
        return df_long

    def __merge_df_blink_rate_into_df_long_gaze(self, df_blink_rates: pd.DataFrame, df_long_gaze: pd.DataFrame, group_names: list[str]) -> pd.DataFrame:
        df_blink_rates["TSGroupNTP"] = pd.to_datetime(df_blink_rates["TSGroupNTP"])
        df_long_blinks: pd.DataFrame = pd.DataFrame()
        # iterate all groups in blink_rate df
        for group_name in group_names:
            df_group = df_blink_rates.loc[df_blink_rates['group_name']==group_name]
            start_TS = df_group.iloc[0]['TSGroupNTP']
            df_group = df_group.copy() # to avoid adding a column on a df slice throwing a warning
            df_group['seconds'] = (df_group['TSGroupNTP'] - start_TS).dt.total_seconds()
            # merge gaze and blink rate per group
            df_group_gaze = df_long_gaze[df_long_gaze['group_name'] == group_name]
            df_group_merged = pd.merge(left=df_group, right=df_group_gaze, on='seconds')
            df_long_blinks = pd.concat([df_long_blinks, df_group_merged])
        df_long_blinks['BPM'] = df_long_blinks[['P1_valid_blink_onsets', 'P2_valid_blink_onsets', 'P3_valid_blink_onsets']].mean(axis=1, skipna=True)        
        return df_long_blinks.drop(columns=['group_name_y', 'TSGroupNTP',
                                            'P1_valid_blink_onsets', 'P2_valid_blink_onsets', 
                                            'P3_valid_blink_onsets','P3_valid_blinks']).rename(
                                                columns={'group_name_x':'group_name'})
    
    def __merge_df_blink_durations_into_df_long_gaze(self, df_bli_dur_dyads: pd.DataFrame, df_bli_dur_triads: pd.DataFrame, df_long_gaze: pd.DataFrame, group_names: list[str]) -> pd.DataFrame:        
        df_bli_dur_dyads["TSGroupNTP"] = pd.to_datetime(df_bli_dur_dyads["TSGroupNTP"])
        df_bli_dur_triads["TSGroupNTP"] = pd.to_datetime(df_bli_dur_triads["TSGroupNTP"])
        df_merged_all:pd.DataFrame = pd.DataFrame()        
        for group_name in group_names:
            if "dyad" in group_name:
                df_group = df_bli_dur_dyads.loc[df_bli_dur_dyads['group_name']==group_name]
            else:
                df_group = df_bli_dur_triads.loc[df_bli_dur_triads['group_name']==group_name]
            start_TS = df_group.iloc[0]['TSGroupNTP']
            df_group = df_group.copy() # to avoid adding a column on a df slice throwing a warning
            df_group['seconds'] = (df_group['TSGroupNTP'] - start_TS).dt.total_seconds()
            # merge gaze and blink rate per group
            df_group_gaze = df_long_gaze[df_long_gaze['group_name'] == group_name]
            df_group_merged = pd.merge(left=df_group, right=df_group_gaze, on='seconds')                        
            # Check if there are Nans, and if so, fill with average of column
            if df_group_merged.isna().values.any():
                col_means = df_group_merged.drop(columns=["group_name_x","group_name_y","group_type"]).mean()
                df_group_merged.fillna(col_means, inplace=True)
            df_merged_all = pd.concat([df_merged_all, df_group_merged])
                    
        return df_merged_all.drop(columns=['group_name_y', 'TSGroupNTP', 'P1_durations', 'P2_durations', 'P3_durations']).rename(columns={'group_name_x':'group_name', 'group_avg_blink_duration':'blink_durations'})

    def __merge_df_speaking_gaze_into_df_long_gaze(self, df_speaking: pd.DataFrame, df_long_gaze: pd.DataFrame, group_names: list[str]) -> pd.DataFrame:
        df_merged_all:pd.DataFrame = pd.DataFrame()
        # merge gaze and speaking per group
        for group_name in group_names:
            df_group_speaking_gaze = df_speaking[df_speaking["session"] == group_name]
            df_group_gaze = df_long_gaze[df_long_gaze['group_name'] == group_name]
            # merge gaze and speaking per group
            cols_speaking_gaze = ["seconds","G_OnSpeaker", "No_G_OnSpeaker", "G_SI", "No_G_SI"]
            df_group_speaking_gaze_sub = df_group_speaking_gaze[cols_speaking_gaze]
            df_group_merged = pd.merge(left=df_group_speaking_gaze_sub, right=df_group_gaze, on='seconds')
            df_merged_all = pd.concat([df_merged_all, df_group_merged])
        return df_merged_all

    def __merge_resample_df_TE_into_df_long_gaze(self, sampling:int, df_TE:pd.DataFrame, df_long_gaze:pd.DataFrame, group_names:list[str]) -> pd.DataFrame:
        df_TE['TS_ms'] = pd.to_timedelta(df_TE['seconds'], unit='s')
        df_TE = df_TE.set_index('TS_ms')
        df_TE = df_TE.resample(f'{30}s').mean()
        df_TE['seconds'] = df_TE.index.total_seconds()
        df_merged_all:pd.DataFrame = pd.DataFrame()        
        for group_name in group_names:
            group_cols = df_TE.columns.str.contains(group_name)
            seconds_cols = df_TE.columns.str.contains('seconds')
            cols = group_cols + seconds_cols
            df_group = df_TE[df_TE.columns[cols]]            
            df_group = df_group.copy() # to avoid adding a column on a df slice throwing a warning
            df_group.dropna(inplace=True)
            df_group.rename(columns={f'task_eng_{group_name}':'TE'}, inplace=True)
            # merge gaze and blink rate per group
            df_group_gaze = df_long_gaze[df_long_gaze['group_name'] == group_name]
            df_group_merged = pd.merge(left=df_group, right=df_group_gaze, on='seconds')
            df_merged_all = pd.concat([df_merged_all, df_group_merged])
                    
        return df_merged_all

    def to_csv(self, filename:str):
        self.df_processed.to_csv(os.path.join(self.data_folder_gaze, filename))

if __name__ == "__main__":
    process_30s: bool = False
    sampling: int = 60
    file_name_gaze:str = ""
    file_name_blink_duration_dyads:str = ""
    file_name_blink_duration_triads:str = ""
    file_name_blink_rate:str =""
    file_name_speaking_gaze:str = ""
    file_name_TE:str = ""
    # 30s
    if process_30s and sampling == 30:
        file_name_gaze = f"all_features_{sampling}s_resampled2025-12-20.csv"
        file_name_blink_duration_dyads = f"blink_duration_per_minute_dyads_{sampling}s_2025-12-22.csv"
        file_name_blink_duration_triads = f"blink_duration_per_minute_triads_{sampling}s_2025-12-22.csv"
        file_name_blink_rate = f"blink_per_minute_all_groups_{sampling}s_2025-12-22.csv"
        file_name_speaking_gaze = f"all_groups_interaction_speaking_per_{sampling}s.csv"
        file_name_TE = "all_groups_interaction_task_eng90Hz.csv"
    # 60s
    else:
        file_name_gaze = 'all_features_60s_resampled.csv'
        file_name_blink_duration_dyads = f"blink_duration_per_minute_dyads_{sampling}s.csv"
        file_name_blink_duration_triads = f"blink_duration_per_minute_triads_{sampling}s.csv"
        file_name_blink_rate = f"blink_per_minute_all_groups.csv"
        file_name_speaking_gaze = f"all_groups_interaction_speaking_x_gaze_per_{sampling}s_2026-04-09.csv"
        # data_file_with_TE = '60s_TE_correlation.csv'

    datasetCtrl: MergedDatasetProcessor = MergedDatasetProcessor(sampling=sampling, filename_gaze=file_name_gaze,
                                                                 filename_blink_duration_dyads=file_name_blink_duration_dyads, 
                                                                 filename_blink_duration_triads=file_name_blink_duration_triads,
                                                                 filename_blink_rate=file_name_blink_rate,
                                                                 filename_speaking_gaze=file_name_speaking_gaze,
                                                                 filename_TE=file_name_TE)

    datasetCtrl.to_csv(f"{sampling}s_TE_correlation_{datetime.datetime.now().date()}.csv")