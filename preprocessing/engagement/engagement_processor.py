import pandas as pd
import numpy as np
import os.path

class EngagementProcessor:

    col_names = ["task_eng", "conf"]
    data_path: str = "data/annotations"
    path_groups_info: str = "data/group_durations_all_commas.csv"
    filename_1_90Hz: str = "group.task engagement.helenrisack.annotation~"
    filename_2_90Hz: str = "task engagement.group.carlosgonzalez.annotation~"
    filename_1_60Hz: str = "task engagement60Hz.group.helenrisack.annotation~"
    filename_2_60Hz: str = "task engagement60Hz.group.carlosgonzalez.annotation~"
    filename_1: str 
    filename_2: str 
    is_file_1_90Hz: bool = False
    is_file_2_90Hz: bool = False
    path_file_1: str
    path_file_2: str 
    interaction_start: float = 0
    interaction_end: float = 0
    interaction_length: float = 0
    freq: int = 0
    group_name: str
    save_df: bool = False
    df_eng_1: pd.DataFrame 
    df_eng_2 : pd.DataFrame 
    df_groups_info: pd.DataFrame
    df_avg_all: pd.DataFrame = pd.DataFrame()
    df_avg_interaction: pd.DataFrame = pd.DataFrame()
    finished_processing: bool = False


    def __init__(self, path_groups_info:str, path_folder: str, group_name: str):
        self.path_groups_info = path_groups_info
        self.group_name = group_name        
        self.df_groups_info = pd.read_csv(self.path_groups_info)
        self.data_path = path_folder
        
        self.freq = self.__get_annotation_freq(self.df_groups_info, self.group_name)
        if self.freq == 90:
            self.filename_1 = self.filename_1_90Hz
            self.filename_2 = self.filename_2_90Hz
            self.is_file_1_90Hz = True; self.is_file_2_90Hz = True
        elif self.freq == 60:
            if os.path.isfile(f"{self.data_path}/{self.filename_1_90Hz}"):
                self.filename_1 = self.filename_1_90Hz; self.is_file_1_90Hz = True
            else:
                self.filename_1 = self.filename_1_60Hz
            if os.path.isfile("{self.data_path}/{self.filename_2_90Hz}"):
                self.filename_2 = self.filename_2_90Hz; self.is_file_2_90Hz = True
            else:
                self.filename_2 = self.filename_2_60Hz
            if self.is_file_1_90Hz and self.is_file_2_90Hz: 
                self.freq = 90
        
        self.path_file_1: str = f"{self.data_path}/{self.filename_1}"
        self.path_file_2: str = f"{self.data_path}/{self.filename_2}"

        self.df_eng_1: pd.DataFrame = pd.read_csv(self.path_file_1, sep=";", names=self.col_names)
        self.df_eng_2 : pd.DataFrame = pd.read_csv(self.path_file_2, sep=";", names=self.col_names)
    
    def __avg_eng_files_TS_secs(self, df_1: pd.DataFrame, df_2: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
        print(f"{self.group_name}: File 1 has {len(df_1)} lines and File 2 has {len(df_2)} lines")
        diff_dfs: int = len(df_1) - len(df_2)
        if diff_dfs != 0:
            if abs(diff_dfs) < 10: 
                if diff_dfs > 0: 
                    df_1.drop(df_1.tail(diff_dfs).index, inplace=True)
                else:
                    df_2.drop(df_2.tail(abs(diff_dfs)).index, inplace=True)
            else:
                print(f"{self.group_name}: Engagement files are not of equal size! Aborting processing")
                return pd.DataFrame(), False
        # drop conf column and make sure both dataframes are numeric
        cols_to_drop: list[str] = ['conf', 'seconds', 'std']
        df_1.drop(cols_to_drop, axis=1, errors='ignore', inplace=True)
        df_2.drop(cols_to_drop, axis=1, errors='ignore', inplace=True)
        # if "conf" in df_1.columns: 
        #     df_1.drop("conf", axis=1, inplace=True)
        df_1['task_eng'] = pd.to_numeric(df_1['task_eng'], errors='coerce')
        # if "conf" in df_2.columns: 
            # df_2.drop("conf", axis=1, inplace=True)
        df_2['task_eng'] = pd.to_numeric(df_2['task_eng'], errors='coerce')
        # clean nans
        df_merged: pd.DataFrame = pd.concat([df_1, df_2], axis=1)
        df_merged.replace('-nan(ind)', np.nan, inplace=True)
        df_merged = df_merged.fillna(0)
        df_avg: pd.DataFrame = pd.DataFrame(df_merged.mean(axis=1), columns=['task_eng'])
        df_avg['std'] = df_merged.std(axis=1)
        # add column for seconds per frame
        ts_secs: list[float] = [x * (1/self.freq) for x in range(len(df_avg))]
        df_avg['seconds'] = ts_secs
        return df_avg, True

    def __get_start_end_interaction(self, df_groups_info: pd.DataFrame, group_name: str) -> tuple[float, float]:
        # get start and end of interaction
        col_mask: pd.Series[bool] = df_groups_info['name'] == group_name
        start = df_groups_info[col_mask]['offset_recording_interaction_start'].values[0]
        length = df_groups_info[col_mask]['duration_interaction'].values[0]
        end = start + length
        return  start, end
    
    def __get_annotation_freq(self, df_groups_info: pd.DataFrame, group_name: str) -> int:
        col_mask: pd.Series[bool] = df_groups_info['name'] == group_name
        freq: int = df_groups_info[col_mask]['freq_task_eng'].values[0]
        return freq

    def __slice_interaction_time(self, df: pd.DataFrame, start: float, end: float) -> pd.DataFrame:
        # slice df to only get interaction time
        index_interaction_start = (df['seconds']-start).abs().argsort()[:1]
        index_interaction_end = (df['seconds']-end).abs().argsort()[:1]
        df_return: pd.DataFrame = pd.DataFrame(df.loc[index_interaction_start.values[0]:index_interaction_end.values[0]])
        # add column for seconds per frame for interaction time
        ts_secs: list[float] = [x * (1/self.freq) for x in range(len(df_return))]
        df_return['seconds_interaction'] = ts_secs
        return df_return
    
    def __interpolate_eng(self, df: pd.DataFrame, freq_original: float, freq_target: float, force_numeric: bool = False) -> pd.DataFrame:
        # Cannot interpolate down
        if freq_original > freq_target:
            print(f"Error: you are trying to interpolate down {freq_original} to {freq_target}. Aborting interpolation...")
            return pd.DataFrame()
        if force_numeric:
            # drop conf column and make sure dataframe is numeric
            df.drop("conf", axis=1, inplace=True)
            df['task_eng'] = pd.to_numeric(df['task_eng'], errors='coerce')
            df.replace('-nan(ind)', np.nan, inplace=True)
            df = df.fillna(0)
        if 'std' not in df.columns:
            std = df.std(axis=1)
            df['std'] = std
        if 'seconds' not in df.columns:
            old_seconds: list[float] = [x * (1/freq_original) for x in range(len(df))]
            df['seconds'] = old_seconds    
        # new timesteps        
        new_seconds = np.arange(df['seconds'].min(), df['seconds'].max(), 1/freq_target)
        # interpolate
        interpolated_eng = np.interp(new_seconds, df['seconds'], df['task_eng'])
        interpolated_std = np.interp(new_seconds, df['seconds'], df['std'])
        df_result = pd.DataFrame({'task_eng': interpolated_eng , 'std' : interpolated_std, 'seconds': new_seconds})
        return df_result

    def process_task_engagement(self, save_to_disk:bool) -> tuple[pd.DataFrame, pd.DataFrame]:
        if self.freq < 90:
            if not self.is_file_1_90Hz:
                self.df_eng_1 = self.__interpolate_eng(self.df_eng_1, self.freq, 90, force_numeric=True)
            if not self.is_file_2_90Hz:
                self.df_eng_2 = self.__interpolate_eng(self.df_eng_2, self.freq, 90, force_numeric=True)
        df_avg, result = self.__avg_eng_files_TS_secs(self.df_eng_1, self.df_eng_2)
        if not result:
            return pd.DataFrame(), pd.DataFrame()
        if self.freq < 90:
            df_avg = self.__interpolate_eng(df_avg, self.freq, 90)
            self.freq = 90
        interaction_start, interaction_end = self.__get_start_end_interaction(self.df_groups_info, group_name=self.group_name)
        df_interaction = self.__slice_interaction_time(df_avg, interaction_start, interaction_end)
        if save_to_disk:
            df_avg.to_csv(f"data/annotations/recording_{self.group_name}/task_engagement{self.freq}Hz_avg_all.csv")
            df_interaction.to_csv(f"data/annotations/recording_{self.group_name}/task_engagement{self.freq}Hz_avg_interaction.csv")
        self.df_avg_all = df_avg
        self.df_avg_interaction = df_interaction
        self.finished_processing = True
        return df_avg, df_interaction

if __name__ == '__main__':
    path_file_1: str = "data/annotations/dyad_01/group.task engagement.helenrisack.annotation~"
    path_file_2: str = "data/annotations/dyad_01/task engagement.group.carlosgonzalez.annotation~"
    path_groups_info: str = "data/group_durations_all_commas.csv"
    folder_path = "data/annotations/dyad_01/"
    group_name: str = "dyad_01"

    processor: EngagementProcessor = EngagementProcessor(path_groups_info=path_groups_info, path_folder=folder_path, group_name=group_name)
    processor.process_task_engagement(save_to_disk=True)