import pandas as pd
import numpy as np

class EngagementProcessor:

    col_names = ["task_eng", "conf"]
    data_path: str = "data/annotations"
    path_groups_info: str = "data/group_durations_all_commas.csv"
    filename_1: str 
    filename_2: str 
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
            self.filename_1 = "group.task engagement.helenrisack.annotation~"
            self.filename_2 = "task engagement.group.carlosgonzalez.annotation~"
        elif self.freq == 60:
            self.filename_1 = "task engagement60Hz.group.helenrisack.annotation~"
            self.filename_2 = "task engagement60Hz.group.carlosgonzalez.annotation~"
        
        self.path_file_1: str = f"{self.data_path}/{self.filename_1}"
        self.path_file_2: str = f"{self.data_path}/{self.filename_2}"

        self.df_eng_1: pd.DataFrame = pd.read_csv(self.path_file_1, sep=";", names=self.col_names)
        self.df_eng_2 : pd.DataFrame = pd.read_csv(self.path_file_2, sep=";", names=self.col_names)

    def __avg_eng_files_TS_secs(self, df_1: pd.DataFrame, df_2: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
        print(f"{self.group_name}: File 1 has {len(df_1)} lines and File 2 has {len(df_2)} lines")
        if len(df_1) != len(df_2):
            print(f"{self.group_name}: Engagement files are not of equal size! Aborting processing")
            return pd.DataFrame(), False
        # drop conf column and make sure both dataframes are numeric
        df_1.drop("conf", axis=1, inplace=True)
        df_1['task_eng'] = pd.to_numeric(df_1['task_eng'], errors='coerce')
        df_2.drop("conf", axis=1, inplace=True)
        df_2['task_eng'] = pd.to_numeric(df_2['task_eng'], errors='coerce')
        # clean nans
        df_merged: pd.DataFrame = pd.concat([df_1, df_2], axis=1)
        df_merged.replace('-nan(ind)', np.nan, inplace=True)
        df_merged = df_merged.fillna(0)
        df_avg: pd.DataFrame = pd.DataFrame(df_merged.mean(axis=1), columns=['task_eng'])
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
        df_return = df.loc[index_interaction_start.values[0]:index_interaction_end.values[0]]
        return df_return
    
    def __interpolate_eng(self, df: pd.DataFrame, freq_original: float, freq_target: float) -> pd.DataFrame:
        # Cannot interpolate down
        if freq_original > freq_target:
            return pd.DataFrame()
        # new timesteps        
        new_seconds = np.arange(df['seconds'].min(), df['seconds'].max(), 1/freq_target)
        # interpolate
        interpolated_values = np.interp(new_seconds, df['seconds'], df['task_eng'])
        df_result = pd.DataFrame({'task_eng': interpolated_values, 'seconds': new_seconds})
        return df_result

    def process_task_engagement(self, save_to_disk:bool) -> tuple[pd.DataFrame, pd.DataFrame]:
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