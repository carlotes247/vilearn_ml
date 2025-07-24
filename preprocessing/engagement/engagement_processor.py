import pandas as pd
import numpy as np

class EngagementProcessor:

    col_names = ["task_eng", "conf"]
    path_file_1: str = "data/annotations/dyad_01/group.task engagement.helenrisack.annotation~"
    path_file_2: str = "data/annotations/dyad_01/task engagement.group.carlosgonzalez.annotation~"
    path_groups_info: str = "data/group_durations_all_commas.csv"
    interaction_start: float = 0
    interaction_end: float = 0
    interaction_length: float = 0
    group_name: str = "dyad_01"
    save_df: bool = True
    df_eng_1: pd.DataFrame 
    df_eng_2 : pd.DataFrame 
    df_groups_info: pd.DataFrame


    def __init__(self, path_groups_info:str, path_file_1: str, path_file_2: str, group_name: str):
        self.path_groups_info = path_groups_info
        self.path_file_1 = path_file_1
        self.path_file_2 = path_file_2
        self.group_name = group_name
        self.df_eng_1: pd.DataFrame = pd.read_csv(self.path_file_1, sep=";", names=self.col_names)
        self.df_eng_2 : pd.DataFrame = pd.read_csv(self.path_file_2, sep=";", names=self.col_names)
        self.df_groups_info = pd.read_csv(self.path_groups_info)


    def __avg_eng_files_TS_secs(self, df_1: pd.DataFrame, df_2: pd.DataFrame, freq: float) -> pd.DataFrame:
        print(f"File 1 has {len(df_1)} lines and File 2 has {len(df_2)} lines")
        if len(df_1) != len(df_2):
            print("Engagement files are not of equal size!")
            return pd.DataFrame()
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
        ts_secs: list[float] = [x * (1/freq) for x in range(len(df_avg))]
        df_avg['seconds'] = ts_secs
        return df_avg

    def __get_start_end_interaction(self, df_groups_info: pd.DataFrame, group_name: str) -> tuple[float, float]:
        # get start and end of interaction
        col_mask: pd.Series[bool] = df_groups_info['name'] == group_name
        start = df_groups_info[col_mask]['offset_recording_interaction_start'].values[0]
        length = df_groups_info[col_mask]['duration_interaction'].values[0]
        end = start + length
        return  start, end

    def __slice_interaction_time(self, df: pd.DataFrame, start: float, end: float) -> pd.DataFrame:
        # slice df to only get interaction time
        index_interaction_start = (df['seconds']-start).abs().argsort()[:1]
        index_interaction_end = (df['seconds']-end).abs().argsort()[:1]
        df_return = df.loc[index_interaction_start.values[0]:index_interaction_end.values[0]]
        return df_return

    def process_task_engagement(self, save_to_disk:bool) -> pd.DataFrame:
        df_avg: pd.DataFrame = self.__avg_eng_files_TS_secs(self.df_eng_1, self.df_eng_2, freq=90)
        interaction_start, interaction_end = self.__get_start_end_interaction(self.df_groups_info, group_name=group_name)
        df_interaction = self.__slice_interaction_time(df_avg, interaction_start, interaction_end)
        if save_to_disk:
            df_avg.to_csv(f"data/annotations/{group_name}/task_engagement_avg_all.csv")
            df_interaction.to_csv(f"data/annotations/{group_name}/task_engagement_avg_interaction.csv")

        return pd.DataFrame()

if __name__ == '__main__':
    path_file_1: str = "data/annotations/dyad_01/group.task engagement.helenrisack.annotation~"
    path_file_2: str = "data/annotations/dyad_01/task engagement.group.carlosgonzalez.annotation~"
    path_groups_info: str = "data/group_durations_all_commas.csv"
    group_name: str = "dyad_01"

    processor: EngagementProcessor = EngagementProcessor(path_groups_info, path_file_1, path_file_2, group_name)
    processor.process_task_engagement(save_to_disk=True)