import pandas as pd
import numpy as np
import os.path

class EngagementProcessor:

    col_names = ["task_eng", "conf"]
    data_path: str = "data/annotations"
    path_groups_info: str = "data/group_durations_all_commas.csv"
    TE_2pass_exists: bool = False
    filename_1_90Hz: str = "group.task engagement.helenrisack.annotation~"
    filename_2_90Hz: str = "task engagement.group.carlosgonzalez.annotation~"
    filename_1_60Hz: str = "task engagement60Hz.group.helenrisack.annotation~"
    filename_2_60Hz: str = "task engagement60Hz.group.carlosgonzalez.annotation~"
    filename_1_2pass: str = "group.task engagement 2pass.helenrisack.annotation~"
    filename_1: str 
    filename_2: str 
    is_file_1_90Hz: bool = False
    is_file_2_90Hz: bool = False
    path_file_1: str
    path_file_1_2pass: str
    path_file_2: str
    interaction_start: float = 0
    interaction_end: float = 0
    interaction_length: float = 0
    freq: int = 0
    group_name: str
    save_df: bool = False
    df_eng_1: pd.DataFrame
    df_eng_1_2pass: pd.DataFrame
    df_eng_2: pd.DataFrame
    df_eng_1_interaction: pd.DataFrame
    df_eng_1_2pass_interaction: pd.DataFrame
    df_eng_2_interaction: pd.DataFrame
    df_eng_discretised_anno1: pd.DataFrame = pd.DataFrame()
    df_eng_discretised_anno1_2pass: pd.DataFrame = pd.DataFrame()
    df_eng_discretised_anno2: pd.DataFrame = pd.DataFrame()
    df_eng_discretised_anno1_interaction: pd.DataFrame = pd.DataFrame()
    df_eng_discretised_anno1_2pass_interaction: pd.DataFrame = pd.DataFrame()
    df_eng_discretised_anno2_interaction: pd.DataFrame = pd.DataFrame()
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
            if os.path.isfile(f"{self.data_path}/{self.filename_2_90Hz}"):
                self.filename_2 = self.filename_2_90Hz; self.is_file_2_90Hz = True
            else:
                self.filename_2 = self.filename_2_60Hz
            if self.is_file_1_90Hz and self.is_file_2_90Hz: 
                self.freq = 90
        
        self.path_file_1: str = f"{self.data_path}/{self.filename_1}"
        self.path_file_1_2pass: str = f"{self.data_path}/{self.filename_1_2pass}"
        self.path_file_2: str = f"{self.data_path}/{self.filename_2}"

        self.df_eng_1: pd.DataFrame = pd.read_csv(self.path_file_1, sep=";", names=self.col_names)

        self.df_eng_2 : pd.DataFrame = pd.read_csv(self.path_file_2, sep=";", names=self.col_names)
        #if the current group has a second pass annotation of TE, than load it.
        if os.path.isfile(self.path_file_1_2pass):
            self.df_eng_1_2pass: pd.DataFrame = pd.read_csv(self.path_file_1_2pass, sep=";", names=self.col_names)
            self.TE_2pass_exists = True

    
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

    def __populate_individual_engagement_df_with_interaction_time(self):
        interaction_start, interaction_end = self.__get_start_end_interaction(self.df_groups_info,
                                                                              group_name=self.group_name)
        self.df_eng_1['seconds'] = [x * (1 / self.freq) for x in range(len(self.df_eng_1))]
        self.df_eng_2['seconds'] = [x * (1 / self.freq) for x in range(len(self.df_eng_2))]

        self.df_eng_1_interaction = pd.DataFrame(self.__slice_interaction_time(self.df_eng_1, interaction_start, interaction_end))
        self.df_eng_2_interaction = pd.DataFrame(self.__slice_interaction_time(self.df_eng_2, interaction_start, interaction_end))

        if self.TE_2pass_exists:
            self.df_eng_1_2pass['seconds'] = [x * (1 / self.freq) for x in range(len(self.df_eng_1_2pass))]
            self.df_eng_1_2pass_interaction = pd.DataFrame(
                self.__slice_interaction_time(self.df_eng_1_2pass, interaction_start, interaction_end))

    def process_task_engagement(self, save_to_disk:bool) -> tuple[pd.DataFrame, pd.DataFrame]:
        if self.freq < 90:
            if not self.is_file_1_90Hz:
                self.df_eng_1 = self.__interpolate_eng(self.df_eng_1, self.freq, 90, force_numeric=True)
            if not self.is_file_2_90Hz:
                self.df_eng_2 = self.__interpolate_eng(self.df_eng_2, self.freq, 90, force_numeric=True)
            # by here the frequency will always be 90Hz because we are interpolating up
            self.freq = 90
        df_avg, result = self.__avg_eng_files_TS_secs(self.df_eng_1, self.df_eng_2)
        if not result:
            return pd.DataFrame(), pd.DataFrame()
        # Currently this will never be called because we always interpolate up to 90
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
        self.__populate_individual_engagement_df_with_interaction_time()
        self.finished_processing = True
        return df_avg, df_interaction

    def process_discretise_task_engagement(self, process_individual_TE: bool = True, save_to_disk: bool = False,
                                           testing_10_bins = False, two_bins_for_two_anno = False, return_2pass = True):
        # discretise first(Helen) and second(Laura) files
        bins_list = [0, .33, .66, 1] #3 windows of equal sizes, from 0 to 1
        bins_list_10 = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1] #10 windows of equal sizes, from 0 to 1 to check the data distribution

        # by using the 10 windows for Laura's annotation, we can see that there aren't a lot of values higher than .7-.8 for some group. so here we try with different top values and then split the window in 3 equal sizes.
        #bins_list_ann_2 = [0, .25, .5, 1] #[using .75 as top val] 3 windows for Laura's annotations as she didn't use too much the vals from .7 onwards
        bins_list_ann_2 = [0, .2, .4, 1] #[using .7 as top val] 3 windows for Laura's annotations as she didn't use too much the vals from .7 onwards

        labels = ['low', 'mid', 'high']
        labels_10 = ['0-.1', '.1-.2', '.2-.3', '.3-.4', '.4-.5', '.5-.6',
                     '.6-.7', '.7-.8', '.8-.9', '.9-1']

        # labels_ann_2 = ['low_.25', 'mid_.5', 'high_1']

        if process_individual_TE:

            df_eng_1 = pd.DataFrame(self.__replace_TE_strings_with_nan(self.df_eng_1))
            df_eng_2 = pd.DataFrame(self.__replace_TE_strings_with_nan(self.df_eng_2))

            interaction_start, interaction_end = self.__get_start_end_interaction(self.df_groups_info,
                                                                                  group_name=self.group_name)
            df_eng_1['seconds']= [x * (1 / self.freq) for x in range(len(df_eng_1))]
            df_eng_2['seconds']= [x * (1 / self.freq) for x in range(len(df_eng_2))]
            if self.TE_2pass_exists:
                df_eng_1_2pass = pd.DataFrame(self.__replace_TE_strings_with_nan(self.df_eng_1_2pass))
                df_eng_1_2pass['seconds'] = [x * (1 / self.freq) for x in range(len(df_eng_1_2pass))]

            if testing_10_bins:
                df_eng_1['task_eng'] = pd.cut(df_eng_1['task_eng'], bins=bins_list_10, labels=labels_10, include_lowest=True)
                df_eng_2['task_eng'] = pd.cut(df_eng_2['task_eng'], bins=bins_list_10, labels=labels_10, include_lowest=True)
                if self.TE_2pass_exists: df_eng_1_2pass['task_eng'] = pd.cut(df_eng_1_2pass['task_eng'], bins=bins_list_10, labels=labels_10, include_lowest=True)
            elif two_bins_for_two_anno:
                df_eng_1['task_eng'] = pd.cut(df_eng_1['task_eng'], bins=bins_list, labels=labels,
                                              include_lowest=True)
                df_eng_2['task_eng'] = pd.cut(df_eng_2['task_eng'], bins=bins_list_ann_2, labels=labels,
                                              include_lowest=True)
                if self.TE_2pass_exists:
                    df_eng_1_2pass['task_eng'] = pd.cut(df_eng_1_2pass['task_eng'], bins=bins_list, labels=labels,
                                                  include_lowest=True)

            else:
                df_eng_1['task_eng'] = pd.cut(df_eng_1['task_eng'], bins=bins_list, labels=labels, include_lowest=True)
                df_eng_2['task_eng'] = pd.cut(df_eng_2['task_eng'], bins=bins_list, labels=labels, include_lowest=True)
                if self.TE_2pass_exists: df_eng_1_2pass['task_eng'] = pd.cut(df_eng_1_2pass['task_eng'], bins=bins_list, labels=labels, include_lowest=True)


            df_eng1_interaction = self.__slice_interaction_time(df_eng_1, interaction_start, interaction_end)
            df_eng2_interaction = self.__slice_interaction_time(df_eng_2, interaction_start, interaction_end)
            if self.TE_2pass_exists: df_eng1_2pass_interaction = self.__slice_interaction_time(df_eng_1_2pass, interaction_start, interaction_end)

            if save_to_disk:
                df_eng1_interaction.to_csv(
                    f"../../data/annotations/recording_{self.group_name}/task_engagement_anno1_discretised{self.freq}Hz.csv")
                df_eng2_interaction.to_csv(
                    f"../../data/annotations/recording_{self.group_name}/task_engagement_anno2_discretised{self.freq}Hz.csv")
                if self.TE_2pass_exists: df_eng1_2pass_interaction.to_csv(
                    f"../../data/annotations/recording_{self.group_name}/task_engagement_anno1_2pass_discretised{self.freq}Hz.csv")

            self.df_eng_discretised_anno1_interaction = df_eng1_interaction
            self.df_eng_discretised_anno1 = df_eng_1
            self.df_eng_discretised_anno2_interaction = df_eng2_interaction
            self.df_eng_discretised_anno2 = df_eng_2
            if self.TE_2pass_exists:
                self.df_eng_discretised_anno1_2pass_interaction = df_eng1_2pass_interaction
                self.df_eng_discretised_anno1_2pass = df_eng_1_2pass
            if return_2pass and self.TE_2pass_exists:
                return df_eng1_2pass_interaction, df_eng2_interaction
            else:
                return df_eng1_interaction, df_eng2_interaction

        # print ('Done')
        # else:
        # TODO: check if the df_avg is created already, or if I need to call a fuction firstto create it and then to discretise it.
        # pd.cut(self.df_avg_interaction, bins=bins_list, labels=labels, include_lowest=True)


    def __replace_TE_strings_with_nan(self, df: pd.DataFrame):
        df.replace(to_replace=r'[^.0-9]', value=np.nan, regex=True, inplace=True)
        df['task_eng'] = pd.to_numeric(df['task_eng'])
        return df


if __name__ == '__main__':
    path_file_1: str = "data/annotations/dyad_01/group.task engagement.helenrisack.annotation~"
    path_file_2: str = "data/annotations/dyad_01/task engagement.group.carlosgonzalez.annotation~"
    path_groups_info: str = "../../data/group_durations_all_commas.csv"
    folder_path = "../../data/annotations/recording_dyad_01"
    group_name: str = "dyad_01"

    processor: EngagementProcessor = EngagementProcessor(path_groups_info=path_groups_info, path_folder=folder_path, group_name=group_name)
    processor.process_discretise_task_engagement(save_to_disk=False)
    # processor.process_task_engagement(save_to_disk=True)