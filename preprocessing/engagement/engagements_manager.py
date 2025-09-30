if __name__ != "__main__":
    from preprocessing.engagement.engagement_processor import EngagementProcessor
else:
    from engagement_processor import EngagementProcessor
import os
import pandas as pd
import numpy as np

class EngagementsManager:

    engagements_list: list[EngagementProcessor] = []
    data_path: str = "data/annotations"
    path_groups_info: str = "data/group_durations_all_commas.csv"
    filename_all_groups_rec_time: str = "all_groups_task_eng90Hz.csv"
    filename_dyads_rec_time: str ="dyads_task_eng90Hz.csv"                    
    filename_triads_rec_time: str ="triads_task_eng90Hz.csv"
    filename_all_groups_interaction_time: str ="all_groups_interaction_task_eng90Hz.csv"
    filename_dyads_interaction_time: str ="dyads_interaction_task_eng90Hz.csv"                    
    filename_triads_interaction_time: str ="triads_interaction_task_eng90Hz.csv"
    loaded_from_disk: bool
    folders: list[str]
    df_avg_eng_all: pd.DataFrame
    df_avg_eng_dyads: pd.DataFrame
    df_avg_eng_triads: pd.DataFrame
    df_avg_eng_all_interaction: pd.DataFrame
    df_avg_eng_dyads_interaction: pd.DataFrame
    df_avg_eng_triads_interaction: pd.DataFrame

    def __init__(self, save_to_disk: bool, load_from_disk:bool, floor_level: bool) -> None:
        self.load_engagements(save_to_disk=save_to_disk, load_from_disk=load_from_disk)
        self.avg_engagements(save_to_disk=save_to_disk)
        # drop columns that are not in floor level if true
        if floor_level:
            self.__drop_floorlevel()
            


    def load_engagements(self, save_to_disk: bool, load_from_disk: bool):
        self.loaded_from_disk = False
        if load_from_disk:            
            self.df_avg_eng_all = pd.read_csv(os.path.join(os.getcwd(), self.data_path, self.filename_all_groups_rec_time))
            self.df_avg_eng_dyads = pd.read_csv(os.path.join(os.getcwd(), self.data_path, self.filename_dyads_rec_time))
            self.df_avg_eng_triads = pd.read_csv(os.path.join(os.getcwd(), self.data_path, self.filename_triads_rec_time))
            self.df_avg_eng_all_interaction = pd.read_csv(os.path.join(os.getcwd(), self.data_path, self.filename_all_groups_interaction_time))
            self.df_avg_eng_dyads_interaction = pd.read_csv(os.path.join(os.getcwd(), self.data_path, self.filename_dyads_interaction_time))
            self.df_avg_eng_triads_interaction = pd.read_csv(os.path.join(os.getcwd(), self.data_path, self.filename_triads_interaction_time))
            self.loaded_from_disk = True
        else:
            self.folders = os.listdir(self.data_path)
            self.folders = [folder for folder in self.folders if not os.path.isfile(f"{self.data_path}/{folder}")]
            for folder in self.folders:
                folder_path: str = f"{self.data_path}/{folder}"
                eng_processor = EngagementProcessor(path_groups_info=self.path_groups_info, path_folder=folder_path, group_name=folder.replace("recording_", ""))
                eng_processor.process_task_engagement(save_to_disk=save_to_disk)
                self.engagements_list.append(eng_processor)       

    def __slice_process_avg_df(self, df_combined: pd.DataFrame, keyword_cols: str):
        cols = df_combined.columns[df_combined.columns.str.contains(keyword_cols)]
        df_eng: pd.DataFrame = pd.DataFrame(df_combined[cols])
        avg_eng = df_eng.mean(axis=1)
        std_eng = df_eng.std(axis=1)
        count_groups = df_eng.count(axis=1)
        df_eng['groups'] = count_groups
        df_eng['avg_task_eng'] = avg_eng
        df_eng['std_avg_task_eng'] = std_eng
        df_eng['seconds'] = df_combined['seconds']     
        return df_eng

    def avg_engagements(self, save_to_disk: bool):
        if len(self.engagements_list) == 0 or self.loaded_from_disk:
            return
        # remove processors that couldn't finish processing
        finished_list: list[EngagementProcessor] = [processor for processor in self.engagements_list if processor.finished_processing]
        # Merge all dataframes on 'seconds' using outer join
        df_combined: pd.DataFrame = pd.DataFrame({'seconds': pd.concat([processor.df_avg_all['seconds'] for processor in finished_list]).unique()})        
        df_combined.sort_values('seconds', inplace=True)
        df_combined_interaction: pd.DataFrame = pd.DataFrame({'seconds_interaction': pd.concat([processor.df_avg_interaction['seconds_interaction'] for processor in finished_list]).unique()})        
        df_combined_interaction.sort_values('seconds_interaction', inplace=True)
        # Add each value series to the merged DataFrame
        for i, processor in enumerate(finished_list):
            df_combined = df_combined.merge(processor.df_avg_all, on='seconds', how='left', suffixes=('', f'_{processor.group_name}'))            
            df_combined_interaction = df_combined_interaction.merge(processor.df_avg_interaction, on='seconds_interaction', how='left', suffixes=('', f'_{processor.group_name}'))            
        df_combined.rename(columns={'task_eng' : 'task_eng_dyad_01', 'std' : 'std_dyad_01'}, inplace=True)
        df_combined_interaction.rename(columns={'task_eng' : 'task_eng_dyad_01', 'seconds' : 'seconds_dyad_01', 'std' : 'std_dyad_01'}, inplace=True)        
        df_combined_interaction.rename(columns={'seconds_interaction' : 'seconds'}, inplace=True)
        # dataframes for dyads and triads
        # dyads
        df_eng_dyads: pd.DataFrame = self.__slice_process_avg_df(df_combined=df_combined, keyword_cols='task_eng_dyad')
        df_eng_dyads_interaction: pd.DataFrame = self.__slice_process_avg_df(df_combined=df_combined_interaction, keyword_cols='task_eng_dyad')
        # triads
        df_eng_triads: pd.DataFrame = self.__slice_process_avg_df(df_combined=df_combined, keyword_cols='task_eng_triad')
        df_eng_triads_interaction: pd.DataFrame = self.__slice_process_avg_df(df_combined=df_combined_interaction, keyword_cols='task_eng_triad')
        # both
        df_combined = self.__slice_process_avg_df(df_combined=df_combined, keyword_cols='task_eng')
        df_combined_interaction = self.__slice_process_avg_df(df_combined=df_combined_interaction, keyword_cols='task_eng')
        # TODO: include a method in the future to keep the std from each group (which now is lost the average std)
        self.df_avg_eng_all = df_combined
        self.df_avg_eng_dyads = df_eng_dyads
        self.df_avg_eng_triads = df_eng_triads
        self.df_avg_eng_all_interaction = df_combined_interaction
        self.df_avg_eng_dyads_interaction = df_eng_dyads_interaction
        self.df_avg_eng_triads_interaction = df_eng_triads_interaction
        if save_to_disk:
            df_combined.to_csv("data/annotations/all_groups_task_eng90Hz.csv")
            df_eng_dyads.to_csv("data/annotations/dyads_task_eng90Hz.csv")                    
            df_eng_triads.to_csv("data/annotations/triads_task_eng90Hz.csv")
            df_combined_interaction.to_csv("data/annotations/all_groups_interaction_task_eng90Hz.csv")
            df_eng_dyads_interaction.to_csv("data/annotations/dyads_interaction_task_eng90Hz.csv")                    
            df_eng_triads_interaction.to_csv("data/annotations/triads_interaction_task_eng90Hz.csv")

    def __drop_floorlevel(self):
        df_details_floorlevel = pd.read_csv(os.path.join(os.getcwd(), 'data', 'group_names_with_time_floorlevel.csv'), sep=';')
        groups_floorlevel = df_details_floorlevel['Group_Name'].to_list()
        self.df_avg_eng_all = self.__drop_cols_not_in(self.df_avg_eng_all, groups_floorlevel)
        self.df_avg_eng_dyads = self.__drop_cols_not_in(self.df_avg_eng_dyads, groups_floorlevel)
        self.df_avg_eng_triads = self.__drop_cols_not_in(self.df_avg_eng_triads, groups_floorlevel)
        self.df_avg_eng_all_interaction = self.__drop_cols_not_in(self.df_avg_eng_all_interaction, groups_floorlevel)
        self.df_avg_eng_dyads_interaction = self.__drop_cols_not_in(self.df_avg_eng_dyads_interaction, groups_floorlevel)
        self.df_avg_eng_triads_interaction = self.__drop_cols_not_in(self.df_avg_eng_triads_interaction, groups_floorlevel)
    
    def __drop_cols_not_in(self, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        cols.append(df.columns[df.columns.str.contains('seconds')][0])
        cols_to_keep = df.columns.str.contains("|".join(cols))
        return df[df.columns[cols_to_keep]]
        
if __name__ == "__main__":    
    mngr_aux: EngagementsManager = EngagementsManager(save_to_disk=False, load_from_disk=True, floor_level=True)
    print("done")