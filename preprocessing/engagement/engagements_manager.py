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
    folders: list[str]
    df_avg_eng_all: pd.DataFrame
    df_avg_eng_dyads: pd.DataFrame
    df_avg_eng_triads: pd.DataFrame

    def __init__(self, save_to_disk: bool) -> None:
        self.load_engagements(save_to_disk=save_to_disk)
        self.avg_engagements(save_to_disk=save_to_disk)

    def load_engagements(self, save_to_disk: bool):
        self.folders = os.listdir(self.data_path)
        self.folders = [folder for folder in self.folders if not os.path.isfile(f"{self.data_path}/{folder}")]
        for folder in self.folders:
            folder_path: str = f"{self.data_path}/{folder}"
            eng_processor = EngagementProcessor(path_groups_info=self.path_groups_info, path_folder=folder_path, group_name=folder.replace("recording_", ""))
            eng_processor.process_task_engagement(save_to_disk=save_to_disk)
            self.engagements_list.append(eng_processor)        

    def avg_engagements(self, save_to_disk: bool):
        if len(self.engagements_list) == 0:
            return
        # remove processors that couldn't finish processing
        finished_list: list[EngagementProcessor] = [processor for processor in self.engagements_list if processor.finished_processing]
        # Merge all dataframes on 'seconds' using outer join
        df_combined: pd.DataFrame = pd.DataFrame({'seconds': pd.concat([processor.df_avg_all['seconds'] for processor in finished_list]).unique()})        
        df_combined.sort_values('seconds', inplace=True)
        # Add each value series to the merged DataFrame
        for i, processor in enumerate(finished_list):
            df_combined = df_combined.merge(processor.df_avg_all, on='seconds', how='left', suffixes=('', f'_{processor.group_name}'))            
        df_combined.rename(columns={'task_eng' : 'task_eng_dyad_01'}, inplace=True)
        # dataframes for dyads and triads
        cols_dyads = df_combined.columns[df_combined.columns.str.contains('dyad')]
        df_eng_dyads: pd.DataFrame = df_combined[cols_dyads]
        avg_eng_dyads = df_eng_dyads.mean(axis=1)
        df_eng_dyads['groups'] = df_eng_dyads.count(axis=1)
        df_eng_dyads['avg_task_eng'] = avg_eng_dyads
        df_eng_dyads['seconds'] = df_combined['seconds']
        cols_triads = df_combined.columns[df_combined.columns.str.contains('triad')]
        df_eng_triads: pd.DataFrame = df_combined[cols_triads]
        avg_eng_triads = df_eng_triads.mean(axis=1)
        df_eng_triads['groups'] = df_eng_triads.count(axis=1)
        df_eng_triads['avg_task_eng'] = avg_eng_triads
        df_eng_triads['seconds'] = df_combined['seconds']
        # drop seconds to not skew mean and get number of groups that still have values
        df_only_values = df_combined.drop('seconds', axis=1)
        df_combined['average_value'] = df_only_values.mean(axis=1)
        df_combined['groups'] = df_only_values.count(axis=1)
        self.df_avg_eng_all = df_combined
        self.df_avg_eng_dyads = df_eng_dyads
        self.df_avg_eng_triads = df_eng_triads
        if save_to_disk:
            df_combined.to_csv("data/annotations/all_groups_task_eng.csv")                    

if __name__ == "__main__":    
    mngr_aux: EngagementsManager = EngagementsManager(False)
    print("done")