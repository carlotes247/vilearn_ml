from engagement_processor import EngagementProcessor
import os
import pandas as pd
import numpy as np

class EngagementsManager:

    engagements_list: list[EngagementProcessor] = []
    data_path: str = "data/annotations"
    path_groups_info: str = "data/group_durations_all_commas.csv"
    folders: list[str]

    def __init__(self, save_to_disk: bool) -> None:
        self.load_engagements(save_to_disk=save_to_disk)
        pass

    def load_engagements(self, save_to_disk: bool):
        self.folders = os.listdir(self.data_path)
        self.folders = [folder for folder in self.folders if not os.path.isfile(f"{self.data_path}/{folder}")]
        for folder in self.folders:
            folder_path: str = f"{self.data_path}/{folder}"
            eng_processor = EngagementProcessor(path_groups_info=self.path_groups_info, path_folder=folder_path, group_name=folder.replace("recording_", ""))
            eng_processor.process_task_engagement(save_to_disk=save_to_disk)
            self.engagements_list.append(eng_processor)
        pass

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
        # drop seconds to not skew mean and get number of groups that still have values
        df_only_values = df_combined.drop('seconds', axis=1)
        df_combined['average_value'] = df_only_values.mean(axis=1)
        df_combined['groups'] = df_only_values.count(axis=1)
        if save_to_disk:
            df_combined.to_csv("data/annotations/all_groups_task_eng.csv")    
        print("done!!")
                

if __name__ == "__main__":
    mngr_aux: EngagementsManager = EngagementsManager(False)
    mngr_aux.avg_engagements(False)
    print("done")