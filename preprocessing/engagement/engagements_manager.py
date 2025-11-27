if __name__ != "__main__":
    from preprocessing.engagement.engagement_processor import EngagementProcessor
else:
    from engagement_processor import EngagementProcessor
import os
import pandas as pd
import numpy as np
import pingouin as pg

class EngagementsManager:

    engagements_list: list[EngagementProcessor] = []
    data_path: str = "../../data/annotations"
    path_groups_info: str = "../../data/group_durations_all_commas.csv"
    filename_all_groups_rec_time: str = "all_groups_task_eng90Hz.csv"
    filename_dyads_rec_time: str ="dyads_task_eng90Hz.csv"                    
    filename_triads_rec_time: str ="triads_task_eng90Hz.csv"
    filename_all_groups_interaction_time: str ="all_groups_interaction_task_eng90Hz.csv"
    filename_dyads_interaction_time: str ="dyads_interaction_task_eng90Hz.csv"                    
    filename_triads_interaction_time: str ="triads_interaction_task_eng90Hz.csv"
    loaded_from_disk: bool
    folders: list[str]
    groups_floorlevel: list[str] = []
    df_avg_eng_all: pd.DataFrame
    df_avg_eng_dyads: pd.DataFrame
    df_avg_eng_triads: pd.DataFrame
    df_avg_eng_all_interaction: pd.DataFrame
    df_avg_eng_dyads_interaction: pd.DataFrame
    df_avg_eng_triads_interaction: pd.DataFrame
    df_percent_TE_discrete: pd.DataFrame
    df_interrater_agreement: pd.DataFrame


    def __init__(self, save_to_disk: bool, load_from_disk:bool, floor_level: bool, discretised_data = False) -> None:
        self.load_engagements(save_to_disk=save_to_disk, load_from_disk=load_from_disk, discretised_data = discretised_data)
        self.avg_engagements(save_to_disk=save_to_disk)
        # drop columns that are not in floor level if true
        self.__get_groups_floorlevel()
        if floor_level:
            self.__drop_floorlevel()
            


    def load_engagements(self, save_to_disk: bool, load_from_disk: bool, discretised_data = False):
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
                if discretised_data:
                    eng_processor.process_discretise_task_engagement(two_bins_for_two_anno=True)
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

    def calculate_discrete_TE_stats(self, save_to_disk=False):
        df_interaction_data = pd.DataFrame()
        for engagement_group_data in self.engagements_list:
            if engagement_group_data.group_name in self.groups_floorlevel:
                percent_TE_anno1 = engagement_group_data.df_eng_discretised_anno1_interaction.task_eng.value_counts()/len(engagement_group_data.df_eng_discretised_anno1_interaction)
                percent_TE_anno2 = engagement_group_data.df_eng_discretised_anno2_interaction.task_eng.value_counts()/len(engagement_group_data.df_eng_discretised_anno2_interaction)
                df_interaction_data[engagement_group_data.group_name+'_anno01'] = percent_TE_anno1
                df_interaction_data[engagement_group_data.group_name+'_anno02'] = percent_TE_anno2

        self.df_percent_TE_discrete = df_interaction_data
        if save_to_disk:
            # df_interaction_data.to_csv("../../data/annotations/TE_discrete_percentages.csv")
            df_interaction_data.to_csv("../../data/annotations/TE_discrete_percentages_.25.5_forAnno2.csv")
        return df_interaction_data

    def calculate_interrater_reliability(self, save_to_disk=False):
        df_interrater_reliability_data = pd.DataFrame(index=['continuous','discrete'])#, 'recording_continuous'])
        #go over all the groups and calculate the interrater reliability (like in nova):
        for engagement_group_data in self.engagements_list:
            if engagement_group_data.group_name in self.groups_floorlevel:
                print (engagement_group_data.group_name)
                current_df_discrete = pd.DataFrame({'anno1':engagement_group_data.df_eng_discretised_anno1_interaction.task_eng,
                                           'anno2':engagement_group_data.df_eng_discretised_anno2_interaction.task_eng})
                current_df_cont = pd.DataFrame(
                    {'anno1': engagement_group_data.df_eng_1_interaction.task_eng,
                     'anno2': engagement_group_data.df_eng_2_interaction.task_eng})
                # current_df_recording_cont = pd.DataFrame(
                #     {'anno1': engagement_group_data.df_eng_1.task_eng,
                #      'anno2': engagement_group_data.df_eng_2.task_eng})

                # current_df_discrete.replace([np.inf, -np.inf], np.nan).dropna(axis=0, inplace=True)
                # current_df_cont.replace([np.inf, -np.inf], np.nan).dropna(axis=0, inplace=True)
                # current_df_discrete.dropna(axis=0, how='any', inplace=True)
                current_df_cont = current_df_cont[pd.to_numeric(current_df_cont['anno2'], errors='coerce').notnull()]
                current_df_cont = current_df_cont[pd.to_numeric(current_df_cont['anno1'], errors='coerce').notnull()]

                current_df_cont.dropna(axis=0, how='any', inplace=True)

                current_df_cont['anno2'] = pd.to_numeric(current_df_cont['anno2'])
                current_df_cont['anno1'] = pd.to_numeric(current_df_cont['anno1'])
                current_df_cont['anno1'] = current_df_cont['anno1'].clip(lower=0)
                current_df_cont['anno2'] = current_df_cont['anno2'].clip(lower=0)

                current_df_discrete.dropna(axis=0, how='any', inplace=True)
                current_df_discrete.replace({'low': 1, 'mid': 2, 'high': 3}, inplace=True)
                # current_df_cont.replace({'low': 1, 'mid': 2, 'high': 3}, inplace=True)
                #make all vals numeric
                current_df_discrete['anno2'] = pd.to_numeric(current_df_discrete['anno2'])
                current_df_discrete['anno1'] = pd.to_numeric(current_df_discrete['anno1'])
                # current_df_cont['anno2'] = pd.to_numeric(current_df_cont['anno2'])
                # current_df_cont['anno1'] = pd.to_numeric(current_df_cont['anno1'])

                # current_df_recording_cont['anno2'] = pd.to_numeric(current_df_recording_cont['anno2'])
                # current_df_recording_cont['anno1'] = pd.to_numeric(current_df_recording_cont['anno1'])

                #get the interrater reliability for the discrete values
                interrater_discrete = pg.cronbach_alpha(data=current_df_discrete)[0]
                interrater_cont = pg.cronbach_alpha(data=current_df_cont)[0]
                # interrater_recording_cont = pg.cronbach_alpha(data=current_df_recording_cont)[0]

                df_interrater_reliability_data [engagement_group_data.group_name+'_interrater_relia'] = \
                    [interrater_cont, interrater_discrete]

        self.df_interrater_reliability = df_interrater_reliability_data
        if save_to_disk:
            # df_interaction_data.to_csv("../../data/annotations/TE_discrete_percentages.csv")
            df_interrater_reliability_data.to_csv("../../data/annotations/TE_interrater_reliability.csv")
        return df_interrater_reliability_data

    def __get_groups_floorlevel(self):
        df_details_floorlevel = pd.read_csv('..\..\data\group_names_with_time_floorlevel.csv', sep=';')
        groups_floorlevel = df_details_floorlevel['Group_Name'].to_list()
        self.groups_floorlevel = groups_floorlevel

    def __drop_floorlevel(self):
        # df_details_floorlevel = pd.read_csv(os.path.join(os.getcwd(), 'data', 'group_names_with_time_floorlevel.csv'), sep=';')
        self.df_avg_eng_all = self.__drop_cols_not_in(self.df_avg_eng_all, self.groups_floorlevel)
        self.df_avg_eng_dyads = self.__drop_cols_not_in(self.df_avg_eng_dyads, self.groups_floorlevel)
        self.df_avg_eng_triads = self.__drop_cols_not_in(self.df_avg_eng_triads, self.groups_floorlevel)
        self.df_avg_eng_all_interaction = self.__drop_cols_not_in(self.df_avg_eng_all_interaction, self.groups_floorlevel)
        self.df_avg_eng_dyads_interaction = self.__drop_cols_not_in(self.df_avg_eng_dyads_interaction, self.groups_floorlevel)
        self.df_avg_eng_triads_interaction = self.__drop_cols_not_in(self.df_avg_eng_triads_interaction, self.groups_floorlevel)
    
    def __drop_cols_not_in(self, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        cols.append(df.columns[df.columns.str.contains('seconds')][0])
        cols_to_keep = df.columns.str.contains("|".join(cols))
        return df[df.columns[cols_to_keep]]
        
if __name__ == "__main__":    
    mngr_aux: EngagementsManager = EngagementsManager(save_to_disk=False, load_from_disk=False, floor_level=True, discretised_data = True)
    mngr_aux.calculate_discrete_TE_stats(save_to_disk=True)
    mngr_aux.calculate_interrater_reliability(save_to_disk=True)
    print("done")