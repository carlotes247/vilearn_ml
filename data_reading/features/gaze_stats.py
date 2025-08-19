import statistics

import pandas as pd
from scipy import stats

from data_reading.groups_manager import GroupsManager
import matplotlib.pyplot as plt
import json

class GazeStats:

    # groups_gaze_with_timestamps: a list of dicts that has for each group:
    # "group_name" string, the group name
    # "group_size" int, group size
    # 'gaze_df': index: timestamp, columns (int:[0,1,2,3]): 'P{1, 2, or 3}_DG_target' (it can take the following values:
    #                           0: no direct gaze, 1: direct gaze towards P1, 2: direct gaze towards P2, 3: direct gaze towards P3)
    #                           and columns (int 0 or 1): 'P1P2_MG', 'P1P3_MG', 'P2P3_MG'
    #           seconds_interaction, seconds_recording: time since the interaction or the recording started
    groups_gaze_with_timestamps: list[dict]

    path_going_up_two_folders = "../../"
    path_prefix_file = path_going_up_two_folders + "data/_path_prefix.txt"
    data_folder_path = path_going_up_two_folders + "data/"


    def __init__(self, group_names: list[str] = [], group_names_filename: str = ""):
        self.groups_gaze_with_timestamps = []
        raw_data = pd.DataFrame

        if len(group_names) == 0 and group_names_filename == "":
            print(f"No file with the group names found. No data in the group_names list.")
            return

        if len(group_names_filename) > 0:
            # build the group_names from the file
            self.populate_groups_gaze_dataset_with_a_subset_timeline(group_names_filename)
        else:
            print(f"Method for using group names as strings (not via a file) not implemented yet.")
            # self.populate_groups_gaze_dataset_with_a_the_full_timeline(group_names)


    def populate_groups_gaze_dataset_with_a_subset_timeline(self, group_names_filename:str):
        fullpath = self.data_folder_path + group_names_filename
        data_for_calculating_subsets = pd.read_csv(fullpath, sep=';')

        for index, row in data_for_calculating_subsets.iterrows():

            current_group_dataframe = pd.DataFrame()
            my_groups_manager = GroupsManager(self.path_prefix_file, self.data_folder_path, specific_group=row['Group_Name_Long'], all_groups_names_path="",
                                              onlyTorch=False, load_individual_p_files=False, print_all_stats=False, print_blink_stats=False, use_async=False, print_debug=False)
            group_data = my_groups_manager.groups[0]
            group_data_raw_df = group_data.group_features_csv_loader.raw_data

            timestamps_string = group_data_raw_df['TSGroupNTP']
            timestamps = pd.to_datetime(timestamps_string, utc=True, format='%Y-%m-%d %H:%M:%S.%f')

            for participant in range(group_data.num_participants):

                #get the direct gaze targeted to one of the other participants for each participants. #
                # It can take one of the values: either 0 (no DG), 1 (looking at P1), 2 (P2), or 3 (P3) '
                # in the raw file: 'TargetGazeP1' 'TargetGazeP2' 'TargetGazeP3'
                participant_DG_target = group_data_raw_df[f'TargetGazeP{participant+1}']
                # turning it into a df
                participant_DG_target_df = pd.DataFrame({f'DG_P{participant + 1}_target': participant_DG_target})
                # adding the df to the large one
                current_group_dataframe = pd.concat([current_group_dataframe, participant_DG_target_df], axis='columns')


            # adding the timestap to the df
            current_group_dataframe = pd.concat([current_group_dataframe, timestamps], axis='columns')
            # add the recording time in seconds as another column
            current_group_dataframe['seconds_recording'] = (current_group_dataframe['TSGroupNTP'] -
                                                              current_group_dataframe['TSGroupNTP'][0]).dt.total_seconds()

            #get the info to calculate the interaction time in seconds
            start_interaction_timestamp = pd.to_datetime(row['TS_Start_Interaction'], utc=True, format='%Y-%m-%d %H:%M:%S.%f')
            end_interaction_timestamp = pd.to_datetime(row['TS_End_Interaction'], utc=True, format='%Y-%m-%d %H:%M:%S.%f')
            duration_interaction = (end_interaction_timestamp - start_interaction_timestamp).total_seconds()

            # add the interaction time in seconds as another column; this will sort out the start (0, negative vals will
            # be removed) and the end will be removed using the duration value
            current_group_dataframe['seconds_interaction'] = (current_group_dataframe['TSGroupNTP'] -
                                                              start_interaction_timestamp).dt.total_seconds()
            #setting to NAN the negative values
            current_group_dataframe['seconds_interaction'] = (current_group_dataframe['seconds_interaction'].mask
                                                              (current_group_dataframe['seconds_interaction'] < 0))
            #setting to NAN the values larger than the duration of the interaction
            current_group_dataframe['seconds_interaction'] = (current_group_dataframe['seconds_interaction'].mask
                                                              (current_group_dataframe['seconds_interaction'] >
                                                               duration_interaction))


            # calculate the MG and add new columns to the df:
            current_group_dataframe["MG_P1P2"] =((current_group_dataframe['DG_P1_target'] == 2)
                                                      & (current_group_dataframe['DG_P2_target'] == 1)).astype(int)
            if group_data.num_participants>2:
                current_group_dataframe["MG_P1P3"] = ((current_group_dataframe['DG_P1_target'] == 3)
                                                           & (current_group_dataframe['DG_P3_target'] == 1)).astype(int)
                current_group_dataframe["MG_P2P3"] = ((current_group_dataframe['DG_P2_target'] == 3)
                                                           & (current_group_dataframe['DG_P3_target'] == 2)).astype(int)


            # put all the info into a dictionary and then add it to a list
            d = {'group_name': row['Group_Name'], 'group_size': group_data.num_participants,
                 'gaze_df': current_group_dataframe}

            # append the dataframe to the list of all the groups.
            self.groups_gaze_with_timestamps.append(d)

    #created dfs for all dyads, all triads and all the groups and returns them; if one of the bool is false, it doesn't return that df
    def populate_dfs_with_group_data(self, df_for_all_dyads_data:bool, df_for_all_triads_data:bool,
                                     df_for_all_group_data:bool, mutual_gaze:bool, direct_gaze:bool):
        dyads_df = pd.DataFrame()
        # dyads_merged_rec_df = pd.DataFrame({'seconds_recording':[]})
        dyads_merged_inter_df = pd.DataFrame({'seconds_interaction':[]})

        triads_df = pd.DataFrame()
        # triads_merged_rec_df = pd.DataFrame({'seconds_recording':[]})
        triads_merged_inter_df = pd.DataFrame({'seconds_interaction':[]})

        all_groups_df = pd.DataFrame()
        # all_groups_merged_rec_df = pd.DataFrame({'seconds_recording':[]})
        all_groups_merged_inter_df = pd.DataFrame({'seconds_interaction':[]})

        for group in self.groups_gaze_with_timestamps:
            print (group['group_name'])
            # change the second column to have only 2 decimal points. That means there would be at most 100 frames per second
            group['gaze_df']['seconds_interaction'] = group['gaze_df']['seconds_interaction'].apply(lambda x: "{:.2f}".format(x))
            group['gaze_df']['seconds_interaction'] = group['gaze_df']['seconds_interaction'].astype(float)
            # group['gaze_df']['seconds_recording'] = group['gaze_df']['seconds_recording'].apply(lambda x: "{:.2f}".format(x))
            group['gaze_df'].drop_duplicates(subset=['seconds_interaction'], inplace=True)
            group['gaze_df'].reset_index(inplace=True)

            if df_for_all_dyads_data and group['group_size'] == 2:
                # dyads_df[group['group_name'] + '_seconds_recording'] = group['gaze_df']["seconds_recording"]
                dyads_df[group['group_name'] + '_seconds_interaction'] = group['gaze_df']["seconds_interaction"]

                #setting a keyword for MG or DG
                if mutual_gaze:
                    dyads_col_keyword = ['MG_P1P2']
                else:
                    dyads_col_keyword = ['DG_P1_target', 'DG_P2_target']

                # dyads_df[group['group_name']+'_MG_P1P2'] = group['gaze_df']["MG_P1P2"]
                dyads_df[group['group_name']+'_'+ dyads_col_keyword[0]] = group['gaze_df'][dyads_col_keyword[0]].clip(0, 1)#clipping here overcomes the isue with values of 2 (or 3 in triads)denoting whom the person was looking at. At this point we only care about a DG, regardless of where that is directed to.
                if direct_gaze:
                    dyads_df[group['group_name'] + '_' + dyads_col_keyword[1]] = group['gaze_df'][dyads_col_keyword[1]].clip(0, 1)


                current_inter_dyad = pd.concat([dyads_df[group['group_name'] +'_'+ dyads_col_keyword[0]],
                                          group['gaze_df']["seconds_interaction"]], axis=1)
                if direct_gaze:
                    current_inter_dyad = pd.concat([dyads_df[group['group_name'] + '_' + dyads_col_keyword[0]],
                                                    dyads_df[group['group_name'] + '_' + dyads_col_keyword[1]],
                                                    group['gaze_df']["seconds_interaction"]], axis=1)

                current_inter_dyad.dropna(axis=0, how='any', inplace=True, ignore_index=True)
                # dyads_merged_rec_df = pd.merge_ordered(dyads_merged_rec_df, current_rec_dyad, on='seconds_recording')
                dyads_merged_inter_df = pd.merge_ordered(dyads_merged_inter_df, current_inter_dyad, on='seconds_interaction')


            if df_for_all_triads_data and group['group_size'] == 3:
                # triads_df[group['group_name'] + '_seconds_recording'] = group['gaze_df']["seconds_recording"]
                triads_df[group['group_name'] + '_seconds_interaction'] = group['gaze_df']["seconds_interaction"]
                # setting a keyword for MG or DG
                if mutual_gaze:
                    triads_col_keyword = ['MG_P1P2', 'MG_P1P3', 'MG_P2P3']
                else:
                    triads_col_keyword = ['DG_P1_target', 'DG_P2_target', 'DG_P3_target']

                triads_df[group['group_name']+'_'+triads_col_keyword[0]] = group['gaze_df'][triads_col_keyword[0]].clip(0,1)
                triads_df[group['group_name']+'_'+triads_col_keyword[1]] = group['gaze_df'][triads_col_keyword[1]].clip(0,1)
                triads_df[group['group_name']+'_'+triads_col_keyword[2]] = group['gaze_df'][triads_col_keyword[2]].clip(0,1)


                # current_rec_triad = pd.concat([triads_df[group['group_name']+'_MG_P1P2'],
                #                            triads_df[group['group_name'] + '_MG_P1P3'],
                #                            triads_df[group['group_name'] + '_MG_P2P3'],
                #                            group['gaze_df']["seconds_recording"]], axis=1)

                current_inter_triad = pd.concat([triads_df[group['group_name']+'_'+triads_col_keyword[0]],
                                           triads_df[group['group_name'] + '_'+triads_col_keyword[1]],
                                           triads_df[group['group_name'] + '_'+triads_col_keyword[2]],
                                           group['gaze_df']["seconds_interaction"]], axis=1)

                current_inter_triad.dropna(axis=0, how='any', inplace=True, ignore_index=True)

                # triads_merged_rec_df = pd.merge_ordered(triads_merged_rec_df, current_rec_triad, on='seconds_recording')
                triads_merged_inter_df = pd.merge_ordered(triads_merged_inter_df, current_inter_triad, on='seconds_interaction')

        # merge dyads and triads by the seconds_recording columns. there will be nan for any missing value
        if df_for_all_group_data:
            all_groups_df = pd.concat([dyads_df, triads_df], axis=1)
            # all_groups_merged_rec_df = pd.merge_ordered(dyads_merged_rec_df, triads_merged_rec_df, on='seconds_recording')
            all_groups_merged_inter_df = pd.merge_ordered(dyads_merged_inter_df, triads_merged_inter_df, on='seconds_interaction')



        return [dyads_df, triads_df, all_groups_df,
                    dyads_merged_inter_df, triads_merged_inter_df, all_groups_merged_inter_df]


if __name__ == "__main__":
    save_to_file = False
    # group_data_time_subset_filename = 'group_names_with_time_subsets.csv'
    group_data_time_subset_filename = 'group_names_with_time_subsetsFullVERSION.csv'
    gaze_stats_subsets = GazeStats(group_names_filename=group_data_time_subset_filename)
    (dyads_df, triads_df, all_groups_df,
     # dyads_merged_rec_df, triads_merged_rec_df, all_groups_merged_rec_df,
     dyads_merged_inter_df, triads_merged_inter_df, all_groups_merged_inter_df) = (
        gaze_stats_subsets.populate_dfs_with_group_data(True, True, True,
                                                        direct_gaze=True, mutual_gaze=False))

    if save_to_file:
        root_path= "../../Recordings/SavedData/"
        dyads_df.to_csv(root_path+"all_dyads_direct_gaze_individualTS.csv")
        triads_df.to_csv(root_path+"all_triads_direct_gaze_individualTS.csv")
        all_groups_df.to_csv(root_path+"all_groups_direct_gaze_individualTS.csv")

        # dyads_merged_rec_df.to_csv(root_path+"all_dyads_mutual_gaze_recording_time.csv")
        # triads_merged_rec_df.to_csv(root_path+"all_triads_mutual_gaze_recording_time.csv")
        # all_groups_merged_rec_df.to_csv(root_path+"all_groups_mutual_gaze_recording_time.csv")

        dyads_merged_inter_df.to_csv(root_path+"all_dyads_direct_gaze_interaction_time.csv")
        triads_merged_inter_df.to_csv(root_path+"all_triads_direct_gaze_interaction_time.csv")
        all_groups_merged_inter_df.to_csv(root_path+"all_groups_direct_gaze_interaction_time.csv")

