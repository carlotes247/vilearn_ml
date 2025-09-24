import statistics

import pandas as pd
from scipy import stats
# Added this try catch because on some machines it cannot find data_reading module from this script
try:
    from data_reading.groups_manager import GroupsManager
except ImportError:
    import sys
    sys.path.append(sys.path[0] + '/../..')
    from data_reading.groups_manager import GroupsManager
# from data_reading.groups_manager import GroupsManager
import matplotlib.pyplot as plt
import json
import os

class GazeStats:

    # groups_gaze_with_timestamps: a list of dicts that has for each group:
    # "group_name" string, the group name
    # "group_size" int, group size
    # 'gaze_df': index: timestamp, columns (int:[0,1,2,3]): 'P{1, 2, or 3}_DG_target' (it can take the following values:
    #                           0: no direct gaze, 1: direct gaze towards P1, 2: direct gaze towards P2, 3: direct gaze towards P3)
    #                           and columns (int 0 or 1): 'P1P2_MG', 'P1P3_MG', 'P2P3_MG'
    #           seconds_interaction, seconds_recording: time since the interaction or the recording started
    groups_gaze_with_timestamps: list[dict]
    dyads_counts_df: pd.DataFrame
    triads_counts_df: pd.DataFrame
    all_groups_counts_df: pd.DataFrame


    path_going_up_two_folders = "../../"
    path_prefix_file = "data/_path_prefix.txt"
    data_folder_path = "data/"
    if not os.path.exists("data"):
        path_prefix_file = path_going_up_two_folders + path_prefix_file
        data_folder_path = path_going_up_two_folders + data_folder_path

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


    def populate_groups_gaze_dataset_with_a_subset_timeline(self, group_names_filename:str, debug_print:bool=False):
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
                # turning it into a d
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

            # TODO: slice dataframe to only have interaction time
            current_group_dataframe = current_group_dataframe[current_group_dataframe['seconds_interaction'].notnull()]

            # calculate the MG and add new columns to the df:
            current_group_dataframe["MG_P1P2"] =((current_group_dataframe['DG_P1_target'] == 2)
                                                      & (current_group_dataframe['DG_P2_target'] == 1)).astype(int)
            # 0 D1 dyads
            if group_data.num_participants == 2:
                current_group_dataframe["0_D1"] =((current_group_dataframe['DG_P1_target'] == 0) 
                                                        & (current_group_dataframe['DG_P2_target'] == 0)).astype(int)

            # triads exclusive configurations
            if group_data.num_participants>2:
                current_group_dataframe["MG_P1P3"] = ((current_group_dataframe['DG_P1_target'] == 3)
                                                           & (current_group_dataframe['DG_P3_target'] == 1)).astype(int)
                current_group_dataframe["MG_P2P3"] = ((current_group_dataframe['DG_P2_target'] == 3)
                                                           & (current_group_dataframe['DG_P3_target'] == 2)).astype(int)

            #perhaps here is a good place to add the calculation of a 1d_DG
            #1d_DG -> one directional direct gaze, P1 looks at P2, but P2 doesn't look at P1 (and viceversa)
            current_group_dataframe["1d_DG_P1"] = ((current_group_dataframe['DG_P1_target'] == 2)
                                                  & (current_group_dataframe['DG_P2_target'] != 1)).astype(int)
            current_group_dataframe["1d_DG_P2"] = ((current_group_dataframe['DG_P2_target'] == 1)
                                                   & (current_group_dataframe['DG_P1_target'] != 2)).astype(int)
            if group_data.num_participants>2:
                current_group_dataframe["1d_DG_P1"] = (((current_group_dataframe['DG_P1_target'] == 2)
                                                       & (current_group_dataframe['DG_P2_target'] != 1))
                                                       | ((current_group_dataframe['DG_P1_target'] == 3)
                                                       & (current_group_dataframe['DG_P3_target'] != 1))).astype(int)

                current_group_dataframe["1d_DG_P2"] = (((current_group_dataframe['DG_P2_target'] == 1)
                                                        & (current_group_dataframe['DG_P1_target'] != 2))
                                                       | ((current_group_dataframe['DG_P2_target'] == 3)
                                                          & (current_group_dataframe['DG_P3_target'] != 2))).astype(int)

                current_group_dataframe["1d_DG_P3"] = (((current_group_dataframe['DG_P3_target'] == 1)
                                                        & (current_group_dataframe['DG_P1_target'] != 3))
                                                       |  ((current_group_dataframe['DG_P3_target'] == 2)
                                                          & (current_group_dataframe['DG_P2_target'] != 3))).astype(int)

                # Logic for triad dynamics configurations
                # 0 D1 (Nobody looks at the other participants)
                current_group_dataframe['0_D1'] = ((current_group_dataframe['DG_P1_target'] == 0) 
                                                        & (current_group_dataframe['DG_P2_target'] == 0)
                                                        & (current_group_dataframe['DG_P3_target'] == 0)).astype(int)
                # 1 D1 (One looks at the other. two, don’t)
                # P1 to anybody, P2 & P3 to nothing
                current_group_dataframe['1_D1'] = (
                    # P1 to anybody, P2 & P3 to nothing
                    (current_group_dataframe['DG_P1_target'] > 0) & (current_group_dataframe['DG_P2_target'] == 0) & (current_group_dataframe['DG_P3_target'] == 0)
                    # P2 to anybody, P1 & P3 to nothing
                    | (current_group_dataframe['DG_P2_target'] > 0) & (current_group_dataframe['DG_P1_target'] == 0) & (current_group_dataframe['DG_P3_target'] == 0)
                    # P3 to anybody, P1 & P2 to nothing
                    | (current_group_dataframe['DG_P3_target'] > 0) & (current_group_dataframe['DG_P1_target'] == 0) & (current_group_dataframe['DG_P2_target'] == 0)
                    ).astype(int)
                # 2 D1 same (Two to the same person, which to nothing;
                current_group_dataframe['2_D1_same'] = (
                    # P1 & P2 to P3, P3 to nothing
                    (current_group_dataframe['DG_P1_target'] == 3) & (current_group_dataframe['DG_P2_target'] == 3) & (current_group_dataframe['DG_P3_target'] == 0)
                    # P1 & P3 to P2, P2 to nothing
                    | (current_group_dataframe['DG_P1_target'] == 2) & (current_group_dataframe['DG_P3_target'] == 2) & (current_group_dataframe['DG_P2_target'] == 0)
                    # P2 & P3 to P1, P1 to nothing
                    | (current_group_dataframe['DG_P2_target'] == 1) & (current_group_dataframe['DG_P3_target'] == 1) & (current_group_dataframe['DG_P1_target'] == 0)
                    ).astype(int)                   
                # 2 D1 different (One to nothing and their two each to a different person)
                current_group_dataframe['2_D1_different'] = (
                    # P1 to nothing, P2 to P3, P3 to P1
                    (current_group_dataframe['DG_P1_target'] == 0) & (current_group_dataframe['DG_P2_target'] == 3) & (current_group_dataframe['DG_P3_target'] == 1)
                    # P1 to nothing, P2 to P1, P3 to P2
                    | (current_group_dataframe['DG_P1_target'] == 0) & (current_group_dataframe['DG_P2_target'] == 1) & (current_group_dataframe['DG_P3_target'] == 2)
                    # P2 to nothing, P1 to P2, P3 to P1
                    | (current_group_dataframe['DG_P2_target'] == 0) & (current_group_dataframe['DG_P1_target'] == 2) & (current_group_dataframe['DG_P3_target'] == 1)
                    # P2 to nothing, P1 to P3, P3 to P2
                    | (current_group_dataframe['DG_P2_target'] == 0) & (current_group_dataframe['DG_P1_target'] == 3) & (current_group_dataframe['DG_P3_target'] == 2)
                    # P3 to nothing, P1 to P2, P2 to P3
                    | (current_group_dataframe['DG_P3_target'] == 0) & (current_group_dataframe['DG_P1_target'] == 2) & (current_group_dataframe['DG_P2_target'] == 3)
                    # P3 to nothing, P1 to P3, P2 to P1
                    | (current_group_dataframe['DG_P3_target'] == 0) & (current_group_dataframe['DG_P1_target'] == 3) & (current_group_dataframe['DG_P2_target'] == 1)                                        
                    ).astype(int)                  
                # 3 D1 (Each to the different one - circle of gaze)
                current_group_dataframe['3_D1'] = (
                    # P1 to P2, P2 to P3, P3 to P1
                    (current_group_dataframe['DG_P1_target'] == 2) & (current_group_dataframe['DG_P2_target'] == 3) & (current_group_dataframe['DG_P3_target'] == 1)
                    # P1 to P3, P2 to P1, P3 to P2
                    | (current_group_dataframe['DG_P1_target'] == 3) & (current_group_dataframe['DG_P2_target'] == 1) & (current_group_dataframe['DG_P3_target'] == 2)
                    ).astype(int) 
                # MG D1 (The third left out is looking at one of the two engaged in MG)
                current_group_dataframe['MG_D1'] = (
                    # P1 to P2, P2 to P1, P3 to anyone
                    (current_group_dataframe['DG_P1_target'] == 2) & (current_group_dataframe['DG_P2_target'] == 1) & (current_group_dataframe['DG_P3_target'] > 0)
                    # P1 to P3, P3 to P1, P2 to anyone
                    | (current_group_dataframe['DG_P1_target'] == 3) & (current_group_dataframe['DG_P3_target'] == 1) & (current_group_dataframe['DG_P2_target'] > 0)
                    # P2 to P3, P3 to P2, P1 to anyone
                    | (current_group_dataframe['DG_P2_target'] == 3) & (current_group_dataframe['DG_P3_target'] == 2) & (current_group_dataframe['DG_P1_target'] > 0)                    
                    ).astype(int) 
                # MG D0  (The third does not look at any of the other 2)
                current_group_dataframe['MG_D0'] = (
                    # P1 to P2, P2 to P1, P3 to nothing
                    (current_group_dataframe['DG_P1_target'] == 2) & (current_group_dataframe['DG_P2_target'] == 1) & (current_group_dataframe['DG_P3_target'] == 0)
                    # P1 to P3, P3 to P1, P2 to nothing
                    | (current_group_dataframe['DG_P1_target'] == 3) & (current_group_dataframe['DG_P3_target'] == 1) & (current_group_dataframe['DG_P2_target'] == 0)
                    # P2 to P3, P3 to P2, P1 to nothing
                    | (current_group_dataframe['DG_P2_target'] == 3) & (current_group_dataframe['DG_P3_target'] == 2) & (current_group_dataframe['DG_P1_target'] == 0)                    
                    ).astype(int)
                # TODO: add original target values to df
                # TODO: add a string column describing which hit was this frame e.g. "P1:P2;P2:XX;P3:P1--2_D1_different"
            
            # counts of occurrences 
            cases_df = current_group_dataframe.drop(['TSGroupNTP', 'seconds_recording', 'seconds_interaction'], axis=1)
            counts: list[pd.Series[int]] = []            
            counts_df: pd.DataFrame = pd.DataFrame()
            counts_list: list[dict] = []
            percentage: float = 0.0
            if debug_print:
                print(f"---- {row['Group_Name']} ---")
            for column in cases_df.columns:
                if debug_print:
                    print(f"---- {column} ---")
                count = cases_df[column].value_counts()     
                count_df = pd.DataFrame(count)           
                dict_info = {'case_name' : f"{row['Group_Name']}_{count.index.name}"}
                for i, value in enumerate(count):
                    index:str = f"{count.index[i]}"
                    dict_info[index] = value
                if len(count) == 2:                            
                    percentage = min(count) / count.sum()
                    dict_info['Percentage'] = percentage 
                    percentage = 0.0                            
                if debug_print:
                    print(count)
                    print(f"Percentage of cases {percentage}")                                               
                counts.append(count)
                counts_list.append(dict_info)

            # reordering the counts df so that percentage is the last column
            counts_df: pd.DataFrame = pd.DataFrame(counts_list)
            cols = ['case_name', '0','1','2', 'Percentage']
            counts_df = counts_df[cols]
            # construct a list of dicts with the info (very pythonic)
            d = {'group_name': row['Group_Name'], 'group_size': group_data.num_participants,
                 'gaze_df': current_group_dataframe, 'counts': counts, 'counts_df': counts_df}    
                
            # append the dict list to the list of all the groups.
            self.groups_gaze_with_timestamps.append(d)

        # Separate groups in dfs for analysis
        if len(self.groups_gaze_with_timestamps) > 0:
            all_groups_list: list[dict] = [d['counts_df'] for d in self.groups_gaze_with_timestamps]
            self.all_groups_counts_df = pd.concat(all_groups_list, ignore_index=True).set_index('case_name')
            dyads_list: list[dict] = [d['counts_df'] for d in self.groups_gaze_with_timestamps if 'dyad' in d['group_name']]
            self.dyads_counts_df = pd.concat(dyads_list, ignore_index=True).set_index('case_name')
            triads_list: list[dict] = [d['counts_df'] for d in self.groups_gaze_with_timestamps if 'triad' in d['group_name']]
            self.triads_counts_df = pd.concat(triads_list, ignore_index=True).set_index('case_name')

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
            ##### group['gaze_df']['seconds_interaction'] = group['gaze_df']['seconds_interaction'].apply(lambda x: "{:.2f}".format(x))
            group['gaze_df']['seconds_interaction'] = group['gaze_df']['seconds_interaction'].astype(float)
            # group['gaze_df']['seconds_recording'] = group['gaze_df']['seconds_recording'].apply(lambda x: "{:.2f}".format(x))
            #### group['gaze_df'].drop_duplicates(subset=['seconds_interaction'], inplace=True)
            group['gaze_df'].reset_index(inplace=True)

            if df_for_all_dyads_data and group['group_size'] == 2:
                # dyads_df[group['group_name'] + '_seconds_recording'] = group['gaze_df']["seconds_recording"]
                dyads_df[group['group_name'] + '_seconds_interaction'] = group['gaze_df']["seconds_interaction"]

                #setting a keyword for MG or DG
                if mutual_gaze:
                    dyads_col_keyword = ['MG_P1P2']
                else:
                    dyads_col_keyword = ['DG_P1_target', 'DG_P2_target', '1d_DG_P1', '1d_DG_P2']

                # dyads_df[group['group_name']+'_MG_P1P2'] = group['gaze_df']["MG_P1P2"]
                for col_key in dyads_col_keyword:
                    dyads_df[group['group_name']+'_'+ col_key] = group['gaze_df'][col_key].clip(0, 1)#clipping here overcomes the isue with values of 2 (or 3 in triads)denoting whom the person was looking at. At this point we only care about a DG, regardless of where that is directed to.

                current_inter_dyad = pd.concat([dyads_df[group['group_name'] +'_'+ dyads_col_keyword[0]],
                                          group['gaze_df']["seconds_interaction"]], axis=1)
                if direct_gaze:
                    current_inter_dyad = pd.concat([dyads_df[group['group_name'] + '_' + dyads_col_keyword[0]],
                                                    dyads_df[group['group_name'] + '_' + dyads_col_keyword[1]],
                                                    dyads_df[group['group_name'] + '_' + dyads_col_keyword[2]],
                                                    dyads_df[group['group_name'] + '_' + dyads_col_keyword[3]],
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
                    triads_col_keyword = ['DG_P1_target', 'DG_P2_target', 'DG_P3_target','1d_DG_P1', '1d_DG_P2', '1d_DG_P3']

                for col_keyword in triads_col_keyword:
                    triads_df[group['group_name']+'_'+col_keyword] = group['gaze_df'][col_keyword].clip(0,1)

                # current_rec_triad = pd.concat([triads_df[group['group_name']+'_MG_P1P2'],
                #                            triads_df[group['group_name'] + '_MG_P1P3'],
                #                            triads_df[group['group_name'] + '_MG_P2P3'],
                #                            group['gaze_df']["seconds_recording"]], axis=1)

                current_inter_triad = pd.concat([triads_df[group['group_name']+'_'+triads_col_keyword[0]],
                                           triads_df[group['group_name'] + '_'+triads_col_keyword[1]],
                                           triads_df[group['group_name'] + '_'+triads_col_keyword[2]],
                                           group['gaze_df']["seconds_interaction"]], axis=1)

                if direct_gaze:
                    current_inter_triad = pd.concat([triads_df[group['group_name'] + '_' + triads_col_keyword[0]],
                                                     triads_df[group['group_name'] + '_' + triads_col_keyword[1]],
                                                     triads_df[group['group_name'] + '_' + triads_col_keyword[2]],
                                                     triads_df[group['group_name'] + '_' + triads_col_keyword[3]],
                                                     triads_df[group['group_name'] + '_' + triads_col_keyword[4]],
                                                     triads_df[group['group_name'] + '_' + triads_col_keyword[5]],
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
    # save config flags
    save_to_file = False
    save_count_df_to_file = True
    save_count_df_each_group = False
    # how many groups to include in logic
    small_subset_groups = False
    floorlevel_subset_groups = False
    all_groups = True
    # path vars depending on group size
    group_data_time_subset_filename = ""
    path_suffix = ""
    if small_subset_groups:
        group_data_time_subset_filename = 'group_names_with_time_subsets.csv'
        path_suffix = "_small_subset"
    elif floorlevel_subset_groups:
        group_data_time_subset_filename = 'group_names_with_time_floorlevel.csv'
        path_suffix = "_floorlevel"
    elif all_groups:
        group_data_time_subset_filename = 'group_names_with_time_subsetsFullVERSION.csv'        
    gaze_stats_subsets = GazeStats(group_names_filename=group_data_time_subset_filename)
    
    # save gaze counts
    if save_count_df_to_file:
        gaze_stats_subsets.dyads_counts_df.to_csv(os.path.join(os.getcwd(), "Recordings", "SavedData", "gaze_counts", f"dyads_counts_gaze{path_suffix}.csv"))
        gaze_stats_subsets.triads_counts_df.to_csv(os.path.join(os.getcwd(), "Recordings", "SavedData", "gaze_counts", f"triads_counts_gaze{path_suffix}.csv"))
        gaze_stats_subsets.all_groups_counts_df.to_csv(os.path.join(os.getcwd(), "Recordings", "SavedData", "gaze_counts", f"all_groups_counts_gaze{path_suffix}.csv"))
        if save_count_df_each_group:
            for group in gaze_stats_subsets.groups_gaze_with_timestamps:
                group_counts_df: pd.DataFrame = group['counts_df']
                path_gaze_counts = os.path.join(os.getcwd(), "Recordings", "SavedData", "gaze_counts", f"{group['group_name']}_counts_gaze{path_suffix}.csv")
                group_counts_df.to_csv(path_gaze_counts)

    (dyads_df, triads_df, all_groups_df,
     # dyads_merged_rec_df, triads_merged_rec_df, all_groups_merged_rec_df,
     dyads_merged_inter_df, triads_merged_inter_df, all_groups_merged_inter_df) = (
        gaze_stats_subsets.populate_dfs_with_group_data(True, True, True,
                                                        direct_gaze=True, mutual_gaze=False))

    if save_to_file:
        root_path= "../../Recordings/SavedData/v2"
        dyads_df.to_csv(root_path+f"all_dyads_mutual_gaze_individualTS{path_suffix}.csv")
        triads_df.to_csv(root_path+f"all_triads_mutual_gaze_individualTS{path_suffix}.csv")
        all_groups_df.to_csv(root_path+f"all_groups_mutual_gaze_individualTS{path_suffix}.csv")

        # dyads_merged_rec_df.to_csv(root_path+"all_dyads_mutual_gaze_recording_time.csv")
        # triads_merged_rec_df.to_csv(root_path+"all_triads_mutual_gaze_recording_time.csv")
        # all_groups_merged_rec_df.to_csv(root_path+"all_groups_mutual_gaze_recording_time.csv")

        dyads_merged_inter_df.to_csv(root_path+f"all_dyads_mutual_gaze_interaction_time{path_suffix}.csv")
        triads_merged_inter_df.to_csv(root_path+f"all_triads_mutual_gaze_interaction_time{path_suffix}.csv")
        all_groups_merged_inter_df.to_csv(root_path+f"all_groups_mutual_gaze_interaction_time{path_suffix}.csv")

