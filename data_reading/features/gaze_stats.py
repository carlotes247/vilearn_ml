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
    #                           and columns (int 0 or 1): 'P1P2_MG', 'P1P3_MG', 'P2P3_MG',
    groups_gaze_with_timestamps: list[dict]

    path_going_up_two_folders = "../../"
    path_prefix_file = path_going_up_two_folders + "data/_path_prefix.txt"
    data_folder_path = path_going_up_two_folders + "data/"

    use_interaction_time:bool = True

    def __init__(self, group_names: list[str] = [], group_names_filename: str = "", use_interaction_time: bool = True):
        self.use_interaction_time = use_interaction_time
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
            my_groups_manager = GroupsManager(self.path_prefix_file, self.data_folder_path, specific_group=row['Group'], all_groups_names_path="",
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
            # drop any repeated timestamps
            current_group_dataframe = current_group_dataframe.drop_duplicates(subset=['TSGroupNTP'])
            # set the index to the timestamp to easily get a subset of it based on the correct group conversation
            current_group_dataframe = current_group_dataframe.set_index('TSGroupNTP')
            # sorting the index (timestamps) as the next fuction won't work on a non-soted list. even though it is sorted
            current_group_dataframe.sort_index(inplace=True)

            if self.use_interaction_time:
                # get the correct subset of the dataframe
                start_time_timestamp = pd.to_datetime(row['Start'], utc=True, format='%Y-%m-%d %H:%M:%S.%f')
                end_time_timestamp = pd.to_datetime(row['End'], utc=True, format='%Y-%m-%d %H:%M:%S.%f')
                current_group_dataframe = group_data.get_subset_df_based_on_interaction_start_and_end(start_time_timestamp,
                                                                                                end_time_timestamp,
                                                                                                current_group_dataframe)

            # calculate the MG and add new columns to the df:
            current_group_dataframe["MG_P1P2"] =((current_group_dataframe['DG_P1_target'] == 2)
                                                      & (current_group_dataframe['DG_P2_target'] == 1)).astype(int)
            if group_data.num_participants>2:
                current_group_dataframe["MG_P1P3"] = ((current_group_dataframe['DG_P1_target'] == 3)
                                                           & (current_group_dataframe['DG_P3_target'] == 1)).astype(int)
                current_group_dataframe["MG_P2P3"] = ((current_group_dataframe['DG_P2_target'] == 3)
                                                           & (current_group_dataframe['DG_P3_target'] == 2)).astype(int)


            # put all the info into a dictionary and then add it to a list
            d = {'group_name': group_data.group_name, 'group_size': group_data.num_participants,
                 'gaze_df': current_group_dataframe}

            # append the dataframe to the list of all the groups.
            self.groups_gaze_with_timestamps.append(d)


if __name__ == "__main__":
    group_data_time_subset_filename = 'group_names_with_time_subsets.csv'
    # group_data_time_subset_filename = 'group_names_with_time_subsetsFullVERSION.csv'
    gaze_stats_subsets = GazeStats(group_names_filename=group_data_time_subset_filename)