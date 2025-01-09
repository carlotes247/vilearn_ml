import datetime
import pandas as pd
from data_reading.groups_manager import GroupsManager

class BlinkStats:

    # dataframe that has for each group:
    # "group_name" string, the group name
    # "group_size" int, group size,
    # "blinks" list (series) of booleans for blinks
    #  "blinks_onset" list (series) of booleans for  blinks_onset,
    # "timestamps" list or series of timestamps
    groups_blinks_with_timestamps: list[pd.DataFrame]

    # list of with the group name, group size, blink rate (list of 2 or 3)
    groups_blink_rate = pd.Series() #this is not good as a Series. It needs to be changed

    path_going_up_two_folders = "../../"
    path_prefix_file = path_going_up_two_folders + "data/_path_prefix.txt"
    data_folder_path = path_going_up_two_folders + "data/"


    # def __init__(self, groups_blinks_with_timestamps: pd.DataFrame):
    def __init__(self, group_names: list[str] = [], group_names_filename: str = ""):
        self.groups_blinks_with_timestamps = []
        raw_data = pd.DataFrame

        if len(group_names) == 0 and group_names_filename == "":
            print(f"No file with the group names found. No data in the group_names list.")
            return

        if len(group_names_filename) > 0:
            #build the group_names from the file
            self.populate_groups_blinks_dataset_with_a_subset_timeline (group_names_filename)
        else:
            self.populate_groups_blinks_dataset_with_a_the_full_timeline(group_names)



    def populate_groups_blinks_dataset_with_a_subset_timeline(self, group_names_filename:str):
        fullpath = self.data_folder_path + group_names_filename
        data_for_calculating_subsets = pd.read_csv(fullpath, sep=';')

        for index, row in data_for_calculating_subsets.iterrows():
            my_groups_manager = GroupsManager(self.path_prefix_file, self.data_folder_path, specific_group=row['Group'],
                                              onlyTorch=False, load_individual_p_files=False, print_all_stats=False,
                                              print_blink_stats=False)
            group_data = my_groups_manager.groups[0]
            valid_blinks, valid_blink_onsets = group_data.group_features_csv_loader.extract_valid_blinks_frames()

            timestamps_string = group_data.group_features_csv_loader.raw_data['TSGroupNTP']
            timestamps = pd.to_datetime(timestamps_string, utc=True, format='%Y-%m-%d %H:%M:%S.%f')

            # calculate the valid subset of the group conversation
            # dt = pd.to_datetime("2016-11-13 22:01:25.450")
            #
            # s = df.loc[df.index.unique()[df.index.unique().get_loc(dt, method='nearest')]]

            start_time_timestamp = pd.to_datetime(row['Start'], utc=True, format='%Y-%m-%d %H:%M:%S.%f')
            index_of_start_timestamp = timestamps.index.get_indexer([start_time_timestamp], method='nearest')
            # index_of_start_timestamp = timestamps.loc[timestamps.index.unique()[timestamps.index.unique().get_loc(start_time_timestamp, method='nearest')]]

            end_time_timestamp = pd.to_datetime(row['End'])
            index_of_end_timestamp = timestamps.index.get_indexer([end_time_timestamp], method='nearest')

            subset_timestamps = timestamps.loc[index_of_start_timestamp:index_of_end_timestamp]
            subset_valid_blinks = dict
            subset_valid_blink_onset = dict

            if "DYAD" in row['Group']:
                group_size = 2
            else:
                group_size = 3

            # calculate the subsets for each participant
            for participant in range(group_size):
                participant_subset_valid_blinks = valid_blinks[participant][index_of_start_timestamp, index_of_end_timestamp]
                participant_subset_valid_blink_onset = valid_blink_onsets[participant][index_of_start_timestamp, index_of_end_timestamp]

            #   add to the dict:
                subset_valid_blinks[participant] = participant_subset_valid_blinks
                subset_valid_blink_onset[participant] = participant_subset_valid_blink_onset


            # put all the info into a ndarray and then add it to a dataframe
            d = {'group_name': row['Group'], 'group_size': group_size, 'blinks': subset_valid_blinks,
                 'blinks_onset': subset_valid_blink_onset, 'timestamps': subset_timestamps}
            # individual_group_blinks_data = pd.DataFrame(data=d)

            # append the dataframe to the list of all the groups.
            self.groups_blinks_with_timestamps.append(d)

    def populate_groups_blinks_dataset_with_a_the_full_timeline (self, group_names:list[str]):
        for group_name in group_names:

            my_groups_manager = GroupsManager(self.path_prefix_file, self.data_folder_path, specific_group= group_name,
                                              onlyTorch=False, load_individual_p_files=False, print_all_stats=False,
                                              print_blink_stats=False)
            group_data = my_groups_manager.groups[0]
            valid_blinks, valid_blink_onsets = group_data.group_features_csv_loader.extract_valid_blinks_frames()

            timestamps_string = group_data.group_features_csv_loader.raw_data['TSGroupNTP']
            timestamps = pd.to_datetime(timestamps_string, utc=True, format='%Y-%m-%d %H:%M:%S.%f')

            if "DYAD" in group_name:
                group_size = 2
            else:
                group_size = 3

            # put all the info into a ndarray and then add it to a dataframe
            d = {'group_name': group_name, 'group_size': group_size, 'blinks': valid_blinks,
                 'blinks_onset':valid_blink_onsets, 'timestamps':timestamps}
            # individual_group_blinks_data = pd.DataFrame(data=d)

            # append the dataframe to the list of all the groups.
            self.groups_blinks_with_timestamps.append(d)



    def calculate_blink_rate(self):
        for group in self.groups_blinks_with_timestamps:
            # calculate here the self.groups_blink_rate
            timestamps = group['timestamps']
            blinks_onset = group['blinks_onset']

            total_minutes_within_the_timespan = (timestamps.values[-1] - timestamps.values[0]).astype('timedelta64[m]')

            blink_rates = []

            for participant_number in range(group["group_size"]):
                individual_blinks_rate = sum(blinks_onset[participant_number]) / total_minutes_within_the_timespan.astype('int')
                # print(total_minutes_within_the_timespan, " ", sum(blinks_onset[participant_number]), " ", blink_rates)
                blink_rates.append(individual_blinks_rate)

            d = {'group_name': group["group_name"],
                 'group_size': group["group_size"],
                 'blink_rates': blink_rates}
            temp_series = pd.Series(d)

            self.groups_blink_rate = pd.concat([self.groups_blink_rate, temp_series], ignore_index=True)
        return

    def get_groups_blinks_data(self):
        return self.groups_blinks_with_timestamps

    def get_groups_blink_rate(self):
        if self.groups_blink_rate.empty:
            self.calculate_blink_rate()
            return self.groups_blink_rate
        else:
            return self.groups_blink_rate



# testing below to see if it works
if __name__ == "__main__":

    group_data_time_subset_filename = 'group_names_with_time_subsets.csv'
    blink_stats_subsets = BlinkStats(group_names_filename=group_data_time_subset_filename)
    blink_rates_subsets = blink_stats_subsets.get_groups_blink_rate()
    print(blink_rates_subsets)



    # groups_list_dyads = ["DYAD_2024_06_14_Seminar_Wue_Session_3_Group_5_TS", "DYAD_2024_06_14_Seminar_Wue_Session_1_Group_2_TS",
    #                      "DYAD_2024_05_07_Seminar_Wue_Session_2_Group_1_TS", "DYAD_2023_12_19_Seminar_Wue_Session_4_Group_5_TS",
    #                      "DYAD_2023_12_19_Seminar_Wue_Session_4_Group_2_TS", "DYAD_2023_12_19_Seminar_Wue_Session_4_Group_1_TS",
    #                      "DYAD_2023_12_19_Seminar_Wue_Session_2_Group_5_TS", "DYAD_2023_12_19_Seminar_Wue_Session_2_Group_4_TS",
    #                      "DYAD_2023_12_19_Seminar_Wue_Session_2_Group_2_TS", "DYAD_2023_12_19_Seminar_Wue_Session_1_Group_2_TS",
    #                      "DYAD_2023_11_06_Seminar_Munich_Session_2_Group_1_TS", "DYAD_2023_11_06_Seminar_Munich_Session_1_Group_1_TS"]
    #
    # groups_list_triads = ["TRIAD_2024_06_14_Seminar_Wue_Session_3_Group_2_TS", "TRIAD_2024_06_14_Seminar_Wue_Session_3_Group_1_TS",
    #                       "TRIAD_2024_06_14_Seminar_Wue_Session_2_Group_1_TS", "TRIAD_2024_06_14_Seminar_Wue_Session_1_Group_1_TS",
    #                       "TRIAD_2024_05_07_Seminar_Wue_Session_2_Group_4_TS", "TRIAD_2024_05_07_Seminar_Wue_Session_2_Group_2_TS",
    #                       "TRIAD_2023_12_19_Seminar_Wue_Session_3_Group_1_TS", "TRIAD_2023_12_19_Seminar_Wue_Session_2_Group_1_TS",
    #                       "TRIAD_2023_12_19_Seminar_Wue_Session_1_Group_1_TS", "TRIAD_2023_10_30_Seminar_Munich_No_VAD",
    #                       "TRIAD_2023_10_23_Seminar_Munich_Session_1_Group_1_TS"]
    #
    # blink_stats_dyads = BlinkStats(groups_list_dyads)
    # blink_rates_dyads = blink_stats_dyads.get_groups_blink_rate()
    # print(blink_rates_dyads)
    #
    # blink_stats_triads = BlinkStats(groups_list_triads)
    # blink_rates_triads = blink_stats_triads.get_groups_blink_rate()
    # print(blink_rates_triads)
    #
    # blinks_file_path = blink_stats_dyads.data_folder_path + 'blinks_rates_all_groups.csv'
    # blink_rates_file = open(blinks_file_path, 'a')
    # blink_rates_file.write(blink_rates_dyads.to_string())
    # blink_rates_file.write(blink_rates_triads.to_string())
    # blink_rates_file.close()
