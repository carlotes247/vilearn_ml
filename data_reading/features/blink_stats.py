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
    def __init__(self, group_names: list[str]):
        self.groups_blinks_with_timestamps = []
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
# if __name__ == "__main__":
#     groups_list = ["TRIAD_2023_10_30_Seminar_Munich_No_VAD","TRIAD_2023_10_23_Seminar_Munich_Session_1_Group_1"]
#     test_blink_stats = BlinkStats(groups_list)
#     blink_rates = test_blink_stats.get_groups_blink_rate()
#     print(blink_rates)