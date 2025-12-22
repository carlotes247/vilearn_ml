import datetime
import statistics

import pandas as pd
from scipy import stats

import os
try:
    from data_reading.groups_manager import GroupsManager
except ImportError:
    import sys
    sys.path.append(os.getcwd())
    from data_reading.groups_manager import GroupsManager

import matplotlib.pyplot as plt
import json

class BlinkStats:

    # list of dicts that has for each group:
    # "group_name" string, the group name
    # "group_size" int, group size
    # 'group_blink_collisions' list of BlinkCollisionsList;
    #              this collisions list will have one element if a dyad (collisions of P1 to P2) or 3 elements if a triad (P1-P2, P2-P3, P1-P3)
    # 'blinks_async_250ms_bins', dict with the keys: -1500, -1250, -1000, -750, -500, -250, 0, 250, 500, 750, 1000, 1250
    #              and the value an int for the number of synced blinks with that time lag between blinks onset. The keys represent
    #              the lower bound, the upper bound of that bin is calculated by adding 250ms.
    # 'blinks_durations_ms', dict with keys: 1,2,3 based on the group size; the value is a list of floats representing   each blink in ms;
    # 'blinks_dataframe': index: timestamp, columns (bool): 'P{1, 2, or 3}_valid_blink_onsets', 'P{1, 2, or 3}_valid_blinks'
    # 'interaction_duration_seconds' : how long was the valid interaction for that group in seconds
    groups_blinks_with_timestamps: list[dict]

    # df with the columns: 'group_name':str, group_size: int, 'count_blinks_participant_reference':float, 'count_sync_blinks':float,
    # 'percent_sync_blinks':float. For triads, these values are averaged.
    group_synced_blinks_percent = pd.DataFrame(columns=['group_name', 'group_size', 'count_blinks_participant_reference',
                                                      'count_sync_blinks', 'percent_sync_blinks'])

    # list of with the group name, group size, blink rate (list of 2 or 3)
    groups_blink_rate = pd.DataFrame(columns=['group_name', 'group_size', 'P1_blink_rate', 'P2_blink_rate',
                                        'P3_blink_rate'])

    # df for storing info on the blinks async in ms; for triads this is averaged per each couple and then for all three couples.
    # "group_name" string, the group name
    # "group_size" int, group size
    # "count_sync_blinks", float, how many synced blinks are there avged for triads
    # "avg_blinks_async_ms", float, the average time in ms (blink asynchrony) between the onset time of blinks of partners.
    groups_avg_blinks_async_ms = pd.DataFrame(columns=['group_name', 'group_size', 'count_sync_blinks',
                                                       'avg_blinks_async_ms'])


    #dict containing the dyads' synced blinks' timelag between the participant's blinks onset
    # dyads_blinks_async_250ms_bins = {'-1.50:-1.25':0, '-1.25:-1.00':0,'-1.00:-0.75':0,'-0.75:-0.50':0,'-0.50:-0.25':0,'-0.25:0.00':0,
    #                                  '0.00:0.25':0,'0.25:0.50':0,'0.50:0.75':0,'0.75:1.00':0,'1.00:1.25':0,'1.25:1.50':0}
    dyads_blinks_async_250ms_bins = {-1500:0, -1250:0, -1000:0, -750:0, -500:0, -250:0,
                                     0:0, 250:0, 500:0, 750:0, 1000:0, 1250:0}


    #when dyads, the P3 is -100
    groups_blink_durations_ms = pd.DataFrame(columns=['group_name', 'group_name_short', 'group_size', 'timestamp_seconds',
                                                      'blink_duration_ms_P1', "blink_duration_ms_P2", 'blink_duration_ms_P3', 'blink_duration_ms_avg'])

    path_going_up_two_folders = "../../"
    path_prefix_file = path_going_up_two_folders + "data/_path_prefix.txt"
    data_folder_path = path_going_up_two_folders + "data/"
    if not os.path.exists(path_prefix_file):
        path_prefix_file = os.path.join(os.getcwd(), "data/_path_prefix.txt")
    if not os.path.exists(data_folder_path):
        data_folder_path = os.path.join(os.getcwd(), "data/")

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
        fullpath = os.path.join(self.data_folder_path, group_names_filename)
        data_for_calculating_subsets = pd.read_csv(fullpath, sep=';')

        for index, row in data_for_calculating_subsets.iterrows():

            current_group_dataframe = pd.DataFrame()
            my_groups_manager = GroupsManager(self.path_prefix_file, self.data_folder_path, specific_group=row['Group_Name_Long'],
                                              onlyTorch=False, load_individual_p_files=False, print_all_stats=False,
                                              print_blink_stats=False,
                                              all_groups_names_path='', use_async=False,  print_debug = False)
            group_data = my_groups_manager.groups[0]
            valid_blinks, valid_blink_onsets = group_data.group_features_csv_loader.extract_valid_blinks_frames()

            current_blinks_durations_ms = dict(group_data.group_features_csv_loader.get_blinks_durations_per_participants())

            timestamps_string = group_data.group_features_csv_loader.raw_data['TSGroupNTP']
            timestamps = pd.to_datetime(timestamps_string, utc=True, format='%Y-%m-%d %H:%M:%S.%f')

            current_group_valid_interaction_time_seconds = 0

            if "DYAD" in row['Group_Name_Long']:
                group_size = 2
            else:
                group_size = 3

            # calculate the subsets for each participant
            for participant in range(group_size):

                # get the blink info for each participant
                participant_valid_blinks = valid_blinks[participant]
                participant_valid_blink_onsets = valid_blink_onsets[participant]

                # turn the blink info into a df
                participant_valid_blinks_df = pd.DataFrame({f'P{participant+1}_valid_blinks': participant_valid_blinks})
                participant_valid_blink_onsets_df = pd.DataFrame({f'P{participant+1}_valid_blink_onsets': participant_valid_blink_onsets})

                # add the blink info to the dataframe as another column
                current_group_dataframe = pd.concat([current_group_dataframe, participant_valid_blinks_df,
                                                  participant_valid_blink_onsets_df], axis='columns')

            # after all participants data finished, add the timestamp to the df
            current_group_dataframe = pd.concat([current_group_dataframe, timestamps], axis='columns')

            # drop any repeated timestamps
            current_group_dataframe = current_group_dataframe.drop_duplicates(subset=['TSGroupNTP'])

            # set the index to the timestamp to easily get a subset of it based on the correct group conversation
            current_group_dataframe = current_group_dataframe.set_index('TSGroupNTP')

            # sorting the index (timestamps) as the next fuction won't work on a non-soted list. even though it is sorted
            current_group_dataframe.sort_index(inplace=True)

            # get the correct subset of the dataframe
            start_time_timestamp = pd.to_datetime(row['TS_Start_Interaction'], utc=True, format='%Y-%m-%d %H:%M:%S.%f')
            # get the index of the start timestamp for the subset window
            index_of_start_timestamp = current_group_dataframe.index.get_indexer([start_time_timestamp], method='nearest')
            start_timestamp_available_in_df = current_group_dataframe.index[index_of_start_timestamp[0]]

            end_time_timestamp = pd.to_datetime(row['TS_End_Interaction'], utc=True, format='%Y-%m-%d %H:%M:%S.%f')
            # get the index of the end timestamp for the subset window
            index_of_end_timestamp = current_group_dataframe.index.get_indexer([end_time_timestamp], method='nearest')
            end_timestamp_available_in_df = current_group_dataframe.index[index_of_end_timestamp[0]]

            # get the dataframe subset
            valid_subset_data = current_group_dataframe[start_timestamp_available_in_df:end_timestamp_available_in_df]

            #calculate the diff between the start and end to get the num of seconds in the interaction
            current_group_valid_interaction_time_seconds = (end_time_timestamp - start_time_timestamp).total_seconds()

            # for calculating the bins for the time lags for each group
            current_blinks_async_250ms_bins = {-1500: 0, -1250: 0, -1000: 0, -750: 0, -500: 0, -250: 0,
                                               0: 0, 250: 0, 500: 0, 750: 0, 1000: 0, 1250: 0}
            # TODO: add to this valid subset data info about the sync blinks; see notebook for more notes
            if group_size == 2:
                collisions_dyad = (group_data.group_features_csv_loader.get_blink_onset_collisions
                                   (valid_subset_data['P1_valid_blink_onsets'].tolist(),valid_subset_data['P2_valid_blink_onsets'].tolist(),
                                    valid_subset_data.index.tolist(), "P1", "P2"))
                group_collisions = [collisions_dyad]

            #   calculate the time lag bins for the synced blinks:
                for collision in collisions_dyad.collisions:
                    bin_timelag = collision.delta_ms/250
                    if bin_timelag<0:
                        bin_multiplier = int(bin_timelag-1)
                    else:
                        bin_multiplier = int(bin_timelag)
                    if bin_multiplier == 6: bin_multiplier = 5 #this is the upper bound of the 1250 to 1500 bin
                    if bin_multiplier == -7: bin_multiplier = -6 #this is the lower bound of the -1500 to -1250 bin
                    index = bin_multiplier*250
                    current_blinks_async_250ms_bins[index] += 1

                #this updates the blinks per 250ms bin to create a rathe of the blinks per minute; this is done by dividing the value in each bin to the len of the interaction in that group
                total_minutes_within_the_timespan = (end_timestamp_available_in_df - start_timestamp_available_in_df).total_seconds() / 60
                current_blinks_async_250ms_bins.update(
                    (bin_timelag, value / total_minutes_within_the_timespan) for bin_timelag, value in current_blinks_async_250ms_bins.items())

            else:
                collisions_triad_P1P2 = (group_data.group_features_csv_loader.get_blink_onset_collisions
                                   (valid_subset_data['P1_valid_blink_onsets'].tolist(),
                                    valid_subset_data['P2_valid_blink_onsets'].tolist(),
                                    valid_subset_data.index.tolist(), "P1", "P2"))
                collisions_triad_P1P3 = (group_data.group_features_csv_loader.get_blink_onset_collisions
                                   (valid_subset_data['P1_valid_blink_onsets'].tolist(),
                                    valid_subset_data['P3_valid_blink_onsets'].tolist(),
                                    valid_subset_data.index.tolist(), "P1", "P3"))
                collisions_triad_P3P2 = (group_data.group_features_csv_loader.get_blink_onset_collisions
                                   (valid_subset_data['P3_valid_blink_onsets'].tolist(),
                                    valid_subset_data['P2_valid_blink_onsets'].tolist(),
                                    valid_subset_data.index.tolist(), "P3", "P2"))
                group_collisions = [collisions_triad_P1P2, collisions_triad_P1P3, collisions_triad_P3P2]

                #calculate the time lag bins for the synced blinks:
                for collisions_data in group_collisions:
                    for collision in collisions_data.collisions:
                        bin_timelag = collision.delta_ms / 250
                        if bin_timelag < 0:
                            bin_multiplier = int(bin_timelag - 1)
                        else:
                            bin_multiplier = int(bin_timelag)
                        if bin_multiplier == 6: bin_multiplier = 5  # this is the upper bound of the 1250 to 1500 bin
                        if bin_multiplier == -7: bin_multiplier = -6  # this is the lower bound of the -1500 to -1250 bin
                        index = bin_multiplier * 250
                        current_blinks_async_250ms_bins[index] += 1

                # dividing by 3 the value for each bin as each triad has 3 couples, and the blink sync has been calculated per couple
                # current_blinks_async_250ms_bins.update((bin_timelag, value/3) for bin_timelag, value in current_blinks_async_250ms_bins.items())

                #this updates the blinks per 250ms bin to create a rather of the blinks per minute; this is done by dividing the value in each bin to the len of the interaction in that group
                total_minutes_within_the_timespan = (end_timestamp_available_in_df - start_timestamp_available_in_df).total_seconds() / 60
                current_blinks_async_250ms_bins.update(
                    (bin_timelag, value / total_minutes_within_the_timespan) for bin_timelag, value in current_blinks_async_250ms_bins.items())

            groups_blink_durations_ms = self.calculate_blink_duration_for_each_minute(current_blinks_durations_ms, current_group_dataframe,
                                                                                      start_timestamp_available_in_df,end_timestamp_available_in_df,
                                                                                      row['Group_Name'])

            # put all the info into a dictionary and then add it to a list
            d = {'group_name_short': row['Group_Name'],
                'group_name': row['Group_Name_Long'], 'group_size': group_size, 'group_blink_collisions': group_collisions,
                 'blinks_async_250ms_bins':current_blinks_async_250ms_bins, 'blinks_dataframe': valid_subset_data,
                 'blinks_durations_ms':current_blinks_durations_ms,
                 'interaction_duration_seconds':current_group_valid_interaction_time_seconds,
                 'groups_blink_durations_ms': groups_blink_durations_ms}
            # append the dataframe to the list of all the groups.
            self.groups_blinks_with_timestamps.append(d)

    # I don't think  this is ever used
    def populate_groups_blinks_dataset_with_a_the_full_timeline (self, group_names:list[str]):
         for group_name in group_names:
             current_group_dataframe = pd.DataFrame()
        #     my_groups_manager = GroupsManager(self.path_prefix_file, self.data_folder_path, specific_group= group_name,
        #                                       onlyTorch=False, load_individual_p_files=False, print_all_stats=False,
        #                                       print_blink_stats=False)
        #     group_data = my_groups_manager.groups[0]
        #     valid_blinks, valid_blink_onsets = group_data.group_features_csv_loader.extract_valid_blinks_frames()
        #
        #     timestamps_string = group_data.group_features_csv_loader.raw_data['TSGroupNTP']
        #     timestamps = pd.to_datetime(timestamps_string, utc=True, format='%Y-%m-%d %H:%M:%S.%f')
        #
        #     if "DYAD" in group_name:
        #         group_size = 2
        #     else:
        #         group_size = 3
        #
        #     for participant in range(group_size):
        #         # get the blink info for each participant
        #         participant_valid_blinks = valid_blinks[participant]
        #         participant_valid_blink_onsets = valid_blink_onsets[participant]
        #
        #         # turn the blink info into a df
        #         participant_valid_blinks_df = pd.DataFrame({f'P{participant+1}_valid_blinks': participant_valid_blinks})
        #         participant_valid_blink_onsets_df = pd.DataFrame({f'P{participant+1}_valid_blink_onsets': participant_valid_blink_onsets})
        #
        #         # add the blink info to a dataframe
        #         current_group_dataframe = pd.concat([current_group_dataframe, participant_valid_blinks_df,
        #                                           participant_valid_blink_onsets_df], axis='columns')
        #
        #     # after all participants data finished, add the timestamp to the df
        #     current_group_dataframe = pd.concat([current_group_dataframe, timestamps], axis='columns')
        #
        #     # drop any repeated timestamps
        #     current_group_dataframe = current_group_dataframe.drop_duplicates(subset=['TSGroupNTP'])
        #
        #     # set the index to the timestamp
        #     current_group_dataframe = current_group_dataframe.set_index('TSGroupNTP')
        #
        #     # put all the info into a ndarray and then add it to a dataframe
        #     d = {'group_name': group_name, 'group_size': group_size, 'blinks_dataframe': current_group_dataframe}
        #
        #     self.groups_blinks_with_timestamps.append(d)

    def calculate_mean_blink_asynchrony_ms(self):
        for group in self.groups_blinks_with_timestamps:
            if group['group_size'] == 2:
                total_async_ms = 0
                collisions = group['group_blink_collisions'][0].collisions # this is a list of BlinkCollision

                for collision in collisions:
                    total_async_ms = total_async_ms + abs(collision.delta_ms)

                avg_async_ms = total_async_ms/len(collisions)

                # create the df for the current dyad
                avg_blinks_async_ms = pd.DataFrame(
                    {'group_name': group['group_name'], 'group_size': group['group_size'],
                     'count_sync_blinks': len(collisions), 'avg_blinks_async_ms': avg_async_ms}, index=[0])

                # add the blink info to a dataframe
                self.groups_avg_blinks_async_ms = pd.concat(
                    [self.groups_avg_blinks_async_ms, avg_blinks_async_ms], ignore_index=True)
            else:
                sum_values = {'total_sync_blinks':0, 'total_avg_blinks_async_ms':0}

                for collisions_data in group['group_blink_collisions']:
                    # now this is for one couple within the triad (3 couples in total)
                    total_async_ms = 0
                    for collision in collisions_data.collisions:
                        total_async_ms = total_async_ms + abs(collision.delta_ms)


                    avg_async_ms = total_async_ms/len(collisions_data.collisions)
                    # add this to the dictionary sum; this will be divided by 3 later on
                    sum_values['total_sync_blinks'] += len(collisions_data.collisions)
                    sum_values['total_avg_blinks_async_ms'] += avg_async_ms

                # create the df for the current triad
                avg_blinks_async_ms = pd.DataFrame(
                    {'group_name': group['group_name'], 'group_size': group['group_size'],
                     'count_sync_blinks': sum_values['total_sync_blinks']/3,
                     'avg_blinks_async_ms': sum_values['total_avg_blinks_async_ms']/3}, index=[0])
                # add it now to the df
                self.groups_avg_blinks_async_ms = pd.concat(
                    [self.groups_avg_blinks_async_ms, avg_blinks_async_ms], ignore_index=True)


    #returns the dataframes: mean_async_vals_per_250ms_bins (containing mean vals for dyads, triads, the result of dyads_1sampleTtest and the stars for the rest (dyads_1sampleTtest_stars)
    def calculate_blink_asynchrony_ms_per_time_windows(self):
        # df for dyads 'blinks_async_250ms_bins'
        dyads_async_250ms_bins = pd.DataFrame(columns=[-1500, -1250, -1000, -750, -500, -250, 0, 250, 500, 750, 1000, 1250])
        # df for triads
        triads_async_250ms_bins = pd.DataFrame(columns=[-1500, -1250, -1000, -750, -500, -250, 0, 250, 500, 750, 1000, 1250])

        # populate the df dyads_async_250ms_bins and df triads_async_250ms_bins from each group dict
        for group in self.groups_blinks_with_timestamps:
            if group['group_size'] == 2:
                current_async_dict = group['blinks_async_250ms_bins']
                current_async_df = pd.DataFrame([current_async_dict])
                dyads_async_250ms_bins = pd.concat([dyads_async_250ms_bins, current_async_df], ignore_index=True)
            else:
                current_async_dict = group['blinks_async_250ms_bins']
                current_async_df = pd.DataFrame([current_async_dict])
                triads_async_250ms_bins = pd.concat([triads_async_250ms_bins, current_async_df], ignore_index=True)

        # calculate the chance level of blinks happening in that time bin
        # and get the mean for each time bin
        mean_async_vals_per_250ms_bins = pd.DataFrame()
        dyads_median = 0
        triads_median = 0
        for column in dyads_async_250ms_bins.columns.tolist():
            dyads_median += statistics.median(dyads_async_250ms_bins[column].tolist())
            triads_median += statistics.median(triads_async_250ms_bins[column].tolist())
            mean_async_vals_per_250ms_bins.loc['dyads', column] = statistics.mean(dyads_async_250ms_bins[column].tolist())
            mean_async_vals_per_250ms_bins.loc['triads', column] = statistics.mean(triads_async_250ms_bins[column].tolist())

        # dividing the median by the number of bin (12) to calculate the chance level
        chance_level_dyads = dyads_median/12
        chance_level_triads = triads_median/12

        # do the on-sample-t-test and save the result in a df, along with the corresponding *s
        for column in dyads_async_250ms_bins.columns.tolist():
            dyads_p_value = stats.ttest_1samp(dyads_async_250ms_bins[column].tolist(),
                              popmean=chance_level_dyads, alternative='greater').pvalue

            dyads_statistic = stats.ttest_1samp(dyads_async_250ms_bins[column].tolist(),
                                               popmean=chance_level_dyads, alternative='greater').statistic
            # dyads_p_value = dyads_1samp_ttest_full_result.pvalue

            mean_async_vals_per_250ms_bins.loc['dyads_1sampleTtest', column] = dyads_p_value
            mean_async_vals_per_250ms_bins.loc['dyads_1sampleTtest_statistic', column] = dyads_statistic
            mean_async_vals_per_250ms_bins.loc['dyads_1sampleTtest_stars', column] = self.get_stars_for_p_value(dyads_p_value)

            triads_p_value = stats.ttest_1samp(triads_async_250ms_bins[column].tolist(),
                                                               popmean=chance_level_triads, alternative='greater').pvalue
            triads_statistic = stats.ttest_1samp(triads_async_250ms_bins[column].tolist(),
                                                              popmean=chance_level_triads, alternative='greater').statistic
            # triads_p_value = triads_1samp_ttest_full_result.pvalue
            mean_async_vals_per_250ms_bins.loc['triads_1sampleTtest', column] = triads_p_value
            mean_async_vals_per_250ms_bins.loc['triads_1sampleTtest_full', column] = triads_statistic
            mean_async_vals_per_250ms_bins.loc['triads_1sampleTtest_stars', column] = self.get_stars_for_p_value(triads_p_value)

        # plot the boxplots for dyads and triads
        # x_axis_labels = ['-1500:\n-1250', '-1250:\n-1000', '-1000:\n-750', '-750:\n-500', '-500:\n-250', '-250:\n0', '0:\n250', '250:\n500', '500:\n750', '750:\n1000', '1000:\n1250', '1250:\n1500']
        x_axis_labels = ['-1500:-1250', '-1250:-1000', '-1000:-750', '-750:-500', '-500:-250', '-250:0', '0:250', '250:500', '500:750', '750:1000', '1000:1250', '1250:1500']

        plt.rcParams['figure.dpi'] = 500
        fig, (dyads_plot, triads_plot) = plt.subplots(1, 2)
        plt.subplots_adjust(hspace=0.55, bottom=0.3)

        fig.set_size_inches(10, 3.5)
        boxplot = dyads_plot.boxplot(dyads_async_250ms_bins, tick_labels=x_axis_labels)
        for median in boxplot['medians']:
            median.set_color('blue')
        dyads_plot.tick_params('x', rotation=65)
        dyads_plot.axhline(y=chance_level_dyads, color='r', linestyle='-')
        y_upper_limit = dyads_async_250ms_bins.max() + 1
        dyads_plot.set_ylim(0, 10) #changing this to 10 to ave the same y axis for both dyads and triads; y_upper_limit.max())#increasing the y axis a bit so that the * would fit
        dyads_plot.set_title('Dyads', fontsize=14)
        dyads_plot.set_ylabel('Blink Rate', fontsize=11)

        #add the stars above the max value in the list
        for column in dyads_async_250ms_bins.columns.tolist():
            # ax.text requires x,y and the text; x values for boxplots works as an index of the boxplot itself, not the value that exists of the x axis. hence the first boxplot is at index 1, second index 2 and so on
            dyads_plot.text(dyads_async_250ms_bins.columns.tolist().index(column) + 1,
                            max(dyads_async_250ms_bins[column])+0.25,
                            mean_async_vals_per_250ms_bins.loc['dyads_1sampleTtest_stars', column],
                            horizontalalignment='center')

        triads_plot.boxplot(triads_async_250ms_bins, tick_labels=x_axis_labels)
        triads_plot.tick_params('x', rotation=65)
        triads_plot.axhline(y=chance_level_triads, color='r', linestyle='-')
        y_upper_limit = triads_async_250ms_bins.max() + 1
        triads_plot.set_ylim(0, y_upper_limit.max())#increasing the y axis a bit so that the * would fit
        # add the stars above the max value in the list
        for column in triads_async_250ms_bins.columns.tolist():
            triads_plot.text(triads_async_250ms_bins.columns.tolist().index(column) + 1,
                             max(triads_async_250ms_bins[column])+0.25,
                             mean_async_vals_per_250ms_bins.loc['triads_1sampleTtest_stars', column],
                             horizontalalignment='center')

        triads_plot.set_title('Triads', fontsize=14)
        # triads_plot.set_ylabel('Blink Rate', fontsize=11)


        #create new fig for the line graph with the means
        fig_mean, ax_mean = plt.subplots()
        # ax_mean.set_xticks(dyads_async_250ms_bins.columns.tolist())
        ax_mean.plot(dyads_async_250ms_bins.columns.tolist(), mean_async_vals_per_250ms_bins.loc['dyads'].values, color='b', label='Dyads')
        ax_mean.plot(triads_async_250ms_bins.columns.tolist(), mean_async_vals_per_250ms_bins.loc['triads'].values, color='g', label='Triads')
        ax_mean.set_xticks(dyads_async_250ms_bins.columns.tolist())

        plt.show()
        return mean_async_vals_per_250ms_bins, dyads_async_250ms_bins, triads_async_250ms_bins

    def calculate_synced_blinks_percent_from_all_blinks(self):
        for group in self.groups_blinks_with_timestamps:
            if group['group_size'] == 2:
                collisions_data = group['group_blink_collisions'][0]

                # calculate the total collisions
                total_collisions = len(collisions_data.collisions)

                ref_count_blink_onsets = group['blinks_dataframe'][f'{collisions_data.ref_name}_valid_blink_onsets'].tolist().count(True)
                adv_count_blink_onsets = group['blinks_dataframe'][f'{collisions_data.adv_name}_valid_blink_onsets'].tolist().count(True)

                # tTotal synced blinks over all blinks in the group. All the blinks are from all participants.
                # The synced ones are happening on both participants, hence the multiplication by 2
                percent_sync_blinks = (total_collisions*2)/(ref_count_blink_onsets + adv_count_blink_onsets)

                synced_blinks_group_info = pd.DataFrame({'group_name':group['group_name'], 'group_size':group['group_size'],
                                                          'count_blinks_participant_reference':ref_count_blink_onsets,
                                                          'count_sync_blinks':total_collisions, 'percent_sync_blinks':percent_sync_blinks},
                                                        index=[0])

                # add the blink info to a dataframe
                self.group_synced_blinks_percent = pd.concat([self.group_synced_blinks_percent, synced_blinks_group_info], ignore_index=True)

            else:
                sum_values = {'count_blink_onsets':0, 'total_collisions':0, 'percent_sync_blinks':0}
                for collisions_data in group['group_blink_collisions']:
                    # calculate the total collisions
                    total_collisions = len(collisions_data.collisions)

                    ref_count_blink_onsets = group['blinks_dataframe'][
                        f'{collisions_data.ref_name}_valid_blink_onsets'].tolist().count(True)
                    adv_count_blink_onsets = group['blinks_dataframe'][
                        f'{collisions_data.adv_name}_valid_blink_onsets'].tolist().count(True)

                    # tTotal synced blinks over all blinks in the group. All the blinks are from all participants.
                    # The synced ones are happening on both participants, hence the multiplication by 2
                    percent_sync_blinks = (total_collisions * 2) / (ref_count_blink_onsets + adv_count_blink_onsets)

                    #udpate sum_values
                    sum_values['count_blink_onsets'] += ref_count_blink_onsets
                    sum_values['total_collisions'] += total_collisions
                    sum_values['percent_sync_blinks'] += percent_sync_blinks

                synced_blinks_group_info = pd.DataFrame(
                    {'group_name': group['group_name'], 'group_size': group['group_size'],
                     'count_blinks_participant_reference': sum_values['count_blink_onsets']/3,
                     'count_sync_blinks': sum_values['total_collisions']/3,
                     'percent_sync_blinks': sum_values['percent_sync_blinks']/3}, index =[0])
                # add the blink info to a dataframe
                self.group_synced_blinks_percent = pd.concat(
                    [self.group_synced_blinks_percent, synced_blinks_group_info], ignore_index=True)


    def calculate_blink_rate(self, sampling: int = 60, save_blinks_per_minute=False):
        if save_blinks_per_minute:
            all_blinks_per_minute_df = pd.DataFrame()
        for group in self.groups_blinks_with_timestamps:
            # calculate here the self.groups_blink_rate
            blinks_df = group['blinks_dataframe']
            timestamps = blinks_df.index.tolist()

            total_minutes_within_the_timespan = (timestamps[-1] - timestamps[0]).total_seconds()/sampling

            blink_rates = []

            for participant_number in range(group["group_size"]):

                participant_blink_onsets = blinks_df[f'P{participant_number+1}_valid_blink_onsets'].tolist()
                individual_blinks_rate = sum(participant_blink_onsets) / total_minutes_within_the_timespan
                blink_rates.append(individual_blinks_rate)

            if group["group_size"] == 2:
                temp_df = pd.DataFrame({'group_name': group['group_name'], 'group_size': group['group_size'],
                                        'P1_blink_rate': blink_rates[0], 'P2_blink_rate': blink_rates[1]}, index = [0])
            else:
                temp_df = pd.DataFrame({'group_name': group['group_name'], 'group_size': group['group_size'],
                                        'P1_blink_rate': blink_rates[0], 'P2_blink_rate': blink_rates[1],
                                        'P3_blink_rate': blink_rates[2]}, index=[0])

            if save_blinks_per_minute:
                df_resampled = blinks_df.resample(f'{sampling}s').sum()
                df_resampled.drop(['P1_valid_blinks', 'P2_valid_blinks'], axis=1, inplace=True)
                df_resampled['group_name']=group['group_name_short']
                all_blinks_per_minute_df = pd.concat([all_blinks_per_minute_df, df_resampled])


            self.groups_blink_rate = pd.concat([self.groups_blink_rate, temp_df], ignore_index=True)
        if save_blinks_per_minute:
            all_blinks_per_minute_df.to_csv("../../data/blink_per_minute_all_groups.csv")

        return

    #returns: group name, group size, P1_avg blink duration in ms, P2_avg ..., P3_avg..., group_avg_blink duration_ms,
    def get_groups_blink_duration(self):
        group_blink_duration_df = pd.DataFrame(columns=['group_name', 'group_size', 'P1_mean_blink_duration_ms',
                                        'P2_mean_blink_duration_ms', 'P3_mean_blink_duration_ms','group_mean_blink_duration_ms'])
        participants_avg_durations = [-1.0, -1.0, -1.0]

        for group in self.groups_blinks_with_timestamps:
            for participant_index in range(group["group_size"]):
                participants_avg_durations[participant_index] = statistics.fmean(group['blinks_durations_ms'][participant_index])

            if group['group_size'] == 2:
                d = pd.DataFrame({'group_name':group['group_name'], 'group_size':group['group_size'],
                                  'P1_mean_blink_duration_ms': participants_avg_durations[0],
                                  'P2_mean_blink_duration_ms': participants_avg_durations[1],
                                  'group_mean_blink_duration_ms':(participants_avg_durations[0] +
                                                                  participants_avg_durations[1])/2}, index=[0])
            else:
                d = pd.DataFrame({'group_name': group['group_name'], 'group_size': group['group_size'],
                                  'P1_mean_blink_duration_ms': participants_avg_durations[0],
                                  'P2_mean_blink_duration_ms': participants_avg_durations[1],
                                  'P3_mean_blink_duration_ms': participants_avg_durations[2],
                                  'group_mean_blink_duration_ms': (participants_avg_durations[0] +
                                                                   participants_avg_durations[1] +
                                                                   participants_avg_durations[2]) / 3}, index=[0])
            group_blink_duration_df = pd.concat([group_blink_duration_df, d], ignore_index=True)

        return group_blink_duration_df

    def calculate_blink_duration_for_each_minute(self, current_blinks_durations_ms, df_blinks,
                                                 start_interaction_time, end_interaction_time, group_name )->pd.DataFrame:
        save_blinks_duration_per_minute = False
        all_blink_durations = pd.DataFrame({'dummy': 1}, index=df_blinks.index)
        # consider only interataction time
        # get the dataframe subset
        interaction_time_blink_durations = all_blink_durations[start_interaction_time:end_interaction_time]
        interaction_time_blink_durations = interaction_time_blink_durations.resample('60s').mean()

        for person in range(len(current_blinks_durations_ms)):
            current_person_blinks = pd.DataFrame({'blink_onsets': df_blinks['P'+str(person+1)+'_valid_blink_onsets']}, index=df_blinks.index)
            current_person_blinks = current_person_blinks.loc[current_person_blinks.blink_onsets, :]
            # print(current_person_blinks.shape[0], len(current_blinks_durations_ms[person]))

            if current_person_blinks.shape[0]>len(current_blinks_durations_ms[person]):#dropping the last values that make the difference between the list of blink durations and the df
                current_person_blinks.drop(index=current_person_blinks.index[
                    - (current_person_blinks.shape[0] - len(current_blinks_durations_ms[0]))], axis=0, inplace=True)
                current_person_blinks['P' + str(person + 1) + '_durations'] = current_blinks_durations_ms[person]
            else:
                current_person_blinks['P'+str(person+1)+'_durations'] = current_blinks_durations_ms[person][:current_person_blinks.shape[0]]

            current_person_blinks = current_person_blinks[start_interaction_time:end_interaction_time]
            current_person_blinks = current_person_blinks.resample('60s').mean()
            interaction_time_blink_durations = pd.concat([interaction_time_blink_durations, current_person_blinks], axis=1)

        if len(current_blinks_durations_ms) ==2:
            interaction_time_blink_durations['group_avg_blink_duration'] = (interaction_time_blink_durations['P1_durations']+
                                                               interaction_time_blink_durations['P2_durations'])/2
        else:
            interaction_time_blink_durations['group_avg_blink_duration'] = (interaction_time_blink_durations['P1_durations'] +
                                                               interaction_time_blink_durations['P2_durations'] +
                                                               interaction_time_blink_durations['P3_durations']) / 3

        interaction_time_blink_durations.drop(columns =['dummy','blink_onsets'], axis=1, inplace=True)
        if save_blinks_duration_per_minute:
            interaction_time_blink_durations['group_name'] = group_name
            interaction_time_blink_durations.to_csv("../../data/blink_duration_per_minute_all_groups.csv",mode='a', header=True)

        return interaction_time_blink_durations

    def get_group_names_and_durations(self):
        groups_name= []
        groups_duration = []

        for group in self.groups_blinks_with_timestamps:
            groups_name.append(group['group_name'])
            groups_duration.append(group['interaction_duration_seconds'])
        temp_dict = {'names': groups_name, 'durations': groups_duration}
        df = pd.DataFrame(temp_dict)
        return df

    def get_groups_blinks_data(self):
        return self.groups_blinks_with_timestamps

    def get_groups_blink_rate(self):
        if self.groups_blink_rate.empty:
            self.calculate_blink_rate()
            return self.groups_blink_rate
        else:
            return self.groups_blink_rate

    def get_group_synced_blink_percent(self):
        if self.group_synced_blinks_percent.empty:
            self.calculate_synced_blinks_percent_from_all_blinks()
            return self.group_synced_blinks_percent
        else:
            return self.group_synced_blinks_percent

    def get_group_avg_blinks_async_ms(self):
        if self.groups_avg_blinks_async_ms.empty:
            self.calculate_mean_blink_asynchrony_ms()
            return self.groups_avg_blinks_async_ms
        else:
            return self.groups_avg_blinks_async_ms


    #determine the number of stars based on the p value. Not sure if it's an automated way to do this, but this should do for now
    def get_stars_for_p_value(self, p_value: float):
        if p_value is not None:
            if p_value < 0.0001:
                return '****'
            elif p_value < 0.001:
                return '***'
            elif p_value < 0.01:
                return '**'
            elif p_value < 0.05:
                return '*'
            else:
                return ' '

    #trying to print to file so it can easily be read afterwards. but the data is a list of dicts, and the dicts also have a df. so I'm not sure if this is the best way to save to file. perhaps the structures should be changed and then reconsider how to save to file
    def write_group_data_to_csv(self, filename_to_dump_group_data):
        file = open(filename_to_dump_group_data, 'w+')
        for group_data_dict in self.groups_blinks_with_timestamps:
            file.write(json.dumps(group_data_dict))

    def read_group_data_from_csv(self, filename_to_dump_group_data):
        file = open(filename_to_dump_group_data, 'r')






# testing below to see if it works
if __name__ == "__main__":

    # group_data_time_subset_filename = 'group_names_with_time_subsets.csv'
    group_data_time_subset_filename = 'group_names_with_time_subsetsFullVERSION.csv'
    blink_stats_subsets = BlinkStats(group_names_filename=group_data_time_subset_filename)

    # get blinks rate
    blink_rates_df = blink_stats_subsets.get_groups_blink_rate()
    # # add blink RATES data to file
    blink_rates_file_path = blink_stats_subsets.data_folder_path + 'blink_rates_all_groups.csv'
    blink_rates_file = open(blink_rates_file_path, 'a')
    blink_rates_file.write(blink_rates_df.to_string())
    blink_rates_file.close()


    # # get blinks duration
    blink_durations_df = blink_stats_subsets.get_groups_blink_duration()
    # add blink DURATIONS data to file
    blinks_durations_file_path = blink_stats_subsets.data_folder_path + 'blink_durations_all_groups.csv'
    blink_durations_file = open(blinks_durations_file_path, 'a')
    # blink_durations_file.write(blink_durations_df.to_string())
    blink_durations_file.close()

    # get blinks sync percent
    #synced_blinks_percent = blink_stats_subsets.get_group_synced_blink_percent()
    # synced_blinks_percent_file_path = blink_stats_subsets.data_folder_path + 'synced_blinks_percent_all_groups.csv'
    # synced_blinks_percent_file = open(synced_blinks_percent_file_path, 'a')
    # synced_blinks_percent_file.write(synced_blinks_percent.to_string())
    # synced_blinks_percent_file.close()

    # get avg blinks async time in ms for each group
    # avg_blinks_async_ms = blink_stats_subsets.get_group_avg_blinks_async_ms()
    # avg_blinks_async_ms_file_path = blink_stats_subsets.data_folder_path + 'avg_blinks_async_ms_all_groups.csv'
    # avg_blinks_async_ms_file = open(avg_blinks_async_ms_file_path, 'a')
    # avg_blinks_async_ms_file.write(avg_blinks_async_ms.to_string())
    # avg_blinks_async_ms_file.close()


    #dyads_ms_per_time_window_df, triads_ms_per_time_window_df, mean_vals_ms_per_time_window_df = blink_stats_subsets.calculate_blink_asynchrony_ms_per_time_windows()
    # async_time_window_ms_file_path = blink_stats_subsets.data_folder_path + 'async_time_window_ms_file_path.csv'
    # async_time_window_ms_file = open(async_time_window_ms_file_path, 'a')
    # async_time_window_ms_file.write(dyads_ms_per_time_window_df.to_string())
    # async_time_window_ms_file.write(triads_ms_per_time_window_df.to_string())
    # async_time_window_ms_file.write(mean_vals_ms_per_time_window_df.to_string())
    # async_time_window_ms_file.close()

    #get the duration of each group
    names_durations_df = blink_stats_subsets.get_group_names_and_durations()
    groups_durations_file_path = blink_stats_subsets.data_folder_path + 'group_durations_all_groups.csv'
    blink_durations_file = open(groups_durations_file_path, 'a')
    # blink_durations_file.write(names_durations_df.to_string())
    blink_durations_file.close()


