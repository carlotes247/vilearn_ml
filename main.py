import numpy as np
from data_reading.groups_manager import GroupsManager
from training.vilearn_train import ViLearnTrainLogic
import torch.utils.data
from plotting.plotterClass import PlotterClass
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches # to display textured bars in plot legends
from textwrap import wrap # to wrap plot labels because group names are too long
import pandas as pd
from data_reading.features.blink_stats import BlinkStats
import datetime 
from data_reading.group import Group

#region METHODS

def calculate_stats(list, stringTag, stringMeasurement):
    """
    (Uses Numpy) Prints stats about a csv file
    """ 
    # Calculate the average of the list
    # Calculate the sum of the list
    total = sum(list)
    # Calculate the length of the list
    count = len(list)
    # Calculate the average of the list
    average = total / count
    # Print the average
    print("--------------- " + stringTag + "STATS -----------------")
    print(stringTag + "List Entries: " + str(count))
    print(stringTag + "List Avg: " + str(average) + stringMeasurement)
    deltaArray = np.array(list)
    print(stringTag + "Deltas Mean: " + str(deltaArray.mean()) + stringMeasurement)
    print(stringTag + "Deltas Min: " + str(deltaArray.min()) + stringMeasurement)
    print(stringTag + "Deltas Max: " + str(deltaArray.max()) + stringMeasurement)
    print(stringTag + "Deltas Median: " + str(np.median(deltaArray)) + stringMeasurement)
    print(stringTag + "Percentage of deltas above average: " + str(np.count_nonzero(deltaArray > average) / count * 100) + " %")
    print(stringTag + "Percentage of deltas above 500ms: " + str(np.count_nonzero(deltaArray > 500) / count * 100) + " %")
    print(stringTag + "Percentage of deltas above 900ms: " + str(np.count_nonzero(deltaArray > 900) / count * 100) + " %")

def read_data_and_calculate_stats (reader, fileName, stringTag):
    """
    Calls all functions written so far to understand the data
    """ 
    listOfDeltas = reader.getDeltasBetweenTimestamps(fileName)
    calculate_stats(listOfDeltas, stringTag, "ms")
    reader.getEyeTrackingData(fileName)

# gets called with a series of timestamps and a list for a particular person
def calculate_blink_per_minute_rate(timestamps, valid_blink_onsets):
    # total_minutes_within_the_timespan = (timestamps.nsmallest(1) - timestamps.nlargest(1))#.minutes
    a = timestamps.values[-1]
    b = timestamps.values[0]
    total_minutes_within_the_timespan = (timestamps.values[-1] - timestamps.values[0]).astype('timedelta64[m]')
    blinks_rate = sum(valid_blink_onsets) / total_minutes_within_the_timespan.astype('int')
    print (total_minutes_within_the_timespan, " ", sum(valid_blink_onsets), " ", blinks_rate)

#endregion 

# create csv_reader obj 
#reader = ViLearnCSVDataLoader()
# Read deltas between timestampts and calculate stats. Pass it a path to a local csv file. Try not to push the csv files not to clutter the repo
#readDataAndCalculateStats(reader, "HCM153.csv", "(Subscription to Eye Event) ")
#readDataAndCalculateStats(reader, "goodUserName.csv", "(Fixed Update) ")
#readDataAndCalculateStats(reader, "HCM153_Administrator2023-09-26__09-39-37.103.csv", "(2023-09-26__09-39-37.103) ")
#readDataAndCalculateStats(reader, "HCM153_Administrator2023-09-26__09-49-31.770.csv", "(023-09-26__09-49-31.770) ")

# Files from 05 October 2023 long test
#readDataAndCalculateStats(reader, "LAPTOP-UJR5JBB1_carlo2023-10-05__15-53-28.036.csv", "(Carlos) ") # this one is massive (200MB) avoid pushing
#readDataAndCalculateStats(reader, "Thomas_HCM153_Administrator2023-10-05__16-13-46.496.csv", "(Thomas) ")
#read_data_and_calculate_stats(reader, "DESKTOP-QTU96C2_vilearn2023-10-22__10-33-39.028.csv", "(Laura) ")

# added this comment to check if git hooks work

if __name__ == "__main__":
    #region VARS
    # Config flags (I might want to move them somewhere else, leave here for the moment)
    load_groups_mngr: bool = False
    train_torch: bool = False
    load_individual_participant_files: bool = False
    use_async: bool = False
    modify_dataframes: bool = False
    my_groups_manager: GroupsManager
    # Debug flags
    print_debug: bool = True
    print_all_stats_p_files: bool = False
    print_blink_stats_p_files: bool = False
    # Plotting flags
    plot_eye_openess: bool = False
    plot_group_duration: bool = False


    # Testing loading data logic 12 April 2024
    path_prefix_file = "data/_path_prefix.txt"
    data_folder_path = "data/"
    all_groups_names_file_path = "data/list_of_all_usable_groups.txt"
    # Leave empty to load data from all groups
    #specific_group = "TRIAD_2023_10_30_Seminar_Munich_No_VAD"
    specific_group = ""

    #endregion

    #region MAIN CODE
    
    if load_groups_mngr:
        # Load all groups
        my_groups_manager = GroupsManager(path_prefix_file, data_folder_path, specific_group=specific_group, 
                                    all_groups_names_path=all_groups_names_file_path, onlyTorch=train_torch,
                                    load_individual_p_files=load_individual_participant_files, 
                                    print_all_stats=print_all_stats_p_files, print_blink_stats=print_blink_stats_p_files,
                                    print_debug=print_debug, use_async=use_async)
    
    if modify_dataframes:
        # MODIFYING DATAFRAMES
        recordings_TS_df = pd.read_csv("data/recording_times_group_info.csv")
        interactions_TS_df = pd.read_csv("data/group_names_with_time_subsetsFullVERSION.csv", sep=';')
        durations_df = pd.read_csv("data/group_durations_all_commas.csv")
        # adding new column to durations_df
        durations_df['offset_recording_interaction_start'] = 0
        for recording_time_group in recordings_TS_df['long_name']:
            record_start_time: datetime.datetime = datetime.datetime.strptime(recordings_TS_df[recordings_TS_df['long_name'] == recording_time_group]['start_recording'].values[0], '%Y-%m-%d %H:%M:%S.%f')
            interaction_start_time: datetime.datetime = datetime.datetime.strptime(interactions_TS_df[interactions_TS_df['Group'] == recording_time_group]['Start'].values[0], '%Y-%m-%d %H:%M:%S.%f')
            offset_record_interaction_times: datetime.timedelta = interaction_start_time - record_start_time
            durations_df.loc[durations_df['long_name'] == recording_time_group, "offset_recording_interaction_start"] = offset_record_interaction_times.total_seconds()
            print(f"{durations_df.loc[durations_df['long_name'] == recording_time_group, 'offset_recording_interaction_start'].values[0]}")
        print("df modified")
    
    if train_torch:
        # Get all groups data as a single dataset
        dataset = my_groups_manager.get_concat_groups_torch_dataset()
        # Split into training and eval datasets
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        training_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        data_loader_train = my_groups_manager.get_vilearn_torch_dataloader(training_dataset)
        data_loader_eval = my_groups_manager.get_vilearn_torch_dataloader(eval_dataset)
        training_class = ViLearnTrainLogic()
        # Train model
        training_class.train_lstm(data_loader_train.group_dataloader, data_loader_eval.group_dataloader)
    if plot_eye_openess:
        groupData = my_groups_manager.groups[0]    
        # this is the raw data from the participant file without it being synced with the other participants
        if (load_individual_participant_files):
            raw_left_eye_openess_data_p1 = groupData.participants[0].movement_data.leftEyeOpeness
            timestamp_unsync = groupData.participants[0].movement_data.overallTsNtpString
        # this is the synced data
        timestamps_string = groupData.group_features_csv_loader.raw_data['TSGroupNTP']
        timestamps = pd.to_datetime(timestamps_string, utc=True, format='%Y-%m-%d %H:%M:%S.%f')
        left_eye_openess_p1 = groupData.group_features_csv_loader.raw_data['LeftEyeOpennesP1']
        left_eye_openess_confidence_p1 = groupData.group_features_csv_loader.raw_data['LeftEyeOpennesConfidenceP1']
        right_eye_openess_p1 = groupData.group_features_csv_loader.raw_data['RightEyeOpennesP1']
        right_eye_openess_confidence_p1 = groupData.group_features_csv_loader.raw_data['RightEyeOpennesConfidenceP1']
        blink_p1 = groupData.group_features_csv_loader.raw_data['BlinkP1']
        valid_blinks, valid_blink_onsets = groupData.group_features_csv_loader.extract_valid_blinks_frames()
        valid_blinks_p1 = valid_blinks[0]
        valid_blink_onsets_p1 = valid_blink_onsets[0]
        valid_blink_onsets_p2 = valid_blink_onsets[1]
        collisions_p1_p2 = groupData.group_features_csv_loader.get_blink_onset_collisions(valid_blink_onsets_p1, valid_blink_onsets_p2, timestamps, "P1", "P2")
        collisions_df = collisions_p1_p2.get_collisions_with_all_TS(timestamps)

        # calculate_blink_per_minute_rate(timestamps, valid_blink_onsets_p1)


        print("open plot window")
        collisions_df.plot()
        plt.title("Blink Sync P1-P2")
        plt.xlabel('Time')
        plt.ylabel('Blinks')

        #plt.plot(collisions_df.index, collisions_df[collisions_df.columns[1]].values, label = "Reference")
        #plt.plot(collisions_df.index, collisions_df[collisions_df.columns[2]].values, label = "Adversary")
        #plt.plot(timestamps, collisions_df)        

        plt.show()

        #plotter = PlotterClass()
        #plotter.plot_eye_blinks(my_groups_manager.groups[0])
    if plot_group_duration:
        print("attempting to plot...")
        # group_data_time_subset_filename = 'group_names_with_time_subsets.csv'
        #group_data_time_subset_filename : str = 'group_names_with_time_subsetsFullVERSION.csv'
        #blinks_stats_instance : BlinkStats = BlinkStats(group_names_filename=group_data_time_subset_filename)
        #blinks_stats_instance : BlinkStats = BlinkStats
        
        #get the duration of each group
        #names_durations_df : pd.DataFrame = blinks_stats_instance.get_group_names_and_durations()
        names_durations_df : pd.DataFrame = pd.read_csv('data/group_durations_all_commas.csv', index_col=0)

        # sort duration from higher to lower
        names_durations_df = names_durations_df.sort_values(by='duration_interaction', ascending=False)
        # distiguish duration dyads vs triads by colour
        color_dyad : str = 'red'
        color_triad : str = 'green'
        color_groups : list[str] = names_durations_df['type'].map({'dyad':color_dyad, 'triad':color_triad}).to_list()
        # distinguish f vs line formation with texture or pattern
        tex_groups : list[str] = names_durations_df['group_formation'].map({'F':'', 'Line':'/'}).to_list()

        plot_bars : bool = False
        plot_duration_lines : bool = True
        # configure plots
        fig,ax = plt.subplots()
        if plot_bars:
            # create bars
            ax.bar(names_durations_df.index, names_durations_df['duration_interaction'],color=color_groups,hatch=tex_groups)
            ax.set_title('Group Durations by Type and Formation')
            # configure legend
            circ1 = mpatches.Patch(facecolor=color_dyad,hatch='',label='dyad')
            circ2= mpatches.Patch(facecolor=color_triad,hatch='',label='triad')
            circ3 = mpatches.Patch(facecolor='white',hatch='///',label='line_formation')
            ax.legend(handles = [circ1, circ2, circ3], loc=1)
            # rotate labels for better readability
            plt.xticks(rotation=90)
        if plot_duration_lines:
            # create bars
            ax.barh(names_durations_df.index, names_durations_df['duration_recording'],color=color_groups,hatch=tex_groups)
            ax.set_title('Group Recording vs Interaction Duration')
            # configure legend
            circ1 = mpatches.Patch(facecolor=color_dyad,hatch='',label='dyad')
            circ2= mpatches.Patch(facecolor=color_triad,hatch='',label='triad')
            circ3 = mpatches.Patch(facecolor='white',hatch='///',label='line_formation')
            ax.legend(handles = [circ1, circ2, circ3], loc=1)
            # rotate labels for better readability
            plt.xticks(rotation=90)
        # draw plot
        plt.show()

        list_group_start_TS = []

        # TODO: Use dummy recording time for the moment
        # TODO: Extract first and last TS from each group features file
        for group in my_groups_manager.groups:
            if (not group.group_feature_data_loaded):
                continue
            print(f"{group.group_name}, init recording: {group.group_feature_frames[0].ts_group},  end recording: {group.group_feature_frames[-1].ts_group_string}, duration {group.recording_duration.total_seconds()}")
            print(f"group found in index list name {True if group.group_name in names_durations_df['long_name'].to_list() else False}")
        # TODO: Line graphs with all tasks duration. The Y axis has each group; the X axis has the time, where 0 is the recording time,
        # and the line will start when the interaction (conversation) starts. This way we can see if there is a large time between 
        # the recording start time and the interaction start time.  Differentiate between line- and F- formation, and between the group size (dyad vs triad).
        

        print("done!")
    #endregion
