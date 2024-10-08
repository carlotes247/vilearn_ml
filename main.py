import numpy as np
from data_reading.groups_manager import GroupsManager
from training.vilearn_train import ViLearnTrainLogic
import torch.utils.data
from plotting.plotterClass import PlotterClass
import matplotlib.pyplot as plt
import pandas as pd

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
    # Config flags (I might want to move them somewhere else, leave here for the moment)
    train_torch = False
    load_individual_participant_files = False
    print_all_stats_p_files = False
    print_blink_stats_p_files = False
    plot_eye_openess = True

    # Testing loading data logic 12 April 2024
    path_prefix_file = "data/_path_prefix.txt"
    data_folder_path = "data/"
    # Leave empty to load data from all groups
    specific_group = "TRIAD_2023_10_30_Seminar_Munich_No_VAD"
    # Load all groups
    my_groups_manager = GroupsManager(path_prefix_file, data_folder_path, specific_group=specific_group, 
                                      onlyTorch=train_torch, load_individual_p_files=load_individual_participant_files, 
                                      print_all_stats=print_all_stats_p_files, print_blink_stats=print_blink_stats_p_files)
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
        print("open plot window")
        collisions_df.plot()
        plt.title("Blink Sync")
        plt.xlabel('Time')
        plt.ylabel('Blinks')
        #plt.plot(collisions_df.index, collisions_df[collisions_df.columns[1]].values, label = "Reference")
        #plt.plot(collisions_df.index, collisions_df[collisions_df.columns[2]].values, label = "Adversary")
        #plt.plot(timestamps, valid_blinks_p1)        
        plt.show()
        #plotter = PlotterClass()
        #plotter.plot_eye_blinks(my_groups_manager.groups[0])
    print("done!")


