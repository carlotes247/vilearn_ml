import numpy as np
from data_reading.groups_manager import GroupsManager
from training.vilearn_train import ViLearnTrainLogic
import torch.utils.data
from plotting.plotterClass import PlotterClass
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches # to display textured bars in plot legends
import matplotlib.ticker as ticker # to control axis ticks
import matplotlib.lines as lines # to display lines in plot legends
from textwrap import wrap # to wrap plot labels because group names are too long
import pandas as pd
from data_reading.features.blink_stats import BlinkStats
import datetime 
from data_reading.group import Group
from preprocessing.engagement.engagements_manager import EngagementsManager

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

def plot_task_engagement_method(fig, ax, df: pd.DataFrame, label: str, legend_info, hvline: float, color: str, text_y: float = 0):
    y = df['avg_task_eng']
    y_std = df['std_avg_task_eng']
    y_std = y_std/2 # divide by two to plot around center of y
    num_dyads = df.groups.drop_duplicates()
    text_y = text_y if text_y > 0 else y.max()
    ax.fill_between(x, y-y_std, y+y_std, facecolor = color, alpha=0.3)
    ax.plot(x, y, color)
    for i, num in enumerate(num_dyads):
        index_num_dyads = num_dyads.index[i]
        pos_x: float = x[index_num_dyads].item()
        eng_level: float = df[df['seconds'] == pos_x]['avg_task_eng'].values[0]
        ax.text(pos_x, text_y, f'{num}')
        ax.axvline(pos_x, ymax=hvline, ymin=0, color=color, linestyle='--', alpha=0.8, linewidth=0.7)
    # configure legend
    line1_legend_d = lines.Line2D([0], [0], color=color, lw=1, label=f'Task Engagement {label}')
    line2_legend_d = lines.Line2D([0], [0], color=color, lw=0.7, linestyle='--', alpha=0.7, label=f'Num Groups in Avg {label}')
    legend_info.extend([line1_legend_d, line2_legend_d])

def plot_task_engagement_formation(fig, ax, group_names: list[str], df: pd.DataFrame, label: str, legend_info, hvline: float, color: str, text_y: float = 0):
    cols = df.columns[df.columns.str.contains('|'.join(group_names))]
    df_formation: pd.DataFrame = pd.DataFrame(df[cols] )
    avg = df_formation.mean(axis=1)
    std = df_formation.std(axis=1)
    num_groups = df_formation.count(axis=1)
    df_formation['avg_task_eng'] = avg
    df_formation['std_avg_task_eng'] = std
    df_formation['groups'] = num_groups
    df_formation['seconds'] = df['seconds']
    plot_task_engagement_method(fig=fig, ax=ax, df=df_formation, label=label, legend_info=legend_info, hvline=hvline, color=color, text_y=text_y)   

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
    plot_group_duration_plots: bool = True # True if you want any plot to appear
    descriptive_stats: bool = False
    plot_total_duration_bars: bool = False
    plot_offset_duration_lines: bool = False
    plot_task_engagement: bool = True # true if you want any task engagement plot
    plot_task_engagement_dyads: bool = True # for dyads task engagement
    plot_task_engagement_triads: bool = False # for triads task engagement
    plot_task_engagement_in_interaction_time: bool = True # to plot in interaction time instead of recording time
    discriminate_group_formation: bool = True # to plot depending on group formation (F or Line)

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
    #region PLOTTING
    if plot_group_duration_plots:
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
        # list of groups on F formation
        mask_F_formation = names_durations_df['group_formation'] == 'F'
        F_group_names: list[str] = names_durations_df[mask_F_formation].index.values
        # list of groups on Line formation
        mask_Line_formation = names_durations_df['group_formation'] == 'Line'
        Line_group_names: list[str] = names_durations_df[mask_Line_formation].index.values

        if descriptive_stats: 
            # mean dyad line
            mean_dyad_Line = names_durations_df[(names_durations_df['type']=='dyad') & (names_durations_df['group_formation']=='Line')].describe()
            # mean dyad F formation
            mean_dyad_F = names_durations_df[(names_durations_df['type']=='dyad') & (names_durations_df['group_formation']=='F')].describe()
            # mean triad line
            mean_triad_Line = names_durations_df[(names_durations_df['type']=='triad') & (names_durations_df['group_formation']=='Line')].describe()
            # mean traid F formation
            mean_triad_F = names_durations_df[(names_durations_df['type']=='triad') & (names_durations_df['group_formation']=='F')].describe()
            print(f"dyad Line mean is: {mean_dyad_Line}")
            print(f"dyad F mean is: {mean_dyad_F}")
            print(f"triad Line mean is: {mean_triad_Line}")
            print(f"triad F mean is: {mean_triad_F}")

        # configure plots
        fig,ax = plt.subplots()
        #region PLOT DURATION BARS
        if plot_total_duration_bars:
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
        #enregion
        #region PLOT OFFSET DURATION LINES
        if plot_offset_duration_lines:         
            # Line graphs with all tasks duration. The Y axis has each group; the X axis has the time, where 0 is the recording time,
            # and the line will start when the interaction (conversation) starts. This way we can see if there is a large time between 
            # the recording start time and the interaction start time.  Differentiate between line- and F- formation, and between the group size (dyad vs triad).
            y: float = 0
            for index, row in names_durations_df.iterrows():
                # First line: recording duration starts at 0
                x1: list[float] = [0, row['duration_interaction']]
                # Second line: interaction duration starts at offset
                x2: list[float] = [row['offset_recording_interaction_start'], row['duration_interaction']]
                
                color: str = color_dyad
                hatch: str = ""
                if row['type'] == 'triad': color = color_triad
                if row['group_formation'] == 'Line': hatch = '/'
                
                # Use barh to plot horizontal bars
                ax.barh(y=y, width=x1[1], left=x1[0], color=color, hatch=hatch, alpha=0.5, edgecolor='black')
                ax.barh(y=y, width=x2[1], left=x2[0], color=color, hatch=hatch, alpha=0.5, edgecolor='black')

                # plt.plot(x1, [y,y], label=f'{index}', color=color, hatch=hatch, alpha=0.5, linewidth=7.0)
                # plt.plot(x2, [y+1, y+1], label=f'{index}', color=color, hatch=hatch, alpha=0.5, linewidth=7.0)
                y += 1

            # Customize y-ticks
            ax.set_yticks(range(len(names_durations_df['duration_interaction'])))
            ax.set_yticklabels([index for index, row in names_durations_df.iterrows()])
            #ax.yaxis.set_major_locator(ticker.MultipleLocator(base=2))
            ax.set_xlabel('Time')
            ax.set_title('Overlapping Bars with Variable Start and End')

            # configure legend
            circ1 = mpatches.Patch(facecolor=color_dyad,hatch='',label='dyad')
            circ2= mpatches.Patch(facecolor=color_triad,hatch='',label='triad')
            circ3 = mpatches.Patch(facecolor='white',hatch='///',label='line_formation')
            ax.legend(handles = [circ1, circ2, circ3], loc=1)

            plt.grid(axis='x')
            plt.tight_layout()

            plt.xlabel('Seconds')
            plt.ylabel('Group')
            plt.title('Group Durations by Type and Formation')
            plt.grid(True)
        #endregion
        #region PLOT TASK ENG
        if plot_task_engagement:
            eng_mngr: EngagementsManager = EngagementsManager(False)            
            #eng_mngr.df_avg_eng_all['average_value'].plot()
            # select interaction df depending on flag
            df_eng_all = eng_mngr.df_avg_eng_all if not plot_task_engagement_in_interaction_time else eng_mngr.df_avg_eng_all_interaction              
            x = df_eng_all['seconds'] 
            legend_info = []
            # dyads
            if plot_task_engagement_dyads:
                df_eng_dyads = eng_mngr.df_avg_eng_dyads if not plot_task_engagement_in_interaction_time else eng_mngr.df_avg_eng_dyads_interaction
                if discriminate_group_formation:
                    plot_task_engagement_formation(fig=fig, group_names=F_group_names, ax=ax, df=df_eng_dyads, label='Dyads F formation', legend_info=legend_info, hvline=0.88, color=color_dyad)
                    plot_task_engagement_formation(fig=fig, group_names=Line_group_names, ax=ax, df=df_eng_dyads, label='Dyads Line formation', legend_info=legend_info, hvline=0.39, color='darkred', text_y=0.3)
                else:
                    plot_task_engagement_method(fig=fig, ax=ax, df=df_eng_dyads, label='Dyads', legend_info=legend_info, hvline=0.88, color=color_dyad)                
            # triads
            if plot_task_engagement_triads:
                df_eng_triads = eng_mngr.df_avg_eng_triads if not plot_task_engagement_in_interaction_time else eng_mngr.df_avg_eng_triads_interaction
                if discriminate_group_formation:
                    plot_task_engagement_formation(fig=fig, group_names=F_group_names, ax=ax, df=df_eng_triads, label='Triads F formation', legend_info=legend_info, hvline=0.88, color='mediumseagreen', text_y=0.74)
                    plot_task_engagement_formation(fig=fig, group_names=Line_group_names, ax=ax, df=df_eng_triads, label='Triads Line formation', legend_info=legend_info, hvline=0.75, color='darkgreen', text_y=0.62)
                else:
                    plot_task_engagement_method(fig=fig, ax=ax, df=df_eng_triads, label='Triads', legend_info=legend_info, hvline=0.92, color=color_triad)

            condition_text: str = ""
            if plot_task_engagement_dyads: condition_text += " Dyads"
            if plot_task_engagement_triads: condition_text += " Triads"
            if plot_task_engagement_in_interaction_time: condition_text += " Interaction Time"
            ax.legend(handles = legend_info, loc=0)
            plt.xlabel('Seconds')
            plt.ylabel(f'Task Engagement{condition_text}')
            plt.title(f'Average Task Engagement{condition_text}')
        # endregion
        # draw plot
        plt.show()

        list_group_start_TS = []        

        print("done!")
    #endregion
    #endregion