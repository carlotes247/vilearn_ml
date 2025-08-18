# load the data into dfs
# transform the seconds column into the correct data type
# create the plot
# show and save the plot

import pandas as pd
import matplotlib.pyplot as plt

def convert_seconds_to_timestamp(df_with_seconds:pd.DataFrame,seconds_col_name:str='seconds_interaction') -> pd.DataFrame:
    df_with_seconds[seconds_col_name+'_ts'] = pd.to_timedelta(df_with_seconds[seconds_col_name], unit='s')

    return df_with_seconds

# returns a df with the timeline changed to avg seconds based on a set timeframe.
# By default the avg is calculated every 30 seconds.
# Important that the column name has the be the one with a timestamp data type (use the convert_seconds_to_timestamp first)
def resample_avg_seconds_using_timeframe(df_default_time_frequency:pd.DataFrame,seconds_col_name:str='seconds_interaction_ts',
                                     timeframe:int=30) -> pd.DataFrame:
    df_resampled = df_default_time_frequency.resample(str(timeframe)+'s', on = seconds_col_name).mean().reset_index()
    new_name = seconds_col_name[:-2]+'window'
    df_resampled[new_name] = df_resampled[seconds_col_name].dt.total_seconds()
    df_resampled.drop([seconds_col_name, seconds_col_name[:-3]], axis=1, inplace=True)
    return df_resampled

def get_grups_interaction_times():
    names_durations_df: pd.DataFrame = pd.read_csv('../data/group_durations_all_commas.csv', index_col=0)

    all_groups_interaction_time:list[float] = names_durations_df['duration_interaction']

    dyads_interaction_time:list[float] = names_durations_df.loc[names_durations_df['type']=='dyad', "duration_interaction"]
    triads_interaction_time:list[float] = names_durations_df.loc[names_durations_df['type']=='triad', "duration_interaction"]

    f_dyads_interaction_time: list[float] = names_durations_df.loc[(names_durations_df['group_formation']=='F')
                                                        & (names_durations_df['type']=='dyad'), "duration_interaction"]

    f_triads_interaction_time: list[float] = names_durations_df.loc[(names_durations_df['group_formation']=='F')
                                                        & (names_durations_df['type']=='triad'), "duration_interaction"]

    return [all_groups_interaction_time, dyads_interaction_time, triads_interaction_time, f_dyads_interaction_time, f_triads_interaction_time]

def get_groups_names_and_formation():
    # get the duration of each group
    names_durations_df: pd.DataFrame = pd.read_csv('data/group_durations_all_commas.csv', index_col=0)
    # dyads_names: list[str] = names_durations_df.loc[names_durations_df['type']=='dyad', "name"]
    # tridas_names: list[str] = names_durations_df.loc[names_durations_df['type']=='triad', "name"]
    f_groups_names: list[str] = names_durations_df.loc[names_durations_df['group_formation']=='F', "name"]
    l_groups_names: list[str] = names_durations_df.loc[names_durations_df['group_formation']=='Line', "name"]

    return f_groups_names, l_groups_names

def create_line_plot(file_path:str, timewindow:int = 30, save_plot = False, account_for_group_formation = False,
                     plot_dyads: bool = True, plot_triads: bool = True, plot_all_groups: bool = True):
    df_data = pd.read_csv(file_path, index_col=0)

    df_data_ts = convert_seconds_to_timestamp(df_data)
    df_resampled = resample_avg_seconds_using_timeframe(df_data_ts, timeframe=timewindow)
    df_resampled.set_index('seconds_interaction_window', inplace=True)

    #make a list with all the dyads
    col_dyads = list(df_resampled.filter(regex='dyad').columns)
    df_dyads = pd.DataFrame({'Dyads: Avg MG': df_resampled[col_dyads].mean(axis=1),
                             'Dyads: Std MG': df_resampled[col_dyads].std(axis=1)/2})

    df_triads = pd.DataFrame({'Triads: Avg MG': df_resampled.drop(col_dyads, axis=1).mean(axis=1),
                              'Triads: Std MG': df_resampled.drop(col_dyads, axis=1).std(axis=1)/2})
    df_avg_all_groups = pd.concat([df_dyads['Dyads: Avg MG'], df_triads['Triads: Avg MG']], axis=1)


    # get the duration_interaction, a list of all interaction times
    dyads_durations = get_grups_interaction_times()[1].sort_values(ascending=False).tolist()
    triads_durations = get_grups_interaction_times()[2].sort_values(ascending=False).tolist()

    ax = df_avg_all_groups.plot.line(color=['red', 'green'], figsize=(12,5))

    ax.fill_between(df_dyads.index, df_dyads['Dyads: Avg MG'] - df_dyads['Dyads: Std MG'],
                    df_dyads['Dyads: Avg MG'] + df_dyads['Dyads: Std MG'], facecolor='red', alpha=0.2)
    for current_duration in dyads_durations:
        current_index = dyads_durations.index(current_duration)
        offset = (current_index%2)/20 #adds an offset every other time so it can be readable
        text_y = 0.5 - offset
        line_y = 0.8 - offset
        ax.text(current_duration, text_y, current_index + 1)
        ax.axvline(current_duration, ymax=line_y, ymin=0, color='red', linestyle='--', alpha=0.8, linewidth=0.7)

    ax.fill_between(df_triads.index, df_triads['Triads: Avg MG'] - df_triads['Triads: Std MG'],
                    df_triads['Triads: Avg MG'] + df_triads['Triads: Std MG'], facecolor='green', alpha=0.2)
    for current_duration in triads_durations:
        current_index = triads_durations.index(current_duration)
        offset = (current_index % 2) / 20  # adds an offset every other time so it can be readable
        text_y = 0.1 - offset
        line_y = 0.3 - offset
        ax.text(current_duration, text_y, current_index + 1)
        ax.axvline(current_duration, ymax=line_y, ymin=0, color='green', linestyle='--', alpha=0.8, linewidth=0.7)

    plt.xlabel('Interaction Time in Seconds ('+ 'windows of ' +str(timewindow)+ 's)' )
    plt.ylabel('Mutual Gaze %')
    plt.title('Mutual Gaze in Dyads and Triads')
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    save_plot = False


    root_path= "../Recordings/SavedData/"
    dyads_df_path = root_path+"all_dyads_mutual_gaze_interaction_time.csv"
    triads_df_path = root_path+"all_triads_mutual_gaze_interaction_time.csv"
    all_groups_df_path = root_path+"all_groups_mutual_gaze_interaction_time.csv"

    # to do: make lists of the dyads and triads including their formation in order to discriminate over them when plotting
    # create_line_plot(dyads_df_path)
    # create_line_plot(triads_df_path)
    create_line_plot(all_groups_df_path, timewindow=5