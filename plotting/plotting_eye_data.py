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

    l_dyads_interaction_time: list[float] = names_durations_df.loc[(names_durations_df['group_formation']=='Line')
                                                        & (names_durations_df['type']=='dyad'), "duration_interaction"]

    f_triads_interaction_time: list[float] = names_durations_df.loc[(names_durations_df['group_formation']=='F')
                                                        & (names_durations_df['type']=='triad'), "duration_interaction"]

    l_triads_interaction_time: list[float] = names_durations_df.loc[(names_durations_df['group_formation'] == 'Line')
                                                                    & (names_durations_df[
                                                                           'type'] == 'triad'), "duration_interaction"]
    groups_inter_time={"all_groups":all_groups_interaction_time,
                       "dyads":dyads_interaction_time, "triads":triads_interaction_time,
                       "dyads_l":l_dyads_interaction_time, "dyads_f":f_dyads_interaction_time,
                       "triads_l":l_triads_interaction_time, "triads_f":f_triads_interaction_time}
    return groups_inter_time

def get_groups_names_and_formation()->pd.DataFrame:
    # get the duration of each group
    names_durations_df: pd.DataFrame = pd.read_csv('../data/group_durations_all_commas.csv', index_col=0)
    df_names_formations = names_durations_df [['group_formation', 'type']]
    return df_names_formations

# returns a list of the group names as they are in tha main df, such as dyad01_MGP2P1 or similar; it filters for groups size (dyads or triads) and for formation (Line or F)
def get_formation_grups_names_from_eye_df(main_df:pd.DataFrame, group_formation_df:pd.DataFrame, group_size:str,
                                          group_formation:str)->list:
    name_columns = []
    for name in list(group_formation_df.loc[(group_formation_df['group_formation'] == group_formation) & (
            group_formation_df['type'] == group_size)].index):
        name_columns.extend(list(main_df.filter(regex=name).columns))
    return name_columns

def plot_avg_and_std_to_existing_graph(ax, main_df:pd.DataFrame, df_column_avg:str, df_column_std:str,
                           list_interaction_duration_descending_order:list,
                           line_colour:str, text_y:float, line_y:float,
                           fill_alpha:float, fill_color:str):

    ax.plot(main_df[df_column_avg], color=line_colour, label=df_column_avg)
    ax.fill_between(main_df.index, main_df[df_column_avg] - main_df[df_column_std],
                    main_df[df_column_avg] + main_df[df_column_std], facecolor=fill_color, alpha=fill_alpha, label = df_column_std)

    for current_duration in list_interaction_duration_descending_order:
        current_index = list_interaction_duration_descending_order.index(current_duration)
        offset = (current_index % 2) / 20  # adds an offset every other time so it can be readable
        text_y_offset = text_y - offset
        line_y_offset = line_y - offset
        ax.text(current_duration, text_y_offset, current_index + 1, color=line_colour)
        ax.axvline(current_duration, ymax=line_y_offset, ymin=0, color=line_colour, linestyle='--', alpha=0.8, linewidth=0.7)

    return ax

#potentially, this can be called also with a list of triad names. but so far I think I can just creat the triads names based on the number of triads existing (14)
def sum_triads_MG_per_group(main_df: pd.DataFrame, group_names:list):
    df_triads_MG_summed = pd.DataFrame(index=main_df.index)
    for triad_name in group_names:
        columns_MG = list(main_df.filter(regex=triad_name).columns)
        df_triads_MG_summed[columns_MG[0]] = main_df[columns_MG].sum(axis=1,min_count=1) #keeping the name of the first MG as it is used further in the code and it works like that for the dyads too
    return df_triads_MG_summed


def create_line_plot(file_path:str, timewindow:int = 30, save_plot = False,
                     dyads:bool = True, triads:bool = True, separate_by_group_formation:bool = True,
                     sum_triads_for_mutual_gaze:bool = True,
                     y_axis_text:str='', figure_title:str=''):

    df_data = pd.read_csv(file_path, index_col=0)
    df_data_ts = convert_seconds_to_timestamp(df_data)
    df_resampled = resample_avg_seconds_using_timeframe(df_data_ts, timeframe=timewindow)
    df_resampled.set_index('seconds_interaction_window', inplace=True)
    df_group_formations = get_groups_names_and_formation()

    fig, ax = plt.subplots(figsize=(12,5))

    if dyads:
        if separate_by_group_formation:
            col_dyads_l = get_formation_grups_names_from_eye_df(df_resampled, df_group_formations, 'dyad', 'Line')
            col_dyads_f = get_formation_grups_names_from_eye_df(df_resampled, df_group_formations, 'dyad', 'F')

            df_dyads_l = pd.DataFrame({'Dyads-L Avg': df_resampled[col_dyads_l].mean(axis=1),
                                     'Dyads-L Std': df_resampled[col_dyads_l].std(axis=1) / 2})
            df_dyads_f = pd.DataFrame({'Dyads-F Avg': df_resampled[col_dyads_f].mean(axis=1),
                                       'Dyads-F Std': df_resampled[col_dyads_f].std(axis=1) / 2})
            dyads_durations_f = get_grups_interaction_times()['dyads_f'].sort_values(ascending=False).tolist()
            dyads_durations_l = get_grups_interaction_times()['dyads_l'].sort_values(ascending=False).tolist()

            ax = plot_avg_and_std_to_existing_graph(ax=ax, main_df=df_dyads_l, df_column_avg='Dyads-L Avg',
                                                    df_column_std='Dyads-L Std',
                                                    list_interaction_duration_descending_order=dyads_durations_l,
                                                    line_colour='red', text_y=0.5, line_y=0.8,
                                                    fill_alpha=0.2, fill_color='red')

            ax = plot_avg_and_std_to_existing_graph(ax=ax, main_df=df_dyads_f, df_column_avg='Dyads-F Avg',
                                                    df_column_std='Dyads-F Std',
                                                    list_interaction_duration_descending_order=dyads_durations_f,
                                                    line_colour='orange', text_y=0.5, line_y=0.8,
                                                    fill_alpha=0.2, fill_color='orange')

        else:
            col_dyads = list(df_resampled.filter(regex='dyad').columns)
            df_dyads = pd.DataFrame({'Dyads Avg': df_resampled[col_dyads].mean(axis=1),
                                     'Dyads Std': df_resampled[col_dyads].std(axis=1) / 2})
            dyads_durations = get_grups_interaction_times()['dyads'].sort_values(ascending=False).tolist()

            ax = plot_avg_and_std_to_existing_graph(ax=ax, main_df = df_dyads, df_column_avg='Dyads Avg',
                                               df_column_std='Dyads Std',
                                               list_interaction_duration_descending_order = dyads_durations,
                                               line_colour = 'red', text_y =  0.5, line_y=0.8,
                                               fill_alpha = 0.2, fill_color='red')

    if triads:
        if (sum_triads_for_mutual_gaze):
            triads_group_names = df_group_formations.loc[df_group_formations['type'] == 'triad'].index
            df_resampled = sum_triads_MG_per_group(df_resampled, triads_group_names)

        if separate_by_group_formation:
            col_triads_l = get_formation_grups_names_from_eye_df(df_resampled, df_group_formations, 'triad', 'Line')
            col_triads_f = get_formation_grups_names_from_eye_df(df_resampled, df_group_formations, 'triad', 'F')

            df_triads_l = pd.DataFrame({'Triads-L Avg': df_resampled[col_triads_l].mean(axis=1),
                                       'Triads-L Std': df_resampled[col_triads_l].std(axis=1) / 2})
            df_triads_f = pd.DataFrame({'Triads-F Avg': df_resampled[col_triads_f].mean(axis=1),
                                       'Triads-F Std': df_resampled[col_triads_f].std(axis=1) / 2})
            triads_durations_f = get_grups_interaction_times()['triads_f'].sort_values(ascending=False).tolist()
            triads_durations_l = get_grups_interaction_times()['triads_l'].sort_values(ascending=False).tolist()

            ax = plot_avg_and_std_to_existing_graph(ax=ax, main_df=df_triads_l, df_column_avg='Triads-L Avg',
                                                    df_column_std='Triads-L Std',
                                                    list_interaction_duration_descending_order=triads_durations_l,
                                                    line_colour='green', text_y=0.08, line_y=0.2,
                                                    fill_alpha=0.2, fill_color='green')

            ax = plot_avg_and_std_to_existing_graph(ax=ax, main_df=df_triads_f, df_column_avg='Triads-F Avg',
                                                    df_column_std='Triads-F Std',
                                                    list_interaction_duration_descending_order=triads_durations_f,
                                                    line_colour='blue', text_y=0.2, line_y=0.4,
                                                    fill_alpha=0.2, fill_color='blue')

        else:
            col_triads = list(df_resampled.filter(regex='triad').columns)
            df_triads = pd.DataFrame({'Triads Avg': df_resampled[col_triads].mean(axis=1),
                                      'Triads Std': df_resampled[col_triads].std(axis=1) / 2})

            triads_durations = get_grups_interaction_times()["triads"].sort_values(ascending=False).tolist()

            ax = plot_avg_and_std_to_existing_graph(ax=ax, main_df = df_triads, df_column_avg='Triads Avg',
                                               df_column_std='Triads Std',
                                               list_interaction_duration_descending_order = triads_durations,
                                               line_colour = 'green', text_y =  0.1, line_y=0.3,
                                               fill_alpha = 0.2, fill_color='green')

    plt.xlabel('Interaction Time in Seconds ('+ 'windows of ' +str(timewindow)+ 's)' )
    plt.ylabel(y_axis_text)
    plt.title(figure_title)
    plt.tight_layout()
    ax.legend(loc='best')
    plt.show()


if __name__ == "__main__":
    save_plot = False


    root_path= "../Recordings/SavedData/"
    all_groups_MG_df_path = root_path+"all_groups_mutual_gaze_interaction_time.csv"

    create_line_plot(all_groups_MG_df_path, timewindow=20, separate_by_group_formation=True,
                     y_axis_text="Mutual Gaze %",
                     figure_title="Mutual Gaze")