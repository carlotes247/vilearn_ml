# load the data into dfs
# transform the seconds column into the correct data type
# create the plot
# show and save the plot
import re

import pandas as pd
import matplotlib.pyplot as plt
import os

class PlottingEyeData:

    all_groups_gaze_counts_df: pd.DataFrame
    dyads_gaze_counts_df: pd.DataFrame
    triads_gaze_counts_df: pd.DataFrame
    gaze_counts_loaded: bool = False

    def __init__(self) -> None:
        pass

    def convert_seconds_to_timestamp(self, df_with_seconds:pd.DataFrame,seconds_col_name:str='seconds_interaction') -> pd.DataFrame:
        df_with_seconds[seconds_col_name+'_ts'] = pd.to_timedelta(df_with_seconds[seconds_col_name], unit='s')

        return df_with_seconds

    # returns a df with the timeline changed to avg seconds based on a set timeframe.
    # By default the avg is calculated every 30 seconds.
    # Important that the column name has the be the one with a timestamp data type (use the convert_seconds_to_timestamp first)
    def resample_avg_seconds_using_timeframe(self, df_default_time_frequency:pd.DataFrame,seconds_col_name:str='seconds_interaction_ts',
                                        timeframe:int=30) -> pd.DataFrame:
        df_resampled = df_default_time_frequency.resample(str(timeframe)+'s', on = seconds_col_name).mean().reset_index()
        new_name = seconds_col_name[:-2]+'window'
        df_resampled[new_name] = df_resampled[seconds_col_name].dt.total_seconds()
        df_resampled.drop([seconds_col_name, seconds_col_name[:-3]], axis=1, inplace=True)
        return df_resampled

    def get_grups_interaction_times(self):
        file_path = '../data/group_durations_all_commas.csv'
        # clean path in case it expects the directory at root level    
        if not os.path.exists(file_path) and "../" in file_path:
            file_path = file_path[3:]
        names_durations_df: pd.DataFrame = pd.read_csv(file_path, index_col=0)

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

    def get_groups_names_and_formation(self)->pd.DataFrame:
        # get the duration of each group
        file_path: str = "../data/group_durations_all_commas.csv"
        # clean path in case it expects the directory at root level
        if not os.path.exists(file_path) and "../" in file_path:
            file_path = file_path[3:]
        names_durations_df: pd.DataFrame = pd.read_csv(file_path, index_col=0)
        df_names_formations = names_durations_df [['group_formation', 'type']]
        return df_names_formations

    # returns a list of the group names as they are in tha main df, such as dyad01_MGP2P1 or similar; it filters for groups size (dyads or triads) and for formation (Line or F)
    def get_formation_grups_names_from_eye_df(self, main_df:pd.DataFrame, group_formation_df:pd.DataFrame, group_size:str,
                                            group_formation:str)->list:
        name_columns = []
        for name in list(group_formation_df.loc[(group_formation_df['group_formation'] == group_formation) & (
                group_formation_df['type'] == group_size)].index):
            name_columns.extend(list(main_df.filter(regex=name).columns))
        return name_columns

    def plot_vline_num_groups(self, ax, current_duration: float, current_index: int, text_y:float, line_y:float, line_colour:str):        
            offset = (current_index % 2) / 20  # adds an offset every other time so it can be readable
            text_y_offset = text_y - offset
            line_y_offset = line_y - offset
            num_groups = current_index
            ax.text(current_duration, text_y_offset, num_groups, color=line_colour)
            ax.axvline(current_duration, ymax=line_y_offset, ymin=0, color=line_colour, linestyle='--', alpha=0.8, linewidth=0.7)


    def plot_avg_and_std_to_existing_graph(self, ax, main_df:pd.DataFrame, df_column_avg:str, df_column_std:str,
                            list_interaction_duration_descending_order:list,
                            line_colour:str, text_y:float, line_y:float,
                            fill_alpha:float, fill_color:str, figure_title:str, timewindow: int, plot_std: bool = True):

        ax.plot(main_df[df_column_avg], color=line_colour, label=df_column_avg)
        if plot_std:
            ax.fill_between(main_df.index, main_df[df_column_avg] - main_df[df_column_std],
                        main_df[df_column_avg] + main_df[df_column_std], facecolor=fill_color, alpha=fill_alpha, label = df_column_std)
        
        self.plot_vline_num_groups(ax=ax, current_duration=0, current_index=len(list_interaction_duration_descending_order), text_y=text_y, line_y=line_y, line_colour=line_colour)
        for current_duration in list_interaction_duration_descending_order:
            current_index = list_interaction_duration_descending_order.index(current_duration)
            self.plot_vline_num_groups(ax=ax, current_duration=current_duration, current_index=current_index, text_y=text_y, line_y=line_y, line_colour=line_colour)
            # offset = (current_index % 2) / 20  # adds an offset every other time so it can be readable
            # text_y_offset = text_y - offset
            # line_y_offset = line_y - offset
            # num_groups = current_index
            # ax.text(current_duration, text_y_offset, num_groups, color=line_colour)
            # ax.axvline(current_duration, ymax=line_y_offset, ymin=0, color=line_colour, linestyle='--', alpha=0.8, linewidth=0.7)

        num_groups = len(list_interaction_duration_descending_order)
        # print descriptive stats
        print(f"{figure_title} for {df_column_avg}: {main_df[df_column_avg].mean()}, std: {main_df[df_column_std].mean()}, groups: {num_groups}, timewindow: {timewindow} secs ")

        return ax

    #potentially, this can be called also with a list of triad names. but so far I think I can just creat the triads names based on the number of triads existing (14)
    def sum_triads_MG_per_group(self, main_df: pd.DataFrame, group_names:list):
        df_triads_MG_summed = pd.DataFrame(index=main_df.index)
        for triad_name in group_names:
            columns_MG = list(main_df.filter(regex=f'{triad_name}_MG_P').columns)
            df_triads_MG_summed[columns_MG[0]] = main_df[columns_MG].sum(axis=1,min_count=1) #keeping the name of the first MG as it is used further in the code and it works like that for the dyads too
        return df_triads_MG_summed


    def create_line_plot(self, fig, ax, file_path:str, timewindow:int = 30, save_plot = False,
                        dyads:bool = True, triads:bool = True, separate_by_group_formation:bool = True,
                        sum_triads_for_mutual_gaze:bool = True,
                        y_axis_text:str='', figure_title:str='', color_dyad:str='red', color_triad:str='green',
                        plt_show: bool = True, plot_std: bool = True, 
                        one_directioned_direct_gaze = True, all_features: bool = False, save_df_resampled_to_file:bool=True):

        # clean path in case it expects the directory at root level
        if not os.path.exists(file_path) and "../" in file_path:
            file_path = file_path[3:]
        
        df_data = pd.read_csv(file_path, index_col=0)
        df_data_ts = self.convert_seconds_to_timestamp(df_data)
        df_resampled = self.resample_avg_seconds_using_timeframe(df_data_ts, timeframe=timewindow)
        df_resampled.set_index('seconds_interaction_window', inplace=True)
        # not ideal, but for 1d direct gaze, I'll drop the columns of DG (not the one directioned ones);
        if one_directioned_direct_gaze:
            r = re.compile(".*1d_DG")
            one_DG_cols = list(filter(r.match,list(df_resampled.columns)))
            DG_cols = list (set(df_resampled.columns) - set(one_DG_cols))
            df_resampled.drop(DG_cols, axis=1, inplace=True)
        df_group_formations = self.get_groups_names_and_formation()
        if dyads:
            if separate_by_group_formation:
                col_dyads_l = self.get_formation_grups_names_from_eye_df(df_resampled, df_group_formations, 'dyad', 'Line')
                col_dyads_f = self.get_formation_grups_names_from_eye_df(df_resampled, df_group_formations, 'dyad', 'F')

                df_dyads_l = pd.DataFrame({'Dyads-L ' + figure_title: df_resampled[col_dyads_l].mean(axis=1),
                                        'Dyads-L ' + figure_title + ' Std': df_resampled[col_dyads_l].std(axis=1) / 2})
                df_dyads_f = pd.DataFrame({'Dyads-F ' + figure_title: df_resampled[col_dyads_f].mean(axis=1),
                                        'Dyads-F ' + figure_title + ' Std': df_resampled[col_dyads_f].std(axis=1) / 2})
                dyads_durations_f = self.get_grups_interaction_times()['dyads_f'].sort_values(ascending=False).tolist()
                dyads_durations_l = self.get_grups_interaction_times()['dyads_l'].sort_values(ascending=False).tolist()

                ax = self.plot_avg_and_std_to_existing_graph(ax=ax, main_df=df_dyads_l, df_column_avg='Dyads-L ' + figure_title,
                                                        df_column_std='Dyads-L ' + figure_title +  ' Std',
                                                        list_interaction_duration_descending_order=dyads_durations_l,
                                                        line_colour=color_dyad, text_y=0.4, line_y=0.6,
                                                        fill_alpha=0.2, fill_color=color_dyad, figure_title=figure_title, timewindow=timewindow, 
                                                        plot_std=plot_std)

                ax = self.plot_avg_and_std_to_existing_graph(ax=ax, main_df=df_dyads_f, df_column_avg='Dyads-F ' + figure_title,
                                                        df_column_std='Dyads-F ' + figure_title +  ' Std',
                                                        list_interaction_duration_descending_order=dyads_durations_f,
                                                        line_colour='orange', text_y=0.6, line_y=0.8,
                                                        fill_alpha=0.2, fill_color='orange', figure_title=figure_title, timewindow=timewindow, 
                                                        plot_std=plot_std)

            else:
                col_dyads = list(df_resampled.filter(regex='dyad').columns)
                df_dyads = pd.DataFrame({'Dyads ' + figure_title: df_resampled[col_dyads].mean(axis=1),
                                        'Dyads ' + figure_title +  ' Std': df_resampled[col_dyads].std(axis=1) / 2})
                dyads_durations = self.get_grups_interaction_times()['dyads'].sort_values(ascending=False).tolist()

                ax = self.plot_avg_and_std_to_existing_graph(ax=ax, main_df = df_dyads, df_column_avg='Dyads ' + figure_title,
                                                df_column_std='Dyads ' + figure_title +  ' Std',
                                                list_interaction_duration_descending_order = dyads_durations,
                                                line_colour = color_dyad, text_y =  0.6, line_y=0.8,
                                                fill_alpha = 0.2, fill_color=color_dyad, figure_title=figure_title, timewindow=timewindow, 
                                                plot_std=plot_std)

        if triads:
            if (sum_triads_for_mutual_gaze):
                triads_group_names = df_group_formations.loc[df_group_formations['type'] == 'triad'].index
                df_MG_P_resampled = self.sum_triads_MG_per_group(df_resampled, triads_group_names)
                df_resampled.update(df_MG_P_resampled)            

            if separate_by_group_formation:
                col_triads_l = self.get_formation_grups_names_from_eye_df(df_resampled, df_group_formations, 'triad', 'Line')
                col_triads_f = self.get_formation_grups_names_from_eye_df(df_resampled, df_group_formations, 'triad', 'F')

                df_triads_l = pd.DataFrame({'Triads-L ' + figure_title: df_resampled[col_triads_l].mean(axis=1),
                                        'Triads-L ' + figure_title +  ' Std': df_resampled[col_triads_l].std(axis=1) / 2})
                df_triads_f = pd.DataFrame({'Triads-F ' + figure_title: df_resampled[col_triads_f].mean(axis=1),
                                        'Triads-F ' + figure_title +  ' Std': df_resampled[col_triads_f].std(axis=1) / 2})
                triads_durations_f = self.get_grups_interaction_times()['triads_f'].sort_values(ascending=False).tolist()
                triads_durations_l = self.get_grups_interaction_times()['triads_l'].sort_values(ascending=False).tolist()

                ax = self.plot_avg_and_std_to_existing_graph(ax=ax, main_df=df_triads_l, df_column_avg='Triads-L ' + figure_title,
                                                        df_column_std='Triads-L ' + figure_title +  ' Std',
                                                        list_interaction_duration_descending_order=triads_durations_l,
                                                        line_colour=color_triad, text_y=0.1, line_y=0.25,
                                                        fill_alpha=0.2, fill_color=color_triad, figure_title=figure_title, timewindow=timewindow, 
                                                        plot_std=plot_std)

                ax = self.plot_avg_and_std_to_existing_graph(ax=ax, main_df=df_triads_f, df_column_avg='Triads-F ' + figure_title,
                                                        df_column_std='Triads-F ' + figure_title +  ' Std',
                                                        list_interaction_duration_descending_order=triads_durations_f,
                                                        line_colour='blue', text_y=0.3, line_y=0.5,
                                                        fill_alpha=0.2, fill_color='blue', figure_title=figure_title, timewindow=timewindow, 
                                                        plot_std=plot_std)

            else:
                col_triads = list(df_resampled.filter(regex='triad').columns)
                df_triads = pd.DataFrame({'Triads ' + figure_title: df_resampled[col_triads].mean(axis=1),
                                        'Triads ' + figure_title + ' Std': df_resampled[col_triads].std(axis=1) / 2})

                triads_durations = self.get_grups_interaction_times()["triads"].sort_values(ascending=False).tolist()

                ax = self.plot_avg_and_std_to_existing_graph(ax=ax, main_df = df_triads, df_column_avg='Triads ' + figure_title,
                                                df_column_std='Triads ' + figure_title + ' Std',
                                                list_interaction_duration_descending_order = triads_durations,
                                                line_colour = color_triad, text_y =  0.2, line_y=0.3,
                                                fill_alpha = 0.2, fill_color=color_triad, figure_title=figure_title, timewindow=timewindow, 
                                                plot_std=plot_std)

        if save_df_resampled_to_file:
            filepath_path = "../Recordings/SavedData/v2_no_low_sampled/"
            filepath_path = os.path.join(os.getcwd(), "Recordings", "SavedData", "v2_no_low_sampled")
            extra_filename = ""
            if all_features:
                extra_filename += "all_features_"
            elif one_directioned_direct_gaze:
                extra_filename += "oneD_DG_"
            else:
                extra_filename += "MG_"
            filepath_path = os.path.join(filepath_path, extra_filename)
            filepath_path = filepath_path + str(timewindow) + "s_" + "resampled.csv"
            df_resampled.to_csv(filepath_path)

        if plt_show:
            plt.xlabel('Interaction Time in Seconds ('+ 'windows of ' +str(timewindow)+ 's)' )
            plt.ylabel(y_axis_text)
            plt.title(figure_title)
            plt.tight_layout()
            ax.legend(loc='best')
            plt.show()

    def load_gaze_counts(self, folder_path: str, floor_level: bool):
        if os.path.exists(folder_path):
            floor_level_suffix: str = "_floorlevel" if floor_level else ""
            self.all_groups_gaze_counts_df = pd.read_csv(os.path.join(folder_path, f"all_groups_counts_gaze{floor_level_suffix}.csv"), index_col=0)
            self.dyads_gaze_counts_df = pd.read_csv(os.path.join(folder_path, f"dyads_counts_gaze{floor_level_suffix}.csv"), index_col=0)
            self.triads_gaze_counts_df = pd.read_csv(os.path.join(folder_path, f"triads_counts_gaze{floor_level_suffix}.csv"), index_col=0)
            self.gaze_counts_loaded = True
    
    def calculate_gaze_count_stats(self, df: pd.DataFrame, configuration: str) -> pd.DataFrame:
        config_df = df.loc[:, 'Percentage':][df.index.str.contains(configuration)].fillna(0)
        result_df = config_df.describe()
        result_df.rename(columns={"Percentage":f"Percentage_{configuration}"}, inplace=True)
        return result_df
    
    def calculate_dyads_gaze_count_stats(self, dyads_df: pd.DataFrame, path_save: str, floor_level: bool, save: bool) -> pd.DataFrame:
        result_df_list: list[pd.DataFrame] = []
        # MG 
        result_df_list.append(self.calculate_gaze_count_stats(dyads_df, "MG_P"))
        # 0 D1
        result_df_list.append(self.calculate_gaze_count_stats(dyads_df, "0_D1"))
        # 1 D1 (1d_DG)
        result_df_list.append(self.calculate_gaze_count_stats(dyads_df, "1d_DG"))
        counts_df = pd.concat(result_df_list, axis=1)
        if path_save:
            floor_level_suffix: str = "_floorlevel" if floor_level else ""
            counts_df.to_csv(os.path.join(path_save, f"dyads_gaze_counts_stats{floor_level_suffix}.csv"))
        return counts_df
        pass

    def calculate_triads_gaze_count_stats(self, triads_df: pd.DataFrame, path_save:str, floor_level: bool, save: bool) -> pd.DataFrame:
        result_df_list: list[pd.DataFrame] = []
        # 0 D1 (Nobody looks at the other participants)
        result_df_list.append(self.calculate_gaze_count_stats(triads_df, "0_D1"))
        # 1 D1 (One looks at the other. two, don’t)
        result_df_list.append(self.calculate_gaze_count_stats(triads_df, "1_D1"))
        # 2 D1 same (Two to the same person, which to nothing)
        result_df_list.append(self.calculate_gaze_count_stats(triads_df, "2_D1_same"))
        # 2 D1 different (One to nothing and their two each to a different person)
        result_df_list.append(self.calculate_gaze_count_stats(triads_df, "2_D1_different"))
        # 3 D1 (Each to the different one - circle of gaze)
        result_df_list.append(self.calculate_gaze_count_stats(triads_df, "3_D1"))
        # MG D1 (The third left out is looking at one of the two engaged in MG)
        result_df_list.append(self.calculate_gaze_count_stats(triads_df, "MG_D1"))
        # MG D0  (The third does not look at any of the other 2)
        result_df_list.append(self.calculate_gaze_count_stats(triads_df, "MG_D0"))
        # MG (it includes both MG_D1 and MG_D0)
        result_df_list.append(self.calculate_gaze_count_stats(triads_df, "MG_P"))
        # 1d_DG (it includes all D1 dynamics except 0_D1)
        result_df_list.append(self.calculate_gaze_count_stats(triads_df, "1d_DG"))
        counts_df = pd.concat(result_df_list, axis=1)
        if path_save and save:
            floor_level_suffix: str = "_floorlevel" if floor_level else ""
            counts_df.to_csv(os.path.join(path_save, f"triads_gaze_counts_stats{floor_level_suffix}.csv"))
        return counts_df
        


if __name__ == "__main__":
    save_plot = False 
    gaze_configs_stats = False
    floorlevel = False
    save_gaze_counts = False
    save_resampled_file = True
    # which features to plot
    all_features_plotting = True
    mutual_gaze_plotting = False
    direct_gate_plotting = False
    root_path= "../Recordings/SavedData/v2_no_low_sampled/"
    # root_path_1d_DG= "../Recordings/SavedData/1d_DG/"
    all_groups_MG_df_path = root_path+"all_groups_mutual_gaze_interaction_time.csv"
    all_groups_DG_df_path = root_path+"all_groups_direct_gaze_interaction_time.csv"
    all_groups_all_features_df_path = root_path+"all_groups_all_features_interaction_time.csv"

    fig, ax = plt.subplots(figsize=(12,5))
    eye_plotter: PlottingEyeData = PlottingEyeData()

    # Gaze conditions stats
    if gaze_configs_stats:
        data_folder: str =  os.path.join(os.getcwd(), "Recordings", "SavedData", "gaze_counts")
        eye_plotter.load_gaze_counts(data_folder, floor_level=floorlevel)
        eye_plotter.calculate_dyads_gaze_count_stats(eye_plotter.dyads_gaze_counts_df, data_folder, floor_level=floorlevel, save=save_gaze_counts)
        eye_plotter.calculate_triads_gaze_count_stats(eye_plotter.triads_gaze_counts_df, data_folder, floor_level=floorlevel, save=save_gaze_counts)

    # Mutual Gaze Plotting
    # eye_plotter.create_line_plot(fig=fig, ax=ax, file_path=all_groups_MG_df_path, timewindow=5, separate_by_group_formation=False,
    #                 sum_triads_for_mutual_gaze=False, dyads=True, triads=True, one_directioned_direct_gaze=False,
    #                 y_axis_text="Mutual Gaze %",  figure_title="Mutual Gaze")

    # Direct Gaze plotting
    # eye_plotter.create_line_plot(fig=fig, ax=ax, file_path=all_groups_DG_df_path, timewindow=10, separate_by_group_formation=False,
    #                  sum_triads_for_mutual_gaze=True, dyads=True, triads=True, one_directioned_direct_gaze=False,
    #                  y_axis_text="One Direction Direct Gaze %",  figure_title="One Direction Direct Gaze")

    # All Features plotting
    if all_features_plotting:
        eye_plotter.create_line_plot(fig=fig, ax=ax, file_path=all_groups_all_features_df_path, timewindow=60, separate_by_group_formation=False,
                     sum_triads_for_mutual_gaze=True, dyads=True, triads=True, one_directioned_direct_gaze=False, all_features=True,
                     y_axis_text="All Eye Gaze Features %",  figure_title="All Eye Gaze Features", save_df_resampled_to_file=save_resampled_file)
