# load the data into dfs
# transform the seconds column into the correct data type
# create the plot
# show and save the plot
import re
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import os
# Added this try catch because on some machines it cannot find folders from working directory 
try:
    from preprocessing.engagement.engagements_manager import EngagementsManager
except ImportError:
    import sys
    sys.path.append(os.getcwd())
    from preprocessing.engagement.engagements_manager import EngagementsManager

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
                        one_directioned_direct_gaze = True, all_features: bool = False, load_task_engagement: bool = False, 
                        save_df_resampled_to_file:bool=True):

        # clean path in case it expects the directory at root level
        if not os.path.exists(file_path) and "../" in file_path:
            file_path = file_path[3:]
        
        df_data = pd.read_csv(file_path, index_col=0)
        df_data_ts = self.convert_seconds_to_timestamp(df_data)

        if load_task_engagement:
            load_all_processed = True # <-- Loads the big merged file from disk
            df1_aligned: pd.DataFrame = pd.DataFrame()
            # loading or processing common dataframe
            if load_all_processed:
                df1_aligned = pd.read_csv(os.path.join(os.getcwd(), "data", "all_features_gaze_eng_all_groups_90Hz.csv"))
            else:
                eng_mngr: EngagementsManager = EngagementsManager(save_to_disk=False, load_from_disk=True, floor_level=True)
                # Merge both df_data together with engagements
                # Start with only the 'second' column
                df1_aligned = eng_mngr.df_avg_eng_all_interaction[["seconds"]].copy()
                df1_aligned.rename(columns={'seconds':'seconds_interaction'}, inplace=True)
                df_aux = df_data_ts
                df_aux['seconds_interaction'] *= 1000 # <-- increasing the dimension of time so we adjust the tolerance of the merge down to the ms
                df1_aligned['seconds_interaction'] *= 1000
                # Merge features with tolerance
                max_tolerance: int = 15  # <-- maximum allowed difference in milliseconds (because the time has increased in magnitude and we need an int here)
                # Loop over features and merge one by one
                diffs = []
                for col in df_aux.columns:
                    if not 'seconds' in col:
                        temp = df_aux.dropna(subset=[col])[["seconds_interaction", col]]                    
                        df1_aligned = pd.merge_asof(
                            df1_aligned.sort_values("seconds_interaction"),
                            temp.sort_values("seconds_interaction"),
                            on="seconds_interaction",
                            direction="nearest",
                            tolerance=max_tolerance
                        )
                        og_last_sec = df_aux['seconds_interaction'][df_aux[col].last_valid_index()]/1000
                        new_last_sec = df1_aligned['seconds_interaction'][df1_aligned[col].last_valid_index()]/1000
                        diff = og_last_sec - new_last_sec
                        #print(f"{col}: og last valid time: {og_last_sec} VS new last sec: {new_last_sec}. Diff = {diff}")
                        diffs.append(diff)
                avg_diff = sum(diffs) / len(diffs)
                # Merge now with actual task engagement
                df1_aligned.rename(columns={'seconds_interaction':'seconds'}, inplace=True)
                df1_aligned['seconds'] /= 1000 # <-- reducing the dimension of time so we adjust the tolerance of the merge down to the ms
                df1_aligned = pd.merge_asof(
                            df1_aligned.sort_values("seconds"),
                            eng_mngr.df_avg_eng_all_interaction.sort_values("seconds"),
                            on="seconds",
                            direction="nearest",
                            tolerance=max_tolerance
                        )            
                print(f"Dfs merged! Avg diff is {avg_diff}")
                df1_aligned.to_csv(os.path.join(os.getcwd(), "data", "all_features_gaze_eng_all_groups_90Hz.csv"))

            # do stuff with it
            seconds_col = df1_aligned.columns[df1_aligned.columns.str.contains('second')]
            dyads_cols = df1_aligned.columns[df1_aligned.columns.str.contains('dyad')]
            triads_cols = df1_aligned.columns[df1_aligned.columns.str.contains('triad')]
            df_data_dyads = df1_aligned[seconds_col.append(dyads_cols)]
            df_data_triads = df1_aligned[seconds_col.append(triads_cols)]
            print("calculating stuff")
            # Features dyads            
            df_details_floorlevel = pd.read_csv(os.path.join(os.getcwd(), 'data', 'group_names_with_time_floorlevel.csv'), sep=';')
            groups_floorlevel = df_details_floorlevel['Group_Name'].to_list()
            dyad_names = [group for group in groups_floorlevel if 'dyad' in group]
            # MG
            self.feature_engagement_avg(dyad_names, group_label='dyad', df=df_data_dyads, feature='MG_P1P2', seconds_col=seconds_col)            
            # 0 D1
            self.feature_engagement_avg(dyad_names, group_label='dyad', df=df_data_dyads, feature='0_D1', seconds_col=seconds_col)
            # 1 D1 (1d_DG)
            #result_df_list.append(self.calculate_gaze_count_stats(dyads_df, "1d_DG"))
            # This needs custom logic because we need to combine the percentage of 1_DG of P1 and P2
            r1 = self.feature_engagement_avg(dyad_names, group_label='dyad', df=df_data_dyads, feature='1d_DG_P1', seconds_col=seconds_col)
            r2 = self.feature_engagement_avg(dyad_names, group_label='dyad', df=df_data_dyads, feature='1d_DG_P2', seconds_col=seconds_col)
            r_both = pd.concat([r1, r2], axis=1)
            df_describe = r_both.loc['mean'].describe()
            print(f'Feature dyad_1d_DG task engagement mean is: {df_describe}')
        
            # Features triads
            print(f'triads')
            triad_names = [group for group in groups_floorlevel if 'triad' in group]
            # 0 D1 (Nobody looks at the other participants)
            self.feature_engagement_avg(triad_names, group_label='triad', df=df_data_triads, feature='0_D1', seconds_col=seconds_col)
            # 1 D1 (One looks at the other. two, don’t)
            self.feature_engagement_avg(triad_names, group_label='triad', df=df_data_triads, feature='1_D1', seconds_col=seconds_col)
            # 2 D1 same (Two to the same person, which to nothing)
            self.feature_engagement_avg(triad_names, group_label='triad', df=df_data_triads, feature='2_D1_same', seconds_col=seconds_col)
            # 2 D1 different (One to nothing and their two each to a different person)
            self.feature_engagement_avg(triad_names, group_label='triad', df=df_data_triads, feature='2_D1_different', seconds_col=seconds_col)
            # 3 D1 (Each to the different one - circle of gaze)
            self.feature_engagement_avg(triad_names, group_label='triad', df=df_data_triads, feature='3_D1', seconds_col=seconds_col)
            # MG D1 (The third left out is looking at one of the two engaged in MG)
            self.feature_engagement_avg(triad_names, group_label='triad', df=df_data_triads, feature='MG_D1', seconds_col=seconds_col)
            # MG D0  (The third does not look at any of the other 2)
            self.feature_engagement_avg(triad_names, group_label='triad', df=df_data_triads, feature='MG_D0', seconds_col=seconds_col)
            # MG (it includes both MG_D1 and MG_D0)
            r1 = self.feature_engagement_avg(triad_names, group_label='triad', df=df_data_triads, feature='MG_P1P2', seconds_col=seconds_col)
            r2 = self.feature_engagement_avg(triad_names, group_label='triad', df=df_data_triads, feature='MG_P1P3', seconds_col=seconds_col)
            r3 = self.feature_engagement_avg(triad_names, group_label='triad', df=df_data_triads, feature='MG_P2P3', seconds_col=seconds_col)
            r_all = pd.concat([r1, r2, r3], axis=1)
            df_describe = r_all.loc['mean'].describe()
            print(f'Feature triad_MG task engagement mean is: {df_describe}')
            # 1d_DG (it includes all D1 dynamics except 0_D1)
            self.feature_engagement_avg(triad_names, group_label='triad', df=df_data_triads, feature='1d_DG_P1', seconds_col=seconds_col)
            self.feature_engagement_avg(triad_names, group_label='triad', df=df_data_triads, feature='1d_DG_P2', seconds_col=seconds_col)
            self.feature_engagement_avg(triad_names, group_label='triad', df=df_data_triads, feature='1d_DG_P3', seconds_col=seconds_col)
            
            print("ajajaj")


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
            filepath_path = filepath_path + str(timewindow) + "s_" + "resampled" + f"{datetime.datetime.today().date()}" + ".csv"
            df_resampled.to_csv(filepath_path)

        if plt_show:
            plt.xlabel('Interaction Time in Seconds ('+ 'windows of ' +str(timewindow)+ 's)' )
            plt.ylabel(y_axis_text)
            plt.title(figure_title)
            plt.tight_layout()
            ax.legend(loc='best')
            plt.show()

    def feature_engagement_avg(self, group_names: list[str], df: pd.DataFrame, group_label: str, feature: str, seconds_col, print_diff_rows: bool = False) -> pd.DataFrame:    
        eng_group_cols = df.columns[df.columns.str.contains(f'task_eng_{group_label}')]
        feature_cols = df.columns[df.columns.str.contains(feature)]
        df_feature = df[seconds_col.append(feature_cols.append(eng_group_cols))]
        diff_rows_te = []
        dfs_mean: list[pd.DataFrame] = []
        for group in group_names:
            group_cols = df_feature.columns.str.contains(group)
            df_group = df_feature[df_feature.columns[group_cols]]
            mask_feature_true = df_group[f'{group}_{feature}'] > 0
            df_mean = df_group[mask_feature_true][[f'task_eng_{group}']].describe()
            dfs_mean.append(df_mean)
            #print(dyad)
            if print_diff_rows:
                diff: int = df_group[f'{group}_{feature}'].last_valid_index() - df_group[f'task_eng_{group}'].last_valid_index()
                diff_rows_te.append(diff)
                print(f'Diff for group {group} in between TE and feature row index is: {diff} rows, or {0.011*diff} secs approx.')      
        df_all_means = pd.concat(dfs_mean, axis=1)
        df_result = pd.DataFrame(df_all_means.loc['mean'].describe())
        print(f'Feature {group_label}_{feature} task engagement mean is: {df_result}')
        if print_diff_rows:
            avg_diff = sum(diff_rows_te)/len(diff_rows_te)  
            print(f'AVG diff in between TE and feature row index is: {avg_diff}, or {0.011*avg_diff} secs approx.')
        return df_result

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
        #result_df_list.append(self.calculate_gaze_count_stats(dyads_df, "1d_DG"))
        # This needs custom logic because we need to combine the percentage of 1_DG of P1 and P2
        d1_1_P1_df = dyads_df.loc[:, 'Percentage':][dyads_df.index.str.contains("1d_DG_P1")].fillna(0) 
        d1_1_P2_df = dyads_df.loc[:, 'Percentage':][dyads_df.index.str.contains("1d_DG_P2")].fillna(0)
        sum_d1 = d1_1_P1_df['Percentage'].values + d1_1_P2_df['Percentage'].values
        d1_1_sum = d1_1_P1_df
        d1_1_sum['Percentage'] = sum_d1
        d1_1_sum = d1_1_sum.describe()
        d1_1_sum.rename(columns={"Percentage":f"Percentage_{'1d_DG'}"}, inplace=True)
        result_df_list.append(d1_1_sum)
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
    gaze_configs_counts_stats = False
    floorlevel = False
    save_gaze_counts = False
    save_resampled_file = True
    resampling_time_window: int = 30
    # Task Engagement
    load_TE: bool = True
    # which features to plot (nothing to do with gaze counts)
    all_features_plotting = True
    mutual_gaze_plotting = False
    direct_gate_plotting = False
    root_path= "../Recordings/SavedData/v2_no_low_sampled/"
    # root_path_1d_DG= "../Recordings/SavedData/1d_DG/"
    path_suffix = "_floorlevel" if floorlevel else ""
    all_groups_MG_df_path = root_path+f"all_groups_mutual_gaze_interaction_time.csv"
    all_groups_DG_df_path = root_path+f"all_groups_direct_gaze_interaction_time.csv"
    all_groups_all_features_df_path = root_path+f"all_groups_all_features_interaction_time{path_suffix}.csv"

    fig, ax = plt.subplots(figsize=(12,5))
    eye_plotter: PlottingEyeData = PlottingEyeData()

    # Gaze conditions stats
    if gaze_configs_counts_stats:
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
        eye_plotter.create_line_plot(fig=fig, ax=ax, file_path=all_groups_all_features_df_path, timewindow=resampling_time_window, separate_by_group_formation=False,
                     sum_triads_for_mutual_gaze=True, dyads=True, triads=True, one_directioned_direct_gaze=False, all_features=True,
                     y_axis_text="All Eye Gaze Features %",  figure_title="All Eye Gaze Features", 
                     load_task_engagement=load_TE,
                     save_df_resampled_to_file=save_resampled_file)
