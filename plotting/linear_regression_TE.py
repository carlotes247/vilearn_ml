import statistics
from pathlib import Path

import pandas as pd
from scipy import stats

from data_reading.groups_manager import GroupsManager
import matplotlib.pyplot as plt
import json
import re

# class that would create files for the df to load easier; then would calculate statistics (linear regression) and plot them with TE as dependent variable
class LinearRegressionTE(object):
    # df that contains the MG, 1d-DG, blink rate, bring duration, blink sync, and TE as columns and each group as row.
    df_all_measures_on_groups_level = pd.DataFrame()

    path_going_up_one_folder = "../"
    path_prefix_gaze_data = "Recordings/SavedData/"
    data_folder_path = path_going_up_one_folder + "data/"

    def __init__(self, filepath_all_measures_on_groups_level: str):

        if Path(filepath_all_measures_on_groups_level).is_file():
            # load file
            self.df_all_measures_on_groups_level = pd.read_csv(filepath_all_measures_on_groups_level)
        else:
          # This means that the file is not created yet so get all the data needed and add it to a df and then print it to file.
            # get MG
            self.df_all_measures_on_groups_level = self.return_MG_avg_per_group()
            # get 1d-DG
            df_1d_DG = self.return_1d_DG_avg_per_group()
            self.df_all_measures_on_groups_level = pd.concat([self.df_all_measures_on_groups_level, df_1d_DG], axis=1)
            # get TE
            df_TE = self.return_TE_avg_per_group()
            self.df_all_measures_on_groups_level = pd.concat([self.df_all_measures_on_groups_level, df_TE], axis=1)
            # get BRM
            df_BRM = self.return_BRM_avg_per_group()
            self.df_all_measures_on_groups_level = pd.concat([self.df_all_measures_on_groups_level, df_BRM], axis=1)
            # get blink_duration
            df_blink_duration = self.return_blink_duration_avg_per_group()
            self.df_all_measures_on_groups_level = pd.concat([self.df_all_measures_on_groups_level, df_blink_duration],
                                                             axis=1)
            # get blink_sync
            df_blink_sync = self.return_blink_sync_avg_per_group()
            self.df_all_measures_on_groups_level = pd.concat([self.df_all_measures_on_groups_level, df_blink_sync],
                                                             axis=1)
            # get group formation
            df_formation = self.get_group_formations()
            self.df_all_measures_on_groups_level = pd.concat([self.df_all_measures_on_groups_level, df_formation],
                                                             axis=1)
            # save to file
            self.df_all_measures_on_groups_level.to_csv(self.path_going_up_one_folder + self.path_prefix_gaze_data +
                                                        "all_groups_features_for_regression.csv")

    def return_MG_avg_per_group(self)-> pd.DataFrame:
        MG_filepath = self.path_going_up_one_folder + self.path_prefix_gaze_data + 'all_groups_mutual_gaze_interaction_time.csv'
        df_MG = pd.read_csv(MG_filepath)

        df_means = pd.DataFrame({'MG': df_MG.mean(axis=0)})
        # df_MG_means.rename(columns={df_MG_means.column[0]: 'MG'}, inplace=True)

        r_dyads = re.compile(".*dyad_")
        df_MG_means = df_means.loc[list(filter(r_dyads.match, list(df_means.index)))]
        df_MG_means.rename(index=lambda s: s[:-8], inplace=True)

        #for triads, there are three values for each group (MGP1P2, P1P3 and P2P3), We are going to add these and create another col : triad_XY_MG
        r_triads = re.compile(".*triad_")
        triads_cols = list(filter(r_triads.match, list(df_MG.columns)))

        for i in range (len(triads_cols)//3):
            if i<9:
                r_num = re.compile(".*triad_0"+str(i+1))
                current_cols = list(filter(r_num.match, triads_cols))
                df_MG_means.loc[current_cols[0][:8]] = df_means.loc[current_cols].sum()
            else:
                r_num = re.compile(".*triad_" + str(i + 1))
                current_cols = list(filter(r_num.match, triads_cols))
                df_MG_means.loc[current_cols[0][:8]] = df_means.loc[current_cols].sum()
        return df_MG_means

    def return_1d_DG_avg_per_group(self)->pd.DataFrame:
        one_DG_filepath = self.path_going_up_one_folder + self.path_prefix_gaze_data + '1d_DG/all_groups_direct_gaze_interaction_time.csv'
        df_1DG = pd.read_csv(one_DG_filepath)

        r_1d = re.compile(".*1d_")
        df_1DG = df_1DG[list(filter(r_1d.match, list(df_1DG.columns)))]
        df_means = pd.DataFrame({'1d_DG': df_1DG.mean(axis=0)})
        df_1DG_mean = pd.DataFrame({'1d_DG':[]})

        dyads_count = 11
        triads_count = 14

        for i in range(dyads_count):
            if i<9:
                r_num = re.compile(".*dyad_0"+str(i+1))
                current_cols = list(filter(r_num.match, list(df_1DG.columns)))
                df_1DG_mean.loc[current_cols[0][:7]] = df_means.loc[current_cols].mean()
            else:
                r_num = re.compile(".*dyad_" + str(i + 1))
                current_cols = list(filter(r_num.match, list(df_1DG.columns)))
                df_1DG_mean.loc[current_cols[0][:7]] = df_means.loc[current_cols].mean()

        for i in range(triads_count):
            if i<9:
                r_num = re.compile(".*triad_0"+str(i+1))
                current_cols = list(filter(r_num.match, list(df_1DG.columns)))
                df_1DG_mean.loc[current_cols[0][:8]] = df_means.loc[current_cols].mean()
            else:
                r_num = re.compile(".*triad_" + str(i + 1))
                current_cols = list(filter(r_num.match, list(df_1DG.columns)))
                df_1DG_mean.loc[current_cols[0][:8]] = df_means.loc[current_cols].mean()

        return df_1DG_mean

    def return_TE_avg_per_group(self)->pd.DataFrame:
        TE_filepath = self.path_going_up_one_folder + 'data/annotations/all_groups_interaction_task_eng90Hz.csv'
        df_TE = pd.read_csv(TE_filepath)
        df_means = pd.DataFrame({'TE': df_TE.mean(axis=0)})
        df_TE_mean = pd.DataFrame({'TE': []})
        # keywords = ['dyad', 'triad']
        # df[df["Title"].apply(lambda x: any(k in x for k in keywords))]
        #
        for indx in df_means.index.values:
            if 'dyad' in indx:
                df_TE_mean.loc[indx[9:]] = df_means.loc[indx]
            elif 'triad' in indx:
                df_TE_mean.loc[indx[9:]] = df_means.loc[indx]
        return df_TE_mean

    def return_BRM_avg_per_group(self)->pd.DataFrame:
        BRM_filepath = self.path_going_up_one_folder + 'data/blink_rates_all_groups.csv'
        df_BRM = pd.read_csv(BRM_filepath ,sep='\s+')
        df_mean_BRM = df_BRM[["group_short_name", "group_mean_blink_rate_ms"]]
        df_mean_BRM.rename(columns={'group_mean_blink_rate_ms':'BRM'}, inplace=True)
        df_mean_BRM.set_index("group_short_name", inplace=True)
        return df_mean_BRM

    def return_blink_duration_avg_per_group(self)->pd.DataFrame:
        duration_filepath = self.path_going_up_one_folder + 'data/blink_durations_all_groups.csv'
        df_duration = pd.read_csv(duration_filepath, sep='\s+')
        df_mean_duration = df_duration[["group_short_name", "group_mean_blink_duration_ms"]]
        df_mean_duration.rename(columns={'group_mean_blink_duration_ms': 'blink_duration'}, inplace=True)
        df_mean_duration.set_index("group_short_name", inplace=True)
        return df_mean_duration

    def return_blink_sync_avg_per_group(self)->pd.DataFrame:
        sync_filepath = self.path_going_up_one_folder + 'data/avg_blinks_async_ms_all_groups.csv'
        df_sync = pd.read_csv(sync_filepath, sep='\s+')
        df_mean_sync = df_sync[["group_short_name", "avg_blinks_sync_percent"]]
        df_mean_sync.rename(columns={'avg_blinks_sync_percent': 'blink_sync'}, inplace=True)
        df_mean_sync.set_index("group_short_name", inplace=True)
        return df_mean_sync

    def get_group_formations(self)->pd.DataFrame:
        info_filepath = self.path_going_up_one_folder + 'data/group_durations_all_commas.csv'
        df_info = pd.read_csv(info_filepath)
        df_info = df_info[["name", "group_formation"]]
        df_info.set_index("name", inplace=True)
        return df_info



if __name__ == "__main__":

    path_going_up_one_folder = "../"
    path_prefix_gaze_data = "Recordings/SavedData/"
    data_filepath = path_going_up_one_folder + path_prefix_gaze_data + "all_groups_features_for_regression.csv"
    l_regression = LinearRegressionTE(data_filepath)