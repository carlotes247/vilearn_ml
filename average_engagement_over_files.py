from datetime import datetime, time, timedelta
import os
import time
import pandas as pd
import numpy as np

datetimes : list[str] = []
col_names = ["task_eng", "conf"]
path_file_1: str = "data/annotations/dyad_01/group.task engagement.helenrisack.annotation~"
path_file_2: str = "data/annotations/dyad_01/task engagement.group.carlosgonzalez.annotation~"
path_groups_info: str = "data/group_durations_all_commas.csv"
offset_interaction_start: float = 0
interaction_length: float = 0
group_name: str = "dyad_01"


def avg_eng_files(df_1: pd.DataFrame, df_2: pd.DataFrame) -> pd.DataFrame:
    print(f"File 1 has {len(df_1)} lines and File 2 has {len(df_2)} lines")
    if len(df_1) != len(df_2):
        print("Engagement files are not of equal size!")
        return pd.DataFrame()
    # drop conf column and make sure both dataframes are numeric
    df_1.drop("conf", axis=1, inplace=True)
    df_1['task_eng'] = pd.to_numeric(df_1['task_eng'], errors='coerce')
    df_2.drop("conf", axis=1, inplace=True)
    df_2['task_eng'] = pd.to_numeric(df_2['task_eng'], errors='coerce')
    # clean nans
    df_merged: pd.DataFrame = pd.concat([df_1, df_2], axis=1)
    df_merged.replace('-nan(ind)', np.nan, inplace=True)
    df_merged = df_merged.fillna(0)
    return pd.DataFrame(df_merged.mean(axis=1), columns=['task_eng'])


if __name__ == '__main__':
    df_eng_1: pd.DataFrame = pd.read_csv(path_file_1, sep=";", names=col_names)
    df_eng_2 : pd.DataFrame = pd.read_csv(path_file_2, sep=";", names=col_names)

    df_avg: pd.DataFrame = avg_eng_files(df_eng_1, df_eng_2)
    df_groups_info = pd.read_csv(path_groups_info)
    col_mask = df_groups_info['name'] == group_name
    offset_interaction_start = df_groups_info[col_mask]['offset_recording_interaction_start'].values[0]
    interaction_length = df_groups_info[col_mask]['duration_interaction'].values[0]
    print("done")