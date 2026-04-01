import pandas as pd
import numpy as np

if __name__ == "__main__": 
    # vars
    upsample: bool = True
    resample: bool = True
    resample_window_size: int = 30
    save_resampled:bool = True
    save_upsampled:bool = True
    save_all_groups_df:bool = True
    floorlevel_only:bool = False
    floorlevel_info_df = pd.read_csv("data/group_names_with_time_floorlevel.csv", sep=";")
    times_info_df = pd.read_csv("data/group_durations_all_commas.csv", sep=",")
    if floorlevel_only:
        group_names = floorlevel_info_df["Group_Name"].to_list()
    else:
        group_names = times_info_df["name"].to_list()
    speaking_configs_list = []
    speaking_all_groups_df_list = []
    speaking_all_groups_df_resampled_list = []
    for group_name in group_names:
        group_dis_df = pd.read_csv(f"data/discover/merged/recording_{group_name}.csv")
        print(group_name)
        # set dyad or triad participants
        participants = ["speaking_p_blue", "speaking_p_green"] if "dyad" in group_name else ["speaking_p_blue", "speaking_p_green", "speaking_p_red"]
        triad = True if "triad" in group_name else False  

        if upsample:
            # resample dataframe to 90 Hz
            group_dis_df['timestamp'] = pd.to_datetime(group_dis_df['time_ms'], unit='ms')
            group_dis_df = group_dis_df.set_index('timestamp')
            # 11 ms expressed in pandas offset alias
            freq = '11ms'          # "L" = millisecond, can also use "11ms"

            # Create a new index that spans the whole range at 11 ms steps
            new_index = pd.date_range(start=group_dis_df.index.min(),
                                    end=group_dis_df.index.max(),
                                    freq=freq)

            # Reindex onto the new grid (introduces NaNs)
            df_resampled = group_dis_df.reindex(new_index)
            df_resampled.index.name = 'timestamp'
            df_merged = pd.merge_asof(df_resampled.sort_values('timestamp'), group_dis_df.sort_values('timestamp'), on='timestamp', direction='nearest')
            df_merged.drop(columns=[c for c in df_merged.columns
                       if c.endswith('_x') and c != 'time_ms_x'], inplace=True)
            # Linear interpolation (default for numeric columns)
            df_interpolated = df_merged.interpolate(method='linear', limit_direction='forward', limit=None).ffill()
            df_interpolated.drop(columns=['time_ms_y'], inplace=True)
            df_interpolated.columns = df_interpolated.columns.str.replace(r'_[xy]$', '', regex=True)
            # Reassign df for later processing
            group_dis_df = df_interpolated
            if save_upsampled:
                group_dis_df.to_csv(f"data/discover/merged/recording_{group_name}_90Hz.csv")
            

        # slice interaction time
        inter_start_ms = times_info_df[times_info_df["name"]==group_name]["offset_recording_interaction_start"].iloc[0]*1000
        inter_length_ms = times_info_df[times_info_df["name"]==group_name]["duration_interaction"].iloc[0]*1000
        inter_end_ms = inter_start_ms+inter_length_ms
        first_inter_frame = (group_dis_df['time_ms']-inter_start_ms).abs().argsort()[:1].iloc[0]
        last_inter_frame = (group_dis_df['time_ms']-inter_end_ms).abs().argsort()[:1].iloc[0]
        group_dis_df = pd.DataFrame(group_dis_df[first_inter_frame:last_inter_frame])

        # create seconds column to match other modalities dfs
        group_dis_df["time_ms"] = group_dis_df["time_ms"] - group_dis_df["time_ms"].iloc[0]                  
        group_dis_df.set_index("timestamp", inplace=True)

        # Speaking configs
        length_df = len(group_dis_df.index)
        # Dyads
        # SP (One speaks)
        SP_mask = group_dis_df[participants].sum(axis=1) == 1
        SP_count = 0
        if True in SP_mask.value_counts():
            SP_count = SP_mask.value_counts()[True]         
        SP_amount = SP_count/length_df
        group_dis_df = group_dis_df.assign(SP=SP_mask.values) # assign to group df
        # SI (All silent (~ symbol negates the dataframe))
        SI_mask = ~group_dis_df[participants].any(axis=1)
        SI_count = 0
        if True in SI_mask.value_counts():
            SI_count = SI_mask.value_counts()[True]
        SI_amount = SI_count/length_df
        group_dis_df = group_dis_df.assign(SI=SI_mask.values)
        # OV (Overlap)
        # All speak
        OV_all_speak_mask = group_dis_df[participants].all(axis=1)
        OV_all_speak_count = 0
        if True in OV_all_speak_mask.value_counts():
            OV_all_speak_count = OV_all_speak_mask.value_counts()[True]  
        OV_all_speak_amount = OV_all_speak_count /length_df
        group_dis_df = group_dis_df.assign(OV_all_SP=OV_all_speak_mask.values)
        # Two speak (only triads)
        OV_two_speak_count = 0
        OV_two_speak_amount = 0
        OV_two_speak_mask = group_dis_df[participants].sum(axis=1) == 2
        group_dis_df = group_dis_df.assign(OV_two_SP=OV_two_speak_mask.values)
        if triad:
            if True in OV_two_speak_mask.value_counts():
                OV_two_speak_count = OV_two_speak_mask.value_counts()[True]
                OV_two_speak_amount = OV_two_speak_count /length_df

        # TODO: calculate features that cross with gaze

        # resample down in windows (specified by window_size)
        if resample:
            # Find all columns that match the pattern or are the name 'session'
            cols_to_drop = group_dis_df.columns[group_dis_df.columns.str.contains('text_p', case=False)]  # all text_p
            cols_to_drop = cols_to_drop.tolist() + ['session']                      # add session

            resampled_df = group_dis_df.drop(columns=cols_to_drop).resample(f'{resample_window_size}s').mean()
            resampled_df['time_ms'] = 60*np.arange(len(resampled_df))
            resampled_df.rename(columns={"time_ms": "seconds"}, inplace=True)
            resampled_df['session'] = group_name
            speaking_all_groups_df_resampled_list.append(resampled_df)            

        group_speaking_config_dict = {
            "name":group_name,
            "type": "triad" if triad else "dyad",
            "SP": SP_amount*100,
            "SI": SI_amount*100,
            "OV_all_sp": OV_all_speak_amount*100,
            "OV_two_sp": OV_two_speak_amount*100
        }
        speaking_configs_list.append(group_speaking_config_dict)
        speaking_all_groups_df_list.append(group_dis_df)

    speaking_confis_pd = pd.DataFrame(speaking_configs_list)
    speaking_all_groups_df = pd.concat(speaking_all_groups_df_list, ignore_index=True)
    speaking_all_groups_df['session'] = speaking_all_groups_df['session'].str.replace(r'^recording_', '', regex=True)
    # save file with all frames
    if save_all_groups_df:
        speaking_all_groups_df.to_csv("data/discover/merged/all_groups_interaction_speaking_90Hz.csv")
    if save_resampled:
        speaking_all_groups_df_resampled = pd.concat(speaking_all_groups_df_resampled_list, ignore_index=True)
        speaking_all_groups_df_resampled.to_csv(f'data/discover/merged/all_groups_interaction_speaking_per_{resample_window_size}s.csv')

    # resample file to 60s

    # all_groups_interaction_task_eng90Hz

    print("hola")