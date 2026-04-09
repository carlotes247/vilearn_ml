import pandas as pd
import numpy as np

if __name__ == "__main__": 
    # vars
    upsample: bool = True
    resample: bool = True
    resample_window_size: int = 30
    save_resampled:bool = True
    save_upsampled:bool = False
    save_all_groups_df:bool = True
    save_group_configs_info_df:bool = True
    floorlevel_only:bool = False
    use_all_data_gaze:bool = True # If true, we use the complete dfs from gaze that includes info about who is looking to who
    floorlevel_info_df = pd.read_csv("data/group_names_with_time_floorlevel.csv", sep=";")
    times_info_df = pd.read_csv("data/group_durations_all_commas.csv", sep=",")
    df_gaze_features = pd.read_csv("data/gaze_features_all_groups_interaction_90Hz_2026-04-01.csv")
    df_gaze_features_dyads_complete:pd.DataFrame = pd.read_csv("data/gaze_features_dyads_interaction_all_data_90Hz_2026-04-08.csv")
    df_gaze_features_triads_complete:pd.DataFrame = pd.read_csv("data/gaze_features_triads_interaction_all_data_90Hz_2026-04-08.csv")
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
        
        # GAZE X SPEAKING FEATURES
        # Calculate features that cross with gaze
        if use_all_data_gaze:            
            df_complete = df_gaze_features_dyads_complete if not triad else df_gaze_features_triads_complete
            df_gaze_group = df_complete[df_complete['group_name'] == group_name]
        else:
            df_gaze_group = df_gaze_features[df_gaze_features['group_name'] == group_name]
        # Create a timestamp index column for merge
        df_gaze_group['timestamp'] = pd.to_datetime(df_gaze_group['seconds'], unit='s')
        df_gaze_group = df_gaze_group.set_index('timestamp')
        df_gaze_group = df_gaze_group.sort_values('timestamp')
        group_dis_df.index.name = ' PEPE'
        group_dis_df['timestamp'] = pd.to_datetime(group_dis_df["time_ms"], unit='ms')
        group_dis_df = group_dis_df.set_index('timestamp')
        group_dis_df = group_dis_df.sort_values('timestamp')
        df_gaze_speaking = pd.merge_asof(group_dis_df, df_gaze_group, on='timestamp', direction='nearest')
        df_gaze_speaking.rename(columns={'1d_DG':'DG'}, inplace=True)
        # df_gaze_speaking_B = pd.merge_asof(df_gaze_group, group_dis_df, on='timestamp', direction='nearest')
        # Transform all gaze features to bool before we can do boolean comparisons
        cols = ['MG', 'DG', '0_D1']
        df_gaze_speaking[cols] = df_gaze_speaking[cols].astype(bool)  
        # G_OnSpeaker, OV is excluded!
        # blue = p1, green = p2, red = p3
        if use_all_data_gaze:
            # dyads
            if not triad:
                df_gaze_speaking['G_OnSpeaker'] = (
                    # p1 (blue) speaks, p2 looks
                    ((df_gaze_speaking['speaking_p_blue'] & (df_gaze_speaking['DG_P2_target'] == 1))
                    # p2 (green) speaks, p1 looks
                    | (df_gaze_speaking['speaking_p_green'] & (df_gaze_speaking['DG_P1_target'] == 2)))
                    # This ensures that there is only one participant talking
                    & (df_gaze_speaking['SP']))
                df_gaze_speaking['G_SP_OnSilent'] = (
                    # p1 (blue) speaks, p1 looks to p2
                    ((df_gaze_speaking['speaking_p_blue'] & (df_gaze_speaking['DG_P1_target'] == 2))
                    # p2 (green) speaks, p2 looks to p1
                    | (df_gaze_speaking['speaking_p_green'] & (df_gaze_speaking['DG_P2_target'] == 1)))
                    # This ensures that there is only one participant talking
                    & (df_gaze_speaking['SP']))
            # triads
            else:
                df_gaze_speaking['G_OnSpeaker'] = (
                    # p1 (blue) speaks, p2 or p3 look at p1
                    ((df_gaze_speaking['speaking_p_blue'] & ((df_gaze_speaking['DG_P2_target'] == 1) | (df_gaze_speaking['DG_P3_target'] == 1)))
                    # p2 (green) speaks, p1 or p3 look at p2
                    | (df_gaze_speaking['speaking_p_green'] & ((df_gaze_speaking['DG_P1_target'] == 2) | (df_gaze_speaking['DG_P3_target'] == 2)))
                    # p3 (red) speaks, p1 or p2 look at p1
                    | (df_gaze_speaking['speaking_p_red'] & ((df_gaze_speaking['DG_P1_target'] == 3) | (df_gaze_speaking['DG_P2_target'] == 3))))
                    # This ensures that there is only one participant talking
                    & (df_gaze_speaking['SP']))
                df_gaze_speaking['G_SP_OnSilent'] = (
                    # p1 (blue) speaks, p1 looks at p2 or p3 
                    ((df_gaze_speaking['speaking_p_blue'] & ((df_gaze_speaking['DG_P1_target'] == 2) | (df_gaze_speaking['DG_P1_target'] == 3)))
                    # p2 (green) speaks, p2 looks at p1 or p3
                    | (df_gaze_speaking['speaking_p_green'] & ((df_gaze_speaking['DG_P2_target'] == 1) | (df_gaze_speaking['DG_P2_target'] == 3)))
                    # p3 (red) speaks, p3 looks at p1 or p2 
                    | (df_gaze_speaking['speaking_p_red'] & ((df_gaze_speaking['DG_P3_target'] == 1) | (df_gaze_speaking['DG_P3_target'] == 2))))
                    # This ensures that there is only one participant talking
                    & (df_gaze_speaking['SP']))
        df_gaze_speaking['G_SP'] = ((df_gaze_speaking['MG'] | df_gaze_speaking['DG']) & df_gaze_speaking['SP'])
        G_SP_count = 0
        G_SP_amount = 0
        G_OnSpeaker_count = 0
        G_OnSpeaker_amount = 0
        G_SP_OnSilent_count = 0
        G_SP_OnSilent_amount = 0
        if True in df_gaze_speaking['G_SP'].value_counts():
            G_SP_count = df_gaze_speaking['G_SP'].value_counts()[True]
            G_SP_amount = G_SP_count/len(df_gaze_speaking)
        if True in df_gaze_speaking['G_OnSpeaker'].value_counts():
            G_OnSpeaker_count = df_gaze_speaking['G_OnSpeaker'].value_counts()[True]
            G_OnSpeaker_amount = G_OnSpeaker_count/len(df_gaze_speaking)
        if True in df_gaze_speaking['G_SP_OnSilent'].value_counts():
            G_SP_OnSilent_count = df_gaze_speaking['G_SP_OnSilent'].value_counts()[True]
            G_SP_OnSilent_amount = G_SP_OnSilent_count/len(df_gaze_speaking)
        # G_OnSpeaker_OV, OV is included exclusively
        # blue = p1, green = p2, red = p3
        if use_all_data_gaze:
            # dyads
            if not triad:
                df_gaze_speaking['G_OnSpeaker_OV'] = (
                    # p1 (blue) speaks, p2 looks
                    ((df_gaze_speaking['speaking_p_blue'] & (df_gaze_speaking['DG_P2_target'] == 1))
                    # p2 (green) speaks, p1 looks
                    | (df_gaze_speaking['speaking_p_green'] & (df_gaze_speaking['DG_P1_target'] == 2)))
                    # This ensures that only overlap frames are included
                    & ((df_gaze_speaking['OV_all_SP']) | (df_gaze_speaking['OV_two_SP'])))
            # triads
            else:
                df_gaze_speaking['G_OnSpeaker_OV'] = (
                    # p1 (blue) speaks, p2 or p3 look at p1
                    ((df_gaze_speaking['speaking_p_blue'] & ((df_gaze_speaking['DG_P2_target'] == 1) | (df_gaze_speaking['DG_P3_target'] == 1)))
                    # p2 (green) speaks, p1 or p3 look at p2
                    | (df_gaze_speaking['speaking_p_green'] & ((df_gaze_speaking['DG_P1_target'] == 2) | (df_gaze_speaking['DG_P3_target'] == 2)))
                    # p3 (red) speaks, p1 or p2 look at p1
                    | (df_gaze_speaking['speaking_p_red'] & ((df_gaze_speaking['DG_P1_target'] == 3) | (df_gaze_speaking['DG_P2_target'] == 3))))
                    # This ensures that only overlap frames are included
                    & ((df_gaze_speaking['OV_all_SP']) | (df_gaze_speaking['OV_two_SP'])))                
        G_OnSpeaker_OV_count = 0
        G_OnSpeaker_OV_amount = 0
        if True in df_gaze_speaking['G_OnSpeaker_OV'].value_counts():
            G_OnSpeaker_OV_count = df_gaze_speaking['G_OnSpeaker_OV'].value_counts()[True]
            G_OnSpeaker_OV_amount = G_OnSpeaker_OV_count/len(df_gaze_speaking)
        # No_G_OnSpeaker, OV is excluded
        # blue = p1, green = p2, red = p3
        if use_all_data_gaze:
            # dyads
            if not triad:
                df_gaze_speaking['No_G_OnSpeaker'] = (
                    # p1 (blue) speaks, p2 looks
                    ((df_gaze_speaking['speaking_p_blue'] & (df_gaze_speaking['DG_P2_target'] == 0))
                    # p2 (green) speaks, p1 looks
                    | (df_gaze_speaking['speaking_p_green'] & (df_gaze_speaking['DG_P1_target'] == 0)))
                    # This ensures that there is only one participant talking
                    & (df_gaze_speaking['SP']))
            # triads
            else:
                df_gaze_speaking['No_G_OnSpeaker'] = (
                    # p1 (blue) speaks, p2 and p3 DON'T look at p1
                    ((df_gaze_speaking['speaking_p_blue'] & ((df_gaze_speaking['DG_P2_target'] != 1) & (df_gaze_speaking['DG_P3_target'] != 1)))
                    # p2 (green) speaks, p1 and p3 DON'T look at p2
                    | (df_gaze_speaking['speaking_p_green'] & ((df_gaze_speaking['DG_P1_target'] != 2) & (df_gaze_speaking['DG_P3_target'] != 2)))
                    # p3 (red) speaks, p1 and p2 DON'T look at p1
                    | (df_gaze_speaking['speaking_p_red'] & ((df_gaze_speaking['DG_P1_target'] != 3) & (df_gaze_speaking['DG_P2_target'] != 3))))
                    # This ensures that there is only one participant talking
                    & (df_gaze_speaking['SP']))
        df_gaze_speaking['No_G_SP'] = ((df_gaze_speaking['0_D1']) & df_gaze_speaking['SP'])
        No_G_SP_count = 0
        No_G_SP_amount = 0
        No_G_OnSpeaker_count = 0
        No_G_OnSpeaker_amount = 0
        if True in df_gaze_speaking['No_G_SP'].value_counts():
            No_G_SP_count = df_gaze_speaking['No_G_SP'].value_counts()[True]
            No_G_SP_amount = No_G_SP_count/len(df_gaze_speaking)
        if True in df_gaze_speaking['No_G_OnSpeaker'].value_counts():
            No_G_OnSpeaker_count = df_gaze_speaking['No_G_OnSpeaker'].value_counts()[True]
            No_G_OnSpeaker_amount = No_G_OnSpeaker_count/len(df_gaze_speaking)
        # G_SI (G means either MG or DG)
        df_gaze_speaking['G_SI'] = ((df_gaze_speaking['MG'] | df_gaze_speaking['DG']) & df_gaze_speaking['SI'])
        G_SI_count = 0
        G_SI_amount = 0
        if True in df_gaze_speaking['G_SI'].value_counts():
            G_SI_count = df_gaze_speaking['G_SI'].value_counts()[True]
            G_SI_amount = G_SI_count/len(df_gaze_speaking)
        # No_G_SI (no G, no talk)
        df_gaze_speaking['No_G_SI'] = (df_gaze_speaking['0_D1'] & df_gaze_speaking['SI'])
        No_G_SI_count = 0
        No_G_SI_amount = 0
        if True in df_gaze_speaking['No_G_SI'].value_counts():
            No_G_SI_count = df_gaze_speaking['No_G_SI'].value_counts()[True]
            No_G_SI_amount = No_G_SI_count/len(df_gaze_speaking)
        

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
            "OV_two_sp": OV_two_speak_amount*100,
            "G_OnSpeaker": G_OnSpeaker_amount*100,
            "G_OnSpeaker_OV": G_OnSpeaker_OV_amount*100,
            "G_SP_OnSilent": G_SP_OnSilent_amount*100,
            "No_G_OnSpeaker": No_G_OnSpeaker_amount*100,
            "G_SP": G_SP_amount*100,
            "No_G_SP": No_G_SP_amount*100,
            "G_SI": G_SI_amount*100,
            "No_G_SI": No_G_SI_amount*100
        }
        speaking_configs_list.append(group_speaking_config_dict)
        speaking_all_groups_df_list.append(group_dis_df)

    speaking_configs_df = pd.DataFrame(speaking_configs_list)
    speaking_all_groups_df = pd.concat(speaking_all_groups_df_list, ignore_index=True)
    speaking_all_groups_df['session'] = speaking_all_groups_df['session'].str.replace(r'^recording_', '', regex=True)
    # save file with all frames
    if save_all_groups_df:
        speaking_all_groups_df.to_csv("data/discover/merged/all_groups_interaction_speaking_90Hz.csv")
    if save_resampled:
        speaking_all_groups_df_resampled = pd.concat(speaking_all_groups_df_resampled_list, ignore_index=True)
        speaking_all_groups_df_resampled.to_csv(f'data/discover/merged/all_groups_interaction_speaking_per_{resample_window_size}s.csv')
    if save_group_configs_info_df:
        speaking_configs_df.to_csv("data/speaking_configs_info.csv")

    print("hola")