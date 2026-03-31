import pandas as pd

if __name__ == "__main__": 
    # vars
    upsample: bool = True
    save_upsampled:bool = True
    floorlevel_info_pd = pd.read_csv("data/group_names_with_time_floorlevel.csv", sep=";")
    times_info_pd = pd.read_csv("data/group_durations_all_commas.csv", sep=",")
    group_names = floorlevel_info_pd["Group_Name"].to_list()
    speaking_configs_list = []
    for group_name in group_names:
        group_dis_pd = pd.read_csv(f"data/discover/merged/recording_{group_name}.csv")
        print(group_name)
        # set dyad or triad participants
        participants = ["speaking_p_blue", "speaking_p_green"] if "dyad" in group_name else ["speaking_p_blue", "speaking_p_green", "speaking_p_red"]
        triad = True if "triad" in group_name else False  

        if upsample:
            # resample dataframe to 90 Hz
            group_dis_pd['timestamp'] = pd.to_datetime(group_dis_pd['time_ms'], unit='ms')
            group_dis_pd = group_dis_pd.set_index('timestamp')
            # 11 ms expressed in pandas offset alias
            freq = '11ms'          # "L" = millisecond, can also use "11ms"

            # Create a new index that spans the whole range at 11 ms steps
            new_index = pd.date_range(start=group_dis_pd.index.min(),
                                    end=group_dis_pd.index.max(),
                                    freq=freq)

            # Reindex onto the new grid (introduces NaNs)
            df_resampled = group_dis_pd.reindex(new_index)
            df_resampled.index.name = 'timestamp'
            df_merged = pd.merge_asof(df_resampled.sort_values('timestamp'), group_dis_pd.sort_values('timestamp'), on='timestamp', direction='nearest')
            df_merged.drop(columns=[c for c in df_merged.columns
                       if c.endswith('_x') and c != 'time_ms_x'], inplace=True)
            # Linear interpolation (default for numeric columns)
            df_interpolated = df_merged.interpolate(method='linear', limit_direction='forward', limit=None).ffill()
            df_interpolated.drop(columns=['time_ms_y'], inplace=True)
            df_interpolated.columns = df_interpolated.columns.str.replace(r'_[xy]$', '', regex=True)
            # Reassign df for later processing
            group_dis_pd = df_interpolated
            if save_upsampled:
                group_dis_pd.to_csv(f"data/discover/merged/recording_{group_name}_90Hz.csv")
            

        # slice interaction time
        inter_start_ms = times_info_pd[times_info_pd["name"]==group_name]["offset_recording_interaction_start"].iloc[0]*1000
        inter_length_ms = times_info_pd[times_info_pd["name"]==group_name]["duration_interaction"].iloc[0]*1000
        inter_end_ms = inter_start_ms+inter_length_ms
        first_inter_frame = (group_dis_pd['time_ms']-inter_start_ms).abs().argsort()[:1].iloc[0]
        last_inter_frame = (group_dis_pd['time_ms']-inter_end_ms).abs().argsort()[:1].iloc[0]
        group_dis_pd = group_dis_pd[first_inter_frame:last_inter_frame]              

        # Speaking configs
        length_df = len(group_dis_pd.index)
        # Dyads
        # SP (One speaks)
        SP_mask = group_dis_pd[participants].sum(axis=1) == 1
        SP_count = 0
        if True in SP_mask.value_counts():
            SP_count = SP_mask.value_counts()[True]         
        SP_amount = SP_count/length_df
        # SI (All silent (~ symbol negates the dataframe))
        SI_mask = ~group_dis_pd[participants].any(axis=1)
        SI_count = 0
        if True in SI_mask.value_counts():
            SI_count = SI_mask.value_counts()[True]
        SI_amount = SI_count/length_df
        # OV (Overlap)
        # All speak
        OV_all_speak_mask = group_dis_pd[participants].all(axis=1)
        OV_all_speak_count = 0
        if True in OV_all_speak_mask.value_counts():
            OV_all_speak_count = OV_all_speak_mask.value_counts()[True]  
        OV_all_speak_amount = OV_all_speak_count /length_df
        # Two speak (only triads)
        OV_two_speak_count = 0
        OV_two_speak_amount = 0
        if triad:
            OV_two_speak_mask = group_dis_pd[participants].sum(axis=1) == 2
            if True in OV_two_speak_mask.value_counts():
                OV_two_speak_count = OV_two_speak_mask.value_counts()[True]
                OV_two_speak_amount = OV_two_speak_count /length_df

        group_speaking_config_dict = {
            "name":group_name,
            "type": "triad" if triad else "dyad",
            "SP": SP_amount*100,
            "SI": SI_amount*100,
            "OV_all_sp": OV_all_speak_amount*100,
            "OV_two_sp": OV_two_speak_amount*100
        }
        speaking_configs_list.append(group_speaking_config_dict)

        print(group_dis_pd.size)
    speaking_confis_pd = pd.DataFrame(speaking_configs_list)

    print("hola")