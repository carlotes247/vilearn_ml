import pandas as pd

if __name__ == "__main__": 
    floorlevel_info_pd = pd.read_csv("data/group_names_with_time_floorlevel.csv", sep=";")
    times_info_pd = pd.read_csv("data/group_durations_all_commas.csv", sep=",")
    group_names = floorlevel_info_pd["Group_Name"].to_list()
    speaking_configs_list = []
    for group_name in group_names:
        group_dis_pd = pd.read_csv(f"data/discover/merged/recording_{group_name}.csv")
        print(group_name)
         # slice interaction time
        inter_start_ms = times_info_pd[times_info_pd["name"]==group_name]["offset_recording_interaction_start"].iloc[0]*1000
        inter_length_ms = times_info_pd[times_info_pd["name"]==group_name]["duration_interaction"].iloc[0]*1000
        inter_end_ms = inter_start_ms+inter_length_ms
        first_inter_frame = (group_dis_pd['time_ms']-inter_start_ms).abs().argsort()[:1].iloc[0]
        last_inter_frame = (group_dis_pd['time_ms']-inter_end_ms).abs().argsort()[:1].iloc[0]
        group_dis_pd = group_dis_pd[first_inter_frame:last_inter_frame]
        # set dyad or triad participants
        participants = ["speaking_p_blue", "speaking_p_green"] if "dyad" in group_name else ["speaking_p_blue", "speaking_p_green", "speaking_p_red"]
        triad = True if "triad" in group_name else False        

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
    print("hola")