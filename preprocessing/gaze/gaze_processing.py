import pandas as pd
import datetime

# Cleans up and processes gaze data into a file to have the features MG, DG, 0DG per frame. The code seems all over the place in other files
class GazeProcessing():
    def __init__(self) -> None:
         pass
    
    def gaze_df_wide_to_long(self, df_wide_gaze, group_names, dyads: bool, triads: bool, keep_all_data: bool) -> pd.DataFrame:
            dyad_features_names = ["MG_P1P2","1d_DG_P1","1d_DG_P2","0_D1"]
            triad_features_names= ["", ""]
            df_long: pd.DataFrame = pd.DataFrame()
            for group_name in group_names:            
                group_cols = df_wide_gaze.columns.str.contains(group_name)
                df_group = df_wide_gaze[df_wide_gaze.columns[group_cols]]
                feature_names = dyad_features_names if "dyad" in group_name else triad_features_names  
                feature_cols = [col for col in df_group.columns if any(term in col for term in feature_names)]
                cols = df_wide_gaze.columns[group_cols]              
                if not keep_all_data:                    
                    df_group = df_group[feature_cols]
                df_group.dropna(inplace=True)
                if dyads and "dyad" in group_name:
                    col_names = ['seconds_interaction', 'MG', 'DG_P1_target', 'DG_P2_target', '1d_DG_P1', '1d_DG_P2', '0_D1'] if keep_all_data else ['MG', '1d_DG_P1', '1d_DG_P2', '0_D1']
                    df_group = df_group.set_axis(col_names, axis=1)
                    df_group['1d_DG'] = df_group['1d_DG_P1'] + df_group['1d_DG_P2']
                    if not keep_all_data:
                        df_group.drop(['1d_DG_P1', '1d_DG_P2'], axis=1, inplace=True)
                    df_group['group_type'] = 'dyad'
                if triads and "triad" in group_name:
                    df_group['MG'] = df_group[f"{group_name}_MG_D1"] + df_group[f"{group_name}_MG_D0"]
                    df_group['0_D1'] = df_group[f"{group_name}_0_D1"]
                    df_group['1d_DG'] = df_group[f"{group_name}_3_D1"] + df_group[f"{group_name}_2_D1_different"] + df_group[f"{group_name}_2_D1_same"] + df_group[f"{group_name}_1_D1"]
                    if not keep_all_data:
                        df_group.drop(cols, axis=1, inplace=True)
                    df_group.columns = df_group.columns.str.replace(f'{group_name}_', '', regex=False)
                    df_group['group_type'] = 'triad'                                
                if dyads and not triads and "triad" in group_name: 
                     continue
                if triads and not dyads and "dyad" in group_name:
                     continue
                if not 'seconds_interaction' in df_group.columns:
                    df_group.insert(0, 'seconds', df_wide_gaze.iloc[df_group.index][f'{group_name}_seconds_interaction'])
                else:
                    df_group.rename(columns={'seconds_interaction':'seconds'}, inplace = True)
                df_group['group_name'] = group_name
                df_long = pd.concat([df_long, df_group])
            df_long = df_long.rename(columns={"seconds_interaction":"seconds"})
            return df_long
    
if __name__ == "__main__":
    df_gaze_features_dyads_wide = pd.read_csv("Recordings/SavedData/v2_no_low_sampled/all_dyads_all_features_individualTS.csv")
    df_gaze_features_triads_wide = pd.read_csv("Recordings/SavedData/v2_no_low_sampled/all_triads_all_features_individualTS.csv")
    times_info_df = pd.read_csv("data/group_durations_all_commas.csv", sep=",")
    group_names = times_info_df["name"].to_list()
    save_to_file:bool = True
    keep_all_data: bool = True # If true we won't drop details on who is looking at who

    gazeProcessor = GazeProcessing()
    df_gaze_features_dyads_long = gazeProcessor.gaze_df_wide_to_long(df_wide_gaze=df_gaze_features_dyads_wide, group_names=group_names, dyads=True, triads=False, keep_all_data=keep_all_data)
    df_gaze_features_triads_long = gazeProcessor.gaze_df_wide_to_long(df_wide_gaze=df_gaze_features_triads_wide, group_names=group_names, dyads=False, triads=True, keep_all_data=keep_all_data)
    # We can concat both dfs if not all data is kept, because we are grouping features into ['MG', '1d_DG_P1', '1d_DG_P2', '0_D1']
    if not keep_all_data:
        df_gaze_features_all_groups_long = pd.concat([df_gaze_features_dyads_long, df_gaze_features_triads_long])
        if save_to_file:
            df_gaze_features_all_groups_long.to_csv(f"data/gaze_features_all_groups_interaction_90Hz_{datetime.datetime.now().date()}.csv")         
    # If we want to keep all data, we need to save them separately
    elif keep_all_data and save_to_file:
        df_gaze_features_dyads_long.to_csv(f"data/gaze_features_dyads_interaction_all_data_90Hz_{datetime.datetime.now().date()}.csv")
        df_gaze_features_triads_long.to_csv(f"data/gaze_features_triads_interaction_all_data_90Hz_{datetime.datetime.now().date()}.csv")
