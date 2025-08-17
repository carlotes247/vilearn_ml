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


def create_line_plot(file_path:str, timewindow:int = 30, save_plot = False):
    df_data = pd.read_csv(file_path, index_col=0)

    df_data_ts = convert_seconds_to_timestamp(df_data)
    df_plot_ready = resample_avg_seconds_using_timeframe(df_data_ts)
    print ("plotting")
    line_plot = df_plot_ready.plot.line(x='seconds_interaction_window').legend(loc='center left',bbox_to_anchor=(1.0, 0.5))
    plt.show()


if __name__ == "__main__":
    save_plot = False


    root_path= "../Recordings/SavedData/"
    dyads_df_path = root_path+"all_dyads_mutual_gaze_interaction_time.csv"
    triads_df_path = root_path+"all_triads_mutual_gaze_interaction_time.csv"
    all_groups_df_path = root_path+"all_groups_mutual_gaze_interaction_time.csv"

    # to do: make lists of the dyads and triads including their formation in order to discriminate over them when plotting
    create_line_plot(dyads_df_path)
    create_line_plot(triads_df_path)
    create_line_plot(all_groups_df_path)