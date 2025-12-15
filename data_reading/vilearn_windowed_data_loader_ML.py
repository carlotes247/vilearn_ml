import pandas as pd
import numpy as np
from sklearn import model_selection
import os

class VilearnWindowedDataLoaderML:
    """
    Loads data in windows ready to be served for ML training
    """

    #region VARS
    # config flags
    use_file_TE: bool = True
    floorlevel_groups: bool = True
    print_folds: bool = False
    debug_all_folds: bool = False
    bins_binary: bool = False
    dyads_only: bool = False
    triads_only: bool = False
    # feature selection vars
    features_selected = []
    # bining vars
    labels_text = []
    # three classes
    bins_three = [-0.1, 0.3, 0.7, 1]
    labels_three = [0, 1, 2]
    labels_text_three = ['low', 'middle', 'high']
    # two classes
    bins_two = [-0.1, 0.5, 1]
    labels_two = [0, 1]
    labels_text_two = ['low', 'high']
    # dataframes
    df_data: pd.DataFrame
    df_X: pd.DataFrame = pd.DataFrame()
    df_y: pd.DataFrame = pd.DataFrame()
    X: pd.DataFrame = pd.DataFrame()
    y: pd.Series = pd.Series()
    X_train: pd.DataFrame = pd.DataFrame()
    y_train: pd.Series = pd.Series()
    X_test: pd.DataFrame = pd.DataFrame()
    y_test: pd.Series = pd.Series()
    # k fold vars
    group_names: np.ndarray
    groups: pd.Series = pd.Series()
    group_kfold: model_selection.GroupKFold = model_selection.GroupKFold()
    # paths
    working_dir = os.getcwd()
    data_folder = 'Recordings/SavedData/v2_no_low_sampled'
    data_file = 'all_features_60s_resampled.csv'
    data_file_with_TE = '60s_TE_correlation.csv'
    floor_level_groups_info_filepath = 'data/group_names_with_time_floorlevel.csv'
    sep = ";" if use_file_TE else ","
    full_data_path = ""
    #endregion

    #region CONSTRUCTOR
    def __init__(self, bins_binary:bool, print_folds: bool, debug_all_folds: bool, floorlevel: bool = True, dyads_only: bool = False, triads_only: bool = False, features_to_select: list[str] = []) -> None:
        self.use_file_TE = True
        self.floorlevel_groups = floorlevel
        self.print_folds = print_folds
        self.debug_all_folds = debug_all_folds
        self.bins_binary = bins_binary
        self.dyads_only = dyads_only
        self.triads_only = triads_only
        # configure features that will be selected
        if len(features_to_select) < 1:
            self.features_selected = self.get_original_features_list()
        else:
            self.features_selected = features_to_select
        # load vilearn windowed data
        if self.use_file_TE:
            full_data_path = os.path.join(self.working_dir, self.data_folder, self.data_file_with_TE)
        else:
            full_data_path = os.path.join(self.working_dir, self.data_folder, self.data_file)
        self.df_data = pd.read_csv(full_data_path, sep=";")
        # drop groups that are not floorlevel
        if self.floor_level_groups_info_filepath:
            df_floorlevel_info = pd.read_csv(self.floor_level_groups_info_filepath, sep=";")
            groups_floorlevel = df_floorlevel_info['Group_Name']
            self.df_data = self.df_data[self.df_data['group_name'].isin(groups_floorlevel)]
            self.df_data.reset_index(drop=True, inplace=True)
        # logic for TE file used for R correlation analysis (long dataframe)
        if self.use_file_TE:
            # binning TE
            bins = []
            labels_text= []
            if self.bins_binary:
                bins = self.bins_two
                labels_text = self.labels_text_two
                pass
            else:
                bins = self.bins_three
                labels_text = self.labels_text_three
                pass
            self.labels_text = labels_text # assign so the selected labels are accessible from outside the class            
            # separate per group if requested
            if self.dyads_only:
                dyad_rows = self.df_data['group_type'] == 'dyad'
                self.df_data = self.df_data[dyad_rows].reset_index(drop=True)
            elif self.triads_only:
                triad_rows = self.df_data['group_type'] == 'triad'
                self.df_data = self.df_data[triad_rows].reset_index(drop=True)
            # separate into features (X) and labels (y)
            self.df_y = self.df_data.loc[:, ['TE']].apply(pd.cut, bins=bins, labels=labels_text)
            self.df_data['TE'] = self.df_y
            self.df_X = self.df_data.loc[:, self.df_data.columns != 'TE']
        # logic for wide dataframe with all gaze configurations (missing TE and blink metrics)
        else:
            # TODO: not implemented, check knn_test.py for some starting logic (unfinished there)
            pass

        # splitting into train, test set
        # since we want whole groups and not individual windows, we use a proxy var to calculate the split
        self.group_names = self.df_X['group_name'].unique()
        # Using GroupkFold to do the split 
        self.X = self.df_data.drop(columns=['seconds_interaction_window','group_type','group_formation','group_name' ,'TE'])
        self.y = self.df_data.loc[:, 'TE']
        self.groups = self.df_data['group_name']
        # Select features if requested
        self.select_features(self.features_selected)

        print("todo")

        self.group_kfold = model_selection.GroupKFold(n_splits=len(self.group_names))
        n_splits = self.group_kfold.get_n_splits(self.X, self.y, self.groups)
        if self.print_folds:
                print(f"Num folds: {n_splits}")
                print(self.group_kfold)
        # debug code to understand if all folds are formed correctly
        if self.debug_all_folds:
            # loop through each fold
            for i, (train_index, test_index) in enumerate(self.group_kfold.split(self.X, self.y, self.groups)):                                    
                # debug info
                if self.print_folds:
                    print(f"Fold {i}:")            
                    print(f"  TRAIN: groups={self.groups[train_index].unique()}, count={len(self.groups[test_index].unique())}")
                    print("")
                    print(f"  TEST: groups={self.groups[test_index].unique()}, count={len(self.groups[test_index].unique())}")
                    if len(self.groups[test_index].unique()) > 1:
                        print("THIS FOLD HAS MORE THAN ONE GROUP!!!")
                    print("============================")    
        
    def select_features(self, features: list[str]) -> None:
        if (features != None or (len(features) > 0) and len(features) <= len(self.features_selected)):
            self.features_selected = features
        if (self.X.columns.to_list() != self.features_selected):
            self.X = self.X[self.features_selected]

    def get_features_selected(self) -> list[str]:
        return self.features_selected
    
    def get_original_features_list(self) -> list[str]:
        return ['MG','1d_DG','BPM','blink_durations']

if __name__ == "__main__":
    test = VilearnWindowedDataLoaderML(bins_binary=False, print_folds=False, 
                                       debug_all_folds=False, triads_only=True,
                                       features_to_select=['1d_DG','BPM','blink_durations'])    