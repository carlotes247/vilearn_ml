import csv
import pandas as pd
import numpy as np
from datetime import datetime
from data_reading.features.group_feature_frame import GroupFeatureFrame
from data_reading.features.blink_feature import BlinkFeature
from data_reading.features.direct_gaze_feature import DirectGazeFeature
from data_reading.features.gaze_behaviour_feature import GazeBehaviourFeature
from data_reading.features.participant_features import ParticipantFeatures
from torch_vilearn.torch_group_dataset import TorchGroupDataset
from torch_vilearn.torch_group_data_loader import TorchGroupDataLoader
import os

class GroupCSVDataLoader:

#region variables

    raw_data: pd.DataFrame
    torch_dataset: TorchGroupDataset
    torch_loader: TorchGroupDataLoader
    onlyTorch: bool
    group_name: str
    fileLoaded: bool

#endregion

#region Constructor

    def __init__(self, filePath: str, onlyTorch: bool):
        if (filePath):
            try:
                self.onlyTorch = onlyTorch
                if (not onlyTorch):
                    self.raw_data = pd.read_csv(filePath, sep=';')
                else:
                    self.torch_dataset = TorchGroupDataset(filePath)
                    self.torch_loader = TorchGroupDataLoader(self.torch_dataset)
                self.group_name = os.path.basename(filePath)
                self.fileLoaded = True
            except FileNotFoundError:
                print(f"Error: File not found when loading group feature data file {filePath}")
                self.fileLoaded = False

#endregion

#region getters
    def extract_group_feature_frames(self) -> list[GroupFeatureFrame]:
        if self.fileLoaded and not self.onlyTorch:
            list_feauture_frames = []
            count_participants = self.raw_data.columns.str.contains("Participant").sum()
            for index, row in self.raw_data.iterrows():
                featureFrame = GroupFeatureFrame(group_name=self.group_name, ts_group_string=row["TSGroupNTP"], cognition=row["GroupCognition"])
                # Construct participant features per participant. Pandas appends a '.1' or '.2' to each repeated header, that way we know to which participant each header belongs to
                for i in range(count_participants):
                    # Construct appended header value
                    suffix = ""
                    if i > 0:
                        suffix = f".{i}"
                    # Construct Gaze Behaviour
                    direct_gaze = DirectGazeFeature(direct_gaze_value=row[f"DirectGaze{suffix}"], participant_name=row[f"Participant{i+1}"], target=row[f"TargetGaze{suffix}"])
                    blink = BlinkFeature(blink=row[f"Blink{suffix}"])
                    gaze_feature = GazeBehaviourFeature(direct_gaze_data=direct_gaze, blink_data=blink)
                    participant_feature = ParticipantFeatures(name=self.raw_data[f"Participant{i+1}"], gaze_behaviours=gaze_feature)
                    featureFrame.add_participant_feature(participant_feature)
                    list_feauture_frames.append(featureFrame)
        else:
            print(f"Can't extract group feature frames because the file wasn't loaded correctly!")
        return list_feauture_frames
    
    def extract_valid_blinks_frames(self) -> dict[int, list[bool]]:
        """
        Returns TWO dicts, one with a list of valid blinks per frame per participant and another with blink onsets per participant
        """
        if self.fileLoaded:
            print("stuff")
            # A blink occurs taking into account previous time. We have a window of 400ms (actually 100-400ms) (Nakano and Miyazaki, 2019) and detect big changes in the eye openness to calculate blink onset/offset, I assume with good eye confidence. It is worth plotting this to understand how the eye opennes and the confidence data looks like over time    
            num_participants = 3 if 'BlinkP3' in self.raw_data else 2
            prior_blink_happenned: dict[int, bool] = dict()
            blinks_participants: dict[int, list[bool]] = dict()
            blinks_onsets_participants: dict[int, list[bool]] = dict()
            blinks_indexes_window: dict[int, list[int]] = dict()            
            # init blink lists            
            for p_index in range(num_participants):
                prior_blink_happenned[p_index] = False
                blinks_list = [False] * len(self.raw_data[f'LeftEyeOpennesP{p_index+1}'])
                blinks_onsets_list = [False] * len(blinks_list)
                blinks_participants[p_index] = blinks_list
                blinks_onsets_participants[p_index] = blinks_onsets_list
                blinks_indexes_window[p_index] = []
            # Row by row
            for index_row, row in self.raw_data.iterrows():
                # Participant per row
                for p_index in range(num_participants):
                    current_blink = row[f'LeftEyeOpennesP{p_index+1}'] and row[f'LeftEyeOpennesConfidenceP{p_index+1}'] and row[f'RightEyeOpennesP{p_index+1}'] and row[f'RightEyeOpennesConfidenceP{p_index+1}']
                    # Three conditions
                    #   1. Start of blink window 
                    #   2. Continue blink window
                    #   3. End blink window
                    # Start of blink window
                    if (prior_blink_happenned[p_index] == False and current_blink == True):                        
                        blinks_onsets_participants[p_index][index_row] = True
                        blinks_participants[p_index][index_row] = True                        
                        blinks_indexes_window[p_index].append(index_row)
                        prior_blink_happenned[p_index] = True                        
                    # Continue blink window
                    elif(prior_blink_happenned[p_index] == True and current_blink == True):
                        blinks_participants[p_index][index_row] = True
                        blinks_indexes_window[p_index].append(index_row)
                        prior_blink_happenned[p_index] = True
                    # End blink window
                    elif(prior_blink_happenned[p_index]== True and current_blink == False):
                        # Check if blink window is not in 100-500ms range
                        first_blink_index = blinks_indexes_window[p_index][0]
                        last_blink_index = blinks_indexes_window[p_index][-1]
                        first_blink_TS = datetime.strptime(self.raw_data.iloc[first_blink_index]['TSGroupNTP'], '%Y-%m-%d %H:%M:%S.%f')
                        last_blink_TS = datetime.strptime(self.raw_data.iloc[last_blink_index]['TSGroupNTP'], '%Y-%m-%d %H:%M:%S.%f')
                        length_blink_ms = (last_blink_TS - first_blink_TS).total_seconds() * 1000;
                        if (length_blink_ms < 100 or length_blink_ms > 500):
                            # Set to false all stored blinks because they are invalid
                            blinks_onsets_participants[p_index][first_blink_index] = False
                            for blink_index in blinks_indexes_window[p_index]:                                
                                blinks_participants[p_index][blink_index] = False
                        # Regardless of validity we clear window to find a new set of blinks ahead in the data
                        blinks_indexes_window[p_index].clear()  
                        prior_blink_happenned[p_index] = False
            # Return final list of valid blinks once all rows are computed
            return blinks_participants, blinks_onsets_participants
        else:
            print(f"Can't extract group feature frames because the file wasn't loaded correctly!")
        return None
#endregion