import csv
import pandas as pd
import numpy as np
from datetime import datetime
from features.group_feature_frame import GroupFeatureFrame
from features.blink_feature import BlinkFeature
from features.direct_gaze_feature import DirectGazeFeature
from features.gaze_behaviour_feature import GazeBehaviourFeature
from features.participant_features import ParticipantFeatures
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
#endregion