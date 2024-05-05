import csv
import pandas as pd
import numpy as np
from datetime import datetime
from features import *
import os

class GroupCSVDataLoader:

#region variables

    raw_data: pd.DataFrame 
    group_name: str

#endregion

#region Constructor

    def __init__(self, filePath: str):
        if (filePath):
            self.raw_data = pd.read_csv(filePath, sep=';')
            self.group_name = os.path.basename(filePath)

#endregion

#region getters
    def extract_group_feature_frames(self) -> list[GroupFeatureFrame]:
        if self.raw_data:
            list_feauture_frames = []
            count_participants = self.raw_data.columns.str.contains("Participant").sum()
            for index, row in self.raw_data.iterrows():
                featureFrame = GroupFeatureFrame(group_name=self.group_name, ts_group_string=row["GroupCognition"], cognition=row["GroupCognition"])
                # Construct participant features per participant. Pandas appends a '.1' or '.2' to each repeated header, that way we know to which participant each header belongs to
                for i in range(count_participants):
                    # Construct appended header value
                    suffix = ""
                    if i > 0:
                        suffix = f".{i+1}"
                    # Construct Gaze Behaviour
                    direct_gaze = DirectGazeFeature(direct_gaze_value=row[f"DirectGaze{suffix}"], participant_name=row[f"Participant{i+1}"], target=row[f"TargetGaze{suffix}"])
                    blink = BlinkFeature(blink=row[f"Blink{suffix}"])
                    gaze_feature = GazeBehaviourFeature(direct_gaze_data=direct_gaze, blink_data=blink)
                    participant_feature = ParticipantFeatures(name=self.raw_data[f"Participant{i+1}"], gaze_behaviours=gaze_feature)
                    list_feauture_frames.append(participant_feature)

        return list_feauture_frames
#endregion