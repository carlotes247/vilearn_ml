import datetime
from data_reading.features.participant_features import ParticipantFeatures

class GroupFeatureFrame:
    """
    Represents a single frame of group features
    """
    group_name: str
    ts_group: datetime.datetime
    ts_group_string: str
    participants_features: list[ParticipantFeatures]
    cognition_value: float

    def __init__(self, group_name: str, ts_group_string: str, cognition: float):
        self.group_name = group_name
        self.ts_group_string = ts_group_string
        self.ts_group = datetime.datetime.strptime(ts_group_string, '%Y-%m-%d %H:%M:%S.%f')
        self.participants_features = []
        self.cognition_value = cognition

    def add_participant_feature(self, participant_feature: ParticipantFeatures):
        self.participants_features.append(participant_feature)