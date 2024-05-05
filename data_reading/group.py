from participant import *
from features import *

class Group:
    """
    An instance of a group, which can be a dyad or triad 
    """
    group_name: str
    num_participants: int
    # paths
    csv_participants_file_paths: list[str]
    audio_file_paths: list[str]
    csv_group_features_file_path: str
    csv_group_cognition_file_path: str
    # participant raw data
    participants: list[Participant]
    # group feature data
    group_feature_frames: list[GroupFeatureFrame]
    group_data_loader: GroupCSVDataLoader

    def __init__(self, csv_paths_participants: list[str], audio_paths: list[str], csv_path_group_features: str, group_name: str,  csv_path_group_annotations: str):
        if len(csv_paths_participants) != len(audio_paths):
            raise Exception(f"Can't create group, lengths of csv and audio paths ({len(csv_paths_participants)} vs {len(audio_paths)}) don't match for group {group_name}")
        self.group_name = group_name
        self.num_participants = len(csv_paths_participants)
        self.csv_participants_file_paths = csv_paths_participants
        self.audio_file_paths = audio_paths
        self.csv_group_features_file_path = csv_path_group_features
        self.csv_group_cognition_file_path = csv_path_group_annotations
        self.participants = []
        # Populate participant data
        for i in range(len(csv_paths_participants)):
            aux_participant = Participant(csv_paths_participants[i], audio_paths[i])
            self.participants.append(aux_participant)
        # If a group feature file is present, load file
        if (csv_path_group_features):
            self.group_data_loader = GroupCSVDataLoader(csv_path_group_features)
        # TODO: Extract feature frames from loaded group features file                
