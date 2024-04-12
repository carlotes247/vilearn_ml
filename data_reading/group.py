from participant import *

class Group:
    """
    An instance of a group, which can be a dyad or triad 
    """
    group_name: str
    csv_file_paths: list[str]
    audio_file_paths: list[str]
    participants: list[Participant]

    def __init__(self, csv_paths: list[str], audio_paths: list[str], group_name: str):
        if len(csv_paths) != len(audio_paths):
            raise Exception(f"Can't create group, lengths of csv and audio paths ({len(csv_paths)} vs {len(audio_paths)}) don't match for group {group_name}")
        self.csv_file_paths = csv_paths
        self.audio_file_paths = audio_paths
        self.participants = []
        for i in range(len(csv_paths)):
            aux_participant = Participant(csv_paths[i], audio_paths[i])
            self.participants.append(aux_participant)
        
