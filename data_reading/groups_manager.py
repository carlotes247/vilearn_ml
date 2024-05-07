from group import Group
import os
from pathlib import Path

class GroupsManager:
    """
    A manager that handles several groups 
    """
    path_prefix_data: str
    groups: list[Group]
    group_participant_csv_paths: list[str]
    group_participant_audio_paths: list[str]
    group_features_path: str


    def __init__(self, path_prefix: str, path_folder_groups: str, specific_group: str):
        # Ignore lines with the # symbol to read the final uncommented line with the path prefix
        with open(path_prefix) as path_prefix_file:
            for line in path_prefix_file:
                if not line.startswith('#'):
                    self.path_prefix_data = line
        self.groups = []
        for group_file_name in os.listdir(path_folder_groups):
            # if we have a specific group to only load data from, skip until that group is loaded
            if (specific_group and specific_group != "" and specific_group != Path(group_file_name).stem):
                continue
            # Construct the full file path
            group_file_path = os.path.join(path_folder_groups, group_file_name)
            # Read contents of path file
            with open(group_file_path) as group_file:
                group_data_paths = group_file.read().splitlines()
            # Separate paths for csv files from paths to audio files
            group_participant_csv_paths = []
            group_participant_audio_paths = []
            group_features_path = ""
            for data_path_line in group_data_paths:
                # Group features file
                if data_path_line.startswith("GroupFeatures: "):
                    group_features_path = os.path.normpath(os.path.join(self.path_prefix_data, data_path_line.removeprefix("GroupFeatures: ")))
                # Participant features
                else:
                    # Construct full path
                    full_data_path = os.path.normpath(os.path.join(self.path_prefix_data, data_path_line))
                    if full_data_path.endswith(".csv"):
                        group_participant_csv_paths.append(full_data_path)
                    elif full_data_path.endswith(".wav"):
                        group_participant_audio_paths.append(full_data_path)
            # Instantiate group and add to list of groups
            aux_group = Group(group_participant_csv_paths, group_participant_audio_paths, group_features_path, group_file_name)
            self.groups.append(aux_group)

            