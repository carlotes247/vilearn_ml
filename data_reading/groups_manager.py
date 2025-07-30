import torch.utils
import torch.utils.data
from data_reading.group import Group
from torch_vilearn.torch_group_dataset import TorchGroupDataset
from torch_vilearn.torch_group_data_loader import TorchGroupDataLoader
import os
from pathlib import Path
import pandas as pd
from typing import Optional
import datetime

class GroupsManager:
    """
    A manager that handles several groups 
    """
    path_prefix_data: str
    specific_group: str
    all_groups_names: list[str]
    groups: list[Group]
    group_participant_csv_paths: list[str]
    group_participant_audio_paths: list[str]
    group_features_path: str
    onlyTorch: bool = True
    groups_torch_data: list[TorchGroupDataset]
    use_async: bool = False
    all_groups_read: bool = False # Flag to check from outside
    load_individual_p_files: bool = False
    print_all_stats: bool = False
    print_blink_stats: bool = False
    print_debug: bool = False

    #region INIT
    def __init__(self, path_prefix: str, path_folder_groups: str, specific_group: str, all_groups_names_path: str, onlyTorch: bool, load_individual_p_files: bool, print_all_stats: bool, print_blink_stats: bool, use_async: bool, print_debug: bool):
        self.onlyTorch = onlyTorch
        self.groups_torch_data = []
        self.specific_group = specific_group
        # modify this to populate the list of all_groups_names from a list;
        if all_groups_names_path and all_groups_names_path != "":
            self.all_groups_names = pd.read_csv(all_groups_names_path).columns.to_list()
        else:
            self.all_groups_names = []
        self.use_async = use_async
        self.path_folder_groups = path_folder_groups
        self.load_individual_p_files = load_individual_p_files
        self.print_all_stats = print_all_stats
        self.print_blink_stats = print_blink_stats
        self.print_debug = print_debug
        # Ignore lines with the # symbol to read the final uncommented line with the path prefix
        with open(path_prefix) as path_prefix_file:
            for line in path_prefix_file:
                if not line.startswith('#'):
                    self.path_prefix_data = line.rstrip()
        self.groups = []
        
        # Read all groups if not in async mode
        if not self.use_async:
            self.read_all_groups_loop(path_folder_groups=self.path_folder_groups, specific_group=self.specific_group, load_individual_p_files=self.load_individual_p_files, print_all_stats=self.print_all_stats, print_blink_stats=self.print_blink_stats)       

    #endregion
    
    def read_group(self, csv_paths_participants: list[str], audio_paths: list[str], csv_path_group_features: str, group_name: str, onlyTorch: bool, load_individual_p_files: bool, print_all_stats: bool, print_blink_stats: bool) -> Group:
        """ NOT WORKING ASYNC BECAUSE pd.read_csv IS USED TO LOAD A GROUP"""
        # Instantiate and load group from disk and add to list of groups    
        aux_group = Group(csv_paths_participants, audio_paths, csv_path_group_features, group_name, onlyTorch=self.onlyTorch, load_individual_p_files=load_individual_p_files, print_all_stats=print_all_stats, print_blink_stats=print_blink_stats, print_debug=self.print_debug)
        self.groups.append(aux_group)    
        return aux_group      
    
    async def read_all_groups_async(self):
        """ Needs to be called from outside to trigger the async load """
        #self.async_event_loop = asyncio.get_running_loop()
        # Create task to read
        num_files_to_read: int = 25
        
        # THIS SHOULD BE AWAITED BUT I REMOVED THE ASYNC FROM HERE BECAUSE IN THE END pd.read_csv IS NOT ASYNCABLE
        self.read_all_groups_loop(path_folder_groups=self.path_folder_groups, specific_group=self.specific_group, load_individual_p_files=self.load_individual_p_files, print_all_stats=self.print_all_stats, print_blink_stats=self.print_blink_stats)

        # asyncio.create_task(self.read_all_groups_loop(path_folder_groups=self.path_folder_groups, specific_group=self.specific_group, load_individual_p_files=self.load_individual_p_files, print_all_stats=self.print_all_stats, print_blink_stats=self.print_blink_stats))
        
        # the async tasks list gets filled in automatically by passing the event loop to the read all groups list
        # while num_files_to_read != len(self.async_tasks):
        #     await asyncio.sleep(0.1)
        # asyncio.gather(*self.async_tasks)
        print("Waiting for files to read...")
        # while num_files_to_read != len(self.groups):
        #     await asyncio.sleep(10)
        for aux_group in self.groups:
            # Add group dataset to internal list of datasets
            if self.onlyTorch and not (aux_group.group_features_csv_loader is None):
                self.groups_torch_data.append(aux_group.group_features_csv_loader.torch_dataset)               
        self.all_groups_read = True
        print("all groups read!")

    def read_all_groups_loop(self, path_folder_groups: str, specific_group: str, load_individual_p_files: bool, print_all_stats: bool, print_blink_stats: bool) -> int:     
        """ Reads all groups in a for loop """   
        num_files: int = 0
        specific_group_exists: bool = (specific_group and specific_group != "")
        for group_file_name in os.listdir(path_folder_groups):
            # if we have a specific group to only load data from, skip until that group is loaded
            if (specific_group_exists and specific_group != Path(group_file_name).stem):
                continue
            # if the file is not in the list of group names to work with we skip to avoid loading errors
            if ((not specific_group_exists) and (not Path(group_file_name).stem in self.all_groups_names)):
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
                # Ignore comments
                if data_path_line.startswith("#"):
                    continue
                # Group features file if one available. It will not load individual participant files
                if data_path_line.startswith("GroupFeatures: "):
                    group_features_path = os.path.normpath(os.path.join(self.path_prefix_data, data_path_line.removeprefix("GroupFeatures: ")))
                # Participant individual files if no group file
                else:
                    # Construct full path
                    full_data_path = os.path.normpath(os.path.join(self.path_prefix_data, data_path_line))
                    if full_data_path.endswith(".csv"):
                        group_participant_csv_paths.append(full_data_path)
                    elif full_data_path.endswith(".wav"):
                        group_participant_audio_paths.append(full_data_path)
            # [NOT WORKING] Async load (fast) 
            if self.use_async:
                # [IT WILL NOT WORK BECAUSE pd.read_csv IS NOT ASYNC!!!] Define array of all tasks to run
                # TODO: write reading implementation that does not realy on pd.read_csv or an asyncable version of it                
                #self.async_tasks.append(task)
                pass
                #print(f"Task added! Num tasks: {len(self.async_tasks)}")
            # Sequential load (slow)
            else:
                # Instantiate and load group from disk and add to list of groups
                aux_group = Group(group_participant_csv_paths, group_participant_audio_paths, group_features_path, group_file_name, onlyTorch=self.onlyTorch, load_individual_p_files=load_individual_p_files, print_all_stats=print_all_stats, print_blink_stats=print_blink_stats, print_debug=self.print_debug)
                # Add group dataset to internal list of datasets
                if self.onlyTorch and not (aux_group.group_features_csv_loader is None):
                    self.groups_torch_data.append(aux_group.group_features_csv_loader.torch_dataset)
                self.groups.append(aux_group)
                num_files +=1                            

        # update read flag if not async load 
        if not self.use_async:
            self.all_groups_read = True       
        
        return num_files

    def get_concat_groups_torch_dataset(self) -> torch.utils.data.ConcatDataset[TorchGroupDataset]:
        if not self.onlyTorch:
            raise Exception("Can't return all torch datasets concatenated because this group manager wasn't created with onlyTorch set to true")        
        return torch.utils.data.ConcatDataset(self.groups_torch_data)
    
    def get_vilearn_torch_dataloader(self, groups_dataset: TorchGroupDataset):
        return TorchGroupDataLoader(groups_dataset)