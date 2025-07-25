from engagement_processor import EngagementProcessor
import os

class EngagementsManager:

    engagements_list: list[EngagementProcessor]
    data_path: str = "data/annotations"
    path_groups_info: str = "data/group_durations_all_commas.csv"
    filename_1: str = "group.task engagement.helenrisack.annotation~"
    filename_2: str = "task engagement.group.carlosgonzalez.annotation~"
    folders: list[str] 

    def __init__(self, save_to_disk: bool) -> None:
        self.load_engagements(save_to_disk=save_to_disk)
        pass

    def load_engagements(self, save_to_disk: bool):
        self.folders = os.listdir(self.data_path)
        for folder in self.folders:
            file_1_path: str = f"{self.data_path}/{folder}/{self.filename_1}"
            file_2_path: str = f"{self.data_path}/{folder}/{self.filename_2}"
            eng_processor = EngagementProcessor(path_groups_info=self.path_groups_info, path_file_1=file_1_path, path_file_2=file_2_path, group_name=folder.replace("recording", ""))
            eng_processor.process_task_engagement(save_to_disk=save_to_disk)
        pass

if __name__ == "__main__":

    print("done")