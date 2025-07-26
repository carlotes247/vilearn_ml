from engagement_processor import EngagementProcessor
import os

class EngagementsManager:

    engagements_list: list[EngagementProcessor] = []
    data_path: str = "data/annotations"
    path_groups_info: str = "data/group_durations_all_commas.csv"
    folders: list[str]

    def __init__(self, save_to_disk: bool) -> None:
        self.load_engagements(save_to_disk=save_to_disk)
        pass

    def load_engagements(self, save_to_disk: bool):
        self.folders = os.listdir(self.data_path)
        for folder in self.folders:
            folder_path: str = f"{self.data_path}/{folder}"
            eng_processor = EngagementProcessor(path_groups_info=self.path_groups_info, path_folder=folder_path, group_name=folder.replace("recording_", ""))
            eng_processor.process_task_engagement(save_to_disk=save_to_disk)
            self.engagements_list.append(eng_processor)
        pass

if __name__ == "__main__":
    mngr_aux: EngagementsManager = EngagementsManager(True)
    print("done")