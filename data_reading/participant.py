from vilearn_csv_data_loader import *

class Participant:
    """
    An instance of a participant, with its movement and audio data 
    """
    csv_file_path = ""
    audio_file_path = ""
    movement_data: ViLearnCSVDataLoader
    audio_data = ""

    def __init__(self, csv_path: str, audio_path: str):
        self.csv_file_path = csv_path
        self.audio_file_path = audio_path
        self.movement_data = ViLearnCSVDataLoader(self.csv_file_path)
        self.audio_data = ""
        # debug loaded data
        self.movement_data.get_eye_tracking_data()