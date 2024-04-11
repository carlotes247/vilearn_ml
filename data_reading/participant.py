import csv_reader

class participant:
    """
    An instance of a participant, with its movement and audio data 
    """
    csv_file_path = ""
    movement_data = csv_reader(csv_file_path)
    audio_data = ""