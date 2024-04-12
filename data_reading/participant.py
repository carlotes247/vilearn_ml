import data_reading.vilearn_csv_data_loader as vilearn_csv_data_loader

class participant:
    """
    An instance of a participant, with its movement and audio data 
    """
    csv_file_path = ""
    movement_data = vilearn_csv_data_loader(csv_file_path)
    audio_data = ""