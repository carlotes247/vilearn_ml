class DirectGazeFeature:
    direct_gaze_flag: bool
    participant_name: str
    target: str

    def __init__(self, direct_gaze_value: bool, participant_name: str, target: str):
        self.direct_gaze_flag = direct_gaze_value
        self.participant_name = participant_name
        self.target = target        