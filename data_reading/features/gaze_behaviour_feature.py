from features.direct_gaze_feature import DirectGazeFeature
from features.blink_feature import BlinkFeature

class GazeBehaviourFeature:
    direct_gaze: DirectGazeFeature
    blink: BlinkFeature

    def __init__(self, direct_gaze_data: DirectGazeFeature, blink_data: BlinkFeature):
        self.direct_gaze = DirectGazeFeature(direct_gaze_data.direct_gaze_flag, direct_gaze_data.participant_name, direct_gaze_data.target)
        self.blink = BlinkFeature(blink_data.blink_flag)
