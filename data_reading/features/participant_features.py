from features.gaze_behaviour_feature import GazeBehaviourFeature

class ParticipantFeatures:
    participant_name: str
    gaze_behaviours: GazeBehaviourFeature

    def __init__(self, name: str, gaze_behaviours: GazeBehaviourFeature):
        self.participant_name = name
        self.gaze_behaviours = gaze_behaviours