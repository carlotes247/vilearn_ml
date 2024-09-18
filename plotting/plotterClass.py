from torch_vilearn.torch_group_dataset import TorchGroupDataset
from data_reading.group import Group
import matplotlib.pyplot as plt

class PlotterClass:
    """
    [CURRENTLY NOT WORKING YET] It seems that matplotlib expects to be called from the main script, and calling it 
    from within a class instance method would require to return a subplot and plot it outside. It doesn't seem worth
    to explore this option much as we can write some dirty code... (I don't like it but we need to move on)
    Can plot vilearn data using matplotlib
    """

    def __init__(self) -> None:
        pass

    def plot_eye_blinks(self, groupData: Group):
        if (groupData is not None and groupData.group_features_csv_loader is not None):
            print("preparing to plot...")
            timestamps = groupData.group_features_csv_loader.raw_data['TSGroupNTP']
            left_eye_openess_p1 = groupData.group_features_csv_loader.raw_data['LeftEyeOpennesP1']
            left_eye_openess_confidence_p1 = groupData.group_features_csv_loader.raw_data['LeftEyeOpennesConfidenceP1']
            right_eye_openess_p1 = groupData.group_features_csv_loader.raw_data['RightEyeOpennesP1']
            right_eye_openess_confidence_p1 = groupData.group_features_csv_loader.raw_data['RightEyeOpennesConfidenceP1']
            print("open plot window")
            plt.plot(left_eye_openess_p1, timestamps)
            plt.show()
        else:
            print("can't plot group because it is null!")
    