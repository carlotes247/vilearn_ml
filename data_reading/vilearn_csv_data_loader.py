# Import the csv module
import csv
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

class ViLearnParticipantCSVLoader:
    """
    Can  read CSV files
    """
    #region Variables

    fileExists: bool
    # raw data loaded by pandas
    raw_data: pd.DataFrame
    # timestamp   
    overallTsString = []
    overallTsNtpString = []
    overallTsParsed = []
    overallTsNtpParsed = []
    imuTS = []
    eyeTS = []
    # device status
    headsetValid = []
    leftHandValid = []
    rightHandValid= []
    isUserPresent = []
    # head
    headAcc = []
    headGyro = []
    # head transform
    headPosition = []
    headRotation = []
    # left hand transform
    leftHandPosition = []
    leftHandRotation = []
    # right hand transform
    rightHandPosition = []
    rightHandRotation = []
    # eye left
    leftEyeGaze = []
    leftPupilPosition = []
    leftPupilDilation = []
    leftEyeOpeness = []
    # eye right
    rightEyeGaze = []
    rightPupilPosition = []
    rightPupilDilation = []
    rightEyeOpeness = []
    # combined gaze
    combinedGaze = []
    # features
    leftHandVelocity = []
    leftHandAngularVelocity = []
    rightHandVelocity = []
    rightHandAngularVelocity = []
    
    #endregion

    #region Constructor
    def __init__(self, path: str):
        """
        (Uses Pandas module) Loads a Vilearn participant dataset to memory
        """
        try:                         
            self.raw_data = pd.read_csv(path, decimal=',', sep=';')
            self.fileExists = True
        except FileNotFoundError:
            self.fileExists = False
            print(f"Error: File nout found when loading participant file {path}!")
        if self.fileExists:
            # timestamp   
            self.overallTsString = self.raw_data['overallTS']
            self.overallTsNtpString = self.raw_data['overallTsNTP']
            self.overallTsParsed = []
            self.overallTsNtpParsed = []
            self.imuTS = self.raw_data['imuTS']
            self.eyeTS = self.raw_data['eyeTS']
            # device status
            self.headsetValid = self.raw_data['headsetValid']
            self.leftHandValid = self.raw_data['leftHandValid']
            self.rightHandValid = self.raw_data['rightHandValid']
            self.isUserPresent = self.raw_data['isUserPresent']
            # head
            self.headAcc = self.raw_data[["headAccX", "headAccY", "headAccZ"]]
            self.headGyro = self.raw_data[["headGyroX", "headGyroY", "headGyroZ"]]
            # head transform
            self.headPosition = self.raw_data[["headPositionX", "headPositionY", "headPositionZ"]]   
            self.headRotation =  self.raw_data[["headRotationX", "headRotationY", "headRotationZ", "headRotationW"]]
            # left hand transform
            self.leftHandPosition = self.raw_data[["leftHandPositionX", "leftHandPositionY", "leftHandPositionZ"]]
            self.leftHandRotation = self.raw_data[["leftHandRotationX", "leftHandRotationY", "leftHandRotationZ", "leftHandRotationW"]]
            # right hand transform
            self.rightHandPosition = self.raw_data[["rightHandPositionX", "rightHandPositionY", "rightHandPositionZ"]]
            self.rightHandRotation = self.raw_data[["rightHandRotationX", "rightHandRotationY", "rightHandRotationZ", "rightHandRotationW"]]
            # eye left
            self.leftEyeGaze = self.raw_data[["leftEyeGazeX", "leftEyeGazeY", "leftEyeGazeZ", "leftEyeGazeConfidence"]]
            self.leftPupilPosition = self.raw_data[["leftEyePupilPosition_PositionX", "leftEyePupilPosition_PositionY", "leftEyePupilPosition_Position_Confidence"]]
            self.leftPupilDilation = self.raw_data[["leftEyePupilDilation", "leftEyePupilDilationConfidence"]]
            self.leftEyeOpeness = self.raw_data[["leftEyeOpeness", "leftEyeOpenessConfidence"]]
            # eye right
            self.rightEyeGaze = self.raw_data[["rightEyeGazeX", "rightEyeGazeY", "rightEyeGazeZ", "rightEyeGazeConfidence"]]
            self.rightPupilPosition = self.raw_data[["rightEyePupilPosition_PositionX", "rightEyePupilPosition_PositionY", "rightEyePupilPosition_Position_Confidence"]]
            self.rightPupilDilation = self.raw_data[["rightEyePupilDilation", "rightEyePupilDilationConfidence"]]
            self.rightEyeOpeness = self.raw_data[["rightEyeOpeness", "rightEyeOpenessConfidence"]]
            # combined gaze
            self.combinedGaze = self.raw_data[["combinedGazeX", "combinedGazeY", "combinedGazeZ"]]
            # features
            self.leftHandVelocity = self.raw_data[["leftHandVelocityX", "leftHandVelocityY", "leftHandVelocityZ"]]
            self.leftHandAngularVelocity = self.raw_data[["leftHandAngularVelocityX", "leftHandAngularVelocityY", "leftHandAngularVelocityZ"]]
            self.rightHandVelocity = self.raw_data[["rightHandVelocityX", "rightHandVelocityY", "rightHandVelocityZ"]]
            self.rightHandAngularVelocity = self.raw_data[["rightHandAngularVelocityX", "rightHandAngularVelocityY", "rightHandAngularVelocityZ"]]
    #endregion
    
    #region Methods
    def get_deltas_between_timestamps(self, path):
        """
        (Uses CSV module) Returns a list of deltas between the timestamps
        of a csv file where the first column are the timestamps
        """
        with open(path, "r") as file:
            # Create a csv reader object
            reader = csv.reader(file)
            # Skip the header row
            next(reader)

            previousMiliseconds = 0
            currentMiliseconds = 0
            listOfDeltas = []
            listOfTimeStamps = []
            timestamptRepeated = False
            firstInit = True
            # Loop through each row in the file
            for row in reader:
                # Get the first column as timestamp
                timestamp = row[0]
                # Parse the timestamp string into a datetime object
                dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S.%f")
                if (not firstInit):
                    prevTimestamp = listOfTimeStamps[-1]
                    timestamptRepeated = timestamp == prevTimestamp
                    #print("dtDiff is: " + str(timestamptRepeated))

                # Avoid duplicate timestamps
                if (firstInit or not timestamptRepeated):
                    # Print the timestamp
                    #print(timestamp)
                    # Add timestamp into list
                    listOfTimeStamps.append(timestamp)
                    # Get the microseconds part
                    microseconds = dt.microsecond
                    # Convert microseconds to milliseconds
                    milliseconds = microseconds // 1000
                    # Print the milliseconds delta
                    currentMiliseconds = milliseconds
                    if(not firstInit):
                        delta = abs(currentMiliseconds - previousMiliseconds)
                        #print(delta)
                        if (delta < 0):
                            print("This delta is negative. What's going on?")
                        listOfDeltas.append(delta)
                    previousMiliseconds = currentMiliseconds
                    firstInit = False

        return listOfDeltas

    def print_eye_tracking_data(self):
        """
        (Uses Pandas module) Returns a list of eye tracking
        of a csv file where the first column are the timestamps
        """
        if self.fileExists:
            # IMU data
            print("--------------------- IMU -----------------------------------")
            # timestamp
            #data = pd.read_csv(path, decimal=',', sep=';')
            print(f"Percentage of NaN values in IMU TimeStamp: {self.imuTS.isna().mean()*100}%")
            print(f"Percentage of NaN values in EYE TimeStamp: {self.eyeTS.isna().mean()*100}%")
            print("-------------------------------------------------------------")
            print(f"Percentage of False values in Head isValid: {self.headsetValid.isin([False]).mean()*100}%")
            print(f"Percentage of False values in leftHandValid: {self.leftHandValid.isin([False]).mean()*100}%")
            print(f"Percentage of False values in rightHandValid: {self.rightHandValid.isin([False]).mean()*100}%")
            print(f"Percentage of False values in isUserPresent: {self.isUserPresent.isin([False]).mean()*100}%")
            print("-------------------------------------------------------------")
            # headAcc
            # headAcc = data[["headAccX", "headAccY", "headAccZ"]]
            print(f"Percentage of -Infinity values in:\n{self.headAcc.isin([-np.inf]).mean()*100}")
            print("-------------------------------------------------------------")
            # headGyro
            # headGyro = data[["headGyroX", "headGyroY", "headGyroZ"]]
            print(f"Percentage of -Infinity values in:\n{self.headGyro.isin([-np.inf]).mean()*100}")
            print("-------------------------------------------------------------")

            # leftEye
            print("----------------- LEFT EYE ----------------------------------")
            # Gaze
            # leftEyeGaze = data[["leftEyeGazeX", "leftEyeGazeY", "leftEyeGazeZ", "leftEyeGazeConfidence"]]
            print(f"Percentage of NaN values in:\n{self.leftEyeGaze.isna().mean()*100}")
            print("-------------------------------------------------------------")
            # PupilPosition_Position
            # leftPupilPosition = data[["leftEyePupilPosition_PositionX", "leftEyePupilPosition_PositionY", "leftEyePupilPosition_Position_Confidence"]]
            print(f"Percentage of NaN values in:\n{self.leftPupilPosition.isna().mean()*100}")
            print(f"Percentage of 0 Confidence entries on Left Pupil Position: {(self.leftPupilPosition['leftEyePupilPosition_Position_Confidence'] == 0).mean()*100}%")
            print("-------------------------------------------------------------")
            # PupilDilation
            # leftPupilDilation = data[["leftEyePupilDilation", "leftEyePupilDilationConfidence"]]
            print(f"Percentage of NaN values in:\n{self.leftPupilDilation.isna().mean()*100}")
            print(f"Percentage of 0 Confidence entries on Left Pupil Dilation: {(self.leftPupilDilation['leftEyePupilDilationConfidence'] == 0).mean()*100}%")
            print("-------------------------------------------------------------")
            # EyeOpeness
            # leftEyeOpeness = data[["leftEyeOpeness", "leftEyeOpenessConfidence"]]
            print(f"Percentage of NaN values in:\n{self.leftEyeOpeness.isna().mean()*100}")
            print(f"Percentage of 0 Confidence entries on Left Eye Openess: {(self.leftEyeOpeness['leftEyeOpenessConfidence'] == 0).mean()*100}%")
            print("-------------------------------------------------------------")

            # rightEye
            print("----------------- RIGHT EYE ---------------------------------")
            # Gaze
            # rightEyeGaze = data[["rightEyeGazeX", "rightEyeGazeY", "rightEyeGazeZ", "rightEyeGazeConfidence"]]
            print(f"Percentage of NaN values in:\n{self.rightEyeGaze.isna().mean()*100}")
            print("-------------------------------------------------------------")
            # PupilPosition_Position
            # rightPupilPosition = data[["rightEyePupilPosition_PositionX", "rightEyePupilPosition_PositionY", "rightEyePupilPosition_Position_Confidence"]]
            print(f"Percentage of NaN values in:\n{self.rightPupilPosition.isna().mean()*100}")
            print(f"Percentage of 0 Confidence entries on right Pupil Position: {(self.rightPupilPosition['rightEyePupilPosition_Position_Confidence'] == 0).mean()*100}%")
            print("-------------------------------------------------------------")
            # PupilDilation
            # rightPupilDilation = data[["rightEyePupilDilation", "rightEyePupilDilationConfidence"]]
            print(f"Percentage of NaN values in:\n{self.rightPupilDilation.isna().mean()*100}")
            print(f"Percentage of 0 Confidence entries on right Pupil Dilation: {(self.rightPupilDilation['rightEyePupilDilationConfidence'] == 0).mean()*100}%")
            print("-------------------------------------------------------------")
            # EyeOpeness
            # rightEyeOpeness = data[["rightEyeOpeness", "rightEyeOpenessConfidence"]]
            print(f"Percentage of NaN values in:\n{self.rightEyeOpeness.isna().mean()*100}")
            print(f"Percentage of 0 Confidence entries on right Eye Openess: {(self.rightEyeOpeness['rightEyeOpenessConfidence'] == 0).mean()*100}%")
            print("-------------------------------------------------------------")

            # combinedGaze
            print("----------------- COMBINED GAZE -----------------------------")
            # combinedGaze = data[["combinedGazeX", "combinedGazeY", "combinedGazeZ"]]
            print(f"Percentage of NaN values in:\n{self.combinedGaze.isna().mean()*100}")
            print("-------------------------------------------------------------")

            # leftHand
            print("----------------- LEFT HAND ---------------------------------")
            # HandVelocity
            # leftHandVelocity = data[["leftHandVelocityX", "leftHandVelocityY", "leftHandVelocityZ"]]
            print(f"Percentage of -Infinity values in:\n{self.leftHandVelocity.isin([-np.inf]).mean()*100}")
            print("-------------------------------------------------------------")
            # HandAngularVelocity
            # leftHandAngularVelocity = data[["leftHandAngularVelocityX", "leftHandAngularVelocityY", "leftHandAngularVelocityZ"]]
            print(f"Percentage of -Infinity values in:\n{self.leftHandAngularVelocity.isin([-np.inf]).mean()*100}")
            print("-------------------------------------------------------------")

            # rightHand
            print("----------------- RIGHT HAND --------------------------------")
            # HandVelocity
            # rightHandVelocity = data[["rightHandVelocityX", "rightHandVelocityY", "rightHandVelocityZ"]]
            print(f"Percentage of -Infinity values in:\n{self.rightHandVelocity.isin([-np.inf]).mean()*100}")
            print("-------------------------------------------------------------")
            # HandAngularVelocity
            # rightHandAngularVelocity = data[["rightHandAngularVelocityX", "rightHandAngularVelocityY", "rightHandAngularVelocityZ"]]
            print(f"Percentage of -Infinity values in:\n{self.rightHandAngularVelocity.isin([-np.inf]).mean()*100}")
            print("-------------------------------------------------------------")

            listOfEyeTracking = []
            firstInit = True

            return listOfEyeTracking
        else:
            print("Can't print eye tracking data from this file because it doesn't exist.")
    
    def print_blink_stats(self):
        return
    #endregion

if __name__ == '__main__':
    data_file_path='../data/DESKTOP-QTU96C2_vilearn2023-12-19__12-26-14.586.csv'
    datafilepath_mun06NovPC5 = '../data/DESKTOP-QTU96C2_vilearn2023-11-06__16-09-26.660.csv'
    # Get all the data paths to test from the txt. This way we don't need to rewrite paths in this class when testing
    file_with_paths = 'data_paths.txt'
    data_files_to_test = []
    with open(file_with_paths, "r") as file:
        data_files_to_test = file.read().splitlines()

    # Load a file and print if there any problems with eye data
    data_loader = ViLearnParticipantCSVLoader(data_files_to_test[0])
    data_loader.print_eye_tracking_data()
    #eyeTracking = reader.getEyeTrackingData(datafilepath_mun06NovPC5)
    #print(eyeTracking)