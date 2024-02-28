# Import the csv module
import csv
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

class csv_reader:
    """
    Can  read CSV files
    """
    def getDeltasBetweenTimestamps(self, path):
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

    def getEyeTrackingData(self, path):
        """
        (Uses Pandas module) Returns a list of eye tracking
        of a csv file where the first column are the timestamps
        """ 
        # IMU data
        print("--------------------- IMU -----------------------------------")
        # timestamp      
        data = pd.read_csv(path)
        print(f"Percentage of NaN values in IMU TimeStamp: {data['imuTS'].isna().mean()*100}%")
        print(f"Percentage of NaN values in EYE TimeStamp: {data['eyeTS'].isna().mean()*100}%")
        print("-------------------------------------------------------------")
        print(f"Percentage of False values in Head isValid: {data['headsetValid'].isin([False]).mean()*100}%")
        print(f"Percentage of False values in leftHandValid: {data['leftHandValid'].isin([False]).mean()*100}%")
        print(f"Percentage of False values in rightHandValid: {data['rightHandValid'].isin([False]).mean()*100}%")
        print(f"Percentage of False values in isUserPresent: {data['isUserPresent'].isin([False]).mean()*100}%")        
        print("-------------------------------------------------------------")
        # headAcc
        headAcc = data[["headAccX", "headAccY", "headAccZ"]]
        print(f"Percentage of -Infinity values in:\n{headAcc.isin([-np.inf]).mean()*100}")        
        print("-------------------------------------------------------------")
        # headGyro
        headGyro = data[["headGyroX", "headGyroY", "headGyroZ"]]
        print(f"Percentage of -Infinity values in:\n{headGyro.isin([-np.inf]).mean()*100}")        
        print("-------------------------------------------------------------")
        
        # leftEye
        print("----------------- LEFT EYE ----------------------------------")
        # Gaze
        leftEyeGaze = data[["leftEyeGazeX", "leftEyeGazeY", "leftEyeGazeZ", "leftEyeGazeConfidence"]]
        print(f"Percentage of NaN values in:\n{leftEyeGaze.isna().mean()*100}")        
        print("-------------------------------------------------------------")
        # PupilPosition_Position
        leftPupilPosition = data[["leftEyePupilPosition_PositionX", "leftEyePupilPosition_PositionY", "leftEyePupilPosition_Position_Confidence"]]
        print(f"Percentage of NaN values in:\n{leftPupilPosition.isna().mean()*100}")  
        print(f"Percentage of 0 Confidence entries on Left Pupil Position: {(data['leftEyePupilPosition_Position_Confidence'] == 0).mean()*100}%")    
        print("-------------------------------------------------------------")
        # PupilDilation
        leftPupilDilation = data[["leftEyePupilDilation", "leftEyePupilDilationConfidence"]]
        print(f"Percentage of NaN values in:\n{leftPupilDilation.isna().mean()*100}")  
        print(f"Percentage of 0 Confidence entries on Left Pupil Dilation: {(data['leftEyePupilDilationConfidence'] == 0).mean()*100}%")    
        print("-------------------------------------------------------------")
        # EyeOpeness
        leftEyeOpeness = data[["leftEyeOpeness", "leftEyeOpenessConfidence"]]
        print(f"Percentage of NaN values in:\n{leftEyeOpeness.isna().mean()*100}")  
        print(f"Percentage of 0 Confidence entries on Left Eye Openess: {(data['leftEyeOpenessConfidence'] == 0).mean()*100}%")
        print("-------------------------------------------------------------")
        
        # rightEye
        print("----------------- RIGHT EYE ---------------------------------")
        # Gaze
        rightEyeGaze = data[["rightEyeGazeX", "rightEyeGazeY", "rightEyeGazeZ", "rightEyeGazeConfidence"]]
        print(f"Percentage of NaN values in:\n{rightEyeGaze.isna().mean()*100}")        
        print("-------------------------------------------------------------")
        # PupilPosition_Position
        rightPupilPosition = data[["rightEyePupilPosition_PositionX", "rightEyePupilPosition_PositionY", "rightEyePupilPosition_Position_Confidence"]]
        print(f"Percentage of NaN values in:\n{rightPupilPosition.isna().mean()*100}")  
        print(f"Percentage of 0 Confidence entries on right Pupil Position: {(data['rightEyePupilPosition_Position_Confidence'] == 0).mean()*100}%")    
        print("-------------------------------------------------------------")
        # PupilDilation
        rightPupilDilation = data[["rightEyePupilDilation", "rightEyePupilDilationConfidence"]]
        print(f"Percentage of NaN values in:\n{rightPupilDilation.isna().mean()*100}")  
        print(f"Percentage of 0 Confidence entries on right Pupil Dilation: {(data['rightEyePupilDilationConfidence'] == 0).mean()*100}%")    
        print("-------------------------------------------------------------")
        # EyeOpeness
        rightEyeOpeness = data[["rightEyeOpeness", "rightEyeOpenessConfidence"]]
        print(f"Percentage of NaN values in:\n{rightEyeOpeness.isna().mean()*100}")  
        print(f"Percentage of 0 Confidence entries on right Eye Openess: {(data['rightEyeOpenessConfidence'] == 0).mean()*100}%")
        print("-------------------------------------------------------------")
        
        # combinedGaze
        print("----------------- COMBINED GAZE -----------------------------")
        combinedGaze = data[["combinedGazeX", "combinedGazeY", "combinedGazeZ"]]
        print(f"Percentage of NaN values in:\n{combinedGaze.isna().mean()*100}")  
        print("-------------------------------------------------------------")

        # leftHand
        print("----------------- LEFT HAND ---------------------------------")
        # HandVelocity
        leftHandVelocity = data[["leftHandVelocityX", "leftHandVelocityY", "leftHandVelocityZ"]]
        print(f"Percentage of -Infinity values in:\n{leftHandVelocity.isin([-np.inf]).mean()*100}")        
        print("-------------------------------------------------------------")
        # HandAngularVelocity
        leftHandAngularVelocity = data[["leftHandAngularVelocityX", "leftHandAngularVelocityY", "leftHandAngularVelocityZ"]]
        print(f"Percentage of -Infinity values in:\n{leftHandAngularVelocity.isin([-np.inf]).mean()*100}")        
        print("-------------------------------------------------------------")

        # rightHand
        print("----------------- RIGHT HAND --------------------------------")
        # HandVelocity
        rightHandVelocity = data[["rightHandVelocityX", "rightHandVelocityY", "rightHandVelocityZ"]]
        print(f"Percentage of -Infinity values in:\n{rightHandVelocity.isin([-np.inf]).mean()*100}")        
        print("-------------------------------------------------------------")
        # HandAngularVelocity
        rightHandAngularVelocity = data[["rightHandAngularVelocityX", "rightHandAngularVelocityY", "rightHandAngularVelocityZ"]]
        print(f"Percentage of -Infinity values in:\n{rightHandAngularVelocity.isin([-np.inf]).mean()*100}")        
        print("-------------------------------------------------------------")

        listOfEyeTracking = []
        firstInit = True

        return listOfEyeTracking
            

# reader = csv_reader()
# reader.getEyeTrackingData("Thomas_HCM153_Administrator2023-10-05__16-13-46.496.csv")