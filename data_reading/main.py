from csv_reader import csv_reader
import numpy as np

def calculateStats(list, stringTag, stringMeasurement):
    """
    (Uses Numpy) Prints stats about a csv file
    """ 
    # Calculate the average of the list
    # Calculate the sum of the list
    total = sum(list)
    # Calculate the length of the list
    count = len(list)
    # Calculate the average of the list
    average = total / count
    # Print the average
    print("--------------- " + stringTag + "STATS -----------------")
    print(stringTag + "List Entries: " + str(count))
    print(stringTag + "List Avg: " + str(average) + stringMeasurement)
    deltaArray = np.array(list)
    print(stringTag + "Deltas Mean: " + str(deltaArray.mean()) + stringMeasurement)
    print(stringTag + "Deltas Min: " + str(deltaArray.min()) + stringMeasurement)
    print(stringTag + "Deltas Max: " + str(deltaArray.max()) + stringMeasurement)
    print(stringTag + "Deltas Median: " + str(np.median(deltaArray)) + stringMeasurement)
    print(stringTag + "Percentage of deltas above average: " + str(np.count_nonzero(deltaArray > average) / count * 100) + " %")
    print(stringTag + "Percentage of deltas above 500ms: " + str(np.count_nonzero(deltaArray > 500) / count * 100) + " %")
    print(stringTag + "Percentage of deltas above 900ms: " + str(np.count_nonzero(deltaArray > 900) / count * 100) + " %")

def readDataAndCalculateStats (reader, fileName, stringTag):
    """
    Calls all functions written so far to understand the data
    """ 
    listOfDeltas = reader.getDeltasBetweenTimestamps(fileName)
    calculateStats(listOfDeltas, stringTag, "ms")
    reader.getEyeTrackingData(fileName)

# create csv_reader obj 
reader = csv_reader()
# Read deltas between timestampts and calculate stats. Pass it a path to a local csv file. Try not to push the csv files not to clutter the repo
#readDataAndCalculateStats(reader, "HCM153.csv", "(Subscription to Eye Event) ")
#readDataAndCalculateStats(reader, "goodUserName.csv", "(Fixed Update) ")
#readDataAndCalculateStats(reader, "HCM153_Administrator2023-09-26__09-39-37.103.csv", "(2023-09-26__09-39-37.103) ")
#readDataAndCalculateStats(reader, "HCM153_Administrator2023-09-26__09-49-31.770.csv", "(023-09-26__09-49-31.770) ")

# Files from 05 October 2023 long test
#readDataAndCalculateStats(reader, "LAPTOP-UJR5JBB1_carlo2023-10-05__15-53-28.036.csv", "(Carlos) ") # this one is massive (200MB) avoid pushing
#readDataAndCalculateStats(reader, "Thomas_HCM153_Administrator2023-10-05__16-13-46.496.csv", "(Thomas) ")
readDataAndCalculateStats(reader, "DESKTOP-QTU96C2_vilearn2023-10-22__10-33-39.028.csv", "(Laura) ")

# added this comment to check if git hooks work