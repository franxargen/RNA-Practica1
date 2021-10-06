import csv
import random
import numpy as np
import matplotlib.pyplot as plt


max = []
min = []

# Writes our data in a specific file
# fileName : The string name of the file
# data : The data we want to write
def writeDataInFile(fileName, data):

    with open(fileName, 'w') as file:
        for line in data:
            writer = csv.writer(file)
            writer.writerow(line)
    file.close()

# Gets data from file and saves it in a numpy array
# fileName : The string name of the file we want to get the data of
def getData(fileName):
    data = []
    with open(fileName, "r") as dataFile:
        csvData = csv.reader(dataFile, quoting=csv.QUOTE_NONNUMERIC)
        print("Getting data from", fileName, "...")
        for line in csvData:
            if len(line) != 0:
                data.append(line)
        dataFile.close()
    dataArray_np = np.asarray(data)
    return dataArray_np

"""
# Save max and min values from data file
for col in range(dataArray.shape[1]):
    max.append(max(col))
    min.append(min(col))

def denormalize(dataArray, max, min)
    for col in range(dataArray.shape[i]):
        list[i] = dataArray[i]*(max[i]-min[i]) + min[i]
"""


# Obtains data from file and applies randomation and normalization. Also, it adds a column of full 1's at the end
# fileName : The string name of the file we want to process
def processData(fileName):

    dataArray = getData(fileName)

    print("Processing data...")
    # Normalize
    dataArray_np = (dataArray - dataArray.min(0)) / dataArray.ptp(0)

    # Randomize
    random.shuffle(dataArray)

    # Column of full 1's
    dataArray_np = np.insert(dataArray_np, dataArray_np.shape[1] - 1, np.ones(len(dataArray_np)), axis=1)
    return dataArray_np

'''These two lines process the data. They are commented because we want to proccess the data once'''
#dataArray = processData("ConcreteData.csv")
#writeDataInFile("processedData.csv", dataArray)

processedData = getData("processedData.csv")

# Split of the data
training = processedData[:int(len(processedData)*0.7)]
validation = processedData[int(len(processedData)*0.7):int(len(processedData)*0.85)]
testing = processedData[int(len(processedData)*0.85):]

# We write them on file to check
writeDataInFile("training", training)
writeDataInFile("validation", validation)
writeDataInFile("testing", testing)

# Calculates the mean 
def calculateMeanSquareError(obtainedWeights, dataSet):
    expectedOutput = getExpectedOutput(dataSet)
    result = 0
    numColumns = dataSet.shape[1]
    for i in range(len(dataSet)):
        obtainedOutput = calculateOutput(dataSet[i, :(numColumns - 1)], obtainedWeights)
        result += (expectedOutput[i] - obtainedOutput) ** 2
    result = result / len(dataSet)
    return result

def getExpectedOutput(data):
    return np.array(data[:, -1])

def calculateOutput(input, wheights):
    resultArray = np.multiply(input, wheights)
    return np.sum(resultArray)

def denormalize(list, max, min):
    
    for i in range(len(list)):
        list[i] = (list[i]-minValue)/(maxValue-minValue)
        (maxValue-minValue)*list[i] = (list[i]-minValue)
    return list

def run(input, weights=[], maxCycles=1000, learningRate=0.0005):

    trainingErrorData = []
    validationErrorData = []

    # Initialization of randow weights and umbral
    if len(weights) == 0:
        weights = np.random.rand(input.shape[1] - 1) - 0.5

    '''Uncomment to check initial values'''    
    #print("Initial Weights : \n", weights)
    #print("Initial Inputs : \n", input)

    # Cycles loop
    for cycle in range(maxCycles):

        print("CYCLE =>", cycle)

        # Data lines loop
        for inputLine in input:
            # We calculate the output using the weights
            obtainedOutput = calculateOutput(inputLine[:-1], weights)

            # We adjust the new weights
            expectedOutput = inputLine[-1]
            deltaValue = learningRate * (expectedOutput - obtainedOutput)
            deltaWeights = deltaValue * inputLine[:-1]
            weights = np.add(weights, deltaWeights)

        # And finally, calculate training and validation errors
        trainingErrorData.append(calculateMeanSquareError(weights, training))
        validationErrorData.append(calculateMeanSquareError(weights, validation))

    return weights, trainingErrorData, validationErrorData


finalWeightsModel, trainingErrorData, validationErrorData = run(training)
#print("Final weights are : \n", finalWeightsModel)

print("Training error ", trainingErrorData)
print("Validation error ", validationErrorData)

# Plot settings
plt.plot(trainingErrorData, color='red')
plt.plot(validationErrorData, color='blue')
plt.xlabel('Cycles')
plt.ylabel('Mean Square Error')
plt.ylim(min(min(trainingErrorData), min(validationErrorData)),
        max(max(trainingErrorData), max(validationErrorData)))
plt.show()

# Final testing error
testingError = calculateMeanSquareError(finalWeightsModel, testing)
print("Testing error = ", testingError)