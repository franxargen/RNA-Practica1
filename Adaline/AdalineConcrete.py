import csv
import random
import numpy as np
import matplotlib.pyplot as plt

# Writes our data in a specific file
# fileName : The string name of the file
# data : The data we want to write
def writeDataInFile(fileName, data):
    with open('./Adaline/' + fileName, 'w') as file:
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

# Function that reverts the normalize filter
def denormalizeOutput(output):
    for i in range(len(output)):
        output[i] = output[i] * (maxValueOutput - minValueOutput) + minValueOutput
    return output

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

''' These two lines process the data and write it on file.
They are commented because we want to proccess the data once '''
#dataArray = processData("ConcreteData.csv")
#writeDataInFile("processedData.csv", dataArray)

# We obtain the raw data from file and print it for checking
rawData = getData("./Adaline/ConcreteData.csv")

# Save max and min values from data file. This will be useful to de-normalization
maxValueOutput = max(rawData[:,-1])
minValueOutput = min(rawData[:,-1])

# We obtain the processed data (randomized and normalized)
processedData = getData("./Adaline/processedData.csv")

# Split of the data
training = processedData[:int(len(processedData)*0.7)]
validation = processedData[int(len(processedData)*0.7):int(len(processedData)*0.85)]
testing = processedData[int(len(processedData)*0.85):]

'''Uncomment to check data in files'''
writeDataInFile("training.csv", training)
writeDataInFile("validation.csv", validation)
writeDataInFile("testing.csv", testing)

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

# This function obtains the output column of the file
def getExpectedOutput(data):
    return np.array(data[:, -1])

# This caculates the linear combination to obtain outputs
def calculateOutput(input, wheights):
    resultArray = np.multiply(input, wheights)
    return np.sum(resultArray)

''' ADALINE ALGORITHM '''
def run(input=[], weights=[], maxCycles=1000, learningRate=0.005):

    trainingErrorData = []
    validationErrorData = []

    # Initialization of random weights and umbral
    if len(weights) == 0:
        weights = np.random.rand(input.shape[1] - 1) - 0.5

    '''Uncomment to check initial values'''    
    #print("Initial Weights : \n", weights)
    #print("Initial Inputs : \n", input)

    # Cycles loop
    print("Generating model...")
    for cycle in range(maxCycles):

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

# We initialize the random weights
weights = np.random.rand(training.shape[1] - 1) - 0.5
maxCycles = 10000
learningRate = 0.05
# We call the algorithm and obtain the model and the errors
finalWeightsModel, trainingErrorData, validationErrorData = run(training, weights=weights, maxCycles=maxCycles, learningRate=learningRate)

'''Uncomment to see final results'''
#print("Final weights are : \n", finalWeightsModel)
#print("Training error ", trainingErrorData)
#print("Validation error ", validationErrorData)

# Calculation of the new optimal cycles
newCycles = validationErrorData.index(min(validationErrorData)) + 1
print("New optimal cycles =", newCycles)

# We call the algorithm and obtain the results with the new cycles
finalWeightsModel, trainingErrorData, validationErrorData = run(training, weights=weights, maxCycles=newCycles, learningRate=learningRate)

with open('finalWeightsModel.csv', 'w') as file:
    np.savetxt(file, finalWeightsModel)

# Plot settings for error
plt.plot(trainingErrorData, color='red', label = 'Training error')
plt.plot(validationErrorData, color='blue', label = 'Validation error')
plt.xlabel('Cycles')
plt.ylabel('Mean Square Error')
# This line adjust the plot's scale
plt.ylim(min(min(trainingErrorData), min(validationErrorData)),
        max(max(trainingErrorData), max(validationErrorData)))
plt.legend(loc="lower right")
plt.show()

# Final testing error
testingError = calculateMeanSquareError(finalWeightsModel, testing)
print("Testing error = ", testingError)

''' Now, we have to generate our outputs with the final model
Once we have our outputs, we have to de-normalize them'''

# Get final obtained outputs
finalObtainedOutputs_norm = []
for line in testing:
    finalObtainedOutputs_norm.append(calculateOutput(line[:-1],finalWeightsModel))
#print("OBTAINED OUTPUTS=", finalObtainedOutputs_norm)

# De-normalize the outputs and sorting
finalObtainedOutputs_denorm = denormalizeOutput(np.asarray(finalObtainedOutputs_norm))

# Get the expected outputs and sorting
expectedOutputs_norm = getExpectedOutput(testing)
expectedOutputs_denorm = denormalizeOutput(np.asarray(expectedOutputs_norm))

comparisonErrors = np.vstack((finalObtainedOutputs_denorm, expectedOutputs_denorm)).T
comparisonErrors = comparisonErrors[comparisonErrors[:,1].argsort()]

with open('obtainedExpectedOutputs.csv', 'w') as file:
    np.savetxt(file, comparisonErrors)

# Plot settings for testing
plt.plot(comparisonErrors[:,0], color='red', label='Obtained outputs')
plt.plot(comparisonErrors[:,1], color='blue', label = 'Expected outputs')
plt.xlabel('Test pattern')
plt.ylabel('Concrete Compressive Strength (MPa)')
plt.legend(loc="lower right")
plt.show()

'''Data for MLP'''
# We delete columns of full 1's of all data sets
training = np.delete(training, training.shape[1] - 2, axis=1)
validation = np.delete(validation, validation.shape[1] - 2, axis=1)
testing = np.delete(testing, testing.shape[1] - 2, axis=1)
# And we write them in file
#writeDataInFile("training.csv", training)
#writeDataInFile("validation.csv", validation)
#writeDataInFile("testing.csv", testing)

