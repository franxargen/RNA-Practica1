import csv
import random
import numpy as np

# Funcion de normalizacion. Se le pasa una columna de datos y los normaliza        
def normalize(list):
    maxValue = max(list)
    minValue = min(list)
    for i in range(len(list)):
        list[i] = (list[i]-minValue)/(maxValue-minValue)
    return list

# Funcion de escritura en archivo
def writeDataInFile(fileName, data):
    '''
    with open(fileName, "w") as file:
        for line in data:
            file.write(str(line).strip("[").strip("]"))
    '''
    with open(fileName, 'w') as file:
        for line in data:
            writer = csv.writer(file)
            writer.writerow(line)
    file.close()


#Array de entradas
dataArray = []

with open("ConcreteData.csv", "r") as dataFile:
    csvData = csv.reader(dataFile, quoting=csv.QUOTE_NONNUMERIC)

    # Obtencion de los datos del archivo
    for line in csvData:
        dataArray.append(line)
    
    dataArray = np.asarray(dataArray)

    # Aleatorizacion
    #random.shuffle(dataArray)

""" writeDataInFile("ObtainedData.csv",dataArray) """
''' Normalizacion y sobreescritura de los elementos de la lista de datos '''
# Iteramos sobre las columnas de los datos
for col in range(dataArray.shape[1]):
    # Guardamos cada una de las columnas normalizadas
    colNomarlized = normalize(dataArray[:,col])  
    
    # Sobreescribimos los nuevos datos
    for line in range(len(dataArray)):
        dataArray[line][col] = colNomarlized[line]
print(dataArray)
training = dataArray[:int(len(dataArray)*0.7)]
validation = dataArray[int(len(dataArray)*0.7):int(len(dataArray)*0.85)]
testing = dataArray[int(len(dataArray)*0.85):]

""" writeDataInFile("training",training)
writeDataInFile("validation",validation)
writeDataInFile("testing",testing)
writeDataInFile("NormalizedData.csv",dataArray) """

def calculateMeanCuadraticError(obtainedWeights, dataSet):
    expectedOutput = getExpectedOutput(dataSet)
    result = 0
    for i in range(len(dataSet)):
        obtainedOutput = calculateOutput(dataSet[i, :9], obtainedWeights)
        result += (obtainedOutput - expectedOutput[i]) ** 2
    result = result / len(dataSet)
    return result

def getExpectedOutput(data):
    # TODO: Probar -1 en vez de 8
    output = [fila[8] for fila in data]
    return np.asarray(output)

def getExpectedOutputValue(data):
    output = data[-1]
    return output

def calculateOutput(input, wheights):
    resultArray = np.multiply(input,wheights)
    resultValue = np.sum(resultArray)
    return resultValue
    
    

def run(input, weights = [] ,maxCycles = 1 ,learningRate = 0.05):
    
    # Inicialización de pesos aleatorios y umbral
    if len(weights) == 0:
        weights = np.random.rand(input.shape[1])
    print("Initial Weights : \n", weights)


    # Anadimos una columna mas con 1's para poder multiplicar el umbral
    input = np.insert(input, input.shape[1] - 1, np.ones(len(input)), axis = 1)
    print("Initial Inputs : \n", input)
    
    # Bucle de ciclos
    for cycle in range(maxCycles):
        
        # Bucle de patrones. Recorremos hasta la penultima columna
        for inputLine in input:
            # TODO : Presentar entrada y calcular salida (1)
            obtainedOutput = calculateOutput(inputLine[:-1],weights)
            
            # TODO : Ajustar pesos y umbral (2)
            # TODO : inputLine solo llega hasta la penultima columna, necesitamos el ultimo valor
            expectedOutput = inputLine[-1]
            #print("Expected output in data line ", inputLine, " \n =", expectedOutput)
            
            deltaValue = learningRate * (obtainedOutput - expectedOutput)
            deltaWeights = deltaValue * inputLine[:-1]
            #print("Delta weights are: \n", deltaWeights)
            weights = np.add(weights, deltaWeights)
            print("New weights are : \n", weights)
        
        # TODO: Hay que normalizar los conjuntos de validacion y test
        trainingError = calculateMeanCuadraticError(weights, training)
        #print("Training Error = ", trainingError)
        validationError = calculateMeanCuadraticError(weights, validation)
        #print("Validation Error = ", validationError)

    return weights

finalWeightsModel = run(training)
print("Final weights are : \n", finalWeightsModel)

testingError = calculateMeanCuadraticError(finalWeightsModel, testing)
print("Testing error = ", testingError)