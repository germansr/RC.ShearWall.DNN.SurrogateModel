"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Project: 
An Open-Source Framework for Modeling RC Shear Walls using Deep Neural Networks

File:    
MainNN.py

Date:    
28.12.2022

Developmed by:
-Ph.D. Candidate German Solorzano
Supervised by:
-Dr. Vagelis Plevris

Sponsored by:
Oslo Metropolitan University, Oslo, Norway.
Department of Civil Engineering and Energy Technology 
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# Header:
# This is the main file for the training of the DNN. 
# This script will train the DNN using the corresponding database in the form of a csv file that is specified in the variable "file1".
# The variable "file2" is the database that is used for validation after the DNN is trained.
# The final DNN is serialized (stored) in the path specified at the variable "path" at the end of the script


import DataUtils as dataUtils
import NeuralNetwork as NeuralNet
import Normalization as normalization
import tensorflow as tf
import time

# set the seed to obtain the same results every time
tf.random.set_seed(50)

# data number of inputs and outputs
nInputs = 11
nOutputs = 6

# skip rows or columns in the data (useful when the data comes with headers or numbering)
startRow = 0
startCol = 0

# read the training data file
file1 = "TrainingDataBases/database_training.csv"
data = dataUtils.readDataFile(file1, startRow, startCol)

# read the validation data file
file2 = 'TrainingDataBases/database_validation.csv'
dataValidation = dataUtils.readDataFile(file2, startRow, startCol)
inputDataValid, outputDataValid = dataUtils.splitInputsOutputs(dataValidation, nInputs, nOutputs)

# simple function to separate the data by inputs and outputs
inputData,outputData = dataUtils.splitInputsOutputs(data, nInputs, nOutputs)

# read the normalization object that is created in the Normalization file, it continas the information regarding the max and min values of the input data
# (the object will store the max and min values of every column so that we can normalize and denormalize the data at anytime)
normalizer = normalization.getNormalizerForSurrogateModel()

# normalize the inputs
normInput = normalizer.normalizeInputs(inputData)

# start to measure the time
startTime = time.time()


# create and train a fully connected BPNN nnet with some pre-defined parameters
nnet, history = NeuralNet.createSequentialModel(normInput, 
                                                outputData,
                                                layerSizes=[200,200,200],
                                                nEpochs=200,
                                                nBatchSize=10,
                                                validationSplit=0.10,
                                                earlyStop=True,
                                                earlyStopPatience=5
                                                )


# Compute MSE, R, and R2.
# if the model has multiple outputs, the reported results are the averages
metrics = dataUtils.getSimpleMetricsAverages(nnet,normalizer,dataValidation,nInputs,nOutputs)
mse = metrics[0]
sR = metrics[1]
sR2 = metrics[2]

print("Total MSE = ", mse)
print("Total R   = ", sR)
print("Total R2  = ", sR2)

# print the loss history
NeuralNet.plotHistoryFrom(history,['mse','val_mse'],['Training','Validation'],1,"Epoch","MSE",metrics)

# print the training time
executionTime = (time.time() - startTime)
print('NN training time in seconds: ' + str(executionTime))

# Get the metrics and plots for the output variable "index".
# index = 1
# dataUtils.getMetricsForVariable(nnet,normalizer,dataValidation,nInputs,nOutputs,index)

# Get the metrics and plots for all the output variables
dataUtils.getMetricsForAllVariable(nnet,normalizer,dataValidation,nInputs,nOutputs)


# save the NN to a file
path='NeuralNetworkWeights/dnn_surrogate_model.h5'
nnet.save(path)




    
