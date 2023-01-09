"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Project: 
An Open-Source Framework for Modeling RC Shear Walls using Deep Neural Networks

File:    
Normalization.py

Date:    
28.12.2022

Developmed by:
-Ph.D. Candidate German Solorzano
Supervised by:
-Dr. Vagelis Plevris

Sponsored by:
Oslo Metropolitan University, Oslo, Norway.
Department of Civil Engineering and Energy Technology 
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# Header:
# THIS CLASS IS IMPORTANT FOR THE PROPER FUNCTION OF THE SURROGATE MODEL
# It is a simple class to help to store the max and min values of the input data and to perform normalization according to those stored values
# The normalizer has to be used everytime the surrogate model has is used because the input values must be normalized before any prediction!

import numpy as np
import InputVariableBounds as inputBounds

class SimpleNormalizer():
    
    def __init__(self, nInputs, nOutputs):
        self.nIn = nInputs
        self.nOut = nOutputs
        dataCols = self.nIn+self.nOut
        minValues = np.empty(shape=(dataCols,))
        maxValues = np.empty(shape=(dataCols,))    
        self.minValuesIn = minValues[0:self.nIn];
        self.maxValuesIn = maxValues[0:self.nIn];
        self.minValuesOut = minValues[self.nIn:self.nIn+self.nOut];
        self.maxValuesOut = maxValues[self.nIn:self.nIn+self.nOut];

    def getMinValueInput(self,index):
        return self.minValuesIn[index]
    
    def getMaxValueInput(self,index):
        return self.maxValuesIn[index]    
    
    def getMinValueOutput(self,index):
        return self.minValuesOut[index]
    
    def getMaxValueOutput(self,index):
        return self.maxValuesOut[index]  
    
    def setManualMinValueInput(self,index,minValue):
        self.minValuesIn[index]=minValue
        
    def setManualMaxValueInput(self,index,maxValue):
        self.maxValuesIn[index]=maxValue

    def normalizeInputs(self,data):
        return self.normalizeData(data,self.minValuesIn,self.maxValuesIn);
    
    def normalizeOutputs(self,data):
        return self.normalizeData(data,self.minValuesOut,self.maxValuesOut);
     
        
    def denormalizeInputs(self,data):
        return self.denormalizeData(data,self.minValuesIn,self.maxValuesIn);
    
    def denormalizeOutputs(self,data):
        return self.denormalizeData(data,self.minValuesOut,self.maxValuesOut);    
       
        
    def normalizeData(self,data,minValues,maxValues):
        nCols = len(minValues);
        nRows =  len(data);
        normalized = np.zeros((nRows,nCols));
        for i in range(nCols):
            normalized[:,i] = (data[:,i]-minValues[i]) / (maxValues[i]-minValues[i]);
        return normalized
    
    def denormalizeData(self,data,minValues,maxValues):
        nCols = len(minValues);
        nRows =  len(data);
        normalized = np.zeros((nRows,nCols));
        for i in range(nCols):
            normalized[:,i] = (maxValues[i]-minValues[i])*data[:,i]+minValues[i];
        return normalized    

# This function contains the information regarding the input data for the RC shear wall surrogate model project

def getNormalizerForSurrogateModel():
    
    # number of inputs of the DNN surrogate model
    nInputs = 11
    
    # number of outputs of the DNN surrogate model
    nOutputs = 6
    
    # normalizer object
    normalizer = SimpleNormalizer(nInputs, nOutputs);
    
    minValues = inputBounds.minValues
    
    maxValues = inputBounds.maxValues
    
    # min values of the 11 input variables for the surrogate model
    normalizer.setManualMinValueInput(0, minValues[0]) 
    normalizer.setManualMinValueInput(1, minValues[1])  
    normalizer.setManualMinValueInput(2, minValues[2])  
    normalizer.setManualMinValueInput(3, minValues[3])  
    normalizer.setManualMinValueInput(4, minValues[4])
    normalizer.setManualMinValueInput(5, minValues[5]) 
    normalizer.setManualMinValueInput(6, minValues[6]) 
    normalizer.setManualMinValueInput(7, minValues[7]) 
    normalizer.setManualMinValueInput(8, minValues[8]) 
    normalizer.setManualMinValueInput(9, minValues[9])
    normalizer.setManualMinValueInput(10, minValues[10]) 
    
    
    # max values of the 11 input variables for the surrogate model
    normalizer.setManualMaxValueInput(0, maxValues[0])
    normalizer.setManualMaxValueInput(1, maxValues[1])
    normalizer.setManualMaxValueInput(2, maxValues[2])
    normalizer.setManualMaxValueInput(3, maxValues[3])
    normalizer.setManualMaxValueInput(4, maxValues[4])
    normalizer.setManualMaxValueInput(5, maxValues[5])
    normalizer.setManualMaxValueInput(6, maxValues[6])
    normalizer.setManualMaxValueInput(7, maxValues[7])
    normalizer.setManualMaxValueInput(8, maxValues[8]) #minimum axial load
    normalizer.setManualMaxValueInput(9, maxValues[9]) #minimum height
    normalizer.setManualMaxValueInput(10, maxValues[10]) #minimum fc  
    
    return normalizer