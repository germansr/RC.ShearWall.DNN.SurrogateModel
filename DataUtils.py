"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Project: 
An Open-Source Framework for Modeling RC Shear Walls using Deep Neural Networks

File:    
DataUtils.py

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
# A simple class to handle a few data operations. Read the comments on the scripts for more details

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


# read a data on a text file and skip the first rows or columns based on the given numbers
def readDataFile(file,startRow=0,startCol=0):
    data = np.loadtxt(file, delimiter=',',skiprows=startRow)
    nCols = len(data[0])
    return data[:,startCol:nCols]

# separate a database into two different arrays based on the number of inputs and outputs
def splitInputsOutputs(data,nInputs,nOutputs):
    inputData = data[:,0:nInputs]
    outputData = data[:,nInputs:nInputs+nOutputs]  
    return inputData,outputData

# compute the basic metrics of a ANN: the average values of the Means Squared Error M(SE), the Person Correlation Coefficient (R), and the Coefficient of Determination (R2)
# the arguments are: the trained NN, the normalizer, the database, the number of inputs and the number of outputs
def getSimpleMetricsAverages(nnet,normalizer,data,nIn,nOut):  
    
    inputData, outputData = splitInputsOutputs(data, nIn, nOut)
    normInput = normalizer.normalizeInputs(inputData)
    
    # use the trained model to predict (the obtained output is normalized if the trining data was normalized)
    predOut = nnet.predict(normInput)
    
    # THE R2 provided by the library is just an average of the R2 computed for each output variable
    # this loop is only to prove that
    sumR2 = 0
    sumR = 0
    sumMSE = 0;
    for i in range(nOut):
        dataReal = outputData[:,i]
        dataPred = predOut[:,i]
        scoreR, _ = pearsonr(dataReal, dataPred)
        scoreR2 = r2_score(dataReal, dataPred)
        scoreMSE = mean_squared_error(dataReal, dataPred)
        sumR2 = sumR2 + scoreR2
        sumR = sumR + scoreR
        sumMSE = sumMSE + scoreMSE
    
    totalR = sumR/nOut
    totalR2 = sumR2/nOut
    totalMSE = sumMSE/nOut
    return [totalMSE,totalR,totalR2]
  
# this routine will create a nice plot of the correlation values between the prediction and the ground truth for all the output variables on a ANN
# the arguments are: the trained NN, the normalizer, the database, the number of inputs and the number of outputs 
def getMetricsForAllVariable(nnet,normalizer,data,nIn,nOut):
    
    plt.rcParams.update({'font.size': 16})
    plt.rc('font', family='TimesNewRomman')
    plt.rcParams["font.family"] = "Times New Roman"
    
    inputData, outputData = splitInputsOutputs(data, nIn, nOut)
    normInput = normalizer.normalizeInputs(inputData)

    # use the trained model to predict (the obtained output is normalized if the trining data was normalized)
    predOut = nnet.predict(normInput)
    
    # create a 2-column figure  with multiple plots inside
    figure, ax = plt.subplots(int(nOut/2), 2)
 
    colA = True 
    rowIndices = []
    for i in range(int(nOut/2)):
        rowIndices.append(i)
        rowIndices.append(i)
    
    for i in range(nOut):
        
       if colA: 
           plotCol = 0
           colA=False
       else:
           plotCol = 1
           colA=True
           
       plotRow = rowIndices[i]
       
       # GROUND TRUTH FOR THE SELECTED INDEX
       ground_truth = outputData[:,i]
       
       # PREDICTION FOR THE SELECTED INDEX
       prediction = predOut[:,i]
       
       # COMPUTE R AND R2
       Rscore, _ = pearsonr(ground_truth, prediction)
       R2score = r2_score(ground_truth, prediction)
       MSE = mean_squared_error(ground_truth, prediction)
       
       # plotting range
       length = max(ground_truth) - min(ground_truth);
       props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
       lims = [min(ground_truth), max(ground_truth)]
  
       # textbox with R, R2, and MSE values
       ax[plotRow, plotCol].text(min(ground_truth)+length*0.025, max(ground_truth)-length*0.050, 
                'MSE= '+str(round(MSE,2))+'\n'+'R = '+str(round(Rscore,5))+'\n'+'R²= '+str(round(R2score,5)), 
                fontsize=10,verticalalignment='top', bbox=props)
         
       scatter = ax[plotRow, plotCol].scatter(ground_truth, prediction,  label="real - prediction", marker = 'x', s=15)
       xyLine = ax[plotRow, plotCol].plot(lims, lims,color="red",label="x = y",linewidth=2)
       
       ax[plotRow, plotCol].set_title("Output "+str(i+1), x=0.5, y=0.85)
       ax[plotRow, plotCol].set_xlim(lims)
       ax[plotRow, plotCol].set_ylim(lims)
       ax[plotRow, plotCol].grid(linestyle='dotted')
       
       # only print the x labels for the bottom plots, and the y labels for the plots in the left
       if not colA:
           ax[plotRow, plotCol].set_ylabel('Prediction')
       if i >= nOut-2: 
         ax[plotRow, plotCol].set_xlabel('Real values')
         
        # plot the legend on the last plot
       if i == nOut-1:
          ax[plotRow, plotCol].legend(loc='lower right') 
       
       
 # this routine will create a nice plot of the correlation values between the prediction and the ground truth for a single output variable
 # the arguments are: the trained NN, the normalizer, the database, the number of inputs and the number of outputs    
def getMetricsForVariable(nnet,normalizer,data,nIn,nOut,index):
    
    plt.rcParams.update({'font.size': 16})
    plt.rc('font', family='TimesNewRomman')
    plt.rcParams["font.family"] = "Times New Roman"
    
    inputData, outputData = splitInputsOutputs(data, nIn, nOut)
    normInput = normalizer.normalizeInputs(inputData)

    # use the trained model to predict (the obtained output is normalized if the trining data was normalized)
    predOut = nnet.predict(normInput)
    
    # GROUND TRUTH FOR THE SELECTED INDEX
    ground_truth = outputData[:,index]
    
    # PREDICTION FOR THE SELECTED INDEX
    prediction = predOut[:,index]
    
    # COMPUTE R AND R2
    Rscore, _ = pearsonr(ground_truth, prediction)
    R2score = r2_score(ground_truth, prediction)
    MSE = mean_squared_error(ground_truth, prediction)
    
    
    fig, ax = plt.subplots()
    
    length = max(ground_truth) - min(ground_truth);
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    lims = [min(ground_truth), max(ground_truth)]
        
    #ax.axis('equal')  
    
    ax.set_xlabel('Real values')
    ax.set_ylabel('Prediction')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.grid(linestyle='dotted')
     
    # textbox with R values
    ax.text(min(ground_truth)+length*0.025, max(ground_truth)-length*0.025, 
             'MSE= '+str(round(MSE,2))+'\n'+'R = '+str(round(Rscore,5))+'\n'+'R²= '+str(round(R2score,5)), 
             fontsize=14,verticalalignment='top', bbox=props)
      
    scatter = ax.scatter(ground_truth, prediction,  label="real - prediction", marker = 'x', s=15)
    xyLine = ax.plot(lims, lims,color="red",label="x = y",linewidth=2)
    
    ax.legend(loc='lower right') 
    
    
    #############################
    # ERROR PLOT
    fig, ax = plt.subplots()
    error = prediction - ground_truth
    ax.grid(linestyle='dotted')
    ax.hist(error, bins=20)
    ax.set_xlabel('Error (real-prediction)')
    ax.set_ylabel('Quantity')
    
    plt.tight_layout()
    
