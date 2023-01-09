"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Project: 
An Open-Source Framework for Modeling RC Shear Walls using Deep Neural Networks

File:    
NeuralNetwork.py

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
# This file creates a BPNN. Read the comments on the scripts for more details


import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense


# routine to create a keras sequential model (backpropagation neural network)
# the parameters are:
    # number of inputs
    # number of outputs
    # an array with the dimension of the hidden layers in the form [number_neurons_layer1,number_neurons_layer2,..,number_neurons_layerN]
    # number of epochs to train
    # the batch size
    # split ratio to use to create the validation data
    # include early stopping?
    # patience, or number of iterations with no improvement before stopping the algorithm
def createSequentialModel(inputs,outputs,layerSizes,nEpochs,nBatchSize,validationSplit,earlyStop=True, earlyStopPatience = 10):
    
    model = Sequential()
    
    # number of inputs (Taken from the size of the given vector)
    nIns = len(inputs[0])
    # number of outputs (Taken from the size of the given vector)
    nOuts = len(outputs[0])
    # number of hidden layers (Taken from the size of the given vector)
    nHiddenLayers = len(layerSizes)
    
    # Create First Layer
    model.add(Dense(units=layerSizes[0], input_dim=nIns , kernel_initializer='random_uniform', activation='relu'))   
    
    # Create hidden  layers
    for i in range(nHiddenLayers-1):
        model.add(Dense(units=layerSizes[i+1], kernel_initializer='random_uniform', activation='relu'))
           
    # Create output layer with linear activation function
    model.add(Dense(units=nOuts, kernel_initializer='random_uniform', activation='linear'))
     
    # Compile and set error metrics
    model.compile(loss="mse", optimizer="adam", metrics=['mse',"accuracy"])
    
    # =============================================================================
    # EARLY STOPPING ALGORITHM 
    # monitor: Quantity to be monitored.
    # min_delta: Minimum change in the monitored quantity to qualify as an improvement, i.e. an absolute change of less than min_delta, will count as no improvement.
    # patience: Number of epochs with no improvement after which training will be stopped.
    # verbose: verbosity mode.
    # mode: One of {"auto", "min", "max"}. In min mode, training will stop when the quantity monitored has stopped decreasing; in "max" mode it will stop when the quantity monitored has stopped increasing; 
    #   in "auto" mode, the direction is automatically inferred from the name of the monitored quantity.
    # baseline: Baseline value for the monitored quantity. Training will stop if the model doesn't show improvement over the baseline.
    # restore_best_weights: Whether to restore model weights from the epoch with the best value of the monitored quantity. 
    #   if False, the model weights obtained at the last step of training are used. An epoch will be restored regardless of the performance relative to the baseline. If no epoch improves on baseline, training will run for patience epochs and restore weights from the best epoch in that set.
    # =============================================================================


    # EARLY STOPPING CRITERIA
    callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=1e-6,
        patience=earlyStopPatience,
        verbose=0,
        mode="auto",
        baseline=None,
        restore_best_weights=True)
    
    model.summary()

    # IF EARLY STOP IS SPECIFIED THEN APPLY IT
    if earlyStop:
        # train model (note: use the normalized sets for the training process)
        history = model.fit(inputs, 
                            outputs, 
                            validation_split=validationSplit, 
                            epochs=nEpochs, 
                            batch_size=nBatchSize, 
                            verbose=1,
                            callbacks=callback)
      
    # DO NOT APPLY EARLY STOP AND RUN THE TOTAL NUMBER OF EPOCHS                        
    else:
        # train model (note: use the normalized sets for the training process)
        history = model.fit(inputs, 
                            outputs, 
                            validation_split=validationSplit, 
                            epochs=nEpochs, 
                            batch_size=nBatchSize, 
                            verbose=1)

    # RETURN THE MODEL AND THE HISTORY INFORMATION (FOR PLOTTING)
    return model, history;



def plotHistory(history,functions,legend):
    plt.figure()
    for f in functions:    
        plt.plot(history.history[f])
        plt.ylabel(f)
        plt.xlabel('epoch')
        plt.show()  

    
def plotHistoryFrom(history,functions,legend,firstEpoch,xlabel,ylabel,metrics=None):
    
    plt.rcParams.update({'font.size': 14})
    plt.rc('font', family='TimesNewRomman')
    plt.rcParams["font.family"] = "Times New Roman"
    
    fig, ax = plt.subplots()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(linestyle='dotted')
    count = 0
    
    colors = ["blue","red","green"]
    lineWidth = [1,1,1]
    lineStyle = ["-", 'dashed', 'dashed']
    
    
    maxY = 0
    for f in functions:    
        y = history.history[f]
        maxY = max(y)
        plt.plot(range(firstEpoch,len(y)),  y[firstEpoch:len(y)], color=colors[count], linewidth=lineWidth[count], linestyle=lineStyle[count] )   
        plt.show() 
        count = count+1
       
    if metrics is not None:
        ax.text(5, maxY*0.70,  
             'MSE= '+str(round(metrics[0],2))+'\n'+'R = '+str(round(metrics[1],5))+'\n'+'R²= '+str(round(metrics[2],5)), 
             fontsize=10,verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))   
        
  
    plt.legend(legend,loc='upper right') 
    plt.tight_layout()