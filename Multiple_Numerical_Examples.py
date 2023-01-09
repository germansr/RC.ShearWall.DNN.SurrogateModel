"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Project: 
An Open-Source Framework for Modeling RC Shear Walls using Deep Neural Networks

File:    
Multiple_Numerical_Examples.py

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
# This file creates random vectors, then, it runs the corresponding pushover analysis using the FEM model and also uses the stored ANN to predict the results
# The curves are presented in a nice 2x1 containing several curves each (see the second figure in section "numerical examples" of the paper).
# The number of curves per plot is controlled by the variable "NumOfCurves". Using a large number will take several minutes to finalize (40 secs per analysis approx)
# The ANN has to be already trained and stored in the path specified in the variable "pathToTheNN"


import numpy as np
import random as rnd
import ShearWallParametrizedAsFunction as shearWallAsFunc
import InputVariableBounds as inputBounds
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import Normalization as normalization
import TrainedNNprediction as testNN

# control how many curves are drawn per plot
NumOfCurves = 3

#  LOAD THE PREVIOUSLY SAVED NEURAL NETWORK MODEL
normalizer = normalization.getNormalizerForSurrogateModel()
pathToTheNN='NeuralNetworkWeights/dnn_surrogate_model.h5'
nnet=load_model(pathToTheNN)
nnet.summary()

# CREATE THE MIN-MAX NORMALIZER
figure, ax = plt.subplots(1, 2)

# curve colors
colors = ["magenta","green","blue","red","orange","grey","springgreen","yellowgreen"]

loopCount = 0
terminate = 5

outsideCount = 0
for k in range(2):
    
    loopCount = 0 
    for i in range(NumOfCurves): 
        
        print("index",i)
        concMaterial=1
        performPushOver = True
        plotValidation = False
        plotDeformedGravity = False
        plotPushOverResults = False
        
        minValues = inputBounds.minValues
        maxValues = inputBounds.maxValues

        # thickness
        t = rnd.uniform(minValues[0],maxValues[0])
        
        # total length of wall
        lw = rnd.uniform(t*6,maxValues[1])
        
        # boudary element length as percentage of the total lenght
        lbe = rnd.uniform(minValues[2],maxValues[2])
        
        # ratio of reinforcement for the web, longitudinal (l) and transversal (t)
        pl_be = rnd.uniform(minValues[3],maxValues[3])
        pt_be = rnd.uniform(minValues[4],maxValues[4])
        
        # ratio of reinforcement for the web, longitudinal (l) and transversal (t)
        pl_web = rnd.uniform(minValues[5], 0.6*pl_be)
        pt_web = rnd.uniform(minValues[6], 0.6*pt_be)
        
        #axial force as percentage of the maximum allowable load for that wall Po = Ag*0.85*f'c
        paxial = rnd.uniform(minValues[7],maxValues[7]) 
        
        # heihg of the wall
        height = rnd.uniform(minValues[8],maxValues[8])
        
        # compressive strength in MPa
        fc = rnd.uniform(minValues[9],maxValues[9])
        
        # yield strength in MPa
        fy = rnd.uniform(minValues[10],maxValues[10])
        
        # other variables
        targetDisp = 0.02
        steps = 200
        increment = 0.02/steps
        
        # discretization size (number of elements in each direction)
        meshHorizontal = 8 # total elements in horizontal direction
        meshBE = 2         # from the total elements this number is used for each boundary element
        meshVertical = 10
        
        # run the non-linear static pushover analysis
        [x,y],ops = shearWallAsFunc.run(t,
                                    lw,
                                    lbe,
                                    pl_be,
                                    pt_be,
                                    pl_web,
                                    pt_web,
                                    paxial,
                                    height,       
                                    fc,
                                    fy,
                                    meshHorizontal,
                                    meshBE,
                                    meshVertical,
                                    targetDisp,
                                    increment,
                                    performPushOver,
                                    plotValidation,
                                    plotDeformedGravity,
                                    plotPushOverResults) 
       
        # PREDICT THE VALUES USING THE STORED NEURAL NETWORK
        inputValues, predOut = testNN.predict(nnet,
                                            normalizer,
                                            t, 
                                            lw, 
                                            lbe, 
                                            pl_be, 
                                            pt_be, 
                                            pl_web,  
                                            pt_web, 
                                            paxial, 
                                            height, 
                                            fc, 
                                            fy
                                            )
        
        # create the discretized curve to be plotted
        xdata = [0,0.5,1.0,2.5,5,10,20]
        ydata = []
        ydata.append(0)
        for val in predOut[0]:
            ydata.append(val)
         

        if loopCount==0:
            ax[k].plot(xdata,ydata, 
                      linewidth=1, 
                      linestyle="-", 
                      color='black',
                      marker = 'x',
                      label="ANN predictions",
                      markersize = 5)
        else:
            ax[k].plot(xdata,ydata, 
                      linewidth=1, 
                      linestyle="-", 
                      color='black',
                      marker = 'x',
                      markersize = 5)
          
        print(y)    
        ax[k].plot(x,y, linewidth=3, 
                 linestyle="-", 
                 label="DP"+str(outsideCount+1),
                 color=np.random.rand(3,),
                 alpha=0.7)
        
        ax[k].grid(linestyle='dotted')
        ax[k].legend(loc='upper left') 
        loopCount = loopCount+1
        outsideCount  = outsideCount+1
        ax[k].set_xlabel('Displacement [mm]')
        
        if k==0:
           ax[k].set_ylabel('Base Shear [kN]')
        


    




