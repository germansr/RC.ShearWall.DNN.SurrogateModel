"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Project: 
An Open-Source Framework for Modeling RC Shear Walls using Deep Neural Networks

File:    
TrainedNNPrediction.py

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
# This file contains a couple of functions to test the trained NN. Read the comments on the functions for more information


import numpy as np
import matplotlib.pyplot as plt
import random as rnd
import ShearWallParametrizedAsFunction as shearWallAsFunc


# Predict the corresponding output for a set of input values
# The required arguments are the trained NN, the normalizer, and the 11 input values
def predict(nnet, normalizer, v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11):
    predIn = np.array([[v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11]]);
    predInNorm = normalizer.normalizeInputs(predIn) #prediction input normalized
    predOut = nnet.predict(predInNorm)
    return predIn, predOut
    

# Create a random vector, perform the static pushover analysis and compare the result to the prediction of the NN, plot the results
# The required arguments are the normalizer and the trained NN.
def testNN(normalizer, nnet):
    
    performPushOver = True
    plotValidation = False
    plotDeformedGravity = False
    plotPushOverResults = False
    
    #-------- GENERATE A RANDOM VECTOR ----------------------------------------------------
    # compressive strength in MPa
    fc = rnd.uniform(25e6,40e6) 
    
    # yield strength in MPa
    fy = rnd.uniform(350e6,450e6) 
    
    # heihg of the wall
    height = rnd.uniform(3,3.5) 
    
    # thickness
    t = rnd.uniform(0.125,0.40)
    
    # total length of wall
    lw = rnd.uniform(t*6,4)
    
    # boudary element length as percentage of the total lenght
    lbe = rnd.uniform(0.15,0.25)
    
    # ratio of reinforcement for the web, longitudinal (l) and transversal (t)
    pl_be = rnd.uniform(0.01,0.035)
    pt_be = rnd.uniform(0.0075,0.015)
    
    # ratio of reinforcement for the web, longitudinal (l) and transversal (t)
    pl_web = rnd.uniform(0.0025, 0.5*pl_be)
    pt_web = rnd.uniform(0.0025, 0.5*pt_be)
    
    #axial force as percentage of the maximum allowable load for that wall Po = Ag*0.85*f'c
    paxial = rnd.uniform(0.010,0.10)     
       
    #----------------------------------------------------------------------------------
    
    targetDisp = 0.02
    steps = 200
    increment = targetDisp/steps
    
    
    #---------  CREATE THE MODEL AND RUN THE ANALYSIS WITH THE RANDOM VECTOR ---------
    [x,y] = shearWallAsFunc.run(t,
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
                                8,
                                2,
                                10,
                                targetDisp,
                                increment,
                                performPushOver,
                                plotValidation,
                                plotDeformedGravity,
                                plotPushOverResults)  
    #----------------------------------------------------------------------------------
       
    # initialize graphic to see the prediction
    plt.figure(figsize=(4,3), dpi=100)
    plt.xlabel('Displacement (mm)')
    plt.ylabel('Base Shear (kN)')
    
    style = dict(color = 'lightgray',
                 linewidth=1, 
                 linestyle=":",
                 )


    # predict the output using the NN
    predIn = np.array([[t,lw,lbe,pl_be,pt_be,pl_web,pt_web,paxial,height,fc,fy]]);
    predInNorm = normalizer.normalizeInputs(predIn) #prediction input normalized
    predOut = nnet.predict(predInNorm)
    #predOut = normalizer.denormalizeOutputs(predOutNorm) #prediction input normalized

    # create the x-y points of the prediction
    x2 = [0,0.5,1.0,2.5,5,10,19.5]
    y2 = []
    y2.append(0)
    for val in predOut[0]:
        y2.append(val)
        
    # plot the predicted curve    
    plt.plot(x2,y2, 
             linewidth=1, 
             linestyle="-", 
             color='black',
             marker = 'x',
             markersize = 5)
    
    # plot grid (vertical lines)
    plt.axvline(x = 0, **style)
    plt.axvline(x = 0.5, **style)
    plt.axvline(x = 1.0, **style)
    plt.axvline(x = 2.5, **style)
    plt.axvline(x = 5.0, **style)
    plt.axvline(x = 10.0, **style)
    plt.axvline(x = 20.0, **style)
    
    # plot grid (horizontal lines)
    stepGridH = 200
    maxy = max(max(y), max(y2)) + stepGridH
    
    for i in range(500):
        val = i*stepGridH
        if val < maxy:
            plt.axhline(y = val, **style)
    
    
    # plot the computed pushover curve
    plt.plot(x,y, linewidth=2, linestyle="-", label='Pushover',color='red')
    plt.show()






