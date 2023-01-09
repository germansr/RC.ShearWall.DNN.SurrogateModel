"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Project: 
An Open-Source Framework for Modeling RC Shear Walls using Deep Neural Networks

File:    
CreateDataBase_Loop.py

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
# This file is a loop that generates random data points, then performs the static pushover analysis, and stores the results in a file

# Note that the script wont override the previous file. At every iteration, 
# the file will open and the new information will be added. Thus, stopping the script and re-running it later is OK!

# The generated file contains 3 rows per data point:
    # the first row is the input vector,
    # the second row is the x-axis values of the pushover plot,
    # and the third row is the y-axis values of the pushover curve

# The generated file must be processed by an other function (CreateRealDataBase.py) to create the training database with one row per data point   


import random as rnd
import os
import ShearWallParametrizedAsFunction as shearWallAsFunc
import csv
import InputVariableBounds as inputBounds


# number of samples to run
samples = 3

# Create a folder to put the output
fileName = "database_complete.csv"            
ResultsDir = 'AnalysisResults'
pathToFile = ResultsDir+"/"+fileName

if not os.path.exists(ResultsDir):
	os.makedirs(ResultsDir)
 
# parameters for the static pushover analysis    
performPushOver = True
plotValidation = False
plotDeformedGravity = False
plotPushOverResults = False    
  
# other variables
targetDisp = 0.02
steps = 200
increment = 0.02/steps

# discretization size (number of elements in each direction)
meshHorizontal = 8 # total elements in horizontal direction
meshBE = 2         # from the total elements this number is used for each boundary element
meshVertical = 10  
  
# loop to generate random input vectors and perform the analysis
for i in range(samples):

    # open file and write the results every time
    f = open(ResultsDir+"/"+fileName, 'a', newline='')
    writer = csv.writer(f)
    
    print("RUNNING SAMPLE: ",i)

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
    
    # save the 11 input values        
    params = [t,lw,lbe,pl_be,pt_be,pl_web,pt_web,paxial,height,fc,fy]

    # store the obtained pushover curve
    # -only keep the data points that converge to more than 1cm of displacement
    maxX = max(x)
    if maxX > 10:
        print("Printing: ",maxX)
        writer.writerow(params)
        writer.writerow(x)
        writer.writerow(y)

    f.close()
    
f.close()

