"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Project: 
An Open-Source Framework for Modeling RC Shear Walls using Deep Neural Networks

File:    
VisualizeCurveDiscretization.py

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
# This file takes the file of the pushover FEM analysis results and selects randomly a few points then plots the corresponding discretizaton
# this file is just basically a visualization tool to check that the curves are being discretized correctly    


import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import random as r
from tensorflow.keras.models import load_model

fileName = "database_complete.csv"            
ResultsDir = 'AnalysisResults'
pathToFile = ResultsDir+"/"+fileName

f = open(pathToFile)
csvreader = csv.reader(f)

rows = []
for row in csvreader:
    float_lst = list(np.array(row, dtype = 'float'))
    rows.append(float_lst)
    
f.close();       

# Create a folder to put the output
fileName = "test.csv"            
ResultsDir = 'AnalysisResults'
pathToFile = ResultsDir+"/"+fileName

if not os.path.exists(ResultsDir):
	os.makedirs(ResultsDir)
    
f = open(pathToFile, 'w', newline='')
writer = csv.writer(f)

stations = [0,0.5,1.0,2.5,5,10,19.5]

plt.figure(figsize=(4,3), dpi=100)
plt.xlabel('Displacement (mm)')
plt.ylabel('Base Shear (kN)')

plotIndex = range(int(len(rows)/3))
#plotIndex = range(10)

loopCount = 0
terminate = 5


index1 = r.randint(0, 2500)
index2 = r.randint(0, 2500)
index3 = r.randint(0, 2500)
index4 = r.randint(0, 2500)


indexes = [index1,index2,index3,index4]
colors = ["magenta","green","blue","red"]
labels = ["DP1","DP2","DP3","DP4"]


style = dict(color = 'lightgray',
             linewidth=1, 
             linestyle=":",
             )

plt.axvline(x = 0, **style)
plt.axvline(x = 0.5, **style)
plt.axvline(x = 1.0, **style)
plt.axvline(x = 2.5, **style)
plt.axvline(x = 5.0, **style)
plt.axvline(x = 10.0, **style)
plt.axvline(x = 20.0, **style)

plt.axhline(y = 0, **style)
plt.axhline(y = 500, **style)
plt.axhline(y = 1000, **style)
plt.axhline(y = 1500, **style)
plt.axhline(y = 2000, **style)


# # LOAD THE PREVIOUSLY SAVED NEURAL NETWORK MODEL
pathToTheNN='NeuralNetworkWeights/dnn_surrogate_model.h5'
nnet=load_model(pathToTheNN)
nnet.summary()



plt.rcParams["font.family"] = "Times New Roman"
plt.tight_layout()

for i in indexes: 
    
    print("index:", i)
    maxV = []
    index = i*3
    params = rows[index]
    x = rows[index+1]
    y = rows[index+2]
    maxX = max(x)
    maxY = max(y)
    maxV.append(maxY)
    #print(maxX,maxY)
    
    # only if it converged to more than 5 mm of displacement, otherwise ignore data point
    if maxX < 10: 
        continue
    
    # discretize the curve into 6 sections, at: 0, 1, 2.5, 5, 10, 19.5
    discPointsX = []
    discPointsY = []
    for station in stations:
        count = 0
        skip = False
        for xval in x:
            if xval >= station:
                skip=True
                if xval > 18 and xval <=20:
                    discPointsX.append(20)
                else:
                    discPointsX.append(x[count])    
                discPointsY.append(y[count])
            count=count+1   
            if skip:
                break;
            
    if len(discPointsY) < len(stations):               
            discPointsX.append(20)
            discPointsY.append(discPointsY[-1]*1.055)
    
    if discPointsY[6] < discPointsY[5]:
        discPointsY[6] = discPointsY[5]*1.055

    
    
    print("input",rows[i*3])      
    print(discPointsX)
    print(discPointsY)
    
    
    
    plt.plot(x,y, linewidth=3, 
             linestyle="-", 
             label=labels[loopCount],
             color=colors[loopCount],
             alpha=0.7)
    
    plt.plot(discPointsX,discPointsY, 
             linewidth=1, 
             linestyle="-", 
             color='black',
             marker = 'x',
             markersize = 5)
    
    dataBaseRow = [*params,*discPointsY]
    writer.writerow(dataBaseRow)
    loopCount = loopCount+1




plt.legend();
plt.show()
    
f.close();



