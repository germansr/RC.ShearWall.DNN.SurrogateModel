"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Project: 
An Open-Source Framework for Modeling RC Shear Walls using Deep Neural Networks

File:    
DiscretizeCurvesAndCreateDataBase.py

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
# This script takes the stored results that contain the full pushover curves obtained from the analysis, wchih are stored at the file: "database_complete.csv".
# Then, it will discretizes all the curves into 6 sections and proceed to create a file with the data as input-output vectors (one vector per row)
# Finally, create two subdatabases one for training and one for validation



import csv
import os
import numpy as np
import matplotlib.pyplot as plt


# Plot the data to visualize the discretizations
plotCurves = False

# File with the FEM analysis database
fileName = "database_complete.csv" 

# File with the training database
fileNameTrainingDataBase = "database_training.csv"  

# File with the validation database
fileNameValidationDataBase = "database_validation.csv"  

# SEPARATE TRAINING AND VALIDATION DATA
# EXAMPLE, SUPPOSE THAT 3000 ANALYSIS ARE AVAILABLE IN THE FILE "database_complete.csv. THEN, USE 2500 FOR TRAINING AND 500 FOR VALIDATION
# NUMBER OF DATA POINTS FOR TRAINING
nTraining = 2500
# NUMBER OF DATA POINTS FOR VALIDATION
nValidation = 150
      


# ---- AUTOMATED FROM THIS POINT ----
        
ResultsDir = 'AnalysisResults'
pathToFile = ResultsDir+"/"+fileName
f = open(pathToFile)
csvreader = csv.reader(f)

# CONVERT THE DATABASE TO A NUMPY ARRAY
rows = []
for row in csvreader:
    float_lst = list(np.array(row, dtype = 'float'))
    rows.append(float_lst)
    
f.close();       

# CREATE A DIRECTORY AND A FILE TO SAVE THE TRAINING DATABASE THAT WILL BE GENERATED
fileName = "database_processed.csv"            
ResultsDir = 'TrainingDataBases'
pathToFile = ResultsDir+"/"+fileName

if not os.path.exists(ResultsDir):
	os.makedirs(ResultsDir)
    
f = open(pathToFile, 'w', newline='')
writer = csv.writer(f)

# STATIONS WHERE THE PUSHOVER CURVE WILL BE CUT FOR THE DISCRETIZATION
stations = [0,0.5,1.0,2.5,5,10,19.5]



# SOME VARIABLES FOR THE LOOP
total = 0

# THE RESULTS FILE HAS 3 ROWS PER DATAPOINT, SO THE LOOP NEEDS TO GO ON ONE THIRD OF THE DATA
plotIndex = range(int(len(rows)/3))


if plotCurves:
    plt.figure(figsize=(4,3), dpi=100)
    plt.xlabel('Displacement (mm)')
    plt.ylabel('Base Shear (kN)')

for i in plotIndex: 
    
    print("index:", i)
    # READ THE x and y DATA OF THE PUSHOVER CURVE
    maxV = []
    index = i*3
    params = rows[index]
    x = rows[index+1]
    y = rows[index+2]
    maxX = max(x)
    maxY = max(y)
    maxV.append(maxY)
    #print(maxX,maxY)
    
    # only if it converged to more than 10 mm of displacement, otherwise ignore data point and continue
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
                discPointsX.append(x[count])
                discPointsY.append(y[count])
            count=count+1   
            if skip:
                break;
    
    # if the analysis did not converge until 20 mm, then there is a missing point
    # to complete the 6 points, the baseshear at 20 mm is taken as the base shear at 10mm multiplied by a factor of 1.055
    # the factor of 1.055 is an approximation that has been calculated with hundred of analyses that do converge until 20mm
    if len(discPointsY) < len(stations):               
            discPointsX.append(20)
            discPointsY.append(discPointsY[-1]*1.055)    

    # avoid softening behaviour due to numerical innestabilities
    # sometimes it appears that the displacement at 20 mm is less than the displacement 10 mm, but that is only because the algorithm has
    # taken a value at a "valley" of a numerical innestable section
    if discPointsY[6] < discPointsY[5]:
        discPointsY[6] = discPointsY[5]*1.055


    total = total + 1 
    dataBaseRow = [*params,*discPointsY[1:7]]       
    writer.writerow(dataBaseRow)

    # plot the data to visualize the discretizations
    if plotCurves:
        plt.plot(discPointsX,discPointsY, linewidth=1, linestyle="-.", label='Pushover',color='black')
        plt.plot(x,y, linewidth=1, linestyle="-", label='Pushover',color='red')
        

 
# show all the pusover curves and their discretization
if plotCurves:
    plt.show()
    
f.close();


# NOW THAT ALL THE ANALYSIS RESULTS HAVE BEEN CONVERTED TO A single-row FORMAT, CHOSE HOW MANY ARE UED FOR TRAINING AND HOW MANY FOR VALIDATION
# NOTE: THIS VALIDATION SET IS A DIFFERENT SET THAT THE ONE THAT IS MONITORED DURING TRAINING. IT IS UED TO MEASURE THE R, R2, AND MSE SCORES MANUALLY AFTER THE TRAINING IS COMPLETED!!

fileName = "database_processed.csv"            
ResultsDir = 'TrainingDataBases'
pathToFile = ResultsDir+"/"+fileName
f = open(pathToFile)
csvreader = csv.reader(f)

# CONVERT THE DATABASE TO A NUMPY ARRAY
rows = []
for row in csvreader:
    float_lst = list(np.array(row, dtype = 'float'))
    rows.append(float_lst)
    
f.close(); 



# CREATE A DIRECTORY AND A FILE TO SAVE THE TRAINING DATABASE THAT WILL BE GENERATED
fileName = fileNameTrainingDataBase     
ResultsDir = 'TrainingDataBases'
pathToFile = ResultsDir+"/"+fileName

if not os.path.exists(ResultsDir):
	os.makedirs(ResultsDir)
    
f = open(pathToFile, 'w', newline='')
writer = csv.writer(f)


for i in range(nTraining):
    writer.writerow(rows[i])
    
    
f.close(); 


# CREATE A DIRECTORY AND A FILE TO SAVE THE TRAINING DATABASE THAT WILL BE GENERATED
fileName = fileNameValidationDataBase           
ResultsDir = 'TrainingDataBases'
pathToFile = ResultsDir+"/"+fileName

if not os.path.exists(ResultsDir):
	os.makedirs(ResultsDir)
    
f = open(pathToFile, 'w', newline='')
writer = csv.writer(f)


for i in range(nTraining,nTraining+nValidation):
    writer.writerow(rows[i])
    
    
f.close(); 



