"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Project: 
An Open-Source Framework for Modeling RC Shear Walls using Deep Neural Networks

File:    
RunValidationExample.py

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
# This file runs the numerical example that has been used to validate the model
# visit this site for more information: https://openseespydoc.readthedocs.io/en/latest/src/RCshearwall.html    

import ShearWallParametrizedAsFunction as shearWall


# -------------- VALIDATION EXAMPLE AS PARAMETER VALUES ------------------
t= 0.125;
lw = 1;
lbe = 0.20;
pl_be = 0.0188496;
pt_be = 0.00282744;
pl_web = 0.00376992;
pt_web = 0.0018095616;
fc = 20.7e6
fy = 392e6
paxial = 246000 / (0.85*20.7e6*t*lw)
height=2.0


targetDisp = 0.020
increment = 0.0001

# print all the figures
performPushOver = True
plotValidation = True
plotDeformedGravity = True
plotPushOverResults = True

# run the parametric model 
[x,y] = shearWall.run(t,
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


