"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Project: 
An Open-Source Framework for Modeling RC Shear Walls using Deep Neural Networks

File:    
InputVariableBounds.py

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
# contains the input and output variable bounds as global variables. 
# this is to facilitate any changes in these values as they are required in multiple subroutines across the project
# note that this file must be imported so that it can be accessed by the other files

minValues = [0.125,   # thickness
             0.75,    # wall length (min t*6)
             0.15,    # BE length (as percentage of wall length)
             0.01,    # BE long reinf ratio
             0.0075,  # BE transv reinf ratio
             0.0025,  # web long reinf ratio
             0.0025,  # web transv reinf ratio
             0.01,    # axial load ratio
             3.0,     # height
             25e6,    # fc
             380e6]   # fy

maxValues = [0.40,    # thickness
             3.00,    # wall length (min t*6)
             0.30,    # BE length (as percentage of wall length)
             0.04,    # BE long reinf ratio
             0.015,   # BE transv reinf ratio
             0.025,   # web long reinf ratio
             0.0080,  # web transv reinf ratio
             0.1,     # axial load ratio
             3.5,     # height
             60e6,    # fc
             600e6]   # fy