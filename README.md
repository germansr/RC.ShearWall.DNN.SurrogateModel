# Abstract
A data-driven surrogate model for analyzing RC shear walls is developed using Deep Neural Networks (DNN). The surrogate model is trained with thousands of FEM simulations to predict the characteristic curve obtained when a static non-linear pushover analysis is performed. The surrogate model is extensively tested and found to exhibit a high degree of accuracy in its predictions while being extremely faster than the FEM analysis. In addition to the presented methodology, the complete code and framework that made this study possible is provided as an open-source project with the intention to bridge the gap between research and practical application of Machine Learning powered techniques in Structural Engineering. The project is developed on Python and includes a parametric FEM model of an RC shear wall in OpenSeesPy, the training and validation of the DNN model in TensorFlow, and an application with an interactive Graphical User Interface to test the methodology and visualize the results. 

# Main Instructions 
Most files are extensively commented and easy to adapt/modify. The workflow is explained as follows:

**1- Quick Test of the FEM model using OpenSeesPy**\
To perform a quick test with the FEM model using OpenSeesPy, run the file "RunValidationExample.py"

**2- Run FEM simulations to create the database**\
Open the file "CreateDataBase_Loop.py", select the number of simulations to run by changing the corresponding variable, and run the file. (important to be consistent with the file name of all the analysis results that are used in the following step)

**3- Data curation and prepare the training database**\
Follow the instructions in the file "DiscretizeCurvesAndCreateDatabase.py" (important to be consistent with the file name of the created training and testing databases that are used in the following step)

**4- Train the ANN surrogate model**\
Run the file "MainNN.py". Follow the instructions and comments in the file to change the ANN structure if neccesary. (important to be consistent with the file name for the serialization of the ANN model which is used by the GUI application)



# About
Development: Ph.D. Candidate German Solorzano (sr.german90@gmail.com)

Supervision: Dr. Vagelis Plevris (vplevris@gmail.com)

Sponsored:  Oslo Metropolitan University (OsloMet), Department of Civil Engineering and Energy Technology, Oslo, Norway.
