
# Modeling RC shear wall using Deep Neural Networks

![alt text](https://github.com/germansr/RC.ShearWall.DNN.SurrogateModel/blob/main/Images/ImageWall.png) 

# Main Instructions 
Most files are extensively commented and easy to adapt/modify. The intended workflow is as follows:

**0- Preparation: install the required libraries**
- OpenseesPy: https://pypi.org/project/openseespy/
- TensorFlow: https://www.tensorflow.org/install
- Numpy: https://numpy.org/install/
- tkinter: https://docs.python.org/3/library/tkinter.html  (usually included in standard python distribution)
- matplotlib:  https://matplotlib.org/stable/users/installing/index.html

**1- Quick test of the FEM model using OpenSeesPy**\
To perform a quick test with the FEM model using OpenSeesPy, run the file "RunValidationExample.py".

**2- Run many FEM simulations to create the database**\
Open the file "CreateDataBase_Loop.py", select the number of simulations to run by changing the corresponding variable, and run the file. This is an expensive step as each simulation takes around 40 seconds to complete. The results are saved to a text file and stored in the folder "AnalysisResults". (important to be consistent with the file names because they are used in the next step).

**3- Data curation and preparation of the training database**\
To create the database run the file "DiscretizeCurvesAndCreateDatabase.py". This script will discretize the pushover curve into 6 sections and create the training and testing data bases. The databses are stored in the folder "TrainingDataBases". (important to be consistent with the file names because they are used in the next step).

**4- Train the ANN surrogate model**\
To train the ANN surrogate model, run the file "MainNN.py". Follow the instructions and comments in the file to change the ANN structure if neccesary. The file "NeuralNetwork.py" constructs the ANN model based on some predefined parameters and the user-defined hyperparameters. (important to be consistent with the file name for the serialization of the ANN model which is used by the GUI application).

**5- Test the methodology with the interactive GUI**\
To open the GUI application, run the file "AppGUI.py". The app loads the pre-trained ANN on opening and performs real-time predictions based on the slider values. Use the sliders to modify the input variables. To test the surrogate model againts the FEM analysis, run the analysis with the button "run FEM analysis". The analysis is performed in the background. After the analysis is completed, the results are shown in the top-right plot area where they can be compared with the surrogate model predictions. 

**Misc**
- The file "ShearWallParametrizedAsFunction.py" ccontains the main function to run the FEM model based on the 11 input values and some other input data.
- The file "InputVariableBounds.py" controls the bounds of the input variables.
- The file "Normalization.py" is a helper class to easily normalize and denormalize data.
- The files "ColorMapFEM.py" and "MyPlottingFEM.py" are various script mainly developed to add visual feedback to the opensees library.
- The files "DataUtils" contains functions to check the performance of the ANN model.

# About
- **Development:** Ph.D. Candidate German Solorzano (sr.german90@gmail.com, https://www.linkedin.com/in/germansolorzano/)
- **Supervision:** Dr. Vagelis Plevris (vplevris@gmail.com, http://www.vplevris.net/en/)
- **Sponsored:**  Oslo Metropolitan University (OsloMet), Department of Civil Engineering and Energy Technology, Oslo, Norway (https://www.oslomet.no/en/research/research-groups/structural-engineering)
