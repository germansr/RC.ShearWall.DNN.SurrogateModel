# Introduction
A data-driven surrogate model for the analysis of RC shear walls is developed using Deep Neural Networks (DNN). The surrogate model is trained with thousands of FEM simulations to predict the characteristic pushover curve obtained when a non-linear static pushover analysis is performed, thus, becoming a computationally efficient substitute for the costly FEM model. The project includes a parametric FEM model of a RC shear wall in OpenSeesPy, the training and validation of the DNN model in TensorFlow, and an application with an interactive Graphical User Interface (GUI) to test the methodology and visualize the results.

# Run the interactive GUI application
To open an interactive GUI to test and visualize the surrogate model results, run the file "AppGUI.py"

# FEM model quick Test
To perform a quick test with the FEM model using OpenSeesPy, run the file "RunValidationExample.py"
