# Abstract
A data-driven surrogate model for analyzing RC shear walls is developed using Deep Neural Networks (DNN). The surrogate model is trained with thousands of FEM simulations to predict the characteristic curve obtained when a static non-linear pushover analysis is performed. The surrogate model is extensively tested and found to exhibit a high degree of accuracy in its predictions while being extremely faster than the FEM analysis. In addition to the presented methodology, the complete code and framework that made this study possible is provided as an open-source project with the intention to bridge the gap between research and practical application of Machine Learning powered techniques in Structural Engineering. The project is developed on Python and includes a parametric FEM model of an RC shear wall in OpenSeesPy, the training and validation of the DNN model in TensorFlow, and an application with an interactive Graphical User Interface to test the methodology and visualize the results. 

# Run the interactive GUI application
To open an interactive GUI to test and visualize the surrogate model results, run the file "AppGUI.py"

# FEM model quick Test
To perform a quick test with the FEM model using OpenSeesPy, run the file "RunValidationExample.py"

# About
Development: Ph.D. Candidate German Solorzano (sr.german90@gmail.com)

Supervision: Dr. Vagelis Plevris (vplevris@gmail.com)

Sponsored:  Oslo Metropolitan University (OsloMet), Department of Civil Engineering and Energy Technology, Oslo, Norway.
