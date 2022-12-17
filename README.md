# Linear State space Dynamic Survival Model
 Linear State space Dynamic Survival Model (LSDSM) is a joint model for longitudinal and survival data, having the longitudinal sub-process as a state space model. This framework has been created on MATLAB. This repository also contains some work on R, more specifically, the model created by Dimitris Rizopoulos was used for comparison: https://github.com/drizopoulos/JMbayes. The proposed model provides an alternative to the standard joint model by introducing a dynamical relationship in the longitudinal sub-process instead of time-basis functions to model the biomarker trajectories.

 ## Prerequisites
  The model was created using MATLAB R2022b. Thus, a recent MATLAB version may be required for proper execution of the algorithm.

  For the standard joint model on R, some packages are required for proper execution. To install these:
  ```r
  install.packages("splines")
  install.packages("JMbayes")
  install.packages("dplyr")
  install.packages("tidyverse")
  install.packages("stringr")
  # if using rstudio
  install.packages(rstudioapi)
  ```

  If you are not using rstudio, then you will need to direct the program to the right path to retrieve the data and to store the outputs.

 ## Contents
  This repository has 3 folders in the time of writing, being _Data_, _LSDSM_, and _R_. The former holds the data that was created by the R files to compare the models. It also contains some results. 
 
  _LSDSM_ folder contains 6 _m_ files at the time of writing:
  - *LSDSM_ALLFUNCS.m* - Contains all the functions required within a class to execute the LSDSM estimation algorithm and other analysis, such as plotting and performance evaluation.
  - *LSDSM_data_single_run.m* - After creating a training and data set, this file can extract the _csv_ data and execute the LSDSM estimation algorithm. The _csv_ file is expected to be in a certain format, where the columns are ordered as _ID_, _survival time_, _event indicator_, _time of capturing longitudinal biomarkers_, _sequence of baseline covariates_, and _sequence of longitudinal biomarkers_. Some configuration is required within this _m_ file to ensure that the data is properly loaded in MATLAB.
  - *LSDSM_data_single_run_compared.m* - Similar to the above, but comparing LSDSM with the standard joint model generated from R.
  - *LSDSM_data_xfold_run_compared.m* - Similar to the above, but requires the _csv_ data files to be already segmented. This was done by the R files. This file executes the LSDSM code *_x_* times, and outputs the performance metrics according to the set configuration.
  - *LSDSM_sim_single_run.m* - Creates a simulation from the LSDSM framework, and executes the estimation algorithm.
  - *LSDSM_sim_multiple_runs.m* - Creates multiple simulations from the LSDSM framework, and executes the estimation algorithm. These simulations can check for performance against increasing number of patients.
  
  _R_ folder contains 2 _R_ files:
  - *JM_PBC_analysis.R* - _JMBayes_ package stores the PBC data set. Hence, this file splits the data into training and testing data sets, which will be made available for the MATLAB files. Also, it estimates the standard joint model using a Bayesian approach, and computes the performance metrics across the chosen landmarks and for a number of horizons and stores them into _csv_ files.
  - *JM_PBC_analysis_xfold.R* - Similar to the above file, but instead of splitting the data once, it splits it according to the requested number of folds. It stores the _csv_ files for the MATLAB file to use. It also computes the performance metrics and stores them into _csv_ files.

 ## How to use
 If you have a specific data set that contains longitudinal and survival data, it is recommended that you use *LSDSM_data_single_run.m* as a starting point. Save the data into the requested format (look at the available _csv_ files for inspiration) in the _Data_ folder. In the _m_ file, make sure that the file directory is chosen properly. Also, set the configuration parameters of your choosing. The model scales accordingly with the set parameters. These include setting `csv_controls` to extract the data as requested, `dim_size` to specify the size of the model, and `landmark_t_arr` to choose the landmarks to test.

 ## Versioning
 This is the first version of LSDSM being published. Future work may follow, including:
 - Introducing mixture models within LSDSM to learn different patient clusters; and
 - Optimisation of the code for faster execution.

 ## Authors
 - Mark Cauchi - Main Author

 ## License
 This project is licensed under the Apache-2.0 License - see the LICENSE.md file for details.

 ## Acknowledgements
 - The University of Sheffield for funding this project; and
 - [Dimitris Rizopoulos](https://github.com/drizopoulos) for making the standard joint model framework available for R and easy to utilise.