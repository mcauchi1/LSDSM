# Linear State space Dynamic Survival Model
 Linear State space Dynamic Survival Model (LSDSM) is a state space model that jointly models longitudinal and survival data. This framework has been created on MATLAB. This repository also contains some work in R, more specifically to split the data into training and testing sets, and to compare with a risk score: http://dx.doi.org/10.1016/j.healun.2017.04.008. The proposed model provides an alternative to risk modelling by introducing a dynamical relationship in the longitudinal sub-process to model the biomarker trajectories. The data used for this paper is available on The University of Sheffield's online data repository (ORDA).

 ## Prerequisites
  The model was created using MATLAB R2022b. Thus, a recent MATLAB version may be required for proper execution of the algorithm.

  For the risk score model in R, some packages are required for proper execution. To install these:
  ```r
  install.packages("dplyr")
  install.packages("tidyverse")
  install.packages("stringr")
  install.packages("lattice")
  # if using rstudio
  install.packages(rstudioapi)
  ```

  If you are not using rstudio, then you will need to direct the program to the right path to retrieve the data and to store the outputs.

 ## Contents
  This repository has 3 folders in the time of writing, being _Data_, _LSDSM_, and _R_. 

  _Data_ folder contains 1 _csv_ file at the time of writing:
  - *PBC_dataset.csv* - Contains an example of how the _csv_ files should be stored based on the PBC dataset. This also shows an example of how data should be stored for LSDSM - ID, event/survival time, binary variable indicating event happened, time of recording longitudinal data, baseline information, longitudinal information.
  _LSDSM_ folder contains 1 folder and 5 _m_ files at the time of writing:
  - *sims_t30* - Contains the results obtained from the simulations at different missing data percentages.
  - *LSDSM_MM_ALLFUNCS.m* - Contains all the functions required within a class to execute the LSDSM estimation algorithm and other analysis, such as plotting and performance evaluation.
  - *LSDSM_MM_PH_data_single_run.m* - After creating a training and data set, this file can extract the _csv_ data and execute the LSDSM estimation algorithm. The _csv_ file is expected to be in a certain format, where the columns are ordered as _ID_, _survival time_, _event indicator_, _time of capturing longitudinal biomarkers_, _sequence of baseline covariates_, and _sequence of longitudinal biomarkers_. Some configuration is required within this _m_ file to ensure that the data is properly loaded in MATLAB.
  - *LSDSM_multiple_runs_sims.m* - Creates multiple simulations from the LSDSM framework, and executes the estimation algorithm. These simulations can check for performance against increasing missing data percentages and number of patients if required.
  - *all_sims_analysis.m* - Compares the results from the multiple runs simulations.
  - *legendUnq.m* - A file to properly label the parameters estimation plot during the EM execution. [Adam Danz (2023). legendUnq (https://www.mathworks.com/matlabcentral/fileexchange/67646-legendunq), MATLAB Central File Exchange. Retrieved October 17, 2023.]
  
  _R_ folder contains 1 _R_ file:
  - *data_and_risk_score.R* - This file splits the data into training and testing data sets, which will be made available for the MATLAB files. Also, it estimates the risk scores using the methods proposed by the above paper.

 ## How to use
 If you have a specific data set that contains longitudinal and survival data, it is recommended that you use *LSDSM_MM_PH_data_single_run.m* as a starting point. Save the data into the requested format (look at the available _csv_ files for inspiration) in the _Data_ folder. In the _m_ file, make sure that the file directory is chosen properly. Also, set the configuration parameters of your choosing. The model scales accordingly with the set parameters. These include setting `csv_controls` to extract the data as requested, `dim_size` to specify the size of the model, `controls` for the EM algorithm controls, and `landmark_t_arr` to choose the landmarks to test.

 ## Versioning
 This is the first version of LSDSM being published. Future work may follow, including:
 - Optimisation of the code for faster execution; and
 - Introducing nonlinearities.

 ## Authors
 - Mark Cauchi - Main Author

 ## License
 This project is licensed under the Apache-2.0 License - see the LICENSE.md file for details.

 ## Acknowledgements
 - The University of Sheffield for funding this project.