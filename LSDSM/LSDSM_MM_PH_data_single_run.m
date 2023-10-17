% Performing a simulation from saved data

clear all;
close all;
rng(31);

%% 1. Retrieve training data and organise into arrays

% Load data
mydir  = pwd;
idcs   = strfind(mydir,'\');
newdir = mydir(1:idcs(end-1)-1); % go to the parent parent folder
newdir_train = strcat(newdir, '\Data\PH_shef_clean\Anon\PH_clean_dataset_w_sex_train_modified_join_whoFC3-4.csv');
M_train = readtable(newdir_train); % store data in table

ignore_all_NaNs = 0; % if 1, then when making predictions, we ignore those patients that have no observed values

allow_plots = 0; % if =1, then plot patient biomarker
% if we are plotting patient biomarkers, this would be the maximum number of plots
no_of_plots = 5; 
Delta = 6; % Time step for the state space model

% H_mat is the matrix that defines the linear combination of hidden states
% to affect the hazard function
H_mat = [1 0];

% Controls
csv_controls.base_cov_col_no = 6; % start of column number of baseline covariates
csv_controls.bio_long_col_no = 9; % start of column number of longitudinal biomarkers
csv_controls.norm_bool = 1; % boolean to normalise all biomarkers
csv_controls.Delta = Delta; % time step size (in months)
csv_controls.t_multiplier = 1; % survival time (in months) = data_observed.surv_time * csv_controls.t_multiplier
csv_controls.censor_time = 10 * 12; % csv_controls.t_multiplier; % (in months)
csv_controls.allow_plots = allow_plots; % shows plots of the observed longitudinal biomarkers
csv_controls.no_of_plots = 5; % maximum number of plots to show
csv_controls.train_test = 'train'; % training and testing data sets are handled differently in the longitudinal data

% dimensions
dim_size.states = 2; % number of hidden states in SSM
dim_size.dyn_states = 1; % number of dynamic states in SSM
dim_size.y = 1; % number of biomarkers considered (observations)
% number of baseline covariates (the +1 is for the intercept)
dim_size.base_cov = csv_controls.bio_long_col_no - csv_controls.base_cov_col_no + 1;
dim_size.alpha_eqns = size(H_mat,1); % number of associations between linear combinations of hidden states and hazard function
dim_size.class_cov = 1;

% Testing variables
landmark_t_arr = [1 2 3 4] * 12; %csv_controls.t_multiplier;

data_observed = LSDSM_MM_ALLFUNCS.read_from_csv(M_train, dim_size, csv_controls);

%% 2. Train the model

no_of_classes = 1; % Number of models to train

% data_controls.MM = 'None'; % 'altMM'
% data_controls.base_haz = 'None'; % 'Weibull'

% Initialisation and fixation of C
C_init = [eye(dim_size.y), zeros(dim_size.y, dim_size.states - dim_size.y)]; % Observation matrix - assumed known

% extract first biomarker value for every patient
num_pats = double(data_observed.Count);
y_arr = LSDSM_MM_ALLFUNCS.extract_field_from_map(data_observed, 'y');

y_init_mat = reshape(y_arr(:,:,1,:), [size(y_arr,1), num_pats]);
% if number of hidden states is a multiple of number of observations
if mod(dim_size.states, dim_size.y) == 0 
    mu_0_init = repmat(nanmean(y_init_mat, 2), dim_size.states / dim_size.y, 1);
else % if one observation has more lagging states associated with it
    temp_mu = nanmean(y_init_mat, 2);
    mu_0_init = [repmat(temp_mu, floor(dim_size.states / dim_size.dyn_states), 1);
                 temp_mu(1:mod(dim_size.states, dim_size.dyn_states))];
end

% Calculate the fraction of missing data
num_obs = LSDSM_MM_ALLFUNCS.extract_field_from_map(data_observed, 'm_i')';
sum_m_i = 0;
for i=1:num_pats
    sum_m_i = sum_m_i + sum(all(~isnan(y_arr(:,:,1:num_obs(i),i))));
end

frac_not_miss = sum_m_i / sum(num_obs);
frac_miss = 1 - frac_not_miss;

% Control which parameters to keep fixed by replacing NaN with a matrix
fixed_params_tmp = struct('A', NaN, 'B', NaN, 'C', C_init, ...
                      'W', NaN, 'V', NaN, ... % 
                      'g_s', NaN, 'a_s', NaN, ...
                      'mu_0', NaN, 'W_0', NaN); % 

fixed_params = containers.Map('KeyType', 'int32', 'ValueType', 'any');
for g=1:no_of_classes
    fixed_params(g) = fixed_params_tmp;
end

% Control which parameters to keep the same across models
same_params = struct('A', 0, 'B', 0, 'C', 0, ...
                      'W', 0, 'V', 0, ...
                      'g_s', 0, 'a_s', 0, ...
                      'mu_0', 0, 'W_0', 0);

%%% EM controls %%%
controls.fixed_params = fixed_params;
controls.same_params = same_params;
controls.EM_iters = 600; % number of iterations for the EM algorithm
controls.mod_KF = true; % 1 means utilise survival information in the modified KF
controls.verbose = true; % If true, it will provide feedback while the algorithm is executing
controls.allow_plots = true; % If true, it will plot the log likelihood over EM iterations and some patient data
controls.max_param_diff = 5e-4; % stopping criterion - stop if difference in all parameters < this value
controls.base_haz = 'None'; % 'Weibull';
controls.mod_KF = true; % 1 means utilise survival information in the modified KF
controls.joseph_form = false; % 1 means that we use Joseph's form in the filter stage
controls.do_EM_better_init_params = false;

% Initialise the parameters by running EM algorithms for every class on a
% small subset of the population
model_coef_init = LSDSM_MM_ALLFUNCS.better_init_params(dim_size, data_observed, fixed_params, Delta, H_mat, ...
                                                        csv_controls.censor_time, no_of_classes, controls);
                                                    
controls.init_params = model_coef_init; % set initial parameters

% Run the EM algorithm for all combined classes
[model_coef_est, max_iter_reached, param_traj, RTS_traj, E_c_ig_allpats] = ...
                LSDSM_MM_ALLFUNCS.LSDSM_MM_EM(dim_size, data_observed, controls, csv_controls.censor_time);


%% 3. Retrieve the testing data set and deduce Performance Metrics
model_coef_test = model_coef_est;
model1 = model_coef_est(1);

% Load test data set
newdir_test = strcat(newdir, '\Data\PH_shef_clean\Anon\PH_clean_dataset_w_sex_test_modified_join_whoFC3-4.csv');
M_test = readtable(newdir_test);

csv_controls.train_test = 'test';

%% 4. Analyse Performance on testing data set

compare_w_risk_score = 0; % comparison with risk score requires different storage of patient information
no_of_t_points = csv_controls.censor_time / Delta;

% Initialise arrays for time-dependent Brier score, prediction error, and area under ROC curve
mult_bs_test = zeros(length(landmark_t_arr), no_of_t_points);
mult_pe_test = zeros(length(landmark_t_arr), no_of_t_points);
mult_auc_test = zeros(length(landmark_t_arr), no_of_t_points);

auc_thresh_test = zeros(length(landmark_t_arr), 1);

% Landmark is used for the testing data, where we forecast from this point
% onwards the hidden state values and the respective survival values
for l_var=1:length(landmark_t_arr)
    landmark_t = landmark_t_arr(l_var); % use data up to this value (in months)
    csv_controls.landmark_idx = int64(landmark_t / csv_controls.Delta);
    
    horiz_int = 12; % include patient if they were not censored before 12 months after landmark
    
    if compare_w_risk_score
        range_of_int = [landmark_t-2 landmark_t];
        
        test_data_observed = LSDSM_MM_ALLFUNCS.read_from_csv_risk_score(M_test, dim_size, csv_controls, range_of_int, ...
                                                                             landmark_t, horiz_int);
    else
        test_data_observed = LSDSM_MM_ALLFUNCS.read_from_csv(M_test, dim_size, csv_controls);
    end

    %% 4a. Performance Metrics - Forecasts
    t_est_idx = floor(csv_controls.censor_time / Delta); % maximum forecast time
                                      
    % Forecasts (longitudinal and survival) using the proper distributions
    pat_data_reduced = LSDSM_MM_ALLFUNCS.forecast_fn(test_data_observed, landmark_t, t_est_idx, csv_controls.censor_time, ...
                                                                     dim_size, model_coef_est, controls, ignore_all_NaNs);

    % find the longest interval to work out the performance metrics
    max_horizon_t = csv_controls.censor_time - landmark_t;

    %% 4b. Performance Metrics - Time-dependent Brier Score
    % Brier score for the forecast function using the proper distributions
    [bs_test] = LSDSM_MM_ALLFUNCS.Brier_Score_fn(pat_data_reduced, landmark_t, max_horizon_t, model_coef_est(1).DeltaT);
    mult_bs_test(l_var, 1:length(bs_test)) = bs_test;

    %% 4c. Performance Metrics - Time-dependent Prediction Error
    % Prediction error for the forecast function using the proper distributions
    [pe_test] = LSDSM_MM_ALLFUNCS.Prediction_Error_fn(pat_data_reduced, landmark_t, max_horizon_t, model_coef_est(1).DeltaT);
    mult_pe_test(l_var, 1:length(pe_test)) = pe_test;

    %% 4d. Performance Metrics - AUC
    % AUC for the forecast function using the proper distributions
    [auc_test] = LSDSM_MM_ALLFUNCS.AUC_fn(pat_data_reduced, landmark_t, max_horizon_t, model_coef_est(1).DeltaT);
    mult_auc_test(l_var, 1:length(auc_test)) = auc_test;
    
    %% 4e. Plot results and some survival prediction plots
    
    LSDSM_MM_ALLFUNCS.plot_bs_pe_auc(pat_data_reduced, bs_test, pe_test, ...
                                     auc_test, landmark_t, csv_controls.censor_time, csv_controls.t_multiplier, Delta)

    if allow_plots
        for ii=1:no_of_plots
            LSDSM_MM_ALLFUNCS.plot_forecast_surv(pat_data_reduced(ii), landmark_t, t_all, Delta)
        end
    end
    
    % AUC by thresholding survival predictions
    delta_evs = LSDSM_MM_ALLFUNCS.extract_field_from_map(pat_data_reduced, 'delta_ev');
    surv_times = LSDSM_MM_ALLFUNCS.extract_field_from_map(pat_data_reduced, 'surv_time');
    
    % Time and index of prediction
    t_int = landmark_t + horiz_int;
    t_idx = t_int / model_coef_est(1).DeltaT;
    
    % true event = 1 if patient experienced event during horizon
    true_event = (delta_evs & surv_times < t_int);
    
    % obtain survival predictions
    pred_surv_arr = LSDSM_MM_ALLFUNCS.extract_field_from_map(pat_data_reduced, 'pred_surv');
    
    % take survival prediction at time index of interest
    pred_surv = reshape(pred_surv_arr(1,t_idx,:), size(true_event));
    
    % Create a logical mask for the elements to exclude - exclude censored
    % patients during horizon
    idx_to_exclude = find(delta_evs == 0 & surv_times < t_int);
    
    excludeMask = true(size(true_event));
    excludeMask(idx_to_exclude) = false;
    
    % keep only those that experience the event within horizon or are
    % monitored for longer than the horizon
    true_event = true_event(excludeMask);
    pred_surv = pred_surv(excludeMask);
    
    thresh_vec = 0:0.01:1; % threshold vector
    
    % confusion matrix
    temp_df = LSDSM_MM_ALLFUNCS.buildConfMat(thresh_vec, pred_surv, true_event);
    
    % plot ROC curve
    figure;
    plot(1-temp_df.specificity, temp_df.sensitivity);
    xlabel('1 - Specificity');
    ylabel('Sensitivity');
    title(sprintf('ROC curve for landmark at %d months and horizon of %d months', landmark_t, horiz_int));
    
    % calculate AUC from ROC curve
    auc = LSDSM_MM_ALLFUNCS.calcAUC(temp_df);
    
    % store AUC
    auc_thresh_test(l_var) = auc;
end

% plot the AUC figures in 2x2 subplots
figure;
for l_var=1:length(landmark_t_arr)
    subplot(2,2,l_var);
    plot(Delta:Delta:10*Delta, mult_auc_test(l_var, 1:10));
    xlabel('Horizon (Months)');
    ylabel('AUC');
    title(sprintf('AUC across several horizons for landmark = %d months', landmark_t_arr(l_var)));
    grid on;
    ylim([0.6 0.95]);
end
sgtitle('AUC across several landmarks and horizons');
