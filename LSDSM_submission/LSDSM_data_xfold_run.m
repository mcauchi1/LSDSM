% Performing a simulation from saved data

clear all;
close all;
rng(31);

% Find the parent folder
mydir  = pwd;
idcs   = strfind(mydir,'\');
newdir = mydir(1:idcs(end)-1); % go to the parent folder

allow_plots = 0; % if =1, then plot patient biomarker
% if we are plotting patient biomarkers, this would be the maximum number of plots
no_of_plots = 5; 

Delta = 6; % time step size (in months)

% Controls
% Controls
csv_controls.base_cov_col_no = 5; % start of column number of baseline covariates
csv_controls.bio_long_col_no = 8; % start of column number of longitudinal biomarkers
csv_controls.norm_bool = 0; % boolean to normalise all biomarkers
csv_controls.Delta = Delta; % time step size (in months)
csv_controls.Delta_SJM = 6; % time step size of the performance metrics for the standard JM
csv_controls.t_multiplier = 12; % survival time (in months) = data_observed.surv_time * csv_controls.t_multiplier
csv_controls.censor_time = 15 * csv_controls.t_multiplier; % (in months)
csv_controls.allow_plots = 1; % shows plots of the observed longitudinal biomarkers
csv_controls.no_of_plots = 5; % maximum number of plots to show
csv_controls.train_test = 'train'; % training and testing data sets are handled differently in the longitudinal data

% dimensions
dim_size.states = 2; % number of hidden states in SSM
dim_size.dyn_states = 1; % number of dynamic states in SSM
dim_size.y = 1; % number of biomarkers considered (observations)
% number of baseline covariates (the +1 is for the intercept)
dim_size.base_cov = csv_controls.bio_long_col_no - csv_controls.base_cov_col_no + 1;

landmark_t_arr = [5.5 7.5 9.5] * csv_controls.t_multiplier;
horizon_arr = [1, 2, 5] * csv_controls.t_multiplier;

no_of_kfold = 5;

% Cell to store the performance metrics for every horizon
perf_metrics_df = cell(1,length(horizon_arr));

% Create a performance metrics dataframe to compare directly with R code
% Number of rows is equivalent to number of landmarks to test
% Number of columns is equivalent to 3 (BS, PE, AUC)
for horiz_idx=1:length(horizon_arr)
    perf_metrics_df{horiz_idx} = zeros(length(landmark_t_arr) * no_of_kfold, 3);
end
    
%% 1. Retrieve training data and organise into arrays
for kfold=1:no_of_kfold
    
    % start from row 2 and column 1
    newdir_train = strcat(newdir, '\Data\PBC_data\PBC_dataset_train_fold', num2str(kfold), '.csv');
    M_train = xlsread(newdir_train);
    
    [data_observed, csv_controls] = LSDSM_ALLFUNCS.read_from_csv(M_train, dim_size, csv_controls);


    %% 2. Train the model

    % Initialisation of parameters
    model_coef_init = LSDSM_ALLFUNCS.initialise_params(dim_size, data_observed, Delta);

    C_init = [1 0]; % Keep C fixed
    % Control which parameters to keep fixed by replacing NaN with a matrix
    fixed_params = struct('A', NaN, 'C', C_init, ...
                          'Gamma', NaN, 'Sigma', NaN, ...
                          'g_s', NaN, 'a_s', NaN, ...
                          'mu_0', NaN, 'V_0', NaN);

    % EM controls
    controls.init_params = model_coef_init;
    controls.fixed_params = fixed_params;
    controls.EM_iters = 600; % number of iterations for the EM algorithm
    controls.mod_KF = true; % 1 means utilise survival information in the modified KF
    controls.verbose = true; % If true, it will provide feedback while the algorithm is executing
    controls.allow_plots = true; % If true, it will plot the log likelihood over EM iterations and some patient data
    controls.max_param_diff = 1e-4; % stopping criterion - stop if difference in all parameters < this value

    [model_coef_est, max_iter_reached, param_traj, RTS_traj] = ...
                    LSDSM_ALLFUNCS.LSDSM_EM(dim_size, data_observed, controls, csv_controls.censor_time);

    if allow_plots
        t_all = linspace(0,max_t*csv_controls.t_multiplier, max_t*csv_controls.t_multiplier/csv_controls.Delta+1);
        for i=1:no_of_plots
            mu_tilde_arr = RTS_traj.mu_tilde(:,:,:,i);
            mu_hat_arr = RTS_traj.mu_hat(:,:,:,i);

            for j=1:dim_size.states
                figure;
                % if j <= no_of_y
                if j <= dim_size.y
                    scatter(t_all, data_observed(i).y(j,:));
                else
                    scatter(t_all, data_observed(i).y(mod(j-1, dim_size.states - dim_size.dyn_states)+1,:));
                end
                hold on;
                % end
                % plot(t_all, squeeze(mu_pred_temp_arr(j,1,:,i))); % one step forward prediction
                plot(t_all, squeeze(mu_tilde_arr(j,1,:)));
                plot(t_all, squeeze(mu_hat_arr(j,1,:)));
                if j <= dim_size.y
                    legend('y', '\mu_{tilde}', '\mu_{hat}');
                else
                    legend('y', '\mu_{tilde}', '\mu_{hat}')
                end
                if data_observed(i).delta_ev == 0
                    title_n = sprintf('Patient was censored at time %.1f months: State %d', data_observed(i).surv_time, j);
                else
                    title_n = sprintf('Patient died at time %.1f months: State %d', data_observed(i).surv_time, j);
                end
                title(title_n)
                if norm_bool
                    ylim([0 1]);
                else
                    ylim([0 normalise_const(mod(j-1, dim_size.states - dim_size.dyn_states)+1)]);
                end
            end

        end
    end

    % Plot the first entry of every parameter across EM iterations
    figure;
    plot(1:max_iter_reached, squeeze(param_traj.A(1,1,1:max_iter_reached)));
    hold on;
    plot(1:max_iter_reached, squeeze(param_traj.C(1,1,1:max_iter_reached)));
    plot(1:max_iter_reached, squeeze(param_traj.Gamma(1,1,1:max_iter_reached)));
    plot(1:max_iter_reached, squeeze(param_traj.Sigma(1,1,1:max_iter_reached)));
    plot(1:max_iter_reached, squeeze(param_traj.g_s(1,1,1:max_iter_reached)));
    plot(1:max_iter_reached, squeeze(param_traj.a_s(1,1,1:max_iter_reached)));
    legend('A_{11}', 'C_{11}', '\Gamma_{11}', '\Sigma_{11}', '\gamma_{1}', '\alpha_{1}');
    xlabel('EM iteration');
    ylabel('Parameter Values');


    %% 3. Retrieve the testing data set and deduce Performance Metrics
    model_coef_test = model_coef_est;

    % Load test data set
    newdir_test = strcat(newdir, '\Data\PBC_data\PBC_dataset_test_fold', num2str(kfold), '.csv');
    M_test = xlsread(newdir_test);

    csv_controls.train_test = 'test';

    no_of_t_points = csv_controls.censor_time / Delta;

    mult_bs_test = zeros(length(landmark_t_arr), no_of_t_points);
    mult_pe_test = zeros(length(landmark_t_arr), no_of_t_points);
    mult_auc_test = zeros(length(landmark_t_arr), no_of_t_points);

    % Landmark is used for the testing data, where we predict from this point
    % onwards the hidden state values and the respective survival values
    for l_var=1:length(landmark_t_arr)
        landmark_t = landmark_t_arr(l_var); % use data up to this value (in months)
        % +1 due to index starts from 1
        csv_controls.landmark_idx = int64(landmark_t / csv_controls.Delta) + 1;

        [test_data_observed, csv_controls] = LSDSM_ALLFUNCS.read_from_csv(M_test, dim_size, csv_controls);

        %% 4a. Performance Metrics - Time-dependent Brier Score

        % +1 since index starts from 1
        t_est_idx = floor(csv_controls.censor_time / csv_controls.Delta)+1;

        % Forecasts
        pat_data_reduced = LSDSM_ALLFUNCS.forecast_fn(test_data_observed, landmark_t, t_est_idx, ...
                                                      csv_controls.censor_time, model_coef_est, controls);

        % find the longest interval to work out the performance metrics
        max_horizon_t = csv_controls.censor_time - landmark_t;

        [bs_test] = LSDSM_ALLFUNCS.Brier_Score_fn(pat_data_reduced, landmark_t, max_horizon_t, model_coef_est);

        %% 4b. Performance Metrics - Time-dependent Prediction Error
        [pe_test] = LSDSM_ALLFUNCS.Prediction_Error_fn(pat_data_reduced, landmark_t, max_horizon_t, model_coef_est);

        %% 4c. Performance Metrics - AUC

        [auc_test] = LSDSM_ALLFUNCS.AUC_fn(pat_data_reduced, landmark_t, max_horizon_t, model_coef_est);

        %% 4e. Performance Metrics - Store results in arrays
        for horiz_idx=1:length(horizon_arr)
            horizon_to_test = horizon_arr(horiz_idx);
            horizon_test_idx = ceil(horizon_to_test / Delta);
            perf_metrics_df{horiz_idx}((kfold-1) * length(landmark_t_arr) + l_var,:) = ...
                [bs_test(horizon_test_idx), pe_test(horizon_test_idx), auc_test(horizon_test_idx)];
        end

    end
end









