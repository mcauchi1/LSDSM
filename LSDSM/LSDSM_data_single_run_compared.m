% Performing a simulation from saved data

clear all;
close all;
rng(31);

% Load results obtained using standard Joint Model
mydir  = pwd;
idcs   = strfind(mydir,'\');
newdir = mydir(1:idcs(end)-1); % go to the parent folder
% R_landmark7_5 = xlsread('Data/PBC_dataset/Results/landmark7_5_several_horiz.csv');
% R_landmark7_5_rand_int = ...
%     xlsread('Data/PBC_dataset/Results/landmark7_5_several_horiz_rand_int_only.csv');
R_landmarks{1} = xlsread(strcat(newdir, '/Data/PBC_data/Results/landmark5_5_several_horiz.csv'));
R_landmarks_rand_int{1} = ...
    xlsread(strcat(newdir, '/Data/PBC_data/Results/landmark5_5_several_horiz_rand_int_only.csv'));
R_landmarks{2} = xlsread(strcat(newdir, '/Data/PBC_data/Results/landmark7_5_several_horiz.csv'));
R_landmarks_rand_int{2} = ...
    xlsread(strcat(newdir, '/Data/PBC_data/Results/landmark7_5_several_horiz_rand_int_only.csv'));
R_landmarks{3} = xlsread(strcat(newdir, '/Data/PBC_data/Results/landmark9_5_several_horiz.csv'));
R_landmarks_rand_int{3} = ...
    xlsread(strcat(newdir, '/Data/PBC_data/Results/landmark9_5_several_horiz_rand_int_only.csv'));

%% 1. Retrieve training data and organise into arrays

% Load data
mydir  = pwd;
idcs   = strfind(mydir,'\');
newdir = mydir(1:idcs(end)-1); % go to the parent folder
newdir_train = strcat(newdir, '\Data\PBC_data\PBC_dataset_train.csv');
M_train = xlsread(newdir_train);

allow_plots = 0; % if =1, then plot patient biomarker
% if we are plotting patient biomarkers, this would be the maximum number of plots
no_of_plots = 5; 
Delta = 6;

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

% Testing variables
landmark_t_arr = [5.5 7.5 9.5] * csv_controls.t_multiplier;
horizon_to_test = 2 * csv_controls.t_multiplier;

data_observed = LSDSM_ALLFUNCS.read_from_csv(M_train, dim_size, csv_controls);


%% 2. Train the model

% https://onlinelibrary.wiley.com/action/downloadSupplement?doi=10.1002%2Fbimj.201600238&file=bimj1810-sup-0002-SuppMat.pdf

% Initialisation of parameters
model_coef_init = LSDSM_ALLFUNCS.initialise_params(dim_size, data_observed, Delta);

C_init = [1 0]; % Keep C fixed
% Control which parameters to keep fixed by replacing NaN with a matrix
fixed_params = struct('A', NaN, 'C', C_init, ...
                      'W', NaN, 'V', NaN, ...
                      'g_s', NaN, 'a_s', NaN, ...
                      'mu_0', NaN, 'W_0', NaN);

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
plot(1:max_iter_reached, squeeze(param_traj.W(1,1,1:max_iter_reached)));
plot(1:max_iter_reached, squeeze(param_traj.V(1,1,1:max_iter_reached)));
plot(1:max_iter_reached, squeeze(param_traj.g_s(1,1,1:max_iter_reached)));
plot(1:max_iter_reached, squeeze(param_traj.a_s(1,1,1:max_iter_reached)));
legend('A_{11}', 'C_{11}', 'W_{11}', 'V_{11}', '\gamma_{1}', '\alpha_{1}');
xlabel('EM iteration');
ylabel('Parameter Values');



%% 3. Retrieve the testing data set and deduce Performance Metrics
model_coef_test = model_coef_est;

% Load test data set
newdir_test = strcat(newdir, '\Data\PBC_data\PBC_dataset_test.csv');
M_test = xlsread(newdir_test);

csv_controls.train_test = 'test';

no_of_t_points = csv_controls.censor_time / Delta;

% Create a performance metrics dataframe to compare directly with R code
% Number of rows is equivalent to number of landmarks to test
% Number of columns is equivalent to 3 (BS, PE, AUC)
perf_metrics_df = zeros(length(landmark_t_arr), 3);
horizon_test_idx = ceil(horizon_to_test / csv_controls.Delta);

mult_bs_test = zeros(length(landmark_t_arr), no_of_t_points);
mult_pe_test = zeros(length(landmark_t_arr), no_of_t_points);
mult_auc_test = zeros(length(landmark_t_arr), no_of_t_points);

% Landmark is used for the testing data, where we predict from this point
% onwards the hidden state values and the respective survival values
for l_var=1:length(landmark_t_arr)
    landmark_t = landmark_t_arr(l_var); % use data up to this value (in months)
    % +1 due to index starts from 1
    csv_controls.landmark_idx = int64(landmark_t / csv_controls.Delta) + 1;
    
    test_data_observed = LSDSM_ALLFUNCS.read_from_csv(M_test, dim_size, csv_controls);

    %% 4a. Performance Metrics - Time-dependent Brier Score

    % +1 since index starts from 1
    t_est_idx = floor(csv_controls.censor_time / csv_controls.Delta)+1;

    % Forecasts
	pat_data_reduced = LSDSM_ALLFUNCS.forecast_fn(test_data_observed, landmark_t, t_est_idx, ...
                                                  csv_controls.censor_time, model_coef_est, controls);

    % find the longest interval to work out the performance metrics
    max_horizon_t = csv_controls.censor_time - landmark_t;

    [bs_test] = LSDSM_ALLFUNCS.Brier_Score_fn(pat_data_reduced, landmark_t, max_horizon_t, model_coef_est);
	mult_bs_test(l_var, 1:length(bs_test)) = bs_test;
    
    %% 4b. Performance Metrics - Time-dependent Prediction Error
    [pe_test] = LSDSM_ALLFUNCS.Prediction_Error_fn(pat_data_reduced, landmark_t, max_horizon_t, model_coef_est);
    
    mult_pe_test(l_var, 1:length(pe_test)) = pe_test;

    %% 4c. Performance Metrics - AUC

    [auc_test] = LSDSM_ALLFUNCS.AUC_fn(pat_data_reduced, landmark_t, max_horizon_t, model_coef_est);
    
	mult_auc_test(l_var, 1:length(auc_test)) = auc_test;
    
    perf_metrics_df(l_var,:) = [bs_test(horizon_test_idx), pe_test(horizon_test_idx), auc_test(horizon_test_idx)];

    %% 5. Plot some Survival prediction plots
    
    reduced_num_pats = double(pat_data_reduced.Count);
    surv_info_mat = zeros(reduced_num_pats, 2); % surv time and delta matrix

    for ii=1:reduced_num_pats
        surv_info_mat(ii,:) = [pat_data_reduced(ii).surv_time, pat_data_reduced(ii).delta_ev];
    end

    hist_count_arr = zeros(1, length(auc_test));
    hist_count_arr_censored = zeros(1, length(auc_test));

    for ii=1:length(hist_count_arr)
        hist_count_arr(ii) = sum(surv_info_mat(surv_info_mat(:,1) < (landmark_t + csv_controls.Delta * ii), 2));
        hist_count_arr_censored(ii) = length(surv_info_mat(surv_info_mat(:,1) > (landmark_t + csv_controls.Delta * ii),2));
    end
    
    curr_results_r = R_landmarks{l_var};
    curr_results_r_rand_int = R_landmarks_rand_int{l_var};
    
    max_horizon = 60;
    no_of_perf_pts = max_horizon / csv_controls.Delta;
    no_of_perf_jm_pts = max_horizon / csv_controls.Delta_SJM;
    
    if l_var == 3 % exceeds censoring time in standard JM
        no_of_perf_pts = no_of_perf_pts - 1;
        no_of_perf_jm_pts = no_of_perf_jm_pts - 1;
    end
    perf_t_pts = csv_controls.Delta:csv_controls.Delta:csv_controls.Delta*no_of_perf_pts;
    perf_jm_t_pts = csv_controls.Delta_SJM:csv_controls.Delta_SJM:csv_controls.Delta_SJM * no_of_perf_jm_pts;
    
    curr_results_r = curr_results_r(:,1:no_of_perf_jm_pts);
    curr_results_r_rand_int = curr_results_r_rand_int(:,1:no_of_perf_jm_pts);
        
    figure;
    t = tiledlayout(4, 1);
    nexttile;
    plot(perf_t_pts, bs_test(1,1:no_of_perf_pts), '-*g');
    hold on;
    plot(perf_jm_t_pts, curr_results_r_rand_int(1,:), '-*b');
    plot(perf_jm_t_pts, curr_results_r(1,:), '-*m');
    legend('Proposed Model', 'Standard JM 1', 'Standard JM 2', 'Location', 'southeast');
    % legend('Proposed Model', 'Standard JM', 'Standard JM (int. only)', 'Location', 'southeast');
    title('Brier Score');
    xlim([0, max_horizon+csv_controls.Delta/2]);
    xticks(0:csv_controls.Delta:csv_controls.Delta*length(hist_count_arr));
    grid on;
    nexttile;
    plot(perf_t_pts, pe_test(1,1:no_of_perf_pts), '-*g');
    hold on;
    plot(perf_jm_t_pts, curr_results_r_rand_int(2,:), '-*b');
    plot(perf_jm_t_pts, curr_results_r(2,:), '-*m');
    % legend('Proposed Model', 'Standard JM', 'Standard JM (int. only)', 'Location', 'southeast');
    legend('Proposed Model', 'Standard JM 1', 'Standard JM 2', 'Location', 'southeast');
    title('Prediction Error');
    xlim([0, max_horizon+csv_controls.Delta/2]);
    xticks(0:csv_controls.Delta:csv_controls.Delta*length(hist_count_arr));
    grid on;
    nexttile;
    plot(perf_t_pts, auc_test(1,1:no_of_perf_pts), '-*g');
    hold on;
    plot(perf_jm_t_pts, curr_results_r_rand_int(3,:), '-*b');
    plot(perf_jm_t_pts, curr_results_r(3,:), '-*m');
    % legend('Proposed Model', 'Standard JM', 'Standard JM (int. only)', 'Location', 'southeast');
    legend('Proposed Model', 'Standard JM 1', 'Standard JM 2', 'Location', 'southeast');
    title('Area under ROC curve');
    xlim([0, max_horizon+csv_controls.Delta/2]);
    xticks(0:csv_controls.Delta:csv_controls.Delta*length(hist_count_arr));
    grid on;
    nexttile;
    bar(csv_controls.Delta:csv_controls.Delta:csv_controls.Delta*length(hist_count_arr), [hist_count_arr; hist_count_arr_censored], "stacked");
    xlim([0, max_horizon+csv_controls.Delta/2]);

    % ylabel('Frequency');
    title('Frequency of Events and Censored Observations');
    lgd = legend('Number of Event Observations', 'Number of Censored Observations');
    lgd.Location = 'southoutside';
    lgd.Orientation = 'horizontal';
    xticks(0:csv_controls.Delta:csv_controls.Delta*length(hist_count_arr));
    title(t, sprintf('Performance metrics across different horizons at landmark = %.1f years', ...
        landmark_t / csv_controls.t_multiplier));
    xlabel(t, 'Horizon (in months)');
    grid on;
        
end


