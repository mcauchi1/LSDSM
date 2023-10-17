%% 1. Initialisation and Simulation

clear all;
close all;

rng(421); % for reproducibility

% Time step information
no_of_t_points = 30; % number of observations
Delta = 1; % time gap between every observation
censor_time = Delta * no_of_t_points; % maximum time observed
t_all = linspace(0, Delta * (no_of_t_points-1), no_of_t_points); % Define the time sequence
t_multiplier = 1; % survival time (in months) = surv_time * t_multiplier

% Additional information
no_of_pat_train = 100; % number of patients
no_of_obs = 1; % number of different biomarkers captured
no_of_classes_true = 1; % the number of present classes
frac_miss_data_train = 0.25; % fraction of missing data (0.25 for 25% missing data)
range_cens = [10 50]; % uniform censoring range
sim_controls.MM = 'None'; %'altMM';
sim_controls.base_haz = 'None'; %'Weibull';

% Controls and testing variables
norm_bool = 0; % If set to 1, this will normalise the biomarker measurements
rand_boolean = 0; % used to randomise every single simulation created - for repeatability, keep it 0
% If set to 0, the random functions will generate the same numbers every
% time the simulation function is run.
no_of_plots = min(5, no_of_pat_train); % plot a maximum of 5 figures
allow_pat_plots = 0; % when =1, plot patient plots
allow_perf_plots = 0; % when =1, plot survival performance plots
landmark_t_arr = [10 15 20] * t_multiplier; % Landmarks to test
ignore_all_NaNs = 0; % if 1, then when making predictions, we ignore those patients that have no observed values

% Initialise the map for the true parameter models
true_model_coef = containers.Map('KeyType', 'int32', 'ValueType', 'any');

% Below are two pre-set parameters based on the number of observations
if no_of_obs==1
    % State space parameters
    a_bar1 = [1.46 -0.48];
    A1 = [a_bar1;
          eye(size(a_bar1,1)) zeros(size(a_bar1,1), size(a_bar1,2) - size(a_bar1,1))];
    C1 = eye(size(a_bar1));
    W1 = (0.2)^2 * eye(size(a_bar1, 1));
    V1 = (0.5)^2 * eye(size(C1,1), size(C1,1));
    G_mat = [eye(size(a_bar1,1));
             zeros(size(a_bar1,2)-size(a_bar1,1))]; % fixed
         
    H_mat = [1 0]; % the relation between hidden states and hazard function - fixed valued
    
    % Initial state parameters
    mu_01 = [10;
             10];
    W_01 = (1)^2 * eye(size(mu_01, 1));

    % survival parameters
    g_s1 = [2.5;
            -0.75];
    a_s1 = [-1.25];
        
    zeta1 = [0];
        
    % Model parameters are placed in a struct data structure
    true_model_coef(1) = struct('A', A1, 'C', C1, 'W', W1, 'V', V1, ...
                                'g_s', g_s1, 'a_s', a_s1, 'DeltaT', Delta, 'G_mat', G_mat, 'H_mat', H_mat, ...
                                'mu_0', mu_01, 'W_0', W_01, 'zeta', zeta1);

elseif no_of_obs==2
    % State space parameters
    a_bar = [1.46 0 -0.48 0;
             0.2 0.5 -0.1 -0.2];
    A = [a_bar;
         eye(size(a_bar,1)) zeros(size(a_bar,1), size(a_bar,2) - size(a_bar,1))];
    C = eye(size(a_bar));
    W = (0.5)^2 * eye(size(a_bar, 1));
    V = (0.5)^2 * eye(size(C,1), size(C,1));
    G_mat = [eye(size(a_bar,1));
             zeros(size(a_bar,2)-size(a_bar,1))];
         
    % Initisl state parameters
    x0 = [10;
          7.5;
          5;
          2.5];
    W0 = (0.5)^2 * eye(size(x0, 1));

    % Survival parameters
    g_s = [1];
    a_s = [-1.75;
           0.1;
           -1;
           0.25];
end

% a preset for the alternative model with survival independent of states
if strcmpi(sim_controls.MM, 'altMM')
    % State space parameters
    a_bar1 = [1.46 -0.48];
    A1 = [a_bar1;
          eye(size(a_bar1,1)) zeros(size(a_bar1,1), size(a_bar1,2) - size(a_bar1,1))];
    C1 = eye(size(a_bar1));
    W1 = (0.75)^2 * eye(size(a_bar1, 1));
    V1 = (0.5)^2 * eye(size(C1,1), size(C1,1));
    G_mat = [eye(size(a_bar1,1));
             zeros(size(a_bar1,2)-size(a_bar1,1))]; % fixed
    
    % Initial state parameters
    mu_01 = [10;
             10];
    W_01 = (1)^2 * eye(size(mu_01, 1));

    % survival parameters
    g_s1 = [-0.5];
    g_s1 = [-0.5;
            1];
    a_s1 = 2.5;
    b_s1 = 50;
       
    zeta1 = [0];
        
    true_model_coef = containers.Map('KeyType', 'int32', 'ValueType', 'any');
    % Model parameters are placed in a struct data structure
    true_model_coef(1) = struct('A', A1, 'C', C1, 'W', W1, 'V', V1, ...
                                'g_s', g_s1, 'a_s', a_s1, 'b_s', b_s1, 'DeltaT', Delta, 'G_mat', G_mat, ...
                                'mu_0', mu_01, 'W_0', W_01, 'zeta', zeta1);
    
end

% set the dimensions of every model characteristic
dim_size.states = size(mu_01, 1);
dim_size.dyn_states = size(a_bar1, 1);
dim_size.y = size(C1, 1);
dim_size.base_cov = size(g_s1, 1);
dim_size.alpha_eqns = size(a_s1, 1);
dim_size.class_cov = size(zeta1, 1);
if isfield(true_model_coef(1), 'B') % if input exists
    dim_size.u = size(B1,2);
end

% create a normal distribution map for the class covariates
class_cov_distributions = containers.Map('KeyType', 'int32', 'ValueType', 'any');
class_cov_distributions(1) = struct('mean', 60, 'std', 10); % resembles age


%% Simulate
no_of_pat_arr = [500]; % array of patient numbers to perform differently sized simulations

% Controls to split the simulations into separate executions - do not clear
% variables between runs
no_of_runs = 10; % the number of runs to perform in this execution
max_no_of_runs = 10; % the number of runs planned
offset_no = 0; % used in case the simulations are split

% If we are starting a fresh set of simulations
if offset_no == 0
    est_model_rmse.train.mu = zeros(dim_size.states,1,max_no_of_runs,length(no_of_pat_arr));
    est_model_rmse.train.surv = zeros(1,1,max_no_of_runs,length(no_of_pat_arr));
    
    true_model_rmse.train.mu = zeros(dim_size.states,1,max_no_of_runs,length(no_of_pat_arr));
    
    est_model_rmse.test.mu = zeros(dim_size.states,1,max_no_of_runs,length(no_of_pat_arr));
    est_model_rmse.test.surv = zeros(1,1,max_no_of_runs,length(no_of_pat_arr));
    
    true_model_rmse.test.mu = zeros(dim_size.states,1,max_no_of_runs,length(no_of_pat_arr));
    
    param_diff_percent_arr = zeros(1,1,max_no_of_runs,length(no_of_pat_arr));
    EM_iter_arr = zeros(1,1,max_no_of_runs,length(no_of_pat_arr));
    
    E_m_i_arr = zeros(1,1,max_no_of_runs,length(no_of_pat_arr));
    E_T_i_arr = zeros(1,1,max_no_of_runs,length(no_of_pat_arr));
    E_delta_i_arr = zeros(1,1,max_no_of_runs,length(no_of_pat_arr));

    % We can store parameters across different runs for comparison
    A_est_runs_arr = zeros(dim_size.states,dim_size.states,max_no_of_runs,length(no_of_pat_arr));
    W_est_runs_arr = zeros(dim_size.dyn_states,dim_size.dyn_states,max_no_of_runs,length(no_of_pat_arr));
    V_est_runs_arr = zeros(dim_size.y,dim_size.y,max_no_of_runs,length(no_of_pat_arr));
    g_s_est_runs_arr = zeros(dim_size.base_cov,1,max_no_of_runs,length(no_of_pat_arr));
    a_s_est_runs_arr = zeros(dim_size.alpha_eqns,1,max_no_of_runs,length(no_of_pat_arr));
    mu_0_est_runs_arr = zeros(dim_size.states,1,max_no_of_runs,length(no_of_pat_arr));
    
    mult_bs_test_runs_arr = zeros(length(landmark_t_arr), no_of_t_points, max_no_of_runs, length(no_of_pat_arr));
    mult_pe_test_runs_arr = zeros(length(landmark_t_arr), no_of_t_points, max_no_of_runs, length(no_of_pat_arr));
    mult_auc_test_runs_arr = zeros(length(landmark_t_arr), no_of_t_points, max_no_of_runs, length(no_of_pat_arr));
end

for iter_no=1:length(no_of_pat_arr)
    
    no_of_pat = no_of_pat_arr(iter_no);
    
    fprintf('Executing %d runs for %d patients \n', no_of_runs, no_of_pat);
    
    for run_no=1+offset_no:no_of_runs+offset_no
        
        tic % used to check the execution time for every simulation
        
        fprintf('Run: %d \n', run_no);
        % Create simulations (longitudinal and survival data) with the above specifications
        [data_latent, data_observed] = LSDSM_MM_ALLFUNCS.sim_obs_surv_pat(no_of_pat, censor_time, true_model_coef, class_cov_distributions, ...
                                                                          dim_size, frac_miss_data_train, range_cens, rand_boolean, ...
                                                                          sim_controls);

        if allow_pat_plots % Plot patient's hidden states and hazard function
            LSDSM_MM_ALLFUNCS.plot_pat_info(no_of_plots, t_all, data_latent, data_observed);
        end

        % Calculate the normalisation constant should we want to normalise
        normalise_const = zeros(dim_size.y, 1);
        for ii=1:data_observed.Count
            normalise_const = max([normalise_const, reshape(squeeze(data_observed(ii).y), dim_size.y, no_of_t_points)], [], 2);
        end

        if allow_pat_plots
            % plot the observations of some patients
            biomarker_to_plot = 1; % Choose which biomarker to plot
            for ii=1:no_of_plots
                figure;
                scatter(t_all, squeeze(data_observed(ii).y(biomarker_to_plot,:)), 'LineWidth', 2);
                xlabel('Time');
                legend('y');
                xlim([t_all(1) t_all(end)]);
                if norm_bool
                    ylim([0 1]);
                else
                    ylim([0 normalise_const(biomarker_to_plot)]);
                end

                grid on;
            end
        end

        % store survival information
        ev_ind = LSDSM_MM_ALLFUNCS.extract_field_from_map(data_observed, 'delta_ev')';
        surv_time = LSDSM_MM_ALLFUNCS.extract_field_from_map(data_observed, 'surv_time')';
        
        % calculate the event rate and the mean survival time
        E_delta_i_arr(:,:,run_no,iter_no) = sum(ev_ind);
        E_T_i_arr(:,:,run_no,iter_no) = mean(surv_time);

        %% 2. Train the model
        
        % find the number of observed measurements
        y_arr = LSDSM_MM_ALLFUNCS.extract_field_from_map(data_observed, 'y');
        num_obs = LSDSM_MM_ALLFUNCS.extract_field_from_map(data_observed, 'm_i')';
        sum_m_i = 0;
        for i=1:no_of_pat
            sum_m_i = sum_m_i + sum(~isnan(y_arr(:,:,1:num_obs(i),i)));
        end
        
        % expected number of observed measurements per patient
        E_m_i = sum_m_i / no_of_pat;
        E_m_i_arr(:,:,run_no,iter_no) = E_m_i;

        no_of_classes = 1; % Number of models to train

        % Control which parameters to keep fixed by replacing NaN with a matrix
        if strcmpi(sim_controls.MM, 'altMM') % if we are using alternative MM
            % fix the variances to higher than expected values - empircally
            % found to produce better results
            fixed_params_tmp = struct('A', NaN, 'B', NaN, 'C', true_model_coef(1).C, ...
                                  'W', 5, 'V', 5, ...
                                  'g_s', NaN, 'a_s', NaN, ...
                                  'mu_0', mu_01, 'W_0', 20 * eye(2)); % 5 * eye(2)

            fixed_params = containers.Map('KeyType', 'int32', 'ValueType', 'any');

            for g=1:no_of_classes
                fixed_params(g) = fixed_params_tmp;
            end

        else % if we are using standard MM
            % do not fix the variances - fix only initial states covariance
            fixed_params_tmp = struct('A', NaN, 'B', NaN, 'C', true_model_coef(1).C, ...
                                  'W', NaN, 'V', NaN, ...
                                  'g_s', NaN, 'a_s', NaN, ...
                                  'mu_0', mu_01, 'W_0', W_01);

            fixed_params = containers.Map('KeyType', 'int32', 'ValueType', 'any');

            for g=1:no_of_classes
                fixed_params(g) = fixed_params_tmp;
            end
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
        controls.base_haz = 'None'; % 'Weibull' - for alternative model
        controls.do_EM_better_init_params = true; % used to use EM on a small dataset for better initialisation
        controls.mod_KF = true; % 1 means utilise survival information in the modified KF
        % if we utilise the alternative model
        if strcmpi(sim_controls.MM, 'altMM') && strcmpi(sim_controls.base_haz, 'Weibull')
            controls.base_haz = 'Weibull'; % 'Weibull';
            controls.mod_KF = false; % true means utilise survival information in the modified KF
            same_params.b_s = 0; % retains the same shape parameter
        end

        % Initialise the parameters
        model_coef_init = LSDSM_MM_ALLFUNCS.better_init_params(dim_size, data_observed, fixed_params, Delta, H_mat, ...
                                                                    censor_time, no_of_classes, controls);

        controls.init_params = model_coef_init; % set initial parameters

        % Run the EM algorithm for all combined classes
        [model_coef_est, max_iter_reached, param_traj, RTS_traj, E_c_ig_allpats] = ...
                        LSDSM_MM_ALLFUNCS.LSDSM_MM_EM(dim_size, data_observed, controls, censor_time);

        if strcmpi(sim_controls.MM, 'altMM') % if we are using alternative MM
            % Re-run the EM algorithm without retaining fixed parameters on the
            % variances, and with the newly found parameter estimates as starting points.
            % Control which parameters to keep fixed by replacing NaN with a matrix

            fixed_params_tmp = struct('A', NaN, 'B', NaN, 'C', true_model_coef(1).C, ...
                                  'W', NaN, 'V', NaN, ...
                                  'g_s', NaN, 'a_s', NaN, ...
                                  'mu_0', mu_01, 'W_0', 20 * eye(2)); % 5 * eye(2)

            fixed_params = containers.Map('KeyType', 'int32', 'ValueType', 'any');

            for g=1:no_of_classes % set fixed parameters for every class
                fixed_params(g) = fixed_params_tmp;
            end

            controls.fixed_params = fixed_params;
            controls.init_params = model_coef_est; % set initial parameters - from previous EM

            % Run the EM algorithm for all combined classes
            [model_coef_est, max_iter_reached, param_traj, RTS_traj, E_c_ig_allpats] = ...
                            LSDSM_MM_ALLFUNCS.LSDSM_MM_EM(dim_size, data_observed, controls, censor_time);
        end % end alternative model

        if allow_pat_plots % plot the filtered and smoothed hidden state trajectories
            for ii=1:no_of_plots
                mu_tilde_arr = RTS_traj.mu_tilde(:,:,:,ii);
                mu_hat_arr = RTS_traj.mu_hat(:,:,:,ii);

                for j=1:dim_size.states
                    figure;
                    if j <= dim_size.y
                        scatter(t_all, squeeze(data_observed(ii).y(j,:)));
                    else
                        scatter(t_all, squeeze(data_observed(ii).y(mod(j-1, dim_size.states - dim_size.dyn_states)+1,:)));
                    end
                    hold on;
                    plot(t_all, squeeze(mu_tilde_arr(j,1,:)));
                    plot(t_all, squeeze(mu_hat_arr(j,1,:)));
                    if j <= dim_size.y
                        legend('y', '\mu_{tilde}', '\mu_{hat}');
                    else
                        legend('y', '\mu_{tilde}', '\mu_{hat}')
                    end
                    if data_observed(ii).delta_ev == 0
                        title_n = sprintf('Patient was censored at time %.1f months: State %d', data_observed(ii).surv_time, j);
                    else
                        title_n = sprintf('Patient died at time %.1f months: State %d', data_observed(ii).surv_time, j);
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
        
        % Store parameter estimates for current simulation
        A_est_runs_arr(:,:,run_no,iter_no) = model_coef_est(1).A;
        W_est_runs_arr(:,:,run_no,iter_no) = model_coef_est(1).W;
        V_est_runs_arr(:,:,run_no,iter_no) = model_coef_est(1).V;
        g_s_est_runs_arr(:,:,run_no,iter_no) = model_coef_est(1).g_s;
        a_s_est_runs_arr(:,:,run_no,iter_no) = model_coef_est(1).a_s;
        mu_0_est_runs_arr(:,:,run_no,iter_no) = model_coef_est(1).mu_0;
        
        % Store number of EM iterations required until convergence for
        % current simulation
        EM_iter_arr(:,:,run_no,iter_no) = max_iter_reached;
        
        if no_of_classes==1
            % Calculate the mu values using the true parameters model
            RTS_true.mu_tilde = zeros(size(RTS_traj{1}.mu_tilde));
            for ii=1:no_of_pat
                pat_ii = data_observed(ii);
                pat_ii.mu_0 = true_model_coef(1).mu_0;
                pat_ii.W_0 = true_model_coef(1).W_0;
                [RTS_true.mu_tilde(:,:,:,ii), V_tilde, log_like_val] = LSDSM_MM_ALLFUNCS.Kalman_filter(pat_ii, true_model_coef(1), ...
                                                                                        censor_time+1, controls);
            end

            [param_diff, param_diff_percent] = LSDSM_MM_ALLFUNCS.find_model_param_diff(true_model_coef(1), model_coef_est(1));

            param_diff_percent_arr(1,1,run_no,iter_no) = nanmean(param_diff_percent)*100;

            %%%%%%%%%%%%%%%%%%
            %%% RMSE of mu %%%
            %%%%%%%%%%%%%%%%%%

            %%% Comparison of states %%%
            % RMSE for true and estimated models for hidden states
            true_model_rmse_states = LSDSM_MM_ALLFUNCS.find_rmse_states(data_latent, data_observed, RTS_true.mu_tilde);
            est_model_rmse_states = LSDSM_MM_ALLFUNCS.find_rmse_states(data_latent, data_observed, RTS_traj{1}.mu_tilde);

            true_model_rmse.train.mu(:,:,run_no,iter_no) = true_model_rmse_states;
            est_model_rmse.train.mu(:,:,run_no,iter_no) = est_model_rmse_states;
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%% RMSE of survival curves %%%
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % extract hidden states into a multi-dimensional array
            x_true_mat = LSDSM_MM_ALLFUNCS.extract_field_from_map(data_latent, 'x_true');
            % calculate the true, true model, and estimated, survival curves
            true_surv_fn = LSDSM_MM_ALLFUNCS.surv_curve_calc(true_model_coef(1), data_observed, x_true_mat);
            est_surv_fn = LSDSM_MM_ALLFUNCS.surv_curve_filt(model_coef_est, data_observed, censor_time, dim_size, controls);

            % RMSE for true and estimated models for survival curves
            est_rmse_surv = LSDSM_MM_ALLFUNCS.find_rmse_surv(true_surv_fn, data_observed, est_surv_fn);
            est_model_rmse.train.surv(:,:,run_no,iter_no) = est_rmse_surv;
        end

        %% 4. Simulate test data
        no_of_pat_test = no_of_pat; % number of patients in testing dataset
        frac_miss_data_test = frac_miss_data_train; % fraction of missing data

        % Create simulations for testing
        [test_data_latent, test_data_observed] = ...
            LSDSM_MM_ALLFUNCS.sim_obs_surv_pat(no_of_pat_test, censor_time, true_model_coef, class_cov_distributions, ...
                                               dim_size, frac_miss_data_train, range_cens, rand_boolean, sim_controls);

        % Plot patient's hidden states and hazard function
        if allow_pat_plots
            LSDSM_MM_ALLFUNCS.plot_pat_info(no_of_plots, t_all, test_data_latent, test_data_observed);
        end
        
        %% 5. Analyse Performance on testing data set
        if no_of_classes==1
            
            % Calculate the mu values using the true and estimated parameters model
            RTS_true.mu_tilde = zeros(size(RTS_traj{1}.mu_tilde));
            RTS_traj_test.mu_tilde = zeros(size(RTS_traj{1}.mu_tilde));
            for ii=1:no_of_pat
                pat_ii = test_data_observed(ii);
                pat_ii.mu_0 = true_model_coef(1).mu_0;
                pat_ii.W_0 = true_model_coef(1).W_0;
                [RTS_true.mu_tilde(:,:,:,ii), V_tilde, log_like_val] = LSDSM_MM_ALLFUNCS.Kalman_filter(pat_ii, true_model_coef(1), ...
                                                                                        censor_time+1, controls);
                [RTS_traj_test.mu_tilde(:,:,:,ii), V_tilde, log_like_val] = LSDSM_MM_ALLFUNCS.Kalman_filter(pat_ii, model_coef_est(1), ...
                                                                                        censor_time+1, controls);
            end

            %%%%%%%%%%%%%%%%%%
            %%% RMSE of mu %%%
            %%%%%%%%%%%%%%%%%%
            %%% Comparison of states %%%
            % RMSE for true and estimated models for hidden states
            true_model_rmse_states = LSDSM_MM_ALLFUNCS.find_rmse_states(test_data_latent, test_data_observed, RTS_true.mu_tilde);
            est_model_rmse_states = LSDSM_MM_ALLFUNCS.find_rmse_states(test_data_latent, test_data_observed, RTS_traj_test.mu_tilde);

            true_model_rmse.test.mu(:,:,run_no,iter_no) = true_model_rmse_states;
            est_model_rmse.test.mu(:,:,run_no,iter_no) = est_model_rmse_states;

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%% RMSE of survival curves %%%
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % extract hidden states into a multi-dimensional array
            x_true_mat = LSDSM_MM_ALLFUNCS.extract_field_from_map(test_data_latent, 'x_true');
            % calculate the true, true model, and estimated, survival curves
            true_surv_fn = LSDSM_MM_ALLFUNCS.surv_curve_calc(true_model_coef(1), test_data_observed, x_true_mat);
            est_surv_fn = LSDSM_MM_ALLFUNCS.surv_curve_filt(model_coef_est, test_data_observed, censor_time, dim_size, controls);

            % RMSE for true and estimated models for survival curves
            est_rmse_surv = LSDSM_MM_ALLFUNCS.find_rmse_surv(true_surv_fn, test_data_observed, est_surv_fn);
            est_model_rmse.test.surv(:,:,run_no,iter_no) = est_rmse_surv;
        end

        %% 5. Analyse Performance on testing data set
        % Initialise arrays for time-dependent Brier score, prediction error, and area under ROC curve
        mult_bs_test = zeros(length(landmark_t_arr), no_of_t_points);
        mult_pe_test = zeros(length(landmark_t_arr), no_of_t_points);
        mult_auc_test = zeros(length(landmark_t_arr), no_of_t_points);

        upd_mult_bs_test = zeros(length(landmark_t_arr), no_of_t_points);
        upd_mult_pe_test = zeros(length(landmark_t_arr), no_of_t_points);
        upd_mult_auc_test = zeros(length(landmark_t_arr), no_of_t_points);

        % Landmark is used for the testing data, where we forecast from this point
        % onwards the hidden state values and the respective survival values
        for l_var=1:length(landmark_t_arr)
            landmark_t = landmark_t_arr(l_var); % use data up to this value (in months)

            %% 5a. Performance Metrics - Forecasts
            t_est_idx = floor(censor_time / Delta); % maximum forecast time

            % Forecasts (longitudinal and survival) using the proper distributions
            pat_data_reduced = LSDSM_MM_ALLFUNCS.forecast_fn(test_data_observed, landmark_t, t_est_idx, ...
                                                                 censor_time, dim_size, model_coef_est, controls, ignore_all_NaNs);

            % find the longest interval to work out the performance metrics
            max_horizon_t = t_est_idx - landmark_t;

            %% 5b. Performance Metrics - Time-dependent Brier Score

            % Brier score for the forecast function using the proper distributions
            [bs_test] = LSDSM_MM_ALLFUNCS.Brier_Score_fn(pat_data_reduced, landmark_t, max_horizon_t, model_coef_est(1).DeltaT);
            upd_mult_bs_test(l_var, 1:length(bs_test)) = bs_test;

            %% 5c. Performance Metrics - Time-dependent Prediction Error

            % Prediction error for the forecast function using the proper distributions
            [pe_test] = LSDSM_MM_ALLFUNCS.Prediction_Error_fn(pat_data_reduced, landmark_t, max_horizon_t, ...
                                                                      model_coef_est(1).DeltaT);
            upd_mult_pe_test(l_var, 1:length(pe_test)) = pe_test;

            %% 5d. Performance Metrics - AUC

            % AUC for the forecast function using the proper distributions
            [auc_test] = LSDSM_MM_ALLFUNCS.AUC_fn(pat_data_reduced, landmark_t, max_horizon_t, model_coef_est(1).DeltaT);
            upd_mult_auc_test(l_var, 1:length(auc_test)) = auc_test;

            %% 5e. Plot results and some survival prediction plots

            LSDSM_MM_ALLFUNCS.plot_bs_pe_auc(pat_data_reduced, bs_test, pe_test, ...
                                             auc_test, landmark_t, censor_time, t_multiplier, Delta)

            if allow_pat_plots
                for ii=1:no_of_plots
                    LSDSM_MM_ALLFUNCS.plot_forecast_surv(pat_data_reduced(ii), landmark_t, t_all, Delta)
                end
            end

        end % end of landmarks for loop
        
        % Store the survival performance metrics
        mult_bs_test_runs_arr(:,:,run_no, iter_no) = upd_mult_bs_test;
        mult_pe_test_runs_arr(:,:,run_no, iter_no) = upd_mult_pe_test;
        mult_auc_test_runs_arr(:,:,run_no, iter_no) = upd_mult_auc_test;

        close all; % figures
        
        save LSDSM_MM_multiple_runs_sim.mat % save current state
        
        toc % used to check the execution time for every simulation
        
    end % end of runs for this num_pats config
end % end of num_pats array

save LSDSM_MM_multiple_runs_sim.mat % save all simulations

%% Post simulations analysis
figure; histogram(A_est_runs_arr(1,1,:,:), 20); title('A_{11}'); xline(A1(1,1), '--r', 'LineWidth', 2);
figure; histogram(A_est_runs_arr(1,2,:,:), 20); title('A_{12}'); xline(A1(1,2), '--r', 'LineWidth', 2);
figure; histogram(W_est_runs_arr(1,1,:,:), 20); title('W'); xline(W1(1,1), '--r', 'LineWidth', 2);
figure; histogram(V_est_runs_arr(1,1,:,:), 20); title('V'); xline(V1(1,1), '--r', 'LineWidth', 2);
figure; histogram(g_s_est_runs_arr(1,1,:,:), 20); title('\gamma_{s1}'); xline(g_s1(1,1), '--r', 'LineWidth', 2);
figure; histogram(g_s_est_runs_arr(2,1,:,:), 20); title('\gamma_{s2}'); xline(g_s1(2,1), '--r', 'LineWidth', 2);
figure; histogram(a_s_est_runs_arr(1,1,:,:), 20); title('\alpha_{s1}'); xline(a_s1(1,1), '--r', 'LineWidth', 2);

% mean parameters
mean_A_est1 = mean(A_est_runs_arr(1,1,:,:));
mean_A_est2 = mean(A_est_runs_arr(1,2,:,:));
mean_W_est = mean(W_est_runs_arr(1,1,:,:));
mean_V_est = mean(V_est_runs_arr(1,1,:,:));
mean_g_s_est1 = mean(g_s_est_runs_arr(1,1,:,:));
mean_g_s_est2 = mean(g_s_est_runs_arr(2,1,:,:));
mean_a_s_est = mean(a_s_est_runs_arr(1,1,:,:));

% bias in parameters
bias_A_est1 = (A1(1,1) - mean_A_est1) / A1(1,1) * 100;
bias_A_est2 = (A1(1,2) - mean_A_est2) / A1(1,2) * 100;
bias_W_est = (W1(1,1) - mean_W_est) / W1(1,1) * 100;
bias_V_est = (V1(1,1) - mean_V_est) / V1(1,1) * 100;
bias_g_s_est1 = (g_s1(1,1) - mean_g_s_est1) / g_s1(1,1) * 100;
bias_g_s_est2 = (g_s1(2,1) - mean_g_s_est2) / g_s1(2,1) * 100;
bias_a_s_est = (a_s1(1,1) - mean_a_s_est) / a_s1(1,1) * 100;

% standard deviation - default divides by N-1 observations
std_A_est1 = std(A_est_runs_arr(1,1,:,:));
std_A_est2 = std(A_est_runs_arr(1,2,:,:));
std_W_est = std(W_est_runs_arr(1,1,:,:));
std_V_est = std(V_est_runs_arr(1,1,:,:));
std_g_s_est1 = std(g_s_est_runs_arr(1,1,:,:));
std_g_s_est2 = std(g_s_est_runs_arr(2,1,:,:));
std_a_s_est = std(a_s_est_runs_arr(1,1,:,:));
