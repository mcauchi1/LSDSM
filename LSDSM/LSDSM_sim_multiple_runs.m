% Performing a simulation

%% 1. Initialisation and Simulation

% clear all;
% close all;

rng(42) % for repeatability

% Time step information
no_of_t_points = 60; % number of observations
Delta = 1; % time gap between every observation
t_all = linspace(0, Delta * (no_of_t_points-1), no_of_t_points); % Define the time sequence
t_multiplier = 1; % survival time (in months) = surv_time * t_multiplier

% Additional information
no_of_obs = 1; % number of different biomarkers captured
frac_miss_data = 0; % fraction of missing data (0.25 for 25% missing data)
range_cens = [10 110]; % uniform censoring range

% Controls and testing variables
rand_boolean = 1; % used to randomise every single simulation created
% If set to 0, the random functions will generate the same numbers every
% time the simulation function is run.
no_of_plots = 5; % plot a maximum of 5 figures
allow_pat_plots = 0; % when =1, will plot patient plots
landmark_t_arr = [10 20 30] * t_multiplier; % Landmarks to test
horizon_to_test = 10 * t_multiplier; % Horizons to test

% Below are two pre-set parameters based on the number of observations
if no_of_obs==1
    % State space parameters
    a_bar = [1.46 -0.48];
    A = [a_bar;
         eye(size(a_bar,1)) zeros(size(a_bar,1), size(a_bar,2) - size(a_bar,1))];
    C = eye(size(a_bar));
    W = (0.5)^2 * eye(size(a_bar, 1));
    V = (0.5)^2 * eye(size(C,1), size(C,1));
    G_mat = [eye(size(a_bar,1));
             zeros(size(a_bar,2)-size(a_bar,1))]; % fixed
    
    % Initial state parameters
    x0 = [10;
          5];
    W0 = (0.5)^2 * eye(size(x0, 1));

    % survival parameters
    g_s = [1.75];
    a_s = [-1;
           -0.5];

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

% Model parameters are placed in a struct data structure
true_model_coef = struct('A', A, 'C', C, 'W', W, 'V', V, ...
                         'g_s', g_s, 'a_s', a_s, 'DeltaT', Delta, 'G_mat', G_mat, ...
                         'mu_0', x0, 'W_0', W0);

% Simulate all patient information (longitudinal and survival data)
dim_size.states = size(x0,1);
dim_size.dyn_states = size(a_bar, 1);
dim_size.y = size(C, 1);
dim_size.base_cov = size(g_s,1);

censor_time = no_of_t_points;

no_of_pat_arr = [10, 20, 50, 100, 150, 200, 300, 500, 750, 1000];

% Controls to split the simulations into separate executions - do not clear
% variables between runs
no_of_runs = 2; % the number of runs to perform in this execution
max_no_of_runs = 10; % the number of runs planned
offset_no = 1;

% If we are starting a fresh set of simulations
if offset_no == 0
    est_model_rmse.train.mu = zeros(dim_size.states,1,max_no_of_runs,length(no_of_pat_arr));
    est_model_rmse.train.surv = zeros(1,1,max_no_of_runs,length(no_of_pat_arr));
    
    true_model_rmse.train.mu = zeros(dim_size.states,1,max_no_of_runs,length(no_of_pat_arr));
    true_model_rmse.train.surv = zeros(1,1,max_no_of_runs,length(no_of_pat_arr));
    
    est_model_rmse.test.mu = zeros(dim_size.states,1,max_no_of_runs,length(no_of_pat_arr));
    est_model_rmse.test.surv = zeros(1,1,max_no_of_runs,length(no_of_pat_arr));
    
    true_model_rmse.test.mu = zeros(dim_size.states,1,max_no_of_runs,length(no_of_pat_arr));
    true_model_rmse.test.surv = zeros(1,1,max_no_of_runs,length(no_of_pat_arr));
    
    param_diff_percent_arr = zeros(1,1,max_no_of_runs,length(no_of_pat_arr));

    % We can store parameters across different runs for comparison
    A_est_runs_arr = zeros(dim_size.states,dim_size.states,max_no_of_runs,length(no_of_pat_arr));
    W_est_runs_arr = zeros(dim_size.dyn_states,dim_size.dyn_states,max_no_of_runs,length(no_of_pat_arr));
    V_est_runs_arr = zeros(dim_size.y,dim_size.y,max_no_of_runs,length(no_of_pat_arr));
    g_s_est_runs_arr = zeros(dim_size.base_cov,1,max_no_of_runs,length(no_of_pat_arr));
    a_s_est_runs_arr = zeros(dim_size.states,1,max_no_of_runs,length(no_of_pat_arr));
    mu_0_est_runs_arr = zeros(dim_size.states,1,max_no_of_runs,length(no_of_pat_arr));
end

for iter_no=1:length(no_of_pat_arr)
    
    no_of_pat = no_of_pat_arr(iter_no);
    
    fprintf('Executing %d runs for %d patients \n', no_of_runs, no_of_pat);
    
    for run_no=1+offset_no:no_of_runs+offset_no
        % Create simulations with the above specifications
        [data_latent, data_observed] = ...
            LSDSM_ALLFUNCS.sim_obs_surv_pat(no_of_pat, censor_time, true_model_coef, frac_miss_data, range_cens, rand_boolean);


        if allow_pat_plots
            LSDSM_ALLFUNCS.plot_pat_info(no_of_plots, t_all, data_latent, data_observed);
        end

        max_y_vals = zeros(dim_size.y, 1);
        for ii=1:data_observed.Count
            max_y_vals = max([max_y_vals, reshape(squeeze(data_observed(ii).y), dim_size.y, no_of_t_points)], [], 2);
        end


        %% 2. Train the model

        %%% EM controls %%%

        % Initialise parameters randomly
        model_coef_init = LSDSM_ALLFUNCS.initialise_params(dim_size, data_observed, Delta);

        % Control which parameters to keep fixed by replacing NaN with a matrix
        fixed_params = struct('A', NaN, 'C', C, ...
                              'W', NaN, 'V', NaN, ...
                              'g_s', NaN, 'a_s', NaN, ...
                              'mu_0', NaN, 'W_0', NaN);

        controls.init_params = model_coef_init;
        controls.fixed_params = fixed_params;
        controls.EM_iters = 600; % number of iterations for the EM algorithm
        controls.mod_KF = true; % 1 means utilise survival information in the modified KF
        controls.verbose = true; % If true, it will provide feedback while the algorithm is executing
        controls.allow_plots = true; % If true, it will plot the log likelihood over EM iterations and some patient data
        controls.max_param_diff = 1e-4; % stopping criterion - stop if difference in all parameters < this value

        [model_coef_est, max_iter_reached, param_traj, RTS_traj] = ...
                        LSDSM_ALLFUNCS.LSDSM_EM(dim_size, data_observed, controls, censor_time);
        if allow_pat_plots
            for i=1:no_of_plots
                mu_tilde_arr = RTS_traj.mu_tilde(:,:,:,i);
                mu_hat_arr = RTS_traj.mu_hat(:,:,:,i);

                for j=1:dim_size.states
                    figure;
                    % if j <= dim_size.y
                    if j <= dim_size.y
                        scatter(t_all, squeeze(data_observed(i).y(j,:)));
                    else
                        scatter(t_all, squeeze(data_observed(i).y(mod(j-1, dim_size.states - dim_size.dyn_states)+1,:)));
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
                    ylim([0 max_y_vals(mod(j-1, dim_size.states - dim_size.dyn_states)+1)]);
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
        
        % Calculate the mu values using the true parameters model
        RTS_true.mu_tilde = zeros(size(RTS_traj.mu_tilde));
        for i=1:no_of_pat
            pat_ii = data_observed(i);
            pat_ii.mu_0 = true_model_coef.mu_0;
            pat_ii.W_0 = true_model_coef.W_0;
            [RTS_true.mu_tilde(:,:,:,i), V_tilde, log_like_val] = LSDSM_ALLFUNCS.Kalman_filter(pat_ii, true_model_coef, ...
                                                                                    censor_time, controls.mod_KF);
        end
        
        % Store results for training
        A_est_runs_arr(:,:,run_no,iter_no) = model_coef_est.A;
        W_est_runs_arr(:,:,run_no,iter_no) = model_coef_est.W;
        V_est_runs_arr(:,:,run_no,iter_no) = model_coef_est.V;
        g_s_est_runs_arr(:,:,run_no,iter_no) = model_coef_est.g_s;
        a_s_est_runs_arr(:,:,run_no,iter_no) = model_coef_est.a_s;
        mu_0_est_runs_arr(:,:,run_no,iter_no) = model_coef_est.mu_0;
        
        [param_diff, param_diff_percent] = LSDSM_ALLFUNCS.find_model_param_diff(true_model_coef, model_coef_est);
        
        param_diff_percent_arr(1,1,run_no,iter_no) = nanmean(param_diff_percent)*100;
        
        %%%%%%%%%%%%%%%%%%
        %%% RMSE of mu %%%
        %%%%%%%%%%%%%%%%%%

        %%% Comparison of states %%%
        % RMSE for true and estimated models for hidden states
        true_model_rmse_states = LSDSM_ALLFUNCS.find_rmse_states(data_latent, data_observed, RTS_true.mu_tilde);
        est_model_rmse_states = LSDSM_ALLFUNCS.find_rmse_states(data_latent, data_observed, RTS_traj.mu_tilde);

        true_model_rmse.train.mu(:,:,run_no,iter_no) = true_model_rmse_states;
        est_model_rmse.train.mu(:,:,run_no,iter_no) = est_model_rmse_states;
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%% RMSE of survival curves %%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % extract hidden states into a multi-dimensional array
        x_true_mat = LSDSM_ALLFUNCS.map_struct_to_4dims_mat(data_latent, 'x_true');
        % calculate the true, true model, and estimated, survival curves
        true_surv_fn = LSDSM_ALLFUNCS.surv_curve_filt(true_model_coef, data_observed, x_true_mat);
        true_model_surv_fn = LSDSM_ALLFUNCS.surv_curve_filt(true_model_coef, data_observed, RTS_true.mu_tilde);
        est_surv_fn = LSDSM_ALLFUNCS.surv_curve_filt(model_coef_est, data_observed, RTS_traj.mu_tilde);

        % RMSE for true and estimated models for survival curves
        true_model_rmse_surv = LSDSM_ALLFUNCS.find_rmse_surv(true_surv_fn, data_observed, true_model_surv_fn);
        est_rmse_surv = LSDSM_ALLFUNCS.find_rmse_surv(true_surv_fn, data_observed, est_surv_fn);

        true_model_rmse.train.surv(:,:,run_no,iter_no) = true_model_rmse_surv;
        est_model_rmse.train.surv(:,:,run_no,iter_no) = est_rmse_surv;

        %% 3. Simulate test data

        % Set same initial conditions for all patients
        no_of_pat_test = no_of_pat;
        x0_test = x0;
        W0_test = W0;
        frac_miss_data_test = frac_miss_data;

        % Create simulations for testing
        [test_data_latent, test_data_observed] = ...
            LSDSM_ALLFUNCS.sim_obs_surv_pat(no_of_pat_test, censor_time, true_model_coef, frac_miss_data_test, range_cens, rand_boolean);

        % Plot testing patients information
        if allow_pat_plots
            LSDSM_ALLFUNCS.plot_pat_info(no_of_plots, t_all, test_data_latent, test_data_observed);
        end
        
        % extract hidden states into a multi-dimensional array
        x_true_mat = LSDSM_ALLFUNCS.map_struct_to_4dims_mat(test_data_latent, 'x_true');
        % calculate the true, true model, and estimated, survival curves
        surv_fn_true_test_curve = LSDSM_ALLFUNCS.surv_curve_filt(true_model_coef, test_data_observed, x_true_mat);

        % Predictions to obtain the smoothed hidden states
        est_model_pat_data_test = LSDSM_ALLFUNCS.predict_fn(test_data_observed, no_of_t_points, censor_time, model_coef_est, controls);
        true_model_pat_data_test = LSDSM_ALLFUNCS.predict_fn(test_data_observed, no_of_t_points, censor_time, true_model_coef, controls);
        
        %%%%%%%%%%%%%%%%%%
        %%% RMSE of mu %%%
        %%%%%%%%%%%%%%%%%%
        % extract predicted states into a multi-dimensional array
        x_true_model_test_mat = zeros([size(true_model_pat_data_test(1).predictions.mu), true_model_pat_data_test.Count]);
        for ii=1:true_model_pat_data_test.Count
            x_true_model_test_mat(:,:,:,ii) = true_model_pat_data_test(ii).predictions.mu;
        end
        
        x_est_model_test_mat = zeros([size(est_model_pat_data_test(1).predictions.mu), est_model_pat_data_test.Count]);
        for ii=1:est_model_pat_data_test.Count
            x_est_model_test_mat(:,:,:,ii) = est_model_pat_data_test(ii).predictions.mu;
        end
        
        %%% Comparison of states %%%
        % RMSE for true and estimated models for hidden states
        true_model_rmse_states = LSDSM_ALLFUNCS.find_rmse_states(test_data_latent, test_data_observed, x_true_model_test_mat);
        est_model_rmse_states = LSDSM_ALLFUNCS.find_rmse_states(test_data_latent, test_data_observed, x_est_model_test_mat);

        true_model_rmse.test.mu(:,:,run_no,iter_no) = true_model_rmse_states;
        est_model_rmse.test.mu(:,:,run_no,iter_no) = est_model_rmse_states;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%% RMSE of survival curves %%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % extract hidden states into a multi-dimensional array
        x_true_mat = LSDSM_ALLFUNCS.map_struct_to_4dims_mat(test_data_latent, 'x_true');
        % calculate the true, true model, and estimated, survival curves
        true_surv_fn = LSDSM_ALLFUNCS.surv_curve_filt(true_model_coef, test_data_observed, x_true_mat);
        true_model_surv_fn = LSDSM_ALLFUNCS.surv_curve_filt(true_model_coef, test_data_observed, x_true_model_test_mat);
        est_surv_fn = LSDSM_ALLFUNCS.surv_curve_filt(model_coef_est, test_data_observed, x_est_model_test_mat);

        % RMSE for true and estimated models for survival curves
        true_model_rmse_surv = LSDSM_ALLFUNCS.find_rmse_surv(true_surv_fn, data_observed, true_model_surv_fn);
        est_rmse_surv = LSDSM_ALLFUNCS.find_rmse_surv(true_surv_fn, data_observed, est_surv_fn);

        true_model_rmse.test.surv(:,:,run_no,iter_no) = true_model_rmse_surv;
        est_model_rmse.test.surv(:,:,run_no,iter_no) = est_rmse_surv;
        
    end
    
end

save LSDSM_multiple_runs_sim.mat

