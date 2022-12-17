classdef LSDSM_ALLFUNCS
    methods(Static)
        
        % Functions used in this script
        function [data_observed, csv_controls] = read_from_csv(M_train, dim_size, csv_controls)
            
            %%% Assumed structure of csv file %%%
            % col 01 = patID
            % col 02 = survTime
            % col 03 = eventInd
            % col 04 = tij
            % col base_cov_col_no = baseline covariate 1
            % col ... = baseline covariate ...
            % col base_cov_col_no + dim_size.base_cov = longitudinal biomarker 1
            % col ... = longitudinal biomarker ...
            
            % Normalisation constants if we wish to normalise the biomarkers
            normalise_const = max(M_train(:,csv_controls.bio_long_col_no:end));

            % Find the maximum time from the training data
            max_t = ceil(max(M_train(:,2))); % ceil(max(surv time))
            % Create an array for time in csv_controls.Delta steps till max_t is reached
            no_of_months = linspace(0,max_t*csv_controls.t_multiplier, max_t*csv_controls.t_multiplier/csv_controls.Delta+1);
            no_of_t_points = length(no_of_months); % total number of points considered
            csv_controls.t_all = no_of_months;
            % csv_controls.censor_time = (no_of_t_points-1)*csv_controls.Delta; % censoring hard threshold (in months)

            data_observed = containers.Map('KeyType', 'int32', 'ValueType', 'any'); % Stores observed patient data
            % Find the number of patients in the training set
            unique_pat_id = unique(M_train(:,1));
            no_of_pat = length(unique_pat_id);

            % Filling in patient information
            for i=1:no_of_pat
                pat_ii = struct();
                pat_ii.id = unique_pat_id(i);
                pat_ii.surv_time = 0;
                pat_ii.delta_ev = 0;
                pat_ii.m_i = 0;
                pat_ii.base_cov = zeros(dim_size.base_cov,1);
                % NaN was used for y as these will be treated as missing values if no
                % observations were made at that time period.
                pat_ii.y = NaN * zeros(dim_size.y, 1, length(no_of_months));
                % Extract the part of matrix that contains only the current patient's info
                curr_ind = find(M_train(:,1) == pat_ii.id);
                % Find the iteration number for every observation made for this patient
                % Note: For training data, we use round function to go to the closest
                % time binning. In testing data, we shall use ceil for those time
                % points that arrive after the landmark of interest. This is because we
                % do not want to use future data after the landmark.
                % E.g. assume landmark is at 12 time points. If there is an observation
                % at 12.4, then using the round function, this will go to 12 time
                % points, and hence, future data is used. With ceil, this will not be
                % utilised.
                if strcmp(csv_controls.train_test, 'train') % if training data set
                    iter_no = round(M_train(curr_ind, 4) * csv_controls.t_multiplier/csv_controls.Delta);
                else % if testing data set
                    iter_no = M_train(curr_ind, 4) * csv_controls.t_multiplier/csv_controls.Delta;
                    iter_no(iter_no<=csv_controls.landmark_idx) = round(iter_no(iter_no<=csv_controls.landmark_idx));
                    iter_no(iter_no>csv_controls.landmark_idx) = ceil(iter_no(iter_no>csv_controls.landmark_idx));
                end
                for j=1:length(curr_ind)
                    if csv_controls.norm_bool
                        pat_ii.y(:,1,iter_no(j)+1) = M_train(curr_ind(j), csv_controls.bio_long_col_no:csv_controls.bio_long_col_no+dim_size.y-1) ...
                                                     ./ normalise_const(1:dim_size.y);
                    else
                        pat_ii.y(:,1,iter_no(j)+1) = M_train(curr_ind(j), csv_controls.bio_long_col_no:csv_controls.bio_long_col_no+dim_size.y-1);
                    end
                end
                % Also note: utilising the above method means that some observations
                % are not utilised, since they happen to fall at the same time bin.

                % If we want to visualise some plots
                if csv_controls.allow_plots && i <= csv_controls.no_of_plots
                    figure;
                    to_plot = M_train(curr_ind, csv_controls.bio_long_col_no);
                    if csv_controls.norm_bool
                        to_plot = to_plot / normalise_const(1);
                    end
                    scatter(M_train(curr_ind, 4), to_plot);
                    xlabel('Time (years)')
                    ylabel('y');
                    hold on;

                    if M_train(curr_ind(j), 3) == 0 % if patient is censored
                        xline(M_train(curr_ind(1), 2), 'g', 'LineWidth', 2);
                        legend('y', 'Censored');
                    else
                        xline(M_train(curr_ind(1), 2), 'r', 'LineWidth', 2);
                        legend('y', 'Event');
                    end

                    xlim([0,max_t]);
                    if csv_controls.norm_bool
                        ylim([0,1]);
                    else
                        ylim([min(M_train(:,csv_controls.bio_long_col_no)) normalise_const(1)]);
                    end
                end

                % Store survival information
                pat_ii.delta_ev = M_train(curr_ind(j), 3);
                pat_ii.surv_time = M_train(curr_ind(1), 2)*csv_controls.t_multiplier;

                % If the survival time is greater than the hard thresholding of the
                % censor time, then the patient is censored at the threshold time
                if pat_ii.surv_time > csv_controls.censor_time
                    pat_ii.delta_ev = 0;
                    pat_ii.surv_time = csv_controls.censor_time;
                end

                % Number of time periods the patient is observed for
                pat_ii.m_i = floor(pat_ii.surv_time/csv_controls.Delta)+1;

                % Store the baseline covariates
                pat_ii.base_cov(1,1) = 1; % intercept
                % Other baseline covariates
                pat_ii.base_cov(2:end,1) = M_train(curr_ind(j), csv_controls.base_cov_col_no:csv_controls.base_cov_col_no+dim_size.base_cov-2); 

                data_observed(i) = pat_ii;
            end
        end
        
        
        % Data simulation
        function [data_latent, data_observed] = ...
            sim_obs_surv_pat(num_pats, cens_time, model_true, frac_miss, cens_range, rand_bool)
        
        
            % We will sample from a uniform distribution to obtain a survival time
            % for a patient, which will be determined through the Inverse Transform
            % Sampling.
            % For the details regarding this sampling of the survival time, look at
            % "Example for a Piecewise Constant Hazard Data Simulation 
            % in R" - Technical Report by Rainer Walke 2010
            
            % These maps will contain a struct for every patient
            % Each struct (for observed data) consists of:
            % -- m_i - Number of observed timepoints for the patient
            % -- delta_ev - Event indicator for the patient
            % -- surv_time - Survival time for the patient
            % -- y - longitudinal biomarkers for the patient
            % -- base_cov - baseline covariates for the patient
            data_observed = containers.Map('KeyType', 'int32', 'ValueType', 'any'); % Stores observed patient data
            data_latent = containers.Map('KeyType', 'int32', 'ValueType', 'any'); % Stores accurate and unobserved patient data
            
            if rand_bool
                reset(RandStream.getGlobalStream,sum(100*clock))
            end
            
            % Determining the number of states and observations from model parameters
            dim_size.x = size(model_true.A, 1);
            dim_size.y = size(model_true.C, 1);

            for ii=1:num_pats % for every patient
                % Initialisations of arrays
                x_cln = zeros(dim_size.x, 1, cens_time); % hidden states without disturbances
                x_true = zeros(dim_size.x, 1, cens_time); % hidden states with disturbances
                haz_true = zeros(1, cens_time); % Hazard function over time
                m_i = 0; % Number of iterations observed for patient
                delta_ev = 0; % event indicator
                surv_time = 0; % survival time
                y = zeros(dim_size.y, 1, cens_time); % longitudinal observations
                base_cov = zeros(size(model_true.g_s, 1), 1); % baseline covariates

                x0_tmp = model_true.mu_0; % set initial condition for patient ii

                % Baseline covariates are randomised from a normal distribution
                % First baseline covariate is the intercept
                if size(model_true.g_s, 1) == 1
                    q_ii = 1;
                else
                    q_ii = randn(size(model_true.g_s)); % N(0,1)
                    q_ii(1) = 1; % intercept
                end
                base_cov(:,1) = q_ii;

                % Survival of the patient - Utilising the inverse transform sampling
                unif_dist = rand(1); % the smaller this value, the higher chance they have to survive
                cum_haz = 0; % initialise cumulative hazard
                x_cln(:,:,1) = x0_tmp; % initialise x with no process noise

                % Finding the sqrt of the variance (to obtain std dev)
                sqrtV0 = chol(model_true.V_0, 'lower');
                x_true(:,:,1) = x0_tmp + sqrtV0 * randn(dim_size.x,1); % initialise x with disturbance

                % enforce a lower limit of 0 on x - assuming negative
                % biomarker values do not exist
                x_true(:,:,1) = max(0, x_true(:,:,1));
                
                % Finding the sqrt of the variance (to obtain std dev)
                sqrtGamma = chol(model_true.Gamma, 'lower');
                sqrtSigma = chol(model_true.Sigma, 'lower');
                
                % initialise y with measurement noise
                y(:,:,1) = model_true.C * x_true(:,:,1) + sqrtSigma * randn(dim_size.y,1); 

                % enforce a lower limit of 0 on y - assuming negative
                % biomarker values do not exist
                y(:,:,1) = max(0, y(:,:,1));

                % Let's assume first observation is always observed
                % Uncomment below if this is not a valid assumption
                % y_miss = rand(dim_size.y, 1) < frac_miss;
                % y_o(y_miss == 1,:,1,ii) = NaN;

                haz_true(1,1) = exp( model_true.g_s' * q_ii + model_true.a_s' * x_true(:,:,1)); % calculate initial hazard
                cum_haz = cum_haz + model_true.DeltaT * haz_true(1,1); % add hazard to cumulative hazard

                % Censoring time is assumed to be drawn from a uniform
                % distribution with the provided limits
                unif_dist_cens = rand(1);
                cens_num = (unif_dist_cens * diff(cens_range) / model_true.DeltaT) + cens_range(1) / model_true.DeltaT;
                max_t_points = min(cens_time, cens_num); % calculate max number of observations for patient

                if log(unif_dist) > - cum_haz % if cumulative hazard exceeds a certain value (based in the sampled value)
                    % T_i = tau_i - ( ln(S_i(t)) / h_i(t) )
                    surv_time = 0 * model_true.DeltaT - (log(unif_dist) + 0) / haz_true(1,1);
                    delta_ev = 1; % patient experienced event
                    m_i = 1; % patient had a single observation (at time = 0)
                else % If they had more observations before death
                    prev_cum_haz = cum_haz; % store current cumulative hazard
                    for j=2:floor(max_t_points) % for the rest of the observations
                        x_cln(:,:,j) = model_true.A * x_cln(:,:,j-1); % calculate x without disturbance
                        % calculate x with disturbance
                        x_true(:,:,j) = model_true.A * x_true(:,:,j-1) + model_true.G_mat * sqrtGamma * randn(size(sqrtGamma,2),1); 

                        % enforce a lower limit of 0 on x - assuming negative
                        % biomarker values do not exist
                        x_true(:,:,j) = max(0, x_true(:,:,j));

                        % calculate y with measurement noise
                        y(:,:,j) = model_true.C * x_true(:,:,j) + sqrtSigma * randn(dim_size.y,1); 
                        
                        % enforce a lower limit of 0 on y - assuming negative
                        % biomarker values do not exist
                        y(:,:,j) = max(0, y(:,:,j));

                        % randomise the missing observations based on the expected fraction of missing values
                        % Note: at some time points, we may have partial missing observations (i.e. only some of the
                        % biomarkers are missing)
                        y_miss = rand(dim_size.y, 1) < frac_miss;
                        y(y_miss == 1,:,j) = NaN;

                        haz_true(1,j) = exp( model_true.g_s' * q_ii + model_true.a_s' * x_true(:,:,j)); % calculate new hazard
                        cum_haz = cum_haz + model_true.DeltaT * haz_true(1,j); % calculate cumulative hazard

                        if log(unif_dist) > - cum_haz % if cumulative hazard exceeds a certain value
                            % T_i = tau_i - ( ln(S_i(t)) + H_i(t) / h_i(t) )
                            surv_time = (j-1) * model_true.DeltaT - (log(unif_dist) + prev_cum_haz) / haz_true(1,j);
                            delta_ev = 1; % patient experienced event
                            m_i = j; % patient had j observations
                            break; % patient died
                        end
                        
                        % if patient did not experience event
                        prev_cum_haz = cum_haz; % store the cumulative hazard function
                        
                    end % End of observation period
                    if delta_ev == 0 % if patient remained event-free
                        surv_time = (max_t_points) * model_true.DeltaT; % calculate survival time
                        m_i = j; % patient had a total of j observations
                    end
                end
                
                % Store data in map
                data_latent(ii) = struct('x_cln', x_cln, 'x_true', x_true, 'haz_true', haz_true);
                data_observed(ii) = struct('m_i', m_i, 'delta_ev', delta_ev, 'surv_time', surv_time, ...
                                           'y', y, 'base_cov', base_cov);
            end
        end
        
        
        function x_true_mat = map_struct_to_4dims_mat(map_tmp, field_name)
            x_true_mat = zeros([size(getfield(map_tmp(1), field_name)), map_tmp.Count]);
            
            for ii=1:map_tmp.Count
                x_true_mat(:,:,:,ii) = getfield(map_tmp(ii), field_name);
            end
        end
        
        % Plot the hidden state and Hazard function for the first num_plots patients
        function plot_pat_info(num_plots, t_arr, data_latent, data_obs)
            for ii=1:num_plots
                figure;
                
                m_i = data_obs(ii).m_i; % number of observations for patient ii
                % Plot the x values
                ax1 = subplot(2,1,1);
                for jj=1:size(data_latent(ii).x_true, 1)
                    plot(t_arr(1:m_i), data_latent(ii).x_true(jj,1:m_i),'DisplayName',sprintf('x%d', jj));
                    hold on;
                end
                
                if data_obs(ii).delta_ev == 0 % If patient did not experience event
                    title_n = sprintf('Patient was censored at time %.2f', data_obs(ii).surv_time);
                else % If patient experienced event
                    title_n = sprintf('Patient died at time %.2f', data_obs(ii).surv_time);
                end
                title(title_n);
                ax1.XLabel.String = 'Time';
                ax1.YLabel.String = 'x';
                xlim([t_arr(1) t_arr(end)]);
                legend();
                grid on;
                
                % Plot the hazard function
                ax2 = subplot(2,1,2);
                plot(t_arr(1:m_i), data_latent(ii).haz_true(1:m_i));
                hold on;
                if data_obs(ii).delta_ev == 0 % If patient did not experience event
                    xline(data_obs(ii).surv_time, '--g');
                else
                    xline(data_obs(ii).surv_time, '--r');
                end
                ax2.XLabel.String = 'Time';
                ax2.YLabel.String = 'Hazard';
                xlim([t_arr(1) t_arr(end)]);
                grid on;
            end
        end
        
        
        function [model_coef_init] = initialise_params(dim_size, data_observed, Delta)
            %%% Initialisation of parameters %%%
            % State space parameters
            a_bar_tmp = 0.5 * [eye(dim_size.dyn_states), eye(dim_size.dyn_states, dim_size.states - dim_size.dyn_states)];
            A_init = [a_bar_tmp; % dynamics matrix in canonical form
                 eye(dim_size.states - dim_size.dyn_states, dim_size.states)];
            C_init = [eye(dim_size.y), zeros(dim_size.y, dim_size.states - dim_size.y)]; % Observation matrix - assumed known
            Gamma_init = (0.5)^2 * eye(size(a_bar_tmp, 1)); % Disturbance matrix
            Sigma_init = (0.5)^2 * eye(size(C_init,1)); % Measurement error matrix
            G_mat = [eye(dim_size.dyn_states); % Matrix linking disturbance with the states
                     zeros(dim_size.states - dim_size.dyn_states, dim_size.dyn_states)];

            % Capture initial observation values for all patients
            y_init_mat = zeros(dim_size.y, data_observed.Count);
            for i=1:data_observed.Count
                y_init_mat(:,i) = data_observed(i).y(:,:,1);
            end

            % Initialise initial state values based on observation data
            % lagging states have same value as initial biomarker values observed
            if mod(dim_size.states, dim_size.y) == 0
                mu_0_init = repmat(nanmean(y_init_mat, 2), dim_size.states / dim_size.y, 1);
            else
                temp_mu = nanmean(y_init_mat, 2);
                mu_0_init = [repmat(temp_mu, floor(dim_size.states / dim_size.dyn_states), 1);
                             temp_mu(1:mod(dim_size.states, dim_size.dyn_states))];
            end

            V_0_init = zeros(dim_size.states, dim_size.states);
            for j=1:dim_size.states
                corr_dyn_state = mod(j-1, dim_size.y) + 1;
                V_0_init(j,j) = 1 * nanvar(y_init_mat(corr_dyn_state,:));
                if j>dim_size.y % for the lagging states, put a higher variance than the sample variance
                    V_0_init(j,j) = 3 * nanvar(y_init_mat(corr_dyn_state,:));
                end
            end

            % Survival parameters
            g_s_init = zeros(dim_size.base_cov, 1); % coefficients linking baseline covariates with hazard function
            a_s_init = zeros(dim_size.states, 1); % coefficients linking hidden states with hazard function

            model_coef_init = struct('A', A_init, 'C', C_init, ...
                                     'Gamma', Gamma_init, 'Sigma', Sigma_init, ...
                                     'g_s', g_s_init, 'a_s', a_s_init, ...
                                     'DeltaT', Delta, 'G_mat', G_mat, ...
                                     'mu_0', mu_0_init, 'V_0', V_0_init);
        end
        
        
        function plot_forecast_surv(pat_ii, landmark_t, t_arr, Delta)
            if pat_ii.delta_ev == 0
                title_n = sprintf('Patient was censored at time %.2f', pat_ii.surv_time);
            else
                title_n = sprintf('Patient died at time %.2f', pat_ii.surv_time);
            end

            figure;
            time_arr_for_surv = [t_arr t_arr(end)+Delta];
            plot(time_arr_for_surv, pat_ii.forecasts.surv);
            hold on;

            ylim([0, 1]);
            xlim([time_arr_for_surv(1), time_arr_for_surv(end)]);
            title(title_n);
            ylabel('Survival Probability');
            xlabel('Time');

            xline(landmark_t, 'm', 'LineWidth', 2)
            if pat_ii.delta_ev == 0
                xline(pat_ii.surv_time, 'g', 'LineWidth', 2);
                legend('Mod', 'Landmark', 'Censored');
            else
                xline(pat_ii.surv_time, 'k', 'LineWidth', 2);
                legend('Mod', 'Landmark', 'Death');
            end

        end
        
        
        function plot_bs_pe_auc(pat_data_reduced, bs_test, pe_test, auc_test, landmark_t, censor_time, t_multiplier, Delta)
            reduced_num_pats = double(pat_data_reduced.Count);
            surv_info_mat = zeros(reduced_num_pats, 2); % surv time and delta matrix

            for ii=1:reduced_num_pats
                surv_info_mat(ii,:) = [pat_data_reduced(ii).surv_time, pat_data_reduced(ii).delta_ev];
            end

            hist_count_arr = zeros(1, length(auc_test));

            for ii=1:length(hist_count_arr)
                hist_count_arr(ii) = sum(surv_info_mat(surv_info_mat(:,1) < (landmark_t + Delta * ii), 2));
            end

            max_time_shown = censor_time - landmark_t + Delta / 2;

            figure;
            t = tiledlayout(4, 1);
            nexttile;
            plot(Delta:Delta:Delta*length(bs_test), bs_test, '-*g');
            legend('Proposed Model', 'Location', 'southeast');
            title('Brier Score');
            xlim([0, max_time_shown]);
            grid on;
            nexttile;
            plot(Delta:Delta:Delta*length(pe_test), pe_test, '-*g');
            legend('Proposed Model', 'Location', 'southeast');
            title('Prediction Error');
            xlim([0, max_time_shown]);
            grid on;
            nexttile;
            plot(Delta:Delta:Delta*length(auc_test), auc_test, '-*g');
            legend('Proposed Model', 'Location', 'southeast');
            title('Area under ROC curve');
            xlim([0, max_time_shown]);
            grid on;
            nexttile;
            bar(Delta:Delta:Delta*length(hist_count_arr), [hist_count_arr; reduced_num_pats - hist_count_arr], "stacked");
            xlim([0, max_time_shown]);
            grid on;

            % ylabel('Frequency');
            title('Frequency of Events and Censored Observations');
            lgd = legend('Number of Event Observations', 'Number of Censored Observations');
            lgd.Location = 'southoutside';
            lgd.Orientation = 'horizontal';
            title(t, sprintf('Performance metrics across different horizons at landmark = %.1f years', ...
                landmark_t / t_multiplier));
            xlabel(t, 'Horizon (in months)');
        end
        
        
        function [auc_test_tmp] = AUC_fn(pat_data, landmark_t_tmp, max_horizon_t_tmp, model_coef_test) 
                                            
            % AUC - As explained by Blanche in their 2014 paper
            % +1 due to index starts from 1
            landmark_idx_tmp = floor(landmark_t_tmp / model_coef_test.DeltaT) + 1;
            
            % initialise the Brier Score array
            bs_test_tmp = zeros(1, int64(max_horizon_t_tmp / model_coef_test.DeltaT));
            
            no_of_pat_auc_test = double(pat_data.Count); % total number of patients at risk at landmark time
            
            surv_info_mat = zeros(no_of_pat_auc_test, 2); % surv time and delta matrix
            
            for ii=1:no_of_pat_auc_test
                surv_info_mat(ii,:) = [pat_data(ii).surv_time, pat_data(ii).delta_ev];
            end

            auc_test_tmp = zeros(1, int64(max_horizon_t_tmp / model_coef_test.DeltaT));

            for horizon_t_tmp=model_coef_test.DeltaT:model_coef_test.DeltaT:max_horizon_t_tmp %max_horizon_t

                % time at which we want to make prediction of survival
                t_est = landmark_t_tmp + horizon_t_tmp;

                % t_est_idx refers to the index for which we will use the survival
                % iteration. idx=1 refers to the index at time=0, which corresponds to
                % the survival of 1 (we know that the patient is alive at time=0).
                % Hence we will use surv(t_est_idx), which refers to the survival value
                % that used the x value of the previous time point, but corresponds to
                % the right index since we shift the index by 1 (i.e. at time=0, idx=1)
                t_est_idx = int64(t_est / model_coef_test.DeltaT) + 1;

                horizon_idx_tmp = int64(horizon_t_tmp / model_coef_test.DeltaT);
                
                weights_crit_test = zeros(no_of_pat_auc_test, 1); % Weightings to be used in AUC calculation

                [G_cens_test, G_time_test] = ecdf(surv_info_mat(:,1),'censoring',surv_info_mat(:,2),'function','survivor');
                
                [G_val, G_idx] = max(G_time_test(G_time_test < t_est));
                if isempty(G_idx)
                    G_idx = 1;
                end

                for ii=1:no_of_pat_auc_test
                    ind_tmp = 0; % Indicator variable for patient that experienced event
                    ind_risk = 0; % Indicator variable for patient still at risk
                    if surv_info_mat(ii,1) <= t_est && surv_info_mat(ii,2) == 1
                        ind_tmp = 1; % experienced event
                    end
                    if surv_info_mat(ii,1) >= t_est
                        ind_risk = 1; % still at risk
                    end

                    G_idx_tmp = G_idx;

                    if ind_tmp % if the patient experienced the event
                        % we modify the index accordingly
                        [G_val, G_idx_tmp] = max(G_time_test(G_time_test < surv_info_mat(ii,1)));
                        if isempty(G_idx_tmp)
                            G_idx_tmp = 1;
                        end
                    end

                    if ind_tmp || ind_risk
                        weights_crit_test(ii) = (ind_tmp + ind_risk) / G_cens_test(G_idx_tmp);
                    end

                end

                auc_num_sum = 0;
                auc_den_sum = 0;

                for ii=1:no_of_pat_auc_test
                    pat_ii = pat_data(ii);
                    D_i = pat_ii.surv_time <= t_est && pat_ii.delta_ev == 1;

                    if D_i
                        % Survival function for patient ii
                        surv_fn_ii = pat_ii.forecasts.surv;
                        W_i = weights_crit_test(ii);

                        for jj=1:no_of_pat_auc_test
                            pat_jj = pat_data(jj);
                            
                            % Survival function for patient jj
                            surv_fn_jj = pat_jj.forecasts.surv;
                            D_j = pat_jj.surv_time <= t_est && pat_jj.delta_ev == 1;
                            W_j = weights_crit_test(jj);

                            % Predicted survival given it is known they
                            % survived beyond landmark time
                            pred_Surv_i = surv_fn_ii(1,t_est_idx) / surv_fn_ii(1,landmark_idx_tmp);
                            pred_Surv_j = surv_fn_jj(1,t_est_idx) / surv_fn_jj(1,landmark_idx_tmp);

                            c_idx_ind = pred_Surv_i < pred_Surv_j;

                            auc_num_sum = auc_num_sum + (c_idx_ind * D_i * (1 - D_j) * W_i * W_j);
                            auc_den_sum = auc_den_sum + (D_i * (1 - D_j) * W_i * W_j);
                            
                        end
                    end
                end

                auc_test_tmp(horizon_idx_tmp) = auc_num_sum / auc_den_sum;
            end
            
        end
        
        
        
        function [pe_test_tmp] = Prediction_Error_fn(pat_data, landmark_t_tmp, max_horizon_t_tmp, model_coef_test) 
            
            % Prediction Error - as detailed by Rizopoulos in his 2017 paper
            % +1 due to index starts from 1
            landmark_idx_tmp = floor(landmark_t_tmp / model_coef_test.DeltaT) + 1;
            
            % initialise the Prediction Error array
            pe_test_tmp = zeros(1, int64(max_horizon_t_tmp / model_coef_test.DeltaT));
            
            no_of_pat_pe_test = double(pat_data.Count); % total number of patients at risk at landmark time

            for horizon_t_tmp=model_coef_test.DeltaT:model_coef_test.DeltaT:max_horizon_t_tmp
                t_est = landmark_t_tmp + horizon_t_tmp;
                t_est_idx = int64(t_est / model_coef_test.DeltaT)+1;

                horizon_idx_tmp = int64(horizon_t_tmp / model_coef_test.DeltaT);

                for ii=1:no_of_pat_pe_test
                    pat_ii = pat_data(ii);
                    surv_fn = pat_ii.forecasts.surv;
                    
                    ind_tmp = 0; % Indicator variable for patient that experienced event
                    ind_risk = 0; % Indicator variable for patient still at risk
                    ind_cens = 0; % Indicator variable for patients censored during horizon
                    if pat_ii.surv_time < t_est && pat_ii.delta_ev == 1
                        ind_tmp = 1; % experienced event
                    end
                    if pat_ii.surv_time >= t_est
                        ind_risk = 1; % still at risk
                    end
                    if pat_ii.surv_time <= t_est && pat_ii.delta_ev == 0
                        ind_cens = 1; % censored
                    end

                    if ind_tmp || ind_risk || ind_cens
                        pred_Surv = surv_fn(1,t_est_idx) / surv_fn(1,landmark_idx_tmp);
                        censor_time_idx = floor(pat_ii.surv_time / model_coef_test.DeltaT)+1;
                        pred_surv_pi = surv_fn(1,t_est_idx) / surv_fn(1,censor_time_idx);
                        
                        to_sum = ind_risk * (1 - pred_Surv)^2 + ind_tmp * (0 - pred_Surv)^2 + ...
                                    ind_cens * (pred_surv_pi * (1 - pred_Surv)^2 + (1 - pred_surv_pi) * (0 - pred_Surv)^2);
                        pe_test_tmp(horizon_idx_tmp) = pe_test_tmp(horizon_idx_tmp) + to_sum;

                    end

                end

                pe_test_tmp(horizon_idx_tmp) = pe_test_tmp(horizon_idx_tmp) / no_of_pat_pe_test; % Dividing by number of patients
            end
            
        end
        
        
        function [bs_test_tmp] = Brier_Score_fn(pat_data, landmark_t_tmp, max_horizon_t_tmp, model_coef_test) 
                                            
            % +1 due to index starts from 1
            landmark_idx_tmp = floor(landmark_t_tmp / model_coef_test.DeltaT) + 1;
            
            % initialise the Brier Score array
            bs_test_tmp = zeros(1, int64(max_horizon_t_tmp / model_coef_test.DeltaT));
            
            no_of_pat_bs_test = double(pat_data.Count); % total number of patients at risk at landmark time
            
            surv_info_mat = zeros(no_of_pat_bs_test, 2); % surv time and delta matrix
            
            for ii=1:no_of_pat_bs_test
                surv_info_mat(ii,:) = [pat_data(ii).surv_time, pat_data(ii).delta_ev];
            end

            % For every horizon we wish to consider
            for horizon_t_tmp=model_coef_test.DeltaT:model_coef_test.DeltaT:max_horizon_t_tmp

                % t_est is the time at which we want to make prediction of survival
                t_est = landmark_t_tmp + horizon_t_tmp;
                
                % t_est_idx refers to the index for which we will use the survival
                % iteration. idx=1 refers to the index at time=0, which corresponds to
                % the survival of 1 (we know that the patient is alive at time=0).
                % Hence we will use surv(t_est_idx), which refers to the survival value
                % that used the x value of the previous time point, but corresponds to
                % the right index since we shift the index by 1 (i.e. at time=0, idx=1)
                t_est_idx = int64(t_est / model_coef_test.DeltaT) + 1;

                horizon_idx_tmp = int64(horizon_t_tmp / model_coef_test.DeltaT);

                % After the censoring name, use the variable:
                % -> 1-delta_ev to obtain the KM estimate for survival function -> S(t) = P(T_i > t)
                % -> delta_ev to obtain the KM estimate for censoring function -> G(t) = P(C > t)
                [G_cens_test, G_time_test] = ecdf(surv_info_mat(:,1),'censoring',surv_info_mat(:,2),'function','survivor');

                % Finding the index for the Censored Kaplan Meier at t_est               
                [G_val, G_idx] = max(G_time_test(G_time_test < t_est));
                if isempty(G_idx)
                    G_idx = 1;
                end

                for ii=1:no_of_pat_bs_test
                    pat_ii = pat_data(ii);
                    surv_fn = pat_ii.forecasts.surv;
                    
                    ind_tmp = 0; % Indicator variable for patient that experienced event
                    ind_risk = 0; % Indicator variable for patient still at risk
                    if pat_ii.surv_time <= t_est && pat_ii.delta_ev == 1
                        ind_tmp = 1; % experienced event
                    end
                    if pat_ii.surv_time >= t_est
                        ind_risk = 1; % still at risk
                    end

                    G_idx_tmp = G_idx;

                    if ind_tmp % if the patient experienced the event
                        % we modify the index accordingly
                        [G_val, G_idx_tmp] = max(G_time_test(G_time_test < pat_ii.surv_time));
                        if isempty(G_idx_tmp)
                            G_idx_tmp = 1;
                        end
                    end

                    if ind_tmp || ind_risk
                        pred_Surv = surv_fn(1,t_est_idx) / surv_fn(1,landmark_idx_tmp);
                        to_sum = (0 - pred_Surv)^2 * ind_tmp / G_cens_test(G_idx_tmp) + ...
                                    (1 - pred_Surv)^2 * ind_risk / G_cens_test(G_idx);
                        bs_test_tmp(horizon_idx_tmp) = bs_test_tmp(horizon_idx_tmp) + to_sum;

                    end

                end

                bs_test_tmp(horizon_idx_tmp) = bs_test_tmp(horizon_idx_tmp) / no_of_pat_bs_test; % Dividing by number of patients
            end
            
        end
        
        
        function [pat_data_out] = predict_fn(pat_data, t_est_idx_tmp, max_censor_time, model_coef_test, controls)
            %%% To be used only in simulations to compare the estimated
            %%% model's performance to that of the true model when all data
            %%% is made available (this function uses the smoothed output
            %%% given all available data).
            
            max_num_pts = ceil(max_censor_time / model_coef_test.DeltaT)+1;
            
            pat_data_out = containers.Map('KeyType', 'int32', 'ValueType', 'any');
            
            for ii=1:pat_data.Count % iterate for every patient

                pat_ii = pat_data(ii);
                
                % Set initial conditions
                pat_ii.mu_0 = model_coef_test.mu_0;
                pat_ii.V_0 = model_coef_test.V_0;

                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %%% Forward recursion - Standard/Modified RTS Filter %%%
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                [pat_ii.mu_tilde, pat_ii.V_tilde] = ...
                    LSDSM_ALLFUNCS.Kalman_filter(pat_ii, model_coef_test, max_num_pts, controls.mod_KF);

                %%% Backward recursion - RTS Smoother %%%
                [pat_ii.predictions.mu, pat_ii.predictions.V, pat_ii.predictions.J] = ...
                    LSDSM_ALLFUNCS.Kalman_smoother(pat_ii, model_coef_test, max_num_pts);

                % Survival curve estimation and forecasts
                cum_haz_fn = 0;
                haz_fn = zeros(1, max_num_pts);

                % patient's survival at time = 0 is 1
                pat_ii.predictions.surv(:,1) = 1;

                for j=1:t_est_idx_tmp
                    % the jth value of x will predict the survival of patient at (j+1)
                    haz_fn(:,j) = exp(model_coef_test.g_s' * pat_ii.base_cov) * ...
                                                          exp(model_coef_test.a_s' * pat_ii.predictions.mu(:,1,j));
                    cum_haz_fn = cum_haz_fn + model_coef_test.DeltaT * haz_fn(:,j);
                    pat_ii.predictions.surv(:,j+1) = exp(-cum_haz_fn);
                end
                
                pat_data_out(ii) = pat_ii;
            end
            
        end
        
        
        function [reduced_pat_data] = forecast_fn(pat_data, landmark_t_tmp, t_est_idx_tmp, ...
                                                  max_censor_time, model_coef_test, controls)
            % +1 due to index starts from 1
            landmark_idx_tmp = floor(landmark_t_tmp / model_coef_test.DeltaT) + 1;
            
            max_num_pts = ceil(max_censor_time / model_coef_test.DeltaT)+1;
            
            % Stores patient data and forecasts for surviving patients beyond landmark time
            reduced_pat_data = containers.Map('KeyType', 'int32', 'ValueType', 'any');
            
            red_ii = 0; % used to keep track of the number of patients that survive beyond the landmark point

            for ii=1:pat_data.Count % iterate for every patient

                pat_ii = pat_data(ii);
                
                if pat_ii.surv_time > landmark_t_tmp % forecast if they survived beyond the landmark
                    
                    % increase the iterative for the reduced patient data set
                    red_ii = red_ii + 1;
                    
                    % Set initial conditions
                    pat_ii.mu_0 = model_coef_test.mu_0;
                    pat_ii.V_0 = model_coef_test.V_0;

                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    %%% Forward recursion - Standard/Modified RTS Filter %%%
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    [pat_ii.mu_tilde, pat_ii.V_tilde] = ...
                        LSDSM_ALLFUNCS.Kalman_filter(pat_ii, model_coef_test, max_num_pts, controls.mod_KF);

                    % Utilise the RTS filter output until the landmark (one time step before)
                    % Here we do not use the RTS smoother since future values may
                    % affect the current values (which in turn will affect the forecasts)
                    pat_ii.forecasts.mu(:,:,1:landmark_idx_tmp-1) = pat_ii.mu_tilde(:,:,1:landmark_idx_tmp-1);
                    pat_ii.forecasts.V(:,:,1:landmark_idx_tmp-1) = pat_ii.V_tilde(:,:,1:landmark_idx_tmp-1);

                    % We cannot use the output of the RTS filter at the landmark time,
                    % as this is utilising survival information of the next period.
                    % However, we can use a single step of the standard RTS filter,
                    % where we utilise just the longitudinal data available at that time.
                    [mu_filt, V_filt, K_filt, P_filt] = LSDSM_ALLFUNCS.KF_single_step(pat_ii.mu_tilde(:,:,landmark_idx_tmp-1), ...
                                                                                      pat_ii.V_tilde(:,:,landmark_idx_tmp-1), ...
                                                                                      pat_ii.y(:,:,landmark_idx_tmp), model_coef_test);

                    % The above filtering will allow us to update the hidden states based
                    % on the longitudinal information available at the landmark. This will
                    % result in predictions starting from the next iteration.
                    pat_ii.forecasts.mu(:,:,landmark_idx_tmp) = mu_filt;
                    pat_ii.forecasts.V(:,:,landmark_idx_tmp) = V_filt;

                    %%%%%%%%%%%%%%%%%
                    %%% Forecasts %%%
                    %%%%%%%%%%%%%%%%%
                    for j=landmark_idx_tmp+1:t_est_idx_tmp
                        % mu_pred_temp_test_arr(:,:,j,ii) = model_coef_est.A * mu_pred_temp_test_arr(:,:,j-1,ii);
                        pat_ii.forecasts.mu(:,:,j) = model_coef_test.A * pat_ii.forecasts.mu(:,:,j-1);
                        pat_ii.forecasts.V(:,:,j) = model_coef_test.A * pat_ii.forecasts.V(:,:,j-1) * model_coef_test.A' ...
                                                                + model_coef_test.G_mat * model_coef_test.Gamma * model_coef_test.G_mat';
                    end
                    
                    % Survival curve estimation and forecasts
                    cum_haz_fn = 0;
                    haz_fn = zeros(1, max_num_pts);
                    
                    % patient's survival at time = 0 is 1
                    pat_ii.forecasts.surv(:,1) = 1;

                    for j=1:t_est_idx_tmp
                        % the jth value of x will predict the survival of patient at (j+1)
                        haz_fn(:,j) = exp(model_coef_test.g_s' * pat_ii.base_cov) * ...
                                                              exp(model_coef_test.a_s' * pat_ii.forecasts.mu(:,1,j));
                        cum_haz_fn = cum_haz_fn + model_coef_test.DeltaT * haz_fn(:,j);
                        pat_ii.forecasts.surv(:,j+1) = exp(-cum_haz_fn);
                    end
                    
                    reduced_pat_data(red_ii) = pat_ii;
                end
            end
        end
        
        
        
        function [model_coef_est_out, max_iter, param_traj, RTS_arrs] = LSDSM_EM(dim_size, pat_data, controls, max_censor_time)
            % FUNCTION NAME:
            %   LSDSM_EM
            %
            % DESCRIPTION:
            %   Executes the Expectation Maximisation (EM) algorithm to
            %   find the parameters for the Linear State space Dynamic
            %   Survival Model (LSDSM), which is a joint model for
            %   longitudinal and survival data.
            %   Longitudinal Sub-process: Linear Gaussian State Space Model
            %   Survival Sub-process: Exponential Survival Model
            %   
            %   For more information, reader is referred to the paper
            %   titled "Individualised Survival Predictions using State
            %   Space Model with Longitudinal and Survival Data".
            %
            % INPUT:
            %   dim_size - (struct) Contains the number of states,
            %   "dynamic" states, and biomarkers
            %   pat_data - (Map) Contains patient data with every key
            %   representing a single patient. In each entry, a struct with
            %   number of iterations observed, boolean representing whether
            %   patient experienced event, survival time, longitudinal
            %   biomarkers, and baseline covariates
            %   controls - (struct) Controls for the EM algorithm,
            %   including number of iterations, maximum parameter 
            %   difference for stopping criteria, initial parameters, fixed
            %   parameters, and boolean for modified filter equations
            %   max_censor_time - (double) Indicates the maximum time to
            %   observe every patient
            %
            % OUTPUT:
            %   model_coef_est_out - (struct) Estimated Parameters
            %   max_iter - (double) Number of EM iterations executed
            %   param_traj_tmp - (struct) Evolution of parameter values
            %   over EM iterations
            %   RTS_arrs - (struct) Contains the filtered and smoothed
            %   outputs of the hidden state trajectories.
            %
            % REVISION HISTORY:
            %   17/12/2022 - mcauchi1
            %       * Initial implementation
            %

            % examine the number of patients
            num_pats = pat_data.Count;

            % set the initial parameters
            model_coef_est = controls.init_params;
            
            % maximum number of time points to consider
            max_num_pts = ceil(max_censor_time / model_coef_est.DeltaT);

            controls.update_pop_mu = 0;
            if size(model_coef_est.mu_0, 3) == 1 % we are updating a single population initial states
                controls.update_pop_mu = 1;
            end

            % Initialise arrays to track parameter changes
            param_traj.A = zeros(dim_size.states, dim_size.states, controls.EM_iters);
            param_traj.C = zeros(dim_size.y, dim_size.states, controls.EM_iters);
            param_traj.Gamma = zeros(dim_size.dyn_states, dim_size.dyn_states, controls.EM_iters);
            param_traj.Sigma = zeros(dim_size.y, dim_size.y, controls.EM_iters);
            param_traj.g_s = zeros(size(pat_data(1).base_cov,1), 1, controls.EM_iters);
            param_traj.a_s = zeros(dim_size.states, 1, controls.EM_iters);
            param_traj.mu_0 = zeros(dim_size.states, 1, num_pats, controls.EM_iters);
            param_traj.V_0 = zeros(dim_size.states, dim_size.states, num_pats, controls.EM_iters);
            
            log_like_val_tot_arr = zeros(num_pats, controls.EM_iters);

            % Set the first value (in iter_EM) as the initial estimates
            param_traj.A(:,:,1) = model_coef_est.A;
            param_traj.C(:,:,1) = model_coef_est.C;
            param_traj.Gamma(:,:,1) = model_coef_est.Gamma;
            param_traj.Sigma(:,:,1) = model_coef_est.Sigma;
            param_traj.g_s(:,:,1) = model_coef_est.g_s;
            param_traj.a_s(:,:,1) = model_coef_est.a_s;

            if controls.update_pop_mu
                param_traj.mu_0(:,:,1) = model_coef_est.mu_0;
                param_traj.V_0(:,:,1) = model_coef_est.V_0;
            else
                param_traj.mu_0(:,:,:,1) = model_coef_est.mu_0;
                param_traj.V_0(:,:,:,1) = model_coef_est.V_0;
            end

            % Store the states obtained from the RTS filter/smoother
            
            RTS_arrs.mu_tilde = zeros(dim_size.states, 1, max_num_pts, num_pats);
            RTS_arrs.V_tilde = zeros(dim_size.states, dim_size.states, max_num_pts, num_pats);
            RTS_arrs.mu_hat = zeros(dim_size.states, 1, max_num_pts, num_pats);
            RTS_arrs.V_hat = zeros(dim_size.states, dim_size.states, max_num_pts, num_pats);
            

            for j=2:controls.EM_iters % iterate the EM algorithm (iter_EM-1) times
                if controls.verbose
                    if mod(j,10) == 0 % Feedback every 10 iterations
                        fprintf("EM Iteration %d / %d. Max Parameter Difference: %.6f \n", j, controls.EM_iters, param_max_diff);
                    end
                end

                % Initialisation of summations to be used at the E and M steps
                % E_sums is a struct containing all summations of the
                % expectations required
                % Stores the sum across all patients of sum_{n=2}^{N} E[x(n) x(n-1)']
                E_sums.xn_xnneg1_from2 = zeros(dim_size.states,dim_size.states);
                % Stores the sum across all patients of sum_{n=i}^{N-1} E[x(n) x(n-1)']
                E_sums.xn_xn_tillNneg1 = zeros(dim_size.states,dim_size.states);
                % Stores the sum across all patients of sum_{n=2}^{N} E[x(n) x(n)']
                E_sums.xn_xn_from2 = zeros(dim_size.states,dim_size.states);
                % Stores the sum across all patients of sum_{n=1}^{N} E[x(n) x(n)']
                E_sums.xn_xn = zeros(dim_size.states,dim_size.states);
                % Stores the sum across all patients of sum_{n=1}^{N} E[x(n)]
                E_sums.xn = zeros(dim_size.states,1);
                % Stores the sum across all patients of sum_{n=1}^{N} y(n) E[x(n)']
                E_sums.yn_xn = zeros(dim_size.y,dim_size.states);
                % Stores the sum across all patients of sum_{n=1}^{N} y(n) y(n)'
                E_sums.yn_yn = zeros(dim_size.y, dim_size.y);
                
                % Stores the sum across all patients of sum_{n=2}^{N} E[x(n) x(n)']
                E_sums.barxn_xnneg1_from3 = zeros(dim_size.dyn_states,dim_size.states);
                % Stores the sum across all patients of sum_{n=i}^{N-1} E[x(n) x(n-1)']
                E_sums.xn_xn_from2_tillNneg1 = zeros(dim_size.states,dim_size.states);

                % Stores the sum across all patients of sum_{n=1}^{N} E[x_bar(n) x(n-1)']
                E_sums.barxn_xnneg1_from2 = zeros(dim_size.dyn_states,dim_size.states);
                % Stores the sum across all patients of sum_{n=1}^{N} E[x_bar(n) x(n-1)']
                E_sums.barxn_barxn_from2 = zeros(dim_size.dyn_states,dim_size.dyn_states);

                E_sums.x0 = zeros(dim_size.states, 1);
                E_sums.x0_x0 = zeros(dim_size.states, dim_size.states);
                
                for ii=1:num_pats % iterate for every patient

                    % Capture information from current patient
                    pat_ii = pat_data(ii);
                    
                    % Set initial state conditions for patient ii
                    if controls.update_pop_mu
                        pat_ii.mu_0 = model_coef_est.mu_0;
                        pat_ii.V_0 = model_coef_est.V_0;
                    else
                        pat_ii.mu_0 = model_coef_est.mu_0(:,:,ii);
                        pat_ii.V_0 = model_coef_est.V_0(:,:,ii);
                    end
                    
                    %%%%%%%%%%%%%%
                    %%% E Step %%%
                    %%%%%%%%%%%%%%

                    %%% Forward recursion - Standard/Modified RTS Filter %%%
                    [pat_ii.mu_tilde, pat_ii.V_tilde, log_like_val] = ...
                        LSDSM_ALLFUNCS.Kalman_filter(pat_ii, model_coef_est, max_num_pts, controls.mod_KF);
                    
                    log_like_val_tot_arr(ii,j) = log_like_val;
                    
                    %%% Backward recursion - RTS Smoother %%%
                    [pat_ii.mu_hat, pat_ii.V_hat, pat_ii.J_hat] = ...
                        LSDSM_ALLFUNCS.Kalman_smoother(pat_ii, model_coef_est, max_num_pts);

                    %%% Compute the required Expectations %%%
                    [pat_ii.E] = LSDSM_ALLFUNCS.compute_E_fns(pat_ii, model_coef_est, max_num_pts);

                    %%%%%%%%%%%%%%
                    %%% M Step %%%
                    %%%%%%%%%%%%%%
                    % These are summing the expectations across every patient for the
                    % relevant times. These will then be utilised to identify the new
                    % parameter estimations.
                    [E_sums] = LSDSM_ALLFUNCS.sum_E_fns(E_sums, pat_ii);

                    % Store the RTS filter and smoother outputs
                    RTS_arrs.mu_tilde(:,:,:,ii) = pat_ii.mu_tilde;
                    RTS_arrs.V_tilde(:,:,:,ii) = pat_ii.V_tilde;
                    RTS_arrs.mu_hat(:,:,:,ii) = pat_ii.mu_hat;
                    RTS_arrs.V_hat(:,:,:,ii) = pat_ii.V_hat;

                end
                
                model_coef_new = LSDSM_ALLFUNCS.M_step_all_pats(pat_data, E_sums, RTS_arrs, model_coef_est, controls);

                % store the old coefficients for stopping criterion
                model_coef_est_old_cmp = rmfield(model_coef_est, {'DeltaT', 'G_mat', 'mu_0', 'V_0'});

                % Update the estimates to be used in the next iteration
                model_coef_est = model_coef_new;
                
                % Store the updated estimates in the respective arrays
                param_traj.A(:,:,j) = model_coef_est.A;
                param_traj.C(:,:,j) = model_coef_est.C;
                param_traj.Gamma(:,:,j) = model_coef_est.Gamma;
                param_traj.Sigma(:,:,j) = model_coef_est.Sigma;
                param_traj.g_s(:,:,j) = model_coef_est.g_s;
                param_traj.a_s(:,:,j) = model_coef_est.a_s;
                if controls.update_pop_mu
                    param_traj.mu_0(:,:,j) = model_coef_est.mu_0;
                    param_traj.V_0(:,:,j) = model_coef_est.V_0;
                else
                    param_traj.mu_0(:,:,:,j) = model_coef_est.mu_0;
                    param_traj.V_0(:,:,:,j) = model_coef_est.V_0;
                end

                model_coef_est_new_cmp = rmfield(model_coef_est, {'DeltaT', 'G_mat', 'mu_0', 'V_0'});

                model_coef_est_out = model_coef_est;

                % Convert structs to vertical arrays to compare parameters
                old_params_pre = cellfun(@(x) reshape(x,[],1), struct2cell(model_coef_est_old_cmp), 'UniformOutput', false);
                old_params_vec = vertcat(old_params_pre{:});

                new_params_pre = cellfun(@(x) reshape(x,[],1), struct2cell(model_coef_est_new_cmp), 'UniformOutput', false);
                new_params_vec = vertcat(new_params_pre{:});

                param_diff = new_params_vec - old_params_vec;

                % Find maximum absolute parameter difference
                param_max_diff = max(abs(param_diff));

                % If stopping criteria is reached
                if param_max_diff < controls.max_param_diff
                    break; % Break from EM algorithm for loop
                end
            end

            max_iter = j;
            
            if controls.allow_plots
                figure;
                plot(sum(log_like_val_tot_arr(:,2:j), 1))
                title('Log likelihood over EM iterations');
                ylabel('Log likelihood');
                xlabel('EM iteration');
            end

            fprintf('Maximum Parameter Difference: %6f at iteration %d \n', param_max_diff, j);
            
        end
        
        
        function [param_diff, param_diff_percent] = find_model_param_diff(true_model_coef, model_coef_est)
            % Finds parameter difference between two models
            % Removes DeltaT, G_mat, mu_0, V_0 from this parameter
            % difference
            true_model_coef_cmp = rmfield(true_model_coef, {'DeltaT', 'G_mat', 'mu_0', 'V_0'});
            model_coef_est_cmp = rmfield(model_coef_est, {'DeltaT', 'G_mat', 'mu_0', 'V_0'});
            
            % Convert structs to vertical arrays to compare parameters
            true_params_pre = cellfun(@(x) reshape(x,[],1), struct2cell(true_model_coef_cmp), 'UniformOutput', false);
            true_params_vec = vertcat(true_params_pre{:});

            est_params_pre = cellfun(@(x) reshape(x,[],1), struct2cell(model_coef_est_cmp), 'UniformOutput', false);
            est_params_vec = vertcat(est_params_pre{:});

            param_diff = true_params_vec - est_params_vec;
            
            param_diff_percent = abs((true_params_vec - est_params_vec) ./ true_params_vec);
            
            param_diff_percent(isinf(param_diff_percent)) = NaN; % In case of a division by zero
        end
        
        
        function rmse_states = find_rmse_states(data_latent, data_observed, comp_signal)
            % Find the root mean square error between comp_signal and true
            % hidden states (only valid for simulations)
            total_sse_val = zeros(size(comp_signal,1), 1);
            total_obs_test = 0;
            t_dim = 3;

            for ii=1:data_observed.Count
                true_sig_ii = data_latent(ii).x_true(:,:,1:data_observed(ii).m_i);
                comp_sig_ii = comp_signal(:,:,1:data_observed(ii).m_i,ii);
                total_obs_test = total_obs_test + data_observed(ii).m_i; % counting the total number of observations


                [sse_val, mse_val, rmse_val] = LSDSM_ALLFUNCS.rmse_fn(true_sig_ii, comp_sig_ii, t_dim);
                total_sse_val(:,1) = total_sse_val(:,1) + sse_val;
            end

            mse_states = total_sse_val / total_obs_test;
            rmse_states = sqrt(mse_states);
        end
        
        
        function rmse_surv = find_rmse_surv(true_surv_fn, data_observed, comp_signal)
            % Find the root mean square error between comp_signal and true
            % hidden states (only valid for simulations)
            total_sse_val = zeros(size(comp_signal,1), 1);
            total_obs_test = 0;
            t_dim = 2;

            for ii=1:data_observed.Count
                % start from 2 since the first survival probability is
                % always equal to 1
                true_sig_ii = true_surv_fn(:,2:data_observed(ii).m_i+1, ii);
                comp_sig_ii = comp_signal(:,2:data_observed(ii).m_i+1, ii);
                total_obs_test = total_obs_test + data_observed(ii).m_i; % counting the total number of observations


                [sse_val, mse_val, rmse_val] = LSDSM_ALLFUNCS.rmse_fn(true_sig_ii, comp_sig_ii, t_dim);
                total_sse_val(:,1) = total_sse_val(:,1) + sse_val;
            end

            mse_surv = total_sse_val / total_obs_test;
            rmse_surv = sqrt(mse_surv);
        end
        
        
        function surv_fn = surv_curve_filt(model_coef_est, data_observed, state_vals)
            surv_fn = zeros(1, size(state_vals,3)+1, data_observed.Count);

            for ii=1:data_observed.Count
                % Initialise hazard
                cum_haz = 0;
                haz_fn = zeros(1, size(state_vals,3), data_observed.Count);
                
                % capture (predicted) hidden states and baseline covariates
                x_ii = state_vals(:,:,:,ii);
                base_cov_ii = data_observed(ii).base_cov;

                % at time = 0, surv = 1
                surv_fn(:,1,ii) = 1;

                m_i = data_observed(ii).m_i;
                for j=1:m_i
                    haz_fn(:,j) = exp(model_coef_est.g_s' * base_cov_ii) * exp(model_coef_est.a_s' * x_ii(:,1,j));
                    cum_haz = cum_haz + model_coef_est.DeltaT * haz_fn(:,j);
                    surv_fn(:,j+1,ii) = exp(-cum_haz);
                end
            end
        end
        
        
        function E_cdll_red = calc_E_cdll_red(coeffs, pat_info, x_i1, mu_0_tmp, V_0_tmp)

            A_tmp = coeffs{1};
            C_tmp = coeffs{2};
            Gamma_tmp = coeffs{3};
            Sigma_tmp = coeffs{4};
            g_s_tmp = coeffs{5};
            a_s_tmp = coeffs{6};
            % model_coef_est.DeltaT = coeffs{7};
            % G_tmp = coeffs{8};
            
            num_pats = pat_info{1};
            DeltaT = pat_info{2};
            iter_obs_pats_tmp = pat_info{3};
            surv_time_pats_tmp = pat_info{4};
            delta_ev_pats_tmp = pat_info{5};
            
            q_pats = pat_info{8};
            
            dim_size.dyn_states = size(Gamma_tmp,1);
            dim_size.states = size(A_tmp, 1);
            dim_size.y = size(C_tmp,1);
            
            % Calculation of the expectation of the complete data log likelihood
            E_cdll_red = 0;

            for ii=1:1
                N_tmp = iter_obs_pats_tmp(ii); % number of observations
                surv_time_tmp = surv_time_pats_tmp(ii); % patient's survival time
                event_ind_tmp = delta_ev_pats_tmp(ii); % indicator for event
                q_pats_i = q_pats(:,:,ii);

                for i=1:1 % For every observation
                    % Check if patient experienced event in current time frame
                    delta_ij = (N_tmp == i) & (event_ind_tmp == 1);

                    % By default, Delta_t is equal to the time step
                    Delta_t = DeltaT;
                    % However, if patient experiences event or is censored
                    % in the current period, then we have to adjust Delta_t
                    % accordingly
                    if surv_time_tmp < i
                        Delta_t = surv_time_tmp - (i - 1)*DeltaT;
                    end

                    scalar_tmp = Delta_t * exp(g_s_tmp' * q_pats_i) * ...
                        exp(a_s_tmp' * x_i1 + 1/2 * a_s_tmp' * V_0_tmp * a_s_tmp);

                    % survival expectation
                    E_cdll_red = E_cdll_red + delta_ij * (g_s_tmp' * q_pats_i + a_s_tmp' * x_i1) - scalar_tmp;
                end
                
                % First hidden state expectation
                E_cdll_red = E_cdll_red -1/2 * ( dim_size.states * log(2*pi) + log(det(V_0_tmp)) + trace( V_0_tmp^-1 * ...
                            (V_0_tmp + (x_i1 - mu_0_tmp) * (x_i1 - mu_0_tmp)')));
            end

            
        end
        
        
        function E_cdll = calc_E_cdll(coeffs, pat_info, cell_of_sums, mu_0_tmp, V_0_tmp)
            
            % sum_xn_tmp = cell_of_sums{1};
            sum_yn_yn_tmp = cell_of_sums{2};
            sum_yn_xn_tmp = cell_of_sums{3};
            sum_xn_xn_tmp = cell_of_sums{4};
%             E_sums.xn_xn_from2 = cell_of_sums{5};
            sum_xn_xn_tillNneg1_tmp = cell_of_sums{6};
%             E_sums.xn_xnneg1_from2 = cell_of_sums{7};
            sum_barxn_barxn_from2_tmp = cell_of_sums{8};
            sum_barxn_xnneg1_from2_tmp = cell_of_sums{9};
%             E_sums.x0 = cell_of_sums{10};
%             E_sums.x0_x0 = cell_of_sums{11};
% 
%             E_sums.barxn_xnneg1_from3 = cell_of_sums{12};
%             E_sums.xn_xn_from2_tillNneg1 = cell_of_sums{13};

            A_tmp = coeffs{1};
            C_tmp = coeffs{2};
            Gamma_tmp = coeffs{3};
            Sigma_tmp = coeffs{4};
            g_s_tmp = coeffs{5};
            a_s_tmp = coeffs{6};
            % model_coef_est.DeltaT = coeffs{7};
            % G_tmp = coeffs{8};
            
            num_pats = pat_info{1};
            DeltaT = pat_info{2};
            iter_obs_pats_tmp = pat_info{3};
            surv_time_pats_tmp = pat_info{4};
            delta_ev_pats_tmp = pat_info{5};
            mu_hat_pats_arr = pat_info{6};
            V_hat_pats_arr = pat_info{7};
            q_pats = pat_info{8};
            
            dim_size.dyn_states = size(sum_barxn_barxn_from2_tmp,1);
            dim_size.states = size(A_tmp, 1);
            dim_size.y = size(C_tmp,1);
            
            % Calculation of the expectation of the complete data log likelihood
            E_cdll = 0;

            for ii=1:num_pats
                N_tmp = iter_obs_pats_tmp(ii); % number of observations
                surv_time_tmp = surv_time_pats_tmp(ii); % patient's survival time
                event_ind_tmp = delta_ev_pats_tmp(ii); % indicator for event

                mu_hat_tmp = mu_hat_pats_arr(:,:,:,ii);
                V_hat_tmp = V_hat_pats_arr(:,:,:,ii);
                q_pats_i = q_pats(:,:,ii);

                for i=1:N_tmp % For every observation
                    % Check if patient experienced event in current time frame
                    delta_ij = (N_tmp == i) & (event_ind_tmp == 1);

                    % By default, Delta_t is equal to the time step
                    Delta_t = DeltaT;
                    % However, if patient experiences event or is censored
                    % in the current period, then we have to adjust Delta_t
                    % accordingly
                    if surv_time_tmp < i
                        Delta_t = surv_time_tmp - (i - 1)*DeltaT;
                    end

                    scalar_tmp = Delta_t * exp(g_s_tmp' * q_pats_i) * ...
                        exp(a_s_tmp' * mu_hat_tmp(:,:,i) + 1/2 * a_s_tmp' * V_hat_tmp(:,:,i) * a_s_tmp);

                    % survival expectation
                    E_cdll = E_cdll + delta_ij * (g_s_tmp' * q_pats_i + a_s_tmp' * mu_hat_tmp(:,:,i)) - scalar_tmp;
                end
                
                % First hidden state expectation
                E_cdll = E_cdll -1/2 * ( dim_size.states * log(2*pi) + log(det(V_0_tmp)) + trace( V_0_tmp^-1 * ...
                            (V_hat_tmp(:,:,1) + (mu_hat_tmp(:,:,1) - mu_0_tmp) * (mu_hat_tmp(:,:,1) - mu_0_tmp)')));
            end

            % observations expectation
            E_cdll = E_cdll - 1/2 * ( sum(iter_obs_pats_tmp) * (dim_size.y * log(2*pi) + log(det(Sigma_tmp))) + ...
                        trace(Sigma_tmp^-1 * (sum_yn_yn_tmp - sum_yn_xn_tmp * C_tmp' ...
                                    - C_tmp * sum_yn_xn_tmp' + C_tmp * sum_xn_xn_tmp * C_tmp' )));

            A_bar_tmp = A_tmp(1:dim_size.dyn_states,:);

            % Hidden states expectation (except the first)
            E_cdll = E_cdll - 1/2 * ( (sum(iter_obs_pats_tmp) - num_pats) * ( dim_size.states * log(2*pi) + log(det(Gamma_tmp))) + ...
                        trace(Gamma_tmp^-1 * (sum_barxn_barxn_from2_tmp - sum_barxn_xnneg1_from2_tmp * A_bar_tmp' ...
                                - A_bar_tmp * sum_barxn_xnneg1_from2_tmp' + A_bar_tmp * sum_xn_xn_tillNneg1_tmp * A_bar_tmp' )));
        end
        
        
        % Expectation including only survival data
        function Gsurv_val = Gsurv(g_a_s_prev, pat_info)
            
            % variables used to evaluate this derivative (d E[.] / d [g_s a_s])
            num_pats = pat_info{1};
            deltaT = pat_info{2};
            iter_obs_pats = pat_info{3};
            surv_time_pats = pat_info{4};
            delta_ev_pats = pat_info{5};
            mu_hat_pats = pat_info{6};
            V_hat_pats = pat_info{7};
            q_pats = pat_info{8};

            % Store the previous values of g_s and a_s
            g_s_prev = g_a_s_prev(1:size(q_pats,1), 1);
            a_s_prev = g_a_s_prev(size(q_pats,1)+1:end, 1);

            % Initiate the output value to zero
            Gsurv_val = 0;

            for ii=1:num_pats % For every patient
                N_tmp = iter_obs_pats(ii); % number of observations
                surv_time_tmp = surv_time_pats(ii); % patient's survival time
                event_ind_tmp = delta_ev_pats(ii); % indicator for event

                mu_hat_tmp = mu_hat_pats(:,:,:,ii);
                V_hat_tmp = V_hat_pats(:,:,:,ii);
                q_pats_i = q_pats(:,:,ii);

                for i=1:N_tmp % For every observation
                    % Check if patient experienced event in current time frame
                    delta_ij = (N_tmp == i) & (event_ind_tmp == 1);
                    
                    % By default, Delta_t is equal to the time step
                    Delta_t = deltaT;
                    % However, if patient experiences event or is censored
                    % in the current period, then we have to adjust Delta_t
                    % accordingly
                    if surv_time_tmp < i
                        Delta_t = surv_time_tmp - (i - 1)*deltaT;
                    end
                    
                    scalar_tmp = Delta_t * exp(g_s_prev' * q_pats_i) * ...
                        exp(a_s_prev' * mu_hat_tmp(:,:,i) + 1/2 * a_s_prev' * V_hat_tmp(:,:,i) * a_s_prev);

                    Gsurv_val = Gsurv_val + delta_ij * g_s_prev' * q_pats_i + ...
                        + delta_ij * a_s_prev' * mu_hat_tmp(:,:,i) - scalar_tmp;
                end
            end
        end


        % First Derivative of the Expectation with respect to g_s and alpha (a_s)
        function dGout = dGdx(g_a_s_prev, dG_data)

            % Store the previous values of g_s and a_s
            g_s_prev = g_a_s_prev(1:size(dG_data.pat_data(1).base_cov,1), 1);
            a_s_prev = g_a_s_prev(size(dG_data.pat_data(1).base_cov,1)+1:end, 1);

            % Initiate the derivatives to zero
            dGdg_s = 0;
            dGda_s = 0;

            for ii=1:dG_data.pat_data.Count % For every patient
                base_cov_ii = dG_data.pat_data(ii).base_cov;
                
                mu_hat_tmp = dG_data.RTS_arrs.mu_hat(:,:,:,ii);
                V_hat_tmp = dG_data.RTS_arrs.V_hat(:,:,:,ii);

                for j=1:dG_data.pat_data(ii).m_i % For every observation
                    % Check if patient experienced event in current time frame
                    [delta_ij, tau_ij] = LSDSM_ALLFUNCS.pat_status_at_j(j, dG_data.pat_data(ii), dG_data.model_coef_est.DeltaT);

                    % Work out the derivatives with respect to survival
                    % parameters
                    scalar_tmp = tau_ij * exp(g_s_prev' * base_cov_ii) * ...
                        exp(a_s_prev' * mu_hat_tmp(:,:,j) + 1/2 * a_s_prev' * V_hat_tmp(:,:,j) * a_s_prev);

                    dGdg_s = dGdg_s + delta_ij * base_cov_ii - scalar_tmp * base_cov_ii;
                    dGda_s = dGda_s + delta_ij * mu_hat_tmp(:,:,j) - ...
                        scalar_tmp * (mu_hat_tmp(:,:,j) + V_hat_tmp(:,:,j) * a_s_prev);
                end
            end

            dGout = [dGdg_s; dGda_s];
            
        end

        % Second Derivative of the Expectation with respect to g_s and alpha (a_s)
        function d2Gout = d2Gdx2(g_a_s_prev, dG_data)

            % Store the previous values of g_s and a_s
            g_s_prev = g_a_s_prev(1:size(dG_data.pat_data(1).base_cov,1), 1);
            a_s_prev = g_a_s_prev(size(dG_data.pat_data(1).base_cov,1)+1:end, 1);

            % Initiate the second derivatives to zero
            d2Gdg_s2 = 0;
            d2Gdg_a_s = 0;
            d2Gda_s2 = 0;

            for ii=1:dG_data.pat_data.Count % For every patient
                base_cov_ii = dG_data.pat_data(ii).base_cov; % baseline covariates for patient ii

                mu_hat_tmp = dG_data.RTS_arrs.mu_hat(:,:,:,ii);
                V_hat_tmp = dG_data.RTS_arrs.V_hat(:,:,:,ii);
                
                for j=1:dG_data.pat_data(ii).m_i % For every observation
                    % Check if patient experienced event in current time frame
                    [delta_ij, tau_ij] = LSDSM_ALLFUNCS.pat_status_at_j(j, dG_data.pat_data(ii), dG_data.model_coef_est.DeltaT);

                    scalar_tmp = tau_ij * exp(g_s_prev' * base_cov_ii) * ...
                        exp(a_s_prev' * mu_hat_tmp(:,:,j) + 1/2 * a_s_prev' * V_hat_tmp(:,:,j) * a_s_prev);

                    mu_sigma_alpha = (mu_hat_tmp(:,:,j) + V_hat_tmp(:,:,j) * a_s_prev);

                    d2Gdg_s2 = d2Gdg_s2 - scalar_tmp * (base_cov_ii * base_cov_ii');
                    d2Gdg_a_s = d2Gdg_a_s - scalar_tmp * (base_cov_ii * mu_sigma_alpha');
                    d2Gda_s2 = d2Gda_s2 - scalar_tmp * ((mu_sigma_alpha * mu_sigma_alpha') + V_hat_tmp(:,:,j));

                end
            end

            d2Gout = [d2Gdg_s2, d2Gdg_a_s;
                      d2Gdg_a_s', d2Gda_s2];
        end

        % First derivative of the exponent of the product of the Standard RTS filter 
        % output with the survival likelihood with respect to the hidden state x
        function dgout = dgdx(x_prev, coeffs)
            % g(x_{i1}) = -delta_{i1} a_s x_{i1} + DeltaT exp(a_s x_{i1}) + 1/(2 Sigma_{i1}) (x_{i1} - mu_{i1})^2
            % g'(x_{i1}) = -delta_{i1} a_s + a_s DeltaT exp(a_s x_{i1}) + 1/(Sigma_{i1}) (x_{i1} - mu_{i1})
            dgout = -coeffs.delta_ij * coeffs.a_s ...
                    + coeffs.tau_ij * exp(coeffs.g_s' * coeffs.base_cov) * exp(coeffs.a_s' * x_prev) * coeffs.a_s ...
                    + coeffs.V_ij^(-1) * (x_prev - coeffs.mu_ij); %dg/dx
        end

        % Second derivative of the exponent of the product of the Standard RTS filter 
        % output with the survival likelihood with respect to the hidden state x
        function d2gout = d2gdx2(x_prev, coeffs)
            % g(x_{i1}) = -delta_{i1} a_s x_{i1} + DeltaT exp(a_s x_{i1}) + 1/(2 Sigma_{i1}) (x_{i1} - mu_{i1})^2
            % g''(x_{i1}) = a_s^2 DeltaT exp(a_s x_{i1}) + 1/(Sigma_{i1})
            d2gout = coeffs.tau_ij * exp(coeffs.g_s' * coeffs.base_cov) * exp(coeffs.a_s' * x_prev) * (coeffs.a_s * coeffs.a_s') ...
                     + coeffs.V_ij^(-1); % d2g/dx^2
        end

        % Newton-Raphson procedure for any equation (provide the first and second derivative functions as parameters)
        function x_NR_tmp = Newton_Raphson(num_dims, max_iter, eps, init_val, dfdx, d2fdx2, coeffs)
            x_NR_tmp_arr = zeros(num_dims, max_iter); % array for storing single Newton-Raphson procedure
            x_NR_tmp_arr(:,1) = init_val; % set initial value
            for jj=2:max_iter
                df = dfdx(x_NR_tmp_arr(:,jj-1), coeffs); %dg/dx
                d2f = d2fdx2(x_NR_tmp_arr(:,jj-1), coeffs); % d2g/dx^2
                
                if abs(det(d2f)) < 1e-10 % preventing matrix errors during inverse operation
                    % x_NR_tmp = x_NR_tmp_arr(:,jj);
                    x_NR_tmp_arr(:,jj) = x_NR_tmp_arr(:,jj-1);
                    break
                end

                try
                    if strcmp(functions(dfdx).function, 'LSDSM_ALLFUNCS.dGdx') % if we are updating the survival parameters
                        if not(isnan(coeffs.controls.fixed_params.g_s)) && not(isnan(coeffs.controls.fixed_params.a_s)) % if both parameters are given
                            x_NR_tmp_arr(:,jj) = x_NR_tmp_arr(:,jj-1);
                        elseif not(isnan(coeffs.controls.fixed_params.g_s)) % if g_s is given
                            idx_tmp = size(coeffs.controls.fixed_params.g_s,1)+1; % find index of where a_s starts
                            % reduce derivatives to update a_s only
                            df = df(idx_tmp:end,1); 
                            d2f = d2f(idx_tmp:end,idx_tmp:end);
                            x_NR_tmp_arr(:,jj) = x_NR_tmp_arr(:,jj-1); % keep same g_s
                            x_NR_tmp_arr(idx_tmp:end,jj) = x_NR_tmp_arr(idx_tmp:end,jj-1) - d2f^(-1) * df; % update a_s
                        elseif not(isnan(coeffs.controls.fixed_params.a_s)) % if a_s is given
                            idx_tmp = size(coeffs.controls.fixed_params.g_s,1); % find index of where g_s finishes
                            % reduce derivatives to update g_s only
                            df = df(1:idx_tmp);
                            d2f = d2f(1:idx_tmp,1:idx_tmp);
                            x_NR_tmp_arr(:,jj) = x_NR_tmp_arr(:,jj-1); % keep same a_s
                            x_NR_tmp_arr(1:idx_tmp,jj) = x_NR_tmp_arr(1:idx_tmp,jj-1) - d2f^(-1) * df; % update g_s
                        else % if they both need to be estimated
                            x_NR_tmp_arr(:,jj) = x_NR_tmp_arr(:,jj-1) - d2f^(-1) * df;
                        end
                    else % for other NR operations
                        x_NR_tmp_arr(:,jj) = x_NR_tmp_arr(:,jj-1) - d2f^(-1) * df;
                    end
                catch
                    disp('Singular Matrix')
                end


                chg = x_NR_tmp_arr(:,jj) - x_NR_tmp_arr(:,jj-1);
                if sqrt(chg' * chg) < eps % converged
                    % x_NR_tmp = x_NR_tmp_arr(:,jj);
                    break
                end
            end
            x_NR_tmp = x_NR_tmp_arr(:,jj);
        end

        % Single iteration in time of the Standard RTS Filter
        function [mu_tmp, V_tmp, K_tmp, P_tmp] = KF_single_step(mu_ineg1, V_ineg1, y_i, model_coef_est)

            [y_tmp_i, C_tmp_i, Sigma_tmp_i, Omega_O, Omega_M, I_mat_O, ...
                        I_mat_M, nabla_ij] = LSDSM_ALLFUNCS.missing_val_matrices(y_i, model_coef_est);

            % -> mu_n = A mu_(n-1) + K_n (y_n - C A mu_(n-1) )
            % -> V_n = (I - K_n C) P_(n-1)
            % -> P_(n-1) = A V_(n-1) A' + Gamma
            % -> K_n = P_(n-1) C' (C P_(n-1) C' + Sigma)^-1
            P_tmp = model_coef_est.A * V_ineg1 * model_coef_est.A' + model_coef_est.G_mat * model_coef_est.Gamma * model_coef_est.G_mat';
            K_tmp = P_tmp * C_tmp_i' * (C_tmp_i * P_tmp * C_tmp_i' + Sigma_tmp_i)^-1;
            mu_tmp = model_coef_est.A * mu_ineg1 + K_tmp * ( y_tmp_i - C_tmp_i *  model_coef_est.A * mu_ineg1 );
            V_tmp = (eye(size(mu_ineg1,1)) - K_tmp * C_tmp_i) * P_tmp;

        end

        % Single iteration in time of the Standard RTS Smoother
        function [mu_hat_tmp, V_hat_tmp, J_tmp] = KS_single_step(mu_hat_iplus1, V_hat_iplus1, mu_tilde_tmp, V_tilde_tmp, model_coef_est)
            
            % Single Smoother step
            P_tmp = model_coef_est.A * V_tilde_tmp * model_coef_est.A' + model_coef_est.G_mat * model_coef_est.Gamma * model_coef_est.G_mat';

            J_tmp = V_tilde_tmp * model_coef_est.A' * P_tmp^-1;

            mu_hat_tmp = mu_tilde_tmp + J_tmp * (mu_hat_iplus1 - model_coef_est.A * mu_tilde_tmp);

            V_hat_tmp = V_tilde_tmp + J_tmp * (V_hat_iplus1 - P_tmp) * J_tmp';
        end

        % Finding the values of  and other matrices that are used in the presence of missing values
        function [y_tmp_i, C_tmp_i, Sigma_tmp_i, Omega_O, Omega_M, I_mat_O, ...
                    I_mat_M, nabla_ij] = missing_val_matrices(y_tmp_i, model_coef_est)

            eye_Y = eye(size(y_tmp_i, 1));

            [row_O, col_O] = find(~isnan(y_tmp_i));
            [row_M, col_M] = find(isnan(y_tmp_i));

            Omega_O = eye_Y(row_O, :);
            Omega_M = eye_Y(row_M, :);

            I_mat_O = Omega_O' * Omega_O;
            I_mat_M = Omega_M' * Omega_M;

            y_tmp_i(isnan(y_tmp_i)) = 0;

            y_tmp_i = I_mat_O * y_tmp_i;
            C_tmp_i = I_mat_O * model_coef_est.C;
            Sigma_tmp_i = I_mat_O * model_coef_est.Sigma * I_mat_O + I_mat_M * model_coef_est.Sigma * I_mat_M;

            Sigma_tmp_OO = Omega_O * model_coef_est.Sigma * Omega_O';
            if isempty(Sigma_tmp_OO)
                % Sigma_tmp_OO = zeros(size(Sigma_tmp_i));
                neg_part = zeros(size(Sigma_tmp_i));
            else
                neg_part = model_coef_est.Sigma * Omega_O' * Sigma_tmp_OO^-1 * Omega_O;
            end
            nabla_ij = eye(size(y_tmp_i,1)) - neg_part;
        end
        
        % Identify the event indicator at time step j for patient i
        function [delta_ij, Delta_t_tmp] = pat_status_at_j(j_tmp, pat_ii, DeltaT)
            delta_ij = (pat_ii.m_i == j_tmp) & (pat_ii.delta_ev == 1);
            Delta_t_tmp = DeltaT;
            if pat_ii.surv_time < j_tmp
                Delta_t_tmp = pat_ii.surv_time - (j_tmp - 1) * DeltaT;
            end
        end
        
        function f11out = f11x_i1(x_val, coeffs)
            % g'(x_{i1}) = - delta_{i1} a_s_g' x_{i1} + DeltaT exp(g_s_g' q_tmp + a_s_g' x_{i1})
            %              + 1/2 (y_{i1} - C_g x_{i1})' V_{g}^-1 (y_{i1} - C_g x_{i1}) 
            %              + 1/2 (x_{i1} - bar(x)_{i1g})' bar(W)_{i1g}^-1 (x_{i1} - bar(x)_{i1g})
            f11out = - coeffs.delta_ij * coeffs.a_s' * x_val ...
                     + coeffs.tau_ij * exp(coeffs.g_s' * coeffs.base_cov) * exp(coeffs.a_s' * x_val) ...
                     + (1/2) * (coeffs.y_ij - coeffs.C * x_val)' * coeffs.Sigma^(-1) * (coeffs.y_ij - coeffs.C * x_val) ...
                     + (1/2) * (x_val - coeffs.pred_mu)' * coeffs.pred_V^(-1) * (x_val - coeffs.pred_mu); %f11(x_{i1})
        end
        
        function df11out = df11dx(x_prev, coeffs)
            % g'(x_{i1}) = -delta_{i1} a_s_g + DeltaT exp(g_s_g' q_tmp + a_s_g' x_{i1}) a_s_g
            %              + C_{g}' V_{g}^-1 (y_{i1} - C_g x_{i1}) + bar(W)_{i1g}^-1 (x_{i1} - bar(x)_{i1g})
            df11out = - coeffs.delta_ij * coeffs.a_s ...
                      + coeffs.tau_ij * exp(coeffs.g_s' * coeffs.base_cov) * exp(coeffs.a_s' * x_prev) * coeffs.a_s ...
                      - coeffs.C' * coeffs.Sigma^(-1) * (coeffs.y_ij - coeffs.C * x_prev) ...
                      + coeffs.pred_V^(-1) * (x_prev - coeffs.pred_mu); %df11/dx
        end
        
        function d2f11out = d2f11dx2(x_prev, coeffs)
            % g'(x_{i1}) = -delta_{i1} a_s_g + DeltaT exp(a_s_g' x_{i1}) a_s_g
            %              + C_{g}' Sigma_{i1}^-1 (y_{i1} - C_g x_{i1}) + bar(W)_{i1g}^-1 (x_{i1} - bar(x)_{i1g})
            d2f11out = coeffs.tau_ij * exp(coeffs.g_s' * coeffs.base_cov) * exp(coeffs.a_s' * x_prev) * (coeffs.a_s * coeffs.a_s') ...
                        + coeffs.C' * coeffs.Sigma^(-1) * coeffs.C + coeffs.pred_V^(-1);
        end
        
        
        % The evaluating the likelihood for step j
        function [like_val] = like_fn_curr_step(x_ij, coeffs)
            
            % f_1 (x_i1) = - delta_{ij} a_s_{g}' x_{i1} 
            %              + Delta t exp{g_s_{g}' q_tmp} exp{a_s_{g}' x_{i1} 
            %              + 1/2 (y_{i1} - C_{g} x_{i1})' V_{g}^(-1} (y_{i1} - C_{g} x_{i1})
            %              + 1/2 (x_{i1} - bar(x)_{i1g})' bar(W)_{i1g}_{g}^(-1} (x_{i1} - bar(x)_{i1g})
            dim_size.states = size(x_ij,1);
            dim_size.y = size(coeffs.Sigma, 1);
            % Evaluate the integral value first
            % 1) find the value of x_{i1} that gives the minimum of f_1 (through NR)
            
            % Newton_Raphson(num_dims, max_iter, eps, init_val, dfdx, d2fdx2, coeffs)
            x_NR = LSDSM_ALLFUNCS.Newton_Raphson(dim_size.states, 100, 1e-6, x_ij, @LSDSM_ALLFUNCS.df11dx, ...
                                                        @LSDSM_ALLFUNCS.d2f11dx2, coeffs);
                                                    
            % 2) Evaluate the integral using Laplace approximation
            hess = LSDSM_ALLFUNCS.d2f11dx2(x_NR, coeffs);
            f1x_i1 = LSDSM_ALLFUNCS.f11x_i1(x_NR, coeffs);
            int_val = (2*pi)^(dim_size.states/2) * det(hess)^(-1/2) * exp(-f1x_i1);
            
            % 3) Evaluate the final expression for the likelihood of the
            % current observations given the past observations
            like_val = exp(coeffs.delta_ij * coeffs.g_s' * coeffs.base_cov) * (2*pi)^(-(dim_size.states + dim_size.y)/2) ...
                        * det(coeffs.Sigma)^(-1/2) * det(coeffs.pred_V)^(-1/2) * int_val;
            
        end

        % The entire Standard/Modified RTS filter procedure for a single patient
        function [mu_out, V_out, log_likelihood_val] = ...
                Kalman_filter(pat_ii, model_coef_est, max_censor_time, mod_KF)
            
            % Initialise required arrays for filter equations
            mu = zeros(size(pat_ii.mu_0,1), 1, max_censor_time); % mean in the forward recursion
            V = zeros(size(pat_ii.mu_0,1), size(pat_ii.mu_0,1), max_censor_time); % variance in the forward recursion
            P = zeros(size(pat_ii.mu_0,1), size(pat_ii.mu_0,1), max_censor_time); % variance of integral part of forward recursion
            K = zeros(size(pat_ii.mu_0,1), size(pat_ii.y,1), max_censor_time); % gain in the forward recursion
            mu_out = zeros(size(pat_ii.mu_0,1), 1, max_censor_time); % mean output by standard/modified RTS filter
            V_out = zeros(size(pat_ii.mu_0,1), size(pat_ii.mu_0,1), max_censor_time); % variance output by standard/modified RTS filter

            [y_tmp_i, C_tmp_i, Sigma_tmp_i, Omega_O, Omega_M, I_mat_O, ...
                        I_mat_M, nabla_ij] = LSDSM_ALLFUNCS.missing_val_matrices(pat_ii.y(:,:,1), model_coef_est);

            % initial value estimation
            K(:,:,1) = pat_ii.V_0 * C_tmp_i' * (C_tmp_i * pat_ii.V_0 * C_tmp_i' + Sigma_tmp_i)^-1;
            mu(:,:,1) = pat_ii.mu_0 + K(:,:,1) * (y_tmp_i - C_tmp_i * pat_ii.mu_0);
            V(:,:,1) = (eye(size(pat_ii.mu_0,1)) - K(:,:,1) * C_tmp_i) * pat_ii.V_0;
            
            % % Check survival information for current time step only
            % delta_ij -> boolean to check if patient died within the first time step
            % tau_ij -> survival time for the considered period (time step)
            [delta_ij, tau_ij] = LSDSM_ALLFUNCS.pat_status_at_j(1, pat_ii, model_coef_est.DeltaT);

            if mod_KF % If we are using survival data to modify the states

                V_tmp_NR = V(:,:,1);
                
                % Find the value of mu and V that approximates the 
                % posterior distribution (correction using longitudinal and 
                % survival data)

                g_fn_coef = struct('delta_ij', delta_ij, 'mu_ij', mu(:,:,1), ...
                                   'V_ij', V_tmp_NR, 'base_cov', pat_ii.base_cov, ...
                                   'g_s', model_coef_est.g_s, 'a_s', model_coef_est.a_s, ...
                                   'tau_ij', tau_ij);
                % g(x_{i1}) = -delta_{i1} a_s x_{i1} + DeltaT exp(a_s x_{i1}) + 1/(2 Sigma_{i1}) (x_{i1} - mu_{i1})^2

                x_NR = LSDSM_ALLFUNCS.Newton_Raphson(size(mu,1), 100, 1e-6, mu(:,:,1), @LSDSM_ALLFUNCS.dgdx, ...
                                                        @LSDSM_ALLFUNCS.d2gdx2, g_fn_coef);

                % Update mu_tilde and V_tilde
                mu_out(:,:,1) = x_NR;

                V_tmp_NR_out = (tau_ij * exp(model_coef_est.g_s' * pat_ii.base_cov) * exp(model_coef_est.a_s' * x_NR) ...
                                * (model_coef_est.a_s * model_coef_est.a_s') ...
                                + V_tmp_NR^-1)^-1;
                V_out(:,:,1) = LSDSM_ALLFUNCS.ensure_sym_mat(V_tmp_NR_out);
                
            else % If we are using the standard filter with no survival data correction
                mu_out(:,:,1) = mu(:,:,1);
                V_out(:,:,1) = LSDSM_ALLFUNCS.ensure_sym_mat(V(:,:,1));
            end
            
            % evaluate the likelihood for the first time step
            f1_fn_coef = struct('delta_ij', delta_ij, 'y_ij', y_tmp_i, ...
                                 'C', C_tmp_i, 'Sigma', model_coef_est.Sigma, ...
                                 'pred_mu', pat_ii.mu_0, 'pred_V', pat_ii.V_0, ...
                                 'base_cov', pat_ii.base_cov, 'g_s', model_coef_est.g_s, ...
                                 'a_s', model_coef_est.a_s, 'tau_ij', tau_ij);

            % We store the log due to the small number rounding errors
            log_likelihood_val = log(LSDSM_ALLFUNCS.like_fn_curr_step(mu_out(:,:,1), f1_fn_coef));


            % filtering the rest of the values/time steps
            for j=2:pat_ii.m_i

                [mu(:,:,j), V(:,:,j), K(:,:,j), P(:,:,j-1)] = ...
                    LSDSM_ALLFUNCS.KF_single_step(mu_out(:,:,j-1), V_out(:,:,j-1), pat_ii.y(:,:,j), model_coef_est);

                % boolean to check if patient died within the first time step
                [delta_ij, tau_ij] = LSDSM_ALLFUNCS.pat_status_at_j(j, pat_ii, model_coef_est.DeltaT);
                
                if mod_KF % If we are using survival data to modify the states

                    V_tmp_NR = V(:,:,j);

                    % Find the value of x that maximises the posterior
                    % distribution
                    g_fn_coef = struct('delta_ij', delta_ij, 'mu_ij', mu(:,:,j), ...
                                   'V_ij', V_tmp_NR, 'base_cov', pat_ii.base_cov, ...
                                   'g_s', model_coef_est.g_s, 'a_s', model_coef_est.a_s, ...
                                   'tau_ij', tau_ij);
                    
                    x_NR = LSDSM_ALLFUNCS.Newton_Raphson(size(mu,1), 100, 1e-6, mu(:,:,j), @LSDSM_ALLFUNCS.dgdx, ...
                                                            @LSDSM_ALLFUNCS.d2gdx2, g_fn_coef);

                    % Update mu_tilde and V_tilde
                    mu_out(:,:,j) = x_NR;
                    
                    V_tmp_NR_out = (tau_ij * exp(model_coef_est.g_s' * pat_ii.base_cov) * exp(model_coef_est.a_s' * x_NR) * (model_coef_est.a_s * model_coef_est.a_s') ...
                                   + V_tmp_NR^-1)^-1;
                    V_out(:,:,j) = V_tmp_NR_out;
                    V_out(:,:,j) = LSDSM_ALLFUNCS.ensure_sym_mat(V_out(:,:,j));
                else
                    mu_out(:,:,j) = mu(:,:,j);
                    V_out(:,:,j) = V(:,:,j);
                    V_out(:,:,j) = LSDSM_ALLFUNCS.ensure_sym_mat(V_out(:,:,j));
                end
                
                % Evaluate the likelihood till the current time step
                [y_tmp_i, C_tmp_i, Sigma_tmp_i, Omega_O, Omega_M, I_mat_O, ...
                        I_mat_M, nabla_ij] = LSDSM_ALLFUNCS.missing_val_matrices(pat_ii.y(:,j), model_coef_est);
                
                % evaluate the likelihood for the current time step
                f1_fn_coef = struct('delta_ij', delta_ij, 'y_ij', y_tmp_i, ...
                                     'C', C_tmp_i, 'Sigma', model_coef_est.Sigma, ...
                                     'pred_mu', model_coef_est.A * mu_out(:,:,j-1), ...
                                     'pred_V', P(:,:,j-1), ...
                                     'base_cov', pat_ii.base_cov, 'g_s', model_coef_est.g_s, ...
                                     'a_s', model_coef_est.a_s, 'tau_ij', tau_ij);
                
                log_likelihood_val = log_likelihood_val + log(LSDSM_ALLFUNCS.like_fn_curr_step(mu_out(:,:,j), f1_fn_coef));
            end
        end

        % The entire RTS Smoother procedure for a single patient
        function [mu_out, V_out, J_out] = Kalman_smoother(pat_ii, model_coef_est, max_censor_time)

            % Initialisation for arrays required for smoother operations
            mu_out = zeros(size(pat_ii.mu_tilde,1), 1, max_censor_time);
            V_out = zeros(size(pat_ii.mu_tilde,1), size(pat_ii.mu_tilde,1), max_censor_time);
            J_out = zeros(size(pat_ii.mu_tilde,1), size(pat_ii.mu_tilde,1), max_censor_time);

            % Last value w.r.t. time is obtained from the forward recursion,
            % since that is the P( x(N) | y(1:N) ), i.e. the posterior of x(N)
            % given all data available. Same goes for V(N).
            % mu_hat(length(t_all)) = mu(length(t_all));
            mu_out(:,:,pat_ii.m_i) = pat_ii.mu_tilde(:,:,pat_ii.m_i);
            % V_hat(length(t_all)) = V(length(t_all));
            V_out(:,:,pat_ii.m_i) = pat_ii.V_tilde(:,:,pat_ii.m_i);

            % Iterate through every time step to find the probabilities given
            % the entire observed data.
            for i=2:pat_ii.m_i
                k = pat_ii.m_i - i + 1; % N-1, N-2, ..., 2, 1

                % [mu_hat_tmp, V_hat_tmp] = KS_single_step(mu_hat_iplus1, V_hat_iplus1, mu_tilde_tmp, V_tilde_tmp)
                [mu_out(:,:,k), V_out(:,:,k), J_out(:,:,k)] = ...
                    LSDSM_ALLFUNCS.KS_single_step(mu_out(:,:,k+1), V_out(:,:,k+1), pat_ii.mu_tilde(:,:,k), ...
                                                    pat_ii.V_tilde(:,:,k), model_coef_est);

                % J(k) = V_tilde(k) * model_coef_est.A' * (model_coef_est.A * V_tilde(k) * model_coef_est.A' + model_coef_est.Gamma)^-1;
                % mu_hat(k) = mu_tilde(k) + J(k) * (mu_hat(k+1) - model_coef_est.A * mu_tilde(k));
                % V_hat(k) = V_tilde(k) + J(k) * (V_hat(k+1) - model_coef_est.A * V_tilde(k) * model_coef_est.A' - model_coef_est.Gamma) * J(k)';
                V_out(:,:,k) = LSDSM_ALLFUNCS.ensure_sym_mat(V_out(:,:,k));
                % Add a small number to the diagonals of V to avoid
                % singular matrices since we require the inverse of V.
                % This seems to happen when we have a single observation
                % for a patient.
                V_out(:,:,k) = V_out(:,:,k) + 1e-9 * eye(size(V_out,1));
            end
        end

        % Compute the required expectations for every time step respectively
        function [E] = compute_E_fns(pat_ii, model_coef_est, max_censor_time)
            
            % E[x_n] = mu_hat_n
            % E[x_n x_(n-1)'] = V_hat_n J_(n-1)' + mu_hat_n mu_hat_(n-1)'
            % E[x_n x_n'] = V_hat_n + mu_hat_n mu_hat_n'

            [idx_present_r, idx_present_c] = find(model_coef_est.G_mat ~= 0); % indices of "dynamic" states

            % Placeholders for the expected values for the current patient
            E.xn = zeros(size(pat_ii.mu_hat,1), 1, max_censor_time);
            E.xn_xnneg1 = zeros(size(pat_ii.mu_hat,1), size(pat_ii.mu_hat,1), max_censor_time);
            E.xn_xn = zeros(size(pat_ii.mu_hat,1), size(pat_ii.mu_hat,1), max_censor_time);
            E.barxn_barxn = zeros(length(idx_present_r), length(idx_present_r), max_censor_time);
            E.barxn_xnneg1 = zeros(length(idx_present_r), size(pat_ii.mu_hat,1), max_censor_time);

            E.yn = zeros(size(pat_ii.y,1), 1, max_censor_time);
            E.yn_yn = zeros(size(pat_ii.y,1), size(pat_ii.y,1), max_censor_time);
            E.yn_xn = zeros(size(pat_ii.y,1), size(pat_ii.mu_hat,1), max_censor_time);

            E.xn = pat_ii.mu_hat; % This is equal to the mean of the normal distribution

            % % x(1) = a11 * x(0) + a12 * x(-1)
            % % x(-1) = (x(1) - a11 * x(0)) / a12
            % tmp_val_x0 = (E_xn_tmp(1,1,2) - A_tmp(1,1) * E_xn_tmp(1,1,1)) / A_tmp(1,2);
            % if ~isinf(tmp_val_x0)
            %     E_xn_tmp(2,1,1) = tmp_val_x0;
            % end

            mu_bar_tmp = pat_ii.mu_hat(idx_present_r,:,:);
            M_tmp = pagemtimes(pat_ii.V_hat(:,:,2:end), 'none', pat_ii.J_hat(:,:,1:end-1), 'transpose');
            E.xn_xnneg1(:,:,2:end) = M_tmp + pagemtimes(pat_ii.mu_hat(:,:,2:end), 'none', ...
                                                                pat_ii.mu_hat(:,:,1:end-1), 'transpose');
            
            E.barxn_xnneg1(:,:,2:end) = M_tmp(idx_present_r,:,:) + ...
                                            pagemtimes(mu_bar_tmp(:,:,2:end), 'none', pat_ii.mu_hat(:,:,1:end-1), 'transpose');
            
            E.xn_xn = pat_ii.V_hat + pagemtimes(pat_ii.mu_hat, 'none', pat_ii.mu_hat, 'transpose');
            E.barxn_barxn = pat_ii.V_hat(idx_present_r, idx_present_r, :) + ...
                    pagemtimes(mu_bar_tmp, 'none', mu_bar_tmp, 'transpose');
            
            % for every time step
            for i=1:pat_ii.m_i
                E.xn_xn(:,:,i) = LSDSM_ALLFUNCS.ensure_sym_mat(E.xn_xn(:,:,i));
                E.barxn_barxn(:,:,i) = LSDSM_ALLFUNCS.ensure_sym_mat(E.barxn_barxn(:,:,i));

                % Expectations involving y
                [y_tmp_i, C_tmp_i, Sigma_tmp_i, Omega_O, Omega_M, I_mat_O, ...
                        I_mat_M, nabla_ij] = LSDSM_ALLFUNCS.missing_val_matrices(pat_ii.y(:,:,i), model_coef_est);

                E.yn(:,:,i) = y_tmp_i - nabla_ij * (y_tmp_i - model_coef_est.C * E.xn(:,:,i));

                E.yn_yn(:,:,i) = I_mat_M * (nabla_ij * model_coef_est.Sigma + ...
                    nabla_ij * model_coef_est.C * pat_ii.V_hat(:,:,i) * model_coef_est.C' * nabla_ij') * I_mat_M + ...
                    E.yn(:,:,i) * E.yn(:,:,i)';

                E.yn_yn(:,:,i) = LSDSM_ALLFUNCS.ensure_sym_mat(E.yn_yn(:,:,i));

                E.yn_xn(:,:,i) = nabla_ij * model_coef_est.C * pat_ii.V_hat(:,:,i) + E.yn(:,:,i) * E.xn(:,:,i)';
            end
        end

        % Ensure Matrix remains symmetric
        function sym_mat = ensure_sym_mat(mat_tmp)
            sym_mat = (mat_tmp + mat_tmp') / 2;
        end
        
        % Compute the required summations of expectations
        function [E_sums] = sum_E_fns(E_sums, pat_ii)

            % Sum across all patients of sum_{n=2}^{N} E[x(n) x(n-1)']
            E_sums.xn_xnneg1_from2 = E_sums.xn_xnneg1_from2 + sum(pat_ii.E.xn_xnneg1(:,:,2:pat_ii.m_i), 3);
            % Sum across all patients of sum_{n=1}^{N-1} E[x(n) x(n)']
            E_sums.xn_xn_tillNneg1 = E_sums.xn_xn_tillNneg1 + sum(pat_ii.E.xn_xn(:,:,1:pat_ii.m_i-1), 3);
            % Sum across all patients of sum_{n=2}^{N} E[x(n) x(n)']
            E_sums.xn_xn_from2 = E_sums.xn_xn_from2 + sum(pat_ii.E.xn_xn(:,:,2:pat_ii.m_i), 3);
            % Sum across all patients of sum_{n=1}^{N} E[x(n) x(n)']
            E_sums.xn_xn = E_sums.xn_xn + sum(pat_ii.E.xn_xn(:,:,1:pat_ii.m_i), 3);
            % Sum across all patients of sum_{n=1}^{N} E[x(n)]
            E_sums.xn = E_sums.xn + sum(pat_ii.E.xn(:,:,1:pat_ii.m_i), 3);
            % Sum across all patients of sum_{n=2}^{N} E[bar_x(n) bar_x(n)']
            E_sums.barxn_barxn_from2 = E_sums.barxn_barxn_from2 + sum(pat_ii.E.barxn_barxn(:,:,2:pat_ii.m_i), 3);
            % Sum across all patients of sum_{n=2}^{N} E[bar_x(n) x(n-1)']
            E_sums.barxn_xnneg1_from2 = E_sums.barxn_xnneg1_from2 + sum(pat_ii.E.barxn_xnneg1(:,:,2:pat_ii.m_i), 3);
            % Sum across all patients of initial states E[x0] ( E[x(1)] since MATLAB starts arrays from 1 )
            E_sums.x0 = E_sums.x0 + pat_ii.E.xn(:,:,1);
            % Sum across all patients of initial states E[x0 x0'] ( E[x(1) x(1)'] since MATLAB starts arrays from 1 )
            E_sums.x0_x0 = E_sums.x0_x0 + pat_ii.E.xn_xn(:,:,1);

            % Sum across all patients of sum_{n=3}^{N} E[x(n) x(n)']
            E_sums.barxn_xnneg1_from3 = E_sums.barxn_xnneg1_from3 + sum(pat_ii.E.barxn_xnneg1(:,:,3:pat_ii.m_i), 3);
            % Sum across all patients of sum_{n=2}^{N} E[x(n) x(n)']
            E_sums.xn_xn_from2_tillNneg1 = E_sums.xn_xn_from2_tillNneg1 + sum(pat_ii.E.xn_xn(:,:,2:pat_ii.m_i-1), 3);

            % Sum across all patients of sum_{n=1}^{N} E[y(n) y(n)']
            E_sums.yn_yn = E_sums.yn_yn + sum(pat_ii.E.yn_yn(:,:,1:pat_ii.m_i), 3);
            % Sum across all patients of sum_{n=1}^{N} E[y(n) x(n)']
            E_sums.yn_xn = E_sums.yn_xn + sum(pat_ii.E.yn_xn(:,:,1:pat_ii.m_i), 3);
        end

        % M Step - Individual characteristics (initial conditions)
        function [mu_0new_tmp, V_0new_tmp] = M_step_indiv_pat(sum_x0_tmp, sum_x0_x0_tmp, num_pats)
            % mu_0new = E[x_1]
            % V_0new = E[x1 x1'] - E[x1] E[x1']
            mu_0new_tmp = sum_x0_tmp / num_pats; % E[x(1)]
            V_0new_tmp = sum_x0_x0_tmp / num_pats - mu_0new_tmp * mu_0new_tmp'; % 1/N * \sum( E[x(1) x(1)'] ) - E[x(1)] E[x(1)']
        end

        % M Step - Population parameters
        function [model_new_coeffs] = M_step_all_pats(pat_data, E_sums, RTS_arrs, model_coef_est, controls)

            % A_new = ( sum_{n=2}^{N} E[x_n x_(n-1)'] ) ( sum_{n=2}^{N} E[x_(n-1) x_(n-1)'] )^-1
            % Gamma_new = 1 / (N-1) * sum_{n=2}^N { E[x_n x_n'] - A_new E[x_(n-1) x_n']
            %                                       - E[x_n x_(n-1)'] A_new' + A_new E[x_(n-1) x_(n-1)'] A_new' }
            % C_new = ( sum_{n=1}^{N} y_n E[x_n'] ) ( sum_{n=1}^{N} E[x_n x_n'])^-1
            % Sigma_new = 1/N * sum_{n=1}^{N} { y_n y_n' - C_new E[x_n] y_n' 
            %                                   - y_n E[x_n'] C_new' + C_new E[x_n x_n'] C_new' }
            
            % Number of patients
            num_pats = double(pat_data.Count);
            N_totpat = 0; % Total number of observations across all patients
            for ii=1:num_pats
                N_totpat = N_totpat + pat_data(ii).m_i;
            end

            [idx_present_r, idx_present_c] = find(model_coef_est.G_mat ~= 0);

            % Update the values after all summations across all patients have been
            % evaluated.

            % A = ( sum_{n=2}^{N} E[ x(n) x(n-1)' ] ) ( sum_{n=1}^{N-1} E[ x(n) x(n)' ] )^-1
            % A_new = sum_xn_xnneg1_from2_tmp * sum_xn_xn_tillNneg1_tmp^-1;
            % Special treatment due to canonical form
            % A_bar_new = sum_barxn_xnneg1_from2_tmp * sum_xn_xn_tillNneg1_tmp^-1;
            A_bar_new = E_sums.barxn_xnneg1_from3 * E_sums.xn_xn_from2_tillNneg1^-1;
            A_new = model_coef_est.A;
            A_new(idx_present_r,:) = A_bar_new;

            if not(isnan(controls.fixed_params.A)) % if A is fixed
                A_new = controls.fixed_params.A;
                A_bar_new = A_new(idx_present_r,:);
            end

            % C = ( sum_{n=1}^{N} y(n) E[ x(n)' ] ) ( sum_{n=1}^N E[ x(n) x(n)' ] )^-1
            C_new = E_sums.yn_xn * E_sums.xn_xn^-1;
            
            if not(isnan(controls.fixed_params.C)) % if C is fixed
                C_new = controls.fixed_params.C;
            end
            % C_new = C_tmp; % Checked for identifiability purposes

            % Gamma = (1 / sum(m_i - 1) ) ( sum_{n=2}^{N} E[ x(n) x(n)' ] 
            %                              - A sum_{n=2}^{N} E[ x(n) x(n-1)' ]' 
            %                              - sum_{n=2}^{N} E[ x(n) x(n-1)' ] A'
            %                              + A sum_{n=1}^{N-1} E[ x(n) x(n)' ] A' )

            Gamma_new = (N_totpat - num_pats)^-1 * ...
                        (E_sums.barxn_barxn_from2 - A_bar_new * E_sums.barxn_xnneg1_from2' ...
                         - E_sums.barxn_xnneg1_from2 * A_bar_new' + A_bar_new * E_sums.xn_xn_tillNneg1 * A_bar_new');
            Gamma_new = LSDSM_ALLFUNCS.ensure_sym_mat(Gamma_new);
            
            if not(isnan(controls.fixed_params.Gamma)) % if Gamma is fixed
                Gamma_new = controls.fixed_params.Gamma;
            end

            % Sigma = (1 / N ) ( sum_{n=1}^{N} y(n) y(n)'
            %                    - C sum_{n=1}^{N} y(n) E[ x(n)' ]' 
            %                    - sum_{n=1}^{N} y(n) E[ x(n)' ] C'
            %                    + C sum_{n=1}^{N} E[ x(n) x(n)' ] C' )

            Sigma_new = (N_totpat)^-1 * (E_sums.yn_yn - C_new * E_sums.yn_xn' ...
                                            - E_sums.yn_xn * C_new' + C_new * E_sums.xn_xn * C_new');
            Sigma_new = LSDSM_ALLFUNCS.ensure_sym_mat(Sigma_new);
            
            if not(isnan(controls.fixed_params.Sigma)) % if Sigma is fixed
                Sigma_new = controls.fixed_params.Sigma;
            end
            % Sigma_new = Sigma_tmp; % Checked for identifiability purposes

            %%%%%%%%%%%%%%%%%%
            %%% NR - alpha %%%
            %%%%%%%%%%%%%%%%%%
            
            dG_data = struct('pat_data', pat_data, 'model_coef_est', model_coef_est, ...
                             'RTS_arrs', RTS_arrs, 'controls', controls);
            
            % Newton_Raphson(num_dims, max_iter, eps, init_val, dGdx, d2Gdx2, pat_info)
            g_a_s_new = LSDSM_ALLFUNCS.Newton_Raphson(size(model_coef_est.g_s,1) + size(model_coef_est.a_s,1), 100, 1e-6, ...
                                                        [model_coef_est.g_s; model_coef_est.a_s], @LSDSM_ALLFUNCS.dGdx, ...
                                                        @LSDSM_ALLFUNCS.d2Gdx2, dG_data);

            g_s_new = g_a_s_new(1:size(model_coef_est.g_s,1), 1);
            if not(isnan(controls.fixed_params.g_s)) % if g_s is fixed
                g_s_new = controls.fixed_params.g_s;
            end

            a_s_new = g_a_s_new(size(model_coef_est.g_s,1)+1:end, 1);
            if not(isnan(controls.fixed_params.a_s)) % if a_s is fixed
                a_s_new = controls.fixed_params.a_s;
            end

            if controls.update_pop_mu
                mu_0new = E_sums.x0 / num_pats; % E[x(1)]
                % We need to check for fixed mu_0 before updating V_0
                if not(isnan(controls.fixed_params.mu_0)) % if mu0 is fixed
                    mu_0new = controls.fixed_params.mu_0;
                end
                % V = 1/N * \sum( E[x(1) x(1)'] ) - E[x(1)] E[x(1)']
                V_0new = E_sums.x0_x0 / num_pats - mu_0new * mu_0new'; 
                V_0new = LSDSM_ALLFUNCS.ensure_sym_mat(V_0new);
                
            else

                mu_0new = reshape(RTS_arrs.mu_hat(:,:,1,:), [size(model_coef_est.A,1), 1, num_pats]);
                % We need to check for fixed mu_0 before updating V_0
                if not(isnan(controls.fixed_params.mu_0)) % if mu0 is fixed
                    mu_0new = controls.fixed_params.mu_0;
                end
                V_0new = reshape(RTS_arrs.V_hat(:,:,1,:), [size(model_coef_est.A,1), size(model_coef_est.A,1), num_pats]);
            end
            
            if not(isnan(controls.fixed_params.V_0)) % if V0 is fixed
                V_0new = controls.fixed_params.V_0;
            end

            model_new_coeffs = struct('A', A_new, 'C', C_new, ...
                                      'Gamma', Gamma_new, 'Sigma', Sigma_new, ...
                                      'g_s', g_s_new, 'a_s', a_s_new, ...
                                      'DeltaT', model_coef_est.DeltaT, 'G_mat', model_coef_est.G_mat, ...
                                      'mu_0', mu_0new, 'V_0', V_0new);
        end

        
        %%% Performance Metrics %%%
        
        % RMSE
        function [sse_value, mse_value, rmse_value] = rmse_fn(true_signal, comp_signal, time_dim)
            error_signal = true_signal - comp_signal;
            sse_value = zeros(size(true_signal, 1), 1);
            for j=1:size(error_signal, time_dim)
                if time_dim == 2
                    sse_value = sse_value + error_signal(:,j).^2;
                elseif time_dim == 3
                    sse_value = sse_value + error_signal(:,:,j).^2;
                end

            end
            mse_value = sse_value / length(error_signal);
            rmse_value = sqrt(mse_value);
        end
        
        
    end
end