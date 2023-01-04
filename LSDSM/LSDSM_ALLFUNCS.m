classdef LSDSM_ALLFUNCS
    methods(Static)
        
        % All functions used for LSDSM are stored within this class
        
        % Function to convert csv to required data structure
        function [data_observed] = read_from_csv(M_mat, dim_size, csv_controls)
            % FUNCTION NAME:
            %   read_from_csv
            %
            % DESCRIPTION:
            %   Function to convert csv to required data structure. data is
            %   assumed to be in long format, i.e. one row for every
            %   longitudinal measurement.
            %
            %   Assumed structure of csv file:
            %     - col 01 = patID
            %     - col 02 = survTime
            %     - col 03 = eventInd
            %     - col 04 = tij
            %     - col base_cov_col_no = baseline covariate 1
            %     - col ... = baseline covariate ...
            %     - col base_cov_col_no + dim_size.base_cov = longitudinal biomarker 1
            %     - col ... = longitudinal biomarker ...
            %
            % INPUT:
            %   M_mat - (matrix) This is the matrix obtained from MATLAB's
            %                    native read csv function. Data is
            %                    expressed in the long format.
            %   dim_size - (struct) Expresses the dimension sizes of the
            %              number of hidden states and dynamic states, 
            %              number of observations, and number of baseline 
            %              covariates.
            %   csv_controls - (struct) Contains controls such as the
            %                  column start for the baseline covariates and
            %                  longitudinal biomarkers, the time step for
            %                  the state space model, etc.
            %
            % OUTPUT:
            %   data_observed - (map) Contains a struct for every patient,
            %                   each retaining the patient's ID, survival
            %                   time, event boolean, number of measurements
            %                   for the state space model, the baseline
            %                   covariates and the longitudinal biomarkers.
            %
            % REVISION HISTORY:
            %   17/12/2022 - mcauchi1
            %       * Initial implementation
            %
            
            % Normalisation constants if we wish to normalise the biomarkers
            normalise_const = max(M_mat(:,csv_controls.bio_long_col_no:end));

            % Find the maximum time from the training data
            max_t = ceil(max(M_mat(:,2))); % ceil(max(surv time))
            % Create an array for time in csv_controls.Delta steps till max_t is reached
            no_of_months = linspace(0,max_t*csv_controls.t_multiplier, max_t*csv_controls.t_multiplier/csv_controls.Delta+1);

            % Stores observed patient data - struct for every patient,
            % stored within a map
            data_observed = containers.Map('KeyType', 'int32', 'ValueType', 'any'); 
            
            % Find the number of patients in the training set
            unique_pat_id = unique(M_mat(:,1));
            no_of_pat = length(unique_pat_id);

            % Filling in patient information
            for i=1:no_of_pat
                pat_ii = struct();
                pat_ii.id = unique_pat_id(i);
                pat_ii.surv_time = 0;
                pat_ii.delta_ev = 0;
                pat_ii.m_i = 0;
                pat_ii.base_cov = zeros(dim_size.base_cov,1);
                % NaN was used for y as these will be treated as missing 
                % values within the state space model if no observations 
                % were made at that time period.
                pat_ii.y = NaN * zeros(dim_size.y, 1, length(no_of_months));
                % Extract the part of matrix that contains only the current
                % patient's info - curr_ind contains the row indices of the
                % matrix that correspond to the current patient
                curr_ind = find(M_mat(:,1) == pat_ii.id);
                % Find the iteration number for every observation made for
                % this patient Note: For training data, we use round
                % function to go to the closest time binning. In testing
                % data, we shall use ceil for those time points that arrive
                % after the landmark of interest. This is because we do not
                % want to use future data after the landmark. E.g. assume
                % landmark is at 12 time points. If there is an observation
                % at 12.4, then using the round function, this will go to
                % 12 time points, and hence, future data is used. With
                % ceil, this measurement will not be utilised.
                if strcmp(csv_controls.train_test, 'train') % if training data set
                    iter_no = round(M_mat(curr_ind, 4) * csv_controls.t_multiplier/csv_controls.Delta);
                else % if testing data set
                    iter_no = M_mat(curr_ind, 4) * csv_controls.t_multiplier/csv_controls.Delta;
                    iter_no(iter_no<=csv_controls.landmark_idx) = round(iter_no(iter_no<=csv_controls.landmark_idx));
                    iter_no(iter_no>csv_controls.landmark_idx) = ceil(iter_no(iter_no>csv_controls.landmark_idx));
                end
                for j=1:length(curr_ind)
                    % Store the patient's longitudinal biomarkers in a 3d
                    % array (num_obs, 1, num_time_steps).
                    if csv_controls.norm_bool
                        pat_ii.y(:,1,iter_no(j)+1) = M_mat(curr_ind(j), csv_controls.bio_long_col_no:csv_controls.bio_long_col_no+dim_size.y-1) ...
                                                     ./ normalise_const(1:dim_size.y);
                    else
                        pat_ii.y(:,1,iter_no(j)+1) = M_mat(curr_ind(j), csv_controls.bio_long_col_no:csv_controls.bio_long_col_no+dim_size.y-1);
                    end
                end
                % Also note: utilising the above method means that some
                % observations are not utilised, since they happen to fall
                % at the same time bin.

                % If we want to visualise some plots
                if csv_controls.allow_plots && i <= csv_controls.no_of_plots
                    figure;
                    to_plot = M_mat(curr_ind, csv_controls.bio_long_col_no);
                    if csv_controls.norm_bool
                        to_plot = to_plot / normalise_const(1);
                    end
                    scatter(M_mat(curr_ind, 4), to_plot);
                    xlabel('Time (years)')
                    ylabel('y');
                    hold on;

                    if M_mat(curr_ind(j), 3) == 0 % if patient is censored
                        xline(M_mat(curr_ind(1), 2), 'g', 'LineWidth', 2);
                        legend('y', 'Censored');
                    else % if patient experiences event
                        xline(M_mat(curr_ind(1), 2), 'r', 'LineWidth', 2);
                        legend('y', 'Event');
                    end

                    xlim([0,max_t]);
                    if csv_controls.norm_bool
                        ylim([0,1]);
                    else
                        ylim([min(M_mat(:,csv_controls.bio_long_col_no)) normalise_const(1)]);
                    end
                end

                % Store survival information
                pat_ii.delta_ev = M_mat(curr_ind(j), 3);
                pat_ii.surv_time = M_mat(curr_ind(1), 2)*csv_controls.t_multiplier;

                % If the survival time is greater than the hard
                % thresholding of the censor time, then the patient is
                % censored at the threshold time
                if pat_ii.surv_time > csv_controls.censor_time
                    pat_ii.delta_ev = 0;
                    pat_ii.surv_time = csv_controls.censor_time;
                end

                % Number of time periods the patient is observed for
                pat_ii.m_i = floor(pat_ii.surv_time/csv_controls.Delta)+1;

                % Store the baseline covariates
                pat_ii.base_cov(1,1) = 1; % intercept
                % Other baseline covariates
                pat_ii.base_cov(2:end,1) = ...
                    M_mat(curr_ind(j), csv_controls.base_cov_col_no:csv_controls.base_cov_col_no+dim_size.base_cov-2); 

                % Store patient data in map
                data_observed(i) = pat_ii;
            end
        end
        
        
        % Data simulation
        function [data_latent, data_observed] = ...
            sim_obs_surv_pat(num_pats, cens_time, model_true, frac_miss, cens_range, rand_bool)
            % FUNCTION NAME:
            %   sim_obs_surv_pat
            %
            % DESCRIPTION:
            %   Function to simulate patient data. This function splits the
            %   data into latent and observed data. 
            %
            %   Inverse Transform Sampling is used to sample the survival
            %   time for every patient. This utilises uniform
            %   distributions.
            %   For the details regarding this sampling of the survival
            %   time, refer to "Example for a Piecewise Constant Hazard
            %   Data Simulation in R" - Technical Report by Rainer Walke
            %   2010
            %
            % INPUT:
            %   num_pats - (double) The number of patients to simulate data
            %              for.
            %   cens_time - (double) The maximum period to observe the
            %               patients.
            %   model_true - (struct) Contains the true parameter values
            %                of the model that are used for the simulation.
            %   frac_miss - (double) The fraction of observations to be
            %               "missing" within the state space model.
            %   cens_range - (vector) Array containing the range at which
            %                to make uniform censoring.
            %                Format: [censor_start censor_end]
            %   rand_bool - (boolean) Used to randomise every simulation,
            %               regardless of the seed value set in the main 
            %               file.
            %
            % OUTPUT:
            %   data_latent - (map) Contains the clean (no process noise)
            %                 and true hidden states, and the hazard
            %                 function.
            %   data_observed - (map) Contains the number of SSM
            %                   observations, survival time, event boolean,
            %                   baseline covariates, and the longitudinal
            %                   biomarker vectors.
            %
            % REVISION HISTORY:
            %   17/12/2022 - mcauchi1
            %       * Initial implementation
            %
        

            % These maps will contain a struct for every patient
            % Each struct (for observed data) consists of:
            % -- m_i - Number of observed timepoints for the patient
            % -- delta_ev - Event indicator for the patient
            % -- surv_time - Survival time for the patient
            % -- y - longitudinal biomarkers for the patient
            % -- base_cov - baseline covariates for the patient
            data_observed = containers.Map('KeyType', 'int32', 'ValueType', 'any'); % Stores observed patient data
            data_latent = containers.Map('KeyType', 'int32', 'ValueType', 'any'); % Stores accurate and unobserved patient data
            
            if rand_bool % if randomisation is set
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
                m_i = 0; % Number of (SSM) iterations observed for patient
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
                unif_dist = rand(1); % the smaller this value, the higher chance the patient has to survive
                cum_haz = 0; % initialise cumulative hazard
                x_cln(:,:,1) = x0_tmp; % initialise x with no process noise

                % Finding the sqrt of the variance (to obtain std dev)
                sqrtV0 = chol(model_true.W_0, 'lower');
                x_true(:,:,1) = x0_tmp + sqrtV0 * randn(dim_size.x,1); % initialise x with disturbance

                % enforce a lower limit of 0 on x - assuming negative
                % biomarker values do not exist
                x_true(:,:,1) = max(0, x_true(:,:,1));
                
                % Finding the sqrt of the variance (to obtain std dev)
                sqrtW = chol(model_true.W, 'lower');
                sqrtV = chol(model_true.V, 'lower');
                
                % initialise y with measurement noise
                y(:,:,1) = model_true.C * x_true(:,:,1) + sqrtV * randn(dim_size.y,1); 

                % enforce a lower limit of 0 on y - assuming negative
                % biomarker values do not exist
                y(:,:,1) = max(0, y(:,:,1));

                %%% Let's assume first observation is always observed.
                %%% Uncomment below if this is not a valid assumption.
                % y_miss = rand(dim_size.y, 1) < frac_miss;
                % y_o(y_miss == 1,:,1,ii) = NaN;

                % calculate initial hazard
                haz_true(1,1) = exp( model_true.g_s' * q_ii + model_true.a_s' * x_true(:,:,1));
                % add hazard to cumulative hazard
                cum_haz = cum_haz + model_true.DeltaT * haz_true(1,1);

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
                        x_true(:,:,j) = model_true.A * x_true(:,:,j-1) + model_true.G_mat * sqrtW * randn(size(sqrtW,2),1); 

                        % enforce a lower limit of 0 on x - assuming negative
                        % biomarker values do not exist
                        x_true(:,:,j) = max(0, x_true(:,:,j));

                        % calculate y with measurement noise
                        y(:,:,j) = model_true.C * x_true(:,:,j) + sqrtV * randn(dim_size.y,1); 
                        
                        % enforce a lower limit of 0 on y - assuming negative
                        % biomarker values do not exist
                        y(:,:,j) = max(0, y(:,:,j));

                        % randomise the missing observations based on the
                        % expected fraction of missing values Note: at some
                        % time points, we may have partial missing
                        % observations (i.e. only some of the biomarkers
                        % are missing)
                        y_miss = rand(dim_size.y, 1) < frac_miss;
                        y(y_miss == 1,:,j) = NaN;

                        % calculate new hazard
                        haz_true(1,j) = exp( model_true.g_s' * q_ii + model_true.a_s' * x_true(:,:,j));
                        % calculate cumulative hazard
                        cum_haz = cum_haz + model_true.DeltaT * haz_true(1,j);

                        if log(unif_dist) > - cum_haz % if cumulative hazard exceeds a certain value
                            % T_i = tau_i - ( ln(S_i(t)) + H_i(t) / h_i(t) )
                            surv_time = (j-1) * model_true.DeltaT - (log(unif_dist) + prev_cum_haz) / haz_true(1,j);
                            delta_ev = 1; % patient experienced event
                            m_i = j; % patient had j observations
                            break; % future observations are not required/possible.
                        end
                        
                        % if patient did not experience event
                        prev_cum_haz = cum_haz; % store the cumulative hazard function
                        
                    end % End of observation period
                    if delta_ev == 0 % if patient remained event-free
                        surv_time = (max_t_points) * model_true.DeltaT; % calculate survival time
                        m_i = j; % patient had a total of j observations
                    end
                end
                
                % Store data in maps
                data_latent(ii) = struct('x_cln', x_cln, 'x_true', x_true, 'haz_true', haz_true);
                data_observed(ii) = struct('m_i', m_i, 'delta_ev', delta_ev, 'surv_time', surv_time, ...
                                           'y', y, 'base_cov', base_cov);
            end
        end
        
        
        function array_tmp = map_struct_to_4dims_mat(map_tmp, field_name)
            % FUNCTION NAME:
            %   map_struct_to_4dims_mat
            %
            % DESCRIPTION:
            %   Combines patient data into a 4d array. Enters within the
            %   patient data that is stored separately in map keys and
            %   extracts the longitudinal data required.
            %
            % INPUT:
            %   map_tmp - (map) Map containing all patient data.
            %   field_name - (string) The name of the field to be
            %                extracted.
            %
            % OUTPUT:
            %   array_tmp - (array) Output in array format.
            %
            % REVISION HISTORY:
            %   17/12/2022 - mcauchi1
            %       * Initial implementation
            %
            
            array_tmp = zeros([size(getfield(map_tmp(1), field_name)), map_tmp.Count]);
            
            for ii=1:map_tmp.Count
                array_tmp(:,:,:,ii) = getfield(map_tmp(ii), field_name);
            end
        end
        

        function plot_pat_info(num_plots, t_arr, data_latent, data_obs)
            % FUNCTION NAME:
            %   plot_pat_info
            %
            % DESCRIPTION:
            %   Creates plots of hidden state trajectories and hazard
            %   function over time for a number of patients. This function
            %   can be used when this latent data is available.
            %
            % INPUT:
            %   num_plots - (double) The number of patients to have their 
            %               data plotted.
            %   t_arr - (array) Array containing the SSM time steps.
            %   data_latent - (map) Contains patients' hidden information,
            %                 including hidden states and hazard function.
            %   data_obs - (map) Contains patients' observed information,
            %              including survival time, event boolean, and 
            %              longitudinal biomarkers.
            %
            % REVISION HISTORY:
            %   17/12/2022 - mcauchi1
            %       * Initial implementation
            %
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
                        % FUNCTION NAME:
            %   initialise_params
            %
            % DESCRIPTION:
            %   Function to initialise the parameters of LSDSM in a generic
            %   manner.
            %
            % INPUT:
            %   dim_size - (struct) Contains the dimension sizes of the
            %              latent states, dynamic states, and the number of
            %              observations and the baseline covariates.
            %   data_observed - (map) Contains the observed data of the
            %                   patients.
            %   Delta - (double) Time step for SSM.
            %
            % OUTPUT:
            %   model_coef_init - (struct) Contains the initiliased
            %                     parameter values of LSDSM.
            %
            % REVISION HISTORY:
            %   17/12/2022 - mcauchi1
            %       * Initial implementation
            %
            
            % State space parameters
            a_bar_tmp = 0.5 * [eye(dim_size.dyn_states), eye(dim_size.dyn_states, dim_size.states - dim_size.dyn_states)];
            A_init = [a_bar_tmp; % Dynamics matrix in canonical form
                 eye(dim_size.states - dim_size.dyn_states, dim_size.states)];
            C_init = [eye(dim_size.y), zeros(dim_size.y, dim_size.states - dim_size.y)]; % Observation matrix
            W_init = (0.5)^2 * eye(size(a_bar_tmp, 1)); % Disturbance matrix
            V_init = (0.5)^2 * eye(size(C_init,1)); % Measurement error matrix
            G_mat = [eye(dim_size.dyn_states); % Matrix linking disturbance with the states
                     zeros(dim_size.states - dim_size.dyn_states, dim_size.dyn_states)];

            % Capture initial observation values for all patients
            y_init_mat = zeros(dim_size.y, data_observed.Count);
            for i=1:data_observed.Count
                y_init_mat(:,i) = data_observed(i).y(:,:,1);
            end

            
            % Initialise initial state values based on observation data
            % lagging states have same value as initial biomarker values observed
            
            % if number of hidden states is a multiple of number of observations
            if mod(dim_size.states, dim_size.y) == 0 
                mu_0_init = repmat(nanmean(y_init_mat, 2), dim_size.states / dim_size.y, 1);
            else % if one observation has more lagging states associated with it
                temp_mu = nanmean(y_init_mat, 2);
                mu_0_init = [repmat(temp_mu, floor(dim_size.states / dim_size.dyn_states), 1);
                             temp_mu(1:mod(dim_size.states, dim_size.dyn_states))];
            end

            % Initial variance is calculated using the variance observed in
            % the first hidden state vector for all patients
            W_0_init = zeros(dim_size.states, dim_size.states);
            for j=1:dim_size.states
                corr_dyn_state = mod(j-1, dim_size.y) + 1;
                W_0_init(j,j) = 1 * nanvar(y_init_mat(corr_dyn_state,:));
                if j>dim_size.y % for the lagging states, put a higher variance than the sample variance
                    W_0_init(j,j) = 3 * nanvar(y_init_mat(corr_dyn_state,:));
                end
            end

            % Survival parameters
            % coefficients linking baseline covariates with hazard function
            g_s_init = zeros(dim_size.base_cov, 1);
            % coefficients linking hidden states with hazard function
            a_s_init = zeros(dim_size.states, 1); 

            % Store all model coefficients within a struct
            model_coef_init = struct('A', A_init, 'C', C_init, ...
                                     'W', W_init, 'V', V_init, ...
                                     'g_s', g_s_init, 'a_s', a_s_init, ...
                                     'DeltaT', Delta, 'G_mat', G_mat, ...
                                     'mu_0', mu_0_init, 'W_0', W_0_init);
        end
        
        
        function plot_forecast_surv(pat_ii, landmark_t, t_arr, Delta)
            % FUNCTION NAME:
            %   plot_forecast_surv
            %
            % DESCRIPTION:
            %   Shows the predicted survival curve from the start of
            %   observation of the patient. It shows the probability of
            %   survival, not the probability of survival given the patient
            %   survived till landmark. For the latter, divide by the
            %   probability of survival at landmark, and ignore previous
            %   values.
            %
            % INPUT:
            %   pat_ii - (struct) Observed data of a single patient,
            %            including survival forecasts, survival time, and 
            %            event indicator.
            %   landmark_t - (double) Time at which forecasts are made.
            %   t_arr - (array) Array containing the SSM time steps.
            %   Delta - (double) Time step for SSM.
            %
            % REVISION HISTORY:
            %   17/12/2022 - mcauchi1
            %       * Initial implementation
            %
            if pat_ii.delta_ev == 0
                title_n = sprintf('Patient was censored at time %.2f', pat_ii.surv_time);
            else
                title_n = sprintf('Patient died at time %.2f', pat_ii.surv_time);
            end

            figure;
            % size of array is one time step longer for survival curves.
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
            % FUNCTION NAME:
            %   plot_bs_pe_auc
            %
            % DESCRIPTION:
            %   Plots the Brier Score, Prediction Error Score, Area Under
            %   ROC Curve, and a stacked histogram showing the number of
            %   patients that experience an event within the horizon of 
            %   interest, and the number of patients that survive beyond 
            %   the horizon of interest.
            %
            % INPUT:
            %   pat_ii - (struct) Observed data of a single patient,
            %            including survival forecasts, survival time, and 
            %            event indicator.
            %   landmark_t - (double) Time at which forecasts are made.
            %   t_arr - (array) Array containing the SSM time steps.
            %   Delta - (double) Time step for SSM.
            %
            % REVISION HISTORY:
            %   17/12/2022 - mcauchi1
            %       * Initial implementation
            %
            
            % Number of patients that are known to survive beyond the
            % landmark.
            reduced_num_pats = double(pat_data_reduced.Count);
            surv_info_mat = zeros(reduced_num_pats, 2); % surv time and delta matrix

            for ii=1:reduced_num_pats % populate the surv time and delta from every patient
                surv_info_mat(ii,:) = [pat_data_reduced(ii).surv_time, pat_data_reduced(ii).delta_ev];
            end

            % Counts the number of patients that experience the event
            % within the horizon and interest, and those that survive
            % beyond the horizon of interest.
            hist_count_arr = zeros(1, length(auc_test));
            hist_cens_arr = zeros(1, length(auc_test));

            for ii=1:length(hist_count_arr)
                hist_count_arr(ii) = sum(surv_info_mat(surv_info_mat(:,1) < (landmark_t + Delta * ii), 2));
                hist_cens_arr(ii) = size(surv_info_mat(surv_info_mat(:,1) > (landmark_t + Delta * ii), :),1);
            end

            % Calculate the maximum forecast time (+Delta/2 for a nicer
            % plot).
            max_time_shown = censor_time - landmark_t + Delta / 2;

            figure;
            t = tiledlayout(4, 1);
            nexttile;
            % Brier Score
            plot(Delta:Delta:Delta*length(bs_test), bs_test, '-*g');
            legend('Proposed Model', 'Location', 'southeast');
            title('Brier Score');
            xlim([0, max_time_shown]);
            grid on;
            nexttile;
            % Prediction Error Score
            plot(Delta:Delta:Delta*length(pe_test), pe_test, '-*g');
            legend('Proposed Model', 'Location', 'southeast');
            title('Prediction Error');
            xlim([0, max_time_shown]);
            grid on;
            nexttile;
            % Area Under ROC Curve
            plot(Delta:Delta:Delta*length(auc_test), auc_test, '-*g');
            legend('Proposed Model', 'Location', 'southeast');
            title('Area under ROC curve');
            xlim([0, max_time_shown]);
            grid on;
            nexttile;
            % Histogram of patients experiencing event and censoring at
            % every forecasted time step
            bar(Delta:Delta:Delta*length(hist_count_arr), [hist_count_arr; hist_cens_arr], "stacked");
            xlim([0, max_time_shown]);
            grid on;
            title('Frequency of Events and Censored Observations');
            lgd = legend('Number of Event Observations', 'Number of Censored Observations');
            lgd.Location = 'southoutside';
            lgd.Orientation = 'horizontal';
            title(t, sprintf('Performance metrics across different horizons at landmark = %.1f years', ...
                landmark_t / t_multiplier));
            xlabel(t, 'Horizon (in months)');
        end
        
        
        function [auc_test_arr] = AUC_fn(pat_data, landmark_t, max_horizon_t, model_coef_est) 
            % FUNCTION NAME:
            %   AUC_fn
            %
            % DESCRIPTION:
            %   Finds the time-dependent Area under ROC curve for several
            %   horizons at a particular landmark. The formulation is based
            %   on the one provided by Blanche et al. in their paper
            %   "Quantifying and Comparing Dynamic Predictive Accuracy of
            %   Joint Models for Longitudinal Marker and Time-to-Event in
            %   Presence of Censoring and Competing Risks".
            %
            % INPUT:
            %   pat_data - (map) Observed data of all patients still at
            %              risk at landmark time, including survival
            %              forecasts, survival time, and event indicator.
            %   landmark_t - (double) Time at which forecasts are made, and
            %                hence, the start of AUC calculations.
            %   max_horizon_t - (double) The maximum horizon to work out
            %                   the AUC calculations.
            %   model_coef_est - (struct) Contains all model parameters for
            %                    LSDSM.
            %
            % OUTPUT:
            %   auc_test_arr - (array) All AUC calculations for the
            %                  horizons and landmark indicated.
            % 
            % REVISION HISTORY:
            %   17/12/2022 - mcauchi1
            %       * Initial implementation
            %
            
            % +1 due to index starts from 1
            landmark_idx_tmp = floor(landmark_t / model_coef_est.DeltaT) + 1;
            
            % total number of patients at risk at landmark time
            no_of_pat_auc_test = double(pat_data.Count);
            
            % surv time and delta matrix
            surv_info_mat = zeros(no_of_pat_auc_test, 2);
            
            for ii=1:no_of_pat_auc_test
                surv_info_mat(ii,:) = [pat_data(ii).surv_time, pat_data(ii).delta_ev];
            end

            % Initialise array to store AUC calculations
            auc_test_arr = zeros(1, int64(max_horizon_t / model_coef_est.DeltaT));

            % For every horizon to test
            for horizon_t_tmp=model_coef_est.DeltaT:model_coef_est.DeltaT:max_horizon_t

                % time at which we want to use survival prediction
                t_est = landmark_t + horizon_t_tmp;

                % t_est_idx refers to the index for which we will use the
                % survival iteration. idx=1 refers to the index at time=0,
                % which corresponds to the survival of 1 (we know that the
                % patient is alive at time=0). Hence we will use
                % surv(t_est_idx), which refers to the survival value that
                % used the x value of the previous time point, but
                % corresponds to the right index since we shift the index
                % by 1 (i.e. at time=0, idx=1)
                t_est_idx = int64(t_est / model_coef_est.DeltaT) + 1;

                % Horizon index - used to store the AUC calculations
                horizon_idx = int64(horizon_t_tmp / model_coef_est.DeltaT);
                
                % Weightings to be used in AUC calculation
                weights_crit_test = zeros(no_of_pat_auc_test, 1);

                % KM censoring curve
                [G_cens_test, G_time_test] = ecdf(surv_info_mat(:,1),'censoring', ...
                                                  surv_info_mat(:,2),'function','survivor');
                
                % Find the censoring index that corresponds to the value
                % right before the prediction time
                [G_val, G_idx] = max(G_time_test(G_time_test < t_est));
                if isempty(G_idx)
                    G_idx = 1;
                end

                % Calculate the associated weight for every patient
                for ii=1:no_of_pat_auc_test % for every patient
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

                auc_num_sum = 0; % AUC numerator
                auc_den_sum = 0; % AUC denominator

                % Calculate the numerator and denominator values
                for ii=1:no_of_pat_auc_test % for every patient
                    pat_ii = pat_data(ii); % Store patient info
                    
                    % Check if patient experienced event within prediction
                    % time
                    D_i = pat_ii.surv_time <= t_est && pat_ii.delta_ev == 1;

                    if D_i % if they experienced event in time of interest
                        % Survival function for patient ii
                        surv_fn_ii = pat_ii.forecasts.surv;
                        W_i = weights_crit_test(ii);

                        for jj=1:no_of_pat_auc_test % for every patient
                            pat_jj = pat_data(jj); % store second patient data
                            
                            % Survival function for patient jj
                            surv_fn_jj = pat_jj.forecasts.surv;
                            % Check if patient experienced event within
                            % prediction time
                            D_j = pat_jj.surv_time <= t_est && pat_jj.delta_ev == 1;
                            W_j = weights_crit_test(jj);

                            % Dynamic predicted survival given it is known
                            % they survived beyond landmark time
                            pred_Surv_i = surv_fn_ii(1,t_est_idx) / surv_fn_ii(1,landmark_idx_tmp);
                            pred_Surv_j = surv_fn_jj(1,t_est_idx) / surv_fn_jj(1,landmark_idx_tmp);

                            % Concordance between patient i and j
                            c_idx_ind = pred_Surv_i < pred_Surv_j;

                            auc_num_sum = auc_num_sum + (c_idx_ind * D_i * (1 - D_j) * W_i * W_j);
                            auc_den_sum = auc_den_sum + (D_i * (1 - D_j) * W_i * W_j);
                            
                        end
                    end
                end

                auc_test_arr(horizon_idx) = auc_num_sum / auc_den_sum;
            end
            
        end
        
        
        function [pe_test_arr] = Prediction_Error_fn(pat_data, landmark_t, max_horizon_t, model_coef_est) 
            % FUNCTION NAME:
            %   Prediction_Error_fn
            %
            % DESCRIPTION:
            %   Finds the Prediction Error for several horizons at a
            %   particular landmark. The formulation is based on the one
            %   provided by Rizopoulos et al. in their paper "Dynamic
            %   predictions with time-dependent covariates in survival
            %   analysis using joint modeling and landmarking".
            %
            % INPUT:
            %   pat_data - (map) Observed data of all patients still at
            %              risk at landmark time, including survival
            %              forecasts, survival time, and event indicator.
            %   landmark_t - (double) Time at which forecasts are made, and
            %                hence, the start of AUC calculations.
            %   max_horizon_t - (double) The maximum horizon to work out
            %                   the AUC calculations.
            %   model_coef_est - (struct) Contains all model parameters for
            %                    LSDSM.
            %
            % OUTPUT:
            %   pe_test_arr - (array) All Prediction Error calculations for 
            %                 the horizons and landmark indicated.
            % 
            % REVISION HISTORY:
            %   17/12/2022 - mcauchi1
            %       * Initial implementation
            %
            
            % +1 due to index starts from 1
            landmark_idx_tmp = floor(landmark_t / model_coef_est.DeltaT) + 1;
            
            % initialise the Prediction Error array
            pe_test_arr = zeros(1, int64(max_horizon_t / model_coef_est.DeltaT));
            
            no_of_pat_pe_test = double(pat_data.Count); % total number of patients at risk at landmark time

            % For every horizon until maximum horizon
            for horizon_t_tmp=model_coef_est.DeltaT:model_coef_est.DeltaT:max_horizon_t
                % Find the prediction time for the current horizon
                t_est = landmark_t + horizon_t_tmp;
                t_est_idx = int64(t_est / model_coef_est.DeltaT)+1;

                horizon_idx_tmp = int64(horizon_t_tmp / model_coef_est.DeltaT);

                for ii=1:no_of_pat_pe_test % for every patient
                    pat_ii = pat_data(ii); % store patient data
                    surv_fn = pat_ii.forecasts.surv; % extract survival forecast
                    
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
                        % Extract the predicted survival, the censoring
                        % index (if censored), and pi as explained in the
                        % formulation
                        pred_Surv = surv_fn(1,t_est_idx) / surv_fn(1,landmark_idx_tmp);
                        censor_time_idx = floor(pat_ii.surv_time / model_coef_est.DeltaT)+1;
                        pred_surv_pi = surv_fn(1,t_est_idx) / surv_fn(1,censor_time_idx);
                        
                        % PE formulation
                        to_sum = ind_risk * (1 - pred_Surv)^2 + ...
                                 ind_tmp * (0 - pred_Surv)^2 + ...
                                 ind_cens * (pred_surv_pi * (1 - pred_Surv)^2 + ...
                                             (1 - pred_surv_pi) * (0 - pred_Surv)^2);
                        pe_test_arr(horizon_idx_tmp) = pe_test_arr(horizon_idx_tmp) + to_sum;

                    end

                end

                % Dividing by number of patients
                pe_test_arr(horizon_idx_tmp) = pe_test_arr(horizon_idx_tmp) / no_of_pat_pe_test; 
            end
            
        end
        
        
        function [bs_test_arr] = Brier_Score_fn(pat_data, landmark_t, max_horizon_t, model_coef_est)              
            % FUNCTION NAME:
            %   Brier_Score_fn
            %
            % DESCRIPTION:
            %   Finds the Brier Score for several horizons at a particular
            %   landmark. The formulation is based on the one provided by
            %   Blanche et al. in their paper "Quantifying and Comparing
            %   Dynamic Predictive Accuracy of Joint Models for
            %   Longitudinal Marker and Time-to-Event in Presence of
            %   Censoring and Competing Risks".
            %
            % INPUT:
            %   pat_data - (map) Observed data of all patients still at
            %              risk at landmark time, including survival
            %              forecasts, survival time, and event indicator.
            %   landmark_t - (double) Time at which forecasts are made, and
            %                hence, the start of AUC calculations.
            %   max_horizon_t - (double) The maximum horizon to work out
            %                   the AUC calculations.
            %   model_coef_est - (struct) Contains all model parameters for
            %                    LSDSM.
            %
            % OUTPUT:
            %   bs_test_arr - (array) All Brier Score calculations for the
            %                 horizons and landmark indicated.
            % 
            % REVISION HISTORY:
            %   17/12/2022 - mcauchi1
            %       * Initial implementation
            %
            
            % +1 due to index starts from 1
            landmark_idx_tmp = floor(landmark_t / model_coef_est.DeltaT) + 1;
            
            % initialise the Brier Score array
            bs_test_arr = zeros(1, int64(max_horizon_t / model_coef_est.DeltaT));
            
            % total number of patients at risk at landmark time
            no_of_pat_bs_test = double(pat_data.Count);
            
            % surv time and delta matrix
            surv_info_mat = zeros(no_of_pat_bs_test, 2);
            
            for ii=1:no_of_pat_bs_test
                surv_info_mat(ii,:) = [pat_data(ii).surv_time, pat_data(ii).delta_ev];
            end

            % For every horizon we wish to consider
            for horizon_t_tmp=model_coef_est.DeltaT:model_coef_est.DeltaT:max_horizon_t

                % Find the prediction time for the current horizon
                t_est = landmark_t + horizon_t_tmp;
                t_est_idx = int64(t_est / model_coef_est.DeltaT) + 1;

                horizon_idx_tmp = int64(horizon_t_tmp / model_coef_est.DeltaT);

                % S(t) = P(T_i > t)
                % G(t) = P(C > t)
                [G_cens_test, G_time_test] = ecdf(surv_info_mat(:,1),'censoring',...
                                                  surv_info_mat(:,2),'function','survivor');

                % Finding the index for the Censored Kaplan Meier at t_est               
                [G_val, G_idx] = max(G_time_test(G_time_test < t_est));
                if isempty(G_idx)
                    G_idx = 1;
                end

                for ii=1:no_of_pat_bs_test % for every patient
                    pat_ii = pat_data(ii); % store patient data
                    surv_fn = pat_ii.forecasts.surv; % extract survival forecasts
                    
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
                        % Dynamic predicted survival
                        pred_Surv = surv_fn(1,t_est_idx) / surv_fn(1,landmark_idx_tmp);
                        % Brier Score formulation
                        to_sum = (0 - pred_Surv)^2 * ind_tmp / G_cens_test(G_idx_tmp) + ...
                                    (1 - pred_Surv)^2 * ind_risk / G_cens_test(G_idx);
                        bs_test_arr(horizon_idx_tmp) = bs_test_arr(horizon_idx_tmp) + to_sum;

                    end

                end

                % Dividing by number of patients
                bs_test_arr(horizon_idx_tmp) = bs_test_arr(horizon_idx_tmp) / no_of_pat_bs_test; 
            end
            
        end
        
        
        function [pat_data_out] = predict_fn(pat_data, t_est_idx, max_censor_time, model_coef_est, controls)
            % FUNCTION NAME:
            %   predict_fn
            %
            % DESCRIPTION:
            %   Finds the smoothed hidden states and survival curves for
            %   the patients given all data is available. This is to be
            %   used when performing simulations such that a comparison can
            %   be made between different models on the smoothed outputs.
            %
            % INPUT:
            %   pat_data - (map) Observed data of all patients including
            %              longitudinal biomarkers, survival time, and 
            %              event indicator.
            %   t_est_idx - (double) Time index to make predictions.
            %   max_censor_time - (double) The maximum censoring time.
            %   model_coef_est - (struct) Contains all model parameters for
            %                    LSDSM.
            %   controls - (struct) Contains the controls for the LSDSM EM,
            %              including the boolean to indicate if survival
            %              data should affect hidden states.
            %
            % OUTPUT:
            %   pat_data_out - (map) All observed patient data, together
            %                  with the prediction outputs.
            % 
            % REVISION HISTORY:
            %   17/12/2022 - mcauchi1
            %       * Initial implementation
            %
            
            % Calculate the maximum index
            max_num_pts = ceil(max_censor_time / model_coef_est.DeltaT)+1;
            
            % Create a map to store the output
            pat_data_out = containers.Map('KeyType', 'int32', 'ValueType', 'any');
            
            for ii=1:pat_data.Count % for every patient

                pat_ii = pat_data(ii); % store patient data
                
                % Set initial conditions
                pat_ii.mu_0 = model_coef_est.mu_0;
                pat_ii.W_0 = model_coef_est.W_0;

                
                % RTS Filter
                [pat_ii.mu_tilde, pat_ii.V_tilde] = ...
                    LSDSM_ALLFUNCS.Kalman_filter(pat_ii, model_coef_est, max_num_pts, controls.mod_KF);

                % RTS Smoother
                [pat_ii.predictions.mu, pat_ii.predictions.V, pat_ii.predictions.J] = ...
                    LSDSM_ALLFUNCS.Kalman_smoother(pat_ii, model_coef_est, max_num_pts);

                % Survival curve estimation and predictions
                cum_haz_fn = 0;
                haz_fn = zeros(1, max_num_pts);

                % patient's survival at time = 0 is 1
                pat_ii.predictions.surv(:,1) = 1;

                for j=1:t_est_idx
                    % the jth value of x will predict the survival of patient at (j+1)
                    haz_fn(:,j) = exp(model_coef_est.g_s' * pat_ii.base_cov) * ...
                                      exp(model_coef_est.a_s' * pat_ii.predictions.mu(:,1,j));
                    cum_haz_fn = cum_haz_fn + model_coef_est.DeltaT * haz_fn(:,j);
                    pat_ii.predictions.surv(:,j+1) = exp(-cum_haz_fn);
                end
                
                pat_data_out(ii) = pat_ii; % Store patient data in output map
            end
        end
        
        
        function [pat_data_out] = forecast_fn(pat_data, landmark_t, t_est_idx, ...
                                                  max_censor_time, model_coef_est, controls)
            % FUNCTION NAME:
            %   forecast_fn
            %
            % DESCRIPTION:
            %   Makes forecasts on hidden states and survival curves for
            %   the patients given the data available until the landmark
            %   time.
            %
            % INPUT:
            %   pat_data - (map) Observed data of all patients including
            %              longitudinal biomarkers, survival time, and 
            %              event indicator.
            %   landmark_t - (double) Landmark time at which to start the
            %                forecasts
            %   t_est_idx - (double) Time index to make forecasts.
            %   max_censor_time - (double) The maximum censoring time.
            %   model_coef_est - (struct) Contains all model parameters for
            %                    LSDSM.
            %   controls - (struct) Contains the controls for the LSDSM EM,
            %              including the boolean to indicate if survival
            %              data should affect hidden states.
            %
            % OUTPUT:
            %   pat_data_out - (map) All observed patient data, together
            %                  with the forecast outputs.
            % 
            % REVISION HISTORY:
            %   17/12/2022 - mcauchi1
            %       * Initial implementation
            %
            
            % Find the landmark index
            % +1 due to index starts from 1
            landmark_idx = floor(landmark_t / model_coef_est.DeltaT) + 1;
            
            % Find the maximum index
            max_num_pts = ceil(max_censor_time / model_coef_est.DeltaT)+1;
            
            % Stores patient data and forecasts for surviving patients beyond landmark time
            pat_data_out = containers.Map('KeyType', 'int32', 'ValueType', 'any');
            
            red_ii = 0; % used to keep track of the number of patients that survive beyond the landmark point

            for ii=1:pat_data.Count % iterate for every patient

                pat_ii = pat_data(ii); % Store the patient data
                
                if pat_ii.surv_time > landmark_t % forecast if they survived beyond the landmark
                    
                    % increase the iterative for the reduced patient data set
                    red_ii = red_ii + 1;
                    
                    % Set initial conditions
                    pat_ii.mu_0 = model_coef_est.mu_0;
                    pat_ii.W_0 = model_coef_est.W_0;

                    
                    % RTS Filter
                    [pat_ii.mu_tilde, pat_ii.V_tilde] = ...
                        LSDSM_ALLFUNCS.Kalman_filter(pat_ii, model_coef_est, max_num_pts, controls.mod_KF);

                    % Utilise the RTS filter output until the landmark (one time step before)
                    pat_ii.forecasts.mu(:,:,1:landmark_idx-1) = pat_ii.mu_tilde(:,:,1:landmark_idx-1);
                    pat_ii.forecasts.V(:,:,1:landmark_idx-1) = pat_ii.V_tilde(:,:,1:landmark_idx-1);

                    % We cannot use the output of the RTS filter at the landmark time,
                    % as this is utilising survival information of the next period.
                    % However, we can use a single step of the standard RTS filter,
                    % where we utilise just the longitudinal data available at that time.
                    [mu_filt, V_filt, K_filt, P_filt] = ...
                            LSDSM_ALLFUNCS.KF_single_step(pat_ii.mu_tilde(:,:,landmark_idx-1), ...
                                                          pat_ii.V_tilde(:,:,landmark_idx-1), ...
                                                          pat_ii.y(:,:,landmark_idx), model_coef_est);

                    % The above filtering will allow us to update the hidden states based
                    % on the longitudinal information available at the landmark. This will
                    % result in predictions starting from the next iteration.
                    pat_ii.forecasts.mu(:,:,landmark_idx) = mu_filt;
                    pat_ii.forecasts.V(:,:,landmark_idx) = V_filt;

                    % Forecasts
                    for j=landmark_idx+1:t_est_idx
                        pat_ii.forecasts.mu(:,:,j) = model_coef_est.A * pat_ii.forecasts.mu(:,:,j-1);
                        pat_ii.forecasts.V(:,:,j) = model_coef_est.A * pat_ii.forecasts.V(:,:,j-1) * model_coef_est.A' ...
                                                     + model_coef_est.G_mat * model_coef_est.W * model_coef_est.G_mat';
                    end
                    
                    % Survival curve estimation and forecasts
                    cum_haz_fn = 0;
                    haz_fn = zeros(1, max_num_pts);
                    
                    % patient's survival at time = 0 is 1
                    pat_ii.forecasts.surv(:,1) = 1;

                    for j=1:t_est_idx
                        % the jth value of x will predict the survival of patient at (j+1)
                        haz_fn(:,j) = exp(model_coef_est.g_s' * pat_ii.base_cov) * ...
                                                              exp(model_coef_est.a_s' * pat_ii.forecasts.mu(:,1,j));
                        cum_haz_fn = cum_haz_fn + model_coef_est.DeltaT * haz_fn(:,j);
                        pat_ii.forecasts.surv(:,j+1) = exp(-cum_haz_fn);
                    end
                    % Store patient data in output map
                    pat_data_out(red_ii) = pat_ii;
                end
            end
        end
        
        
        function [model_coef_est_out, max_iter, param_traj, RTS_arrs] = ...
                        LSDSM_EM(dim_size, pat_data, controls, max_censor_time)
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
            %              "dynamic" states, and biomarkers
            %   pat_data - (map) Contains patient data with every key
            %              representing a single patient. In each entry, a
            %              struct with number of iterations observed,
            %              boolean representing whether patient experienced
            %              event, survival time, longitudinal biomarkers,
            %              and baseline covariates.
            %   controls - (struct) Controls for the EM algorithm,
            %              including number of iterations, maximum
            %              parameter difference for stopping criteria,
            %              initial parameters, fixed parameters, and
            %              boolean for modified filter equations
            %   max_censor_time - (double) Indicates the maximum time to
            %                     observe every patient
            %
            % OUTPUT:
            %   model_coef_est_out - (struct) Estimated Parameters
            %   max_iter - (double) Number of EM iterations executed
            %   param_traj_tmp - (struct) Evolution of parameter values
            %                    over EM iterations
            %   RTS_arrs - (struct) Contains the filtered and smoothed
            %              outputs of the hidden state trajectories.
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
            param_traj.W = zeros(dim_size.dyn_states, dim_size.dyn_states, controls.EM_iters);
            param_traj.V = zeros(dim_size.y, dim_size.y, controls.EM_iters);
            param_traj.g_s = zeros(size(pat_data(1).base_cov,1), 1, controls.EM_iters);
            param_traj.a_s = zeros(dim_size.states, 1, controls.EM_iters);
            param_traj.mu_0 = zeros(dim_size.states, 1, num_pats, controls.EM_iters);
            param_traj.W_0 = zeros(dim_size.states, dim_size.states, num_pats, controls.EM_iters);
            
            % Initialise the log likelihood array
            log_like_val_tot_arr = zeros(num_pats, controls.EM_iters);

            % Set the first value (in iter_EM) as the initial estimates
            param_traj.A(:,:,1) = model_coef_est.A;
            param_traj.C(:,:,1) = model_coef_est.C;
            param_traj.W(:,:,1) = model_coef_est.W;
            param_traj.V(:,:,1) = model_coef_est.V;
            param_traj.g_s(:,:,1) = model_coef_est.g_s;
            param_traj.a_s(:,:,1) = model_coef_est.a_s;

            if controls.update_pop_mu % If we're updating global initial conditions
                param_traj.mu_0(:,:,1) = model_coef_est.mu_0;
                param_traj.W_0(:,:,1) = model_coef_est.W_0;
            else
                param_traj.mu_0(:,:,:,1) = model_coef_est.mu_0;
                param_traj.W_0(:,:,:,1) = model_coef_est.W_0;
            end

            % Store the states obtained from the RTS filter/smoother
            RTS_arrs.mu_tilde = zeros(dim_size.states, 1, max_num_pts, num_pats);
            RTS_arrs.V_tilde = zeros(dim_size.states, dim_size.states, max_num_pts, num_pats);
            RTS_arrs.mu_hat = zeros(dim_size.states, 1, max_num_pts, num_pats);
            RTS_arrs.V_hat = zeros(dim_size.states, dim_size.states, max_num_pts, num_pats);
            

            for j=2:controls.EM_iters % iterate the EM algorithm (iter_EM-1) times
                if controls.verbose
                    if mod(j,10) == 0 % Feedback every 10 iterations
                        fprintf("EM Iteration %d / %d. Max Parameter Difference: %.6f \n", ...
                                    j, controls.EM_iters, param_max_diff);
                    end
                end

                % Initialisation of summations to be used at the E and M steps
                % E_sums is a struct containing all summations of the
                % expectations required
                % n - longitudinal iterations
                % N - number of longitudinal measurements across time
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
                
                % Stores the sum across all patients of E[x(0)]
                E_sums.x0 = zeros(dim_size.states, 1);
                % Stores the sum across all patients of E[x(0) x(0)']
                E_sums.x0_x0 = zeros(dim_size.states, dim_size.states);
                
                
                for ii=1:num_pats % for every patient
                    % Capture information from current patient
                    pat_ii = pat_data(ii);
                    
                    % Set initial state conditions for patient ii
                    if controls.update_pop_mu
                        pat_ii.mu_0 = model_coef_est.mu_0;
                        pat_ii.W_0 = model_coef_est.W_0;
                    else
                        pat_ii.mu_0 = model_coef_est.mu_0(:,:,ii);
                        pat_ii.W_0 = model_coef_est.W_0(:,:,ii);
                    end
                    
                    %%%%%%%%%%%%%%
                    %%% E Step %%%
                    %%%%%%%%%%%%%%
                    %%% Forward recursion - Standard/Modified RTS Filter %%%
                    [pat_ii.mu_tilde, pat_ii.V_tilde, log_like_val] = ...
                        LSDSM_ALLFUNCS.Kalman_filter(pat_ii, model_coef_est, max_num_pts, controls.mod_KF);
                    
                    % Store the log likelihood value for this patient
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
                
                % Find the updated parameters
                model_coef_new = LSDSM_ALLFUNCS.M_step(pat_data, E_sums, RTS_arrs, model_coef_est, controls);

                % store the old coefficients for stopping criterion
                model_coef_est_old = model_coef_est;

                % Update the estimates to be used in the next iteration
                model_coef_est = model_coef_new;
                
                % Store the updated estimates in the respective arrays
                param_traj.A(:,:,j) = model_coef_est.A;
                param_traj.C(:,:,j) = model_coef_est.C;
                param_traj.W(:,:,j) = model_coef_est.W;
                param_traj.V(:,:,j) = model_coef_est.V;
                param_traj.g_s(:,:,j) = model_coef_est.g_s;
                param_traj.a_s(:,:,j) = model_coef_est.a_s;
                if controls.update_pop_mu
                    param_traj.mu_0(:,:,j) = model_coef_est.mu_0;
                    param_traj.W_0(:,:,j) = model_coef_est.W_0;
                else
                    param_traj.mu_0(:,:,:,j) = model_coef_est.mu_0;
                    param_traj.W_0(:,:,:,j) = model_coef_est.W_0;
                end
                
                model_coef_est_out = model_coef_est; % Update output parameters

                [param_diff, param_diff_percent] = ...
                        LSDSM_ALLFUNCS.find_model_param_diff(model_coef_est, model_coef_est_old);

                % Find maximum absolute parameter difference
                param_max_diff = max(abs(param_diff));

                % If stopping criteria is reached
                if param_max_diff < controls.max_param_diff
                    break; % Break from EM algorithm for loop
                end
            end

            max_iter = j;
            
            if controls.allow_plots
                % Plot the log likelihood over EM iterations
                figure;
                plot(sum(log_like_val_tot_arr(:,2:j), 1))
                title('Log likelihood over EM iterations');
                ylabel('Log likelihood');
                xlabel('EM iteration');
            end

            fprintf('Maximum Parameter Difference: %6f at iteration %d \n', param_max_diff, j);
        end
        
        
        function [param_diff, param_diff_percent] = find_model_param_diff(model_coef1, model_coef2)
            % FUNCTION NAME:
            %   find_model_param_diff
            %
            % DESCRIPTION:
            %   Finds the difference in parameters between two models.
            %
            % INPUT:
            %   model_coef1 - (struct) Contains the parameters for the
            %                 first model
            %   model_coef2 - (struct) Contains the parameters for the
            %                 second model
            %
            % OUTPUT:
            %   param_diff - (array) Array of parameter differences
            %   param_diff_percent - (array) Percentage difference between
            %                        the models for every parameter
            %
            % REVISION HISTORY:
            %   17/12/2022 - mcauchi1
            %       * Initial implementation
            %
            
            % Removes DeltaT, G_mat, mu_0, W_0 from this parameter difference
            true_model_coef_cmp = rmfield(model_coef1, {'DeltaT', 'G_mat', 'mu_0', 'W_0'});
            model_coef_est_cmp = rmfield(model_coef2, {'DeltaT', 'G_mat', 'mu_0', 'W_0'});
            
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
            % FUNCTION NAME:
            %   find_rmse_states
            %
            % DESCRIPTION:
            %   Find the root mean square error between comp_signal and
            %   true hidden states (only valid for simulations).
            %
            % INPUT:
            %   data_latent - (map) Contains the hidden data for all
            %                 patients.
            %   data_observed - (map) Contains the observed data for all
            %                   patients.
            %   comp_signal - (array) Contains estimated state trajectory
            %                 for every patient.
            %
            % OUTPUT:
            %   rmse_states - (double) Root Mean Square Error between the
            %                 true and comparison signals.
            %
            % REVISION HISTORY:
            %   17/12/2022 - mcauchi1
            %       * Initial implementation
            %
            

            total_sse_val = zeros(size(comp_signal,1), 1);
            total_obs_test = 0;
            t_dim = 3; % time is along this dimension in the array

            for ii=1:data_observed.Count % for every patient
                % Extract the true hidden state trajectories
                true_sig_ii = data_latent(ii).x_true(:,:,1:data_observed(ii).m_i);
                % Extract the comparison signals
                comp_sig_ii = comp_signal(:,:,1:data_observed(ii).m_i,ii);
                % count the total number of observations
                total_obs_test = total_obs_test + data_observed(ii).m_i; 

                % Find the sum of square error
                [sse_val, mse_val, rmse_val] = LSDSM_ALLFUNCS.rmse_fn(true_sig_ii, comp_sig_ii, t_dim);
                total_sse_val = total_sse_val + sse_val;
            end

            % Mean square error
            mse_states = total_sse_val / total_obs_test;
            % Root mean square error
            rmse_states = sqrt(mse_states);
        end
        
        
        function rmse_surv = find_rmse_surv(true_surv_fn, data_observed, comp_signal)
            % FUNCTION NAME:
            %   find_rmse_surv
            %
            % DESCRIPTION:
            %   Find the root mean square error between comp_signal and
            %   true survival trajectory (only valid for simulations).
            %
            % INPUT:
            %   true_surv_fn - (array) Contains the true survival curves 
            %                  for every patient.
            %   data_observed - (map) Contains the observed data for all
            %                   patients.
            %   comp_signal - (array) Contains estimated survival curves
            %                 for every patient.
            %
            % OUTPUT:
            %   rmse_surv - (double) Root Mean Square Error between the
            %                 true and comparison survival signals.
            %
            % REVISION HISTORY:
            %   17/12/2022 - mcauchi1
            %       * Initial implementation
            %
            
            % Initialise sum of square error
            total_sse_val = zeros(size(comp_signal,1), 1);
            total_obs_test = 0;
            t_dim = 2; % time is along this dimension in the array

            for ii=1:data_observed.Count
                % start from 2 since the first survival probability is
                % always equal to 1
                % Extract the true survival curve
                true_sig_ii = true_surv_fn(:,2:data_observed(ii).m_i+1, ii);
                % Extract the estimated survival curve
                comp_sig_ii = comp_signal(:,2:data_observed(ii).m_i+1, ii);
                % count the total number of observations
                total_obs_test = total_obs_test + data_observed(ii).m_i;

                % Calculate the sum of square error
                [sse_val, mse_val, rmse_val] = LSDSM_ALLFUNCS.rmse_fn(true_sig_ii, comp_sig_ii, t_dim);
                total_sse_val(:,1) = total_sse_val(:,1) + sse_val;
            end

            % Mean square error
            mse_surv = total_sse_val / total_obs_test;
            % Root mean square error
            rmse_surv = sqrt(mse_surv);
        end
        
        
        function surv_fn = surv_curve_calc(model_coef_est, data_observed, state_vals)
            % FUNCTION NAME:
            %   surv_curve_calc
            %
            % DESCRIPTION:
            %   Calculates the survival curves for all patients based on
            %   the provided model, hidden state values, and the observed
            %   data.
            %
            % INPUT:
            %   model_coef_est - (struct) Model parameters.
            %   data_observed - (map) Contains the observed data for all
            %                   patients.
            %   state_vals - (array) Hidden state trajectories for all
            %                patients.
            %
            % OUTPUT:
            %   surv_fn - (array) The survival curves estimation for all
            %             patients.
            %
            % REVISION HISTORY:
            %   17/12/2022 - mcauchi1
            %       * Initial implementation
            %
            
            % Initialise placeholder for survival trajectories
            surv_fn = zeros(1, size(state_vals,3)+1, data_observed.Count);

            for ii=1:data_observed.Count % for every patient
                % Initialise hazard
                cum_haz = 0;
                haz_fn = zeros(1, size(state_vals,3), data_observed.Count);
                
                % capture (predicted) hidden states and baseline covariates
                x_ii = state_vals(:,:,:,ii);
                base_cov_ii = data_observed(ii).base_cov;

                % at time = 0, surv = 1
                surv_fn(:,1,ii) = 1;

                % Number of longitudinal observations for patient ii
                m_i = data_observed(ii).m_i;
                for j=1:m_i % for every time index
                    % Calculate current hazard
                    haz_fn(:,j) = exp(model_coef_est.g_s' * base_cov_ii) * exp(model_coef_est.a_s' * x_ii(:,1,j));
                    % Update cumulative hazard
                    cum_haz = cum_haz + model_coef_est.DeltaT * haz_fn(:,j);
                    % Record new survival probability
                    surv_fn(:,j+1,ii) = exp(-cum_haz);
                end
            end
        end


        function dGout = dGdx(g_a_s_prev, dG_data)
            % FUNCTION NAME:
            %   dGdx
            %
            % DESCRIPTION:
            %   First Derivative of the Expectation with respect to the
            %   survival parameters: gamma (g_s) and alpha (a_s).
            %
            % INPUT:
            %   g_a_s_prev - (array) Previous parameter values of g_s and
            %                a_s.
            %   dG_data - (struct) Contains the required data to compute
            %             the derivative.
            %
            % OUTPUT:
            %   dGout - (array) The derivative of the expectation with
            %           respect to the survival parameters.
            %
            % REVISION HISTORY:
            %   17/12/2022 - mcauchi1
            %       * Initial implementation
            %

            % Store the previous values of g_s and a_s
            g_s_prev = g_a_s_prev(1:size(dG_data.pat_data(1).base_cov,1), 1);
            a_s_prev = g_a_s_prev(size(dG_data.pat_data(1).base_cov,1)+1:end, 1);

            % Initiate the derivatives to zero
            dGdg_s = 0;
            dGda_s = 0;

            for ii=1:dG_data.pat_data.Count % For every patient
                % Extract the patient data
                pat_ii = dG_data.pat_data(ii);
                % Extract the baseline covariates
                base_cov_ii = pat_ii.base_cov;
                
                % Extract the smoothed outputs
                mu_hat_tmp = dG_data.RTS_arrs.mu_hat(:,:,:,ii);
                V_hat_tmp = dG_data.RTS_arrs.V_hat(:,:,:,ii);

                for j=1:pat_ii.m_i % For every observation
                    % Check if patient experienced event in current time frame
                    [delta_ij, tau_ij] = LSDSM_ALLFUNCS.pat_status_at_j(j, pat_ii, dG_data.model_coef_est.DeltaT);

                    % Auxiliary variable
                    scalar_tmp = tau_ij * exp(g_s_prev' * base_cov_ii) * ...
                        exp(a_s_prev' * mu_hat_tmp(:,:,j) + 1/2 * a_s_prev' * V_hat_tmp(:,:,j) * a_s_prev);

                    % Work out the derivatives with respect to survival parameters
                    dGdg_s = dGdg_s + delta_ij * base_cov_ii - scalar_tmp * base_cov_ii;
                    dGda_s = dGda_s + delta_ij * mu_hat_tmp(:,:,j) - ...
                        scalar_tmp * (mu_hat_tmp(:,:,j) + V_hat_tmp(:,:,j) * a_s_prev);
                end
            end

            dGout = [dGdg_s; dGda_s];
        end

        
        function d2Gout = d2Gdx2(g_a_s_prev, dG_data)
            % FUNCTION NAME:
            %   d2Gdx2
            %
            % DESCRIPTION:
            %   Hessian matrix (second derivative) of the Expectation with
            %   respect to the survival parameters: gamma (g_s) and alpha
            %   (a_s).
            %
            % INPUT:
            %   g_a_s_prev - (array) Previous parameter values of g_s and
            %                a_s.
            %   dG_data - (struct) Contains the required data to compute
            %             the derivative.
            %
            % OUTPUT:
            %   d2Gout - (array) The second derivative of the expectation 
            %            with respect to the survival parameters.
            %
            % REVISION HISTORY:
            %   17/12/2022 - mcauchi1
            %       * Initial implementation
            %

            % Store the previous values of g_s and a_s
            g_s_prev = g_a_s_prev(1:size(dG_data.pat_data(1).base_cov,1), 1);
            a_s_prev = g_a_s_prev(size(dG_data.pat_data(1).base_cov,1)+1:end, 1);

            % Initiate the second derivatives to zero
            d2Gdg_s2 = 0;
            d2Gdg_a_s = 0;
            d2Gda_s2 = 0;

            for ii=1:dG_data.pat_data.Count % For every patient
                % Extract patient data
                pat_ii = dG_data.pat_data(ii);
                % Extract baseline covariates
                base_cov_ii = pat_ii.base_cov;

                % Extract smoothed outputs
                mu_hat_tmp = dG_data.RTS_arrs.mu_hat(:,:,:,ii);
                V_hat_tmp = dG_data.RTS_arrs.V_hat(:,:,:,ii);
                
                for j=1:pat_ii.m_i % For every observation
                    % Check if patient experienced event in current time frame
                    [delta_ij, tau_ij] = LSDSM_ALLFUNCS.pat_status_at_j(j, pat_ii, dG_data.model_coef_est.DeltaT);

                    % Auxiliary variables
                    scalar_tmp = tau_ij * exp(g_s_prev' * base_cov_ii) * ...
                        exp(a_s_prev' * mu_hat_tmp(:,:,j) + 1/2 * a_s_prev' * V_hat_tmp(:,:,j) * a_s_prev);

                    mu_sigma_alpha = (mu_hat_tmp(:,:,j) + V_hat_tmp(:,:,j) * a_s_prev);
                    
                    % Second derivative calculations
                    d2Gdg_s2 = d2Gdg_s2 - scalar_tmp * (base_cov_ii * base_cov_ii');
                    d2Gdg_a_s = d2Gdg_a_s - scalar_tmp * (base_cov_ii * mu_sigma_alpha');
                    d2Gda_s2 = d2Gda_s2 - scalar_tmp * ((mu_sigma_alpha * mu_sigma_alpha') + V_hat_tmp(:,:,j));

                end
            end

            % Store the Hessian matrix in the correct format
            d2Gout = [d2Gdg_s2, d2Gdg_a_s;
                      d2Gdg_a_s', d2Gda_s2];
        end

        
        function dgout = dgdx(x_prev, coeffs)
            % FUNCTION NAME:
            %   dgdx
            %
            % DESCRIPTION:
            %   First derivative of the exponent of the product of the
            %   Standard RTS filter and the survival likelihood, with
            %   respect to the hidden states.
            %
            % INPUT:
            %   x_prev - (array) Hidden state values at the previous NR
            %            iteration
            %   coeffs - (struct) Contains the required data to compute
            %            the derivative.
            %
            % OUTPUT:
            %   dgout - (array) The derivative with respect to the hidden
            %           states.
            %
            % REVISION HISTORY:
            %   17/12/2022 - mcauchi1
            %       * Initial implementation
            %
            
            dgout = - coeffs.delta_ij * coeffs.a_s ...
                    + coeffs.tau_ij * exp(coeffs.g_s' * coeffs.base_cov) * exp(coeffs.a_s' * x_prev) * coeffs.a_s ...
                    + coeffs.Sigma_ij^(-1) * (x_prev - coeffs.mu_ij);
        end


        function d2gout = d2gdx2(x_prev, coeffs)
            % FUNCTION NAME:
            %   d2gdx2
            %
            % DESCRIPTION:
            %   Second derivative of the exponent of the product of the
            %   Standard RTS filter and the survival likelihood, with
            %   respect to the hidden states.
            %
            % INPUT:
            %   x_prev - (array) Hidden state values at the previous NR
            %            iteration
            %   coeffs - (struct) Contains the required data to compute
            %            the derivative.
            %
            % OUTPUT:
            %   d2gout - (array) The second derivative with respect to the 
            %            hidden states.
            %
            % REVISION HISTORY:
            %   17/12/2022 - mcauchi1
            %       * Initial implementation
            %
            
            d2gout = coeffs.tau_ij * exp(coeffs.g_s' * coeffs.base_cov) * ... 
                        exp(coeffs.a_s' * x_prev) * (coeffs.a_s * coeffs.a_s') ...
                     + coeffs.Sigma_ij^(-1);
        end

        
        function x_NR_tmp = Newton_Raphson(init_val, max_iter, eps, dfdx, d2fdx2, coeffs)
            % FUNCTION NAME:
            %   Newton_Raphson
            %
            % DESCRIPTION:
            %   Newton Raphson's iterative method to find the roots (in
            %   this case, to find where the derivatives are at zero).
            %
            % INPUT:
            %   init_val - (array) The initial value to start NR iterations
            %              from.
            %   max_iter - (double) The maximum number of iterations to
            %              find the roots.
            %   eps - (double) Stopping criteria based on absolute
            %         difference in the differentiating variable.
            %   dfdx - (function name) Function finding the first
            %          derivative with respect to the variable of interest.
            %   d2fdx2 - (function name) Function finding the second
            %            derivative with respect to the variable of
            %            interest.
            %   coeffs - (struct) Contains all required data to calculate
            %            the above first and second derivatives.
            %
            % OUTPUT:
            %   x_NR_tmp - (array) The stationary point for the function.
            %
            % REVISION HISTORY:
            %   17/12/2022 - mcauchi1
            %       * Initial implementation
            %
            
            num_dims = size(init_val, 1); % dimension size
            % initialise array for storing single Newton-Raphson procedure
            x_NR_tmp_arr = zeros(num_dims, max_iter);
            x_NR_tmp_arr(:,1) = init_val; % set initial value
            for jj=2:max_iter % NR iterations
                % Find the first derivative
                df = dfdx(x_NR_tmp_arr(:,jj-1), coeffs);
                % Find the second derivative
                d2f = d2fdx2(x_NR_tmp_arr(:,jj-1), coeffs);
                
                if abs(det(d2f)) < 1e-10 % preventing matrix errors during inverse operation
                    x_NR_tmp_arr(:,jj) = x_NR_tmp_arr(:,jj-1);
                    break
                end

                try
                    if strcmp(functions(dfdx).function, 'LSDSM_ALLFUNCS.dGdx')
                        % if we are updating the survival parameters
                        
                        % Parameters to remain fixed throughout EM
                        fixed_params_tmp = coeffs.controls.fixed_params;
                        
                        if not(isnan(fixed_params_tmp.g_s)) && not(isnan(fixed_params_tmp.a_s))
                            % if both parameters are given
                            x_NR_tmp_arr(:,jj) = x_NR_tmp_arr(:,jj-1);
                            
                        elseif not(isnan(fixed_params_tmp.g_s))
                            % if g_s is given
                            
                            % find index of where a_s starts
                            idx_tmp = size(fixed_params_tmp.g_s,1)+1;
                            % reduce derivatives to update a_s only
                            df = df(idx_tmp:end,1); 
                            d2f = d2f(idx_tmp:end,idx_tmp:end);
                            
                            % keep same g_s
                            x_NR_tmp_arr(:,jj) = x_NR_tmp_arr(:,jj-1); 
                            % update a_s
                            x_NR_tmp_arr(idx_tmp:end,jj) = x_NR_tmp_arr(idx_tmp:end,jj-1) - d2f^(-1) * df;
                            
                        elseif not(isnan(fixed_params_tmp.a_s))
                            % if a_s is given
                            
                            idx_tmp = size(fixed_params_tmp.g_s,1); % find index of where g_s finishes
                            % reduce derivatives to update g_s only
                            df = df(1:idx_tmp);
                            d2f = d2f(1:idx_tmp,1:idx_tmp);
                            
                            % keep same a_s
                            x_NR_tmp_arr(:,jj) = x_NR_tmp_arr(:,jj-1);
                            % update g_s
                            x_NR_tmp_arr(1:idx_tmp,jj) = x_NR_tmp_arr(1:idx_tmp,jj-1) - d2f^(-1) * df; 
                            
                        else % if they both need to be estimated
                            x_NR_tmp_arr(:,jj) = x_NR_tmp_arr(:,jj-1) - d2f^(-1) * df;
                        end
                    else % for other NR operations
                        x_NR_tmp_arr(:,jj) = x_NR_tmp_arr(:,jj-1) - d2f^(-1) * df;
                    end
                catch
                    disp('Singular Matrix')
                end

                % Calculate the change in the differentiating variable
                chg = x_NR_tmp_arr(:,jj) - x_NR_tmp_arr(:,jj-1);
                if sqrt(chg' * chg) < eps % converged
                    break
                end
            end
            % Output the final computed value
            x_NR_tmp = x_NR_tmp_arr(:,jj);
        end


        function [mu_tmp, V_tmp, K_tmp, P_tmp] = KF_single_step(mu_ineg1, V_ineg1, y_i, model_coef_est)
            % FUNCTION NAME:
            %   KF_single_step
            %
            % DESCRIPTION:
            %   Single iteration in time of the RTS Filter using
            %   longitudinal data.
            %
            % INPUT:
            %   mu_ineg1 - (array) Filtered mean at the previous iteration.
            %   V_ineg1 - (array) Filtered variance at the previous
            %             iteration.
            %   y_i - (array) The current biomarker observations.
            %   model_coef_est - (struct) Model parameters.
            %
            % OUTPUT:
            %   mu_tmp - (array) Filtered mean at the current iteration
            %            using the current longitudinal data.
            %   V_tmp - (array) Filtered variance at the current iteration
            %           using the current longitudinal data.
            %   K_tmp - (array) Kalman gain at the current iteration
            %   P_tmp - (array) Prediction variance at the previous
            %           iteration.
            %
            % REVISION HISTORY:
            %   17/12/2022 - mcauchi1
            %       * Initial implementation
            %

            % Identify and restructure the missing values accordingly
            [y_tmp_i, C_tmp_i, V_tmp_i, Omega_O, Omega_M, I_mat_O, ...
                        I_mat_M, nabla_ij] = LSDSM_ALLFUNCS.missing_val_matrices(y_i, model_coef_est);

            P_tmp = model_coef_est.A * V_ineg1 * model_coef_est.A' ...
                    + model_coef_est.G_mat * model_coef_est.W * model_coef_est.G_mat';
            K_tmp = P_tmp * C_tmp_i' * (C_tmp_i * P_tmp * C_tmp_i' + V_tmp_i)^-1;
            mu_tmp = model_coef_est.A * mu_ineg1 + K_tmp * ( y_tmp_i - C_tmp_i *  model_coef_est.A * mu_ineg1 );
            V_tmp = (eye(size(mu_ineg1,1)) - K_tmp * C_tmp_i) * P_tmp;
        end

        
        function [mu_hat_tmp, V_hat_tmp, J_tmp] = KS_single_step(mu_hat_iplus1, V_hat_iplus1, ...
                                                                 mu_tilde_tmp, V_tilde_tmp, model_coef_est)
            % FUNCTION NAME:
            %   KS_single_step
            %
            % DESCRIPTION:
            %   Single iteration in time of the Standard RTS Smoother.
            %
            % INPUT:
            %   mu_hat_iplus1 - (array) Smoothed mean at the next
            %                   iteration.
            %   V_hat_iplus1 - (array) Smoothed variance at the next
            %                  iteration.
            %   mu_tilde_tmp - (array) Filtered mean at the current
            %                  iteration.
            %   V_tilde_tmp - (array) Filtered variance at the current
            %                 iteration.
            %   model_coef_est - (struct) Model parameters.
            %
            % OUTPUT:
            %   mu_hat_tmp - (array) Smoothed mean at the current
            %                iteration.
            %   V_hat_tmp - (array) Smoothed variance at the current
            %               iteration.
            %   J_tmp - (array) Auxiliary matrix
            %
            % REVISION HISTORY:
            %   17/12/2022 - mcauchi1
            %       * Initial implementation
            %
            
            P_tmp = model_coef_est.A * V_tilde_tmp * model_coef_est.A' ...
                    + model_coef_est.G_mat * model_coef_est.W * model_coef_est.G_mat';
            J_tmp = V_tilde_tmp * model_coef_est.A' * P_tmp^-1;
            mu_hat_tmp = mu_tilde_tmp + J_tmp * (mu_hat_iplus1 - model_coef_est.A * mu_tilde_tmp);
            V_hat_tmp = V_tilde_tmp + J_tmp * (V_hat_iplus1 - P_tmp) * J_tmp';
        end

        
        function [y_tmp_i, C_tmp_i, V_tmp_i, Omega_O, Omega_M, I_mat_O, ...
                    I_mat_M, nabla_ij] = missing_val_matrices(y_tmp_i, model_coef_est)
            % FUNCTION NAME:
            %   missing_val_matrices
            %
            % DESCRIPTION:
            %   Arranges the biomarker array at the current time iteration,
            %   replacing NaNs, and informing the algorithm with the
            %   missing observations.
            %
            % INPUT:
            %   y_tmp_i - (array) Biomarker measurements at the current
            %             iteration.
            %   model_coef_est - (struct) Model parameters.
            %
            % OUTPUT:
            %   y_tmp_i - (array) Corrected biomarker measurements at the
            %             current iteration.
            %   C_tmp_i - (array) Observation matrix after correcting for
            %             missing observations.
            %   V_tmp_i - (array) Observation variance matrix after
            %             correcting for the missing observations.
            %   Omega_O - (array) Matrix used to extract only the observed
            %             measurements.
            %   Omega_M - (array) Matrix used to extract only the missing
            %             measurements.
            %   I_mat_O - (array) Matrix used to zero out the missing
            %             measurements, retaining the original size of the
            %             matrix.
            %   I_mat_M - (array) Matrix used to zero out the observed
            %             measurements, retaining the original size of the
            %             matrix.
            %   nabla_ij - (array) Auxiliary variable.
            %
            % REVISION HISTORY:
            %   17/12/2022 - mcauchi1
            %       * Initial implementation
            %
            
            % Identity matrix of the size of number of biomarkers
            eye_Y = eye(size(y_tmp_i, 1));

            % Finding the rows that are observed/missing
            [row_O, col_O] = find(~isnan(y_tmp_i));
            [row_M, col_M] = find(isnan(y_tmp_i));

            % Creating the Omega matrices
            Omega_O = eye_Y(row_O, :);
            Omega_M = eye_Y(row_M, :);

            % Created the altered identity matrices
            I_mat_O = Omega_O' * Omega_O;
            I_mat_M = Omega_M' * Omega_M;

            % Setting value to zero if equals to NaN
            y_tmp_i(isnan(y_tmp_i)) = 0;

            % Arranging the vectors and matrices
            y_tmp_i = I_mat_O * y_tmp_i;
            C_tmp_i = I_mat_O * model_coef_est.C;
            V_tmp_i = I_mat_O * model_coef_est.V * I_mat_O + I_mat_M * model_coef_est.V * I_mat_M;

            % Calculation of the auxiliary variable
            V_tmp_OO = Omega_O * model_coef_est.V * Omega_O';
            if isempty(V_tmp_OO)
                neg_part = zeros(size(V_tmp_i));
            else
                neg_part = model_coef_est.V * Omega_O' * V_tmp_OO^-1 * Omega_O;
            end
            nabla_ij = eye(size(y_tmp_i,1)) - neg_part;
        end
        
        
        function [delta_ij, Delta_t_tmp] = pat_status_at_j(j_tmp, pat_ii, DeltaT)
            % FUNCTION NAME:
            %   pat_status_at_j
            %
            % DESCRIPTION:
            %   Identify the event indicator at time step j for patient i.
            %
            % INPUT:
            %   j_tmp - (double) Current iteration index.
            %   pat_ii - (struct) Contains observed data on the current
            %            patient.
            %   DeltaT - (double) Time step for the SSM.
            %
            % OUTPUT:
            %   delta_ij - (boolean) Event indicator at time step j for
            %              patient i.
            %   Delta_t_tmp - (double) Period between current time and next
            %                 time step/censoring/event (whichever comes
            %                 first).
            %
            % REVISION HISTORY:
            %   17/12/2022 - mcauchi1
            %       * Initial implementation
            %
            
            delta_ij = (pat_ii.m_i == j_tmp) & (pat_ii.delta_ev == 1);
            Delta_t_tmp = DeltaT;
            % If patient does not survive another DeltaT time
            if pat_ii.surv_time < j_tmp * DeltaT
                % Arrange Delta_t for this time step
                Delta_t_tmp = pat_ii.surv_time - (j_tmp - 1) * DeltaT;
            end
        end
        
        function f1out = f1x_ij(x_val, coeffs)
            % FUNCTION NAME:
            %   f1x_ij
            %
            % DESCRIPTION:
            %   Finds the negative exponent of the likelihood function of
            %   the current time step at the current value of x.
            %
            % INPUT:
            %   x_val - (array) Current hidden state values
            %   coeffs - (struct) Contains all required data to compute
            %            this function.
            %
            % OUTPUT:
            %   f1out - (double) Negative exponent of the likelihood value
            %           at the current time step.
            %
            % REVISION HISTORY:
            %   17/12/2022 - mcauchi1
            %       * Initial implementation
            %
            
            f1out = - coeffs.delta_ij * coeffs.a_s' * x_val ...
                     + coeffs.tau_ij * exp(coeffs.g_s' * coeffs.base_cov) * exp(coeffs.a_s' * x_val) ...
                     + (1/2) * (coeffs.y_ij - coeffs.C * x_val)' * coeffs.V^(-1) * (coeffs.y_ij - coeffs.C * x_val) ...
                     + (1/2) * (x_val - coeffs.pred_mu)' * coeffs.pred_V^(-1) * (x_val - coeffs.pred_mu);
        end
        
        function df1out = df1dx(x_val, coeffs)
            % FUNCTION NAME:
            %   df1dx
            %
            % DESCRIPTION:
            %   Finds the derivative of the negative exponent of the
            %   likelihood function of the current time step at the current
            %   value of x.
            %
            % INPUT:
            %   x_val - (array) Current hidden state values
            %   coeffs - (struct) Contains all required data to compute
            %            this function.
            %
            % OUTPUT:
            %   df1out - (double) Derivative of the negative exponent of
            %            the likelihood value at the current time step.
            %
            % REVISION HISTORY:
            %   17/12/2022 - mcauchi1
            %       * Initial implementation
            %
            
            df1out = - coeffs.delta_ij * coeffs.a_s ...
                      + coeffs.tau_ij * exp(coeffs.g_s' * coeffs.base_cov) * exp(coeffs.a_s' * x_val) * coeffs.a_s ...
                      - coeffs.C' * coeffs.V^(-1) * (coeffs.y_ij - coeffs.C * x_val) ...
                      + coeffs.pred_V^(-1) * (x_val - coeffs.pred_mu);
        end
        
        function d2f1out = d2f1dx2(x_val, coeffs)
            % FUNCTION NAME:
            %   d2f1dx2
            %
            % DESCRIPTION:
            %   Finds the second derivative of the negative exponent of the
            %   likelihood function of the current time step at the current
            %   value of x.
            %
            % INPUT:
            %   x_val - (array) Current hidden state values
            %   coeffs - (struct) Contains all required data to compute
            %            this function.
            %
            % OUTPUT:
            %   d2f1out - (double) Second derivative of the negative 
            %             exponent of the likelihood value at the current
            %             time step.
            %
            % REVISION HISTORY:
            %   17/12/2022 - mcauchi1
            %       * Initial implementation
            %
            
            d2f1out = coeffs.tau_ij * exp(coeffs.g_s' * coeffs.base_cov) ...
                            * exp(coeffs.a_s' * x_val) * (coeffs.a_s * coeffs.a_s') ...
                        + coeffs.C' * coeffs.V^(-1) * coeffs.C + coeffs.pred_V^(-1);
        end
        
        
        function [like_val] = like_fn_curr_step(x_ij, coeffs)
            % FUNCTION NAME:
            %   like_fn_curr_step
            %
            % DESCRIPTION:
            %   Evaluates the contribution of the current time step to the
            %   likelihood function (for patient i at time step j) using
            %   Laplace Approximation to evaluate the integral.
            %
            % INPUT:
            %   x_ij - (array) Current hidden state values
            %   coeffs - (struct) Contains all required data to compute
            %            this function.
            %
            % OUTPUT:
            %   like_val - (double) The likelihood contribution of patient
            %              i for time step j.
            %
            % REVISION HISTORY:
            %   17/12/2022 - mcauchi1
            %       * Initial implementation
            %
            
            % Find the dimension sizes of the hidden states and biomarker
            % observations
            dim_size.states = size(x_ij,1);
            dim_size.y = size(coeffs.V, 1);
            
            % 1) Find the value of x_{ij} that gives the minimum of f_1
            %    (through NR)
            x_NR = LSDSM_ALLFUNCS.Newton_Raphson(x_ij, 100, 1e-6, @LSDSM_ALLFUNCS.df1dx, ...
                                                        @LSDSM_ALLFUNCS.d2f1dx2, coeffs);
                                                    
            % 2) Evaluate the integral using Laplace approximation
            hess = LSDSM_ALLFUNCS.d2f1dx2(x_NR, coeffs);
            f1x_ij = LSDSM_ALLFUNCS.f1x_ij(x_NR, coeffs);
            int_val = (2*pi)^(dim_size.states/2) * det(hess)^(-1/2) * exp(-f1x_ij);
            
            % 3) Evaluate the final expression for the likelihood of the
            %    current observations given the past observations
            like_val = exp(coeffs.delta_ij * coeffs.g_s' * coeffs.base_cov) ...
                        * (2*pi)^(-(dim_size.states + dim_size.y)/2) ...
                        * det(coeffs.V)^(-1/2) * det(coeffs.pred_V)^(-1/2) * int_val;
            
        end


        function [mu_out, Sigma_out, log_likelihood_val] = ...
                Kalman_filter(pat_ii, model_coef_est, max_censor_time, mod_KF)
            % FUNCTION NAME:
            %   Kalman_filter
            %
            % DESCRIPTION:
            %   Executes the filtering part of the RTS smoother for a
            %   single patient. This function contains the required
            %   adaptations to make it suitable for LSDSM.
            %
            % INPUT:
            %   pat_ii - (struct) Contains all observed data of the current
            %            patient.
            %   model_coef_est - (struct) Model parameters.
            %   max_censor_time - (double) Maximum period of observation.
            %   mod_KF - (boolean) Indicator to use the survival data to
            %            correct the hidden state values.
            %
            % OUTPUT:
            %   mu_out - (array) Filtered mean for the entire observation
            %            trajectory.
            %   Sigma_out - (array) Filtered covariance for the entire 
            %               observation trajectory.
            %   log_likelihood_val - (double) Log likelihood contribution
            %                        of the current patient.
            %
            % REVISION HISTORY:
            %   17/12/2022 - mcauchi1
            %       * Initial implementation
            %
            
            % Initialise required arrays for filter
            % Mean in the forward recursion
            mu = zeros(size(pat_ii.mu_0,1), 1, max_censor_time);
            % Covariance in the forward recursion
            Sigma = zeros(size(pat_ii.mu_0,1), size(pat_ii.mu_0,1), max_censor_time);
            % Prediction covariance of forward recursion
            P = zeros(size(pat_ii.mu_0,1), size(pat_ii.mu_0,1), max_censor_time);
            % Kalman gain in the forward recursion
            K = zeros(size(pat_ii.mu_0,1), size(pat_ii.y,1), max_censor_time);
            % Mean output of the filter
            mu_out = zeros(size(pat_ii.mu_0,1), 1, max_censor_time);
            % Covariance output of the filter
            Sigma_out = zeros(size(pat_ii.mu_0,1), size(pat_ii.mu_0,1), max_censor_time);

            % We store the log due to the small number rounding errors
            log_likelihood_val = 0; % Initialise at 0

            for j=1:pat_ii.m_i % for every time step observed
                
                % boolean to check if patient died within the first time step
                [delta_ij, tau_ij] = LSDSM_ALLFUNCS.pat_status_at_j(j, pat_ii, model_coef_est.DeltaT);
                
                % Correct the observation vectors/matrices to account for
                % missing observations
                [y_tmp_i, C_tmp_i, V_tmp_i, Omega_O, Omega_M, I_mat_O, I_mat_M, nabla_ij] ...
                    = LSDSM_ALLFUNCS.missing_val_matrices(pat_ii.y(:,:,j), model_coef_est);
                
                if j==1 % Initialisation of filter
                    
                    % initialise filter
                    K(:,:,1) = pat_ii.W_0 * C_tmp_i' * (C_tmp_i * pat_ii.W_0 * C_tmp_i' + V_tmp_i)^-1;
                    mu(:,:,1) = pat_ii.mu_0 + K(:,:,1) * (y_tmp_i - C_tmp_i * pat_ii.mu_0);
                    Sigma(:,:,1) = (eye(size(pat_ii.mu_0,1)) - K(:,:,1) * C_tmp_i) * pat_ii.W_0;
                    
                    % Prepare the coefficients for the first contribution
                    % of the log likelihood
                    f1_fn_coef = struct('delta_ij', delta_ij, 'y_ij', y_tmp_i, ...
                                         'C', C_tmp_i, 'V', model_coef_est.V, ...
                                         'pred_mu', pat_ii.mu_0, 'pred_V', pat_ii.W_0, ...
                                         'base_cov', pat_ii.base_cov, 'g_s', model_coef_est.g_s, ...
                                         'a_s', model_coef_est.a_s, 'tau_ij', tau_ij);
                else % if j > 1
                    % Correction using the longitudinal data
                    [mu(:,:,j), Sigma(:,:,j), K(:,:,j), P(:,:,j-1)] = ...
                        LSDSM_ALLFUNCS.KF_single_step(mu_out(:,:,j-1), Sigma_out(:,:,j-1), ...
                                                      pat_ii.y(:,:,j), model_coef_est);

                    % Prepare the coefficients for the rest of the
                    % contributions of the log likelihood
                    f1_fn_coef = struct('delta_ij', delta_ij, 'y_ij', y_tmp_i, ...
                                         'C', C_tmp_i, 'V', model_coef_est.V, ...
                                         'pred_mu', model_coef_est.A * mu_out(:,:,j-1), ...
                                         'pred_V', P(:,:,j-1), ...
                                         'base_cov', pat_ii.base_cov, 'g_s', model_coef_est.g_s, ...
                                         'a_s', model_coef_est.a_s, 'tau_ij', tau_ij);
                end

                if mod_KF % Correction using survival data

                    Sigma_tmp_NR = Sigma(:,:,j);

                    % Find the value of x that maximises the posterior
                    % distribution
                    g_fn_coef = struct('delta_ij', delta_ij, 'mu_ij', mu(:,:,j), ...
                                   'Sigma_ij', Sigma_tmp_NR, 'base_cov', pat_ii.base_cov, ...
                                   'g_s', model_coef_est.g_s, 'a_s', model_coef_est.a_s, ...
                                   'tau_ij', tau_ij);
                    
                    % Use Newton Raphson's iterative method to approximate
                    % the posterior as a Gaussian distribution
                    x_NR = LSDSM_ALLFUNCS.Newton_Raphson(mu(:,:,j), 100, 1e-6, @LSDSM_ALLFUNCS.dgdx, ...
                                                            @LSDSM_ALLFUNCS.d2gdx2, g_fn_coef);

                    % Update mu_tilde and Sigma_tilde
                    mu_out(:,:,j) = x_NR;
                    
                    Sigma_tmp_NR_out = (tau_ij * exp(model_coef_est.g_s' * pat_ii.base_cov) ...
                                    * exp(model_coef_est.a_s' * x_NR) * (model_coef_est.a_s * model_coef_est.a_s') ...
                                        + Sigma_tmp_NR^-1)^-1;
                    Sigma_out(:,:,j) = Sigma_tmp_NR_out;
                    Sigma_out(:,:,j) = LSDSM_ALLFUNCS.ensure_sym_mat(Sigma_out(:,:,j));
                    
                else % If we are not correcting the states using survival data
                    mu_out(:,:,j) = mu(:,:,j);
                    Sigma_out(:,:,j) = Sigma(:,:,j);
                    Sigma_out(:,:,j) = LSDSM_ALLFUNCS.ensure_sym_mat(Sigma_out(:,:,j));
                end
                
                % Calculate the log likelihood contribution of the current
                % time step and add it to the overall contribution
                log_likelihood_val = log_likelihood_val + log(LSDSM_ALLFUNCS.like_fn_curr_step(mu_out(:,:,j), f1_fn_coef));
            end
        end

        
        function [mu_out, V_out, J_out] = Kalman_smoother(pat_ii, model_coef_est, max_censor_time)
            % FUNCTION NAME:
            %   Kalman_smoother
            %
            % DESCRIPTION:
            %   Executes the smoothing part of the RTS smoother for a
            %   single patient.
            %
            % INPUT:
            %   pat_ii - (struct) Contains all observed data of the current
            %            patient.
            %   model_coef_est - (struct) Model parameters.
            %   max_censor_time - (double) Maximum period of observation.
            %
            % OUTPUT:
            %   mu_out - (array) Smoother mean for the entire observation
            %            trajectory.
            %   Sigma_out - (array) Smoother covariance for the entire 
            %               observation trajectory.
            %   J_out - (array) Auxiliary matrices at every time point
            %
            % REVISION HISTORY:
            %   17/12/2022 - mcauchi1
            %       * Initial implementation
            %

            % Initialisation for arrays required for smoother operations
            mu_out = zeros(size(pat_ii.mu_tilde,1), 1, max_censor_time);
            V_out = zeros(size(pat_ii.mu_tilde,1), size(pat_ii.mu_tilde,1), max_censor_time);
            J_out = zeros(size(pat_ii.mu_tilde,1), size(pat_ii.mu_tilde,1), max_censor_time);

            % Last value w.r.t. time is obtained from the forward recursion,
            % since that is the P( x(N) | y(1:N) ), i.e. the posterior of x(N)
            % given all data available. Same goes for V(N).
            mu_out(:,:,pat_ii.m_i) = pat_ii.mu_tilde(:,:,pat_ii.m_i);
            V_out(:,:,pat_ii.m_i) = pat_ii.V_tilde(:,:,pat_ii.m_i);

            % Iterate through every time step to find the probabilities given
            % the entire observed data.
            for i=2:pat_ii.m_i
                k = pat_ii.m_i - i + 1; % N-1, N-2, ..., 2, 1

                % Find the smoother output of the current time step
                [mu_out(:,:,k), V_out(:,:,k), J_out(:,:,k)] = ...
                    LSDSM_ALLFUNCS.KS_single_step(mu_out(:,:,k+1), V_out(:,:,k+1), pat_ii.mu_tilde(:,:,k), ...
                                                    pat_ii.V_tilde(:,:,k), model_coef_est);

                V_out(:,:,k) = LSDSM_ALLFUNCS.ensure_sym_mat(V_out(:,:,k));
                % Add a small number to the diagonals of V to avoid
                % singular matrices since we require the inverse of V.
                V_out(:,:,k) = V_out(:,:,k) + 1e-9 * eye(size(V_out,1));
            end
        end

        
        function E = compute_E_fns(pat_ii, model_coef_est, max_censor_time)
            % FUNCTION NAME:
            %   compute_E_fns
            %
            % DESCRIPTION:
            %   Finds the required expectations for the EM algorithm for
            %   the current patient.
            %
            % INPUT:
            %   pat_ii - (struct) Contains all observed data of the current
            %            patient.
            %   model_coef_est - (struct) Model parameters.
            %   max_censor_time - (double) Maximum period of observation.
            %
            % OUTPUT:
            %   E - (struct) Expectations for the current patient.
            %
            % REVISION HISTORY:
            %   17/12/2022 - mcauchi1
            %       * Initial implementation
            %
            
            % Find the indices of "dynamic" states
            [idx_present_r, idx_present_c] = find(model_coef_est.G_mat ~= 0);

            % Placeholders for the expected values for the current patient
            E.xn = zeros(size(pat_ii.mu_hat,1), 1, max_censor_time);
            E.xn_xnneg1 = zeros(size(pat_ii.mu_hat,1), size(pat_ii.mu_hat,1), max_censor_time);
            E.xn_xn = zeros(size(pat_ii.mu_hat,1), size(pat_ii.mu_hat,1), max_censor_time);
            E.barxn_barxn = zeros(length(idx_present_r), length(idx_present_r), max_censor_time);
            E.barxn_xnneg1 = zeros(length(idx_present_r), size(pat_ii.mu_hat,1), max_censor_time);
            E.yn = zeros(size(pat_ii.y,1), 1, max_censor_time);
            E.yn_yn = zeros(size(pat_ii.y,1), size(pat_ii.y,1), max_censor_time);
            E.yn_xn = zeros(size(pat_ii.y,1), size(pat_ii.mu_hat,1), max_censor_time);

            % The expectation of x is the smoothed distribution
            E.xn = pat_ii.mu_hat;

            % Extract the dynamic states
            mu_bar_tmp = pat_ii.mu_hat(idx_present_r,:,:);
            
            M_tmp = pagemtimes(pat_ii.V_hat(:,:,2:end), 'none', pat_ii.J_hat(:,:,1:end-1), 'transpose');
            E.xn_xnneg1(:,:,2:end) = M_tmp + pagemtimes(pat_ii.mu_hat(:,:,2:end), 'none', ...
                                                                pat_ii.mu_hat(:,:,1:end-1), 'transpose');
            
            E.barxn_xnneg1(:,:,2:end) = M_tmp(idx_present_r,:,:) + ...
                                            pagemtimes(mu_bar_tmp(:,:,2:end), 'none', ...
                                                       pat_ii.mu_hat(:,:,1:end-1), 'transpose');
            
            E.xn_xn = pat_ii.V_hat + pagemtimes(pat_ii.mu_hat, 'none', pat_ii.mu_hat, 'transpose');
            E.barxn_barxn = pat_ii.V_hat(idx_present_r, idx_present_r, :) + ...
                    pagemtimes(mu_bar_tmp, 'none', mu_bar_tmp, 'transpose');
            
            for i=1:pat_ii.m_i % for every time step
                E.xn_xn(:,:,i) = LSDSM_ALLFUNCS.ensure_sym_mat(E.xn_xn(:,:,i));
                E.barxn_barxn(:,:,i) = LSDSM_ALLFUNCS.ensure_sym_mat(E.barxn_barxn(:,:,i));

                % Expectations involving y
                % Modify the observation vectors and matrices to account
                % for the missing measurements
                [y_tmp_i, C_tmp_i, V_tmp_i, Omega_O, Omega_M, I_mat_O, ...
                        I_mat_M, nabla_ij] = LSDSM_ALLFUNCS.missing_val_matrices(pat_ii.y(:,:,i), model_coef_est);

                E.yn(:,:,i) = y_tmp_i - nabla_ij * (y_tmp_i - model_coef_est.C * E.xn(:,:,i));

                E.yn_yn(:,:,i) = I_mat_M * (nabla_ij * model_coef_est.V + ...
                    nabla_ij * model_coef_est.C * pat_ii.V_hat(:,:,i) * model_coef_est.C' * nabla_ij') * I_mat_M + ...
                    E.yn(:,:,i) * E.yn(:,:,i)';

                E.yn_yn(:,:,i) = LSDSM_ALLFUNCS.ensure_sym_mat(E.yn_yn(:,:,i));

                E.yn_xn(:,:,i) = nabla_ij * model_coef_est.C * pat_ii.V_hat(:,:,i) + E.yn(:,:,i) * E.xn(:,:,i)';
            end
        end

        
        function sym_mat = ensure_sym_mat(mat_tmp)
            % FUNCTION NAME:
            %   ensure_sym_mat
            %
            % DESCRIPTION:
            %   Ensures symmetry in matrix by finding the average of the
            %   matrix and its transpose.
            %
            % INPUT:
            %   mat_tmp - (array) Matrix to ensure symmetry.
            %
            % OUTPUT:
            %   sym_mat - (array) Symmetric matrix output.
            %
            % REVISION HISTORY:
            %   17/12/2022 - mcauchi1
            %       * Initial implementation
            %
            sym_mat = (mat_tmp + mat_tmp') / 2;
        end
        
        
        function [E_sums] = sum_E_fns(E_sums, pat_ii)
            % FUNCTION NAME:
            %   sum_E_fns
            %
            % DESCRIPTION:
            %   This function continues to add towards the total sum of
            %   expectations that are required in the M step of the EM
            %   algorithm.
            %
            % INPUT:
            %   E_sums - (struct) Contains the summations of the 
            %            expectations all the previously evaluated
            %            patients.
            %   pat_ii - (struct) Contains all observed data of the current
            %            patient.
            %
            % OUTPUT:
            %   E_sums - (struct) New summations of all required
            %            expectations, including the current patient's
            %            information.
            %
            % REVISION HISTORY:
            %   17/12/2022 - mcauchi1
            %       * Initial implementation
            %

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


        function [model_new_coeffs] = M_step(pat_data, E_sums, RTS_arrs, model_coef_est, controls)
            % FUNCTION NAME:
            %   M_step
            %
            % DESCRIPTION:
            %   Finds the updated parameters of LSDSM keeping the hidden
            %   states fixed. This is the M step of the EM algorithm.
            %
            % INPUT:
            %   pat_data - (map) Contains the observed data for all
            %              patients.
            %   E_sums - (struct) Contains the summations of expectations 
            %            for all patients.
            %   RTS_arrs - (struct) Contains the output arrays obtained
            %              from the RTS smoother.
            %   model_coef_est - (struct) Current model parameters.
            %   controls - (struct) EM algorithm controls, including
            %              information on which parameters to keep fixed.
            %
            % OUTPUT:
            %   model_new_coeffs - (struct) Updated model parameters.
            %
            % REVISION HISTORY:
            %   17/12/2022 - mcauchi1
            %       * Initial implementation
            %
            
            % Number of patients
            num_pats = double(pat_data.Count);
            
            % Total number of observations across all patients
            N_totpat = 0;
            for ii=1:num_pats
                N_totpat = N_totpat + pat_data(ii).m_i;
            end

            % Find the indices for the dynamic states
            [idx_present_r, idx_present_c] = find(model_coef_est.G_mat ~= 0);

            %%% State Space Parameters %%%
            % Update A
            if isnan(controls.fixed_params.A)
                % Special treatment due to canonical form
                % Also, note that we use the sum from 3 instead of from 2.
                % This is because it was found to be more stable for the EM
                % algorithm.
                A_bar_new = E_sums.barxn_xnneg1_from3 * E_sums.xn_xn_from2_tillNneg1^-1;
                A_new = model_coef_est.A;
                A_new(idx_present_r,:) = A_bar_new;
            else % if A is fixed
                A_new = controls.fixed_params.A;
                A_bar_new = A_new(idx_present_r,:);
            end

            % Update C
            if isnan(controls.fixed_params.C)
                C_new = E_sums.yn_xn * E_sums.xn_xn^-1;
            else % if C is fixed
                C_new = controls.fixed_params.C;
            end
            
            % Update W
            if isnan(controls.fixed_params.W)
                W_new = (N_totpat - num_pats)^-1 * (E_sums.barxn_barxn_from2 - A_bar_new * E_sums.barxn_xnneg1_from2' ...
                         - E_sums.barxn_xnneg1_from2 * A_bar_new' + A_bar_new * E_sums.xn_xn_tillNneg1 * A_bar_new');
                W_new = LSDSM_ALLFUNCS.ensure_sym_mat(W_new);
            else % if W is fixed
                W_new = controls.fixed_params.W;
            end

            % Update V
            if isnan(controls.fixed_params.V)
                V_new = (N_totpat)^-1 * (E_sums.yn_yn - C_new * E_sums.yn_xn' ...
                                            - E_sums.yn_xn * C_new' + C_new * E_sums.xn_xn * C_new');
                V_new = LSDSM_ALLFUNCS.ensure_sym_mat(V_new);
            else % if V is fixed
                V_new = controls.fixed_params.V;
            end
            
            % Update mu0 and W0
            if controls.update_pop_mu
                if isnan(controls.fixed_params.mu_0)
                    mu_0new = E_sums.x0 / num_pats;
                else % if mu0 is fixed
                    mu_0new = controls.fixed_params.mu_0;
                end
                
                if isnan(controls.fixed_params.W_0)
                    W_0new = E_sums.x0_x0 / num_pats - mu_0new * mu_0new'; 
                    W_0new = LSDSM_ALLFUNCS.ensure_sym_mat(W_0new);
                else % if W0 is fixed
                    W_0new = controls.fixed_params.W_0;
                end
                
            else
                if isnan(controls.fixed_params.mu_0)
                    mu_0new = reshape(RTS_arrs.mu_hat(:,:,1,:), [size(model_coef_est.A,1), 1, num_pats]);
                else % if mu0 is fixed
                    mu_0new = controls.fixed_params.mu_0;
                end
                if isnan(controls.fixed_params.W_0)
                    W_0new = reshape(RTS_arrs.V_hat(:,:,1,:), ...
                                     [size(model_coef_est.A,1), size(model_coef_est.A,1), num_pats]);
                else % if W0 is fixed
                    W_0new = controls.fixed_params.W_0;
                end
            end

            
            %%% Survival Parameters %%%
            dG_data = struct('pat_data', pat_data, 'model_coef_est', model_coef_est, ...
                             'RTS_arrs', RTS_arrs, 'controls', controls);
            
            % Use Newton Raphson to identify the survival parameters that
            % maximise the expectation of the complete data log likelihood
            g_a_s_new = LSDSM_ALLFUNCS.Newton_Raphson([model_coef_est.g_s; model_coef_est.a_s], 100, 1e-6, ...
                                                        @LSDSM_ALLFUNCS.dGdx, ...
                                                        @LSDSM_ALLFUNCS.d2Gdx2, dG_data);

            g_s_new = g_a_s_new(1:size(model_coef_est.g_s,1), 1);
            if not(isnan(controls.fixed_params.g_s)) % if g_s is fixed
                g_s_new = controls.fixed_params.g_s;
            end

            a_s_new = g_a_s_new(size(model_coef_est.g_s,1)+1:end, 1);
            if not(isnan(controls.fixed_params.a_s)) % if a_s is fixed
                a_s_new = controls.fixed_params.a_s;
            end
            
            % Store the updated parameters
            model_new_coeffs = struct('A', A_new, 'C', C_new, ...
                                      'W', W_new, 'V', V_new, ...
                                      'g_s', g_s_new, 'a_s', a_s_new, ...
                                      'DeltaT', model_coef_est.DeltaT, 'G_mat', model_coef_est.G_mat, ...
                                      'mu_0', mu_0new, 'W_0', W_0new);
        end

        
        function [sse_value, mse_value, rmse_value] = rmse_fn(signal1, signal2, time_dim)
            % FUNCTION NAME:
            %   rmse_fn
            %
            % DESCRIPTION:
            %   Finds the sum of squares error, mean square error, and root
            %   mean square error between two signals.
            %
            % INPUT:
            %   signal1 - (array) The first signal.
            %   signal2 - (array) The second signal.
            %   time_dim - (double) The dimension where time is located
            %              within the signals.
            %
            % OUTPUT:
            %   sse_value - (double) Sum of squares error.
            %   mse_value - (double) Mean square error.
            %   rmse_value - (double) Root mean square error.
            %
            % REVISION HISTORY:
            %   17/12/2022 - mcauchi1
            %       * Initial implementation
            %
            
            % Find the difference between the signals
            error_signal = signal1 - signal2;
            
            % Evaluate the sum of squares error
            sse_value = zeros(size(signal1, 1), 1);
            for j=1:size(error_signal, time_dim)
                if time_dim == 2 % If time iterates in the second dimension of the array
                    sse_value = sse_value + error_signal(:,j).^2;
                elseif time_dim == 3 % If time iterates in the third dimension of the array
                    sse_value = sse_value + error_signal(:,:,j).^2;
                end
            end
            % Evaluate the mean square error
            mse_value = sse_value / length(error_signal);
            % Evaluate the root mean square error
            rmse_value = sqrt(mse_value);
        end
        
    end
end