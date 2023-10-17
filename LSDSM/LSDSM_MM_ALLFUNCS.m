classdef LSDSM_MM_ALLFUNCS
    % Author:        Mark Cauchi
    % Created on:    17/12/2022
    % Last Modified: 13/10/2023
    % Description:   All functions required to execute training and
    %                predictions using Linear State Space Dynamic Survival
    %                Model (LSDSM) with the Mixture Model (MM) extension.
    %                This model jointly handles longitudinal and survival
    %                data by making both dependent on some latent variables
    %                (state trajectories and classes). The sub-processes
    %                are defined as: 
    %                - Longitudinal Sub-process: Linear Gaussian State 
    %                  Space Model 
    %                - Survival Sub-process: Proportional Hazards Model 
    %
    %                For more information, reader is referred to the paper
    %                titled "Individualised Survival Predictions using
    %                State Space Model with Longitudinal and Survival
    %                Data".
    methods(Static)
        
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
            %   M_mat - (matrix/table) This is the matrix obtained from 
            %           MATLAB's native read csv function. Data is 
            %           expressed in the long format.
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
            %   10/09/2023 - mcauchi1
            %       * Included support for table in place of the matrix
            %
            
            if istable(M_mat)
                
                % Convert table to cell matrix
                M_mat_cell = table2cell(M_mat);
                
                % Normalisation constants if we wish to normalise the biomarkers
                normalise_const = max(cellfun(@(x) x, M_mat_cell(:,csv_controls.bio_long_col_no:end)));

                % Find the maximum time from the training data
                max_t = ceil(max(cellfun(@(x) x, M_mat_cell(:,2)))); % ceil(max(surv time))
                % Create an array for time in csv_controls.Delta steps till max_t is reached
                no_of_months = linspace(0,max_t*csv_controls.t_multiplier, max_t*csv_controls.t_multiplier/csv_controls.Delta+1);

                % Stores observed patient data - struct for every patient,
                % stored within a map
                data_observed = containers.Map('KeyType', 'int32', 'ValueType', 'any'); 

                % Find the number of patients in the training set
                unique_pat_id = unique(M_mat_cell(:,1));
                no_of_pat = length(unique_pat_id);

                for i=1:no_of_pat % for every patient
                    % Filling in patient information
                    pat_ii = struct();
                    pat_ii.id = unique_pat_id(i);
                    pat_ii.surv_time = 0;
                    pat_ii.delta_ev = 0;
                    pat_ii.m_i = 0;
                    pat_ii.base_cov = zeros(dim_size.base_cov,1);
                    pat_ii.class_cov = zeros(dim_size.class_cov,1);
                    % NaN was used for y as these will be treated as
                    % missing values within the state space model if no
                    % observations were made at that time period.
                    pat_ii.y = NaN * zeros(dim_size.y, 1, length(no_of_months));
                    % Extract the part of matrix that contains only the
                    % current patient's info - curr_ind contains the row
                    % indices of the matrix that correspond to the current
                    % patient
                    curr_ind = find(strcmp(M_mat_cell(:,1), pat_ii.id));
                    % Find the iteration number for every observation made
                    % for this patient Note: For training data, we use
                    % round function to go to the closest time binning. In
                    % testing data, we shall use ceil for those time points
                    % that arrive after the landmark of interest. This is
                    % because we do not want to use future data after the
                    % landmark. E.g. assume landmark is at 12 time points.
                    % If there is an observation at 12.4, then using the
                    % round function, this will go to 12 time points, and
                    % hence, future data is used. With ceil, this
                    % measurement will not be utilised.
                    if strcmpi(csv_controls.train_test, 'train') % if training data set
                        iter_no = round(cell2mat(M_mat_cell(curr_ind, 4)) * csv_controls.t_multiplier/csv_controls.Delta);
                    else % if testing data set
                        iter_no = cell2mat(M_mat_cell(curr_ind, 4)) * csv_controls.t_multiplier/csv_controls.Delta;
                        iter_no(iter_no<=csv_controls.landmark_idx) = round(iter_no(iter_no<=csv_controls.landmark_idx));
                        iter_no(iter_no>csv_controls.landmark_idx) = ceil(iter_no(iter_no>csv_controls.landmark_idx));
                    end
                    
                    for j=1:length(curr_ind)
                        % Store the patient's longitudinal biomarkers in a 3d
                        % array (num_obs, 1, num_time_steps).
                        if csv_controls.norm_bool
                            pat_ii.y(:,1,iter_no(j)+1) = ...
                                cell2mat(M_mat_cell(curr_ind(j), csv_controls.bio_long_col_no:csv_controls.bio_long_col_no+dim_size.y-1)) ...
                                                         ./ normalise_const(1:dim_size.y);
                        else
                            pat_ii.y(:,1,iter_no(j)+1) = ...
                                cell2mat(M_mat_cell(curr_ind(j), csv_controls.bio_long_col_no:csv_controls.bio_long_col_no+dim_size.y-1));
                        end
                    end
                    % Also note: utilising the above method means that some
                    % observations are not utilised, since they happen to fall
                    % at the same time bin.

                    % Class covariates
                    if dim_size.class_cov == 1
                        pat_ii.class_cov = 1;
                    else
                        pat_ii.class_cov(1) = 1;
                        % for every class covariate we have
                        for j=2:dim_size.class_cov
                            pat_ii.class_cov(j) = normrnd(curr_dist.mean, curr_dist.std);
                        end
                    end

                    % If we want to visualise some plots
                    if csv_controls.allow_plots && i <= csv_controls.no_of_plots
                        figure;
                        to_plot = M_mat_cell(curr_ind, csv_controls.bio_long_col_no);
                        if csv_controls.norm_bool
                            to_plot = to_plot / normalise_const(1);
                        end
                        scatter(cell2mat(M_mat_cell(curr_ind, 4)), cell2mat(to_plot));
                        xlabel('Time (years)')
                        ylabel('y');
                        hold on;

                        if cell2mat(M_mat_cell(curr_ind(j), 3)) == 0 % if patient is censored
                            xline(cell2mat(M_mat_cell(curr_ind(1), 2)), 'g', 'LineWidth', 2);
                            legend('y', 'Censored');
                        else % if patient experiences event
                            xline(cell2mat(M_mat_cell(curr_ind(1), 2)), 'r', 'LineWidth', 2);
                            legend('y', 'Event');
                        end

                        xlim([0,max_t]);
                        if csv_controls.norm_bool
                            ylim([0,1]);
                        else
                            ylim([min(cell2mat(M_mat_cell(:,csv_controls.bio_long_col_no))) normalise_const(1)]);
                        end
                    end

                    % Store survival information
                    pat_ii.delta_ev = cell2mat(M_mat_cell(curr_ind(j), 3));
                    pat_ii.surv_time = cell2mat(M_mat_cell(curr_ind(1), 2))*csv_controls.t_multiplier;

                    % If the survival time is greater than the hard
                    % thresholding of the censor time, then the patient is
                    % censored at the threshold time
                    if pat_ii.surv_time > csv_controls.censor_time
                        pat_ii.delta_ev = 0;
                        pat_ii.surv_time = csv_controls.censor_time;
                    end

                    % Number of time periods the patient is observed for
                    % as it is, if the last measurement time coincides with
                    % the survival time, it will be lost (this makes sense
                    % with our model since the current measurement affects
                    % the next time step for survival).
                    pat_ii.m_i = floor(pat_ii.surv_time/csv_controls.Delta)+1;

                    % Store the baseline covariates
                    pat_ii.base_cov(1,1) = 1; % intercept
                    % Other baseline covariates
                    if dim_size.base_cov > 1
                        pat_ii.base_cov(2:end,1) = ...
                            cell2mat(M_mat_cell(curr_ind(j), csv_controls.base_cov_col_no:csv_controls.base_cov_col_no+dim_size.base_cov-2));
                    end

                    % Class covariates
                    pat_ii.class_cov(1) = 1;
                    if dim_size.class_cov > 1
                        % Other class covariates
                        pat_ii.class_cov(2:end) = ...
                            cell2mat(M_mat_cell(curr_ind(j), csv_controls.base_cov_col_no:csv_controls.base_cov_col_no+dim_size.class_cov-2)); 
                    end
                    % Store patient data in map
                    data_observed(i) = pat_ii;
                end
                
            else % if M_mat is a matrix
                % Normalisation constants if we wish to normalise the biomarkers
                normalise_const = max(M_mat(:,csv_controls.bio_long_col_no:end));

                % Find the maximum time from the training data
                max_t = ceil(max(M_mat(:,2))); % ceil(max(surv time))
                % Create an array for time in csv_controls.Delta steps till
                % max_t is reached
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
                    pat_ii.class_cov = zeros(dim_size.class_cov,1);
                    % NaN was used for y as these will be treated as
                    % missing values within the state space model if no
                    % observations were made at that time period.
                    pat_ii.y = NaN * zeros(dim_size.y, 1, length(no_of_months));
                    % Extract the part of matrix that contains only the
                    % current patient's info - curr_ind contains the row
                    % indices of the matrix that correspond to the current
                    % patient
                    curr_ind = find(M_mat(:,1) == pat_ii.id);
                    % Find the iteration number for every observation made
                    % for this patient Note: For training data, we use
                    % round function to go to the closest time binning. In
                    % testing data, we shall use ceil for those time points
                    % that arrive after the landmark of interest. This is
                    % because we do not want to use future data after the
                    % landmark. E.g. assume landmark is at 12 time points.
                    % If there is an observation at 12.4, then using the
                    % round function, this will go to 12 time points, and
                    % hence, future data is used. With ceil, this
                    % measurement will not be utilised.
                    if strcmpi(csv_controls.train_test, 'train') % if training data set
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
                            pat_ii.y(:,1,iter_no(j)+1) = ...
                                M_mat(curr_ind(j), csv_controls.bio_long_col_no:csv_controls.bio_long_col_no+dim_size.y-1) ...
                                                         ./ normalise_const(1:dim_size.y);
                        else
                            pat_ii.y(:,1,iter_no(j)+1) = ...
                                M_mat(curr_ind(j), csv_controls.bio_long_col_no:csv_controls.bio_long_col_no+dim_size.y-1);
                        end
                    end
                    % Also note: utilising the above method means that some
                    % observations are not utilised, since they happen to fall
                    % at the same time bin.

                    % Class covariates
                    if dim_size.class_cov == 1
                        pat_ii.class_cov = 1;
                    else
                        pat_ii.class_cov(1) = 1;
                        % for every class covariate we have
                        for j=2:dim_size.class_cov
                            pat_ii.class_cov(j) = normrnd(curr_dist.mean, curr_dist.std);
                        end
                    end

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

                    % Class covariates
                    pat_ii.class_cov(1) = 1;
                    if dim_size.class_cov > 1
                        % Other class covariates
                        pat_ii.class_cov(2:end) = ...
                            M_mat(curr_ind(j), csv_controls.base_cov_col_no:csv_controls.base_cov_col_no+dim_size.class_cov-2); 
                    end
                    % Store patient data in map
                    data_observed(i) = pat_ii;
                end
            
            end
        end
        
        
        function [data_observed] = read_from_csv_risk_score(M_mat, dim_size, csv_controls, range_int, landmark, horiz)
            % FUNCTION NAME:
            %   read_from_csv_risk_score
            %
            % DESCRIPTION:
            %   Function to convert csv to required data structure. data is
            %   assumed to be in long format, i.e. one row for every
            %   longitudinal measurement.
            %   Different from read_from_csv since it sorts data to have
            %   same information as the risk score would
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
            %   M_mat - (matrix/table) This is the matrix obtained from 
            %           MATLAB's native read csv function. Data is 
            %           expressed in the long format.
            %   dim_size - (struct) Expresses the dimension sizes of the
            %              number of hidden states and dynamic states, 
            %              number of observations, and number of baseline 
            %              covariates.
            %   csv_controls - (struct) Contains controls such as the
            %                  column start for the baseline covariates and
            %                  longitudinal biomarkers, the time step for
            %                  the state space model, etc.
            %   range_int - (array) The range of interest for longitudinal
            %               measurements. Includes only patients that have
            %               measurements within this range.
            %   landmark - (double) Landmark where survival predictions are
            %              made.
            %   horizon - (double) The length of the prediction window.
            %
            % OUTPUT:
            %   data_observed - (map) Contains a struct for every patient,
            %                   each retaining the patient's ID, survival
            %                   time, event boolean, number of measurements
            %                   for the state space model, the baseline
            %                   covariates and the longitudinal biomarkers.
            %
            % REVISION HISTORY:
            %   10/09/2023 - mcauchi1
            %       * Initial implementation
            %
            
            if istable(M_mat) % if it is a table
                
                % convert to cell arrays
                M_mat_cell = table2cell(M_mat);
                
                % Normalisation constants if we wish to normalise the biomarkers
                normalise_const = max(cellfun(@(x) x, M_mat_cell(:,csv_controls.bio_long_col_no:end)));

                % Find the maximum time from the training data
                max_t = ceil(max(cellfun(@(x) x, M_mat_cell(:,2)))); % ceil(max(surv time))
                % Create an array for time in csv_controls.Delta steps till max_t is reached
                no_of_months = linspace(0,max_t*csv_controls.t_multiplier, max_t*csv_controls.t_multiplier/csv_controls.Delta+1);

                % Stores observed patient data - struct for every patient,
                % stored within a map
                data_observed = containers.Map('KeyType', 'int32', 'ValueType', 'any'); 

                % Find the number of patients in the training set
                unique_pat_id = unique(M_mat_cell(:,1));
                no_of_pat = length(unique_pat_id);
                
                red_i = 0;

                % Filling in patient information
                for i=1:no_of_pat
                    pat_ii = struct();
                    pat_ii.id = unique_pat_id(i);
                    pat_ii.surv_time = 0;
                    pat_ii.delta_ev = 0;
                    pat_ii.m_i = 0;
                    pat_ii.base_cov = zeros(dim_size.base_cov,1);
                    pat_ii.class_cov = zeros(dim_size.class_cov,1);
                    % NaN was used for y as these will be treated as
                    % missing values within the state space model if no
                    % observations were made at that time period.
                    pat_ii.y = NaN * zeros(dim_size.y, 1, length(no_of_months));
                    % Extract the part of matrix that contains only the
                    % current patient's info - curr_ind contains the row
                    % indices of the matrix that correspond to the current
                    % patient
                    curr_ind = find(strcmp(M_mat_cell(:,1), pat_ii.id));
                    % Find the iteration number for every observation made
                    % for this patient Note: For training data, we use
                    % round function to go to the closest time binning. In
                    % testing data, we shall use ceil for those time points
                    % that arrive after the landmark of interest. This is
                    % because we do not want to use future data after the
                    % landmark. E.g. assume landmark is at 12 time points.
                    % If there is an observation at 12.4, then using the
                    % round function, this will go to 12 time points, and
                    % hence, future data is used. With ceil, this
                    % measurement will not be utilised.
                    
                    % Store survival information
                    pat_ii.delta_ev = cell2mat(M_mat_cell(curr_ind(1), 3));
                    pat_ii.surv_time = cell2mat(M_mat_cell(curr_ind(1), 2))*csv_controls.t_multiplier;

                    % If the survival time is greater than the hard
                    % thresholding of the censor time, then the patient is
                    % censored at the threshold time
                    if pat_ii.surv_time > csv_controls.censor_time
                        pat_ii.delta_ev = 0;
                        pat_ii.surv_time = csv_controls.censor_time;
                    end
                    
                    % timings of measurements for current patient
                    tij_arr = cell2mat(M_mat_cell(curr_ind, 4));
                    % include patient if they have a measurement within the
                    % range of interest
                    include_pat = any(tij_arr >= range_int(1) & tij_arr <= range_int(2));
                    
                    if include_pat
                        % include patient if they are not censored before
                        % the horizon period ends since risk score will not
                        % know if they survived the entire period
                        include_pat = include_pat & not(pat_ii.surv_time < (landmark + horiz) & pat_ii.delta_ev == 0);
                    end
                    
                    if include_pat % if patient is to be included
                        
                        red_i = red_i + 1; % increase patient iterative
                    
                        if strcmpi(csv_controls.train_test, 'train') % if training data set
                            % round to the nearest time step
                            iter_no = round(tij_arr * csv_controls.t_multiplier/csv_controls.Delta);
                        else % if testing data set
                            % future measurements cannot be rounded down to
                            % the nearest time step
                            iter_no = tij_arr * csv_controls.t_multiplier/csv_controls.Delta;
                            iter_no(iter_no<=csv_controls.landmark_idx) = round(iter_no(iter_no<=csv_controls.landmark_idx));
                            iter_no(iter_no>csv_controls.landmark_idx) = ceil(iter_no(iter_no>csv_controls.landmark_idx));
                        end
                        for j=1:length(curr_ind)
                            % Store the patient's longitudinal biomarkers in a 3d
                            % array (num_obs, 1, num_time_steps).
                            if csv_controls.norm_bool
                                pat_ii.y(:,1,iter_no(j)+1) = ...
                                    cell2mat(M_mat_cell(curr_ind(j), csv_controls.bio_long_col_no:csv_controls.bio_long_col_no+dim_size.y-1)) ...
                                                             ./ normalise_const(1:dim_size.y);
                            else
                                pat_ii.y(:,1,iter_no(j)+1) = ...
                                    cell2mat(M_mat_cell(curr_ind(j), csv_controls.bio_long_col_no:csv_controls.bio_long_col_no+dim_size.y-1));
                            end
                        end
                        % Also note: utilising the above method means that some
                        % observations are not utilised, since they happen to fall
                        % at the same time bin.

                        % Class covariates
                        if dim_size.class_cov == 1
                            pat_ii.class_cov = 1;
                        else
                            pat_ii.class_cov(1) = 1;
                            % for every class covariate we have
                            for j=2:dim_size.class_cov
                                pat_ii.class_cov(j) = normrnd(curr_dist.mean, curr_dist.std);
                            end
                        end

                        % If we want to visualise some plots
                        if csv_controls.allow_plots && i <= csv_controls.no_of_plots
                            figure;
                            to_plot = M_mat_cell(curr_ind, csv_controls.bio_long_col_no);
                            if csv_controls.norm_bool
                                to_plot = to_plot / normalise_const(1);
                            end
                            scatter(cell2mat(M_mat_cell(curr_ind, 4)), cell2mat(to_plot));
                            xlabel('Time (years)')
                            ylabel('y');
                            hold on;

                            if cell2mat(M_mat_cell(curr_ind(j), 3)) == 0 % if patient is censored
                                xline(cell2mat(M_mat_cell(curr_ind(1), 2)), 'g', 'LineWidth', 2);
                                legend('y', 'Censored');
                            else % if patient experiences event
                                xline(cell2mat(M_mat_cell(curr_ind(1), 2)), 'r', 'LineWidth', 2);
                                legend('y', 'Event');
                            end

                            xlim([0,max_t]);
                            if csv_controls.norm_bool
                                ylim([0,1]);
                            else
                                ylim([min(cell2mat(M_mat_cell(:,csv_controls.bio_long_col_no))) normalise_const(1)]);
                            end
                        end

                        % Number of time periods the patient is observed
                        % for as it is, if the last measurement time
                        % coincides with the survival time, it will be lost
                        % (this makes sense with our model since the
                        % current measurement affects the next time step
                        % for survival).
                        pat_ii.m_i = floor(pat_ii.surv_time/csv_controls.Delta)+1;

                        % Store the baseline covariates
                        pat_ii.base_cov(1,1) = 1; % intercept
                        % Other baseline covariates
                        pat_ii.base_cov(2:end,1) = ...
                            cell2mat(M_mat_cell(curr_ind(j), csv_controls.base_cov_col_no:csv_controls.base_cov_col_no+dim_size.base_cov-2)); 

                        % Class covariates
                        pat_ii.class_cov(1) = 1;
                        if dim_size.class_cov > 1
                            % Other class covariates
                            pat_ii.class_cov(2:end) = ...
                                cell2mat(M_mat_cell(curr_ind(j), csv_controls.base_cov_col_no:csv_controls.base_cov_col_no+dim_size.class_cov-2)); 
                        end
                        % Store patient data in map
                        data_observed(red_i) = pat_ii;
                    end
                end
                
            else % if M_mat is a matrix
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
                    pat_ii.class_cov = zeros(dim_size.class_cov,1);
                    % NaN was used for y as these will be treated as
                    % missing values within the state space model if no
                    % observations were made at that time period.
                    pat_ii.y = NaN * zeros(dim_size.y, 1, length(no_of_months));
                    % Extract the part of matrix that contains only the
                    % current patient's info - curr_ind contains the row
                    % indices of the matrix that correspond to the current
                    % patient
                    curr_ind = find(M_mat(:,1) == pat_ii.id);
                    % Find the iteration number for every observation made
                    % for this patient Note: For training data, we use
                    % round function to go to the closest time binning. In
                    % testing data, we shall use ceil for those time points
                    % that arrive after the landmark of interest. This is
                    % because we do not want to use future data after the
                    % landmark. E.g. assume landmark is at 12 time points.
                    % If there is an observation at 12.4, then using the
                    % round function, this will go to 12 time points, and
                    % hence, future data is used. With ceil, this
                    % measurement will not be utilised.
                    if strcmpi(csv_controls.train_test, 'train') % if training data set
                        iter_no = round(M_mat(curr_ind, 4) * csv_controls.t_multiplier/csv_controls.Delta);
                    else % if testing data set
                        iter_no = M_mat(curr_ind, 4) * csv_controls.t_multiplier/csv_controls.Delta;
                        iter_no(iter_no<=csv_controls.landmark_idx) = round(iter_no(iter_no<=csv_controls.landmark_idx));
                        iter_no(iter_no>csv_controls.landmark_idx) = ceil(iter_no(iter_no>csv_controls.landmark_idx));
                    end
                    for j=1:length(curr_ind)
                        % Store the patient's longitudinal biomarkers in a 
                        % 3d array (num_obs, 1, num_time_steps).
                        if csv_controls.norm_bool
                            pat_ii.y(:,1,iter_no(j)+1) = ...
                                M_mat(curr_ind(j), csv_controls.bio_long_col_no:csv_controls.bio_long_col_no+dim_size.y-1) ...
                                                         ./ normalise_const(1:dim_size.y);
                        else
                            pat_ii.y(:,1,iter_no(j)+1) = ...
                                M_mat(curr_ind(j), csv_controls.bio_long_col_no:csv_controls.bio_long_col_no+dim_size.y-1);
                        end
                    end
                    % Also note: utilising the above method means that some
                    % observations are not utilised, since they happen to fall
                    % at the same time bin.

                    % Class covariates
                    if dim_size.class_cov == 1
                        pat_ii.class_cov = 1;
                    else
                        pat_ii.class_cov(1) = 1;
                        % for every class covariate we have
                        for j=2:dim_size.class_cov
                            pat_ii.class_cov(j) = normrnd(curr_dist.mean, curr_dist.std);
                        end
                    end

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

                    % Class covariates
                    pat_ii.class_cov(1) = 1;
                    if dim_size.class_cov > 1
                        % Other class covariates
                        pat_ii.class_cov(2:end) = ...
                            M_mat(curr_ind(j), csv_controls.base_cov_col_no:csv_controls.base_cov_col_no+dim_size.class_cov-2); 
                    end
                    % Store patient data in map
                    data_observed(i) = pat_ii;
                end
            
            end
        end
        
        
        function [data_latent, data_observed] = ...
            sim_obs_surv_pat(num_pats, cens_time, models_true, class_cov_dist, dim_size, frac_miss, cens_range, rand_bool, sim_controls)
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
            %   class_cov_dist - (map) Contains the parameters for the
            %                    normal distributions for the class
            %                    covariates.
            %   dim_size - (struct) Contains the dimension sizes of states,
            %              observations, baseline covariates, and input (if
            %              applicable).
            %   frac_miss - (double) The fraction of observations to be
            %               "missing" within the state space model.
            %   cens_range - (vector) Array containing the range at which
            %                to make uniform censoring.
            %                Format: [censor_start censor_end]
            %   rand_bool - (boolean) Used to randomise every simulation,
            %               regardless of the seed value set in the main 
            %               file.
            %   sim_controls - (struct) Controls to create the simulations
            %                  including the type of mixture model, and the
            %                  baseline hazard function (if applicable).
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
            %   24/01/2023 - mcauchi1
            %       * Introduced class_cov_dist to create custom normal
            %       distributions for the class covariates
            %       * Adapted to allow for mixture models
            %   22/02/2023 - mcauchi1
            %       * Fixed an issue where we had the last measurement for
            %       censored patients as missing
            %   15/03/2023 - mcauchi1
            %       * Introduced input to the state space model
            %   23/03/2023 - mcauchi1
            %       * Introduced the alternative mixture model extension
            %       with the Weibull baseline hazard function.
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
            
            num_classes = double(models_true.Count);
            
            curr_model = models_true(1);
            
            % Find the maximum index for the longitudinal observations
            max_cens_idx = ceil(cens_time / curr_model.DeltaT);

            for ii=1:num_pats % for every patient
                % Initialisations of arrays
                x_cln = zeros(dim_size.states, 1, max_cens_idx); % hidden states without disturbances
                x_true = zeros(dim_size.states, 1, max_cens_idx); % hidden states with disturbances
                haz_true = zeros(1, max_cens_idx); % Hazard function over time
                m_i = 0; % Number of (SSM) iterations observed for patient
                delta_ev = 0; % event indicator
                surv_time = 0; % survival time
                y = zeros(dim_size.y, 1, max_cens_idx); % longitudinal observations
                base_cov = zeros(dim_size.base_cov, 1); % baseline covariates
                class_cov = zeros(dim_size.class_cov, 1); % class covariates
                class_prob = zeros(num_classes, 1); % probability of being in a particular class
                
                if isfield(dim_size, 'u') % if input is affecting the state space model
                    u = zeros(dim_size.u, 1, max_cens_idx); % input array
                    input_range_chg = [5 20]; % it takes between 5-20 indices to change input value
                end

                % Baseline covariates are randomised from a normal distribution
                if strcmpi(sim_controls.MM, 'altMM') % if we are using the alternative mixture model
                    if strcmpi(sim_controls.base_haz, 'Weibull') % if we have the Weibull baseline hazard function
                        q_ii = randn(dim_size.base_cov,1); % N(0,1) - no intercept for this format as Weibull will take care of it
                    end
                else % if we are using the standard mixture model
                    % First baseline covariate is the intercept
                    if dim_size.base_cov == 1
                        q_ii = 1;
                    else
                        q_ii = randn(dim_size.base_cov,1); % N(0,1)
                        q_ii(1) = 1; % intercept
                    end
                end
                
                % Store the survival baseline covariates
                base_cov(:,1) = q_ii;
                
                % Class covariates
                if dim_size.class_cov == 1
                    class_cov = 1;
                else
                    class_cov(1) = 1;
                    % for every class covariate we have
                    for j=2:dim_size.class_cov
                        % extract the normal distribution for the current
                        % class variable
                        curr_dist = class_cov_dist(j-1);
                        % sample from that distribution
                        class_cov(j) = normrnd(curr_dist.mean, curr_dist.std);
                    end
                end
                
                % Generate a number between 0 and 1 to choose the class for
                % this patient
                unif_class = rand(1);
                % Calculate prior probabilities for all classes to choose
                % the class for the patient with these covariates
                for g=1:num_classes
                    model_g = models_true(g);
                    class_prob(g) = exp(model_g.zeta' * class_cov);
                end
                
                class_prob = class_prob / sum(class_prob); % softmax regression
                
                cum_class = 0; % cumulative distribution for probability of class
                
                class_pat = 0; % class number for current patient
                for g=1:num_classes
                    cum_class = cum_class + class_prob(g); % add the probability
                    if unif_class <= cum_class % if number within cumulative value
                        class_pat = g; % class g is chosen
                        break;
                    end
                end
                
                curr_model = models_true(g); % choose the correct model
                
                x0_tmp = curr_model.mu_0; % set initial condition for patient ii

                % Survival of the patient - Utilising the inverse transform sampling
                unif_dist = rand(1); % the smaller this value, the higher chance the patient has to survive
                
                % initial state value and observation calculation
                x_cln(:,:,1) = x0_tmp; % initialise x with no process noise

                % Finding the sqrt of the variance (to obtain std dev)
                sqrtW0 = chol(curr_model.W_0, 'lower');
                x_true(:,:,1) = x0_tmp + sqrtW0 * randn(dim_size.states,1); % initialise x with disturbance

                if isfield(curr_model, 'B') % if the input is affecting the state space model
                    input_chg_rand = rand(1); % sample from a uniform distribution to choose time of change
                    % find the corresponding index at which time changes
                    % according to the chosen range
                    input_changes_idx = round(1 + input_range_chg(1) + input_chg_rand * diff(input_range_chg));
                    input_changes_idx = min(input_changes_idx, max_cens_idx); % maximum is at censor time
                    u(:,:,1:input_changes_idx) = 0.1 * poissrnd(1,[dim_size.u,1]); % sample from a poisson distribution
                    % note that the input does not affect the initial value
                    % as this will alter the distribution of the initial
                    % state, and we assume that the input at time t=-1 is
                    % at zero.
                end

                % enforce a lower limit of 0 on x - assuming negative
                % biomarker values do not exist
                x_true(:,:,1) = max(0, x_true(:,:,1));

                % Finding the sqrt of the variance (to obtain std dev)
                sqrtW = chol(curr_model.W, 'lower');
                sqrtV = chol(curr_model.V, 'lower');

                % initialise y with measurement noise
                y(:,:,1) = curr_model.C * x_true(:,:,1) + sqrtV * randn(dim_size.y,1); 

                % enforce a lower limit of 0 on y - assuming negative
                % biomarker values do not exist
                y(:,:,1) = max(0, y(:,:,1));

                %%% Let's assume first observation is always observed.
                %%% Uncomment below if this is not a valid assumption.
                % y_miss = rand(dim_size.y, 1) < frac_miss;
                % y_o(y_miss == 1,:,1,ii) = NaN;
                
                if strcmpi(sim_controls.MM, 'altMM') % if we are using the alternative mixture model extension
                    if strcmpi(sim_controls.base_haz, 'Weibull') % if we are using the Weibull baseline hazard function
                        
                        % Censoring time is assumed to be drawn from a 
                        % uniform distribution with the provided limits
                        unif_dist_cens = rand(1);
                        cens_num = unif_dist_cens * diff(cens_range) + cens_range(1);
                        max_cens_time = min(cens_time, cens_num);
                        
                        % calculate the survival time of the patient using
                        % the inverse transform sampling
                        true_surv_time = curr_model.b_s * (- exp( curr_model.g_s' * q_ii).^(-1) .* log(unif_dist)).^(1/curr_model.a_s);
                        delta_ev = true_surv_time < max_cens_time; % patient experienced event if surv_time is smaller than censoring time
                        surv_time = min(true_surv_time, max_cens_time);
                        
                        % calculate the number of observations for patient
                        % ceil(.) is used since the first observation is
                        % made at time t=0.
                        m_i = ceil(surv_time / curr_model.DeltaT);
                        
                        % Find the time array for the observations from 0
                        % till the last observation time - used to
                        % calculate the true hazard of the patient
                        time_values = 0:curr_model.DeltaT:(ceil(surv_time / curr_model.DeltaT)-curr_model.DeltaT);
                        haz_true(1,1:m_i) = curr_model.a_s / curr_model.b_s * (time_values).^(curr_model.a_s) * exp(curr_model.g_s' * q_ii);
                        
                        for j=2:m_i % for the rest of the observations
                            x_cln(:,:,j) = curr_model.A * x_cln(:,:,j-1); % calculate x without disturbance
                            % calculate x with disturbance
                            x_true(:,:,j) = curr_model.A * x_true(:,:,j-1) + curr_model.G_mat * sqrtW * randn(size(sqrtW,2),1); 

                            if isfield(curr_model, 'B') % if the input is affecting the state space model
                                % Note that the input at the previous iteration affects the current x.
                                % This is because of causality and the fact that if a patient is prescribed
                                % medication at this time, we see a change due to medication in the next time step.
                                x_true(:,:,j) = x_true(:,:,j) + curr_model.B * u(:,:,j-1); % update x with input u
                                x_cln(:,:,j) = x_cln(:,:,j) + curr_model.B * u(:,:,j-1); % update clean x with input u
                                if j==input_changes_idx % if it is time to change the input value
                                    input_chg_rand = rand(1); % sample from a uniform distribution to choose time of change
                                    % find the corresponding index at which time changes
                                    % according to the chosen range
                                    input_changes_idx = round(j + input_range_chg(1) + input_chg_rand * diff(input_range_chg));
                                    input_changes_idx = min(input_changes_idx, m_i); % maximum is at patient's censor time
                                    u(:,:,j:input_changes_idx) = 0.25 * poissrnd(1,[dim_size.u,1]); % sample from a poisson distribution
                                end
                            end

                            % % enforce a lower limit of 0 on x - assuming negative
                            % % biomarker values do not exist
                            % x_true(:,:,j) = max(0, x_true(:,:,j));

                            % calculate y with measurement noise
                            y(:,:,j) = curr_model.C * x_true(:,:,j) + sqrtV * randn(dim_size.y,1); 

                            % % enforce a lower limit of 0 on y - assuming negative
                            % % biomarker values do not exist
                            % y(:,:,j) = max(0, y(:,:,j));

                            % randomise the missing observations based on the
                            % expected fraction of missing values Note: at some
                            % time points, we may have partial missing
                            % observations (i.e. only some of the biomarkers
                            % are missing)
                            y_miss = rand(dim_size.y, 1) < frac_miss;
                            y(y_miss == 1,:,j) = NaN;

                        end % End of observation period
                    end
                    
                else % if we are using the standard mixture model extension
                    cum_haz = 0; % initialise cumulative hazard

                    % calculate initial hazard
                    haz_true(1,1) = exp( curr_model.g_s' * q_ii + curr_model.a_s' * curr_model.H_mat * x_true(:,:,1));
                    % add hazard to cumulative hazard
                    cum_haz = cum_haz + curr_model.DeltaT * haz_true(1,1);

                    % Censoring time is assumed to be drawn from a uniform
                    % distribution with the provided limits
                    unif_dist_cens = rand(1);
                    cens_num = unif_dist_cens * diff(cens_range) + cens_range(1);
                    max_t_points = min(cens_time, cens_num) / curr_model.DeltaT; % calculate max number of observations for patient

                    if log(unif_dist) > - cum_haz % if cumulative hazard exceeds a certain value (based in the sampled value)
                        % T_i = tau_i - ( ln(S_i(t)) / h_i(t) )
                        surv_time = 0 * curr_model.DeltaT - (log(unif_dist) + 0) / haz_true(1,1);
                        delta_ev = 1; % patient experienced event
                        m_i = 1; % patient had a single observation (at time = 0)
                    else % If they had more observations before death
                        prev_cum_haz = cum_haz; % store current cumulative hazard
                        for j=2:ceil(max_t_points) % for the rest of the observations
                            x_cln(:,:,j) = curr_model.A * x_cln(:,:,j-1); % calculate x without disturbance
                            % calculate x with disturbance
                            x_true(:,:,j) = curr_model.A * x_true(:,:,j-1) + curr_model.G_mat * sqrtW * randn(size(sqrtW,2),1); 

                            if isfield(curr_model, 'B') % if the input is affecting the state space model
                                % Note that the input at the previous iteration affects the current x.
                                % This is because of causality and the fact that if a patient is prescribed
                                % medication at this time, we see a change due to medication in the next time step.
                                x_true(:,:,j) = x_true(:,:,j) + curr_model.B * u(:,:,j-1); % update x with input u
                                x_cln(:,:,j) = x_cln(:,:,j) + curr_model.B * u(:,:,j-1); % update clean x with input u
                                if j==input_changes_idx
                                    input_chg_rand = rand(1); % sample from a uniform distribution to choose time of change
                                    % find the corresponding index at which time changes
                                    % according to the chosen range
                                    input_changes_idx = round(j + input_range_chg(1) + input_chg_rand * diff(input_range_chg));
                                    input_changes_idx = min(input_changes_idx, ceil(max_t_points)); % maximum is at patient's censor time
                                    u(:,:,j:input_changes_idx) = 0.25 * poissrnd(1,[dim_size.u,1]); % sample from a poisson distribution
                                end
                            end

                            % enforce a lower limit of 0 on x - assuming negative
                            % biomarker values do not exist
                            x_true(:,:,j) = max(0, x_true(:,:,j));

                            % calculate y with measurement noise
                            y(:,:,j) = curr_model.C * x_true(:,:,j) + sqrtV * randn(dim_size.y,1); 

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
                            haz_true(1,j) = exp( curr_model.g_s' * q_ii + curr_model.a_s' * curr_model.H_mat * x_true(:,:,j));
                            % calculate cumulative hazard
                            cum_haz = cum_haz + curr_model.DeltaT * haz_true(1,j);

                            if log(unif_dist) > - cum_haz % if cumulative hazard exceeds a certain value
                                % T_i = tau_i - ( ln(S_i(t)) + H_i(t) / h_i(t) )
                                surv_time = (j-1) * curr_model.DeltaT - (log(unif_dist) + prev_cum_haz) / haz_true(1,j);
                                delta_ev = 1; % patient experienced event
                                m_i = j; % patient had j observations
                                break; % future observations are not required/possible.
                            end

                            % if patient did not experience event
                            prev_cum_haz = cum_haz; % store the cumulative hazard function

                        end % End of observation period
                        if delta_ev == 0 % if patient remained event-free
                            surv_time = (max_t_points) * curr_model.DeltaT; % calculate survival time
                            m_i = j; % patient had a total of j observations
                        end
                    end
                end
                
                % Store data in maps
                data_latent(ii) = struct('x_cln', x_cln, 'x_true', x_true, 'haz_true', haz_true, 'class', class_pat);
                data_observed(ii) = struct('m_i', m_i, 'delta_ev', delta_ev, 'surv_time', surv_time, ...
                                           'y', y, 'base_cov', base_cov, 'class_cov', class_cov);
                
                if isfield(dim_size, 'u') % if we have input to the state space model
                    data_observed(ii) = setfield(data_observed(ii), 'u', u);
                end
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
        
        
        function output_arr = extract_field_from_map(map_tmp, field_name)
            % FUNCTION NAME:
            %   extract_field_from_map
            %
            % DESCRIPTION:
            %   Function to extract a particular field from a map of
            %   patients and store the arrays within a single nd-array. The
            %   patient iteration is always kept at the last dimension.
            %   
            %   E.g. to extract the baseline covariates for every patient,
            %   or the survival times of all patients.
            %
            % INPUT:
            %   map_tmp - (map) Map containing all patient data.
            %   field_name - (string) The name of the field to be
            %                extracted.
            %
            % OUTPUT:
            %   output_arr - (array) Output in array format.
            %
            % REVISION HISTORY:
            %   22/02/2023 - mcauchi1
            %       * Initial implementation
            %
            
            % if the field is a double (not an array)
            if isequal(size(getfield(map_tmp(1), field_name)), [1 1])
                output_arr = zeros([1, map_tmp.Count]); % initialise array as (1 x n) vector
            else
                % else, initialise array with correct dimensions, keeping
                output_arr = zeros([size(getfield(map_tmp(1), field_name)), map_tmp.Count]);
            end
            
            % identify the number of dimensions, and populate idx with ":"
            % indicating all iterations in that dimension
            nd = ndims(output_arr);
            idx = cell(1, nd);
            idx(:) = {':'};
            
            for ii=1:map_tmp.Count % for every patient
                idx{end} = ii; % iterate over the last dimension (indicating patient number)
                % extract and store the patient's data in the output array
                output_arr(idx{:}) = getfield(map_tmp(ii), field_name);
            end
        end
   
        
        function plot_forecast_surv(pat_ii, landmark_t, t_arr, Delta)
            % FUNCTION NAME:
            %   plot_forecast_surv
            %
            % DESCRIPTION:
            %   Shows the predicted survival curve from the start of
            %   observation of the patient. It shows the probability of
            %   survival as deduced by the forecast function
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
            title(t, sprintf('Performance metrics across different horizons at landmark = %.1f months', ...
                landmark_t / t_multiplier));
            xlabel(t, 'Horizon (in months)');
        end
        
        
        function [auc_test_arr] = AUC_fn(pat_data, landmark_t, max_horizon_t, deltaT) 
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
            %   deltaT - (double) Time step for SSM
            %
            % OUTPUT:
            %   auc_test_arr - (array) All AUC calculations for the
            %                  horizons and landmark indicated.
            % 
            % REVISION HISTORY:
            %   17/12/2022 - mcauchi1
            %       * Initial implementation
            %   06/04/2023 - mcauchi1
            %       * Replaced the parameter for the entire estimated
            %       models with deltaT since we are only using this
            %       variable inside that struct.
            %       * Removed the +1 from all indices. Now the first time
            %       point of survival represents the time point after
            %       observing the first period.
            %
            
            % +1 due to index starts from 1 - REMOVED
            landmark_idx_tmp = floor(landmark_t / deltaT);
            
            % total number of patients at risk at landmark time
            no_of_pat_auc_test = double(pat_data.Count);
            
            % surv time and delta matrix
            surv_info_mat = zeros(no_of_pat_auc_test, 2);
            
            for ii=1:no_of_pat_auc_test % for every patient
                % store the survival information
                surv_info_mat(ii,:) = [pat_data(ii).surv_time, pat_data(ii).delta_ev];
            end

            % Initialise array to store AUC calculations
            auc_test_arr = zeros(1, int64(max_horizon_t / deltaT));

            % For every horizon to test
            for horizon_t_tmp=deltaT:deltaT:max_horizon_t

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
                t_est_idx = int64(t_est / deltaT);

                % Horizon index - used to store the AUC calculations
                horizon_idx = int64(horizon_t_tmp / deltaT);
                
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
                        surv_fn_ii = pat_ii.pred_surv;
                        W_i = weights_crit_test(ii);

                        for jj=1:no_of_pat_auc_test % for every patient
                            pat_jj = pat_data(jj); % store second patient data
                            
                            % Survival function for patient jj
                            surv_fn_jj = pat_jj.pred_surv;
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
        
        
        function conf_mat = buildConfMat(thresh_vec, predVar, trueVar)
            % FUNCTION NAME:
            %   buildConfMat
            %
            % DESCRIPTION:
            %   Builds the confusion matrix at every required threshold.
            %
            % INPUT:
            %   thresh_vec - (array) Threshold values to consider.
            %   predVar - (double) Predicted variable for every patient.
            %             E.g. Survival probability.
            %   trueVar - (boolean) Whether the patient survived or not.
            %
            % OUTPUT:
            %   conf_mat - (table) Table of all confusion matrix properties
            %              including true/false positives/negatives, and
            %              sensitivity, specificity, and accuracy.
            % 
            % REVISION HISTORY:
            %   01/09/2023 - mcauchi1
            %       * Initial implementation
            %
            
            % This function assumes that event is 1 if predicted value is smaller than
            % the threshold value
            % e.g. when survival prediction is smaller than X, then patient is expected
            % to experience the event

            % create a table to place the outcomes within it
            conf_mat = array2table(zeros(0, 5), 'VariableNames', {'Thresh', 'TP', 'FP', 'FN', 'TN'});

            % for every threshold value required
            for idx = 1:length(thresh_vec)
                s_thresh = thresh_vec(idx);

                % create row to place the outcome for that threshold value
                conf_mat(end + 1, :) = {s_thresh, 0, 0, 0, 0};

                conf_mat{end, 'TP'} = sum(predVar < s_thresh & trueVar == 1);
                conf_mat{end, 'FP'} = sum(predVar < s_thresh & trueVar == 0);
                conf_mat{end, 'FN'} = sum(predVar >= s_thresh & trueVar == 1);
                conf_mat{end, 'TN'} = sum(predVar >= s_thresh & trueVar == 0);
            end

            % calculate sensitivity, specificity, and accuracy
            conf_mat.sensitivity = conf_mat.TP ./ (conf_mat.TP + conf_mat.FN);
            conf_mat.specificity = conf_mat.TN ./ (conf_mat.TN + conf_mat.FP);
            conf_mat.accuracy = (conf_mat.TP + conf_mat.TN) ./ (conf_mat.TP + conf_mat.TN + conf_mat.FP + conf_mat.FN);
        end
        
        
        function auc = calcAUC(conf_mat)
            % FUNCTION NAME:
            %   calcAUC
            %
            % DESCRIPTION:
            %   Calculates the area under the ROC curve using sensitivity
            %   and specificity values.
            %
            % INPUT:
            %   conf_mat - (table) Table of all confusion matrix properties
            %              including sensitivity, specificity, and
            %              accuracy.
            %
            % OUTPUT:
            %   auc - (double) The area under the ROC curve value.
            % 
            % REVISION HISTORY:
            %   01/09/2023 - mcauchi1
            %       * Initial implementation
            
            % calculate the widths of the trapezoids
            xvals = diff(1 - conf_mat.specificity);

            % calculate the average of two consecutive sensitivity values
            yvals = (conf_mat.sensitivity(2:end) + conf_mat.sensitivity(1:end-1)) / 2;

            % area under the curve
            auc = sum(xvals .* yvals);
        end


        function [pe_test_arr] = Prediction_Error_fn(pat_data, landmark_t, max_horizon_t, deltaT) 
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
            %   deltaT - (double) Time step for SSM
            %
            % OUTPUT:
            %   pe_test_arr - (array) All Prediction Error calculations for 
            %                 the horizons and landmark indicated.
            % 
            % REVISION HISTORY:
            %   17/12/2022 - mcauchi1
            %       * Initial implementation
            %   06/04/2023 - mcauchi1
            %       * Replaced the parameter for the entire estimated
            %       models with deltaT since we are only using this
            %       variable inside that struct.
            %       * Removed the +1 from all indices. Now the first time
            %       point of survival represents the time point after
            %       observing the first period.
            %
            
            % +1 due to index starts from 1 - REMOVED
            landmark_idx_tmp = floor(landmark_t / deltaT);
            
            % initialise the Prediction Error array
            pe_test_arr = zeros(1, int64(max_horizon_t / deltaT));
            
            no_of_pat_pe_test = double(pat_data.Count); % total number of patients at risk at landmark time

            % For every horizon until maximum horizon
            for horizon_t_tmp=deltaT:deltaT:max_horizon_t
                % Find the prediction time for the current horizon
                t_est = landmark_t + horizon_t_tmp;
                t_est_idx = int64(t_est / deltaT);

                horizon_idx_tmp = int64(horizon_t_tmp / deltaT);

                for ii=1:no_of_pat_pe_test % for every patient
                    pat_ii = pat_data(ii); % store patient data
                    surv_fn = pat_ii.pred_surv; % extract survival forecast
                    
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
                        censor_time_idx = floor(pat_ii.surv_time / deltaT);
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
        
        
        function [bs_test_arr] = Brier_Score_fn(pat_data, landmark_t, max_horizon_t, deltaT)              
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
            %                hence, the start of BS calculations.
            %   max_horizon_t - (double) The maximum horizon to work out
            %                   the AUC calculations.
            %   deltaT - (double) Time step for SSM
            %
            % OUTPUT:
            %   bs_test_arr - (array) All Brier Score calculations for the
            %                 horizons and landmark indicated.
            % 
            % REVISION HISTORY:
            %   17/12/2022 - mcauchi1
            %       * Initial implementation
            %   06/04/2023 - mcauchi1
            %       * Replaced the parameter for the entire estimated
            %       models with deltaT since we are only using this
            %       variable inside that struct.
            %       * Removed the +1 from all indices. Now the first time
            %       point of survival represents the time point after
            %       observing the first period.
            %
            
            % +1 due to index starts from 1 - REMOVED
            landmark_idx_tmp = floor(landmark_t / deltaT);
            
            % initialise the Brier Score array
            bs_test_arr = zeros(1, int64(max_horizon_t / deltaT));
            
            % total number of patients at risk at landmark time
            no_of_pat_bs_test = double(pat_data.Count);
            
            % surv time and delta matrix
            surv_info_mat = zeros(no_of_pat_bs_test, 2);
            
            for ii=1:no_of_pat_bs_test
                surv_info_mat(ii,:) = [pat_data(ii).surv_time, pat_data(ii).delta_ev];
            end

            % For every horizon we wish to consider
            for horizon_t_tmp=deltaT:deltaT:max_horizon_t

                % Find the prediction time for the current horizon
                t_est = landmark_t + horizon_t_tmp;
                t_est_idx = int64(t_est / deltaT);

                horizon_idx_tmp = int64(horizon_t_tmp / deltaT);

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
                    surv_fn = pat_ii.pred_surv; % extract survival forecasts
                    
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
            % TO DO:
            %   update to work with multiple classes
            %   soft and hard assignments
            
            % Calculate the maximum index
            max_num_pts = ceil(max_censor_time / model_coef_est.DeltaT);
            
            % Create a map to store the output
            pat_data_out = containers.Map('KeyType', 'int32', 'ValueType', 'any');
            
            for ii=1:pat_data.Count % for every patient

                pat_ii = pat_data(ii); % store patient data
                
                % Set initial conditions
                pat_ii.mu_0 = model_coef_est.mu_0;
                pat_ii.W_0 = model_coef_est.W_0;

                
                % RTS Filter
                [pat_ii.mu_tilde, pat_ii.V_tilde] = ...
                    LSDSM_MM_ALLFUNCS.Kalman_filter(pat_ii, model_coef_est, max_num_pts, controls);

                % RTS Smoother
                [pat_ii.predictions.mu, pat_ii.predictions.V, pat_ii.predictions.J] = ...
                    LSDSM_MM_ALLFUNCS.Kalman_smoother(pat_ii, model_coef_est, max_num_pts);

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
        
        
        function [pat_data_out] = forecast_fn(pat_data, landmark_t, t_est_idx, max_censor_time, ...
                                                        dim_size, model_coef_est_all_classes, controls, ignore_all_NaNs)
            % FUNCTION NAME:
            %   forecast_fn
            %
            % DESCRIPTION:
            %   Makes forecasts on hidden states and survival curves for
            %   the patients given the data available until the landmark
            %   time.
            %   Utilises proper probability distributions to identify an
            %   approximation to the probability that the patient survives
            %   beyond a certain time point.
            %
            % INPUT:
            %   pat_data - (map) Observed data of all patients including
            %              longitudinal biomarkers, survival time, and 
            %              event indicator.
            %   landmark_t - (double) Landmark time at which to start the
            %                forecasts
            %   t_est_idx - (double) Time index to make forecasts.
            %   dim_size - (struct) Contains the dimension sizes of the
            %              latent states, dynamic states, and the number of
            %              observations and the baseline covariates.
            %   max_censor_time - (double) The maximum censoring time.
            %   model_coef_est_all_classes - (map) Contains model
            %                                parameters for all classes.
            %   controls - (struct) Contains the controls for the LSDSM EM,
            %              including the boolean to indicate if survival
            %              data should affect hidden states.
            %   ignore_all_NaNs - (boolean) Includes patient if they have
            %                     at least one measurement in observation
            %                     window.
            %
            % OUTPUT:
            %   pat_data_out - (map) All observed patient data, together
            %                  with the forecast outputs.
            % 
            % REVISION HISTORY:
            %   17/12/2022 - mcauchi1
            %       * Initial implementation
            %   06/04/2023 - mcauchi1
            %       * Removed the +1 from all indices. Now the first time
            %       point of survival represents the time point after
            %       observing the first period.
            %   10/09/2023 - mcauchi1
            %       * Works with multiple classes
            %
            % TO DO:
            %   hard assignments option
            %
            
            % Find the landmark index
            % +1 due to index starts from 1 - REMOVED
            % if landmark is at time 30, then idx would be 30 as well. This
            % means that we shall be using the 30th observation which
            % corresponds to time 29 in the filtering steps, and this
            % corresponds to the 30th iterate in the survival array, which
            % corresponds to time 30 (survival array is assumed to start 
            % from time 1).
            % In other words, longitudinal trajectories start from time 0
            % while survival curves start from time 1.
            landmark_idx = floor(landmark_t / model_coef_est_all_classes(1).DeltaT);
            
            % Find the maximum index - the only time we do not add +1
            % (since we are assuming the above structure)
            max_num_pts = floor(max_censor_time / model_coef_est_all_classes(1).DeltaT);
            
            % Find the number of classes
            num_classes = double(model_coef_est_all_classes.Count);
            % Initialise cells to retain forecast arrays for every class
            forecast_arrs = cell(num_classes, 1);
            
            % Initialise some arrays to be used for forecasting
            forecasts_tmp.mu_pred = zeros(dim_size.states, 1, max_num_pts);
            forecasts_tmp.V_pred = zeros(dim_size.states, dim_size.states, max_num_pts);
            forecasts_tmp.mu_corr = zeros(dim_size.states, 1, max_num_pts);
            forecasts_tmp.V_corr = zeros(dim_size.states, dim_size.states, max_num_pts);
            forecasts_tmp.surv_phi = zeros(1, 1, max_num_pts);
            forecasts_tmp.survival = zeros(1, 1, max_num_pts);
            
            % controls for Newton Raphson procedure
            NR_controls = struct('max_iter', 100, 'eps', 1e-6, 'damp_factor', 0.5, 'minimising', 1);
            
            % Stores patient data and forecasts for patients that survive beyond landmark time
            pat_data_out = containers.Map('KeyType', 'int32', 'ValueType', 'any');
            
            red_ii = 0; % used to keep track of the number of patients that survive beyond the landmark point

            for ii=1:pat_data.Count % iterate for every patient

                pat_ii = pat_data(ii); % Store the patient data
                
                include_pat = 1;
                if ignore_all_NaNs % include patient if they have at least one observed measurement before landmark
                    y_ii = pat_ii.y;
                    include_pat = not(all(all(isnan(y_ii(:,:,1:landmark_idx+1))))); % 2 all functions in case of multivariate biomarkers
                end
                
                if pat_ii.surv_time > landmark_t && include_pat % forecast if they survived beyond the landmark
                    
                    pat_ii.id = ii; % store patient ID (since numbering will change in new map)
                    
                    overall_pred_surv = zeros(1, 1, max_num_pts);
                    
                    % increase the iterative for the reduced patient data set
                    red_ii = red_ii + 1;

                    for g=1:num_classes % populate forecast arrays for every class
                        forecast_arrs{g} = forecasts_tmp;
                    end

                    % responsibility of class g for patient i
                    E_c_ig = zeros(1, num_classes);
                    prior_probs = zeros(1, num_classes);

                    % Prior Probabilities
                    % Calculate numerator of prior probability for each class
                    for g=1:num_classes
                        zeta_class_g = model_coef_est_all_classes(g).zeta;
                        prior_probs(g) = exp(zeta_class_g' * pat_ii.class_cov);
                    end
                    % Normalise these prior probabilities
                    prior_probs = prior_probs / sum(prior_probs);

                    for g=1:num_classes % for every class

                        % Set coefficients for the g-th model
                        model_coef_est = model_coef_est_all_classes(g);

                        % Set initial conditions
                        pat_ii.mu_0 = model_coef_est.mu_0;
                        pat_ii.W_0 = model_coef_est.W_0;

                        % Set initial state conditions for patient ii
                        pat_ii.mu_0 = model_coef_est.mu_0;
                        pat_ii.W_0 = model_coef_est.W_0;

                        pat_ii_tmp = pat_ii; % create a temporary struct to use within the RTS filter

                        pat_ii_tmp.m_i = landmark_idx; % set the number of observations equal to the landmark index
                        pat_ii_tmp.delta_ev = 0; % since they survived beyond landmark

                        %%% Forward recursion - Standard/Modified RTS Filter %%%
                        [pat_ii.mu_tilde, pat_ii.V_tilde, log_like_val] = ...
                            LSDSM_MM_ALLFUNCS.Kalman_filter(pat_ii_tmp, model_coef_est, max_num_pts, controls);

                        % Calculate the numerator for E[c_ig]
                        E_c_ig(g) = exp(log_like_val) * prior_probs(g);

                        % Utilise the RTS filter output until the landmark (one time step before)
                        % This corresponds to landmark_idx
                        forecast_arrs{g}.mu_corr(:,:,1:landmark_idx) = pat_ii.mu_tilde(:,:,1:landmark_idx);
                        forecast_arrs{g}.V_corr(:,:,1:landmark_idx) = pat_ii.V_tilde(:,:,1:landmark_idx);

                        % patient is known to be alive until the
                        % landmark time
                        forecast_arrs{g}.surv_phi(:,:,1:landmark_idx) = 1;
                        forecast_arrs{g}.survival(:,:,1:landmark_idx) = 1;
                    end

                    % store the forecasts for this patient for every
                    % class
                    pat_ii.forecasts = forecast_arrs;

                    % Calculate the posterior probability of patient i belonging to class g
                    E_c_ig = E_c_ig / sum(E_c_ig);

                    % If we obtain likelihood values of 0 for all classes
                    % resort to the prior probabilities for E[c_ig]
                    if all(isnan(E_c_ig))
                        E_c_ig = prior_probs;
                    end

                    % Forecasts
                    for g=1:num_classes % for every class
                        % Set coefficients for the g-th model
                        model_coef_est = model_coef_est_all_classes(g);

                        % initialisations of forecasts for easy access
                        fore_tmp = pat_ii.forecasts{g};

                        if ~controls.mod_KF % if using the alternative mixture model extension
                            if strcmpi(controls.base_haz, 'Weibull') % if using the Weibull baseline hazard function
                                surv_prob_at_lm = exp( - (landmark_t / model_coef_est.b_s)^model_coef_est.a_s ...
                                                       * exp(model_coef_est.g_s' * pat_ii.base_cov));
                            end
                        end

                        for j=landmark_idx+1:t_est_idx % for every horizon
                            % 1. Predictions
                            if j==landmark_idx+1 % if this is the first horizon
                                % At this point, an observation at the landmark time might be available, but
                                % it was not used since the Kalman filter assumes that the patient survives 
                                % another period. Hence we can simply filter using the observations only.
                                if isfield(pat_ii, 'u') % if input to SSM exists
                                    [mu_filt, V_filt, K_filt, P_filt] = ...
                                                LSDSM_MM_ALLFUNCS.KF_single_step(fore_tmp.mu_corr(:,:,landmark_idx), ...
                                                                              fore_tmp.V_corr(:,:,landmark_idx), ...
                                                                              pat_ii.y(:,:,landmark_idx+1), model_coef_est, ...
                                                                              pat_ii.u(:,:,landmark_idx));
                                else % if there is no input
                                    [mu_filt, V_filt, K_filt, P_filt] = ...
                                                LSDSM_MM_ALLFUNCS.KF_single_step(fore_tmp.mu_corr(:,:,landmark_idx), ...
                                                                              fore_tmp.V_corr(:,:,landmark_idx), ...
                                                                              pat_ii.y(:,:,landmark_idx+1), model_coef_est);
                                end
                                % this corresponds to the predicted
                                % variables
                                fore_tmp.mu_pred(:,:,j) = mu_filt;
                                fore_tmp.V_pred(:,:,j) = V_filt;

                            else % for all other points
                                % predict hidden states variables one-step forward
                                fore_tmp.mu_pred(:,:,j) = model_coef_est.A * fore_tmp.mu_corr(:,:,j-1);
                                if isfield(pat_ii, 'u') % if input to SSM exists
                                    fore_tmp.mu_pred(:,:,j) = fore_tmp.mu_pred(:,:,j) + model_coef_est.B * pat_ii.u(:,:,j-1);
                                end
                                fore_tmp.V_pred(:,:,j) = model_coef_est.A * fore_tmp.V_corr(:,:,j-1) * model_coef_est.A' ...
                                                    + model_coef_est.G_mat * model_coef_est.W * model_coef_est.G_mat';
                            end

                            if ~controls.mod_KF % if using the alternative mixture model extension
                                if strcmpi(controls.base_haz, 'Weibull') % if using the Weibull baseline hazard function
                                    curr_time = j * model_coef_est.DeltaT;
                                    surv_prob = exp( - (curr_time / model_coef_est.b_s)^model_coef_est.a_s ...
                                                                    * exp(model_coef_est.g_s' * pat_ii.base_cov));
                                    fore_tmp.survival(:,:,j) = surv_prob / surv_prob_at_lm;
                                end
                            else
                                % 2. Corrections
                                % Find the value of x that maximises the 
                                % posterior distribution assuming that the 
                                % patient survived the previous time point
                                g_fn_coef = struct('delta_ij', 0, 'mu_ij', fore_tmp.mu_pred(:,:,j), ...
                                               'Sigma_ij', fore_tmp.V_pred(:,:,j), 'base_cov', pat_ii.base_cov, ...
                                               'g_s', model_coef_est.g_s, 'a_s', model_coef_est.a_s, ...
                                               'tau_ij', model_coef_est.DeltaT, 'H_mat', model_coef_est.H_mat);

                                % Use Newton Raphson's iterative method to approximate
                                % the posterior as a Gaussian distribution
                                x_NR = LSDSM_MM_ALLFUNCS.Newton_Raphson(fore_tmp.mu_pred(:,:,j), NR_controls, @LSDSM_MM_ALLFUNCS.f_KF, ...
                                                                        @LSDSM_MM_ALLFUNCS.dfdx_KF, @LSDSM_MM_ALLFUNCS.d2fdx2_KF, g_fn_coef);

                                % find the hessian matrix
                                hess = LSDSM_MM_ALLFUNCS.d2fdx2_KF(x_NR, g_fn_coef);

                                % Update mu_corr and V_corr
                                fore_tmp.mu_corr(:,:,j) = x_NR;
                                fore_tmp.V_corr(:,:,j) = (hess)^-1;
                                fore_tmp.V_corr(:,:,j) = LSDSM_MM_ALLFUNCS.ensure_sym_mat(fore_tmp.V_corr(:,:,j));

                                % 3. Survival probabilities
                                % Use Laplace approximation to estimate the
                                % integral that leads to the probability of
                                % survival within a period of interest.
                                exponent_val = LSDSM_MM_ALLFUNCS.f_KF(x_NR, g_fn_coef);
                                int_val = (2*pi)^(dim_size.states/2) * det(hess)^(-1/2) * exp(-exponent_val);
                                surv_period = (2*pi)^(-dim_size.states/2) * det(fore_tmp.V_pred(:,:,j))^(-1/2) * int_val;
                                if surv_period > 1 % survival probability should never be greater than 1
                                    error('Probability of survival for this period is greater than 1.');
                                end
                                % Store the survival variables
                                fore_tmp.surv_phi(:,:,j) = surv_period;
                                fore_tmp.survival(:,:,j) = fore_tmp.survival(:,:,j-1) * fore_tmp.surv_phi(:,:,j);
                            end
                        end
                        % Store the forecast arrays
                        pat_ii.forecasts{g} = fore_tmp;
                    end

                    % calculate the overall survival of the patient
                    % (which is a weighted average of all classes).
                    for g=1:num_classes % for every class
                        % Add the contribution of class g for the
                        % survival of patient ii
                        overall_pred_surv = overall_pred_surv + E_c_ig(g) * pat_ii.forecasts{g}.survival;
                    end

                    % Store the survival in a (1 x horizons) vector
                    pat_ii.pred_surv = squeeze(overall_pred_surv)';

                    % Store patient data in output map
                    pat_data_out(red_ii) = pat_ii;
                end
            end
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
            
            % initialise the sum of squared error array
            total_sse_val = zeros(size(comp_signal,1), 1);
            total_obs_test = 0; % number of observations for all patients
            t_dim = 3; % time is along this dimension in the array

            for ii=1:data_observed.Count % for every patient
                % Extract the true hidden state trajectories
                true_sig_ii = data_latent(ii).x_true(:,:,2:data_observed(ii).m_i);
                % Extract the comparison signals
                comp_sig_ii = comp_signal(:,:,2:data_observed(ii).m_i,ii);
                % count the total number of observations
                total_obs_test = total_obs_test + data_observed(ii).m_i - 1; 

                % Find the sum of square error
                [sse_val, ~] = LSDSM_MM_ALLFUNCS.rmse_fn(true_sig_ii, comp_sig_ii, t_dim);
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
            %               true and comparison survival signals.
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
                [sse_val, ~] = LSDSM_MM_ALLFUNCS.rmse_fn(true_sig_ii, comp_sig_ii, t_dim);
                total_sse_val = total_sse_val + sse_val;
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
            %   data. Valid for simulations only where true parameters are
            %   known.
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
            % TO BE UPDATED
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
                    haz_fn(:,j) = exp(model_coef_est.g_s' * base_cov_ii) * exp(model_coef_est.a_s' * model_coef_est.H_mat * x_ii(:,1,j));
                    % Update cumulative hazard
                    cum_haz = cum_haz + model_coef_est.DeltaT * haz_fn(:,j);
                    % Record new survival probability
                    surv_fn(:,j+1,ii) = exp(-cum_haz);
                end
            end
        end
        
        
        function surv_fn_all = surv_curve_filt(model_coef_est_all_classes, pat_data, max_censor_time, dim_size, controls)
            % FUNCTION NAME:
            %   surv_curve_filt
            %
            % DESCRIPTION:
            %   Calculates the survival curves for all patients based on
            %   the provided model, hidden state values, and the observed
            %   data. Filters the states and performs survival analysis
            %   at every step. This is used in simulations to check the
            %   estimated model's ability to track the true survival curve.
            %
            % INPUT:
            %   model_coef_est_all_classes - (map) Model parameters for all
            %                                classes
            %   pat_data - (map) Observed data of all patients including
            %              longitudinal biomarkers, survival time, and 
            %              event indicator.
            %   max_censor_time - (double) The maximum censoring time.
            %   dim_size - (struct) Contains the dimension sizes of states,
            %              observations, baseline covariates, and input (if
            %              applicable).
            %   controls - (struct) Contains the controls for the LSDSM EM,
            %              including the boolean to indicate if survival
            %              data should affect hidden states.
            %
            % OUTPUT:
            %   surv_fn_all - (array) The survival curves estimation for 
            %                 all patients.
            %
            % REVISION HISTORY:
            %   01/09/2023 - mcauchi1
            %       * Initial implementation
            %
            
            % placeholder for survival probabilities for every patient at
            % every time point
            surv_fn_all = zeros(1, max_censor_time+1, pat_data.Count);
            
            % Find the maximum index
            max_num_pts = floor(max_censor_time / model_coef_est_all_classes(1).DeltaT);
            
            % Find the number of classes
            num_classes = double(model_coef_est_all_classes.Count);
            % Initialise cells to retain forecast arrays for every class
            forecast_arrs = cell(num_classes, 1);
            
            % Initialise some arrays to be used for forecasting
            forecasts_tmp.mu_long_corr = zeros(dim_size.states, 1, max_num_pts);
            forecasts_tmp.V_long_corr = zeros(dim_size.states, dim_size.states, max_num_pts);
            forecasts_tmp.mu_corr = zeros(dim_size.states, 1, max_num_pts);
            forecasts_tmp.V_corr = zeros(dim_size.states, dim_size.states, max_num_pts);
            forecasts_tmp.surv_phi = zeros(1, 1, max_num_pts+1);
            forecasts_tmp.survival = zeros(1, 1, max_num_pts+1);
            
            % controls for Newton Raphson procedure
            NR_controls = struct('max_iter', 100, 'eps', 1e-6, 'damp_factor', 0.5, 'minimising', 1);

            for ii=1:pat_data.Count % iterate for every patient

                pat_ii = pat_data(ii); % Store the patient data
                
                pat_ii.id = ii; % store patient ID (since numbering will change in new map)

                overall_pred_surv = zeros(1, 1, max_num_pts+1);

                for g=1:num_classes % populate forecast arrays for every class
                    forecast_arrs{g} = forecasts_tmp;
                end

                % responsibility of class g for patient i
                E_c_ig = zeros(1, num_classes);
                prior_probs = zeros(1, num_classes);

                % Prior Probabilities
                % Calculate numerator of prior probability for each class
                for g=1:num_classes
                    zeta_class_g = model_coef_est_all_classes(g).zeta;
                    prior_probs(g) = exp(zeta_class_g' * pat_ii.class_cov);
                end
                % Normalise these prior probabilities
                prior_probs = prior_probs / sum(prior_probs);

                % For every class, we perform the Kalman filter to extract
                % the likelihood, which allows us to identify the
                % probabilities of patient belonging to each class.
                % These probabilities will then be used in the survival
                % forecasts.
                for g=1:num_classes % for every class

                    % Set coefficients for the g-th model
                    model_coef_est = model_coef_est_all_classes(g);

                    % Set initial conditions
                    pat_ii.mu_0 = model_coef_est.mu_0;
                    pat_ii.W_0 = model_coef_est.W_0;

                    pat_ii_tmp = pat_ii; % create a temporary struct to use within the RTS filter

                    %%% Forward recursion - Standard/Modified RTS Filter %%%
                    [pat_ii.mu_tilde, pat_ii.V_tilde, log_like_val] = ...
                        LSDSM_MM_ALLFUNCS.Kalman_filter(pat_ii_tmp, model_coef_est, max_num_pts, controls);

                    % Calculate the numerator for E[c_ig]
                    E_c_ig(g) = exp(log_like_val) * prior_probs(g);

                    % Utilise the RTS filter output until the landmark (one time step before)
                    % This corresponds to landmark_idx
                    forecast_arrs{g}.mu_corr = pat_ii.mu_tilde;
                    forecast_arrs{g}.V_corr = pat_ii.V_tilde;

                    % patient is known to be alive until the landmark time
                    forecast_arrs{g}.surv_phi(:,:,1) = 1;
                    forecast_arrs{g}.survival(:,:,1) = 1;
                end

                % store the forecasts for this patient for every class
                pat_ii.forecasts = forecast_arrs;

                % Calculate the posterior probability of patient i belonging to class g
                E_c_ig = E_c_ig / sum(E_c_ig);

                % If we obtain likelihood values of 0 for all classes
                % resort to the prior probabilities for E[c_ig]
                if all(isnan(E_c_ig))
                    E_c_ig = prior_probs;
                end

                % Forecasts
                for g=1:num_classes % for every class
                    % Set coefficients for the g-th model
                    model_coef_est = model_coef_est_all_classes(g);

                    % initialisations of forecasts for easy access
                    fore_tmp = pat_ii.forecasts{g};

                    if ~controls.mod_KF % if using the alternative mixture model extension
                        if strcmpi(controls.base_haz, 'Weibull') % if using the Weibull baseline hazard function
                            surv_prob_at_lm = exp( - (landmark_t / model_coef_est.b_s)^model_coef_est.a_s ...
                                                   * exp(model_coef_est.g_s' * pat_ii.base_cov));
                        end
                    end

                    for j=1:pat_ii.m_i

                        if ~controls.mod_KF % if using the alternative mixture model extension
                            if strcmpi(controls.base_haz, 'Weibull') % if using the Weibull baseline hazard function
                                curr_time = j * model_coef_est.DeltaT;
                                surv_prob = exp( - (curr_time / model_coef_est.b_s)^model_coef_est.a_s ...
                                                                * exp(model_coef_est.g_s' * pat_ii.base_cov));
                                fore_tmp.survival(:,:,j+1) = surv_prob / surv_prob_at_lm;
                            end
                        else
                            
                            if j>1
                                if isfield(pat_ii, 'u') % if input to SSM exists
                                    [mu_filt, V_filt, K_filt, P_filt] = ...
                                                LSDSM_MM_ALLFUNCS.KF_single_step(fore_tmp.mu_corr(:,:,j-1), ...
                                                                              fore_tmp.V_corr(:,:,j-1), ...
                                                                              pat_ii.y(:,:,j), model_coef_est, ...
                                                                              pat_ii.u(:,:,j-1));
                                else % if there is no input
                                    [mu_filt, V_filt, K_filt, P_filt] = ...
                                                LSDSM_MM_ALLFUNCS.KF_single_step(fore_tmp.mu_corr(:,:,j-1), ...
                                                                              fore_tmp.V_corr(:,:,j-1), ...
                                                                              pat_ii.y(:,:,j), model_coef_est);
                                end
                                % this corresponds to the predicted
                                % variables
                                fore_tmp.mu_long_corr(:,:,j) = mu_filt;
                                fore_tmp.V_long_corr(:,:,j) = V_filt;
                            else % if j==1
                                [y_tmp_i, C_tmp_i, V_tmp_i, Omega_O, Omega_M, I_mat_O, I_mat_M, nabla_ij] ...
                                        = LSDSM_MM_ALLFUNCS.missing_val_matrices(pat_ii.y(:,:,j), model_coef_est);

                                % Correction using the longitudinal data
                                K(:,:,1) = pat_ii.W_0 * C_tmp_i' * (C_tmp_i * pat_ii.W_0 * C_tmp_i' + V_tmp_i)^-1;
                                fore_tmp.mu_long_corr(:,:,1) = pat_ii.mu_0 + K(:,:,1) * (y_tmp_i - C_tmp_i * pat_ii.mu_0);
                                fore_tmp.V_long_corr(:,:,1) = (eye(size(pat_ii.mu_0,1)) - K(:,:,1) * C_tmp_i) * pat_ii.W_0;
                            end
                                
                            % 2. Corrections
                            % Find the value of x that maximises the 
                            % posterior distribution assuming that the 
                            % patient survived the previous time point
                            g_fn_coef = struct('delta_ij', 0, 'mu_ij', fore_tmp.mu_long_corr(:,:,j), ...
                                           'Sigma_ij', fore_tmp.V_long_corr(:,:,j), 'base_cov', pat_ii.base_cov, ...
                                           'g_s', model_coef_est.g_s, 'a_s', model_coef_est.a_s, ...
                                           'tau_ij', model_coef_est.DeltaT, 'H_mat', model_coef_est.H_mat);

                            % Use Newton Raphson's iterative method to approximate
                            % the posterior as a Gaussian distribution
                            x_NR = LSDSM_MM_ALLFUNCS.Newton_Raphson(fore_tmp.mu_long_corr(:,:,j), NR_controls, @LSDSM_MM_ALLFUNCS.f_KF, ...
                                                                    @LSDSM_MM_ALLFUNCS.dfdx_KF, @LSDSM_MM_ALLFUNCS.d2fdx2_KF, g_fn_coef);

                            % find the hessian matrix
                            hess = LSDSM_MM_ALLFUNCS.d2fdx2_KF(x_NR, g_fn_coef);

                            % Update mu_corr and V_corr
                            fore_tmp.mu_corr(:,:,j) = x_NR;
                            fore_tmp.V_corr(:,:,j) = (hess)^-1;
                            fore_tmp.V_corr(:,:,j) = LSDSM_MM_ALLFUNCS.ensure_sym_mat(fore_tmp.V_corr(:,:,j));
                            % 3. Survival probabilities
                            % Use Laplace approximation to estimate the
                            % integral that leads to the probability of
                            % survival within a period of interest.
                            exponent_val = LSDSM_MM_ALLFUNCS.f_KF(fore_tmp.mu_corr(:,:,j), g_fn_coef);
                            int_val = (2*pi)^(dim_size.states/2) * det(hess)^(-1/2) * exp(-exponent_val);
                            surv_period = (2*pi)^(-dim_size.states/2) * det(fore_tmp.V_corr(:,:,j))^(-1/2) * int_val;
                            if surv_period > 1 % survival probability should never be greater than 1
                                error('Probability of survival for this period is greater than 1.');
                            end
                            % Store the survival variables
                            fore_tmp.surv_phi(:,:,j+1) = surv_period;
                            fore_tmp.survival(:,:,j+1) = fore_tmp.survival(:,:,j) * fore_tmp.surv_phi(:,:,j+1);
                        end
                    end
                    % Store the forecast arrays
                    pat_ii.forecasts{g} = fore_tmp;
                end

                % calculate the overall survival of the patient (which is a weighted average of all classes).
                for g=1:num_classes % for every class
                    % Add the contribution of class g for the survival of patient ii
                    overall_pred_surv = overall_pred_surv + E_c_ig(g) * pat_ii.forecasts{g}.survival;
                end

                % Store the survival in a (1 x horizons) vector
                pat_ii.pred_surv = squeeze(overall_pred_surv)';

                surv_fn_all(:,:,ii) = pat_ii.pred_surv;
            end
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
            
            % Removes DeltaT and G_mat from this parameter difference
            model1_coef_cmp = rmfield(model_coef1, {'DeltaT', 'G_mat'});
            model2_coef_cmp = rmfield(model_coef2, {'DeltaT', 'G_mat'});
            
            % Convert structs to vertical arrays to compare parameters:
            % of model 1
            model1_params_pre = cellfun(@(x) reshape(x,[],1), struct2cell(model1_coef_cmp), 'UniformOutput', false);
            model1_params_vec = vertcat(model1_params_pre{:});

            % of model 2
            model2_params_pre = cellfun(@(x) reshape(x,[],1), struct2cell(model2_coef_cmp), 'UniformOutput', false);
            model2_params_vec = vertcat(model2_params_pre{:});

            % find the difference
            param_diff = model1_params_vec - model2_params_vec;
            
            % find the percentage difference
            param_diff_percent = abs((model1_params_vec - model2_params_vec) ./ model1_params_vec);
            
            % In case of a division by zero - set to NaN
            param_diff_percent(isinf(param_diff_percent)) = NaN;
        end
        
        
        function [models_init] = initialise_params(dim_size, data_observed, fixed_params, Delta, H_mat, num_classes, controls)
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
            %   fixed_params - (map) Contains struct for every class to fix
            %                  certain parameters in the initialisation.
            %   Delta - (double) Time step for SSM.
            %   H_mat - (matrix) Matrix determining the linear combination
            %           of states to affect the hazard function.
            %   num_classes - (double) The number of models to initialise.
            %   controls - (struct) Controls for the EM algorithm,
            %              including number of iterations, maximum
            %              parameter difference for stopping criteria,
            %              initial parameters, fixed parameters, boolean
            %              for modified filter equations, and type of
            %              baseline hazard function.
            %
            % OUTPUT:
            %   model_coef_init - (map) Contains the initiliased
            %                     parameter values of LSDSM for the 
            %                     required number of classes.
            %
            % REVISION HISTORY:
            %   17/12/2022 - mcauchi1
            %       * Initial implementation
            %   05/01/2023 - mcauchi1
            %       * Introduced initialisation of multiple models
            %   07/02/2023 - mcauchi1
            %       * Introduced fixed_params variable that fixes the
            %       initialisations as required
            %   23/03/2023 - mcauchi1
            %       * Introduced initialisation for alternative mixture
            %       model extension
            %   01/09/2023 - mcauchi1
            %       * Introduced the H matrix to have the survival function
            %       depend on a pre-determined linear combination of
            %       states.
            %
            
            % Initialise map to hold the different models
            models_init = containers.Map('KeyType', 'int32', 'ValueType', 'any');
            
            for g=1:num_classes % for every class
                
                % store fixed parameters set
                fixed_params_tmp = fixed_params(g);
                
                % State space parameters
                A_init = fixed_params_tmp.A;
                if isnan(A_init)
                    a_bar_tmp = randn(1) * [eye(dim_size.dyn_states), ...
                                                eye(dim_size.dyn_states, dim_size.states - dim_size.dyn_states)];
                    a_bar_tmp = 0.5*rand(1) * [eye(dim_size.dyn_states), ...
                                                eye(dim_size.dyn_states, dim_size.states - dim_size.dyn_states)];
                    A_init = [a_bar_tmp; % Dynamics matrix in canonical form
                         eye(dim_size.states - dim_size.dyn_states, dim_size.states)];
                end
                
                C_init = fixed_params_tmp.C;
                if isnan(C_init)
                    C_init = [eye(dim_size.y), zeros(dim_size.y, dim_size.states - dim_size.y)]; % Observation matrix
                end
                
                W_init = fixed_params_tmp.W;
                if isnan(W_init)
                    W_init = (1)^2 * eye(size(a_bar_tmp, 1)); % Disturbance matrix
                end
                
                V_init = fixed_params_tmp.V;
                if isnan(V_init)
                    V_init = (1)^2 * eye(size(C_init,1)); % Measurement error matrix
                end
                
                G_mat = [eye(dim_size.dyn_states); % Matrix linking disturbance with the states
                         zeros(dim_size.states - dim_size.dyn_states, dim_size.dyn_states)];

                % Capture initial observation values for all patients
                y_init_mat = zeros(dim_size.y, data_observed.Count);
                for i=1:data_observed.Count
                    y_init_mat(:,i) = data_observed(i).y(:,:,1);
                end

                % Initialise initial state values based on observation data
                % lagging states have same value as initial biomarker values observed

                mu_0_init = fixed_params_tmp.mu_0;
                if isnan(mu_0_init)
                    % if number of hidden states is a multiple of number of observations
                    if mod(dim_size.states, dim_size.y) == 0 
                        mu_0_init = repmat(nanmean(y_init_mat, 2), dim_size.states / dim_size.y, 1);
                    else % if one observation has more lagging states associated with it
                        temp_mu = nanmean(y_init_mat, 2);
                        mu_0_init = [repmat(temp_mu, floor(dim_size.states / dim_size.dyn_states), 1);
                                     temp_mu(1:mod(dim_size.states, dim_size.dyn_states))];
                    end
                end

                W_0_init = fixed_params_tmp.W_0;
                if isnan(W_0_init)
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
                end

                % Survival parameters
                % coefficients linking baseline covariates with hazard function
                g_s_init = fixed_params_tmp.g_s;
                if isnan(g_s_init)
                    g_s_init = randn(dim_size.base_cov, 1);
                end
                
                % coefficients linking hidden states with hazard function
                a_s_init = fixed_params_tmp.a_s;            
                if isnan(a_s_init) % if not fixed
                    a_s_init = 0 * randn(size(H_mat,1), 1); 
                    if ~controls.mod_KF % if we are using the alternative mixture model extension
                        if strcmpi(controls.base_haz, 'Weibull')
                            % These cannot be set to negative values
                            a_s_init = 1; % Weibull shape parameter
                        end
                    end
                end
                b_s_init = 1; % Weibull scale parameter

                % Class parameters
                zeta_init = zeros(dim_size.class_cov, 1);

                % Store all model coefficients within a struct
                models_init(g) = struct('A', A_init, 'C', C_init, ...
                                         'W', W_init, 'V', V_init, ...
                                         'g_s', g_s_init, 'a_s', a_s_init, ...
                                         'DeltaT', Delta, 'G_mat', G_mat, 'H_mat', H_mat, ...
                                         'mu_0', mu_0_init, 'W_0', W_0_init, ...
                                         'zeta', zeta_init);
                                     
                if ~controls.mod_KF % if we are using the alternative mixture model extension
                    if strcmpi(controls.base_haz, 'Weibull')
                        models_init(g) = setfield(models_init(g), 'b_s', b_s_init);
                    end
                end
                
            end
        end
        
        
        function [models_init] = better_init_params(dim_size, data_observed, fixed_params, Delta, H_mat, max_censor_time, num_classes, controls)
            % FUNCTION NAME:
            %   better_init_params
            %
            % DESCRIPTION:
            %   Function to initialise the parameters of LSDSM in a smarter
            %   way. It begins by randomly initialising the parameters,
            %   followed by an update on the A matrix to obtain better
            %   hidden state trajectories. Finally, it extracts a small
            %   sample from the population, and runs the EM algorithm to
            %   obtain better initialisations for the overall EM algorithm.
            %   This is done for every class separately.
            %
            % INPUT:
            %   dim_size - (struct) Contains the dimension sizes of the
            %              latent states, dynamic states, and the number of
            %              observations and the baseline covariates.
            %   data_observed - (map) Contains the observed data of the
            %                   patients.
            %   fixed_params - (map) Contains struct for every class to fix
            %                  certain parameters in the initialisation.
            %   Delta - (double) Time step for SSM.
            %   H_mat - (matrix) Matrix determining the linear combination
            %           of states to affect the hazard function.
            %   max_censor_time - (double) Maximum period of observation.
            %   num_classes - (double) The number of models to initialise.
            %   controls - (struct) Controls for the EM algorithm,
            %              including number of iterations, maximum
            %              parameter difference for stopping criteria,
            %              initial parameters, fixed parameters, and
            %              boolean for modified filter equations
            %
            % OUTPUT:
            %   models_init - (map) Contains the initiliased parameter
            %                 values of LSDSM for the required number of
            %                 classes.
            %
            % REVISION HISTORY:
            %   22/02/2023 - mcauchi1
            %       * Initial implementation
            %   23/03/2023 - mcauchi1
            %       * Introduced survival parameters for Weibull baseline
            %       hazard function
            %   01/07/2023 - mcauchi1
            %       * Finds a better start for the A matrix for faster EM
            %       convergence
            %   01/09/2023 - mcauchi1
            %       * Introduced the H matrix to have the survival function
            %       depend on a pre-determined linear combination of
            %       states.
            %
            
            % Initialise map to hold the different models
            models_init = containers.Map('KeyType', 'int32', 'ValueType', 'any');
            
            num_pats = double(data_observed.Count); % number of patients
            
            % maximum number of time points to consider - the +1 is used in
            % case there are 21 measurements for a max obs time of 20.
            max_num_pts = ceil(max_censor_time / Delta) + 1;
            
            % Sample some of the patients, up to a maximum of 100 patients
            % for every class - extract random sample for all classes (this
            % results in sampling without replacement)
            num_red_pats = min(100, floor(num_pats/num_classes));
            rand_sample = randsample(num_pats, num_red_pats * num_classes);
            
            for g=1:num_classes % for every class
                
                % Store the fixed parameter values for this class
                fixed_params_tmp = fixed_params(g);
                
                % Start by randomly initialising the values
                
                % State space parameters
                A_init = fixed_params_tmp.A;
                if isnan(A_init) % if not fixed
                    a_bar_tmp = randn(1) * [eye(dim_size.dyn_states), ...
                                                eye(dim_size.dyn_states, dim_size.states - dim_size.dyn_states)];
                    a_bar_tmp = 0.5*rand(1) * [eye(dim_size.dyn_states), ...
                                                eye(dim_size.dyn_states, dim_size.states - dim_size.dyn_states)];
                    A_init = [a_bar_tmp; % Dynamics matrix in canonical form
                         eye(dim_size.states - dim_size.dyn_states, dim_size.states)];
                end
                
                if isfield(dim_size , 'u')
                    B_init = fixed_params_tmp.B;
                    if isnan(B_init) % if not fixed
                        B_init = zeros(dim_size.states, dim_size.u);
                    end
                end
                
                C_init = fixed_params_tmp.C;
                if isnan(C_init) % if not fixed
                    C_init = [eye(dim_size.y), zeros(dim_size.y, dim_size.states - dim_size.y)]; % Observation matrix
                end
                
                W_init = fixed_params_tmp.W;
                if isnan(W_init) % if not fixed
                    W_init = (0.01)^2 * eye(size(a_bar_tmp, 1)); % Disturbance matrix
                end
                
                V_init = fixed_params_tmp.V;
                if isnan(V_init) % if not fixed
                    V_init = (0.01)^2 * eye(size(C_init,1)); % Measurement error matrix
                end
                
                G_mat = [eye(dim_size.dyn_states); % Matrix linking disturbance with the states
                         zeros(dim_size.states - dim_size.dyn_states, dim_size.dyn_states)];

                % The initial conditions for the hidden states can be more
                % informed by observing the first values
                % Capture initial observation values for all patients
                y_arr = LSDSM_MM_ALLFUNCS.extract_field_from_map(data_observed, 'y');
                y_init_mat = reshape(y_arr(:,:,1,:), [size(y_arr,1), num_pats]);

                % Initialise initial state values based on observation data
                % lagging states have same value as initial biomarker
                % values observed
                mu_0_init = fixed_params_tmp.mu_0;
                if isnan(mu_0_init) % if not fixed
                    % if number of hidden states is a multiple of number of observations
                    if mod(dim_size.states, dim_size.y) == 0 
                        mu_0_init = repmat(nanmean(y_init_mat, 2), dim_size.states / dim_size.y, 1);
                    else % if one observation has more lagging states associated with it
                        temp_mu = nanmean(y_init_mat, 2);
                        mu_0_init = [repmat(temp_mu, floor(dim_size.states / dim_size.dyn_states), 1);
                                     temp_mu(1:mod(dim_size.states, dim_size.dyn_states))];
                    end
                end

                W_0_init = fixed_params_tmp.W_0;
                if isnan(W_0_init) % if not fixed
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
                end

                % Survival parameters
                % coefficients linking baseline covariates with hazard function
                g_s_init = fixed_params_tmp.g_s;
                if isnan(g_s_init) % if not fixed
                    g_s_init = randn(dim_size.base_cov, 1);
                end
                
                % coefficients linking hidden states with hazard function
                a_s_init = fixed_params_tmp.a_s;
                if isnan(a_s_init) % if not fixed
                    a_s_init = 0 * randn(size(H_mat,1), 1); 
                    if ~controls.mod_KF % if we are using the alternative mixture model extension
                        if strcmpi(controls.base_haz, 'Weibull')
                            % These cannot be set to negative values
                            a_s_init = 1;
                        end
                    end
                end
                b_s_init = 1;

                % Class parameters
                zeta_init = zeros(dim_size.class_cov, 1);

                % Store all model coefficients within a struct
                models_init(g) = struct('A', A_init, 'C', C_init, ...
                                        'W', W_init, 'V', V_init, ...
                                        'g_s', g_s_init, 'a_s', a_s_init, ...
                                        'DeltaT', Delta, 'G_mat', G_mat, 'H_mat', H_mat, ...
                                        'mu_0', mu_0_init, 'W_0', W_0_init, ...
                                        'zeta', zeta_init);
            
                if isfield(dim_size, 'u')
                    models_init(g) = setfield(models_init(g), 'B', B_init);
                end
                if ~controls.mod_KF % if we are using the alternative mixture model extension
                    if strcmpi(controls.base_haz, 'Weibull')
                        models_init(g) = setfield(models_init(g), 'b_s', b_s_init);
                    end
                end
                                     
                % Create the sampled population
                rand_sample_class = rand_sample((g-1)*num_red_pats+1:g*num_red_pats);
                red_keys = 1:num_red_pats;
                red_values = data_observed.values;
                red_values = red_values(rand_sample_class);

                red_data_observed = containers.Map(red_keys, red_values);
                
                % Update the A matrix for a single iteration - a bad
                % initialisation of A may result in a less optimal solution
                % and may take very long to converge
                if isnan(fixed_params_tmp.A) % if not fixed
                    
                    for j=1:100 % improve A until no further improvements can be made with the same parameters
                        % auxiliary variables for updating A
                        E_sums.barxn_xnneg1_from3 = zeros(dim_size.dyn_states,dim_size.states);
                        E_sums.xn_xn_from2_tillNneg1 = zeros(dim_size.states,dim_size.states);

                        % Set coefficients for the g-th model
                        model_coef_est = models_init(g);

                        for ii=1:num_red_pats % iterate for every patient in the reduced population
                            % Capture information from current patient
                            pat_ii = red_data_observed(ii);

                            % Set initial state conditions for patient ii
                            pat_ii.mu_0 = model_coef_est.mu_0;
                            pat_ii.W_0 = model_coef_est.W_0;

                            %%% Forward recursion - Standard/Modified RTS Filter %%%
                            [pat_ii.mu_tilde, pat_ii.V_tilde, log_like_val] = ...
                                LSDSM_MM_ALLFUNCS.Kalman_filter(pat_ii, model_coef_est, max_num_pts, controls);

                            %%% Backward recursion - RTS Smoother %%%
                            [pat_ii.mu_hat, pat_ii.V_hat, pat_ii.J_hat] = ...
                                LSDSM_MM_ALLFUNCS.Kalman_smoother(pat_ii, model_coef_est, max_num_pts);

                            %%% Compute the required Expectations %%%
                            [pat_ii.E] = LSDSM_MM_ALLFUNCS.compute_E_fns(pat_ii, model_coef_est, max_num_pts);

                            % Sum the required expectations for the update equation of A
                            E_sums.barxn_xnneg1_from3 = E_sums.barxn_xnneg1_from3 + sum(pat_ii.E.barxn_xnneg1(:,:,3:pat_ii.m_i), 3);
                            E_sums.xn_xn_from2_tillNneg1 = E_sums.xn_xn_from2_tillNneg1 + sum(pat_ii.E.xn_xn(:,:,2:pat_ii.m_i-1), 3);
                        end

                        % Update A
                        A_bar_new = E_sums.barxn_xnneg1_from3 * E_sums.xn_xn_from2_tillNneg1^-1;
                        A_new = model_coef_est.A;
                        A_new(1:dim_size.dyn_states,:) = A_bar_new;
                        models_init(g) = setfield(models_init(g), 'A', A_new);
                        
                        A_diff = A_new(:) - model_coef_est.A(:);
                        
                        if abs(A_diff) < 1e-4
                            break;
                        end
                    end

                end
                
                % Run the EM algorithm for a single class using this
                % reduced population with quicker to reach convergence
                % criteria
                if controls.do_EM_better_init_params
                    controls.init_params = containers.Map(1, models_init(g));
                    controls.max_param_diff = 1e-2;
                    controls.fixed_params = containers.Map(1, fixed_params(g));
                    controls.EM_iters = 200;
                    [model_coef_est_tmp, ~] = LSDSM_MM_ALLFUNCS.LSDSM_MM_EM(dim_size, red_data_observed, controls, max_censor_time);

                    % Store the more informed initialised parameters
                    models_init(g) = model_coef_est_tmp(1);
                end
            end
        end
        
        
        function ensure_correct_dims(pat_data, dim_size)
            % FUNCTION NAME:
            %   ensure_correct_dims
            %
            % DESCRIPTION:
            %   Checks that the dimensions of the dim_size variable are
            %   properly set. Will stop the program run if they are not.
            %
            % INPUT:
            %   pat_data - (map) Observed data of all patients including
            %              longitudinal biomarkers, survival time, and 
            %              event indicator.
            %   dim_size - (struct) Contains dimension sizes for variables
            %              and model parameters.
            % 
            % REVISION HISTORY:
            %   07/02/2023 - mcauchi1
            %       * Initial implementation
            %   15/03/2023 - mcauchi1
            %       * Introduced assertion for input dimensions
            %
            
            pat_ii = pat_data(1); % extract data from the first patient
            
            assert(size(pat_ii.y,1) == dim_size.y, ...
                            'Dimensions of y are not in agreement between observations and dim_size\n');
            assert(size(pat_ii.base_cov,1) == dim_size.base_cov, ...
                            'Dimensions of base_cov are not in agreement between observations and dim_size\n');
            assert(size(pat_ii.class_cov,1) == dim_size.class_cov, ...
                            'Dimensions of class_cov are not in agreement between observations and dim_size\n');
            
            if isfield(pat_ii, 'u')
                assert(size(pat_ii.u,1) == dim_size.u, ...
                            'Dimensions of u are not in agreement between observations and dim_size\n');
            end
        end
        
        
        function ensure_correct_param_dims(all_model_params, dim_size, controls)
            % FUNCTION NAME:
            %   ensure_correct_param_dims
            %
            % DESCRIPTION:
            %   Checks that the dimensions of the model parameters are
            %   correct. Will stop the program run if they are not.
            %
            % INPUT:
            %   all_model_params - (map) Model parameters for all classes.
            %   dim_size - (struct) Contains dimension sizes for variables
            %              and model parameters.
            %   controls - (struct) Contains the controls for the LSDSM EM,
            %              including the boolean to indicate if survival
            %              data should affect hidden states.
            % 
            % REVISION HISTORY:
            %   07/02/2023 - mcauchi1
            %       * Initial implementation
            %   15/03/2023 - mcauchi1
            %       * Introduced parameter check for input matrix
            %   23/03/2023 - mcauchi1
            %       * Introduced parameter check for Weibull baseline
            %       hazard function parameters
            %
            
            assert(dim_size.states >= dim_size.dyn_states, ...
                    'Number of dynamic states should not be greater than number of states');
            
            num_classes = double(all_model_params.Count); % count the number of classes
            
            for g=1:num_classes % for every class
                model_g = all_model_params(g);
                
                % State space parameters
                assert(isequal(size(model_g.A,1), size(model_g.A,2), dim_size.states), ...
                    'A should be (%d x %d) matrix, not (%s)\n', dim_size.states, dim_size.states, ...
                                                                regexprep(num2str(size(model_g.A)), ' +', ' x '));
                                                            
                if isfield(model_g, 'B')
                    assert(isequal(size(model_g.B), [dim_size.states, dim_size.u]), ...
                        'B should be (%d x %d) matrix, not (%s)\n', dim_size.states, dim_size.u, ...
                                                                    regexprep(num2str(size(model_g.B)), ' +', ' x '));
                end
                                                                     
                assert(isequal(size(model_g.C), [dim_size.y, dim_size.states]), ...
                    'C should be (%d x %d) matrix, not (%s)\n', dim_size.y, dim_size.states, ...
                                                                regexprep(num2str(size(model_g.C)), ' +', ' x '));
                
                assert(isequal(size(model_g.W,1), size(model_g.W,2), dim_size.dyn_states), ...
                    'W should be (%d x %d) matrix, not (%s)\n', dim_size.dyn_states, dim_size.dyn_states, ...
                                                                regexprep(num2str(size(model_g.W)), ' +', ' x '));
                assert(isequal(size(model_g.V,1), size(model_g.V,2), dim_size.y), ...
                    'V should be (%d x %d) matrix, not (%s)\n', dim_size.y, dim_size.y, ...
                                                                regexprep(num2str(size(model_g.V)), ' +', ' x '));
                
                assert(isequal(size(model_g.mu_0), [dim_size.states, 1]), ...
                    'mu_0 should be (%d x %d) matrix, not (%s)\n', dim_size.states, 1, ...
                                                                   regexprep(num2str(size(model_g.mu_0)), ' +', ' x '));
                assert(isequal(size(model_g.W_0,1), size(model_g.W_0,2), dim_size.states), ...
                    'W_0 should be (%d x %d) matrix, not (%s)\n', dim_size.states, dim_size.states, ...
                                                                  regexprep(num2str(size(model_g.W_0)), ' +', ' x '));
                            
                % Survival parameters
                assert(isequal(size(model_g.g_s), [dim_size.base_cov,1]), ...
                    'g_s should be (%d x %d) matrix, not (%s)\n', dim_size.base_cov, 1, ...
                                                                  regexprep(num2str(size(model_g.g_s)), ' +', ' x '));
                                                              
                if controls.mod_KF
                    assert(isequal(size(model_g.a_s), [dim_size.alpha_eqns,1]), ...
                        'a_s should be (%d x %d) matrix, not (%s)\n', dim_size.alpha_eqns, 1, ...
                                                                      regexprep(num2str(size(model_g.a_s)), ' +', ' x '));
                else % if we are using the alternative mixture model extension
                    if strcmpi(controls.base_haz, 'Weibull')
                        assert(isequal(size(model_g.a_s), [1,1]), ...
                            'a_s should be a scalar, not a (%s) matrix \n', regexprep(num2str(size(model_g.a_s)), ' +', ' x '));
                        assert(isequal(size(model_g.b_s), [1,1]), ...
                            'b_s should be a scalar, not a (%s) matrix \n', regexprep(num2str(size(model_g.b_s)), ' +', ' x '));
                    end
                end
                                                              
                % Class parameters
                assert(isequal(size(model_g.zeta), [dim_size.class_cov,1]), ...
                    'zeta should be (%d x %d) matrix, not (%s)\n', dim_size.class_cov, 1, ...
                                                                   regexprep(num2str(size(model_g.zeta)), ' +', ' x '));
            end
        end
        
        
        function ensure_correct_fixed_param_dims(all_fixed_params, dim_size, controls)
            % FUNCTION NAME:
            %   ensure_correct_fixed_param_dims
            %
            % DESCRIPTION:
            %   Checks that the dimensions of the user-set fixed model
            %   parameters are correct. Will stop the program run if they
            %   are not.
            %
            % INPUT:
            %   all_fixed_params - (map) User-set model parameters for all 
            %                      classes.
            %   dim_size - (struct) Contains dimension sizes for variables
            %              and model parameters.
            %   controls - (struct) Contains the controls for the LSDSM EM,
            %              including the boolean to indicate if survival
            %              data should affect hidden states.
            % 
            % REVISION HISTORY:
            %   07/02/2023 - mcauchi1
            %       * Initial implementation
            %   15/03/2023 - mcauchi1
            %       * Introduced parameter check for input matrix
            %   23/03/2023 - mcauchi1
            %       * Introduced parameter check for Weibull baseline
            %       hazard function
            %
            
            num_classes = double(all_fixed_params.Count); % number of classes
            
            for g=1:num_classes % for every class
                fixed_g = all_fixed_params(g); % store the fixed parameters for current class
                
                if not(isnan(fixed_g.A))
                    assert(isequal(size(fixed_g.A,1), size(fixed_g.A,2), dim_size.states), ...
                        'A should be (%d x %d) matrix, not (%s) (Fixed Parameters)\n', dim_size.states, dim_size.states, ...
                                                                    regexprep(num2str(size(fixed_g.A)), ' +', ' x '));
                end
                
                if isfield(fixed_g, 'B') && not(isnan(fixed_g.B))
                    assert(isequal(size(fixed_g.B), [dim_size.states, dim_size.u]), ...
                        'B should be (%d x %d) matrix, not (%s) (Fixed Parameters)\n', dim_size.states, dim_size.u, ...
                                                                    regexprep(num2str(size(model_g.B)), ' +', ' x '));
                end
                
                if not(isnan(fixed_g.C))
                    assert(isequal(size(fixed_g.C), [dim_size.y, dim_size.states]), ...
                        'C should be (%d x %d) matrix, not (%s) (Fixed Parameters)\n', dim_size.y, dim_size.states, ...
                                                                    regexprep(num2str(size(fixed_g.C)), ' +', ' x '));
                end
                
                if not(isnan(fixed_g.W))
                    assert(isequal(size(fixed_g.W,1), size(fixed_g.W,2), dim_size.dyn_states), ...
                        'W should be (%d x %d) matrix, not (%s) (Fixed Parameters)\n', dim_size.dyn_states, dim_size.dyn_states, ...
                                                                    regexprep(num2str(size(fixed_g.W)), ' +', ' x '));
                end
                
                if not(isnan(fixed_g.V))
                    assert(isequal(size(fixed_g.V,1), size(fixed_g.V,2), dim_size.y), ...
                        'V should be (%d x %d) matrix, not (%s) (Fixed Parameters)\n', dim_size.y, dim_size.y, ...
                                                                    regexprep(num2str(size(fixed_g.V)), ' +', ' x '));
                end
                
                if not(isnan(fixed_g.mu_0))
                    assert(isequal(size(fixed_g.mu_0), [dim_size.states, 1]), ...
                        'mu_0 should be (%d x %d) matrix, not (%s) (Fixed Parameters)\n', dim_size.states, 1, ...
                                                                       regexprep(num2str(size(fixed_g.mu_0)), ' +', ' x '));
                end
                
                if not(isnan(fixed_g.W_0))
                    assert(isequal(size(fixed_g.W_0,1), size(fixed_g.W_0,2), dim_size.states), ...
                        'W_0 should be (%d x %d) matrix, not (%s) (Fixed Parameters)\n', dim_size.states, dim_size.states, ...
                                                                      regexprep(num2str(size(fixed_g.W_0)), ' +', ' x '));
                end
                
                if not(isnan(fixed_g.g_s))
                    assert(isequal(size(fixed_g.g_s), [dim_size.base_cov,1]), ...
                        'g_s should be (%d x %d) matrix, not (%s) (Fixed Parameters)\n', dim_size.base_cov, 1, ...
                                                                      regexprep(num2str(size(fixed_g.g_s)), ' +', ' x '));
                end
                
                if not(isnan(fixed_g.a_s))
                    if controls.mod_KF
                        assert(isequal(size(fixed_g.a_s), [dim_size.alpha_eqns,1]), ...
                            'a_s should be (%d x %d) matrix, not (%s) (Fixed Parameters)\n', dim_size.alpha_eqns, 1, ...
                                                                          regexprep(num2str(size(fixed_g.a_s)), ' +', ' x '));
                    else % if we are using the alternative mixture model extension
                        if strcmpi(controls.base_haz, 'Weibull')
                            assert(isequal(size(fixed_g.a_s), [1,1]), ...
                                'a_s should be a scalar, not a (%s) matrix \n', regexprep(num2str(size(fixed_g.a_s)), ' +', ' x '));
                            assert(isequal(size(fixed_g.b_s), [1,1]), ...
                                'b_s should be a scalar, not a (%s) matrix \n', regexprep(num2str(size(fixed_g.b_s)), ' +', ' x '));
                        end
                    end
                end
                
            end
        end
        
        
        function ensure_same_params_correctness(same_params)
            % FUNCTION NAME:
            %   ensure_same_params_correctness
            %
            % DESCRIPTION:
            %   When utilising same parameters across all classes (e.g.
            %   same dynamics matrix A across all classes), certain
            %   parameters have to be kept the same across all classes (in
            %   the same example, W has to be the same across all classes).
            %
            % INPUT:
            %   same_params - (struct) Binary variables indicating whether
            %                 field is the same across all classes or not.
            % 
            % REVISION HISTORY:
            %   07/02/2023 - mcauchi1
            %       * Initial implementation
            %   15/03/2023 - mcauchi1
            %       * Introduced check to ensure that A and B are kept the
            %       same across classes if one of them is forced to stay
            %       the same
            %
            
            if same_params.A && not(same_params.W) % if same A but not same W
                error('Illegal Case: When forcing A to be the same across all classes, W is also required to be the same');
            end
            if isfield(same_params, 'B') % if there is input to the SSM
                if same_params.A ~= same_params.B % if same A but not same W
                    error('Illegal Case: When forcing A to be the same across all classes, B is also required to be the same');
                end
            end
            if same_params.C && not(same_params.V) % if same C but not same V
                error('Illegal Case: When forcing C to be the same across all classes, V is also required to be the same');
            end
            if same_params.mu_0 && not(same_params.W_0) % if same mu_0 but not same W_0
                error('Illegal Case: When forcing mu_0 to be the same across all classes, W_0 is also required to be the same');
            end
        end
        
        
        function [param_traj] = initialise_param_trajectories(model_coef_tmp, dim_size, controls)
            % FUNCTION NAME:
            %   initialise_param_trajectories
            %
            % DESCRIPTION:
            %   Initialisation of the parameter trajectories over EM
            %   iterations that can be used to visualise the evolution of
            %   the estimated parameters.
            %
            % INPUT:
            %   model_coef_tmp - (map) Contains current model estimates for
            %                    all classes.
            %   dim_size - (struct) Contains dimension sizes for variables
            %              and model parameters.
            %   controls - (struct) Controls for the EM algorithm,
            %              including number of iterations.
            %
            % OUTPUT:
            %   param_traj - (map) Containing placeholders for parameter
            %                estimates for all classes evolving over EM
            %                iterations.
            %
            % REVISION HISTORY:
            %   22/02/2023 - mcauchi1
            %       * Initial implementation
            %   15/03/2023 - mcauchi1
            %       * Introduced the trajectory for the input matrix (B)
            %
            
            num_classes = double(model_coef_tmp.Count);
            
            % Initialise map to track parameter changes for all classes
            param_traj = containers.Map('KeyType', 'int32', 'ValueType', 'any');
            
            % Initialise all parameter arrays with the last dimension
            % representing the EM iterations
            traj_struct.A = zeros(dim_size.states, dim_size.states, controls.EM_iters);
            traj_struct.C = zeros(dim_size.y, dim_size.states, controls.EM_iters);
            traj_struct.W = zeros(dim_size.dyn_states, dim_size.dyn_states, controls.EM_iters);
            traj_struct.V = zeros(dim_size.y, dim_size.y, controls.EM_iters);
            traj_struct.g_s = zeros(dim_size.base_cov, 1, controls.EM_iters);
            if controls.mod_KF
                traj_struct.a_s = zeros(size(model_coef_tmp(1).H_mat,1), 1, controls.EM_iters);
            else
                if strcmpi(controls.base_haz, 'Weibull')
                    traj_struct.a_s = zeros(1, 1, controls.EM_iters);
                    traj_struct.b_s = zeros(1, 1, controls.EM_iters);
                end
            end
            traj_struct.mu_0 = zeros(dim_size.states, 1, controls.EM_iters);
            traj_struct.W_0 = zeros(dim_size.states, dim_size.states, controls.EM_iters);
            traj_struct.zeta = zeros(dim_size.class_cov, 1, controls.EM_iters);
            
            if isfield(dim_size, 'u')
                traj_struct.B = zeros(dim_size.states, dim_size.u, controls.EM_iters);
            end
            
            for g=1:num_classes % for every class
                % Set the first value (in iter_EM) as the initial estimates
                traj_struct.A(:,:,1) = model_coef_tmp(g).A;
                traj_struct.C(:,:,1) = model_coef_tmp(g).C;
                traj_struct.W(:,:,1) = model_coef_tmp(g).W;
                traj_struct.V(:,:,1) = model_coef_tmp(g).V;
                traj_struct.g_s(:,:,1) = model_coef_tmp(g).g_s;
                traj_struct.a_s(:,:,1) = model_coef_tmp(g).a_s;
                if ~controls.mod_KF
                    if strcmpi(controls.base_haz, 'Weibull')
                        traj_struct.b_s(:,:,1) = model_coef_tmp(g).b_s;
                    end
                end
                traj_struct.mu_0(:,:,1) = model_coef_tmp(g).mu_0;
                traj_struct.W_0(:,:,1) = model_coef_tmp(g).W_0;
                traj_struct.zeta(:,:,1) = model_coef_tmp(g).zeta;
                
                if isfield(dim_size, 'u')
                    traj_struct.B(:,:,1) = model_coef_tmp(g).B;
                end
                
                param_traj(g) = traj_struct; % store into map entry
            end
        end
        
        
        function [param_traj] = update_param_trajectories(model_coef_new_all, param_traj, curr_iter)
            % FUNCTION NAME:
            %   update_param_trajectories
            %
            % DESCRIPTION:
            %   Updates the parameter trajectories over EM iterations to be
            %   able to visualise the evolution of the estimated
            %   parameters.
            %
            % INPUT:
            %   model_coef_new_all - (map) Contains the new model estimates
            %                        for all classes.
            %   param_traj - (map) Containing placeholders for parameter
            %                estimates for all classes evolving over EM
            %                iterations (same as output).
            %   curr_iter - (double) Current EM iteration.
            %
            % OUTPUT:
            %   param_traj - (map) Containing placeholders for parameter
            %                estimates for all classes evolving over EM
            %                iterations.
            %
            % REVISION HISTORY:
            %   23/03/2023 - mcauchi1
            %       * Initial implementation
            %
            
            num_classes = double(model_coef_new_all.Count); % number of classes
            
            for g=1:num_classes % for every class
                % Store the updated estimates in the respective arrays
                traj_struct = param_traj(g);
                model_coef_new = model_coef_new_all(g);

                traj_struct.A(:,:,curr_iter) = model_coef_new.A;
                traj_struct.C(:,:,curr_iter) = model_coef_new.C;
                traj_struct.W(:,:,curr_iter) = model_coef_new.W;
                traj_struct.V(:,:,curr_iter) = model_coef_new.V;
                traj_struct.g_s(:,:,curr_iter) = model_coef_new.g_s;
                traj_struct.a_s(:,:,curr_iter) = model_coef_new.a_s;
                traj_struct.mu_0(:,:,curr_iter) = model_coef_new.mu_0;
                traj_struct.W_0(:,:,curr_iter) = model_coef_new.W_0;
                traj_struct.zeta(:,:,curr_iter) = model_coef_new.zeta;
                
                if isfield(model_coef_new, 'B')
                    traj_struct.B(:,:,curr_iter) = model_coef_new.B;
                end
                if isfield(model_coef_new, 'b_s')
                    traj_struct.b_s(:,:,curr_iter) = model_coef_new.b_s;
                end

                % store the updated trajectories
                param_traj(g) = traj_struct;
            end
        end
        
        
        function [RTS_arrs] = initialise_RTS_arrs(num_classes, dim_size, max_num_pts, num_pats)
            % FUNCTION NAME:
            %   initialise_RTS_arrs
            %
            % DESCRIPTION:
            %   Initialisation of RTS arrays that will contain the filtered
            %   and smoothed trajectories of the hidden states.
            %   
            % INPUT:
            %   num_classes - (double) The number of classes considered.
            %   dim_size - (struct) Contains dimension sizes for variables
            %              and model parameters.
            %   max_num_pts - (double) The maximum number of observations
            %                 that can be made for every patient.
            %   num_pats - (double) The number of patients in the
            %              population.
            %
            % OUTPUT:
            %   RTS_arrs - (cell) A cell array for every class containing
            %              a struct holding the filtered and smoothed
            %              hidden state trajectories.
            %
            % REVISION HISTORY:
            %   22/02/2023 - mcauchi1
            %       * Initial implementation
            %
            
            % Initialise cells to retain RTS arrs for every class
            RTS_arrs = cell(num_classes, 1);
            
            % RTS_tmp are used to store the states obtained from the RTS 
            % filter/smoother - initialise at zero
            RTS_tmp.mu_tilde = zeros(dim_size.states, 1, max_num_pts, num_pats);
            RTS_tmp.V_tilde = zeros(dim_size.states, dim_size.states, max_num_pts, num_pats);
            RTS_tmp.mu_hat = zeros(dim_size.states, 1, max_num_pts, num_pats);
            RTS_tmp.V_hat = zeros(dim_size.states, dim_size.states, max_num_pts, num_pats);
            
            for g=1:num_classes % populate RTS_arrs for every class
                RTS_arrs{g} = RTS_tmp;
            end
        end
        
        
        function [E_sums] = initialise_E_sums(num_classes, dim_size)
            % FUNCTION NAME:
            %   initialise_E_sums
            %
            % DESCRIPTION:
            %   Initialisation of summations to be used at the E and M
            %   steps. E_sums is a map containing structs containing all
            %   summations of the expectations required. 
            %   Notation: 
            %   -> n - longitudinal iterations 
            %   -> N - number of longitudinal measurements across time
            %   
            % INPUT:
            %   num_classes - (double) The number of classes considered.
            %   dim_size - (struct) Contains dimension sizes for variables
            %              and model parameters.
            %
            % OUTPUT:
            %   E_sums - (cell) A cell array for every class containing a
            %            struct of zero arrays for every required
            %            expectation in the update step.
            %
            % REVISION HISTORY:
            %   22/02/2023 - mcauchi1
            %       * Initial implementation
            %   15/03/2023 - mcauchi1
            %       * Initialised new expectations required with the
            %       inclusion of the input (if applicable)
            %
            
            % Initialise the cell array
            E_sums = cell(num_classes, 1);

            % Stores the sum across all patients of sum_{n=2}^{N} E[x(n) x(n-1)']
            E_sums_struct.xn_xnneg1_from2 = zeros(dim_size.states,dim_size.states);
            % Stores the sum across all patients of sum_{n=1}^{N-1} E[x(n) x(n-1)']
            E_sums_struct.xn_xn_tillNneg1 = zeros(dim_size.states,dim_size.states);
            % Stores the sum across all patients of sum_{n=2}^{N} E[x(n) x(n)']
            E_sums_struct.xn_xn_from2 = zeros(dim_size.states,dim_size.states);
            % Stores the sum across all patients of sum_{n=1}^{N} E[x(n) x(n)']
            E_sums_struct.xn_xn = zeros(dim_size.states,dim_size.states);
            % Stores the sum across all patients of sum_{n=1}^{N} E[x(n)]
            E_sums_struct.xn = zeros(dim_size.states,1);
            % Stores the sum across all patients of sum_{n=1}^{N} y(n) E[x(n)']
            E_sums_struct.yn_xn = zeros(dim_size.y,dim_size.states);
            % Stores the sum across all patients of sum_{n=1}^{N} y(n) y(n)'
            E_sums_struct.yn_yn = zeros(dim_size.y, dim_size.y);

            % Stores the sum across all patients of sum_{n=3}^{N} E[x_bar(n) x(n-1)']
            E_sums_struct.barxn_xnneg1_from3 = zeros(dim_size.dyn_states,dim_size.states);
            % Stores the sum across all patients of sum_{n=2}^{N-1} E[x(n) x(n)']
            E_sums_struct.xn_xn_from2_tillNneg1 = zeros(dim_size.states,dim_size.states);
            
            if isfield(dim_size, 'u') % if there is input affecting SSM
                % Stores the sum across all patients of sum_{n=3}^{N} E[x_bar(n) hat(xu)(n-1)']
                E_sums_struct.barxn_hatxunneg1_from3 = zeros(dim_size.dyn_states,dim_size.u);
                % Stores the sum across all patients of sum_{n=2}^{N-1} E[x(n) x(n)']
                E_sums_struct.hatxun_hatxun_from2_tillNneg1 = zeros(dim_size.states+dim_size.u,dim_size.states+dim_size.u);

                % Stores the sum across all patients of sum_{n=2}^{N} E[x_bar(n)] u(n-1)'
                E_sums_struct.barxn_unneg1_from2 = zeros(dim_size.dyn_states,dim_size.u);
                % Stores the sum across all patients of sum_{n=1}^{N-1} E[x(n)] u(n)'
                E_sums_struct.xn_un_tillNneg1 = zeros(dim_size.states,dim_size.u);
                % Stores the sum across all patients of sum_{n=1}^{N-1} u(n) u(n)'
                E_sums_struct.un_un_tillNneg1 = zeros(dim_size.u,dim_size.u);
            end

            % Stores the sum across all patients of sum_{n=2}^{N} E[x_bar(n) x(n-1)']
            E_sums_struct.barxn_xnneg1_from2 = zeros(dim_size.dyn_states,dim_size.states);
            % Stores the sum across all patients of sum_{n=2}^{N} E[x_bar(n) x_bar(n)']
            E_sums_struct.barxn_barxn_from2 = zeros(dim_size.dyn_states,dim_size.dyn_states);

            % Stores the sum across all patients of E[x(1)]
            E_sums_struct.x0 = zeros(dim_size.states, 1);
            % Stores the sum across all patients of E[x(1) x(1)']
            E_sums_struct.x0_x0 = zeros(dim_size.states, dim_size.states);
            
            % To be used when the dependency from the initial observation
            % and state is removed from estimation of dynamics matrix A
            E_sums_struct.barxn_barxn_from3 = zeros(dim_size.dyn_states,dim_size.dyn_states);
            E_sums_struct.yn_yn_from2 = zeros(dim_size.y, dim_size.y);
            E_sums_struct.yn_xn_from2 = zeros(dim_size.y,dim_size.states);

            for g=1:num_classes % populate the E_sums cell for all classes
                E_sums{g} = E_sums_struct;
            end
        end
        
        
        function fig_cells = initialise_dynamic_param_plots(num_classes)
            % FUNCTION NAME:
            %   initialise_dynamic_param_plots
            %
            % DESCRIPTION:
            %   Initialisation of the dynamic plots that will keep track of
            %   the log likelihood, the class membership for different
            %   classes, and the parameter trajectories, all across EM
            %   iterations.
            %
            % INPUT:
            %   num_classes - (double) Number of classes considered.
            %
            % OUTPUT:
            %   fig_cells - (cell) A cell array to retain the figures
            %               placeholders.
            %
            % REVISION HISTORY:
            %   23/03/2023 - mcauchi1
            %       * Initial implementation
            %
            
            % Placeholder for all EM figures
            fig_cells = {};
            
            % Log likelihood figure
            fig_cells{1} = figure;
            plot([0]);
            title('Log likelihood over EM iterations');
            ylabel('Log likelihood');
            xlabel('EM iteration');
            
            % Class membership percentage figure
            fig_cells{2} = figure;
            plot([0]);
            title('Class membership over EM iterations');
            ylabel('Class membership probability');
            xlabel('EM iteration');
            
            % Figure for every class for parameter estimates
            for g=1:num_classes
                fig_cells{g+2} = figure;
                plot([0]);
                title(sprintf('Parameter estimates over EM iterations for class %d', g));
                ylabel('Parameter values');
                xlabel('EM iteration');
            end
        end
        
        
        function update_dynamic_param_plots(EM_j, fig_cells, model_coef_est, log_like_val_tot_arr, E_c_ig_EM, param_traj)
            % FUNCTION NAME:
            %   update_dynamic_param_plots
            %
            % DESCRIPTION:
            %   Updates the dynamic plots of the log likelihood, the class
            %   membership for different classes, and the parameter
            %   trajectories, all across EM iterations.
            %   
            % INPUT:
            %   EM_j - (double) Current EM iteration.
            %   fig_cells - (cell) A cell array to retain the figures
            %               placeholders.
            %   model_coef_est - (struct) the parameter estimates of one of
            %                    the classes to be used to extract
            %                    parameter sizes.
            %   log_like_val_tot_arr - (array) Contains the log likelihood
            %                          contribution by every patient at
            %                          every EM iteration.
            %   E_c_ig_EM - (array) Contains the sum of responsibilities
            %               across all patients for every class and EM
            %               iteration.
            %   param_traj - (map) Containing parameter estimates for all
            %                classes evolving over EM iterations.
            %
            % REVISION HISTORY:
            %   23/03/2023 - mcauchi1
            %       * Initial implementation
            %
            
            num_pats = size(log_like_val_tot_arr, 1); % number of patients
            num_classes = double(param_traj.Count); % number of classes
            
            % Plot the log likelihood value across every EM iteration
            figure(fig_cells{1});
            cla
            plot(2:EM_j, sum(log_like_val_tot_arr(:,2:EM_j), 1));
            grid on;

            % Plot the total responsibility of every class across every EM
            % iteration
            figure(fig_cells{2});
            cla
            % Divide E[c_ig] by the number of patients to provide fractions
            % (that sum to 1) rather than absolute values
            plot(2:EM_j, E_c_ig_EM(:,2:EM_j)' / double(num_pats));
            grid on;
            
            % Define a set of visually distinct colours
            new_colours = {'#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#42d4f4', '#f032e6', ...
                           '#9A6324', '#800000', '#000075', '#a9a9a9', '#ffffff', '#000000'};

            for g=1:num_classes % for every class
                % Plot the parameter estimates of that class for every EM
                % iteration
                figure(fig_cells{g+2});
                cla
                
                % Get all parameter names
                fn = fieldnames(param_traj(g));
                hold on;
                for k=1:numel(fn) % for every parameter
                    % plot all of its values (if it is a vector/matrix, it 
                    % will contain multiple values) with the same colour
                    % and legend name.
                    curr_param_tmp = param_traj(g).(fn{k});
                    plot(1:EM_j, reshape(curr_param_tmp(:,:,1:EM_j), [size(model_coef_est.(fn{k})(:), 1), EM_j])', ...
                    'DisplayName', fn{k}, 'Color', new_colours{k});
                end
                grid on;
                % Creates a legend with unique fields (does not repeat for
                % every multipled value of a parameter (e.g. matrix)
                legend(legendUnq(fig_cells{g+2}));
            end
            % Forces the diagrams to be updated
            drawnow
        end
        
        
        function k = calc_num_params(dim_size, controls)
            % FUNCTION NAME:
            %   calc_num_params
            %
            % DESCRIPTION:
            %   Calculates the number of parameters to be estimated for the
            %   proposed configuration.
            %
            % INPUT:
            %   dim_size - (struct) Contains dimension sizes for variables
            %              and model parameters.
            %   controls - (struct) Controls for the EM algorithm,
            %              including fixed and same parameters sets.
            %
            % OUTPUT:
            %   k - (double) Total number of parameters
            %
            % REVISION HISTORY:
            %   01/04/2023 - mcauchi1
            %       * Initial implementation
            %   01/09/2023 - mcauchi1
            %       * Introduced the H matrix to have the survival function
            %       depend on a pre-determined linear combination of
            %       states.
            %
            
            k = 0; % initialise number of parameters
            
            m = dim_size.states / dim_size.y; % autoregressive order
            m_y = dim_size.y; % number of measurements
            m_x = m * m_y; % number of states - should be equal to dim_size.states
            if isfield(dim_size, 'u')
                m_u = dim_size.u; % number of inputs
            end
            m_g = dim_size.base_cov; % number of baseline covariates
            
            % number of classes
            num_classes = double(controls.fixed_params.Count);
            
            for g=1:num_classes % for every class
                % store the fixed parameters
                curr_fixed_params = controls.fixed_params(g);
                
                if isnan(curr_fixed_params.A) % if not fixed
                    % if first class or not the same parameters across all
                    % classes
                    if not(g>1 && controls.same_params.A)
                        k = k + (m_y * m_x);
                    end
                end
                
                if isfield(dim_size, 'u') && isnan(curr_fixed_params.B) % if not fixed
                    % if first class or not the same parameters across all
                    % classes
                    if not(g>1 && controls.same_params.B)
                        k = k + (m_y * m_u);
                    end
                end
                
                if isnan(curr_fixed_params.C) % if not fixed
                    % if first class or not the same parameters across all
                    % classes
                    if not(g>1 && controls.same_params.C)
                        k = k + (m_y * m_x);
                    end
                end
                
                if isnan(curr_fixed_params.W) % if not fixed
                    % if first class or not the same parameters across all
                    % classes
                    if not(g>1 && controls.same_params.W)
                        k = k + (m_y * (m_y + 1) / 2);
                    end
                end
                
                if isnan(curr_fixed_params.V) % if not fixed
                    % if first class or not the same parameters across all
                    % classes
                    if not(g>1 && controls.same_params.V)
                        k = k + (m_y * (m_y + 1) / 2);
                    end
                end
                
                if isnan(curr_fixed_params.g_s) % if not fixed
                    % if first class or not the same parameters across all
                    % classes
                    if not(g>1 && controls.same_params.g_s)
                        k = k + m_g;
                    end
                end
                
                if isnan(curr_fixed_params.a_s) % if not fixed
                    % if first class or not the same parameters across all
                    % classes
                    if not(g>1 && controls.same_params.a_s)
                        k = k + dim_size.alpha_eqns;
                    end
                end
                
                if isnan(curr_fixed_params.mu_0) % if not fixed
                    % if first class or not the same parameters across all
                    % classes
                    if not(g>1 && controls.same_params.mu_0)
                        k = k + m_x;
                    end
                end
                
                if isnan(curr_fixed_params.W_0) % if not fixed
                    % if first class or not the same parameters across all
                    % classes
                    if not(g>1 && controls.same_params.W_0)
                        k = k + (m_x * (m_x + 1) / 2);
                    end
                end
                
                if g>1 % class covariates
                    % no parameters for the first class - ensures
                    % identifiability
                    k = k + dim_size.class_cov;
                end
                
            end
        end
        
        
        function log_like = calc_log_like(pat_data, num_classes, model_coef_est_all_classes, controls, max_censor_time)
            % FUNCTION NAME:
            %   calc_log_like
            %
            % DESCRIPTION:
            %   Calculates the log likelihood with the current estimated
            %   parameters
            %
            % INPUT:
            %   pat_data - (map) Observed data of all patients including
            %              longitudinal biomarkers, survival time, and 
            %              event indicator.
            %   num_classes - (double) Number of classes considered.
            %   model_coef_est_all_classes - (map) Model parameters for all
            %                                classes.
            %   controls - (struct) Contains the controls for the LSDSM EM,
            %              including the boolean to indicate if survival
            %              data should affect hidden states.
            %   max_censor_time - (double) Maximum time considered.
            %
            % OUTPUT:
            %   log_like - (double) Log likelihood value
            %
            % REVISION HISTORY:
            %   01/04/2023 - mcauchi1
            %       * Initial implementation
            %
            
            % number of patients
            num_pats = double(pat_data.Count);
            
            % maximum number of time points to consider
            max_num_pts = ceil(max_censor_time / model_coef_est_all_classes(1).DeltaT);
            
            % initialise the log likelihood variable
            log_like = 0;
            
            for ii=1:num_pats % for every patient
                    
                % Capture information from current patient
                pat_ii = pat_data(ii);

                % responsibility of class g for patient i
                E_c_ig = zeros(1, num_classes);
                prior_probs = zeros(1, num_classes);

                % Prior Probabilities
                % Calculate numerator of prior probability for each class
                for g=1:num_classes
                    zeta_class_g = model_coef_est_all_classes(g).zeta;
                    prior_probs(g) = exp(zeta_class_g' * pat_ii.class_cov);
                end
                % Normalise these prior probabilities
                prior_probs = prior_probs / sum(prior_probs);

                for g=1:num_classes % for every class
                    % Set coefficients for the g-th model
                    model_coef_est = model_coef_est_all_classes(g);

                    % Set initial state conditions for patient ii
                    pat_ii.mu_0 = model_coef_est.mu_0;
                    pat_ii.W_0 = model_coef_est.W_0;

                    % Forward recursion - Standard/Modified RTS Filter
                    [pat_ii.mu_tilde, pat_ii.V_tilde, log_like_val] = ...
                        LSDSM_MM_ALLFUNCS.Kalman_filter(pat_ii, model_coef_est, max_num_pts, controls);

                    % Store the log likelihood value for this patient
                    E_c_ig(g) = exp(log_like_val) * prior_probs(g);
                end

                % Find the log likelihood for patient ii at iteration j
                log_like = log_like + log(sum(E_c_ig));
            end
        end
        
        
        function AIC_val = AIC(dim_size, pat_data, model_coef_est_all_classes, controls, max_censor_time)
            % FUNCTION NAME:
            %   AIC
            %
            % DESCRIPTION:
            %   Calculates the Akaike Information Criterion (AIC) value.
            %
            % INPUT:
            %   dim_size - (struct) Contains dimension sizes for variables
            %              and model parameters.
            %   pat_data - (map) Observed data of all patients including
            %              longitudinal biomarkers, survival time, and 
            %              event indicator.
            %   model_coef_est_all_classes - (map) Model parameters for all
            %                                classes.
            %   controls - (struct) Contains the controls for the LSDSM EM,
            %              including the fixed/same parameters set
            %   max_censor_time - (double) Maximum time considered.
            %
            % OUTPUT:
            %   AIC_val - (double) Akaike Information Criterion value.
            %
            % REVISION HISTORY:
            %   01/04/2023 - mcauchi1
            %       * Initial implementation
            %
            
            % number of classes
            num_classes = double(controls.fixed_params.Count);
            
            % calculate the number of estimated parameters
            k = LSDSM_MM_ALLFUNCS.calc_num_params(dim_size, controls);
            
            % calculate the log likelihood
            log_like = LSDSM_MM_ALLFUNCS.calc_log_like(pat_data, num_classes, model_coef_est_all_classes, controls, max_censor_time);
            
            % calculate Akaike information criterion
            AIC_val = 2 * (k - log_like);
            
        end
        
        
        function BIC_val = BIC(dim_size, pat_data, model_coef_est_all_classes, controls, max_censor_time)
            % FUNCTION NAME:
            %   BIC
            %
            % DESCRIPTION:
            %   Calculates the Bayesian Information Criterion (AIC) value.
            %
            % INPUT:
            %   dim_size - (struct) Contains dimension sizes for variables
            %              and model parameters.
            %   pat_data - (map) Observed data of all patients including
            %              longitudinal biomarkers, survival time, and 
            %              event indicator.
            %   model_coef_est_all_classes - (map) Model parameters for all
            %                                classes.
            %   controls - (struct) Contains the controls for the LSDSM EM,
            %              including the fixed/same parameters set
            %   max_censor_time - (double) Maximum time considered.
            %
            % OUTPUT:
            %   BIC_val - (double) Bayesian Information Criterion value.
            %
            % REVISION HISTORY:
            %   01/04/2023 - mcauchi1
            %       * Initial implementation
            %
            
            % number of data points
            n = sum(LSDSM_MM_ALLFUNCS.extract_field_from_map(pat_data, 'm_i'));
            
            % number of classes
            num_classes = double(controls.fixed_params.Count);
            
            % calculate the number of estimated parameters
            k = LSDSM_MM_ALLFUNCS.calc_num_params(dim_size, controls);
            
            % calculate the log likelihood
            log_like = LSDSM_MM_ALLFUNCS.calc_log_like(pat_data, num_classes, model_coef_est_all_classes, controls, max_censor_time);
            
            % calculate Bayesian information criterion
            BIC_val = k * log(n) - 2 * log_like;
            
        end
        
        
        function [model_coef_est_out, max_iter, param_traj, RTS_arrs, E_c_ig_allpats] = ...
                LSDSM_MM_EM(dim_size, pat_data, controls, max_censor_time)
            % FUNCTION NAME:
            %   LSDSM_MM_EM
            %
            % DESCRIPTION:
            %   Executes the Expectation Maximisation (EM) algorithm to
            %   find the parameters for the Linear State space Dynamic
            %   Survival Model (LSDSM) with the mixture models extension.
            %   - Longitudinal Sub-process: Linear Gaussian State Space
            %   Model 
            %   - Survival Sub-process: Proportional Hazards Model
            %   
            % INPUT:
            %   dim_size - (struct) Contains dimension sizes for variables
            %              and model parameters.
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
            %   model_coef_est_out - (map) Estimated Parameters for the
            %                        number of classes defined.
            %   max_iter - (double) Number of EM iterations executed.
            %   param_traj_tmp - (map) Evolution of parameter values
            %                    over EM iterations for all classes.
            %   RTS_arrs - (cell) Contains the filtered and smoothed
            %              outputs of the hidden state trajectories for all
            %              classes.
            %   E_c_ig_allpats - (array) (N x G) matrix containing the
            %                    responsibility that every class has on
            %                    every patient at the final iteration.
            %
            % REVISION HISTORY:
            %   17/12/2022 - mcauchi1
            %       * Initial implementation
            %   12/01/2023 - mcauchi1
            %       * Updated to include multiple models for different
            %       classes
            %   22/02/2023 - mcauchi1
            %       * Improved readability and conciseness
            %   15/03/2023 - mcauchi1
            %       * Included the functionality to plot the parameter
            %       values over EM iterations
            %
            
            % extract the number of patients
            num_pats = pat_data.Count;

            % set the initial parameters - make a copy of the map
            model_coef_est_all_classes = containers.Map(controls.init_params.keys, controls.init_params.values);
            
            % Check for errors and illegal cases
            LSDSM_MM_ALLFUNCS.ensure_correct_dims(pat_data, dim_size);
            LSDSM_MM_ALLFUNCS.ensure_correct_param_dims(model_coef_est_all_classes, dim_size, controls);
            LSDSM_MM_ALLFUNCS.ensure_correct_fixed_param_dims(controls.fixed_params, dim_size, controls);
            LSDSM_MM_ALLFUNCS.ensure_same_params_correctness(controls.same_params);
            
            % Retrieve the number of classes
            num_classes = double(model_coef_est_all_classes.Count);
            
            % maximum number of time points to consider - the +1 is used in
            % case there are 21 measurements for a max obs time of 20.
            max_num_pts = ceil(max_censor_time / model_coef_est_all_classes(1).DeltaT) + 1;
            
            % Initialise the log likelihood array
            log_like_val_tot_arr = zeros(num_pats, controls.EM_iters);
            
            % Responsibility of classes over EM iterations
            E_c_ig_EM = zeros(num_classes, controls.EM_iters);

            % Initialise map to track parameter changes for all classes
            [param_traj] = LSDSM_MM_ALLFUNCS.initialise_param_trajectories(model_coef_est_all_classes, dim_size, controls);
            
            % Initialise cells to retain RTS arrs for every class
            RTS_arrs = LSDSM_MM_ALLFUNCS.initialise_RTS_arrs(num_classes, dim_size, max_num_pts, num_pats);
            
            if controls.allow_plots % if we are allowing plots, initialise the dynamic plots
                fig_cells = LSDSM_MM_ALLFUNCS.initialise_dynamic_param_plots(num_classes);
            end
            

            for j=2:controls.EM_iters % iterate the EM algorithm (iter_EM-1) times
                
                % Initialise sums of expectations to be used in update equations
                E_sums = LSDSM_MM_ALLFUNCS.initialise_E_sums(num_classes, dim_size);
                
                % Initialise the responsibilities that each class holds for every patient
                E_c_ig_allpats = zeros(num_pats, num_classes);

                for ii=1:num_pats % for every patient
                    
                    % Capture information from current patient
                    pat_ii = pat_data(ii);
                    
                    % Create a space to store the expectations for every class for the current patient
                    pat_ii.E = cell(1, num_classes);
                    
                    % responsibility of class g for patient i
                    E_c_ig = zeros(1, num_classes);
                    prior_probs = zeros(1, num_classes);
                    
                    %%% Prior Probabilities %%%
                    % Calculate numerator of prior probability for each class
                    for g=1:num_classes
                        zeta_class_g = model_coef_est_all_classes(g).zeta;
                        prior_probs(g) = exp(zeta_class_g' * pat_ii.class_cov);
                    end
                    % Normalise these prior probabilities
                    prior_probs = prior_probs / sum(prior_probs);
                    
                    for g=1:num_classes % for every class
                        
                        % Set coefficients for the g-th model
                        model_coef_est = model_coef_est_all_classes(g);
                        
                        % Set initial state conditions for patient ii
                        pat_ii.mu_0 = model_coef_est.mu_0;
                        pat_ii.W_0 = model_coef_est.W_0;
                        
                        %%%%%%%%%%%%%%
                        %%% E Step %%%
                        %%%%%%%%%%%%%%
                        %%% Forward recursion - Standard/Modified RTS Filter %%%
                        [pat_ii.mu_tilde, pat_ii.V_tilde, log_like_val] = ...
                            LSDSM_MM_ALLFUNCS.Kalman_filter(pat_ii, model_coef_est, max_num_pts, controls);
                        
                        % Store the log likelihood value for this patient
                        E_c_ig(g) = exp(log_like_val) * prior_probs(g);

                        %%% Backward recursion - RTS Smoother %%%
                        [pat_ii.mu_hat, pat_ii.V_hat, pat_ii.J_hat] = ...
                            LSDSM_MM_ALLFUNCS.Kalman_smoother(pat_ii, model_coef_est, max_num_pts);

                        %%% Compute the required Expectations %%%
                        [pat_ii.E{g}] = LSDSM_MM_ALLFUNCS.compute_E_fns(pat_ii, model_coef_est, max_num_pts);

                        % Store the RTS filter and smoother outputs
                        RTS_arrs{g}.mu_tilde(:,:,:,ii) = pat_ii.mu_tilde;
                        RTS_arrs{g}.V_tilde(:,:,:,ii) = pat_ii.V_tilde;
                        RTS_arrs{g}.mu_hat(:,:,:,ii) = pat_ii.mu_hat;
                        RTS_arrs{g}.V_hat(:,:,:,ii) = pat_ii.V_hat;
                    end
                    
                    % Find the log likelihood for patient ii at iteration j
                    log_like_val_tot_arr(ii,j) = log(sum(E_c_ig));
                    
                    % Calculate the posterior probability of patient i belonging to class g
                    E_c_ig = E_c_ig / sum(E_c_ig);
                    
                    % If we obtain likelihood values of 0 for all classes
                    % resort to the prior probabilities for E[c_ig]
                    if all(isnan(E_c_ig))
                        E_c_ig = prior_probs;
                    end
                    
                    % Store values in a global matrix (to be used for
                    % evaluating the class parameters)
                    E_c_ig_allpats(ii,:) = E_c_ig;
                    
                    %%%%%%%%%%%%%%
                    %%% M Step %%%
                    %%%%%%%%%%%%%%
                    % Add the contribution of the current patient to the
                    % sum of expectations for every class (to be used in
                    % the M step)
                    E_sums = LSDSM_MM_ALLFUNCS.sum_E_fns(E_sums, E_c_ig, pat_ii);
                    
                end
                
                % The total responsibility of every class for the whole population
                E_c_ig_EM(:,j) = sum(E_c_ig_allpats,1);
                
                % Find the updated parameters
                model_coef_new_all_classes = LSDSM_MM_ALLFUNCS.M_step(pat_data, E_sums, RTS_arrs, E_c_ig_allpats, ...
                                                                   model_coef_est_all_classes, controls, j);
                                                               
                % Update parameter trajectories
                param_traj = LSDSM_MM_ALLFUNCS.update_param_trajectories(model_coef_new_all_classes, param_traj, j);
                
                % vectors to store the absolute/percentage differences of
                % estimated parameters between iterations
                param_diff_all = [];
                param_diff_percent_all = [];
                
                for g=1:num_classes % for every class
                    % store the old coefficients to check for stopping criterion
                    model_coef_old = model_coef_est_all_classes(g);
                    
                    % Store the new coefficients for stopping criterion and
                    % for the next EM step
                    model_coef_new = model_coef_new_all_classes(g);
                    model_coef_est_all_classes(g) = model_coef_new;
                    
                    [param_diff, param_diff_percent] = ...
                        LSDSM_MM_ALLFUNCS.find_model_param_diff(model_coef_new, model_coef_old);
                    
                    param_diff_all = [param_diff_all; param_diff];
                    param_diff_percent_all = [param_diff_percent_all; param_diff_percent];
                end
                
                % store the model parameters for all classes in the output map
                model_coef_est_out = model_coef_est_all_classes;

                % find the maximum absolute difference
                param_max_diff = max(abs(param_diff_all));

                % if stopping criteria is reached
                if param_max_diff < controls.max_param_diff
                    break; % Break from EM algorithm for loop
                end
                
                if controls.verbose
                    if mod(j,10) == 0 % Feedback every 10 iterations
                        fprintf("EM Iteration %4d / %d. Max Parameter Difference: %.6f \n", ...
                                    j, controls.EM_iters, param_max_diff);
                        
                        if controls.allow_plots % if we are allowing plots, update the dynamic plots
                            LSDSM_MM_ALLFUNCS.update_dynamic_param_plots(j, fig_cells, model_coef_est_all_classes(1), ...
                                                                         log_like_val_tot_arr, E_c_ig_EM, param_traj);
                        end
                    end
                end
                
            end % end of EM algorithm

            max_iter = j; % number of EM iterations performed
            
            if controls.allow_plots
                LSDSM_MM_ALLFUNCS.update_dynamic_param_plots(j, fig_cells, model_coef_est_all_classes(1), ...
                                                             log_like_val_tot_arr, E_c_ig_EM, param_traj);
            end

            fprintf('Maximum Parameter Difference: %6f at iteration %d \n', param_max_diff, j);
        end
        
        
        function [mu_out, Sigma_out, log_likelihood_val] = Kalman_filter(pat_ii, model_coef_est, max_censor_time, controls)
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
            %   controls - (struct) Controls including boolean to check if
            %              the alternative mixture model representation is
            %              being used and the type of baseline hazard
            %              function being used.
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
            %   22/02/2023 - mcauchi1
            %       * Introduced NR controls to give more flexibility in
            %       the optimisation of the hidden state values and the
            %       calculation of the observed likelihood function
            %   15/03/2023 - mcauchi1
            %       * Introduced the case of having an input affecting the
            %       state space model
            %   23/03/2023 - mcauchi1
            %       * Made the required changes for the alternative mixture
            %       model extension
            %   01/09/2023 - mcauchi1
            %       * Introduced the H matrix to have the survival function
            %       depend on a pre-determined linear combination of
            %       states.
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
            
            % controls for Newton Raphson procedure
            NR_controls = struct('max_iter', 100, 'eps', 1e-6, 'damp_factor', 0.5, 'minimising', 1);

            for j=1:pat_ii.m_i % for every time step observed
                
                % boolean to check if patient died within the first time step
                [delta_ij, tau_ij] = LSDSM_MM_ALLFUNCS.pat_status_at_j(j, pat_ii, model_coef_est.DeltaT);
                
                % Correct the observation vectors/matrices to account for missing observations
                [y_tmp_i, C_tmp_i, V_tmp_i, Omega_O, Omega_M, I_mat_O, I_mat_M, nabla_ij] ...
                    = LSDSM_MM_ALLFUNCS.missing_val_matrices(pat_ii.y(:,:,j), model_coef_est);
                
                if j==1 % Initialisation of filter
                    % Correction using the longitudinal data
                    K(:,:,1) = pat_ii.W_0 * C_tmp_i' * (C_tmp_i * pat_ii.W_0 * C_tmp_i' + V_tmp_i)^-1;
                    mu(:,:,1) = pat_ii.mu_0 + K(:,:,1) * (y_tmp_i - C_tmp_i * pat_ii.mu_0);
                    Sigma(:,:,1) = (eye(size(pat_ii.mu_0,1)) - K(:,:,1) * C_tmp_i) * pat_ii.W_0;
                    
                    % Prepare the coefficients for the first contribution of the log likelihood
                    f1_fn_coef = struct('delta_ij', delta_ij, 'y_ij', y_tmp_i, ...
                                         'C', C_tmp_i, 'V', V_tmp_i, 'Omega_O', Omega_O, ...
                                         'pred_mu', pat_ii.mu_0, 'pred_V', pat_ii.W_0, ...
                                         'base_cov', pat_ii.base_cov, 'g_s', model_coef_est.g_s, ...
                                         'a_s', model_coef_est.a_s, 'tau_ij', tau_ij, 'H_mat', model_coef_est.H_mat);
                else % if j > 1
                    % Correction using the longitudinal data
                    if isfield(pat_ii, 'u') % if input to SSM exists
                        [mu(:,:,j), Sigma(:,:,j), K(:,:,j), P(:,:,j-1)] = ...
                        LSDSM_MM_ALLFUNCS.KF_single_step(mu_out(:,:,j-1), Sigma_out(:,:,j-1), ...
                                                      pat_ii.y(:,:,j), model_coef_est, pat_ii.u(:,:,j-1));
                    else
                        [mu(:,:,j), Sigma(:,:,j), K(:,:,j), P(:,:,j-1)] = ...
                            LSDSM_MM_ALLFUNCS.KF_single_step(mu_out(:,:,j-1), Sigma_out(:,:,j-1), ...
                                                          pat_ii.y(:,:,j), model_coef_est);
                    end

                    % calculate the one-step forward predicted mu
                    mu_pred = model_coef_est.A * mu_out(:,:,j-1);
                    if isfield(model_coef_est, 'B') % if the input to SSM is present
                        mu_pred = mu_pred + model_coef_est.B * pat_ii.u(:,:,j-1);
                    end
                    
                    % Prepare the coefficients for the rest of the contributions of the log likelihood
                    f1_fn_coef = struct('delta_ij', delta_ij, 'y_ij', y_tmp_i, ...
                                         'C', C_tmp_i, 'V', V_tmp_i, 'Omega_O', Omega_O, ...
                                         'pred_mu', mu_pred, ...
                                         'pred_V', P(:,:,j-1), ...
                                         'base_cov', pat_ii.base_cov, 'g_s', model_coef_est.g_s, ...
                                         'a_s', model_coef_est.a_s, 'tau_ij', tau_ij, 'H_mat', model_coef_est.H_mat);
                end

                if controls.mod_KF % Correction using survival data
                    Sigma_tmp_NR = Sigma(:,:,j);

                    % Find the value of x that maximises the posterior distribution
                    g_fn_coef = struct('delta_ij', delta_ij, 'mu_ij', mu(:,:,j), ...
                                   'Sigma_ij', Sigma_tmp_NR, 'base_cov', pat_ii.base_cov, ...
                                   'g_s', model_coef_est.g_s, 'a_s', model_coef_est.a_s, ...
                                   'tau_ij', tau_ij, 'H_mat', model_coef_est.H_mat);
                    
                    % Use Newton Raphson's iterative method to approximate the posterior as a Gaussian distribution
                    x_NR = LSDSM_MM_ALLFUNCS.Newton_Raphson(mu(:,:,j), NR_controls, @LSDSM_MM_ALLFUNCS.f_KF, ...
                                                            @LSDSM_MM_ALLFUNCS.dfdx_KF, @LSDSM_MM_ALLFUNCS.d2fdx2_KF, g_fn_coef);

                    % Update mu_tilde and Sigma_tilde
                    mu_out(:,:,j) = x_NR;
                    
                    Sigma_tmp_NR_out = (tau_ij * exp(model_coef_est.g_s' * pat_ii.base_cov) ...
                                                * exp(model_coef_est.a_s' * model_coef_est.H_mat * x_NR) ...
                                                * (model_coef_est.H_mat' * model_coef_est.a_s * model_coef_est.a_s' * model_coef_est.H_mat) ...
                                        + Sigma_tmp_NR^-1 )^-1;
                                    
                    Sigma_out(:,:,j) = Sigma_tmp_NR_out;
                    Sigma_out(:,:,j) = LSDSM_MM_ALLFUNCS.ensure_sym_mat(Sigma_out(:,:,j));
                    
                else % If we are not correcting the states (directly) using survival data
                    mu_out(:,:,j) = mu(:,:,j);
                    Sigma_out(:,:,j) = Sigma(:,:,j);
                    Sigma_out(:,:,j) = LSDSM_MM_ALLFUNCS.ensure_sym_mat(Sigma_out(:,:,j));
                end
                
                % Calculate the log likelihood contribution of the current
                % time step and add it to the overall contribution
                log_likelihood_val = log_likelihood_val ...
                        + log(LSDSM_MM_ALLFUNCS.like_fn_curr_step(mu_out(:,:,j), NR_controls, f1_fn_coef, controls.mod_KF));
                    
            end % end of for loop
            
            if ~controls.mod_KF % if we are using the alternative mixture model extension
                if strcmpi(controls.base_haz, 'Weibull')
                    alpha_w = model_coef_est.a_s; % alpha as defined in Weibull
                    beta_w = model_coef_est.b_s; % beta as defined in Weibull
                    base_cov_haz = exp(model_coef_est.g_s' * pat_ii.base_cov); % hazard contribution by baseline covariates
                    
                    % calculate the hazard and survival contributions to the likelihood function
                    haz_contrib = ( alpha_w / (beta_w^alpha_w) * pat_ii.surv_time^(alpha_w - 1) * base_cov_haz )^pat_ii.delta_ev;
                    cum_haz = ( pat_ii.surv_time / beta_w )^alpha_w * base_cov_haz;
                    surv_contrib = exp( - cum_haz );
                    
                    % Add the survival information contribution to the observed log likelihood
                    log_likelihood_val = log_likelihood_val + log( haz_contrib * surv_contrib );
                end
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
            %   15/03/2023 - mcauchi1
            %       * Introduced the effect of the input to the state space
            %       model (if applicable)
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
            J_out(:,:,pat_ii.m_i) = zeros(size(pat_ii.mu_tilde,1), size(pat_ii.mu_tilde,1));

            % Iterate through every time step to find the probabilities given
            % the entire observed data.
            for i=2:pat_ii.m_i
                k = pat_ii.m_i - i + 1; % N-1, N-2, ..., 2, 1

                % Find the smoother output of the current time step
                if isfield(pat_ii, 'u') % if input to SSM exists
                    [mu_out(:,:,k), V_out(:,:,k), J_out(:,:,k)] = ...
                    LSDSM_MM_ALLFUNCS.KS_single_step(mu_out(:,:,k+1), V_out(:,:,k+1), pat_ii.mu_tilde(:,:,k), ...
                                                    pat_ii.V_tilde(:,:,k), model_coef_est, pat_ii.u(:,:,k));
                                                
                else % if there is no input
                    [mu_out(:,:,k), V_out(:,:,k), J_out(:,:,k)] = ...
                    LSDSM_MM_ALLFUNCS.KS_single_step(mu_out(:,:,k+1), V_out(:,:,k+1), pat_ii.mu_tilde(:,:,k), ...
                                                    pat_ii.V_tilde(:,:,k), model_coef_est);
                end

                V_out(:,:,k) = LSDSM_MM_ALLFUNCS.ensure_sym_mat(V_out(:,:,k));
                % Add a small number to the diagonals of V to avoid
                % singular matrices since we require the inverse of V.
                V_out(:,:,k) = V_out(:,:,k) + 1e-9 * eye(size(V_out,1));
            end
        end
        
        
        function [mu_tmp, Sigma_tmp, K_tmp, P_tmp] = KF_single_step(mu_ineg1, Sigma_ineg1, y_i, model_coef_est, u_ineg1)
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
            %   u_ineg1 - (array - optional) Input to the SSM (if 
            %             applicable).
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
            %   15/03/2023 - mcauchi1
            %       * Introduced the input in SSM to the filter
            %
            
            if nargin < 5 % if input is not provided
                u_ineg1 = NaN;
            end

            % Identify and restructure the missing values accordingly
            [y_tmp_i, C_tmp_i, V_tmp_i, Omega_O, Omega_M, I_mat_O, ...
                        I_mat_M, nabla_ij] = LSDSM_MM_ALLFUNCS.missing_val_matrices(y_i, model_coef_est);

            % evaluate the predicted value of the state
            mu_pred_i = model_coef_est.A * mu_ineg1;
            if not(isnan(u_ineg1))
                mu_pred_i = mu_pred_i + model_coef_est.B * u_ineg1;
            end
            
            % standard filter approach for linear Gaussian SSM
            P_tmp = model_coef_est.A * Sigma_ineg1 * model_coef_est.A' ...
                    + model_coef_est.G_mat * model_coef_est.W * model_coef_est.G_mat';
            K_tmp = P_tmp * C_tmp_i' * (C_tmp_i * P_tmp * C_tmp_i' + V_tmp_i)^-1;
            mu_tmp = mu_pred_i + K_tmp * ( y_tmp_i - C_tmp_i * mu_pred_i );
            Sigma_tmp = (eye(size(mu_ineg1,1)) - K_tmp * C_tmp_i) * P_tmp;
        end

        
        function [mu_hat_tmp, V_hat_tmp, J_tmp] = KS_single_step(mu_hat_iplus1, V_hat_iplus1, mu_tilde_tmp, V_tilde_tmp, model_coef_est, u_i)
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
            %   u_i - (array - optional) Input to the SSM (if applicable).
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
            %   15/03/2023 - mcauchi1
            %       * Introduced the effect of input to SSM
            %
            
            if nargin < 6 % if there is no input
                u_i = NaN;
            end
            
            % calculate the one step ahead state
            mu_1step_ahead = model_coef_est.A * mu_tilde_tmp;
            if not(isnan(u_i)) % if there is an input
                mu_1step_ahead = mu_1step_ahead + model_coef_est.B * u_i;
            end
            
            % Standard smoother procedure for linear Gaussian SSM
            P_tmp = model_coef_est.A * V_tilde_tmp * model_coef_est.A' ...
                    + model_coef_est.G_mat * model_coef_est.W * model_coef_est.G_mat';
            J_tmp = V_tilde_tmp * model_coef_est.A' * P_tmp^-1;
            mu_hat_tmp = mu_tilde_tmp + J_tmp * (mu_hat_iplus1 - mu_1step_ahead);
            V_hat_tmp = V_tilde_tmp + J_tmp * (V_hat_iplus1 - P_tmp) * J_tmp';
        end

        
        function [y_tmp_i, C_tmp_i, V_tmp_i, Omega_O, Omega_M, I_mat_O, I_mat_M, nabla_ij] = ...
                missing_val_matrices(y_tmp_i, model_coef_est)
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
        
        
        function like_val = like_fn_curr_step(x_ij, NR_controls, coeffs, mod_KF)
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
            %   NR_controls - (struct) Contains the Newton Raphson controls
            %                 to identify the likelihood value, including
            %                 maximum number of iteration, stopping
            %                 criterion, damping factor, and boolean to
            %                 check if we are minimising a function.
            %   coeffs - (struct) Contains all required data to compute
            %            this function.
            %   mod_KF - (bool) Tells us if algorithm is using survival
            %            data to (directly) correct the hidden states.
            %
            % OUTPUT:
            %   like_val - (double) The likelihood contribution of patient
            %              i for time step j.
            %
            % REVISION HISTORY:
            %   17/12/2022 - mcauchi1
            %       * Initial implementation
            %   22/02/2023 - mcauchi1
            %       * Introduced NR controls to give more flexibility in
            %       its computation
            %   23/03/2023 - mcauchi1
            %       * Evaluates the likelihood for the current step for the
            %       alternative mixture model extension as well
            %   01/04/2023 - mcauchi1
            %       * Fixed error where likelihood was incorrectly biased
            %       in the absence of measurements
            %
            
            % Find the dimension sizes of the hidden states and biomarker observations
            dim_size.states = size(x_ij,1);
            dim_size.y = size(coeffs.Omega_O, 1); % observed vector only
            
            if mod_KF % if we are using the standard MM
                % 1) Find the value of x_{ij} that gives the minimum of f_1 (through NR)
                x_NR = LSDSM_MM_ALLFUNCS.Newton_Raphson(x_ij, NR_controls, @LSDSM_MM_ALLFUNCS.f1x_ij, @LSDSM_MM_ALLFUNCS.df1dx, ...
                                                            @LSDSM_MM_ALLFUNCS.d2f1dx2, coeffs);

                % 2) Evaluate the integral using Laplace approximation
                hess = LSDSM_MM_ALLFUNCS.d2f1dx2(x_NR, coeffs);
                f1x_ij = LSDSM_MM_ALLFUNCS.f1x_ij(x_NR, coeffs);
                int_val = (2*pi)^(dim_size.states/2) * det(hess)^(-1/2) * exp(-f1x_ij);

                % 3) Evaluate the final expression for the likelihood of
                %    the current observations given the past observations
                if isempty(coeffs.Omega_O) % if there are no measurements
                    % likelihood does not include the measurement variance
                    like_val = exp(coeffs.delta_ij * coeffs.g_s' * coeffs.base_cov) ...
                                * (2*pi)^(-(dim_size.states + dim_size.y)/2) ...
                                * det(coeffs.pred_V)^(-1/2) * int_val;
                else % if there are some/all measurements observed
                    % likelihood includes the observed parts of the
                    % measurement variance
                    V_o = coeffs.Omega_O * coeffs.V * coeffs.Omega_O';
                    like_val = exp(coeffs.delta_ij * coeffs.g_s' * coeffs.base_cov) ...
                                * (2*pi)^(-(dim_size.states + dim_size.y)/2) ...
                                * det(V_o)^(-1/2) * det(coeffs.pred_V)^(-1/2) * int_val;
                end
                        
            else % if we are using the alternative mixture model extension
                var_obs = coeffs.C * coeffs.pred_V * coeffs.C' + coeffs.V;
                % use only observed values
                y_o = coeffs.Omega_O * coeffs.y_ij;
                C_o = coeffs.Omega_O * coeffs.C;
                var_obs = coeffs.Omega_O * var_obs * coeffs.Omega_O';
                
                like_val = (2*pi)^(-dim_size.y/2) * det(var_obs)^(-1/2) ...
                            * exp( - (1/2) * (y_o - C_o * coeffs.pred_mu)' ...
                                    * var_obs^(-1) * (y_o - C_o * coeffs.pred_mu) );
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
            %   15/03/2023 - mcauchi1
            %       * Introduced new expectations required if there is
            %       input present for the state space model
            %
            
            % Find the indices of "dynamic" states
            [idx_present_r, idx_present_c] = find(model_coef_est.G_mat ~= 0);

            % Placeholders for the expected values for the current patient
            E.xn = zeros(size(pat_ii.mu_hat,1), 1, max_censor_time);
            E.xn_xnneg1 = zeros(size(pat_ii.mu_hat,1), size(pat_ii.mu_hat,1), max_censor_time);
            E.xn_xn = zeros(size(pat_ii.mu_hat,1), size(pat_ii.mu_hat,1), max_censor_time);
            E.barxn = zeros(length(idx_present_r), 1, max_censor_time);
            E.barxn_barxn = zeros(length(idx_present_r), length(idx_present_r), max_censor_time);
            E.barxn_xnneg1 = zeros(length(idx_present_r), size(pat_ii.mu_hat,1), max_censor_time);
            E.yn = zeros(size(pat_ii.y,1), 1, max_censor_time);
            E.yn_yn = zeros(size(pat_ii.y,1), size(pat_ii.y,1), max_censor_time);
            E.yn_xn = zeros(size(pat_ii.y,1), size(pat_ii.mu_hat,1), max_censor_time);

            % The expectation of x is the smoothed distribution
            E.xn = pat_ii.mu_hat;

            % Extract the dynamic states
            mu_bar_tmp = pat_ii.mu_hat(idx_present_r,:,:);
            E.barxn = mu_bar_tmp;
            
            % covariance matrix of x_n and x_{n-1}
            M_tmp = pagemtimes(pat_ii.V_hat(:,:,2:end), 'none', pat_ii.J_hat(:,:,1:end-1), 'transpose');
            
            % find some of the expectations efficiently using pagemtimes
            E.xn_xnneg1(:,:,2:end) = M_tmp + pagemtimes(pat_ii.mu_hat(:,:,2:end), 'none', ...
                                                                pat_ii.mu_hat(:,:,1:end-1), 'transpose');
            
            E.barxn_xnneg1(:,:,2:end) = M_tmp(idx_present_r,:,:) + ...
                                            pagemtimes(mu_bar_tmp(:,:,2:end), 'none', ...
                                                       pat_ii.mu_hat(:,:,1:end-1), 'transpose');
            
            E.xn_xn = pat_ii.V_hat + pagemtimes(pat_ii.mu_hat, 'none', pat_ii.mu_hat, 'transpose');
            E.barxn_barxn = pat_ii.V_hat(idx_present_r, idx_present_r, :) + ...
                    pagemtimes(mu_bar_tmp, 'none', mu_bar_tmp, 'transpose');
                
            if isfield(pat_ii, 'u') % if there is input to SSM
                % introduce the required expectations for the input adaptation
                E.barxn_unneg1 = zeros(length(idx_present_r), size(pat_ii.u, 1), max_censor_time);
                E.barxn_unneg1(:,:,2:end) = pagemtimes(mu_bar_tmp(:,:,2:end), 'none', pat_ii.u(:,:,1:end-1), 'transpose');
                E.xn_un = pagemtimes(pat_ii.mu_hat, 'none', pat_ii.u, 'transpose');
                E.un_un = pagemtimes(pat_ii.u, 'none', pat_ii.u, 'transpose');
                E.barxn_hatxunneg1 = [E.barxn_xnneg1 E.barxn_unneg1];
                E.hatxun_hatxun = [E.xn_xn E.xn_un;
                                   pagetranspose(E.xn_un), E.un_un];
            end
            
            for i=1:pat_ii.m_i % for every time step
                E.xn_xn(:,:,i) = LSDSM_MM_ALLFUNCS.ensure_sym_mat(E.xn_xn(:,:,i));
                E.barxn_barxn(:,:,i) = LSDSM_MM_ALLFUNCS.ensure_sym_mat(E.barxn_barxn(:,:,i));

                % Expectations involving y:
                % Modify the observation vectors and matrices to account
                % for the missing measurements
                [y_tmp_i, C_tmp_i, V_tmp_i, Omega_O, Omega_M, I_mat_O, ...
                        I_mat_M, nabla_ij] = LSDSM_MM_ALLFUNCS.missing_val_matrices(pat_ii.y(:,:,i), model_coef_est);

                E.yn(:,:,i) = y_tmp_i - nabla_ij * (y_tmp_i - model_coef_est.C * E.xn(:,:,i));

                E.yn_yn(:,:,i) = I_mat_M * (nabla_ij * model_coef_est.V + ...
                    nabla_ij * model_coef_est.C * pat_ii.V_hat(:,:,i) * model_coef_est.C' * nabla_ij') * I_mat_M + ...
                    E.yn(:,:,i) * E.yn(:,:,i)';

                E.yn_yn(:,:,i) = LSDSM_MM_ALLFUNCS.ensure_sym_mat(E.yn_yn(:,:,i));

                E.yn_xn(:,:,i) = nabla_ij * model_coef_est.C * pat_ii.V_hat(:,:,i) + E.yn(:,:,i) * E.xn(:,:,i)';
            end
        end

        
        function E_sums = sum_E_fns(E_sums, E_c_ig, pat_ii)
            % FUNCTION NAME:
            %   sum_E_fns
            %
            % DESCRIPTION:
            %   This function continues to add towards the total sum of
            %   expectations that are required in the M step of the EM
            %   algorithm.
            %
            % INPUT:
            %   E_sums - (cell) Contains the summations of the 
            %            expectations all the previously evaluated
            %            patients. Every cell is defined for every class
            %            containing a struct.
            %   E_c_ig - (array) Contains the expectations that patient i
            %            belongs to every class
            %   pat_ii - (struct) Contains all observed data of the current
            %            patient.
            %
            % OUTPUT:
            %   E_sums - (cell) New summations of all required
            %            expectations. Every cell is defined for every 
            %            class containing a struct.
            %
            % REVISION HISTORY:
            %   17/12/2022 - mcauchi1
            %       * Initial implementation
            %   12/01/2023 - mcauchi1
            %       * Adapted the function to calculate the expectations
            %       for every class
            %   15/03/2023 - mcauchi1
            %       * Included the summations required to involve the
            %       effect of input to SSM
            %

            num_classes = size(E_sums, 1); % number of classes
            
            % number of observations for current patient
            m_i = pat_ii.m_i;
            
            for g=1:num_classes % for every class
                % Extract the total sum of expectations for current class
                E_tmp = E_sums{g};
                % Extract expectations generated by class g for current
                % patient
                pat_E_tmp = pat_ii.E{g};
                
                % Sum across all patients of sum_{n=2}^{N} E[x(n) x(n-1)']
                E_tmp.xn_xnneg1_from2 = E_tmp.xn_xnneg1_from2 ...
                                            + E_c_ig(g) * sum(pat_E_tmp.xn_xnneg1(:,:,2:m_i), 3);
                % Sum across all patients of sum_{n=1}^{N-1} E[x(n) x(n)']
                E_tmp.xn_xn_tillNneg1 = E_tmp.xn_xn_tillNneg1 ...
                                            + E_c_ig(g) * sum(pat_E_tmp.xn_xn(:,:,1:m_i-1), 3);
                % Sum across all patients of sum_{n=2}^{N} E[x(n) x(n)']
                E_tmp.xn_xn_from2 = E_tmp.xn_xn_from2 ...
                                        + E_c_ig(g) * sum(pat_E_tmp.xn_xn(:,:,2:m_i), 3);
                % Sum across all patients of sum_{n=1}^{N} E[x(n) x(n)']
                E_tmp.xn_xn = E_tmp.xn_xn + E_c_ig(g) * sum(pat_E_tmp.xn_xn(:,:,1:m_i), 3);
                % Sum across all patients of sum_{n=1}^{N} E[x(n)]
                E_tmp.xn = E_tmp.xn + E_c_ig(g) * sum(pat_E_tmp.xn(:,:,1:m_i), 3);
                % Sum across all patients of sum_{n=2}^{N} E[bar_x(n) bar_x(n)']
                E_tmp.barxn_barxn_from2 = E_tmp.barxn_barxn_from2 ...
                                              + E_c_ig(g) * sum(pat_E_tmp.barxn_barxn(:,:,2:m_i), 3);
                % Sum across all patients of sum_{n=2}^{N} E[bar_x(n) x(n-1)']
                E_tmp.barxn_xnneg1_from2 = E_tmp.barxn_xnneg1_from2 ...
                                               + E_c_ig(g) * sum(pat_E_tmp.barxn_xnneg1(:,:,2:m_i), 3);
                % Sum across all patients of initial states E[x0] ( E[x(1)] since MATLAB starts arrays from 1 )
                E_tmp.x0 = E_tmp.x0 + E_c_ig(g) * pat_E_tmp.xn(:,:,1);
                % Sum across all patients of initial states E[x0 x0'] 
                % ( E[x(1) x(1)'] since MATLAB starts arrays from 1 )
                E_tmp.x0_x0 = E_tmp.x0_x0 + E_c_ig(g) * pat_E_tmp.xn_xn(:,:,1);

                % Sum across all patients of sum_{n=3}^{N} E[x(n) x(n)']
                E_tmp.barxn_xnneg1_from3 = E_tmp.barxn_xnneg1_from3 ...
                                               + E_c_ig(g) * sum(pat_E_tmp.barxn_xnneg1(:,:,3:m_i), 3);
                % Sum across all patients of sum_{n=2}^{N-1} E[x(n) x(n)']
                E_tmp.xn_xn_from2_tillNneg1 = E_tmp.xn_xn_from2_tillNneg1 ...
                                                  + E_c_ig(g) * sum(pat_E_tmp.xn_xn(:,:,2:m_i-1), 3);
                                              
                if isfield(pat_ii, 'u') % if there is an input present affecting the SSM
                    % Find the required summations to involve input
                    
                    % Sum across all patients of sum_{n=3}^{N} E[x(n) hat_xu(n)']
                    E_tmp.barxn_hatxunneg1_from3 = E_tmp.barxn_hatxunneg1_from3 ...
                                                   + E_c_ig(g) * sum(pat_E_tmp.barxn_hatxunneg1(:,:,3:m_i), 3);
                    % Sum across all patients of sum_{n=2}^{N-1} E[hat_xu(n) hat_xu(n)']
                    E_tmp.hatxun_hatxun_from2_tillNneg1 = E_tmp.hatxun_hatxun_from2_tillNneg1 ...
                                                      + E_c_ig(g) * sum(pat_E_tmp.hatxun_hatxun(:,:,2:m_i-1), 3);
                    
                    % Sum across all patients of sum_{n=2}^{N} E[bar_x(n) u(n-1)']
                    E_tmp.barxn_unneg1_from2 = E_tmp.barxn_unneg1_from2 ...
                                                      + E_c_ig(g) * sum(pat_E_tmp.barxn_unneg1(:,:,2:m_i), 3);
                    % Sum across all patients of sum_{n=1}^{N-1} E[x(n) u(n)']
                    E_tmp.xn_un_tillNneg1 = E_tmp.xn_un_tillNneg1 ...
                                                      + E_c_ig(g) * sum(pat_E_tmp.xn_un(:,:,1:m_i-1), 3);
                    % Sum across all patients of sum_{n=1}^{N-1} E[u(n) u(n)']
                    E_tmp.un_un_tillNneg1 = E_tmp.un_un_tillNneg1 ...
                                                      + E_c_ig(g) * sum(pat_E_tmp.un_un(:,:,1:m_i-1), 3);
                                                  
                end

                % Sum across all patients of sum_{n=1}^{N} E[y(n) y(n)']
                E_tmp.yn_yn = E_tmp.yn_yn + E_c_ig(g) * sum(pat_E_tmp.yn_yn(:,:,1:m_i), 3);
                % Sum across all patients of sum_{n=1}^{N} E[y(n) x(n)']
                E_tmp.yn_xn = E_tmp.yn_xn + E_c_ig(g) * sum(pat_E_tmp.yn_xn(:,:,1:m_i), 3);
                
                
                % Testing - removing the dependency from the initial
                % observation and state
                E_tmp.barxn_barxn_from3 = E_tmp.barxn_barxn_from3 ...
                                              + E_c_ig(g) * sum(pat_E_tmp.barxn_barxn(:,:,3:m_i), 3);
                E_tmp.yn_yn_from2 = E_tmp.yn_yn_from2 + E_c_ig(g) * sum(pat_E_tmp.yn_yn(:,:,2:m_i), 3);  
                E_tmp.yn_xn_from2 = E_tmp.yn_xn_from2 + E_c_ig(g) * sum(pat_E_tmp.yn_xn(:,:,2:m_i), 3);
                                          
                % End of testing
                
                E_sums{g} = E_tmp;
            end
        end

        
        function [all_models_new] = M_step(pat_data, E_sums_all_classes, RTS_arrs, E_c_ig_allpats, models_all_classes, controls, iter_num)
            % FUNCTION NAME:
            %   M_step
            %
            % DESCRIPTION:
            %   Finds the updated parameters of mixture model LSDSM keeping
            %   the hidden states fixed. This is the M step of the EM
            %   algorithm.
            %
            % INPUT:
            %   pat_data - (map) Contains the observed data for all
            %              patients.
            %   E_sums_all_classes - (cell) Contains the summations of expectations 
            %                        in structs for all patients for all
            %                        classes.
            %   RTS_arrs - (cell) Contains the output arrays obtained
            %              from the RTS smoother for every class in
            %              separate cells.
            %   E_c_ig_allpats - (array) Contains the expectations of
            %                    patient i belonging to class g for all
            %                    classes.
            %   models_all_classes - (map) Current model parameters for all
            %                        classes.
            %   controls - (struct) EM algorithm controls, including
            %              information on which parameters to keep fixed.
            %   iter_num - (double) Current EM iteration. Used to delay the
            %              updates of the class coefficients to ensure that
            %              the models have settled slightly before giving
            %              structure to prior probabilities.
            %
            % OUTPUT:
            %   all_models_new - (map) Updated model parameters for all
            %                    classes.
            %
            % REVISION HISTORY:
            %   17/12/2022 - mcauchi1
            %       * Initial implementation
            %   12/01/2023 - mcauchi1
            %       * Updated the function to include parameter updates for
            %       all classes
            %   22/02/2023 - mcauchi1
            %       * Included the functionality to retain the same
            %       selected parameters across classes
            %   15/03/2023 - mcauchi1
            %       * Included the update equation for the input matrix (B)
            %       which updates concurrently with the dynamics matrix (A)
            %       * Made the required changes for the disturbance matrix
            %       (W) in case input is present
            %   01/09/2023 - mcauchi1
            %       * Introduced the H matrix to have the survival function
            %       depend on a pre-determined linear combination of
            %       states.
            %

            % Number of patients
            num_pats = double(pat_data.Count);
            
            % number of iterations
            iter_obs_vector = LSDSM_MM_ALLFUNCS.extract_field_from_map(pat_data, 'm_i')';
            % class covariates
            class_cov_matrix = reshape(LSDSM_MM_ALLFUNCS.extract_field_from_map(pat_data, 'class_cov'), ...
                                            size(pat_data(1).class_cov,1), num_pats);
            
            N_totpat = sum(iter_obs_vector); % total number of observations across all patients
            
            % Find the indices for the dynamic states
            [idx_present_r, idx_present_c] = find(models_all_classes(1).G_mat ~= 0);
            
            num_classes = double(models_all_classes.Count); % number of classes
            
            % Booleans to check which parameters are required to be the same across all classes
            same_params = controls.same_params;
            
            % Map to store the new updated parameters
            all_models_new = containers.Map('KeyType', 'int32', 'ValueType', 'any');
            
            for g=1:num_classes % for every class
                % Identify the parameters to keep fixed and the current model parameters
                fixed_params = controls.fixed_params(g);
                model_g = models_all_classes(g);
                
                % Extract the summations of expectations for the current class
                E_sums = E_sums_all_classes{g};

                if isfield(model_g, 'B') % if there is input affecting the SSM
                    % Update hat(AB)
                    if same_params.A % if A is the same for all classes
                        % note: if A is the same for all classes, so is B
                        % if A and B are to be estimated
                        if g == 1 % if this is the first class we are estimating A
                            temp_sum1 = 0;
                            temp_sum2 = 0;
                            for gg=1:num_classes % sum expectations over all classes
                                temp_sum1 = temp_sum1 + E_sums_all_classes{gg}.barxn_hatxunneg1_from3;
                                temp_sum2 = temp_sum2 + E_sums_all_classes{gg}.hatxun_hatxun_from2_tillNneg1;
                            end
                            hatAB_bar_new = temp_sum1 * temp_sum2^-1;
                            hatAB_new = [model_g.A model_g.B];
                            hatAB_new(idx_present_r,:) = hatAB_bar_new;
                        else % if we already calculated A and B matrices for all classes
                            hatAB_new = [all_models_new(1).A all_models_new(1).B];
                        end
                        
                    else % if A and B are to be calculated for all classes
                        hatAB_bar_new = E_sums.barxn_hatxunneg1_from3 * E_sums.hatxun_hatxun_from2_tillNneg1^-1;
                        hatAB_new = [model_g.A model_g.B];
                        hatAB_new(idx_present_r,:) = hatAB_bar_new;
                    end
                    
                    A_new = hatAB_new(:, 1:size(model_g.A,2));
                    B_new = hatAB_new(:, size(model_g.A,2)+1:end);
                    
                    if not(isnan(fixed_params.A)) % if A is fixed (given)
                        A_new = fixed_params.A;
                    end
                    if not(isnan(fixed_params.B)) % if B is fixed (given)
                        B_new = fixed_params.B;
                    end
                    
                else % if there is no input to SSM
                    % Update A
                    if same_params.A % if A is the same for all classes
                        if not(isnan(fixed_params.A)) % if A is fixed (given)
                            A_new = fixed_params.A;
                        else % if A is to be estimated
                            if g == 1 % if this is the first class we are estimating A
                                temp_sum1 = 0;
                                temp_sum2 = 0;
                                for gg=1:num_classes % sum expectations over all classes
                                    temp_sum1 = temp_sum1 + E_sums_all_classes{gg}.barxn_xnneg1_from3;
                                    temp_sum2 = temp_sum2 + E_sums_all_classes{gg}.xn_xn_from2_tillNneg1;
                                end
                                A_bar_new = temp_sum1 * temp_sum2^-1;
                                A_new = model_g.A;
                                A_new(idx_present_r,:) = A_bar_new;
                            else % if we already calculated the A matrix for all classes
                                A_new = all_models_new(1).A;
                            end
                        end
                        
                    else % if A is to be calculated separately for all classes
                        if not(isnan(fixed_params.A)) % if A is fixed
                            A_new = fixed_params.A;
                        else % if A is to be calculated
                            A_bar_new = E_sums.barxn_xnneg1_from3 * E_sums.xn_xn_from2_tillNneg1^-1;
                            A_new = model_g.A;
                            A_new(idx_present_r,:) = A_bar_new;
                        end
                    end
                end

                % Updating C
                if same_params.C % if C is the same for all classes
                    if not(isnan(fixed_params.C)) % if C is fixed (given)
                        C_new = fixed_params.C;
                    else % if C is to be estimated
                        if g == 1 % if this is the first class we are estimating C
                            temp_sum1 = 0;
                            temp_sum2 = 0;
                            for gg=1:num_classes % sum expectations over all classes
                                temp_sum1 = temp_sum1 + E_sums_all_classes{gg}.yn_xn;
                                temp_sum2 = temp_sum2 + E_sums_all_classes{gg}.xn_xn;
                            end
                            C_new = temp_sum1 * temp_sum2^-1;
                        else % if we already calculated the C matrix for all classes
                            C_new = all_models_new(1).C;
                        end
                    end
                    
                else % if C is to be calculated separately for all classes
                    if not(isnan(fixed_params.C)) % if C is fixed
                        C_new = fixed_params.C;
                    else % if C is to be calculated
                        C_new = E_sums.yn_xn * E_sums.xn_xn^-1;
                    end
                end
                
                % Updating mu0
                if same_params.mu_0 % if mu0 is the same for all classes
                    if not(isnan(fixed_params.mu_0)) % if mu0 is fixed (given)
                        mu_0new = fixed_params.mu_0;
                    else % if mu0 is to be estimated
                        if g == 1 % if this is the first class we are estimating mu0                            
                            temp_sum1 = 0;

                            for gg=1:num_classes
                                temp_sum1 = temp_sum1 + E_sums_all_classes{gg}.x0;
                            end
                            mu_0new = (num_pats)^-1 * temp_sum1;
                        else % if we already calculated the mu0 matrix for all classes
                            mu_0new = all_models_new(1).mu_0;
                        end
                    end
                    
                else % if mu0 is to be calculated separately for all classes
                    if not(isnan(fixed_params.mu_0)) % if mu0 is fixed
                        mu_0new = fixed_params.mu_0;
                    else % if mu0 is to be calculated
                        % E[x(1)]
                        mu_0new = E_sums.x0 / sum(E_c_ig_allpats(:,g));
                    end
                end
                
                % place updated estimates in the map
                all_models_new(g) = struct('A', A_new, 'C', C_new, 'mu_0', mu_0new, ...
                                           'DeltaT', model_g.DeltaT, 'G_mat', model_g.G_mat, 'H_mat', model_g.H_mat);
                                       
                if isfield(model_g, 'B') % if there is input affecting the SSM
                    all_models_new(g) = setfield(all_models_new(g), 'B', B_new);
                end
            end
            
            % This M step is split in 2 for loops. This is because some
            % parameters (variance parameters) require the updates of other
            % parameters across all classes (not just the current class) in
            % case we have the same variance across all classes
            
            for g=1:num_classes % for every class
                % extract the fixed parameters for class g
                fixed_params = controls.fixed_params(g);
                % extract the parameter estimates for class g
                model_g = models_all_classes(g);
                % extract the summations of expectations for class g
                E_sums = E_sums_all_classes{g};
                
                % Updating W
                if same_params.W % if W is the same for all classes
                    if not(isnan(fixed_params.W)) % if W is fixed (given)
                        W_new = fixed_params.W;
                    else % if W is to be estimated
                        if g == 1 % if this is the first class we are estimating W
                            temp_sum1 = 0;

                            for gg=1:num_classes % sum expectations over all classes
                                A_bar_new = all_models_new(gg).A(idx_present_r,:);
                                temp_sum1 = temp_sum1 + (E_sums_all_classes{gg}.barxn_barxn_from2 ...
                                    - A_bar_new * E_sums_all_classes{gg}.barxn_xnneg1_from2' ...
                                    - E_sums_all_classes{gg}.barxn_xnneg1_from2 * A_bar_new' ...
                                    + A_bar_new * E_sums_all_classes{gg}.xn_xn_tillNneg1 * A_bar_new');
                                
                                if isfield(model_g, 'B')
                                    B_bar_new = all_models_new(gg).B(idx_present_r,:);
                                    temp_sum1 = temp_sum1 + ( - E_sums_all_classes{gg}.barxn_unneg1_from2 * B_bar_new' ...
                                        + A_bar_new * E_sums_all_classes{gg}.xn_un_tillNneg1 * B_bar_new' ...
                                        - B_bar_new * E_sums_all_classes{gg}.barxn_unneg1_from2' ...
                                        + B_bar_new * E_sums_all_classes{gg}.xn_un_tillNneg1' * A_bar_new' ...
                                        + B_bar_new * E_sums_all_classes{gg}.un_un_tillNneg1 * B_bar_new');
                                end
                            end
                            W_new = (N_totpat - num_pats)^-1 * temp_sum1;
                        else % if we already calculated the W matrix for all classes
                            W_new = all_models_new(1).W;
                        end
                    end
                    
                else % if W is to be calculated separately for all classes
                    if not(isnan(fixed_params.W)) % if W is fixed
                        W_new = fixed_params.W;
                    else % if W is to be calculated
                        
                        A_bar_new = all_models_new(g).A(idx_present_r,:); % retrieve updated A for class g
                        
                        temp_sum1 = (E_sums.barxn_barxn_from2 - A_bar_new * E_sums.barxn_xnneg1_from2' ...
                                     - E_sums.barxn_xnneg1_from2 * A_bar_new' ...
                                     + A_bar_new * E_sums.xn_xn_tillNneg1 * A_bar_new');
                                 
                        if isfield(model_g, 'B')
                            B_bar_new = all_models_new(g).B(idx_present_r,:);
                            temp_sum1 = temp_sum1 + ( - E_sums.barxn_unneg1_from2 * B_bar_new' ...
                                    + A_bar_new * E_sums.xn_un_tillNneg1 * B_bar_new' ...
                                    - B_bar_new * E_sums.barxn_unneg1_from2' ...
                                    + B_bar_new * E_sums.xn_un_tillNneg1' * A_bar_new' ...
                                    + B_bar_new * E_sums.un_un_tillNneg1 * B_bar_new');
                        end
                        W_new = (E_c_ig_allpats(:,g)' * (iter_obs_vector - 1))^-1 * temp_sum1;
                        
                        W_new = LSDSM_MM_ALLFUNCS.ensure_sym_mat(W_new);
                    end
                end
                
                % Updating V
                if same_params.V % if V is the same for all classes
                    if not(isnan(fixed_params.V)) % if V is fixed (given)
                        V_new = fixed_params.V;
                    else % if V is to be estimated
                        if g == 1 % if this is the first class we are estimating V
                            temp_sum1 = 0;

                            for gg=1:num_classes
                                C_new_tmp = all_models_new(gg).C; % retrieve updated C for class g
                                
                                temp_sum1 = temp_sum1 + (E_sums_all_classes{gg}.yn_yn ...
                                    - C_new_tmp * E_sums_all_classes{gg}.yn_xn' ...
                                    - E_sums_all_classes{gg}.yn_xn * C_new_tmp' ...
                                    + C_new_tmp * E_sums_all_classes{gg}.xn_xn * C_new_tmp');
                            end
                            V_new = (N_totpat)^-1 * temp_sum1;
                        else % if we already calculated the V matrix for all classes
                            V_new = all_models_new(1).V;
                        end
                    end
                    
                else % if V is to be calculated separately for all classes
                    if not(isnan(fixed_params.V)) % if V is fixed
                        V_new = fixed_params.V;
                    else % if V is to be calculated
                        C_new_tmp = all_models_new(g).C;
                        V_new = (E_c_ig_allpats(:,g)' * iter_obs_vector)^-1 * ...
                                    (E_sums.yn_yn - C_new_tmp * E_sums.yn_xn' ...
                                                  - E_sums.yn_xn * C_new_tmp' ...
                                                  + C_new_tmp * E_sums.xn_xn * C_new_tmp');
                        
                        V_new = LSDSM_MM_ALLFUNCS.ensure_sym_mat(V_new);
                    end
                end
                
                % Updating W_0
                if same_params.W_0 % if W_0 is the same for all classes
                    if not(isnan(fixed_params.W_0)) % if W_0 is fixed (given)
                        W_0new_tmp = fixed_params.W_0;
                    else % if W_0 is to be estimated
                        if g == 1 % if this is the first class we are estimating W_0
                            temp_sum1 = 0;

                            for gg=1:num_classes
                                mu0_new_tmp = all_models_new(gg).mu_0;
                                temp_sum1 = temp_sum1 + E_sums_all_classes{gg}.x0_x0 ...
                                    - sum(E_c_ig_allpats(:,gg)) * (mu0_new_tmp * mu0_new_tmp');
                            end
                            W_0new_tmp = (num_pats)^-1 * temp_sum1;
                        else % if we already calculated the W_0 matrix for all classes
                            W_0new_tmp = all_models_new(1).W_0;
                        end
                    end
                    
                else % if W_0 is to be calculated separately for all classes
                    if not(isnan(fixed_params.W_0)) % if W_0 is fixed
                        W_0new_tmp = fixed_params.W_0;
                    else % if W_0 is to be calculated
                        mu_0new = all_models_new(g).mu_0;
                        W_0new_tmp = E_sums.x0_x0 / sum(E_c_ig_allpats(:,g)) - mu_0new * mu_0new'; 
                        W_0new_tmp = LSDSM_MM_ALLFUNCS.ensure_sym_mat(W_0new_tmp);
                    end
                end
                
                % place updated estimates in the map
                all_models_new(g) = setfield(all_models_new(g), 'W', W_new);
                all_models_new(g) = setfield(all_models_new(g), 'V', V_new);
                all_models_new(g) = setfield(all_models_new(g), 'W_0', W_0new_tmp);
            end
            
            % Updating survival parameters
            % NR controls configuration
            NR_controls.max_iter = 100;
            NR_controls.eps = 1e-6;
            NR_controls.damp_factor = 0.5;
                        
            % Case 1: same gamma and alpha across all classes
            if same_params.g_s && same_params.a_s
                update_case = 1;
                
                % extract the first model (does not make a difference which in reality)
                model_g = models_all_classes(1);
                
                % Prepare coefficients to help identify the updates in NR
                dG_data = struct('pat_data', pat_data, 'model_coef_est', model_g, ...
                                 'RTS_arrs', {RTS_arrs}, 'controls', controls, ...
                                 'E_c_ig_allpats', E_c_ig_allpats, 'class_num', 1, ...
                                 'num_classes', num_classes, 'same_b_s', same_params.b_s);
                             
                
                % update the parameters
                if controls.mod_KF % if we are using the standard MM
                    limits_in_place = 0; % =1 -> used to stop NR procedure if hessian gets too small/large
                    g_a_s_new = LSDSM_MM_ALLFUNCS.Newton_Raphson_surv_params([model_g.g_s; model_g.a_s], NR_controls, ...
                                                                @LSDSM_MM_ALLFUNCS.surv_E_val, @LSDSM_MM_ALLFUNCS.dGdx, ...
                                                                @LSDSM_MM_ALLFUNCS.d2Gdx2, dG_data, update_case, limits_in_place);
                    
                else % if we are using the alternative mixture model extension
                    if strcmpi(controls.base_haz, 'Weibull')
                        g_a_s_new = LSDSM_MM_ALLFUNCS.Newton_Raphson_weibull_surv_params([model_g.g_s; model_g.a_s], NR_controls, ...
                                                                @LSDSM_MM_ALLFUNCS.weibull_E_val, @LSDSM_MM_ALLFUNCS.dGdx_alt, ...
                                                                @LSDSM_MM_ALLFUNCS.d2Gdx2_alt, dG_data, update_case);
                    end
                end
                
                % separate the gamma and alpha parameters
                g_s_new = g_a_s_new(1:size(model_g.g_s,1), 1);
                a_s_new = g_a_s_new(size(model_g.g_s,1)+1:end, 1);
                
                % store the same parameters across all classes
                for gg=1:num_classes
                    all_models_new(gg) = setfield(all_models_new(gg), 'g_s', g_s_new);
                    all_models_new(gg) = setfield(all_models_new(gg), 'a_s', a_s_new);
                end
            
            
            % Case 2: same gamma but different alpha across all classes
            elseif same_params.g_s && not(same_params.a_s)
                update_case = 2;
                
                % extract the first model (does not make a difference which in reality)
                model_g = models_all_classes(1);
                
                % Prepare coefficients to help identify the updates in NR
                dG_data = struct('pat_data', pat_data, 'model_coef_est', model_g, ...
                                 'RTS_arrs', {RTS_arrs}, 'controls', controls, ...
                                 'E_c_ig_allpats', E_c_ig_allpats, 'class_num', g, ...
                                 'num_classes', num_classes);

                
                % extract the same gamma vector and different alpha vectors across different classes
                g_a_s_old = [model_g.g_s];
                for gg=1:num_classes
                    model_g = models_all_classes(gg);
                    g_a_s_old = [g_a_s_old;
                                 model_g.a_s];
                end
                
                % update the parameters
                if controls.mod_KF 
                    limits_in_place = 0; % =1 -> used to stop NR procedure if hessian gets too small/large
                    g_a_s_new = LSDSM_MM_ALLFUNCS.Newton_Raphson_surv_params(g_a_s_old, NR_controls, ...
                                                                @LSDSM_MM_ALLFUNCS.surv_E_val, @LSDSM_MM_ALLFUNCS.dGdx_case2, ...
                                                                @LSDSM_MM_ALLFUNCS.d2Gdx2_case2, dG_data, update_case, limits_in_place);
                    
                else % if we are using the alternative mixture model extension
                    if strcmpi(controls.base_haz, 'Weibull')
                        g_a_s_new = LSDSM_MM_ALLFUNCS.Newton_Raphson_weibull_surv_params(g_a_s_old, NR_controls, ...
                                                                @LSDSM_MM_ALLFUNCS.weibull_E_val, @LSDSM_MM_ALLFUNCS.dGdx_alt_case2, ...
                                                                @LSDSM_MM_ALLFUNCS.d2Gdx2_alt_case2, dG_data, update_case);
                    end
                end
                
                % Find the sizes of the survival parameters
                size_g_s = size(model_g.g_s,1);
                size_a_s = size(model_g.a_s,1);
                
                % Extract the same gamma parameter vector
                g_s_new = g_a_s_new(1:size_g_s, 1);
                
                for gg=1:num_classes
                    % Extract the start and end indices for the parameter vector a_s for class gg
                    start_idx = size_g_s + (gg-1)*size_a_s + 1;
                    end_idx = size_g_s + gg*size_a_s;
                    
                    % Extract the alpha parameter vector for class gg
                    a_s_new = g_a_s_new(start_idx:end_idx, 1);
                    
                    % Store the new values in the updated map
                    all_models_new(gg) = setfield(all_models_new(gg), 'g_s', g_s_new);
                    all_models_new(gg) = setfield(all_models_new(gg), 'a_s', a_s_new);
                end
            
            
            % Case 3: different gamma but same alpha across all classes
            elseif not(same_params.g_s) && same_params.a_s
                update_case = 3;
                
                % extract the first model (does not make a difference which in reality)
                model_g = models_all_classes(1);
                
                % Prepare coefficients to help identify the updates in NR
                dG_data = struct('pat_data', pat_data, 'model_coef_est', model_g, ...
                                 'RTS_arrs', {RTS_arrs}, 'controls', controls, ...
                                 'E_c_ig_allpats', E_c_ig_allpats, 'class_num', g, ...
                                 'num_classes', num_classes);
                             
                % extract different gamma vectors across different classes and the same alpha vector
                g_a_s_old = [];
                for gg=1:num_classes
                    model_g = models_all_classes(gg);
                    g_a_s_old = [g_a_s_old;
                                 model_g.g_s];
                end
                
                g_a_s_old = [g_a_s_old;
                             model_g.a_s];
                
                % update the parameters
                if controls.mod_KF 
                    limits_in_place = 0; % =1 -> used to stop NR procedure if hessian gets too small/large
                    g_a_s_new = LSDSM_MM_ALLFUNCS.Newton_Raphson_surv_params(g_a_s_old, NR_controls, ...
                                                                @LSDSM_MM_ALLFUNCS.surv_E_val, @LSDSM_MM_ALLFUNCS.dGdx_case3, ...
                                                                @LSDSM_MM_ALLFUNCS.d2Gdx2_case3, dG_data, update_case, limits_in_place);
                    
                else % if we are using the alternative mixture model extension
                    if strcmpi(controls.base_haz, 'Weibull')
                        g_a_s_new = LSDSM_MM_ALLFUNCS.Newton_Raphson_weibull_surv_params(g_a_s_old, NR_controls, ...
                                                                @LSDSM_MM_ALLFUNCS.weibull_E_val, @LSDSM_MM_ALLFUNCS.dGdx_alt_case3, ...
                                                                @LSDSM_MM_ALLFUNCS.d2Gdx2_alt_case3, dG_data, update_case);
                    end
                end
                
                % Find the sizes of the survival parameters
                size_g_s = size(model_g.g_s,1);
                size_a_s = size(model_g.a_s,1);
                
                % Extract the alpha parameter vector
                a_s_new = g_a_s_new(end-size_a_s+1:end, 1);
                
                for gg=1:num_classes
                    % Extract the start and end indices for the parameter vector g_s for class gg
                    start_idx = (gg-1)*size_g_s + 1;
                    end_idx = gg*size_g_s;
                    
                    % Extract the gamma parameter vector for class gg
                    g_s_new = g_a_s_new(start_idx:end_idx, 1);
                    
                    % Store the new values in the updated map
                    all_models_new(gg) = setfield(all_models_new(gg), 'g_s', g_s_new);
                    all_models_new(gg) = setfield(all_models_new(gg), 'a_s', a_s_new);
                end
            
            % Case 4: different gamma and alpha across all classes
            else
                update_case = 4;
                
                for g=1:num_classes % for every class
                    % Extract model g
                    model_g = models_all_classes(g);
                    
                    % Prepare coefficients to help identify the updates in NR
                    dG_data = struct('pat_data', pat_data, 'model_coef_est', model_g, ...
                                     'RTS_arrs', RTS_arrs{g}, 'controls', controls, ...
                                     'E_c_ig', E_c_ig_allpats(:,g), 'class_num', g, ...
                                     'num_classes', num_classes);

                    % update the parameters for class g
                    if controls.mod_KF
                        limits_in_place = 0; % =1 -> used to stop NR procedure if hessian gets too small/large
                        g_a_s_new = LSDSM_MM_ALLFUNCS.Newton_Raphson_surv_params([model_g.g_s; model_g.a_s], NR_controls, ...
                                                                    @LSDSM_MM_ALLFUNCS.surv_E_val, @LSDSM_MM_ALLFUNCS.dGdx, ...
                                                                    @LSDSM_MM_ALLFUNCS.d2Gdx2, dG_data, update_case, limits_in_place);
                        
                    else % if we are using the alternative mixture model extension
                        if strcmpi(controls.base_haz, 'Weibull')
                            g_a_s_new = LSDSM_MM_ALLFUNCS.Newton_Raphson_weibull_surv_params([model_g.g_s; model_g.a_s], NR_controls, ...
                                                                    @LSDSM_MM_ALLFUNCS.weibull_E_val, @LSDSM_MM_ALLFUNCS.dGdx_alt, ...
                                                                    @LSDSM_MM_ALLFUNCS.d2Gdx2_alt, dG_data, update_case);
                        end
                    end

                    % separate the gamma and alpha parameters
                    g_s_new = g_a_s_new(1:size(model_g.g_s,1), 1);
                    a_s_new = g_a_s_new(size(model_g.g_s,1)+1:end, 1);
                    
                    % store the new values in the updated map
                    all_models_new(g) = setfield(all_models_new(g), 'g_s', g_s_new);
                    all_models_new(g) = setfield(all_models_new(g), 'a_s', a_s_new);
                    
                end
                
            end % end of survival cases updates
            
            if ~controls.mod_KF % if we are using the alternative mixture model extension
                if strcmpi(controls.base_haz, 'Weibull')
                    % update beta (scale parameter)
                    for gg=1:num_classes % for every class
                        % extract the gamma and alpha survival parameters
                        g_a_s_new = [all_models_new(gg).g_s; all_models_new(gg).a_s];
                        dG_data.E_c_ig = E_c_ig_allpats(:,gg); % set expectation for current class
                        
                        if update_case == 1 && same_params.b_s
                            % if we have same b_s across all classes, set
                            % responsibility for all patients equal to 1
                            dG_data.E_c_ig = ones(num_pats, 1);
                        end
                        b_s_new = LSDSM_MM_ALLFUNCS.weibull_beta_val(g_a_s_new, dG_data);
                        
                        % store the new beta in the updated map
                        all_models_new(gg) = setfield(all_models_new(gg), 'b_s', b_s_new);
                    end
                end
            end
            
            % Updating the class coefficients
            if iter_num > 10 % update if there have been at least 9 EM iterations

                if num_classes == 1 % if there is only one class
                    % retain the same values
                    zeta_new = models_all_classes(1).zeta;
                    all_models_new(1) = setfield(all_models_new(1), 'zeta', zeta_new);
                    all_models_new(1) = orderfields(all_models_new(1), models_all_classes(1));
                    
                else % if there is more than one class
                    % Store the required data to perform Newton Raphson
                    dzeta_data = struct('Q_mat', class_cov_matrix, 'E_c_ig_all', E_c_ig_allpats);

                    % Number of coefficients for every class
                    zeta_num_coeffs = size(model_g.zeta,1);
                    % Stack the coefficients for all classes
                    zeta_tmp = zeros(zeta_num_coeffs*num_classes,1);
                    for g=1:num_classes
                        zeta_tmp((g-1)*zeta_num_coeffs+1:g*zeta_num_coeffs,1) = models_all_classes(g).zeta;
                    end
                    
                    % Controls for NR optimisation
                    NR_controls = struct('max_iter', 100, 'eps', 1e-6, 'damp_factor', 0.5, 'minimising', 0);

                    % update the zeta parameters
                    zeta_new = LSDSM_MM_ALLFUNCS.Newton_Raphson(zeta_tmp, NR_controls, @LSDSM_MM_ALLFUNCS.f_zeta, ...
                                                                @LSDSM_MM_ALLFUNCS.dGdzeta, ...
                                                                @LSDSM_MM_ALLFUNCS.d2Gdzeta2, dzeta_data);

                    for g=1:num_classes % for every class
                        % store the coefficients for every model
                        all_models_new(g) = setfield(all_models_new(g), ...
                                                     'zeta', zeta_new((g-1)*zeta_num_coeffs+1:g*zeta_num_coeffs,1));
                                                 
                        % use the same order for the new model struct
                        % this is important for parameter comparison
                        all_models_new(g) = orderfields(all_models_new(g), models_all_classes(g));
                    end
                end
            else % do not update the zeta parameters yet
                for g=1:num_classes % for every class
                    % store the coefficients for every model
                    all_models_new(g) = setfield(all_models_new(g), ...
                                                 'zeta', models_all_classes(g).zeta);
                                             
                    % use the same order for the new model struct
                    % this is important for parameter comparison
                    all_models_new(g) = orderfields(all_models_new(g), models_all_classes(g));
                end
            end
            
        end

        
        function x_NR_tmp = Newton_Raphson(init_val, NR_controls, f_x, dfdx, d2fdx2, coeffs, limits_in_place)
            % FUNCTION NAME:
            %   Newton_Raphson
            %
            % DESCRIPTION:
            %   Newton Raphson's iterative method to find the roots (in
            %   this case, to find where the derivatives are at zero). If
            %   big jump is observed on updated x, this resorts to Damped
            %   Newton's method, which allows the solution to constantly
            %   optimise the function by ensuring that the step size leads
            %   to an increase/decrease in function.
            %
            % INPUT:
            %   init_val - (array) The initial value to start NR iterations
            %              from.
            %   NR_controls - (struct) Contains maximum number of
            %                 iteration, stopping criterion, damping
            %                 factor, and boolean to check if we are
            %                 minimising a function.
            %   f_x - (function name) Function to maximise finding the 
            %         value at the current x value.
            %   dfdx - (function name) Function finding the first
            %          derivative with respect to the variable of interest.
            %   d2fdx2 - (function name) Function finding the second
            %            derivative with respect to the variable of
            %            interest.
            %   coeffs - (struct) Contains all required data to calculate
            %            the above first and second derivatives.
            %   limits_in_place - (boolean) if true, it will break the NR
            %                     algorithm if the determinant of the
            %                     Hessian or its values get too small/big.
            %                     Default: true.
            %
            % OUTPUT:
            %   x_NR_tmp - (array) The stationary point for the function.
            %
            % REVISION HISTORY:
            %   17/12/2022 - mcauchi1
            %       * Initial implementation
            %   12/01/2023 - mcauchi1
            %       * Introduced rules for updating zeta
            %       * Introduced limits_in_place variable
            %   22/02/2023 - mcauchi1
            %       * Introduced damping in Newton's method to ensure that
            %       the solution is constantly improving
            %       * Improved execution time by only using damped Newton's
            %       method if a big change is observed in the x values
            %       * Introduced NR controls to retain all controls for the
            %       optimisation procedure
            %
            
            if (nargin<7) || isempty(limits_in_place) % if not defined
              limits_in_place = 1;
            end
            
            num_dims = size(init_val, 1); % dimension size
            
            % initialise array for storing single Newton-Raphson procedure
            x_NR_tmp_arr = zeros(num_dims, NR_controls.max_iter); % array for storing single Newton-Raphson procedure
            x_NR_tmp_arr(:,1) = init_val; % set initial value
            
            start_idx = 1; % changes if we are updating zeta
            if strcmp(functions(dfdx).function, 'LSDSM_MM_ALLFUNCS.dGdzeta')
                % if we are updating the class coefficients
                % - do not update the first class parameters

                % Find the number of classes and the number of
                % parameters in each class
                num_classes = size(coeffs.E_c_ig_all, 2);
                num_class_params = num_dims / num_classes;
                
                start_idx = num_class_params+1;
            end
            
            
            for jj=2:NR_controls.max_iter % NR iterations
                % Find the first derivative
                df = dfdx(x_NR_tmp_arr(:,jj-1), coeffs);
                % Find the second derivative
                d2f = d2fdx2(x_NR_tmp_arr(:,jj-1), coeffs);

                if limits_in_place
                    % preventing matrix errors during inverse operation
                    if abs(det(d2f)) < 1e-100
                        x_NR_tmp_arr(:,jj) = x_NR_tmp_arr(:,jj-1);
                        fprintf('limit\n');
                        break
                    end

                    if all(all(abs(d2f) > 1e100)) || all(all(abs(d2f) < 1e-100))
                        x_NR_tmp_arr(:,jj) = x_NR_tmp_arr(:,jj-1);
                        fprintf('limit\n');
                        break
                    end
                end
                
                % the magnitude and direction of the step change in parameters
                step_NR = d2f^(-1) * df;

                % update parameters
                x_NR_tmp_arr(:,jj) = x_NR_tmp_arr(:,jj-1); % copy the previous parameters
                x_NR_tmp_arr(start_idx:end,jj) = x_NR_tmp_arr(start_idx:end,jj-1) - step_NR;

                % Checking for largest change in x
                [max_chg, max_idx] = max(abs(x_NR_tmp_arr(:,jj-1) - x_NR_tmp_arr(:,jj)));
                frac_diff = abs(max_chg / x_NR_tmp_arr(max_idx,jj-1));

                % if the difference is too large (relatively).
                if frac_diff > 1 && abs(x_NR_tmp_arr(max_idx,jj-1)) > 1
                    % if big change is observed in x use damped Newton's
                    % method to ensure improvement in function
                    E_improved = 0; % boolean to check if solution improved
                    damping = 1; % Initialise the damping at 1 (no damping)

                    % calculate the value of the function we are maximising/minimising
                    E_new = f_x(x_NR_tmp_arr(:,jj-1), coeffs);
                    E_old = E_new;

                    % while E_new is complex, E_new has deteriorated, or E_new is NaN
                    while not(isreal(E_new)) || not(E_improved) || isnan(E_new)
                        % damped Newton's method - update parameters
                        x_NR_tmp_arr(:,jj) = x_NR_tmp_arr(:,jj-1); % copy the previous parameters
                        x_NR_tmp_arr(start_idx:end,jj) = x_NR_tmp_arr(start_idx:end,jj-1) - damping * step_NR;

                        % Evaluate new E value
                        E_new = f_x(x_NR_tmp_arr(:,jj), coeffs);

                        % make damping stronger
                        damping = NR_controls.damp_factor * damping;

                        E_improved = E_old <= E_new; % boolean to check if solution improved
                        if NR_controls.minimising % if we are finding the minimum
                            E_improved = E_old >= E_new; % improvement criteria changes
                        end

                        if max(abs(x_NR_tmp_arr(:,jj) - x_NR_tmp_arr(:,jj-1))) < NR_controls.eps
                            % max change is smaller than stopping criteria
                            break;
                        end

                        if NR_controls.damp_factor == 1 && not(E_improved) % no damping set
                            fprintf("Cannot improve with damped Newton's method");
                            break;
                        end

                    end
                end

                % Calculate the change in the differentiating variable
                chg = x_NR_tmp_arr(:,jj) - x_NR_tmp_arr(:,jj-1);
                if  max(abs(chg)) < NR_controls.eps % converged
                    break
                end
            end % end of for loop
            
            % Output the final computed value
            x_NR_tmp = x_NR_tmp_arr(:,jj);
        end
        
        
        function x_NR_tmp = Newton_Raphson_weibull_surv_params(init_val, NR_controls, f_x, dfdx, d2fdx2, coeffs, update_case)
            % FUNCTION NAME:
            %   Newton_Raphson_weibull_surv_params
            %
            % DESCRIPTION:
            %   Newton Raphson's iterative method to find the roots (in
            %   this case, to find where the derivatives are at zero). This
            %   is specific for the update equations of the survival
            %   parameters since it has many scenarios. This function
            %   assumes a Weibull baseline hazard function (alternative
            %   MM).
            %
            % INPUT:
            %   init_val - (array) The initial value to start NR iterations
            %              from.
            %   NR_controls - (struct) Contains maximum number of
            %                 iteration, stopping criterion, damping
            %                 factor, and boolean to check if we are
            %                 minimising a function.
            %   f_x - (function name) Function to maximise finding the 
            %         value at the current parameter values.
            %   dfdx - (function name) Function finding the first
            %          derivative with respect to the variable of interest.
            %   d2fdx2 - (function name) Function finding the second
            %            derivative with respect to the variable of
            %            interest.
            %   coeffs - (struct) Contains all required data to calculate
            %            the above first and second derivatives.
            %   limits_in_place - (boolean) if true, it will break the NR
            %                     algorithm if the determinant of the
            %                     Hessian or its values get too small/big.
            %                     Default: true.
            %
            % OUTPUT:
            %   x_NR_tmp - (array) The stationary point for the function.
            %
            % REVISION HISTORY:
            %   23/03/2022 - mcauchi1
            %       * Initial implementation
            %
            
            num_classes = length(coeffs.RTS_arrs); % number of classes
            num_dims = size(init_val, 1); % dimension size
            
            % initialise array for storing single Newton-Raphson procedure
            x_NR_tmp_arr = zeros(num_dims, NR_controls.max_iter); % array for storing single Newton-Raphson procedure
            x_NR_tmp_arr(:,1) = init_val; % set initial value
            
            % Check which parameters we need to update (in case of fixed parameters)
            dim_g_s = size(coeffs.model_coef_est.g_s,1); % size of gamma
            dim_a_s = size(coeffs.model_coef_est.a_s,1); % size of alpha
            updating_idx = 1:size(init_val,1); % stores the indices of parameters to be updated
            fixed_params_tmp = coeffs.controls.fixed_params; % fixed parameters for all classes
            
            % arrange the updating indices
            if update_case == 2 % same gamma different alpha
                
                for gg=1:num_classes % for every class
                    % Start from the back since we shall be removing indices
                    gk = num_classes-gg+1;

                    if not(isnan(fixed_params_tmp(gk).a_s)) % if a_s for class gk is fixed
                        % remove the indices corresponding to this parameter vector
                        idx1 = (gk-1)*dim_a_s + dim_g_s + 1;
                        idx2 = gk*dim_a_s + dim_g_s;
                        updating_idx(idx1:idx2) = [];
                    end
                end

                if not(isnan(fixed_params_tmp(1).g_s)) % if g_s is fixed
                    % remove the indices corresponding to the g_s parameter vector
                    updating_idx(1:dim_g_s) = [];
                end

            elseif update_case == 3

                if not(isnan(fixed_params_tmp(1).a_s)) % if a_s is fixed
                    % remove the indices corresponding to the a_s parameter vector
                    updating_idx(end-dim_a_s+1:end) = [];
                end
                
                for gg=1:num_classes % for every class
                    % Start from the back since we shall be removing indices
                    gk = num_classes-gg+1;

                    if not(isnan(fixed_params_tmp(gk).g_s)) % if g_s for class gk is fixed
                        % remove the indices corresponding to this parameter vector
                        idx1 = (gk-1)*dim_g_s + 1;
                        idx2 = gk*dim_g_s;
                        updating_idx(idx1:idx2) = [];
                    end
                end

            else % update_case == 1 || update_case == 4
                % Parameters to remain fixed throughout EM
                gk = coeffs.class_num;

                % if a_s is fixed
                if not(isnan(fixed_params_tmp(gk).a_s))
                    % do not update this parameter
                    updating_idx(dim_g_s+1:end) = [];
                end

                % if g_s is fixed
                if not(isnan(fixed_params_tmp(gk).g_s))
                    % do not update this parameter
                    updating_idx(1:dim_g_s) = [];
                end

            end
            
            for jj=2:NR_controls.max_iter % NR iterations
                
                if update_case == 1 % same gamma and alpha across classes
                    
                    if coeffs.same_b_s % if we have same b_s, then we have all survival parameters the same
                        % thus, we can set the expectation for every
                        % patient to 1, since we have to find 1 parameter
                        % for every class.
                        % Note that summing across all classes as is done
                        % in the else part would result into assuming that
                        % b_s being not the same.
                        coeffs_g = coeffs;
                        coeffs_g.E_c_ig = ones(size(coeffs.E_c_ig_allpats,1),1);
                        % Find the first derivative
                        df = dfdx(x_NR_tmp_arr(:,jj-1), coeffs_g);
                        % Find the second derivative
                        d2f = d2fdx2(x_NR_tmp_arr(:,jj-1), coeffs_g);
                        
                    else % if b_s is not the same across all classes
                        df = zeros(size(init_val));
                        d2f = zeros(size(init_val,1), size(init_val,1));
                        for g=1:num_classes % for every class
                            % extract the coefficients and store the
                            % responsibilities of class g
                            coeffs_g = coeffs;
                            coeffs_g.E_c_ig = coeffs.E_c_ig_allpats(:,g);
                            
                            % Find the first derivative
                            df_g = dfdx(x_NR_tmp_arr(:,jj-1), coeffs_g);
                            % Find the second derivative
                            d2f_g = d2fdx2(x_NR_tmp_arr(:,jj-1), coeffs_g);

                            % add the contribution for each class
                            df = df + df_g;
                            d2f = d2f + d2f_g;
                        end
                    end
                else
                    % case 2: same gamma different alpha across classes
                    % case 3: different gamma same alpha across classes
                    % case 4: different gamma and  alpha across classes
                    
                    % Find the first derivative
                    df = dfdx(x_NR_tmp_arr(:,jj-1), coeffs);
                    % Find the second derivative
                    d2f = d2fdx2(x_NR_tmp_arr(:,jj-1), coeffs);
                end

                % use Cholesky factorisation to ensure that the matrix is
                % symmetric positive definite. Since we are maximising the
                % function instead of minimising, we want to ensure that
                % the matrix is negative definite, and hence, we put a
                % minus sign in front of the hessian matrix (for the
                % check).
                [R, dir_flag] = chol(-d2f, 'lower');
                
                % filter those parameters that are to be updated
                df = df(updating_idx,1); 
                d2f = d2f(updating_idx,updating_idx);


                if dir_flag % if Hessian is not symmetric negative definite (for maximisation)
                    % use the simple backtracking optimisation
                    update_vec = df; % since we are maximising, we move in the direction of the gradient
                else
                    % use damped Newton's method
                    % Newton's direction is opposite to the gradient since we want to
                    % find the value that sends the gradient to zero
                    update_vec = - d2f^-1 * df;
                end

                E_val_new = NaN; % initialise the new expectation as NaN (ensures to update at least once)
                damping = 1; % initialise damping factor at 1 (no damping)

                % calculate the current expectation value
                E_val_prev = f_x(x_NR_tmp_arr(:,jj-1), coeffs, update_case);

                % while E_new is complex, E_new has not improved, or E_new is NaN
                while not(isreal(E_val_new)) || E_val_prev > E_val_new || isnan(E_val_new)

                    % retain the parameters that are set to fixed
                    x_NR_tmp_arr(:,jj) = x_NR_tmp_arr(:,jj-1); 
                    
                    % update the rest
                    x_NR_tmp_arr(updating_idx,jj) = x_NR_tmp_arr(updating_idx,jj-1) + damping * update_vec;

                    % calculate new expectation
                    E_val_new = f_x(x_NR_tmp_arr(:,jj), coeffs, update_case);

                    % update the damping factor
                    damping = damping * NR_controls.damp_factor;
                end

                % Calculate the change in the differentiating variable
                chg = x_NR_tmp_arr(:,jj) - x_NR_tmp_arr(:,jj-1);
                if max(abs(chg)) < NR_controls.eps % converged
                    break
                end
            end
            % Output the final computed value
            x_NR_tmp = x_NR_tmp_arr(:,jj);
        end
        
        
        function x_NR_tmp = Newton_Raphson_surv_params(init_val, NR_controls, f_x, dfdx, d2fdx2, coeffs, update_case, limits_in_place)
            % FUNCTION NAME:
            %   Newton_Raphson_surv_params
            %
            % DESCRIPTION:
            %   Newton Raphson's iterative method to find the roots (in
            %   this case, to find where the derivatives are at zero). This
            %   is specific for the update equations of the survival
            %   parameters since it has many scenarios (standard MM).
            %
            % INPUT:
            %   init_val - (array) The initial value to start NR iterations
            %              from.
            %   NR_controls - (struct) Contains maximum number of
            %                 iteration, stopping criterion, damping
            %                 factor, and boolean to check if we are
            %                 minimising a function.
            %   f_x - (function name) Function to maximise finding the 
            %         value at the current parameter values.
            %   dfdx - (function name) Function finding the first
            %          derivative with respect to the variable of interest.
            %   d2fdx2 - (function name) Function finding the second
            %            derivative with respect to the variable of
            %            interest.
            %   coeffs - (struct) Contains all required data to calculate
            %            the above first and second derivatives.
            %   update_case - (double) Provides information on which
            %                 survival parameters are kept the same across
            %                 classes.
            %   limits_in_place - (boolean) if true, it will break the NR
            %                     algorithm if the determinant of the
            %                     Hessian or its values get too small/big.
            %                     Default: true.
            %
            % OUTPUT:
            %   x_NR_tmp - (array) The stationary point for the function.
            %
            % REVISION HISTORY:
            %   17/12/2022 - mcauchi1
            %       * Initial implementation
            %   12/01/2023 - mcauchi1
            %       * Introduced rules for updating zeta
            %       * Introduced limits_in_place variable
            %   22/02/2023 - mcauchi1
            %       * Introduced NR controls to retain all controls for the
            %       optimisation procedure
            %   14/04/2023 - mcauchi1
            %       * Introduced function to check if parameters are being
            %       optimised
            %
            
            % Limits in place is used to stop the NR procedure if the
            % derivatives get too small. If =0, it will only stop if
            % converged.
            if (nargin<8) || isempty(limits_in_place) % if not defined
              limits_in_place = 1;
            end
            
            num_classes = length(coeffs.RTS_arrs); % number of classes
            num_dims = size(init_val, 1); % dimension size
            
            % initialise array for storing single Newton-Raphson procedure
            x_NR_tmp_arr = zeros(num_dims, NR_controls.max_iter); % array for storing single Newton-Raphson procedure
            x_NR_tmp_arr(:,1) = init_val; % set initial value
            
            % Check which parameters we need to update (in case of fixed parameters)
            dim_g_s = size(coeffs.model_coef_est.g_s,1); % size of gamma
            dim_a_s = size(coeffs.model_coef_est.a_s,1); % size of alpha
            updating_idx = 1:size(init_val,1); % stores the indices of parameters to be updated
            fixed_params_tmp = coeffs.controls.fixed_params; % fixed parameters for all classes
            
            % arrange the updating indices
            if update_case == 2 % same gamma different alpha
                
                for gg=1:num_classes % for every class
                    % Start from the back since we shall be removing indices
                    gk = num_classes-gg+1;

                    if not(isnan(fixed_params_tmp(gk).a_s)) % if a_s for class gk is fixed
                        % remove the indices corresponding to this parameter vector
                        idx1 = (gk-1)*dim_a_s + dim_g_s + 1;
                        idx2 = gk*dim_a_s + dim_g_s;
                        updating_idx(idx1:idx2) = [];
                    end
                end

                if not(isnan(fixed_params_tmp(1).g_s)) % if g_s is fixed
                    % remove the indices corresponding to the g_s parameter vector
                    updating_idx(1:dim_g_s) = [];
                end

            elseif update_case == 3

                if not(isnan(fixed_params_tmp(1).a_s)) % if a_s is fixed
                    % remove the indices corresponding to the a_s parameter vector
                    updating_idx(end-dim_a_s+1:end) = [];
                end
                
                for gg=1:num_classes % for every class
                    % Start from the back since we shall be removing indices
                    gk = num_classes-gg+1;

                    if not(isnan(fixed_params_tmp(gk).g_s)) % if g_s for class gk is fixed
                        % remove the indices corresponding to this parameter vector
                        idx1 = (gk-1)*dim_g_s + 1;
                        idx2 = gk*dim_g_s;
                        updating_idx(idx1:idx2) = [];
                    end
                end

            else % update_case == 1 || update_case == 4
                % Parameters to remain fixed throughout EM
                gk = coeffs.class_num;

                % if a_s is fixed
                if not(isnan(fixed_params_tmp(gk).a_s))
                    % do not update this parameter
                    updating_idx(dim_g_s+1:end) = [];
                end

                % if g_s is fixed
                if not(isnan(fixed_params_tmp(gk).g_s))
                    % do not update this parameter
                    updating_idx(1:dim_g_s) = [];
                end

            end
            
            for jj=2:NR_controls.max_iter % NR iterations
                
                if update_case == 1 % same gamma and alpha across classes
                    
                    % initialise derivatives at zero
                    df = zeros(size(init_val));
                    d2f = zeros(size(init_val,1), size(init_val,1));
                    
                    for g=1:num_classes % for every class
                        % store the coefficients and introduce the
                        % responsibility for this class
                        coeffs_g = coeffs;
                        coeffs_g.RTS_arrs = coeffs.RTS_arrs{g};
                        coeffs_g.E_c_ig = coeffs.E_c_ig_allpats(:,g);
                        
                        % Find the first derivative
                        df_g = dfdx(x_NR_tmp_arr(:,jj-1), coeffs_g);
                        % Find the second derivative
                        d2f_g = d2fdx2(x_NR_tmp_arr(:,jj-1), coeffs_g);
                        
                        % add the contribution of every class
                        df = df + df_g;
                        d2f = d2f + d2f_g;
                    end
                    
                else
                    % case 2: same gamma different alpha across classes
                    % case 3: different gamma same alpha across classes
                    % case 4: different gamma and  alpha across classes
                    
                    % Find the first derivative
                    df = dfdx(x_NR_tmp_arr(:,jj-1), coeffs);
                    % Find the second derivative
                    d2f = d2fdx2(x_NR_tmp_arr(:,jj-1), coeffs);
                end
                
                if limits_in_place
                    % preventing matrix errors during inverse operation
                    if abs(det(d2f)) < 1e-100
                        x_NR_tmp_arr(:,jj) = x_NR_tmp_arr(:,jj-1);
                        fprintf('limit\n');
                        break
                    end

                    if all(all(abs(d2f) > 1e100)) || all(all(abs(d2f) < 1e-100))
                        x_NR_tmp_arr(:,jj) = x_NR_tmp_arr(:,jj-1);
                        fprintf('limit\n');
                        break
                    end
                end

                % remove those entries that are not to be updated
                df = df(updating_idx,1); 
                d2f = d2f(updating_idx,updating_idx);
                
                % use Cholesky factorisation to ensure that the matrix is
                % symmetric positive definite. Since we are maximising the
                % function instead of minimising, we want to ensure that
                % the matrix is negative definite, and hence, we put a
                % minus sign in front of the hessian matrix (for the
                % check).
                [R, dir_flag] = chol(-d2f, 'lower');
                
                % if Hessian is not symmetric negative definite (for maximisation)
                if dir_flag || ~isfinite(det(d2f)) || det(d2f) == 0 
                    % use the simple backtracking optimisation
                    update_vec = df; % since we are maximising, we move in the direction of the gradient
                else
                    % use damped Newton's method
                    % Newton's direction is opposite to the gradient since we want to
                    % find the value that sends the gradient to zero
                    update_vec = - d2f^-1 * df;

                end

                E_val_new = NaN; % initialise the new expectation as NaN (ensures to update at least once)
                damping = 1; % initialise damping factor at 1 (no damping)

                % calculate the current expectation value
                E_val_prev = f_x(x_NR_tmp_arr(:,jj-1), coeffs, update_case);

                % while E_new is complex, E_new has not improved, or E_new is NaN
                while not(isreal(E_val_new)) || E_val_prev > E_val_new || isnan(E_val_new)

                    % retain the parameters that are set to fixed
                    x_NR_tmp_arr(:,jj) = x_NR_tmp_arr(:,jj-1); 
                    
                    % update the rest
                    x_NR_tmp_arr(updating_idx,jj) = x_NR_tmp_arr(updating_idx,jj-1) + damping * update_vec;

                    % calculate new expectation
                    E_val_new = f_x(x_NR_tmp_arr(:,jj), coeffs, update_case);

                    % update the damping factor
                    damping = damping * NR_controls.damp_factor;
                end

                % Calculate the change in the differentiating variable
                chg = x_NR_tmp_arr(:,jj) - x_NR_tmp_arr(:,jj-1);
                if max(abs(chg)) < NR_controls.eps % converged
                    break
                end
            end
            % Output the final computed value
            x_NR_tmp = x_NR_tmp_arr(:,jj);
        end

        
        function y = f_KF(x, coeffs)
            % FUNCTION NAME:
            %   f_KF
            %
            % DESCRIPTION:
            %   Finds the value of the exponent of the product of the
            %   Standard RTS filter and the survival likelihood, for the
            %   given hidden states.
            %
            % INPUT:
            %   x - (array) Hidden state values
            %   coeffs - (struct) Contains the required data to compute
            %            the function value at the given x.
            %
            % OUTPUT:
            %   y - (double) Function value at x
            %
            % REVISION HISTORY:
            %   22/02/2023 - mcauchi1
            %       * Initial implementation
            %   01/09/2023 - mcauchi1
            %       * Introduced the H matrix to have the survival function
            %       depend on a pre-determined linear combination of
            %       states.
            %
            
            y = - coeffs.delta_ij * (coeffs.g_s' * coeffs.base_cov + coeffs.a_s' * coeffs.H_mat * x) ...
                    + coeffs.tau_ij * exp(coeffs.g_s' * coeffs.base_cov) * exp(coeffs.a_s' * coeffs.H_mat * x) ...
                    + 1/2 * (x - coeffs.mu_ij)' * coeffs.Sigma_ij^(-1) * (x - coeffs.mu_ij);
        end
        
        
        function dfout = dfdx_KF(x_prev, coeffs)
            % FUNCTION NAME:
            %   dfdx_KF
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
            %   dfout - (array) The derivative with respect to the hidden
            %           states.
            %
            % REVISION HISTORY:
            %   17/12/2022 - mcauchi1
            %       * Initial implementation
            %   01/09/2023 - mcauchi1
            %       * Introduced the H matrix to have the survival function
            %       depend on a pre-determined linear combination of
            %       states.
            %
            
            dfout = (coeffs.tau_ij * exp(coeffs.g_s' * coeffs.base_cov) * exp(coeffs.a_s' * coeffs.H_mat * x_prev) ...
                            - coeffs.delta_ij) * coeffs.H_mat' * coeffs.a_s ...
                    + coeffs.Sigma_ij^(-1) * (x_prev - coeffs.mu_ij);
        end


        function d2fout = d2fdx2_KF(x_prev, coeffs)
            % FUNCTION NAME:
            %   d2fdx2_KF
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
            %   d2fout - (array) The second derivative with respect to the 
            %            hidden states.
            %
            % REVISION HISTORY:
            %   17/12/2022 - mcauchi1
            %       * Initial implementation
            %   01/09/2023 - mcauchi1
            %       * Introduced the H matrix to have the survival function
            %       depend on a pre-determined linear combination of
            %       states.
            %
            
            d2fout = coeffs.tau_ij * exp(coeffs.g_s' * coeffs.base_cov) * ... 
                        exp(coeffs.a_s' * coeffs.H_mat * x_prev) * (coeffs.H_mat' * coeffs.a_s * coeffs.a_s' * coeffs.H_mat) ...
                     + coeffs.Sigma_ij^(-1);
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
            %   01/09/2023 - mcauchi1
            %       * Introduced the H matrix to have the survival function
            %       depend on a pre-determined linear combination of
            %       states.
            %
            
            if isempty(coeffs.Omega_O) % if there are no observed values
                f1out = - coeffs.delta_ij * coeffs.a_s' * coeffs.H_mat * x_val ...
                         + coeffs.tau_ij * exp(coeffs.g_s' * coeffs.base_cov) * exp(coeffs.a_s' * coeffs.H_mat * x_val) ...
                         + (1/2) * (x_val - coeffs.pred_mu)' * coeffs.pred_V^(-1) * (x_val - coeffs.pred_mu);
            else % if there are some observed values
                % extract only the observed y values and calculate the
                % likelihood on those
                y_o = coeffs.Omega_O * coeffs.y_ij;
                C_o = coeffs.Omega_O * coeffs.C;
                V_o = coeffs.Omega_O * coeffs.V * coeffs.Omega_O';
                f1out = - coeffs.delta_ij * coeffs.a_s' * coeffs.H_mat * x_val ...
                         + coeffs.tau_ij * exp(coeffs.g_s' * coeffs.base_cov) * exp(coeffs.a_s' * coeffs.H_mat * x_val) ...
                         + (1/2) * (y_o - C_o * x_val)' * V_o^(-1) * (y_o - C_o * x_val) ...
                         + (1/2) * (x_val - coeffs.pred_mu)' * coeffs.pred_V^(-1) * (x_val - coeffs.pred_mu);
            end
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
            %   01/09/2023 - mcauchi1
            %       * Introduced the H matrix to have the survival function
            %       depend on a pre-determined linear combination of
            %       states.
            %
            
            df1out = (coeffs.tau_ij * exp(coeffs.g_s' * coeffs.base_cov) * exp(coeffs.a_s' * coeffs.H_mat * x_val) ...
                                - coeffs.delta_ij) * coeffs.H_mat' * coeffs.a_s ...
                      - coeffs.C' * coeffs.V^(-1) * (coeffs.y_ij - coeffs.C * x_val) ...
                      + coeffs.pred_V^(-1) * (x_val - coeffs.pred_mu);
        end
        
        
        function d2f1out = d2f1dx2(x_val, coeffs)
            % FUNCTION NAME:
            %   d2f1dx2
            %
            % DESCRIPTION:
            %   Finds the hessian matrix (second derivative) of the
            %   negative exponent of the likelihood function of the current
            %   time step at the current value of x.
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
            %   01/09/2023 - mcauchi1
            %       * Introduced the H matrix to have the survival function
            %       depend on a pre-determined linear combination of
            %       states.
            %
            
            d2f1out = coeffs.tau_ij * exp(coeffs.g_s' * coeffs.base_cov) * exp(coeffs.a_s' * coeffs.H_mat * x_val) ...
                                * (coeffs.H_mat' * coeffs.a_s * coeffs.a_s' * coeffs.H_mat) ...
                        + coeffs.C' * coeffs.V^(-1) * coeffs.C + coeffs.pred_V^(-1);
        end
        
        
        function E_val_part = surv_E_val_part(g_a_s_curr, E_data, E_c_ig)
            % FUNCTION NAME:
            %   surv_E_val_part
            %
            % DESCRIPTION:
            %   Evaluates part of the expectation which involves the
            %   survival parameters (standard MM).
            %
            % INPUT:
            %   g_a_s_curr - (array) Current parameter values of g_s and 
            %                a_s for all classes.
            %   E_data - (struct) Contains the required data to compute
            %            the expectation.
            %   E_c_ig - (array) Contains the responsibility of class g for
            %            every patient.
            %
            % OUTPUT:
            %   E_val_part - (double) Part of the expectation utilising the
            %                current values of the survival parameters.
            %
            % REVISION HISTORY:
            %   14/04/2023 - mcauchi1
            %       * Initial implementation
            %   01/09/2023 - mcauchi1
            %       * Introduced the H matrix to have the survival function
            %       depend on a pre-determined linear combination of
            %       states.
            %
            
            % sizes of g_s and a_s
            size_g_s = size(E_data.model_coef_est.g_s,1);
            
            % extract the gamma and alpha values
            g_s_prev = g_a_s_curr(1:size_g_s,1);
            a_s_prev = g_a_s_curr(size_g_s+1:end,1);
            
            E_val_part = 0;
            
            H_mat = E_data.model_coef_est.H_mat;

            for ii=1:E_data.pat_data.Count % For every patient
                % Extract the patient data
                pat_ii = E_data.pat_data(ii);
                % Extract the baseline covariates
                base_cov_ii = pat_ii.base_cov;
                
                % Extract the smoothed outputs
                mu_hat_tmp = E_data.RTS_arrs.mu_hat(:,:,:,ii);
                V_hat_tmp = E_data.RTS_arrs.V_hat(:,:,:,ii);
                
                % Temporary values to be scaled by E_c_ig
                E_val_part_tmp = 0;

                for j=1:pat_ii.m_i % For every observation
                    % Check if patient experienced event in current time frame
                    [delta_ij, tau_ij] = LSDSM_MM_ALLFUNCS.pat_status_at_j(j, pat_ii, E_data.model_coef_est.DeltaT);

                    % Auxiliary variable
                    scalar_tmp = tau_ij * exp(g_s_prev' * base_cov_ii) * ...
                        exp(a_s_prev' * H_mat * mu_hat_tmp(:,:,j) + 1/2 * a_s_prev' * H_mat * V_hat_tmp(:,:,j) * H_mat' * a_s_prev);

                    % add the contribution of this patient to the survival expectation
                    E_val_part_tmp = E_val_part_tmp + delta_ij * g_s_prev' * base_cov_ii ...
                                            + delta_ij * a_s_prev' * H_mat * mu_hat_tmp(:,:,j) - scalar_tmp;
                end
                
                % Add this patient's contributions to the gradients
                E_val_part = E_val_part + E_c_ig(ii) * E_val_part_tmp;
            end
        end
        
        
        function E_val = surv_E_val(g_a_s_curr, E_data, update_case)
            % FUNCTION NAME:
            %   surv_E_val
            %
            % DESCRIPTION:
            %   Evaluates the expectation only involving the survival
            %   parameters (standard MM). This method takes care of any
            %   update case (for all scenarios of retaining the same
            %   parameters across all classes).
            %
            % INPUT:
            %   g_a_s_curr - (array) Current parameter values of g_s and 
            %                a_s for all classes.
            %   E_data - (struct) Contains the required data to compute
            %            the expectation.
            %   update_case - (double) Informs the function of which
            %                 survival parameters are being kept the same.
            %
            % OUTPUT:
            %   E_val - (double) Expectation utilising only the current
            %           values of the survival parameters.
            %
            % REVISION HISTORY:
            %   14/04/2023 - mcauchi1
            %       * Initial implementation
            %
            
            % initialise the expectation to zero
            E_val = 0;
            
            % evaluate the number of classes
            num_classes = E_data.num_classes;
            
            size_g_s = size(E_data.model_coef_est.g_s,1);
            size_a_s = size(E_data.model_coef_est.a_s,1);
            
            if update_case == 1 % if both g_s and a_s are kept the same across all classes
                for gg=1:num_classes  % for every class
                    % store the previous values of g_s and a_s
                    g_s_prev = g_a_s_curr(1:size_g_s, 1);
                    a_s_prev = g_a_s_curr(size_g_s+1:end, 1);

                    % responsibilities for all patients for class g
                    E_c_ig = E_data.E_c_ig_allpats(:,gg);
                    
                    E_data_tmp = E_data;
                    E_data_tmp.RTS_arrs = E_data.RTS_arrs{gg};

                    % evaluate the expectation for this class and add
                    % its contribution to the total expectation
                    E_val = E_val + LSDSM_MM_ALLFUNCS.surv_E_val_part([g_s_prev; a_s_prev], E_data_tmp, E_c_ig);
                end
                
            elseif update_case == 2 % same gamma different alpha
                % Store the previous values of same g_s and different a_s across all classes
                g_s_prev = g_a_s_curr(1:size_g_s, 1);
                a_s_all_prev = g_a_s_curr(size_g_s+1:end, 1);
                
                for gg=1:num_classes % for every class
                    % Find the start and end indices to extract a_s of the current class gg
                    idx_stt = (gg-1)*size_a_s+1;
                    idx_end = gg*size_a_s;
                    a_s_prev = a_s_all_prev(idx_stt:idx_end,1);

                    % responsibilities for all patients for class g
                    E_c_ig = E_data.E_c_ig_allpats(:,gg);
                    
                    E_data_tmp = E_data;
                    E_data_tmp.RTS_arrs = E_data.RTS_arrs{gg};
                    
                    % evaluate the expectation for this class and add its
                    % contribution to the total expectation
                    E_val = E_val + LSDSM_MM_ALLFUNCS.surv_E_val_part([g_s_prev; a_s_prev], E_data_tmp, E_c_ig);
                end
                
            elseif update_case == 3 % different gamma same alpha
                % store the previous values of different g_s and same a_s across all classes
                g_s_all_prev = g_a_s_curr(1:end-size_a_s, 1);
                a_s_prev = g_a_s_curr(end-size_a_s+1:end, 1);
                
                for gg=1:num_classes
                    % Find the start and end indices to extract g_s of the current class gg
                    idx_stt = (gg-1)*size_g_s+1;
                    idx_end = gg*size_g_s;
                    g_s_prev = g_s_all_prev(idx_stt:idx_end,1);

                    % responsibilities for all patients for class g
                    E_c_ig = E_data.E_c_ig_allpats(:,gg);
                    
                    E_data_tmp = E_data;
                    E_data_tmp.RTS_arrs = E_data.RTS_arrs{gg};
                    
                    % evaluate the expectation for this class and add its
                    % contribution to the total expectation
                    E_val = E_val + LSDSM_MM_ALLFUNCS.surv_E_val_part([g_s_prev; a_s_prev], E_data_tmp, E_c_ig);
                end
                
            else % update_case == 4
                % store the previous values of g_s and a_s
                g_s_prev = g_a_s_curr(1:size_g_s, 1);
                a_s_prev = g_a_s_curr(size_g_s+1:end, 1);
                
                % evaluate the expectation for the current class
                E_val = LSDSM_MM_ALLFUNCS.surv_E_val_part([g_s_prev; a_s_prev], E_data, E_data.E_c_ig);
                
            end
        end
        
        
        function dGout = dGdx(g_a_s_prev, dG_data)
            % FUNCTION NAME:
            %   dGdx
            %
            % DESCRIPTION:
            %   First Derivative of the Expectation with respect to the
            %   survival parameters: gamma (g_s) and alpha (a_s). This
            %   function assumes a single class (standard MM).
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
            %   01/09/2023 - mcauchi1
            %       * Introduced the H matrix to have the survival function
            %       depend on a pre-determined linear combination of
            %       states.
            %

            % Store the previous values of g_s and a_s
            g_s_prev = g_a_s_prev(1:size(dG_data.pat_data(1).base_cov,1), 1);
            a_s_prev = g_a_s_prev(size(dG_data.pat_data(1).base_cov,1)+1:end, 1);
            
            H_mat = dG_data.model_coef_est.H_mat;

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
                
                % Temporary values to be scaled by E_c_ig
                dGdg_s_tmp = 0;
                dGda_s_tmp = 0;

                for j=1:pat_ii.m_i % For every observation
                    % Check if patient experienced event in current time frame
                    [delta_ij, tau_ij] = LSDSM_MM_ALLFUNCS.pat_status_at_j(j, pat_ii, dG_data.model_coef_est.DeltaT);

                    % Auxiliary variable
                    scalar_tmp = tau_ij * exp(g_s_prev' * base_cov_ii) * ...
                                    exp(a_s_prev' * H_mat * mu_hat_tmp(:,:,j) + 1/2 * a_s_prev' * H_mat * V_hat_tmp(:,:,j) * H_mat' * a_s_prev);

                    % Work out the derivatives with respect to survival parameters
                    dGdg_s_tmp = dGdg_s_tmp + delta_ij * base_cov_ii - scalar_tmp * base_cov_ii;
                    dGda_s_tmp = dGda_s_tmp + delta_ij * H_mat * mu_hat_tmp(:,:,j) ...
                                        - scalar_tmp * (H_mat * mu_hat_tmp(:,:,j) + H_mat * V_hat_tmp(:,:,j) * H_mat' * a_s_prev);
                end
                
                % Add this patient's contributions to the gradients
                dGdg_s = dGdg_s + dG_data.E_c_ig(ii) * dGdg_s_tmp;
                dGda_s = dGda_s + dG_data.E_c_ig(ii) * dGda_s_tmp;
            end

            % Store in the output vector
            dGout = [dGdg_s; dGda_s];
        end

        
        function d2Gout = d2Gdx2(g_a_s_prev, dG_data)
            % FUNCTION NAME:
            %   d2Gdx2
            %
            % DESCRIPTION:
            %   Hessian matrix (second derivative) of the Expectation with
            %   respect to the survival parameters: gamma (g_s) and alpha
            %   (a_s). This function assumes a single class (standard MM).
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
            %   01/09/2023 - mcauchi1
            %       * Introduced the H matrix to have the survival function
            %       depend on a pre-determined linear combination of
            %       states.
            %

            % Store the previous values of g_s and a_s
            g_s_prev = g_a_s_prev(1:size(dG_data.pat_data(1).base_cov,1), 1);
            a_s_prev = g_a_s_prev(size(dG_data.pat_data(1).base_cov,1)+1:end, 1);
            
            H_mat = dG_data.model_coef_est.H_mat;

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
                
                d2Gdg_s2_tmp = 0;
                d2Gdg_a_s_tmp = 0;
                d2Gda_s2_tmp = 0;
                
                for j=1:pat_ii.m_i % For every observation
                    % Check if patient experienced event in current time frame
                    [delta_ij, tau_ij] = LSDSM_MM_ALLFUNCS.pat_status_at_j(j, pat_ii, dG_data.model_coef_est.DeltaT);

                    % Auxiliary variables
                    scalar_tmp = tau_ij * exp(g_s_prev' * base_cov_ii) * ...
                                    exp(a_s_prev' * H_mat * mu_hat_tmp(:,:,j) + 1/2 * a_s_prev' * H_mat * V_hat_tmp(:,:,j) * H_mat' * a_s_prev);

                    mu_sigma_alpha = (H_mat * mu_hat_tmp(:,:,j) + H_mat * V_hat_tmp(:,:,j) * H_mat' * a_s_prev);
                    
                    % Second derivative calculations
                    d2Gdg_s2_tmp = d2Gdg_s2_tmp - scalar_tmp * (base_cov_ii * base_cov_ii');
                    d2Gdg_a_s_tmp = d2Gdg_a_s_tmp - scalar_tmp * (base_cov_ii * mu_sigma_alpha');
                    d2Gda_s2_tmp = d2Gda_s2_tmp - scalar_tmp * ((mu_sigma_alpha * mu_sigma_alpha') + H_mat * V_hat_tmp(:,:,j) * H_mat');

                end
                % Add this patient's contributions to the second derivatives
                d2Gdg_s2 = d2Gdg_s2 + dG_data.E_c_ig(ii) * d2Gdg_s2_tmp;
                d2Gdg_a_s = d2Gdg_a_s + dG_data.E_c_ig(ii) * d2Gdg_a_s_tmp;
                d2Gda_s2 = d2Gda_s2 + dG_data.E_c_ig(ii) * d2Gda_s2_tmp;
            end

            % Store the Hessian matrix in the correct format
            d2Gout = [d2Gdg_s2, d2Gdg_a_s;
                      d2Gdg_a_s', d2Gda_s2];
        end

        
        function dGout = dGdx_case2(g_a_s_prev, dG_data)
            % FUNCTION NAME:
            %   dGdx_case2
            %
            % DESCRIPTION:
            %   First Derivative of the Expectation with respect to the
            %   survival parameters: gamma (g_s) and alpha (a_s), where it
            %   is assumed that g_s is the same across all classes, while
            %   a_s is different across classes (standard MM).
            %
            % INPUT:
            %   g_a_s_prev - (array) Previous parameter values of g_s and
            %                a_s. Only one g_s vector should be in this
            %                array, while the a_s across all classes should
            %                be included.
            %   dG_data - (struct) Contains the required data to compute
            %             the derivative.
            %
            % OUTPUT:
            %   dGout - (array) The derivative of the expectation with
            %           respect to the survival parameters.
            %
            % REVISION HISTORY:
            %   07/02/2023 - mcauchi1
            %       * Initial implementation
            %   01/09/2023 - mcauchi1
            %       * Introduced the H matrix to have the survival function
            %       depend on a pre-determined linear combination of
            %       states.
            %
                             
            % Identify the nmber of classes
            num_classes = length(dG_data.RTS_arrs);
            % Retrieve the SSM time step
            DeltaT_tmp = dG_data.model_coef_est.DeltaT;
            % Store the sizes for gamma and alpha
            size_g_s = size(dG_data.model_coef_est.g_s,1);
            size_a_s = size(dG_data.model_coef_est.a_s,1);
            
            % Store the previous values of same g_s and different a_s across all classes
            g_s_prev = g_a_s_prev(1:size_g_s, 1);
            a_s_all_prev = g_a_s_prev(size_g_s+1:end, 1);
            
            H_mat = dG_data.model_coef_est.H_mat;

            % Initiate the derivatives to zero
            dGdg_s = zeros(size(g_s_prev));
            dGda_s = zeros(size(a_s_all_prev));

            for ii=1:dG_data.pat_data.Count % For every patient
                % Extract the patient data
                pat_ii = dG_data.pat_data(ii);
                % Extract the baseline covariates
                base_cov_ii = pat_ii.base_cov;
                
                for gg=1:num_classes
                    % Extract the smoothed outputs for current class
                    mu_hat_tmp = dG_data.RTS_arrs{gg}.mu_hat(:,:,:,ii);
                    V_hat_tmp = dG_data.RTS_arrs{gg}.V_hat(:,:,:,ii);
                
                    % Temporary values to be scaled by E_c_ig
                    dGdg_s_tmp = zeros(size_g_s, 1);
                    dGda_s_tmp = zeros(size_a_s, 1);
                    
                    % Find the start and end indices to extract a_s of the current class gg
                    idx_stt = (gg-1)*size_a_s+1;
                    idx_end = gg*size_a_s;
                    a_s_tmp = a_s_all_prev(idx_stt:idx_end,1);

                    for j=1:pat_ii.m_i % For every observation
                        % Check if patient experienced event in current time frame
                        [delta_ij, tau_ij] = LSDSM_MM_ALLFUNCS.pat_status_at_j(j, pat_ii, DeltaT_tmp);

                        % Auxiliary variable
                        scalar_tmp = tau_ij * exp(g_s_prev' * base_cov_ii) * ...
                            exp(a_s_tmp' * H_mat * mu_hat_tmp(:,:,j) + 1/2 * a_s_tmp' * H_mat * V_hat_tmp(:,:,j) * H_mat' * a_s_tmp);

                        % Work out the derivatives with respect to survival parameters
                        dGdg_s_tmp = dGdg_s_tmp + delta_ij * base_cov_ii - scalar_tmp * base_cov_ii;
                        dGda_s_tmp = dGda_s_tmp + delta_ij * H_mat * mu_hat_tmp(:,:,j) - ...
                            scalar_tmp * (H_mat * mu_hat_tmp(:,:,j) + H_mat * V_hat_tmp(:,:,j) * H_mat' * a_s_tmp);
                    end

                    % Add this patient's contribution to the gradients
                    dGdg_s = dGdg_s + dG_data.E_c_ig_allpats(ii,gg) * dGdg_s_tmp;
                    dGda_s(idx_stt:idx_end,1) = dGda_s(idx_stt:idx_end,1) + dG_data.E_c_ig_allpats(ii,gg) * dGda_s_tmp;
                end
            end

            % Store the gradients in the output vector
            dGout = [dGdg_s; dGda_s];
        end
        
        
        function d2Gout = d2Gdx2_case2(g_a_s_prev, dG_data)
            % FUNCTION NAME:
            %   d2Gdx2_case2
            %
            % DESCRIPTION:
            %   Hessian matrix (second derivative) of the Expectation with
            %   respect to the survival parameters: gamma (g_s) and alpha
            %   (a_s), where it is assumed that g_s is the same across all
            %   classes, while a_s is different across classes (standard
            %   MM).
            %
            % INPUT:
            %   g_a_s_prev - (array) Previous parameter values of g_s and
            %                a_s. Only one g_s vector should be in this
            %                array, while the a_s across all classes should
            %                be included.
            %   dG_data - (struct) Contains the required data to compute
            %             the derivative.
            %
            % OUTPUT:
            %   d2Gout - (array) The second derivative of the expectation 
            %            with respect to the survival parameters.
            %
            % REVISION HISTORY:
            %   07/02/2023 - mcauchi1
            %       * Initial implementation
            %   01/09/2023 - mcauchi1
            %       * Introduced the H matrix to have the survival function
            %       depend on a pre-determined linear combination of
            %       states.
            %

            % Identify the nmber of classes
            num_classes = length(dG_data.RTS_arrs);
            % Retrieve the SSM time step
            DeltaT_tmp = dG_data.model_coef_est.DeltaT;
            % Store the sizes for gamma and alpha
            size_g_s = size(dG_data.model_coef_est.g_s, 1);
            size_a_s = size(dG_data.model_coef_est.a_s, 1);
            
            % Store the previous values of same g_s and different a_s
            % across all classes
            g_s_prev = g_a_s_prev(1:size_g_s, 1);
            a_s_all_prev = g_a_s_prev(size_g_s+1:end, 1);
            
            H_mat = dG_data.model_coef_est.H_mat;

            % Initiate the second derivatives to zero
            d2Gdg_s2 = zeros(size_g_s, size_g_s);
            d2Gdg_a_s = zeros(size_g_s, size(a_s_all_prev,1));
            d2Gda_s2 = zeros(size(a_s_all_prev,1), size(a_s_all_prev,1));

            for ii=1:dG_data.pat_data.Count % For every patient
                % Extract patient data
                pat_ii = dG_data.pat_data(ii);
                % Extract baseline covariates
                base_cov_ii = pat_ii.base_cov;
                
                for gg=1:num_classes
                    % Extract the smoothed outputs for current class
                    mu_hat_tmp = dG_data.RTS_arrs{gg}.mu_hat(:,:,:,ii);
                    V_hat_tmp = dG_data.RTS_arrs{gg}.V_hat(:,:,:,ii);
                
                    % Temporary values to be scaled by E_c_ig
                    d2Gdg_s2_tmp = zeros(size_g_s, size_g_s);
                    d2Gdg_a_s_tmp = zeros(size_g_s, size_a_s);
                    d2Gda_s2_tmp = zeros(size_a_s, size_a_s);
                    
                    % Find the start and end indices to extract a_s of the
                    % current class gg
                    idx_stt = (gg-1)*size_a_s+1;
                    idx_end = gg*size_a_s;
                    a_s_tmp = a_s_all_prev(idx_stt:idx_end,1);
                
                    for j=1:pat_ii.m_i % For every observation
                        % Check if patient experienced event in current time frame
                        [delta_ij, tau_ij] = LSDSM_MM_ALLFUNCS.pat_status_at_j(j, pat_ii, DeltaT_tmp);

                        % Auxiliary variables
                        scalar_tmp = tau_ij * exp(g_s_prev' * base_cov_ii) * ...
                            exp(a_s_tmp' * H_mat * mu_hat_tmp(:,:,j) + 1/2 * a_s_tmp' * H_mat * V_hat_tmp(:,:,j) * H_mat' * a_s_tmp);

                        mu_sigma_alpha = (H_mat * mu_hat_tmp(:,:,j) + H_mat * V_hat_tmp(:,:,j) * H_mat' * a_s_tmp);

                        % Second derivative calculations
                        d2Gdg_s2_tmp = d2Gdg_s2_tmp - scalar_tmp * (base_cov_ii * base_cov_ii');
                        d2Gdg_a_s_tmp = d2Gdg_a_s_tmp - scalar_tmp * (base_cov_ii * mu_sigma_alpha');
                        d2Gda_s2_tmp = d2Gda_s2_tmp - scalar_tmp * ((mu_sigma_alpha * mu_sigma_alpha') + H_mat * V_hat_tmp(:,:,j) * H_mat');

                    end
                    
                    % Add this patient's contributions to the second derivatives
                    d2Gdg_s2 = d2Gdg_s2 + dG_data.E_c_ig_allpats(ii,gg) * d2Gdg_s2_tmp;
                    d2Gdg_a_s(:,idx_stt:idx_end) = d2Gdg_a_s(:,idx_stt:idx_end) + dG_data.E_c_ig_allpats(ii,gg) * d2Gdg_a_s_tmp;
                    d2Gda_s2(idx_stt:idx_end, idx_stt:idx_end) ...
                            = d2Gda_s2(idx_stt:idx_end, idx_stt:idx_end) + dG_data.E_c_ig_allpats(ii,gg) * d2Gda_s2_tmp;
                end
            end

            % Store the Hessian matrix in the correct format
            d2Gout = [d2Gdg_s2, d2Gdg_a_s;
                      d2Gdg_a_s', d2Gda_s2];
        end
        
        
        function dGout = dGdx_case3(g_a_s_prev, dG_data)
            % FUNCTION NAME:
            %   dGdx_case3
            %
            % DESCRIPTION:
            %   First Derivative of the Expectation with respect to the
            %   survival parameters: gamma (g_s) and alpha (a_s), where it
            %   is assumed that g_s is different across classes, while a_s  
            %   is the same across all classes (standard MM).
            %
            % INPUT:
            %   g_a_s_prev - (array) Previous parameter values of g_s and
            %                a_s. The g_s across all classes should be
            %                included, while only one a_s should be in this
            %                array.
            %   dG_data - (struct) Contains the required data to compute
            %             the derivative.
            %
            % OUTPUT:
            %   dGout - (array) The derivative of the expectation with
            %           respect to the survival parameters.
            %
            % REVISION HISTORY:
            %   07/02/2023 - mcauchi1
            %       * Initial implementation
            %   01/09/2023 - mcauchi1
            %       * Introduced the H matrix to have the survival function
            %       depend on a pre-determined linear combination of
            %       states.
            %
                             
            % Identify the nmber of classes
            num_classes = length(dG_data.RTS_arrs);
            % Retrieve the SSM time step
            DeltaT_tmp = dG_data.model_coef_est.DeltaT;
            % Store the sizes for gamma and alpha
            size_g_s = size(dG_data.model_coef_est.g_s, 1);
            size_a_s = size(dG_data.model_coef_est.a_s, 1);
            
            % Store the previous values of same g_s and different a_s across all classes
            g_s_all_prev = g_a_s_prev(1:end-size_a_s, 1);
            a_s_prev = g_a_s_prev(end-size_a_s+1:end, 1);
            
            H_mat = dG_data.model_coef_est.H_mat;

            % Initiate the derivatives to zero
            dGdg_s = zeros(size(g_s_all_prev));
            dGda_s = zeros(size(a_s_prev));

            for ii=1:dG_data.pat_data.Count % For every patient
                % Extract the patient data
                pat_ii = dG_data.pat_data(ii);
                % Extract the baseline covariates
                base_cov_ii = pat_ii.base_cov;
                
                for gg=1:num_classes
                    % Extract the smoothed outputs for current class
                    mu_hat_tmp = dG_data.RTS_arrs{gg}.mu_hat(:,:,:,ii);
                    V_hat_tmp = dG_data.RTS_arrs{gg}.V_hat(:,:,:,ii);
                
                    % Temporary values to be scaled by E_c_ig
                    dGdg_s_tmp = zeros(size_g_s, 1);
                    dGda_s_tmp = zeros(size_a_s, 1);
                    
                    % Find the start and end indices to extract g_s of the current class gg
                    idx_stt = (gg-1)*size_g_s+1;
                    idx_end = gg*size_g_s;
                    g_s_tmp = g_s_all_prev(idx_stt:idx_end,1);

                    for j=1:pat_ii.m_i % For every observation
                        % Check if patient experienced event in current time frame
                        [delta_ij, tau_ij] = LSDSM_MM_ALLFUNCS.pat_status_at_j(j, pat_ii, DeltaT_tmp);

                        % Auxiliary variable
                        scalar_tmp = tau_ij * exp(g_s_tmp' * base_cov_ii) * ...
                            exp(a_s_prev' * H_mat * mu_hat_tmp(:,:,j) + 1/2 * a_s_prev' * H_mat * V_hat_tmp(:,:,j) * H_mat' * a_s_prev);

                        % Work out the derivatives with respect to survival parameters
                        dGdg_s_tmp = dGdg_s_tmp + delta_ij * base_cov_ii - scalar_tmp * base_cov_ii;
                        dGda_s_tmp = dGda_s_tmp + delta_ij * H_mat * mu_hat_tmp(:,:,j) - ...
                            scalar_tmp * (H_mat * mu_hat_tmp(:,:,j) + H_mat * V_hat_tmp(:,:,j) * H_mat' * a_s_prev);
                    end

                    % Add this patient's contributions to the gradients
                    dGdg_s(idx_stt:idx_end,1) = dGdg_s(idx_stt:idx_end,1) + dG_data.E_c_ig_allpats(ii,gg) * dGdg_s_tmp;
                    dGda_s = dGda_s + dG_data.E_c_ig_allpats(ii,gg) * dGda_s_tmp;
                end
            end

            % Store in the output vector
            dGout = [dGdg_s; dGda_s];
        end
        
        
        function d2Gout = d2Gdx2_case3(g_a_s_prev, dG_data)
            % FUNCTION NAME:
            %   d2Gdx2_case3
            %
            % DESCRIPTION:
            %   Hessian matrix (second derivative) of the Expectation with
            %   respect to the survival parameters: gamma (g_s) and alpha
            %   (a_s), where it is assumed that g_s is different across 
            %   classes, while a_s is the same across all classes (standard
            %   MM).
            %
            % INPUT:
            %   g_a_s_prev - (array) Previous parameter values of g_s for
            %                all classes and the same a_s. The g_s across
            %                all classes should be included, while only one
            %                a_s should be in this array.
            %   dG_data - (struct) Contains the required data to compute
            %             the derivative.
            %
            % OUTPUT:
            %   d2Gout - (array) The second derivative of the expectation 
            %            with respect to the survival parameters.
            %
            % REVISION HISTORY:
            %   07/02/2023 - mcauchi1
            %       * Initial implementation
            %   01/09/2023 - mcauchi1
            %       * Introduced the H matrix to have the survival function
            %       depend on a pre-determined linear combination of
            %       states.
            %

            % Identify the nmber of classes
            num_classes = length(dG_data.RTS_arrs);
            % Retrieve the SSM time step
            DeltaT_tmp = dG_data.model_coef_est.DeltaT;
            % Store the sizes for gamma and alpha
            size_g_s = size(dG_data.model_coef_est.g_s, 1);
            size_a_s = size(dG_data.model_coef_est.a_s, 1);
            
            % Store the previous values of same g_s and different a_s across all classes
            g_s_all_prev = g_a_s_prev(1:size_g_s*num_classes, 1);
            a_s_prev = g_a_s_prev(num_classes*size_g_s+1:end, 1);
            
            H_mat = dG_data.model_coef_est.H_mat;

            % Initiate the second derivatives to zero
            d2Gdg_s2 = zeros(size(g_s_all_prev,1), size(g_s_all_prev,1));
            d2Gdg_a_s = zeros(size(g_s_all_prev,1), size_a_s);
            d2Gda_s2 = zeros(size_a_s, size_a_s);

            for ii=1:dG_data.pat_data.Count % For every patient
                % Extract patient data
                pat_ii = dG_data.pat_data(ii);
                % Extract baseline covariates
                base_cov_ii = pat_ii.base_cov;
                
                for gg=1:num_classes
                    % Extract the smoothed outputs for current class
                    mu_hat_tmp = dG_data.RTS_arrs{gg}.mu_hat(:,:,:,ii);
                    V_hat_tmp = dG_data.RTS_arrs{gg}.V_hat(:,:,:,ii);
                
                    % Temporary values to be scaled by E_c_ig
                    d2Gdg_s2_tmp = zeros(size_g_s, size_g_s);
                    d2Gdg_a_s_tmp = zeros(size_g_s, size_a_s);
                    d2Gda_s2_tmp = zeros(size_a_s, size_a_s);
                    
                    % Find the start and end indices to extract g_s of the current class gg
                    idx_stt = (gg-1)*size_g_s+1;
                    idx_end = gg*size_g_s;
                    g_s_tmp = g_s_all_prev(idx_stt:idx_end,1);
                
                    for j=1:pat_ii.m_i % For every observation
                        % Check if patient experienced event in current time frame
                        [delta_ij, tau_ij] = LSDSM_MM_ALLFUNCS.pat_status_at_j(j, pat_ii, DeltaT_tmp);

                        % Auxiliary variables
                        scalar_tmp = tau_ij * exp(g_s_tmp' * base_cov_ii) * ...
                            exp(a_s_prev' * H_mat * mu_hat_tmp(:,:,j) + 1/2 * a_s_prev' * H_mat * V_hat_tmp(:,:,j) * H_mat' * a_s_prev);

                        mu_sigma_alpha = (H_mat * mu_hat_tmp(:,:,j) + H_mat * V_hat_tmp(:,:,j) * H_mat' * a_s_prev);

                        % Second derivative calculations
                        d2Gdg_s2_tmp = d2Gdg_s2_tmp - scalar_tmp * (base_cov_ii * base_cov_ii');
                        d2Gdg_a_s_tmp = d2Gdg_a_s_tmp - scalar_tmp * (base_cov_ii * mu_sigma_alpha');
                        d2Gda_s2_tmp = d2Gda_s2_tmp - scalar_tmp * ((mu_sigma_alpha * mu_sigma_alpha') + H_mat * V_hat_tmp(:,:,j) * H_mat');

                    end
                    
                    % Add this patient's contributions to the second derivatives
                    d2Gdg_s2(idx_stt:idx_end, idx_stt:idx_end) ...
                        = d2Gdg_s2(idx_stt:idx_end, idx_stt:idx_end) + dG_data.E_c_ig_allpats(ii,gg) * d2Gdg_s2_tmp;
                    d2Gdg_a_s(idx_stt:idx_end,:) = d2Gdg_a_s(idx_stt:idx_end,:) + dG_data.E_c_ig_allpats(ii,gg) * d2Gdg_a_s_tmp;
                    d2Gda_s2 = d2Gda_s2 + dG_data.E_c_ig_allpats(ii,gg) * d2Gda_s2_tmp;
                end
            end

            % Store the Hessian matrix in the correct format
            d2Gout = [d2Gdg_s2, d2Gdg_a_s;
                      d2Gdg_a_s', d2Gda_s2];
        end
        
        
        function E_val_part = weibull_E_val_part(g_a_s_curr, E_data, E_c_ig)
            % FUNCTION NAME:
            %   weibull_E_val_part
            %
            % DESCRIPTION:
            %   Evaluates part of the expectation which involves the
            %   survival parameters for the Weibull baseline hazard
            %   function (alternative MM).
            %
            % INPUT:
            %   g_a_s_curr - (array) Current parameter values of g_s and 
            %                a_s for all classes.
            %   E_data - (struct) Contains the required data to compute
            %            the expectation.
            %   E_c_ig - (array) Contains the responsibility of class g for
            %            every patient.
            %
            % OUTPUT:
            %   E_val_part - (double) Part of the expectation utilising the
            %                current values of the survival parameters.
            %
            % REVISION HISTORY:
            %   23/03/2023 - mcauchi1
            %       * Initial implementation
            %
            
            % extract the gamma and alpha values
            g_s_prev = g_a_s_curr(1:end-1,1);
            a_s_prev = g_a_s_curr(end,1);
            
            % Extract auxiliary patient information
            T_i = LSDSM_MM_ALLFUNCS.extract_field_from_map(E_data.pat_data, 'surv_time')';
            delta_i = LSDSM_MM_ALLFUNCS.extract_field_from_map(E_data.pat_data, 'delta_ev')';
            base_cov_mat = squeeze(LSDSM_MM_ALLFUNCS.extract_field_from_map(E_data.pat_data, 'base_cov'))'; % n x q matrix

            % this counts the number of effective events in this class
            n_d = E_c_ig' * delta_i;

            % calculate the auxiliary variables
            logT = log(T_i);
            Talpha_go = E_c_ig.* T_i.^a_s_prev .* exp(g_s_prev' * base_cov_mat')';
            
            % calculate part of the expectation
            E_val_part = n_d * log(a_s_prev) - n_d * log(sum(Talpha_go) / n_d) ...
                        + (a_s_prev - 1) * sum(E_c_ig .* delta_i .* logT) + sum(E_c_ig .* delta_i .* (g_s_prev' * base_cov_mat')') - n_d;
        end
        
        
        function E_val = weibull_E_val(g_a_s_curr, E_data, update_case)
            % FUNCTION NAME:
            %   weibull_E_val
            %
            % DESCRIPTION:
            %   Evaluates the expectation only involving the
            %   survival parameters for the Weibull baseline hazard
            %   function (alternative MM). This method takes care of any
            %   update case (for all scenarios of retaining the same
            %   parameters across all classes).
            %
            % INPUT:
            %   g_a_s_curr - (array) Current parameter values of g_s and 
            %                a_s for all classes.
            %   E_data - (struct) Contains the required data to compute
            %            the expectation.
            %   update_case - (double) Informs the function of which
            %                 survival parameters are being kept the same.
            %
            % OUTPUT:
            %   E_val - (double) Expectation utilising only the current
            %           values of the survival parameters.
            %
            % REVISION HISTORY:
            %   23/03/2023 - mcauchi1
            %       * Initial implementation
            %
            
            % initialise the expectation to zero
            E_val = 0;
            
            % evaluate the number of classes
            num_classes = E_data.num_classes;
            
            if update_case == 1 % if both g_s and a_s are kept the same across all classes
                if E_data.same_b_s % if we have the same beta (scale parameter for Weibull) across classes
                    % store the previous values of g_s and a_s
                    g_s_prev = g_a_s_curr(1:end-1, 1);
                    a_s_prev = g_a_s_curr(end, 1);
                    
                    % responsibilities for all patients for class g
                    % if we have the same survival parameters across all
                    % classes, then it is like we have only 1 class
                    E_c_ig = ones(size(E_data.E_c_ig_allpats,1),1);
                    
                    % evaluate the expectation
                    E_val = LSDSM_MM_ALLFUNCS.weibull_E_val_part([g_s_prev; a_s_prev], E_data, E_c_ig);
                    
                else % if we have the different beta (scale parameter for Weibull) across classes
                    for gg=1:num_classes  % for every class
                        % store the previous values of g_s and a_s
                        g_s_prev = g_a_s_curr(1:end-1, 1);
                        a_s_prev = g_a_s_curr(end, 1);

                        % responsibilities for all patients for class g
                        E_c_ig = E_data.E_c_ig_allpats(:,gg);

                        % evaluate the expectation for this class and add
                        % its contribution to the total expectation
                        E_val = E_val + LSDSM_MM_ALLFUNCS.weibull_E_val_part([g_s_prev; a_s_prev], E_data, E_c_ig);
                    end
                end
                
            elseif update_case == 2 % same gamma different alpha
                % store the sizes for gamma and alpha
                size_g_s = size(E_data.model_coef_est.g_s,1);
                size_a_s = size(E_data.model_coef_est.a_s,1);

                % Store the previous values of same g_s and different a_s across all classes
                g_s_prev = g_a_s_curr(1:size_g_s, 1);
                a_s_all_prev = g_a_s_curr(size_g_s+1:end, 1);
                
                for gg=1:num_classes % for every class
                    % Find the start and end indices to extract a_s of the current class gg
                    idx_stt = (gg-1)*size_a_s+1;
                    idx_end = gg*size_a_s;
                    a_s_prev = a_s_all_prev(idx_stt:idx_end,1);

                    % responsibilities for all patients for class g
                    E_c_ig = E_data.E_c_ig_allpats(:,gg);
                    
                    % evaluate the expectation for this class and add its
                    % contribution to the total expectation
                    E_val = E_val + LSDSM_MM_ALLFUNCS.weibull_E_val_part([g_s_prev; a_s_prev], E_data, E_c_ig);
                end
                
            elseif update_case == 3 % different gamma same alpha
                size_g_s = size(E_data.model_coef_est.g_s,1);
            
                % store the previous values of different g_s and same a_s across all classes
                g_s_all_prev = g_a_s_curr(1:end-1, 1);
                a_s_prev = g_a_s_curr(end, 1);
                
                for gg=1:num_classes
                    % Find the start and end indices to extract g_s of the current class gg
                    idx_stt = (gg-1)*size_g_s+1;
                    idx_end = gg*size_g_s;
                    g_s_prev = g_s_all_prev(idx_stt:idx_end,1);

                    % responsibilities for all patients for class g
                    E_c_ig = E_data.E_c_ig_allpats(:,gg);
                    
                    % evaluate the expectation for this class and add its
                    % contribution to the total expectation
                    E_val = E_val + LSDSM_MM_ALLFUNCS.weibull_E_val_part([g_s_prev; a_s_prev], E_data, E_c_ig);
                end
                
            else % update_case == 4
                % store the previous values of g_s and a_s
                g_s_prev = g_a_s_curr(1:end-1, 1);
                a_s_prev = g_a_s_curr(end, 1);
                
                % evaluate the expectation for the current class
                E_val = LSDSM_MM_ALLFUNCS.weibull_E_val_part([g_s_prev; a_s_prev], E_data, E_data.E_c_ig);
                
            end
        end
        
        
        function b_sw = weibull_beta_val(g_a_s_curr, E_data)
            % FUNCTION NAME:
            %   weibull_beta_val
            %
            % DESCRIPTION:
            %   Finds the optimised value for beta (scale) for a given
            %   configuration of gamma and alpha values for a Weibull
            %   baseline hazard function.
            %
            % INPUT:
            %   g_a_s_curr - (array) Current parameter values of g_s and 
            %                a_s for all classes.
            %   E_data - (struct) Contains the patient data and the
            %            expectation of classes.
            %
            % OUTPUT:
            %   b_sw - (double) beta (scale) parameter for the Weibull
            %          baseline hazard function
            %
            % REVISION HISTORY:
            %   23/03/2023 - mcauchi1
            %       * Initial implementation
            %
            
            % Store the previous values of g_s and a_s
            g_s_prev = g_a_s_curr(1:end-1, 1);
            a_s_prev = g_a_s_curr(end, 1);
            
            % responsibilities for all patients for class g
            E_c_ig = E_data.E_c_ig;
    
            % extract patient information
            T_i = LSDSM_MM_ALLFUNCS.extract_field_from_map(E_data.pat_data, 'surv_time')'; % n x 1 vector
            delta_i = LSDSM_MM_ALLFUNCS.extract_field_from_map(E_data.pat_data, 'delta_ev')'; % n x 1 vector
            base_cov_mat = squeeze(LSDSM_MM_ALLFUNCS.extract_field_from_map(E_data.pat_data, 'base_cov'))'; % n x q matrix
            
            % auxiliary variables
            Talpha_go = E_c_ig.* T_i.^a_s_prev .* exp(g_s_prev' * base_cov_mat')';
            n_d = E_c_ig' * delta_i; % this counts the number of effective events in this class
            
            b_sw = (sum(Talpha_go) / n_d)^(1/a_s_prev);
        end
        
        
        function dGout = dGdx_alt(g_a_s_prev, dG_data)
            % FUNCTION NAME:
            %   dGdx_alt
            %
            % DESCRIPTION:
            %   First Derivative of the Expectation with respect to the
            %   survival parameters: gamma (g_s) and alpha (a_s). This
            %   function assumes a single class (alternative MM).
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
            %   23/03/2023 - mcauchi1
            %       * Initial implementation
            %

            % Store the previous values of g_s and a_s
            g_s_prev = g_a_s_prev(1:end-1, 1);
            a_s_prev = g_a_s_prev(end, 1);
            
            % responsibilities for all patients for class g
            E_c_ig = dG_data.E_c_ig;
    
            % extract patient information
            T_i = LSDSM_MM_ALLFUNCS.extract_field_from_map(dG_data.pat_data, 'surv_time')'; % n x 1 vector
            delta_i = LSDSM_MM_ALLFUNCS.extract_field_from_map(dG_data.pat_data, 'delta_ev')'; % n x 1 vector
            base_cov_mat = squeeze(LSDSM_MM_ALLFUNCS.extract_field_from_map(dG_data.pat_data, 'base_cov'))'; % n x q matrix

            % this counts the number of effective events in this class
            n_d = E_c_ig' * delta_i;

            % calculate the auxiliary variables
            logT = log(T_i);
            Talpha_go = E_c_ig.* T_i.^a_s_prev .* exp(g_s_prev' * base_cov_mat')';
            logTtimesTalpha = sum(logT .* Talpha_go);
            Talpha_go_timesomega = sum(Talpha_go .* base_cov_mat,1)'; % (q x 1) vector

            % Find the gradients
            dGdg_s = - n_d * ( Talpha_go_timesomega / sum(Talpha_go) ) + sum(E_c_ig .* delta_i .* base_cov_mat,1)';
            dGda_s = n_d / a_s_prev - n_d * (logTtimesTalpha / sum(Talpha_go)) + sum(E_c_ig .* delta_i .* logT);

            % Store in the output vector
            dGout = [dGdg_s; dGda_s];
        end
        
        
        function d2Gout = d2Gdx2_alt(g_a_s_prev, dG_data)
            % FUNCTION NAME:
            %   d2Gdx2_alt
            %
            % DESCRIPTION:
            %   Hessian matrix (second derivative) of the Expectation with
            %   respect to the survival parameters: gamma (g_s) and alpha
            %   (a_s). This function assumes a single class (alternative 
            %   MM).
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
            %   23/03/2023 - mcauchi1
            %       * Initial implementation
            %

            % Store the previous values of g_s and a_s
            g_s_prev = g_a_s_prev(1:end-1, 1);
            a_s_prev = g_a_s_prev(end, 1);
            
            % responsibilities for all patients for class g
            E_c_ig = dG_data.E_c_ig;
            
            % extract patient information
            T_i = LSDSM_MM_ALLFUNCS.extract_field_from_map(dG_data.pat_data, 'surv_time')';
            delta_i = LSDSM_MM_ALLFUNCS.extract_field_from_map(dG_data.pat_data, 'delta_ev')';
            base_cov_mat = squeeze(LSDSM_MM_ALLFUNCS.extract_field_from_map(dG_data.pat_data, 'base_cov'))'; % n x q matrix

            % this counts the number of effective events in this class
            n_d = E_c_ig' * delta_i;

            % calculate the auxiliary variables
            logT = log(T_i);
            Talpha_go = E_c_ig.* T_i.^a_s_prev .* exp(g_s_prev' * base_cov_mat')';
            logTtimesTalpha = sum(logT .* Talpha_go);
            logT2timesTalpha = sum(logT.^2 .* Talpha_go);
            logTtimesTalphatimesgamma = sum(logT .* Talpha_go .* base_cov_mat)';
            Talpha_go_timesomega = sum(Talpha_go .* base_cov_mat,1)'; % (q x 1) vector
            Talphatimesomegaomega = zeros(size(g_s_prev,1),size(g_s_prev,1)); % (q x q) vector
            for i=1:length(T_i)
                Talphatimesomegaomega = Talphatimesomegaomega + Talpha_go(i) * (base_cov_mat(i,:)' * base_cov_mat(i,:));
            end

            % Find the components for the Hessian matrix
            d2Gdg_s2 = - n_d * ( Talphatimesomegaomega / sum(Talpha_go) ...
                                    - (Talpha_go_timesomega / sum(Talpha_go)) * (Talpha_go_timesomega' / sum(Talpha_go)));
                                
            d2Gdg_a_s = - n_d * ( logTtimesTalphatimesgamma / sum(Talpha_go) ...
                          - (logTtimesTalpha / sum(Talpha_go)) * (Talpha_go_timesomega / sum(Talpha_go)) );

            d2Gda_s2 = - n_d / a_s_prev^2 - n_d * ( (logT2timesTalpha / sum(Talpha_go)) - (logTtimesTalpha / (sum(Talpha_go)))^2);

            % Store the Hessian matrix in the correct format
            d2Gout = [d2Gdg_s2, d2Gdg_a_s;
                      d2Gdg_a_s', d2Gda_s2];
        end
        
        
        function dGout = dGdx_alt_case2(g_a_s_prev, dG_data)
            % FUNCTION NAME:
            %   dGdx_alt_case2
            %
            % DESCRIPTION:
            %   First Derivative of the Expectation with respect to the
            %   survival parameters: gamma (g_s) and alpha (a_s), where it
            %   is assumed that g_s is the same across all classes, while
            %   a_s is different across classes. Here, alpha refers to
            %   shape parameter for the Weibull baseline hazard function
            %   (alternative MM).
            %
            % INPUT:
            %   g_a_s_prev - (array) Previous parameter values of g_s and
            %                a_s. Only one g_s vector should be in this
            %                array, while the a_s across all classes should
            %                be included.
            %   dG_data - (struct) Contains the required data to compute
            %             the derivative.
            %
            % OUTPUT:
            %   dGout - (array) The derivative of the expectation with
            %           respect to the survival parameters.
            %
            % REVISION HISTORY:
            %   23/03/2023 - mcauchi1
            %       * Initial implementation
            %
                             
            % Identify the number of classes
            num_classes = dG_data.num_classes;
            % Store the sizes for gamma and alpha
            size_g_s = size(dG_data.model_coef_est.g_s,1);
            size_a_s = size(dG_data.model_coef_est.a_s,1);
            
            % Store the previous values of same g_s and different a_s across all classes
            g_s_prev = g_a_s_prev(1:size_g_s, 1);
            a_s_all_prev = g_a_s_prev(size_g_s+1:end, 1);

            % Initiate the derivatives to zero
            dGdg_s = zeros(size(g_s_prev));
            dGda_s = zeros(size(a_s_all_prev));
            
            % Extract patient data to be used in this derivative
            T_i = LSDSM_MM_ALLFUNCS.extract_field_from_map(dG_data.pat_data, 'surv_time')'; % n x 1 matrix
            delta_i = LSDSM_MM_ALLFUNCS.extract_field_from_map(dG_data.pat_data, 'delta_ev')'; % n x 1 matrix
            base_cov_mat = squeeze(LSDSM_MM_ALLFUNCS.extract_field_from_map(dG_data.pat_data, 'base_cov'))'; % n x q matrix
            
            for gg=1:num_classes
                % Find the start and end indices to extract a_s of the current class gg
                idx_stt = (gg-1)*size_a_s+1;
                idx_end = gg*size_a_s;
                a_s_prev = a_s_all_prev(idx_stt:idx_end,1);

                % responsibilities for all patients for class g
                E_c_ig = dG_data.E_c_ig_allpats(:,gg);

                % this counts the number of effective events in this class
                n_d = E_c_ig' * delta_i;

                % calculate the auxiliary variables
                logT = log(T_i);
                Talpha_go = E_c_ig.* T_i.^a_s_prev .* exp(g_s_prev' * base_cov_mat')';
                logTtimesTalpha = sum(logT .* Talpha_go);
                Talpha_go_timesomega = sum(Talpha_go .* base_cov_mat,1)'; % (q x 1) vector

                % Find the gradients
                dGdg_s_tmp = - n_d * ( Talpha_go_timesomega / sum(Talpha_go) ) + sum(E_c_ig .* delta_i .* base_cov_mat,1)';
                dGda_s_tmp = n_d / a_s_prev - n_d * (logTtimesTalpha / sum(Talpha_go)) + sum(E_c_ig .* delta_i .* logT);
                
                % Add this patient's contribution to the gradients
                dGdg_s = dGdg_s + dGdg_s_tmp;
                dGda_s(idx_stt:idx_end,1) = dGda_s_tmp;
            end
            
            % Store in the output vector
            dGout = [dGdg_s; dGda_s];
        end
        
        
        function d2Gout = d2Gdx2_alt_case2(g_a_s_prev, dG_data)
            % FUNCTION NAME:
            %   d2Gdx2_alt_case2
            %
            % DESCRIPTION:
            %   Hessian matrix (second derivative) of the expectation with
            %   respect to the survival parameters: gamma (g_s) and alpha
            %   (a_s), where it is assumed that g_s is the same across all
            %   classes, while a_s is different across classes. Here, alpha
            %   refers to shape parameter for the Weibull baseline hazard
            %   function (alternative MM).
            %
            % INPUT:
            %   g_a_s_prev - (array) Previous parameter values of g_s and
            %                a_s. Only one g_s vector should be in this
            %                array, while the a_s across all classes should
            %                be included.
            %   dG_data - (struct) Contains the required data to compute
            %             the derivative.
            %
            % OUTPUT:
            %   d2Gout - (array) The second derivative of the expectation 
            %            with respect to the survival parameters.
            %
            % REVISION HISTORY:
            %   23/03/2023 - mcauchi1
            %       * Initial implementation
            %
            
            % Identify the nmber of classes
            num_classes = dG_data.num_classes;
            % Store the sizes for gamma and alpha
            size_g_s = size(dG_data.model_coef_est.g_s,1);
            size_a_s = size(dG_data.model_coef_est.a_s,1);
            
            % Store the previous values of same g_s and different a_s across all classes
            g_s_prev = g_a_s_prev(1:size_g_s, 1);
            a_s_all_prev = g_a_s_prev(size_g_s+1:end, 1);

            % Initiate the second derivatives to zero
            d2Gdg_s2 = zeros(size_g_s, size_g_s);
            d2Gdg_a_s = zeros(size_g_s, size(a_s_all_prev,1));
            d2Gda_s2 = zeros(size(a_s_all_prev,1), size(a_s_all_prev,1));
            
            % Extract patient data to be used in this derivative
            T_i = LSDSM_MM_ALLFUNCS.extract_field_from_map(dG_data.pat_data, 'surv_time')';
            delta_i = LSDSM_MM_ALLFUNCS.extract_field_from_map(dG_data.pat_data, 'delta_ev')';
            base_cov_mat = squeeze(LSDSM_MM_ALLFUNCS.extract_field_from_map(dG_data.pat_data, 'base_cov'))'; % n x q matrix
            
            for gg=1:num_classes % for every class
                % Find the start and end indices to extract a_s of the current class gg
                idx_stt = (gg-1)*size_a_s+1;
                idx_end = gg*size_a_s;
                a_s_prev = a_s_all_prev(idx_stt:idx_end,1);

                % responsibilities for all patients for class g
                E_c_ig = dG_data.E_c_ig_allpats(:,gg);

                % this counts the number of effective events in this class
                n_d = E_c_ig' * delta_i;

                % calculate the auxiliary variables
                logT = log(T_i);
                Talpha_go = E_c_ig.* T_i.^a_s_prev .* exp(g_s_prev' * base_cov_mat')';
                logTtimesTalpha = sum(logT .* Talpha_go);
                logT2timesTalpha = sum(logT.^2 .* Talpha_go);
                logTtimesTalphatimesgamma = sum(logT .* Talpha_go .* base_cov_mat)';
                Talpha_go_timesomega = sum(Talpha_go .* base_cov_mat,1)'; % (q x 1) vector
                Talphatimesomegaomega = zeros(size_g_s,size_g_s); % (q x q) vector
                
                for i=1:length(T_i)
                    Talphatimesomegaomega = Talphatimesomegaomega + Talpha_go(i) * (base_cov_mat(i,:)' * base_cov_mat(i,:));
                end
                
                % Find the components for the Hessian matrix
                d2Gdg_s2_tmp = - n_d * ( Talphatimesomegaomega / sum(Talpha_go) ...
                                        - (Talpha_go_timesomega / sum(Talpha_go)) * (Talpha_go_timesomega' / sum(Talpha_go)));
                                    
                d2Gdg_a_s_tmp = - n_d * ( logTtimesTalphatimesgamma / sum(Talpha_go) ...
                              - (logTtimesTalpha / sum(Talpha_go)) * (Talpha_go_timesomega / sum(Talpha_go)) );
                
                d2Gda_s2_tmp = - n_d / a_s_prev^2 - n_d * ( (logT2timesTalpha / sum(Talpha_go)) - (logTtimesTalpha / (sum(Talpha_go)))^2 );

                % Add this patients' contributions to the second derivatives
                d2Gdg_s2 = d2Gdg_s2 + d2Gdg_s2_tmp;
                d2Gdg_a_s(:,idx_stt:idx_end) = d2Gdg_a_s_tmp;
                d2Gda_s2(idx_stt:idx_end, idx_stt:idx_end) = d2Gda_s2_tmp;
            end
            
            % Store the Hessian matrix in the correct format
            d2Gout = [d2Gdg_s2, d2Gdg_a_s;
                      d2Gdg_a_s', d2Gda_s2];
        end
        
        
        function dGout = dGdx_alt_case3(g_a_s_prev, dG_data)
            % FUNCTION NAME:
            %   dGdx_alt_case3
            %
            % DESCRIPTION:
            %   First Derivative of the Expectation with respect to the
            %   survival parameters: gamma (g_s) and alpha (a_s), where it
            %   is assumed that g_s is different across classes, while a_s
            %   is the same across all classes. Here, alpha refers to shape
            %   parameter for the Weibull baseline hazard function
            %   (alternative MM).
            %
            % INPUT:
            %   g_a_s_prev - (array) Previous parameter values of g_s and
            %                a_s. The g_s across all classes should be
            %                included, while only one a_s should be in this
            %                array.
            %   dG_data - (struct) Contains the required data to compute
            %             the derivative.
            %
            % OUTPUT:
            %   dGout - (array) The derivative of the expectation with
            %           respect to the survival parameters.
            %
            % REVISION HISTORY:
            %   23/03/2023 - mcauchi1
            %       * Initial implementation
            %
                             
            % Identify the nmber of classes
            num_classes = dG_data.num_classes;
            % Store the sizes for gamma and alpha
            size_g_s = size(dG_data.model_coef_est.g_s,1);
            
            % Store the previous values of same g_s and different a_s across all classes
            g_s_all_prev = g_a_s_prev(1:end-1, 1);
            a_s_prev = g_a_s_prev(end, 1);

            % Initiate the derivatives to zero
            dGdg_s = zeros(size(g_s_all_prev));
            dGda_s = zeros(size(a_s_all_prev));
            
            % Extract patient data to be used in this derivative
            T_i = LSDSM_MM_ALLFUNCS.extract_field_from_map(dG_data.pat_data, 'surv_time')';
            delta_i = LSDSM_MM_ALLFUNCS.extract_field_from_map(dG_data.pat_data, 'delta_ev')';
            base_cov_mat = squeeze(LSDSM_MM_ALLFUNCS.extract_field_from_map(dG_data.pat_data, 'base_cov'))'; % n x q matrix
            
            for gg=1:num_classes
                % Find the start and end indices to extract a_s of the current class gg
                idx_stt = (gg-1)*size_g_s+1;
                idx_end = gg*size_g_s;
                g_s_prev = g_s_all_prev(idx_stt:idx_end,1);

                % responsibilities for all patients for class g
                E_c_ig = dG_data.E_c_ig_allpats(:,gg);

                % this counts the number of effective events in this class
                n_d = E_c_ig' * delta_i;

                % calculate the auxiliary variables
                logT = log(T_i);
                Talpha_go = E_c_ig.* T_i.^a_s_prev .* exp(g_s_prev' * base_cov_mat')';
                logTtimesTalpha = sum(logT .* Talpha_go);
                Talpha_go_timesomega = sum(Talpha_go .* base_cov_mat,1)'; % (q x 1) vector

                % Find the gradients
                dGdg_s_tmp = - n_d * ( Talpha_go_timesomega / sum(Talpha_go) ) + sum(E_c_ig .* delta_i .* base_cov_mat,1)';
                dGda_s_tmp = n_d / a_s_prev - n_d * (logTtimesTalpha / sum(Talpha_go)) + sum(E_c_ig .* delta_i .* logT);
                
                % Add the contributions to the gradients
                dGdg_s(idx_stt:idx_end,1) = dGdg_s_tmp;
                dGda_s = dGda_s + dGda_s_tmp;
            end
            
            % Store in the output vector
            dGout = [dGdg_s; dGda_s];
        end
        
        
        function d2Gout = d2Gdx2_alt_case3(g_a_s_prev, dG_data)
            % FUNCTION NAME:
            %   d2Gdx2_alt_case3
            %
            % DESCRIPTION:
            %   Hessian matrix (second derivative) of the expectation with
            %   respect to the survival parameters: gamma (g_s) and alpha
            %   (a_s), where it is assumed that g_s is different across
            %   classes, while a_s is the same across all classes. Here,
            %   alpha refers to shape parameter for the Weibull baseline
            %   hazard function (alternative MM).
            %
            % INPUT:
            %   g_a_s_prev - (array) Previous parameter values of g_s and
            %                a_s. The g_s across all classes should be
            %                included, while only one a_s should be in this
            %                array.
            %   dG_data - (struct) Contains the required data to compute
            %             the derivative.
            %
            % OUTPUT:
            %   d2Gout - (array) The second derivative of the expectation 
            %            with respect to the survival parameters.
            %
            % REVISION HISTORY:
            %   23/03/2023 - mcauchi1
            %       * Initial implementation
            %
            
            % Identify the nmber of classes
            num_classes = dG_data.num_classes;
            % Store the sizes for gamma and alpha
            size_g_s = size(dG_data.model_coef_est.g_s,1);
            
            % Store the previous values of same g_s and different a_s across all classes
            g_s_all_prev = g_a_s_prev(1:end-1, 1);
            a_s_prev = g_a_s_prev(end, 1);

            % Initiate the second derivatives to zero
            d2Gdg_s2 = zeros(size(g_s_all_prev,1), size(g_s_all_prev,1));
            d2Gdg_a_s = zeros(size(g_s_all_prev,1), size(a_s_prev,1));
            d2Gda_s2 = zeros(size(a_s_prev,1), size(a_s_prev,1));
            
            % Extract patient data to be used in this derivative
            T_i = LSDSM_MM_ALLFUNCS.extract_field_from_map(dG_data.pat_data, 'surv_time')'; % n x 1 vector
            delta_i = LSDSM_MM_ALLFUNCS.extract_field_from_map(dG_data.pat_data, 'delta_ev')'; % n x 1 vector
            base_cov_mat = squeeze(LSDSM_MM_ALLFUNCS.extract_field_from_map(dG_data.pat_data, 'base_cov'))'; % n x q matrix
            
            for gg=1:num_classes % for every class
                % Find the start and end indices to extract a_s of the current class gg
                idx_stt = (gg-1)*size_g_s+1;
                idx_end = gg*size_g_s;
                g_s_prev = g_s_all_prev(idx_stt:idx_end,1);

                % responsibilities for all patients for class g
                E_c_ig = dG_data.E_c_ig_allpats(:,gg);

                % this counts the number of effective events in this class
                n_d = E_c_ig' * delta_i;

                % calculate the auxiliary variables
                logT = log(T_i);
                Talpha_go = E_c_ig.* T_i.^a_s_prev .* exp(g_s_prev' * base_cov_mat')';
                logTtimesTalpha = sum(logT .* Talpha_go);
                logT2timesTalpha = sum(logT.^2 .* Talpha_go);
                logTtimesTalphatimesgamma = sum(logT .* Talpha_go .* base_cov_mat)';
                Talpha_go_timesomega = sum(Talpha_go .* base_cov_mat,1)'; % (q x 1) vector
                Talphatimesomegaomega = zeros(size_g_s,size_g_s); % (q x q) vector
                
                for i=1:length(T_i)
                    Talphatimesomegaomega = Talphatimesomegaomega + Talpha_go(i) * (base_cov_mat(i,:)' * base_cov_mat(i,:));
                end
                
                % Find the components for the Hessian matrix
                d2Gdg_s2_tmp = - n_d * ( Talphatimesomegaomega / sum(Talpha_go) ...
                                        - (Talpha_go_timesomega / sum(Talpha_go)) * (Talpha_go_timesomega' / sum(Talpha_go)));
                
                d2Gdg_a_s_tmp = - n_d * ( logTtimesTalphatimesgamma / sum(Talpha_go) ...
                              - (logTtimesTalpha / sum(Talpha_go)) * (Talpha_go_timesomega / sum(Talpha_go)) );

                d2Gda_s2_tmp = - n_d / a_s_prev^2 - n_d * ( (logT2timesTalpha / sum(Talpha_go)) - (logTtimesTalpha / (sum(Talpha_go)))^2);
                
                % Add this contributions to the second derivatives
                d2Gdg_s2(idx_stt:idx_end, idx_stt:idx_end) = d2Gdg_s2_tmp;
                d2Gdg_a_s(idx_stt:idx_end,:) = d2Gdg_a_s_tmp;
                d2Gda_s2 = d2Gda_s2 + d2Gda_s2_tmp;
            end
            
            % Store the Hessian matrix in the correct format
            d2Gout = [d2Gdg_s2, d2Gdg_a_s;
                      d2Gdg_a_s', d2Gda_s2];
        end
        
        
        function y = f_zeta(zeta, coeffs)
            % FUNCTION NAME:
            %   dGdzeta
            %
            % DESCRIPTION:
            %   Finds the value of the Expectation containing only the
            %   elements involving zeta.
            %
            % INPUT:
            %   zeta - (array) Current parameter values zeta.
            %   coeffs - (struct) Contains the required data to compute
            %            the required expectation value.
            %
            % OUTPUT:
            %   y - (double) The value of the expectation at zeta.
            %
            % REVISION HISTORY:
            %   22/02/2023 - mcauchi1
            %       * Initial implementation
            %
            
            % Find the number of parameters for every class, the number of
            % classes, and the number of patients.
            num_class_params = size(coeffs.Q_mat, 1);
            num_classes = size(coeffs.E_c_ig_all, 2);
            num_pats = size(coeffs.E_c_ig_all, 1);
            
            % B_mat as |R^( n  x  G ) -> E[c_ig]
            E_c_ig_mat = coeffs.E_c_ig_all;
            
            % Split parameters in the form of |R^( m_c  x  G )
            params_split_in_class = zeros(num_class_params, num_classes);
            for g=1:num_classes % for every class
                params_split_in_class(:,g) = zeta((g-1)*num_class_params+1:g*num_class_params,1);
            end
            
            % Calculate the numerator of prior probabilities for every patient and every class
            priors_mat = zeros(size(E_c_ig_mat));
            for ii=1:num_pats % for every patient
                for g=1:num_classes % for every class
                    priors_mat(ii,g) = exp(params_split_in_class(:,g)' * coeffs.Q_mat(:,ii));
                end
            end
            
            % Normalise so that each row sums up to 1 -> prior probabilities
            priors_mat = priors_mat./sum(priors_mat,2);
            
            y = 0;
            for ii=1:num_pats % for every patient
                for g=1:num_classes % for every class
                    y = y + E_c_ig_mat(ii,g) * log(priors_mat(ii,g));
                end
            end
        end
        
        
        function dGout = dGdzeta(zeta_prev, coeffs)
            % FUNCTION NAME:
            %   dGdzeta
            %
            % DESCRIPTION:
            %   First Derivative of the Expectation with respect to all
            %   class parameters zeta.
            %
            % INPUT:
            %   zeta_prev - (array) Previous parameter values of zeta.
            %   coeffs - (struct) Contains the required data to compute
            %            the derivative.
            %
            % OUTPUT:
            %   dGout - (array) The derivative of the expectation with
            %           respect to the class parameters.
            %
            % REVISION HISTORY:
            %   12/01/2023 - mcauchi1
            %       * Initial implementation
            %
            
            % We find this derivative for G-1 classes for identifiability reasons.
            
            % Find the number of parameters for every class, the number of
            % classes, and the number of patients.
            num_class_params = size(coeffs.Q_mat, 1);
            num_classes = size(coeffs.E_c_ig_all, 2);
            num_pats = size(coeffs.E_c_ig_all, 1);
            
            % B represents the expected probabilities of belonging to a 
            % class minus the prior probabilities of belonging to a class.
            % B = E[c_ig] - pi_ig
            % B as |R^( n  x  G )
            B_mat = coeffs.E_c_ig_all;
            
            % Split parameters in the form of |R^( m_c  x  G )
            params_split_in_class = zeros(num_class_params, num_classes);
            for g=1:num_classes
                params_split_in_class(:,g) = zeta_prev((g-1)*num_class_params+1:g*num_class_params,1);
            end
            
            % Calculate the numerator of prior probabilities for every patient and every class
            B_mat_neg = zeros(size(B_mat));
            for ii=1:num_pats
                for g=1:num_classes
                    B_mat_neg(ii,g) = exp(params_split_in_class(:,g)' * coeffs.Q_mat(:,ii));
                end
            end
            
            % Normalise so that each row sums up to 1 -> prior probabilities
            B_mat_neg = B_mat_neg./sum(B_mat_neg,2);
            
            % B = E[c_ig] - pi_ig
            B_mat = B_mat - B_mat_neg;
            
            % For identifiability reasons, we do not update the parameters for the first class
            B_mat = B_mat(:,2:num_classes);
            
            % Q * B' = |R^( (m_c  x  n)  x  (n  x  (G-1))) = |R^( m_c  x  (G-1) )
            dGout = coeffs.Q_mat * B_mat;
            
            % Stack the column matrices -> |R^( (G-1)m_c  x  1 )
            dGout = dGout(:);

        end
        
        
        function d2Gout = d2Gdzeta2(zeta_prev, coeffs)
            % FUNCTION NAME:
            %   d2Gdzeta2
            %
            % DESCRIPTION:
            %   Hessian matrix (second derivative) of the Expectation with
            %   respect to all class parameters zeta.
            %
            % INPUT:
            %   zeta_prev - (array) Previous parameter values of zeta.
            %   coeffs - (struct) Contains the required data to compute
            %            the derivative.
            %
            % OUTPUT:
            %   d2Gout - (array) The second derivative of the expectation 
            %            with respect to the class parameters.
            %
            % REVISION HISTORY:
            %   12/01/2023 - mcauchi1
            %       * Initial implementation
            %
            
            % We find this derivative for G-1 classes for identifiability reasons.
            
            % Find the number of parameters for every class, the number of
            % classes, and the number of patients.
            num_class_params = size(coeffs.Q_mat, 1);
            num_classes = size(coeffs.E_c_ig_all, 2);
            num_pats = size(coeffs.E_c_ig_all, 1);
            
            % the outer product of the class coefficients for every patient
            qqt = zeros(num_class_params, num_class_params, num_pats);
            for ii=1:num_pats
                qqt(:,:,ii) = coeffs.Q_mat(:,ii) * coeffs.Q_mat(:,ii)';
            end

            % pi as |R^( n  x  G ) -> list of prior probabilities
            pi_ig_mat = zeros(num_pats, num_classes);
            
            % Split parameters in the form of |R^( m_c  x  G )
            params_split_in_class = zeros(num_class_params, num_classes);
            for g=1:num_classes
                params_split_in_class(:,g) = zeta_prev((g-1)*num_class_params+1:g*num_class_params,1);
            end
            
            % Calculate the numerator of prior probabilities for every patient and every class
            for ii=1:num_pats
                for g=1:num_classes
                    pi_ig_mat(ii,g) = exp(params_split_in_class(:,g)' * coeffs.Q_mat(:,ii));
                end
            end
            
            % Normalise so that each row sums up to 1 -> prior probabilities
            pi_ig_mat = pi_ig_mat./sum(pi_ig_mat,2);
            
            % For identifiability reasons, we do not update the parameters for the first class
            pi_ig_mat = pi_ig_mat(:,2:num_classes);
            
            % Initialise the Hessian matrix -> |R^( (G-1)m_c  x  (G-1)m_c )
            d2Gout = zeros((num_classes-1)*num_class_params, (num_classes-1)*num_class_params);
            
            % iterate across every class (except first class, hence the -1)
            % and work out the Hessian
            % pi_ig (I(g=l) - pi_il) * qq'
            for g=1:num_classes-1
                for gg=1:num_classes-1
                    temp_val = zeros(num_class_params, num_class_params);
                    for ii=1:num_pats
                         temp_val = temp_val + pi_ig_mat(ii,g) * ((g==gg) - pi_ig_mat(ii,gg)) * qqt(:,:,ii);
                    end
                    d2Gout((g-1)*num_class_params+1:g*num_class_params, ...
                           (gg-1)*num_class_params+1:gg*num_class_params) = - temp_val;
                end
            end
        end


    end
end