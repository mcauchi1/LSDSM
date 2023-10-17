clear all;
close all;

num_sims = 100; % number of simulations performed in each file
miss_frac_arr = [0 25 50 75]; % missing data configurations
num_configs = length(miss_frac_arr);
%% - load all configurations into matrices

% placeholders for all relevant information
A1_all = zeros(num_sims,num_configs);
A2_all = zeros(num_sims,num_configs);
W1_all = zeros(num_sims,num_configs);
V1_all = zeros(num_sims,num_configs);
g_s1_all = zeros(num_sims,num_configs);
g_s2_all = zeros(num_sims,num_configs);
a_s1_all = zeros(num_sims,num_configs);
EM_iter_all = zeros(num_sims,num_configs);

est_model_rmse_train_mu1 = zeros(num_sims,num_configs);
est_model_rmse_train_surv = zeros(num_sims,num_configs);
est_model_rmse_test_mu1 = zeros(num_sims,num_configs);
est_model_rmse_test_surv = zeros(num_sims,num_configs);
true_model_rmse_train_mu1 = zeros(num_sims,num_configs);
true_model_rmse_test_mu = zeros(num_sims,num_configs);
true_model_rmse_train_surv = zeros(num_sims,num_configs);
true_model_rmse_test_mu1 = zeros(num_sims,num_configs);
true_model_rmse_test_surv = zeros(num_sims,num_configs);

E_m_i_all = zeros(num_sims,num_configs);
E_T_i_all = zeros(num_sims,num_configs);
E_delta_i_all = zeros(num_sims,num_configs);

for miss_i=1:length(miss_frac_arr) % for every missing data configuration
    curr_miss_frac = miss_frac_arr(miss_i);
    % load respective file
    load_file = sprintf('sims_t30\\LSDSM_multiple_runs_sim_500pats_%dmiss.mat', curr_miss_frac);
    load(load_file);
    
    % store information
    A1_all(:,miss_i) = squeeze(A_est_runs_arr(1,1,:,:));
    A2_all(:,miss_i) = squeeze(A_est_runs_arr(1,2,:,:));
    W1_all(:,miss_i) = squeeze(W_est_runs_arr(1,1,:,:));
    V1_all(:,miss_i) = squeeze(V_est_runs_arr(1,1,:,:));
    g_s1_all(:,miss_i) = squeeze(g_s_est_runs_arr(1,1,:,:));
    g_s2_all(:,miss_i) = squeeze(g_s_est_runs_arr(2,1,:,:));
    a_s1_all(:,miss_i) = squeeze(a_s_est_runs_arr(1,1,:,:));
    EM_iter_all(:,miss_i) = EM_iter_arr;
    
    est_model_rmse_train_mu1(:,miss_i) = squeeze(est_model_rmse.train.mu(1,:,:));
    est_model_rmse_train_surv(:,miss_i) = squeeze(est_model_rmse.train.surv(1,:,:));
    est_model_rmse_test_mu1(:,miss_i) = squeeze(est_model_rmse.test.mu(1,:,:));
    est_model_rmse_test_surv(:,miss_i) = squeeze(est_model_rmse.test.surv(1,:,:));
    true_model_rmse_train_mu1(:,miss_i) = squeeze(true_model_rmse.train.mu(1,:,:));
    true_model_rmse_train_surv(:,miss_i) = squeeze(true_model_rmse.train.surv(1,:,:));
    true_model_rmse_test_mu1(:,miss_i) = squeeze(true_model_rmse.test.mu(1,:,:));
    true_model_rmse_test_surv(:,miss_i) = squeeze(true_model_rmse.test.surv(1,:,:));
    
    E_m_i_all(:,miss_i) = squeeze(E_m_i_arr);
    E_T_i_all(:,miss_i) = squeeze(E_T_i_arr);
    E_delta_i_all(:,miss_i) = squeeze(E_delta_i_arr);
    
end

%% Basic analysis
mean_A_est1 = mean(A1_all, 1);
mean_A_est2 = mean(A2_all, 1);
mean_W_est = mean(W1_all, 1);
mean_V_est = mean(V1_all, 1);
mean_g_s_est1 = mean(g_s1_all, 1);
mean_g_s_est2 = mean(g_s2_all, 1);
mean_a_s_est = mean(a_s1_all, 1);

std_A_est1 = std(A1_all, 0, 1);
std_A_est2 = std(A2_all, 0, 1);
std_W_est = std(W1_all, 0, 1);
std_V_est = std(V1_all, 0, 1);
std_g_s_est1 = std(g_s1_all, 0, 1);
std_g_s_est2 = std(g_s2_all, 0, 1);
std_a_s_est = std(a_s1_all, 0, 1);

bias_A_est1 = (A1(1,1) - mean_A_est1) / A1(1,1) * 100;
bias_A_est2 = (A1(1,2) - mean_A_est2) / A1(1,2) * 100;
bias_W_est = (W1(1,1) - mean_W_est) / W1(1,1) * 100;
bias_V_est = (V1(1,1) - mean_V_est) / V1(1,1) * 100;
bias_g_s_est1 = (g_s1(1,1) - mean_g_s_est1) / g_s1(1,1) * 100;
bias_g_s_est2 = (g_s1(2,1) - mean_g_s_est2) / g_s1(2,1) * 100;
bias_a_s_est = (a_s1(1,1) - mean_a_s_est) / a_s1(1,1) * 100;

std_percent_A_est1 = std_A_est1 ./ mean_A_est1 * 100;
std_percent_A_est2 = std_A_est2 ./ mean_A_est2 * 100;
std_percent_W_est = std_W_est ./ mean_W_est * 100;
std_percent_V_est = std_V_est ./ mean_V_est * 100;
std_percent_g_s_est1 = std_g_s_est1 ./ mean_g_s_est1 * 100;
std_percent_g_s_est2 = std_g_s_est2 ./ mean_g_s_est2 * 100;
std_percent_a_s_est = std_a_s_est ./ mean_a_s_est * 100;

rmse_train_mu1 = mean(est_model_rmse_train_mu1);
rmse_train_surv = mean(est_model_rmse_train_surv);
rmse_test_mu1 = mean(est_model_rmse_test_mu1);
rmse_test_surv = mean(est_model_rmse_test_surv);
rmse_true_train_mu1 = mean(true_model_rmse_train_mu1);
rmse_true_train_surv = mean(true_model_rmse_train_surv);
rmse_true_test_mu1 = mean(true_model_rmse_test_mu1);
rmse_true_test_surv = mean(true_model_rmse_test_surv);

mean_E_m_i = mean(E_m_i_all);
mean_E_T_i = mean(E_T_i_all);
mean_E_delta_i = mean(E_delta_i_all);

%% Subplots of histograms of parameter estimates
figure;
t = tiledlayout(7,num_configs);
for i=1:length(miss_frac_arr)
    curr_miss_frac = miss_frac_arr(i);
    
    nexttile
    title1 = sprintf('%d%% missing observations\n', curr_miss_frac);
    histogram(A1_all(:,i), 20); xline(A1(1,1), '--r', 'LineWidth', 2); xlim([1.36 1.56]); ylim([0 15]); title(title1);
    if i==1
        ylabel(sprintf('A_{11}'),'Rotation',0,'fontweight','bold'); 
        ylh = get(gca,'ylabel');
        gyl = get(ylh);
        ylp = get(ylh, 'Position');
        set(ylh, 'Rotation',0, 'Position',ylp, 'VerticalAlignment','middle', 'HorizontalAlignment','right')
    end
end

for i=1:length(miss_frac_arr)
    nexttile; histogram(A2_all(:,i), 20); xline(A1(1,2), '--r', 'LineWidth', 2); xlim([-0.58 -0.38]); ylim([0 15]);
    if i==1
        ylabel(sprintf('A_{12}'),'Rotation',0,'fontweight','bold'); 
        ylh = get(gca,'ylabel');
        gyl = get(ylh);
        ylp = get(ylh, 'Position');
        set(ylh, 'Rotation',0, 'Position',ylp, 'VerticalAlignment','middle', 'HorizontalAlignment','right')
    end
end
for i=1:length(miss_frac_arr)
    nexttile; histogram(W1_all(:,i), 20); xline(W1(1,1), '--r', 'LineWidth', 2); xlim([0.02 0.06]); ylim([0 20]);
    if i==1
        ylabel(sprintf('W'),'Rotation',0,'fontweight','bold'); 
        ylh = get(gca,'ylabel');
        gyl = get(ylh);
        ylp = get(ylh, 'Position');
        set(ylh, 'Rotation',0, 'Position',ylp, 'VerticalAlignment','middle', 'HorizontalAlignment','right')
    end
end

for i=1:length(miss_frac_arr)
    nexttile; histogram(V1_all(:,i), 20); xline(V1(1,1), '--r', 'LineWidth', 2); xlim([0.2 0.3]);  ylim([0 15]);
    if i==1
        ylabel(sprintf('V'),'Rotation',0,'fontweight','bold'); 
        ylh = get(gca,'ylabel');
        gyl = get(ylh);
        ylp = get(ylh, 'Position');
        set(ylh, 'Rotation',0, 'Position',ylp, 'VerticalAlignment','middle', 'HorizontalAlignment','right')
    end
end
for i=1:length(miss_frac_arr)
    nexttile; histogram(g_s1_all(:,i), 20); xline(g_s1(1,1), '--r', 'LineWidth', 2); xlim([1.35 3.65]); ylim([0 25]);
    if i==1
        ylabel(sprintf('\\gamma_{s1}'),'Rotation',0,'fontweight','bold'); 
        ylh = get(gca,'ylabel');
        gyl = get(ylh);
        ylp = get(ylh, 'Position');
        set(ylh, 'Rotation',0, 'Position',ylp, 'VerticalAlignment','middle', 'HorizontalAlignment','right')
    end
end
for i=1:length(miss_frac_arr)
    nexttile; histogram(g_s2_all(:,i), 20); xline(g_s1(2,1), '--r', 'LineWidth', 2); xlim([-1.1 -0.4]); ylim([0 20]);
    if i==1
        ylabel(sprintf('\\gamma_{s2}'),'Rotation',0,'fontweight','bold'); 
        ylh = get(gca,'ylabel');
        gyl = get(ylh);
        ylp = get(ylh, 'Position');
        set(ylh, 'Rotation',0, 'Position',ylp, 'VerticalAlignment','middle', 'HorizontalAlignment','right')
    end
end
for i=1:length(miss_frac_arr)
    nexttile; histogram(a_s1_all(:,i), 20); xline(a_s1(1,1), '--r', 'LineWidth', 2); xlim([-1.55 -0.95]); ylim([0 20]);
    if i==1
        ylabel(sprintf('\\alpha_{s1}'),'Rotation',0,'fontweight','bold'); 
        ylh = get(gca,'ylabel');
        gyl = get(ylh);
        ylp = get(ylh, 'Position');
        set(ylh, 'Rotation',0, 'Position',ylp, 'VerticalAlignment','middle', 'HorizontalAlignment','right')
    end
end

t.Padding = 'compact';
t.TileSpacing = 'compact';

% Export a plot in higher resolution
% ax = gcf;
% exportgraphics(ax,"myplot.png","Resolution",300)