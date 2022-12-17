# function to replace the empty (NULL) rows in a matrix with zeros
replace_empty_with_zero <- function(x) {
  if (nrow(x) == 0) {
    x <- c(0,0)
  }
  x
}

library("MASS")
library("splines")
library("JMbayes")
library("xtable")
library("lattice")
library("dplyr")
library("tidyverse")
library("stringr")

# if using rstudio, get the current path and set it to wd using this code
library(rstudioapi)
current_path = rstudioapi::getActiveDocumentContext()$path 
setwd(dirname(current_path))

################
### Controls ###
################

## General controls
# if not using rstudio, write absolute path to make sure to save in correct location
parent_folder_data <- "..\\Data\\PBC_data\\"
results_folder <- "..\\Data\\PBC_data\\Results\\"
allow_plots <- 1

## JM training controls
MCMC_iter <- 100000
MCMC_thin <- 1
MCMC_burnin <- 10000

# rand_int_only <- 0 means use random effects across all basis functions
#               <- 1 means use random effects at intercept only
rand_int_only <- 1

## testing controls
Delta_t <- 0.5 # spacing for the horizons
landmarks_to_test = c(5.5, 7.5, 9.5) # landmarks to test in years
max_horizon_t <- NULL

###########################
### 1. Data Preparation ###
###########################

# create the folders to store the data and the results (produces warning if already created)
dir.create(file.path(parent_folder_data))
dir.create(file.path(results_folder))

# storing the df to manipulate it
orig_df <- pbc2

# log(1+y) transformation
orig_df$y <- log(1+orig_df$serBilir)

orig_df <- orig_df %>% 
  rename(
    EvTime = years,
    event = status2,
    time = year,
    group = drug
  )

orig_df <- orig_df[, c('id', 'EvTime', 'event', 'time', 'age', 'sex', 'group', 'y')]

orig_df$sex <- as.numeric(orig_df$sex) - 1 # male = 0, female = 1
orig_df$group <- as.numeric(orig_df$group) - 1 # placebo = 0, D-pencil = 1

orig_df.id = orig_df[!duplicated(orig_df$id),] # one entry per patient

smp_size <- floor(0.5 * nrow(orig_df.id)) # 50% training set, 50% testing set
# set the seed for reproducible results
set.seed(1)
train_ind <- sample(seq_len(nrow(orig_df.id)), size = smp_size) # training set samples

# split data set into training and testing sets
train_data.id <- orig_df.id[train_ind, ]
test_data.id <- orig_df.id[-train_ind, ]

# order according to ID
train_data.id <- train_data.id[order(train_data.id$id), ]
test_data.id <- test_data.id[order(test_data.id$id), ]

# split data set into training and testing sets for the longitudinal data
train_data <- orig_df[orig_df$id %in% train_data.id$id, ]
test_data <- orig_df[orig_df$id %in% test_data.id$id, ]

# store data in csv files - to be used in MATLAB for comparing models
train_file_loc = paste(parent_folder_data, "PBC_dataset_train.csv", sep="")
test_file_loc = paste(parent_folder_data, "PBC_dataset_test.csv", sep="")
write.csv(train_data, train_file_loc, row.names = FALSE)
write.csv(test_data, test_file_loc, row.names = FALSE)

#########################################
### 2. Train the standard Joint Model ###
#########################################

# train the mixed effects model (spline-based)
if (rand_int_only) {
  lmeFit <- lme(y ~ ns(time, 3, Bound = c(0,14.31)),
                random = ~ 1|id,
                data = train_data)
} else {
  lmeFit <- lme(y ~ ns(time, 3, Bound = c(0,14.31)),
                random = list(id = pdDiag(form =~ns(time, 3, Bound = c(0,14.31)))),
                data = train_data)
}

# Fit the survival model
survFit <- coxph(Surv(EvTime, event) ~ age + sex + group, data = train_data.id, x = TRUE)

# Fit the joint model with the current biomarker value affecting the hazard function
jointFit1 <- jointModelBayes(lmeFit, survFit, timeVar = "time", 
                             n.iter = MCMC_iter, n.thin = MCMC_thin,
                             n.burnin = MCMC_burnin)

# find the maximum observed time and place that as the maximum censoring time
censor_time <- floor(max(orig_df.id$EvTime))
if (is.null(max_horizon_t)) { # if not defined
  max_horizon_t <- censor_time - landmarks_to_test[1]
}
horiz_arr <- seq(Delta_t, max_horizon_t, Delta_t) # create a set of horizons to test


#########################
### 3. Test the model ###
#########################
test_model = jointFit1

for (landmark in landmarks_to_test) { # for every landmark
  
  # create a dataframe to hold results for BS, PE, AUC
  perf_metrics_df <- data.frame(matrix(vector(), 3, 0,
                                       dimnames=list(c(), c())),
                                stringsAsFactors=F)
  
  # filter the data to retain those that are censored/experience event after landmark time
  df_test <- test_data[test_data$EvTime > landmark,]
  df_test <- df_test[df_test$time <= landmark,]
  df_test.ID <- df_test %>% group_by(id) %>% top_n(1,time)
  
  # number of patients that are still at risk at the landmark time
  no_of_pat_filt = nrow(df_test.ID)
  
  for (t_horiz in horiz_arr) { # for every horizon to consider
    col_title <- paste(c("horiz_", t_horiz), collapse="") # create a column header
    
    t_est <- landmark + t_horiz # time at which we make predictions
    
    # make dynamic survival predictions
    surv_predict_vals <- survfitJM(test_model, df_test, idVar = "id", simulate = FALSE,
                                   survTimes = t_est,
                                   last.time = rep(c(landmark), each=nrow(df_test.ID)))
    # extract survival predictions
    surv_pred_df <- do.call(rbind.data.frame, surv_predict_vals$summaries)
    
    
    ######################################
    ### 3.1 Time-dependent Brier Score ###
    ######################################
    
    # Kaplan Meier curve for censored observations
    KM_cens <- survfit(Surv(EvTime, 1-event) ~ 1, data = df_test.ID)
    if (allow_plots) {
      plot(KM_cens, lwd = 2, mark.time = FALSE, xlim = c(-1.5, 15),
           xlab = "Follow-up Time (years)", ylab = "KM Censoring curve")
    }
    
    # KM (censored) variables
    G_time = KM_cens$time
    G_cens = KM_cens$surv
    
    # Finding the index for the Censored Kaplan Meier at t_est
    G_idx = 0
    if (length(G_time[G_time < t_est]) > 0) {
      G_idx = which.max(G_time[G_time < t_est])
    }
    
    
    # Checking every patient and adding their score to the Brier Score
    N_BS = no_of_pat_filt # number of patients left
    BS_val = 0
    
    tmp_val = 0 # checking whether the censoring weights add up to 1
    no_of_pats_used = 0
    
    # creating a data frame with the required variables to calculate performance metrics
    # survival prediction, experienced event in time frame, censored, censoring index
    subj_at_risk <- df_test.ID
    subj_at_risk$predSurv <- surv_pred_df$predSurv
    subj_at_risk$ind_tmp <- subj_at_risk$EvTime <= t_est & subj_at_risk$event == 1
    subj_at_risk$ind_risk <- subj_at_risk$EvTime >= t_est
    subj_at_risk$G_idx <- G_idx
    
    if (G_idx > 0) { # if KM value for censoring is smaller than 1
      # arrange G_idx for those patients that experience event in time frame of interest
      red_subj_at_risk <- subj_at_risk[which(subj_at_risk$ind_tmp == 1), ]
      for (i in 1:nrow(red_subj_at_risk)) { # for those subjects that experience event in time frame
        red_subj_at_risk$G_idx[i] <- 0
        # change censoring index
        if (length(G_time[G_time < red_subj_at_risk$EvTime[i]]) > 0 & nrow(red_subj_at_risk) > 0) {
          red_subj_at_risk$G_idx[i] <- which.max(G_time[G_time < red_subj_at_risk$EvTime[i]])
        }
      }
      subj_at_risk[which(subj_at_risk$ind_tmp == 1), 'G_idx'] <- red_subj_at_risk$G_idx
    }
    
    # if G_idx = 0, then KM value for censoring = 1
    subj_at_risk$G_cens <- 1
    
    # if there are patients with G_idx > 0 (i.e. G_cens != 1)
    if (length(which(subj_at_risk$G_idx > 0) > 0)) {
      # find the KM value for censoring for those patients at time of interest
      subj_at_risk[which(subj_at_risk$G_idx > 0), 'G_cens'] <- 
        G_cens[unlist(subj_at_risk[which(subj_at_risk$G_idx > 0), 'G_idx'])]
    }
    
    # Calculate Brier Score for every patient
    subj_at_risk$BS <- (0 - subj_at_risk$predSurv)^2 * subj_at_risk$ind_tmp / subj_at_risk$G_cens + 
      (1 - subj_at_risk$predSurv)^2 * subj_at_risk$ind_risk / subj_at_risk$G_cens
    
    # Find the total Brier Score
    BS_val <- sum(subj_at_risk$BS) / N_BS
    
    
    ##########################
    ### 3.2 AUC estimation ###
    ##########################
    
    # checking every pair of patients, and adding their score to the AUC
    AUC_val = 0
    AUC_num_sum = 0
    AUC_den_sum = 0
    
    # Start by identifying the indicator variables in new columns
    subj_at_risk$indGreater <- subj_at_risk$EvTime >= t_est
    subj_at_risk$indGreater <- subj_at_risk$indGreater*1
    subj_at_risk$indSmaller <- 1 - subj_at_risk$indGreater
    subj_at_risk$indSmallerEv <- subj_at_risk$indSmaller * subj_at_risk$event
    subj_at_risk$W = 0
    
    # checking for the number of pairs tested
    auc_count_var <- 0
    
    # Start by identifying the indicator variables in new columns
    subj_at_risk$indGreater <- subj_at_risk$EvTime >= t_est
    subj_at_risk$indGreater <- subj_at_risk$indGreater*1
    subj_at_risk$indSmaller <- 1 - subj_at_risk$indGreater
    subj_at_risk$indSmallerEv <- subj_at_risk$indSmaller * subj_at_risk$event
    subj_at_risk$W = 0
    
    # checking for the number of pairs tested
    auc_count_var <- 0
    
    # preparing the weights for the equation for every patient
    subj_at_risk$W <- (subj_at_risk$ind_tmp + subj_at_risk$ind_risk) / subj_at_risk$G_cens
    
    # for every patient i
    for (i in 1:no_of_pat_filt) {
      
      # if patient i experienced the event, we proceed with the AUC for every pair
      if (subj_at_risk$indSmallerEv[i]) {
        D_i <- subj_at_risk$indSmallerEv[i]
        W_i <- subj_at_risk$W[i]
        for (j in 1:no_of_pat_filt) {
          D_j <- subj_at_risk$indSmallerEv[j]
          W_j <- subj_at_risk$W[j]
          
          # concordance index (=1 if patient i has lower survival than patient j)
          c_idx_ind <- subj_at_risk$predSurv[i] < subj_at_risk$predSurv[j]
          
          # add to the numerator and denominator for calculation of AUC
          AUC_num_sum <- AUC_num_sum + (c_idx_ind * D_i * (1 - D_j) * W_i * W_j)
          AUC_den_sum <- AUC_den_sum + (D_i * (1 - D_j) * W_i * W_j)
          
          auc_count_var <- auc_count_var + D_i * (1 - D_j)
        }
      }
      
    }
    
    AUC_val <- AUC_num_sum / AUC_den_sum
    
    
    ############################
    ### 3.3 Prediction Error ###
    ############################
    
    # find the values of 'true_pi', which are pi(u|T_i) where u is t_est, and T_i
    # is the time of death/censoring
    survTime_from_T_i <- survfitJM(test_model, df_test, idVar = "id", simulate = FALSE,
                                   survTimes = t_est,
                                   last.time = df_test.ID$EvTime)
    
    # some predictions will have empty rows - we wish to retain these rows since we then
    # want to join it with the subj_at_risk df
    list_of_preds <- survTime_from_T_i$summaries
    surv_pred_list <- lapply(list_of_preds, replace_empty_with_zero)
    true_pi_tmp <- do.call(rbind.data.frame, surv_pred_list)
    
    subj_at_risk$true_pi <- true_pi_tmp$predSurv
    
    # calculate prediciton error for every patient
    subj_at_risk$predErr <- subj_at_risk$indGreater * (1 - subj_at_risk$predSurv)^2 +
      subj_at_risk$event * subj_at_risk$indSmaller * (0 - subj_at_risk$predSurv)^2 + 
      (1 - subj_at_risk$event) * subj_at_risk$indSmaller * 
      (subj_at_risk$true_pi * (1 - subj_at_risk$predSurv)^2 +
         (1 - subj_at_risk$true_pi) * (0 - subj_at_risk$predSurv)^2)
    
    # Sum over the entire predErr column and divide by number of patients (rows)
    PE_val <- sum(subj_at_risk$predErr) / nrow(subj_at_risk)
    
    # store the performance metrics for this landmark and horizon
    perf_metrics_vec <- c(BS_val, PE_val, AUC_val)
    perf_metrics_df[, col_title] = perf_metrics_vec # row for every landmark
  }
  
  # when the results for a particular landmark are deduced, store the results
  end_string <- paste("landmark", landmark, "_several_horiz", sep="")
  end_string <- str_replace_all(end_string, "\\.", "_")
  if (rand_int_only) {
    results_file_loc = paste(results_folder, end_string, "_rand_int_only.csv", sep="")
  } else {
    results_file_loc = paste(results_folder, end_string, ".csv", sep="")
  }
  
  write.csv(perf_metrics_df, results_file_loc, row.names = FALSE)
  
}
