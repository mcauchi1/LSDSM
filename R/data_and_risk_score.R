
buildConfMat <- function(preddf, thresh_vec, predVar, trueVar) {
  # This function assumes that event is 1 if predicted value is smaller than
  # the threshold value
  # e.g. when survival prediction is smaller than X, then patient is expected
  # to experience the event
  
  # create dataframe to place the outcomes within it
  temp_df <- data.frame(matrix(ncol = 5, nrow = 0))
  colnames(temp_df) <- c("Thresh", "TP", "FP", "FN", "TN")
  
  # for every threshold value required
  for (s_thresh in thresh_vec) {
    # create row to place the outcome for that threshold value
    temp_df[nrow(temp_df) + 1,] = c(s_thresh, 0, 0, 0, 0)
    
    temp_df[nrow(temp_df), "TP"] = sum(preddf[predVar] < s_thresh & preddf[trueVar] == 1)
    temp_df[nrow(temp_df), "FP"] = sum(preddf[predVar] < s_thresh & preddf[trueVar] == 0)
    temp_df[nrow(temp_df), "FN"] = sum(preddf[predVar] >= s_thresh & preddf[trueVar] == 1)
    temp_df[nrow(temp_df), "TN"] = sum(preddf[predVar] >= s_thresh & preddf[trueVar] == 0)
  }
  
  # calculate sensitivity, specificity, and accuracy
  temp_df$sensitivity = temp_df$TP / (temp_df$TP + temp_df$FN)
  temp_df$specificity = temp_df$TN / (temp_df$TN + temp_df$FP)
  temp_df$accuracy = (temp_df$TP + temp_df$TN) / 
    (temp_df$TP + temp_df$TN + temp_df$FP + temp_df$FN)
  
  return(temp_df)
}

# plots the ROC curve from a list of dataframes
plotROCcurve <- function(DFs_list, labels, 
                         color_list=c('blue', 'orange', 'red', 'green', 'cyan')) {
  if (length(DFs_list) != length(labels)){
    print("Length of list of DFs is not equal to length of labels")
    return(0)
  }
  
  if (length(DFs_list) > length(color_list)) {
    print("Not enough colours listed (default = 5)")
    return(0)
  }
  x_roc <- seq(from=0, to=1, by=0.01)
  y_roc <- x_roc
  plot(x_roc, y_roc, type='l',
       col='black', main="ROC curves for different thresholds",
       xlab="1 - Specificity", ylab="Sensitivity", lwd=2, lty=2)
  
  for (ii in 1:length(DFs_list)) { # for every dataframe
    temp_df <- DFs_list[[ii]]
    lines(1-temp_df$specificity, temp_df$sensitivity, type='l',
          col=color_list[ii], main="ROC curves for different thresholds",
          xlab="1 - Specificity", ylab="Sensitivity", lwd=2)
    
  }
  
  legend("bottomright", 
         legend=labels,
         col=color_list[1:length(labels)],
         lty=rep(c(1),each=length(labels)), lwd=rep(c(2),each=length(labels)))
  
  return("Plot")
}

# calculates the area under the ROC curve
calcAUC <- function(ss_df) {
  # calculate the widths of the trapezoids
  xvals = diff(1-ss_df$specificity)
  
  # calculate the average of two consecutive sensitivity values
  yvals = (ss_df$sensitivity[-1] + ss_df$sensitivity[-length(ss_df$sensitivity)]) / 2
  
  auc = sum(xvals * yvals)
  
  return(auc)
}

# prepares the risk dataframe to include the patients that meet the criteria
prep_risk_df <- function(df, range_test, landmark, horiz, str_landmark) {
  
  df <- df[(df$time >= range_test[1] & df$time <= range_test[2]),]
  df <- df[!duplicated(df$id),] # one entry per patient
  # drop the censored patients within horizon period
  if (str_landmark == 'landmark') {
    horiz_time = landmark + horiz
    df <- df %>%
      filter(!(EvTime < horiz_time & event == 0))
    # those that survived beyond horizon period are assumed event-free
    df <- df %>%
      mutate(event = ifelse(EvTime > horiz_time, 0, 1))
  } else {
    df <- df %>%
      filter(!(EvTime < (time + horiz) & event == 0))
    # those that survived beyond horizon period are assumed event-free
    df <- df %>%
      mutate(event = ifelse(EvTime > (time + horiz), 0, 1))
  }
  
  return(df)
}

# necessary libraries
library("dplyr")
library("tidyverse")
library("stringr")
library("lattice")

# if using rstudio, get the current path and set it to wd using this code
library(rstudioapi)
current_path = rstudioapi::getActiveDocumentContext()$path 
setwd(dirname(current_path))

################
### Controls ###
################
## General controls
# if not using rstudio, write absolute path to make sure to save in correct location
parent_folder_data <- "..\\Data\\PH_shef_clean\\Anon\\"
save_folder_data <- "..\\Data\\PH_shef_clean\\Anon\\risk_score\\"
set.seed(1)

###########################
### 1. Data Preparation ###
###########################

# create the folders to store the data and the results (produces warning if already created)
dir.create(file.path(parent_folder_data))
dir.create(file.path(save_folder_data))

# store all relevant dfs
pat_dem_df <- read.csv(paste(parent_folder_data, "anon_patientdemographics.csv", sep=""), 
                       fileEncoding="UTF-8-BOM")
pat_dx_df <- read.csv(paste(parent_folder_data, "anon_diagnosis.csv", sep=""),
                      fileEncoding="UTF-8-BOM")
exercise_df <- read.csv(paste(parent_folder_data, "anon_new_exercisetestpage.csv", sep=""),
                        fileEncoding="UTF-8-BOM")

# choose the columns to keep
pat_dem_df <- pat_dem_df[, c('patientdemographics_sthnumber_text', 
                             'age_at_dx',
                             'patientdemographics_dateofdeath_text_mad')]
pat_dx_df <- pat_dx_df[, c('patientdemographics_sthnumber_text', 
                           'referralpage_finalprimaryphdiagnosis0_text',
                           'referralpage_whofunctionalclass_item')]
exercise_df <- exercise_df[, c('patientdemographics_sthnumber_text', 
                               'breathtests_testdate_date_mad',
                               'exercisetestpage_walkingdistance_value')]

# rename columns
pat_dem_df <- pat_dem_df %>% rename(id = patientdemographics_sthnumber_text,
                                    age = age_at_dx,
                                    date_death = patientdemographics_dateofdeath_text_mad)
pat_dx_df <- pat_dx_df %>% rename(id = patientdemographics_sthnumber_text,
                                  dx = referralpage_finalprimaryphdiagnosis0_text,
                                  WHOfc = referralpage_whofunctionalclass_item)
exercise_df <- exercise_df %>% rename(id = patientdemographics_sthnumber_text,
                                      date_e = breathtests_testdate_date_mad,
                                      walkDist = exercisetestpage_walkingdistance_value)

# remove duplicates from patient demographics and diagnosis tables
pat_dem_df <- pat_dem_df[!duplicated(pat_dem_df$id), ]
pat_dx_df <- pat_dx_df[!duplicated(pat_dx_df$id), ]

# clean the test data
exercise_df[!is.na(exercise_df['date_e']) & 
              exercise_df['date_e'] < 0 & exercise_df['date_e'] > -3, 'date_e'] <- 0

exercise_df <- exercise_df[!is.na(exercise_df['date_e']) & exercise_df['date_e'] >= 0, ]

# sort by id and date
exercise_df <- exercise_df[with(exercise_df, order(id,date_e)),]

# start merging dataframes
allpats <- merge(x = pat_dem_df, y = pat_dx_df, by = "id", all = TRUE)
allpats <- allpats[!is.na(allpats$id),]
allpats_ex <- merge(x = allpats, y = exercise_df, by = "id", all = TRUE)

orig_df <- allpats_ex

orig_df <- orig_df[!is.na(orig_df$date_e), ]

max_times <- orig_df %>%
  group_by(id) %>%
  summarize(survTime = max(date_e))

# merge the 'survTime' values back into the original dataframe
orig_df <- left_join(orig_df, max_times, by = "id")

# arrange the dataframe by 'id' and 'date_e' again
orig_df <- orig_df %>% arrange(id, date_e)

# Replace NA values in 'survTime' and 'date_death' with -1
orig_df$survTime[is.na(orig_df$surv_time)] <- -1
orig_df$date_death[is.na(orig_df$date_death)] <- -1

# Find the maximum between 'survTime' and 'date_death' and store the result in 'survTime'
orig_df$survTime <- 
  ifelse(orig_df$survTime >= orig_df$date_death, orig_df$survTime, orig_df$date_death)

# maximum time considered
max_t <- ceiling(max(orig_df$date_e))
max_t <- min(10*12, max_t) # enforce max observation period of 10 years
orig_df <- orig_df[orig_df$date_e <= max_t, ]

orig_df$eventInd = 1
orig_df[orig_df$date_death == -1, 'eventInd'] = 0
orig_df[orig_df$survTime > max_t, 'eventInd'] = 0
orig_df[orig_df$survTime > max_t, 'survTime'] = max_t

orig_df <- orig_df %>% 
  rename(
    id = id,
    baseAge = age,
    EvTime = survTime,
    event = eventInd,
    time = date_e
  )

# biomarker of interest
orig_df$y <- orig_df$walkDist

orig_df <- orig_df[complete.cases(orig_df), ] # remove rows containing NAs

norm_const <- 1000 # max value for the feature
orig_df$y[orig_df$y > norm_const] <- norm_const

# Create dummy variables for 'Class II', 'Class III', and 'Class IV'
orig_df <- orig_df %>%
  mutate(whoFC2 = ifelse(WHOfc == 'Class II', 1, 0),
         whoFC3 = ifelse(WHOfc == 'Class III', 1, 0),
         whoFC4 = ifelse(WHOfc == 'Class IV', 1, 0))

orig_df <- orig_df[, c('id', 'EvTime', 'event', 'time', 'dx', 'baseAge', 
                       'whoFC2', 'whoFC3', 'whoFC4', 'y')]

orig_df <- orig_df[orig_df$y >= 0, ] # remove y with negative values

# remove patients that have less than freq_req longitudinal measurements
freq_req <- 2
orig_df <- orig_df[orig_df$id %in% names(which(table(orig_df$id) >= 2)), ]

orig_df <- orig_df[orig_df$dx == 'B', ] # PAH patients only

orig_df.id = orig_df[!duplicated(orig_df$id),] # one entry per patient

smp_size <- floor(0.7 * nrow(orig_df.id)) # 70% training set, 30% testing set

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
train_file_loc = paste(save_folder_data, "PH_clean_dataset_train.csv", sep="")
test_file_loc = paste(save_folder_data, "PH_clean_dataset_test.csv", sep="")
write.csv(train_data, train_file_loc, row.names = FALSE)
write.csv(test_data, test_file_loc, row.names = FALSE)

# risk score based on thresholding
df_test <- test_data
landmark <- 0
horizon <- 36
accept_test_date_range = c(landmark-3, landmark+3)
year0_curr_df <- prep_risk_df(df_test, accept_test_date_range, landmark, horizon, 'landmark')

landmark <- 12
accept_test_date_range = c(landmark-2, landmark)
year1_curr_df <- prep_risk_df(df_test, accept_test_date_range, landmark, horizon, 'landmark')

landmark <- 2*12
accept_test_date_range = c(landmark-2, landmark)
year2_curr_df <- prep_risk_df(df_test, accept_test_date_range, landmark, horizon, 'landmark')

landmark <- 3*12
accept_test_date_range = c(landmark-2, landmark)
year3_curr_df <- prep_risk_df(df_test, accept_test_date_range, landmark, horizon, 'landmark')

landmark <- 4*12
accept_test_date_range = c(landmark-2, landmark)
year4_curr_df <- prep_risk_df(df_test, accept_test_date_range, landmark, horizon, 'landmark')

# risk scores - obtain sensitivity and specificity
walk_thresh_vals <- seq(from=0, to=1000, by=10)

conf_mat_risk0year_df <- buildConfMat(year0_curr_df, walk_thresh_vals, 'y', 'event')
conf_mat_risk1year_df <- buildConfMat(year1_curr_df, walk_thresh_vals, 'y', 'event')
conf_mat_risk2year_df <- buildConfMat(year2_curr_df, walk_thresh_vals, 'y', 'event')
conf_mat_risk3year_df <- buildConfMat(year3_curr_df, walk_thresh_vals, 'y', 'event')
conf_mat_risk4year_df <- buildConfMat(year4_curr_df, walk_thresh_vals, 'y', 'event')

plotROCcurve(list(conf_mat_risk0year_df, conf_mat_risk1year_df,
                  conf_mat_risk2year_df, conf_mat_risk3year_df,
                  conf_mat_risk4year_df), 
             c("Risk at Baseline", "Risk at 1 year",
               "Risk at 2 years", "Risk at 3 years",
               "Risk at 4 years"))

risk0year_auc = calcAUC(conf_mat_risk0year_df)
risk1year_auc = calcAUC(conf_mat_risk1year_df)
risk2year_auc = calcAUC(conf_mat_risk2year_df)
risk3year_auc = calcAUC(conf_mat_risk3year_df)
risk4year_auc = calcAUC(conf_mat_risk4year_df)
