install.packages("lm.beta")
use("lm.beta")


# load data file for all 20 groups (all floor level and without outlier)
samelevel_60s_data <-read.csv('C:/Users/dobregeo/Documents/GitHub/vilearn_ml/Recordings/SavedData/v2_no_low_sampled/60s_TE_correlation(participants_same_level_20groups).csv', sep=';')

#create the multiple regression model
TE_samelevel_all_groups_model = lm(formula = TE ~ MG + X1d_DG + BPM + blink_durations, samelevel_60s_data)

#show the model results
summary(TE_samelevel_all_groups_model)

#show the beta values for each independent variable in the model
lm.beta(TE_samelevel_all_groups_model)

#split the data into dyads and triad 
X <-split(samelevel_60s_data, samelevel_60s_data$group_type)
samelevel_D_60s_data <- X[[1]]
samelevel_T_60s_data <- X[[2]]

#create the models for both dyads and triads
TE_samelevel_D_model = lm(formula = TE ~ MG + X1d_DG + BPM + blink_durations, samelevel_D_60s_data)
TE_samelevel_T_model = lm(formula = TE ~ MG + X1d_DG + BPM + blink_durations, samelevel_T_60s_data)

#show the model results and the beta info for each independent variable
summary(TE_samelevel_D_model)
lm.beta(TE_samelevel_D_model)
summary(TE_samelevel_T_model)
lm.beta(TE_samelevel_T_model)