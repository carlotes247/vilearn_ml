install.packages("lm.beta")
use("lm.beta")


# load data file for all dyads and triads separately as there are different DG configurations.(the groups are at floor level and without outlier)
samelevel_60s_DGconfigs_T_data <-read.csv('C:/Users/dobregeo/Documents/GitHub/vilearn_ml/Recordings/SavedData/v2_no_low_sampled/60s_TE_correlation_T_(participants_same_level_12groups).csv', sep=';')

samelevel_60s_DGconfigs_D_data <-read.csv('C:/Users/dobregeo/Documents/GitHub/vilearn_ml/Recordings/SavedData/v2_no_low_sampled/60s_TE_correlation_D_(participants_same_level_8groups).csv', sep=';')


#create the multiple regression models for the triads and dyads
TE_samelevel_DGconfigs_T_model = lm(formula = TE ~ BPM + blink_durations + X0_D1 + X1_D1 + X2_D1_same + X2_D1_different + X3_D1 +  MG_D0 + MG_D1, samelevel_60s_DGconfigs_T_data)

TE_samelevel_DGconfigs_D_model = lm(formula = TE ~ BPM + blink_durations +  MG  + X1d_DG + X0_D1, samelevel_60s_DGconfigs_D_data)


#show the model results and the beta values for each independent variable in the model
summary(TE_samelevel_DGconfigs_T_model)
lm.beta(TE_samelevel_DGconfigs_T_model)
summary(TE_samelevel_DGconfigs_D_model)
lm.beta(TE_samelevel_DGconfigs_D_model)

