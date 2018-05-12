#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 08:57:46 2018

@author: angus
"""

import os
os.chdir('/home/angus/projects/betting')

import numpy as np
import pandas as pd
from tqdm import tqdm
import xgboost as xgb
import sklearn as skl
from sklearn import linear_model
import datetime
import glob
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import optimizers
from keras import initializers
from keras import models
from keras import regularizers
from sklearn.preprocessing import LabelEncoder
from scipy.stats import norm

# read data
combined_data_added_features = pd.read_csv('all_data/combined_data_added_features.csv')



# put features into easily callable lists
homeWDLfeatures_W = [feature for feature in combined_data_added_features.columns.values if ((feature.startswith('home')) & (feature.endswith('_W')))]
homeWDLfeatures_D = [feature for feature in combined_data_added_features.columns.values if ((feature.startswith('home')) & (feature.endswith('_D')))]
homeWDLfeatures_L = [feature for feature in combined_data_added_features.columns.values if ((feature.startswith('home')) & (feature.endswith('_L')))]
homeGoalsForfeatures = [feature for feature in combined_data_added_features.columns.values if feature.startswith('homePRGoals')]
homeGoalsAgainstfeatures = [feature for feature in combined_data_added_features.columns.values if feature.startswith('homePRGoalsAgaints')]
homePROppositionPointsfeatures = [feature for feature in combined_data_added_features.columns.values if feature.startswith('homePROppositionPoints')]
homePRIsHomeFeatures = [feature for feature in combined_data_added_features.columns.values if feature.startswith('homePRIsHomeGame')]
homePRShotsForFeatures = [feature for feature in combined_data_added_features.columns.values if feature.startswith('homePRShotsFor')]
homePRShotsAgainstFeatures = [feature for feature in combined_data_added_features.columns.values if feature.startswith('homePRShotsAgainst')]
homePRSOTForFeatures = [feature for feature in combined_data_added_features.columns.values if feature.startswith('homePRSOTFor')]
homePRSOTAgainstFeatures = [feature for feature in combined_data_added_features.columns.values if feature.startswith('homePRSOTAgainst')]
homePRCornersForFeatures = [feature for feature in combined_data_added_features.columns.values if feature.startswith('homePRCornersFor')]
homePRCornersAgainstFeatures = [feature for feature in combined_data_added_features.columns.values if feature.startswith('homePRCornersAgainst')]
homePRFoulsForFeatures = [feature for feature in combined_data_added_features.columns.values if feature.startswith('homePRFoulsFor')]
homePRFoulsAgainstFeatures = [feature for feature in combined_data_added_features.columns.values if feature.startswith('homePRFoulsAgainst')]
homePRYellowsForFeatures = [feature for feature in combined_data_added_features.columns.values if feature.startswith('homePRYellowsFor')]
homePRYellowsAgainstFeatures = [feature for feature in combined_data_added_features.columns.values if feature.startswith('homePRYellowsAgainst')]
homePRRedsForFeatures = [feature for feature in combined_data_added_features.columns.values if feature.startswith('homePRRedsFor')]
homePRRedsAgainstFeatures = [feature for feature in combined_data_added_features.columns.values if feature.startswith('homePRRedsAgainst')]

awayWDLfeatures_W = [feature for feature in combined_data_added_features.columns.values if ((feature.startswith('away')) & (feature.endswith('_W')))]
awayWDLfeatures_D = [feature for feature in combined_data_added_features.columns.values if ((feature.startswith('away')) & (feature.endswith('_D')))]
awayWDLfeatures_L = [feature for feature in combined_data_added_features.columns.values if ((feature.startswith('away')) & (feature.endswith('_L')))]
awayGoalsForfeatures = [feature for feature in combined_data_added_features.columns.values if feature.startswith('awayPRGoals')]
awayGoalsAgainstfeatures = [feature for feature in combined_data_added_features.columns.values if feature.startswith('awayPRGoalsAgaints')]
awayPROppositionPointsfeatures = [feature for feature in combined_data_added_features.columns.values if feature.startswith('awayPROppositionPoints')]
awayPRIsHomeFeatures = [feature for feature in combined_data_added_features.columns.values if feature.startswith('awayPRIsHomeGame')]
awayPRShotsForFeatures = [feature for feature in combined_data_added_features.columns.values if feature.startswith('awayPRShotsFor')]
awayPRShotsAgainstFeatures = [feature for feature in combined_data_added_features.columns.values if feature.startswith('awayPRShotsAgainst')]
awayPRSOTForFeatures = [feature for feature in combined_data_added_features.columns.values if feature.startswith('awayPRSOTFor')]
awayPRSOTAgainstFeatures = [feature for feature in combined_data_added_features.columns.values if feature.startswith('awayPRSOTAgainst')]
awayPRCornersForFeatures = [feature for feature in combined_data_added_features.columns.values if feature.startswith('awayPRCornersFor')]
awayPRCornersAgainstFeatures = [feature for feature in combined_data_added_features.columns.values if feature.startswith('awayPRCornersAgainst')]
awayPRFoulsForFeatures = [feature for feature in combined_data_added_features.columns.values if feature.startswith('awayPRFoulsFor')]
awayPRFoulsAgainstFeatures = [feature for feature in combined_data_added_features.columns.values if feature.startswith('awayPRFoulsAgainst')]
awayPRYellowsForFeatures = [feature for feature in combined_data_added_features.columns.values if feature.startswith('awayPRYellowsFor')]
awayPRYellowsAgainstFeatures = [feature for feature in combined_data_added_features.columns.values if feature.startswith('awayPRYellowsAgainst')]
awayPRRedsForFeatures = [feature for feature in combined_data_added_features.columns.values if feature.startswith('awayPRRedsFor')]
awayPRRedsAgainstFeatures = [feature for feature in combined_data_added_features.columns.values if feature.startswith('awayPRRedsAgainst')]


divFeatures = ['N1', 'E1', 'SC3', 'SC2', 'D1', 'B1', 'I2', 'G1', 'E3', 'T1', 'EC','D2', 'F1', 'I1', 'P1', 'SP2', 'SP1', 'E2', 'SC0', 'E0', 'F2', 'SC1']

# remove rows which don't have sufficient past history
past_features_to_include = 20
past_features_to_include_alternative = 20
combined_data_added_features_with_history = combined_data_added_features[combined_data_added_features['homePRYear'+str(past_features_to_include)].notnull()]

# optional: remove rows without corners history etc
combined_data_added_features_with_history = combined_data_added_features_with_history[combined_data_added_features_with_history[homePRShotsForFeatures+homePRShotsAgainstFeatures+homePRSOTForFeatures+homePRSOTAgainstFeatures+homePRCornersForFeatures+homePRCornersAgainstFeatures+homePRFoulsForFeatures+homePRFoulsAgainstFeatures+homePRYellowsForFeatures+homePRYellowsAgainstFeatures+homePRRedsForFeatures+homePRRedsAgainstFeatures+
                                                                                                                                awayPRShotsForFeatures+awayPRShotsAgainstFeatures+awayPRSOTForFeatures+awayPRSOTAgainstFeatures+awayPRCornersForFeatures+awayPRCornersAgainstFeatures+awayPRFoulsForFeatures+awayPRFoulsAgainstFeatures+awayPRYellowsForFeatures+awayPRYellowsAgainstFeatures+awayPRRedsForFeatures+awayPRRedsAgainstFeatures].isnull().sum(axis=1)==0]

combined_data_added_features_with_history = combined_data_added_features_with_history[combined_data_added_features_with_history['BbAv<2.5'].notnull()]

combined_data_added_features_with_history['BbAv>2.5prob'] = 1/combined_data_added_features_with_history['BbAv>2.5']/(1/combined_data_added_features_with_history['BbAv>2.5']+1/combined_data_added_features_with_history['BbAv<2.5'])
combined_data_added_features_with_history['BbAv<2.5prob'] = 1/combined_data_added_features_with_history['BbAv<2.5']/(1/combined_data_added_features_with_history['BbAv>2.5']+1/combined_data_added_features_with_history['BbAv<2.5'])
combined_data_added_features_with_history['BbAv2.5bookiesgain'] = (1/combined_data_added_features_with_history['BbAv>2.5']+1/combined_data_added_features_with_history['BbAv<2.5'])
np.mean(combined_data_added_features_with_history['BbAv2.5bookiesgain'])
np.min(combined_data_added_features_with_history['BbAv2.5bookiesgain'])
np.max(combined_data_added_features_with_history['BbAv2.5bookiesgain'])
np.std(combined_data_added_features_with_history['BbAv2.5bookiesgain'])

combined_data_added_features_with_history['totalGoals'] = combined_data_added_features_with_history['FTHG']+combined_data_added_features_with_history['FTAG']
combined_data_added_features_with_history['goals>2.5'] = (combined_data_added_features_with_history['totalGoals']>2.5)*1

bookiesFeatures = ['BbAv>2.5prob', 'BbAv<2.5prob', 'BbAv2.5bookiesgain']







# train one model on all data to get initial weights
# then bootstrap data and train 19 more models (hopefully only have to do a few iterations for each given initial weights)
train_to_season = 2016
predictors = ['seasonWeek']+homeWDLfeatures_W[:past_features_to_include]+homeWDLfeatures_D[:past_features_to_include]+homeWDLfeatures_L[:past_features_to_include]+homeGoalsForfeatures[:past_features_to_include]+homeGoalsAgainstfeatures[:past_features_to_include]+homePROppositionPointsfeatures[:past_features_to_include]+awayWDLfeatures_W[:past_features_to_include]+awayWDLfeatures_D[:past_features_to_include]+awayWDLfeatures_L[:past_features_to_include]+awayGoalsForfeatures[:past_features_to_include]+awayGoalsAgainstfeatures[:past_features_to_include]+awayPROppositionPointsfeatures[:past_features_to_include]+divFeatures+homePRIsHomeFeatures+awayPRIsHomeFeatures+homePRShotsForFeatures[:past_features_to_include]+homePRShotsAgainstFeatures[:past_features_to_include]+homePRSOTForFeatures[:past_features_to_include]+homePRSOTAgainstFeatures[:past_features_to_include]+homePRFoulsForFeatures[:past_features_to_include]+homePRFoulsAgainstFeatures[:past_features_to_include]+homePRYellowsForFeatures[:past_features_to_include]+homePRYellowsAgainstFeatures[:past_features_to_include]+homePRRedsForFeatures[:past_features_to_include]+homePRRedsAgainstFeatures[:past_features_to_include]+awayPRShotsForFeatures[:past_features_to_include]+awayPRShotsAgainstFeatures[:past_features_to_include]+awayPRSOTForFeatures[:past_features_to_include]+awayPRSOTAgainstFeatures[:past_features_to_include]+awayPRFoulsForFeatures[:past_features_to_include]+awayPRFoulsAgainstFeatures[:past_features_to_include]+awayPRYellowsForFeatures[:past_features_to_include]+awayPRYellowsAgainstFeatures[:past_features_to_include]+awayPRRedsForFeatures[:past_features_to_include]+awayPRRedsAgainstFeatures[:past_features_to_include]+bookiesFeatures
predictors_alt = ['seasonWeek']+homeWDLfeatures_W[:past_features_to_include]+homeWDLfeatures_D[:past_features_to_include]+homeWDLfeatures_L[:past_features_to_include]+homeGoalsForfeatures[:past_features_to_include]+homeGoalsAgainstfeatures[:past_features_to_include]+homePROppositionPointsfeatures[:past_features_to_include]+awayWDLfeatures_W[:past_features_to_include]+awayWDLfeatures_D[:past_features_to_include]+awayWDLfeatures_L[:past_features_to_include]+awayGoalsForfeatures[:past_features_to_include]+awayGoalsAgainstfeatures[:past_features_to_include]+awayPROppositionPointsfeatures[:past_features_to_include]+divFeatures+homePRIsHomeFeatures+awayPRIsHomeFeatures+homePRShotsForFeatures[:past_features_to_include]+homePRShotsAgainstFeatures[:past_features_to_include]+homePRSOTForFeatures[:past_features_to_include]+homePRSOTAgainstFeatures[:past_features_to_include]+homePRFoulsForFeatures[:past_features_to_include]+homePRFoulsAgainstFeatures[:past_features_to_include]+homePRYellowsForFeatures[:past_features_to_include]+homePRYellowsAgainstFeatures[:past_features_to_include]+homePRRedsForFeatures[:past_features_to_include]+homePRRedsAgainstFeatures[:past_features_to_include]+awayPRShotsForFeatures[:past_features_to_include]+awayPRShotsAgainstFeatures[:past_features_to_include]+awayPRSOTForFeatures[:past_features_to_include]+awayPRSOTAgainstFeatures[:past_features_to_include]+awayPRFoulsForFeatures[:past_features_to_include]+awayPRFoulsAgainstFeatures[:past_features_to_include]+awayPRYellowsForFeatures[:past_features_to_include]+awayPRYellowsAgainstFeatures[:past_features_to_include]+awayPRRedsForFeatures[:past_features_to_include]+awayPRRedsAgainstFeatures[:past_features_to_include]
train_x = combined_data_added_features_with_history[predictors][combined_data_added_features_with_history['seasonEndYear']<=train_to_season].values
train_x_alt = combined_data_added_features_with_history[predictors_alt][combined_data_added_features_with_history['seasonEndYear']<=train_to_season].values
train_col_means = train_x.mean(axis=0)
train_col_stds = train_x.astype(float).std(axis=0)*10
train_col_stds = train_col_stds + (train_col_stds==0)*1 # adds 1 where the stdev is 0 to not break division
subset = [predictor in predictors_alt for predictor in predictors]
train_x = (train_x - train_col_means)/train_col_stds
train_x_alt = (train_x_alt - train_col_means[subset])/(train_col_stds[subset])

train_y = combined_data_added_features_with_history['goals>2.5'][combined_data_added_features_with_history['seasonEndYear']<=train_to_season].values
#train_y_alt = (pd.get_dummies(combined_data_added_features_with_history['FTR']))[combined_data_added_features_with_history['seasonEndYear']==train_to_season].values
test_x = combined_data_added_features_with_history[predictors][combined_data_added_features_with_history['seasonEndYear']==(train_to_season+1)].values
test_x_alt = combined_data_added_features_with_history[predictors_alt][combined_data_added_features_with_history['seasonEndYear']==(train_to_season+1)].values
test_x = (test_x - train_col_means)/train_col_stds
test_x_alt = (test_x_alt - train_col_means[subset])/train_col_stds[subset]
test_y = combined_data_added_features_with_history['goals>2.5'][combined_data_added_features_with_history['seasonEndYear']==(train_to_season+1)].values


input_dimension = len(predictors)
input_dimension_alt = len(predictors_alt)

# optimizers
optim_rmsprop = optimizers.RMSprop(lr=0.0001, rho=0.9)
optim_sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.5, nesterov=True)
optim_adagrad = optimizers.Adagrad(lr=0.1, decay=0)
optim_adam = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0)

number_epochs_init=50
batch_sizes=2**7
val_split=0.1
dropout = 0.0
weights = 1/np.array(((combined_data_added_features_with_history['FTR']=='H')*combined_data_added_features_with_history['B365H']+(combined_data_added_features_with_history['FTR']=='D')*combined_data_added_features_with_history['B365D']+(combined_data_added_features_with_history['FTR']=='A')*combined_data_added_features_with_history['B365A'])[combined_data_added_features_with_history['seasonEndYear']<=train_to_season])
#weights = (combined_data_added_features_with_history['seasonEndYear']-min(combined_data_added_features_with_history['seasonEndYear']))[combined_data_added_features_with_history['seasonEndYear']<=train_to_season].values/max(combined_data_added_features_with_history['seasonEndYear']-min(combined_data_added_features_with_history['seasonEndYear']))
#weights = np.zeros(train_x.shape[0])+1
weights_alt = np.zeros(train_x.shape[0])+1



### train base model
model1 = Sequential()
model1.add(Dropout(dropout, input_shape=(input_dimension_alt,)))
model1.add(Dense(input_dimension_alt, input_dim=input_dimension_alt, activation='relu',
           kernel_regularizer=regularizers.l1(0.0001),
           #activity_regularizer=regularizers.l1(0.01)
           ))
model1.add(Dropout(dropout))
model1.add(Dense(input_dimension_alt, activation='relu',
           kernel_regularizer=regularizers.l1(0.0001),
           #activity_regularizer=regularizers.l1(0.01)
           ))
model1.add(Dense(input_dimension_alt, activation='relu',
           kernel_regularizer=regularizers.l1(0.0001)
           ))
model1.add(Dense(1, activation='sigmoid'))
model1.compile(loss='binary_crossentropy', optimizer=optim_sgd, metrics=['accuracy'])
model1.fit(train_x_alt,train_y,epochs=number_epochs_init,batch_size=batch_sizes,validation_split=val_split, sample_weight=weights_alt)



### samole data and retrain 19 more models
number_models = 20
model2=models.clone_model(model1)
model3=models.clone_model(model1)
model4=models.clone_model(model1)
model5=models.clone_model(model1)
model6=models.clone_model(model1)
model7=models.clone_model(model1)
model8=models.clone_model(model1)
model9=models.clone_model(model1)
model10=models.clone_model(model1)
model11=models.clone_model(model1)
model12=models.clone_model(model1)
model13=models.clone_model(model1)
model14=models.clone_model(model1)
model15=models.clone_model(model1)
model16=models.clone_model(model1)
model17=models.clone_model(model1)
model18=models.clone_model(model1)
model19=models.clone_model(model1)
model20=models.clone_model(model1)

model_list=[model2, model3, model4, model5, model6, model7, model8, model9, model10, model11,
            model12, model13, model14, model15, model16, model17, model18, model19, model20]

number_epochs_refit=5
sample_proportion = 0.5
samples = []
for i in range(number_models-1):
    samples.append(list(np.random.choice(train_x_alt.shape[0], int(train_x_alt.shape[0]*0.2))))

for i in range(len(model_list)):
    print('refitting model ', i+2)
    model_list[i].set_weights(model1.get_weights())
    model_list[i].compile(loss='binary_crossentropy', optimizer=optim_adagrad, metrics=['accuracy'])
    model_list[i].fit(train_x_alt[samples[i],:],train_y[samples[i]],epochs=number_epochs_refit,batch_size=batch_sizes,validation_split=val_split, sample_weight=weights_alt[samples[i]])

# use this bit of code to train some models more if need be
model10.set_weights(model1.get_weights())
model10.compile(loss='binary_crossentropy', optimizer=optim_adagrad, metrics=['accuracy'])
model10.fit(train_x_alt[samples[8],:],train_y[samples[8]],epochs=5,batch_size=batch_sizes,validation_split=val_split, sample_weight=weights_alt[samples[8]])


### create some predictions from each model
test_year = 2017
test_x = combined_data_added_features_with_history[predictors][combined_data_added_features_with_history['seasonEndYear']==test_year].values
test_x_alt = combined_data_added_features_with_history[predictors_alt][combined_data_added_features_with_history['seasonEndYear']==(test_year)].values
test_x = (test_x - train_col_means)/train_col_stds
test_x_alt = (test_x_alt - train_col_means[subset])/train_col_stds[subset]
test_y = combined_data_added_features_with_history['goals>2.5'][combined_data_added_features_with_history['seasonEndYear']==(test_year)].values


predictions_and_bets = combined_data_added_features_with_history[['Date','BbAv>2.5','BbAv<2.5', 'BbAv>2.5prob', 'BbAv<2.5prob', 'BbAv2.5bookiesgain','totalGoals', 'goals>2.5']][combined_data_added_features_with_history['seasonEndYear']==(test_year)]
full_model_list = [model1]+model_list
for i in tqdm(range(len(full_model_list))):
    model = full_model_list[i]
    test_set = test_x_alt
    predictions_and_bets['preds'+str(i+1)] = model.predict(test_set)

predsList = ['preds'+str(i+1) for i in range(len(full_model_list))]

# calculate mean and st dev of predictions
predictions_and_bets['preds_mean'] = np.mean(predictions_and_bets[predsList], axis=1)
predictions_and_bets['preds_sd'] = np.std(predictions_and_bets[predsList], axis=1)

# calculate lower probability for a given confidence
confidence = 0.7
predictions_and_bets['lower_confidence'] = norm.ppf(1-confidence, predictions_and_bets['preds_mean'], predictions_and_bets['preds_sd'])
predictions_and_bets['upper_confidence'] = norm.ppf(confidence, predictions_and_bets['preds_mean'], predictions_and_bets['preds_sd'])

# assign bets based on the lower confidence probability being above a threshold higher than the bookies' probability
probability_margin = 0.2
predictions_and_bets['over_lower_conf_based_bets'] = (predictions_and_bets['lower_confidence']>(1/predictions_and_bets['BbAv>2.5']+probability_margin))*1
predictions_and_bets['under_lower_conf_based_bets'] = (1-(predictions_and_bets['upper_confidence'])>(1/predictions_and_bets['BbAv<2.5']+probability_margin))*1

# calculate winnings based on lower confidence probability bets
predictions_and_bets['over_lower_conf_winnings'] = predictions_and_bets['over_lower_conf_based_bets']*predictions_and_bets['BbAv>2.5']*(predictions_and_bets['goals>2.5'])
predictions_and_bets['under_lower_conf_winnings'] = predictions_and_bets['under_lower_conf_based_bets']*predictions_and_bets['BbAv<2.5']*(1-predictions_and_bets['goals>2.5'])


print('total bets: ',sum(predictions_and_bets['over_lower_conf_based_bets'])+sum(predictions_and_bets['under_lower_conf_based_bets']),
      'total winnings: ',sum(predictions_and_bets['over_lower_conf_winnings'])+sum(predictions_and_bets['under_lower_conf_winnings']),
      'return: ', ((sum(predictions_and_bets['over_lower_conf_winnings'])+sum(predictions_and_bets['under_lower_conf_winnings']))/(sum(predictions_and_bets['over_lower_conf_based_bets'])+sum(predictions_and_bets['under_lower_conf_based_bets']))-1)*100, '%')
print('over bets: ',sum(predictions_and_bets['over_lower_conf_based_bets']), 'winnings: ',sum(predictions_and_bets['over_lower_conf_winnings']))
print('under bets: ',sum(predictions_and_bets['under_lower_conf_based_bets']), 'winnings: ',sum(predictions_and_bets['under_lower_conf_winnings']))






### betting strategies
# 1. bet max(0,x/c )% of total money where x = prediction - odds based probability
predictions_and_bets['Apred_minus_odds_prob'] = predictions_and_bets['Alower_confidence'] - (1/predictions_and_bets['PSA'])
predictions_and_bets['Apred_minus_odds_prob'] = predictions_and_bets['Apred_minus_odds_prob']*(predictions_and_bets['Apred_minus_odds_prob']>0)
predictions_and_bets['Dpred_minus_odds_prob'] = predictions_and_bets['Dlower_confidence'] - (1/predictions_and_bets['PSD'])
predictions_and_bets['Dpred_minus_odds_prob'] = predictions_and_bets['Dpred_minus_odds_prob']*(predictions_and_bets['Dpred_minus_odds_prob']>0)
predictions_and_bets['Hpred_minus_odds_prob'] = predictions_and_bets['Hlower_confidence'] - (1/predictions_and_bets['PSH'])
predictions_and_bets['Hpred_minus_odds_prob'] = predictions_and_bets['Hpred_minus_odds_prob']*(predictions_and_bets['Hpred_minus_odds_prob']>0)

bets_scaling = 1/10
predictions_and_bets['Aproportion_bet'] = predictions_and_bets['Apred_minus_odds_prob']*bets_scaling
predictions_and_bets['Dproportion_bet'] = predictions_and_bets['Dpred_minus_odds_prob']*bets_scaling
predictions_and_bets['Hproportion_bet'] = predictions_and_bets['Hpred_minus_odds_prob']*bets_scaling

predictions_and_bets['Areturn'] = 1-predictions_and_bets['Aproportion_bet'] + predictions_and_bets['Aproportion_bet']*predictions_and_bets['PSA']*(predictions_and_bets['FTR']=='A')
predictions_and_bets['Dreturn'] = 1-predictions_and_bets['Dproportion_bet'] + predictions_and_bets['Dproportion_bet']*predictions_and_bets['PSD']*(predictions_and_bets['FTR']=='D')
predictions_and_bets['Hreturn'] = 1-predictions_and_bets['Hproportion_bet'] + predictions_and_bets['Hproportion_bet']*predictions_and_bets['PSH']*(predictions_and_bets['FTR']=='H')

predictions_and_bets['AcumulativeReturn'] = [np.prod(predictions_and_bets['Areturn'][:i+1]) for i in range(predictions_and_bets.shape[0])]
predictions_and_bets['DcumulativeReturn'] = [np.prod(predictions_and_bets['Dreturn'][:i+1]) for i in range(predictions_and_bets.shape[0])]
predictions_and_bets['HcumulativeReturn'] = [np.prod(predictions_and_bets['Hreturn'][:i+1]) for i in range(predictions_and_bets.shape[0])]

predictions_and_bets['CombCumulativeReturn'] = np.prod(predictions_and_bets[['AcumulativeReturn','DcumulativeReturn','HcumulativeReturn']], axis=1)

plt.plot(predictions_and_bets['AcumulativeReturn'])
plt.plot(predictions_and_bets['DcumulativeReturn'])
plt.plot(predictions_and_bets['HcumulativeReturn'])
plt.plot(predictions_and_bets['CombCumulativeReturn'])



