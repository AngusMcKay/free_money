#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 10:45:54 2018

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
from sklearn.preprocessing import LabelEncoder

# read data
combined_data_added_features = pd.read_csv('all_data/combined_data_added_features.csv')

# sort out extra odds
combined_data_added_features['PSH'] = np.where(combined_data_added_features['PSH'].isnull(), combined_data_added_features['B365H'], combined_data_added_features['PSH'])
combined_data_added_features['PSA'] = np.where(combined_data_added_features['PSA'].isnull(), combined_data_added_features['B365A'], combined_data_added_features['PSA'])
combined_data_added_features['PSD'] = np.where(combined_data_added_features['PSD'].isnull(), combined_data_added_features['B365D'], combined_data_added_features['PSD'])

combined_data_added_features['BWH'] = np.where(combined_data_added_features['BWH'].isnull(), combined_data_added_features['B365H'], combined_data_added_features['BWH'])
combined_data_added_features['BWA'] = np.where(combined_data_added_features['BWA'].isnull(), combined_data_added_features['B365A'], combined_data_added_features['BWA'])
combined_data_added_features['BWD'] = np.where(combined_data_added_features['BWD'].isnull(), combined_data_added_features['B365D'], combined_data_added_features['BWD'])

combined_data_added_features['VCH'] = np.where(combined_data_added_features['VCH'].isnull(), combined_data_added_features['B365H'], combined_data_added_features['VCH'])
combined_data_added_features['VCA'] = np.where(combined_data_added_features['VCA'].isnull(), combined_data_added_features['B365A'], combined_data_added_features['VCA'])
combined_data_added_features['VCD'] = np.where(combined_data_added_features['VCD'].isnull(), combined_data_added_features['B365D'], combined_data_added_features['VCD'])

# add bookies probabilities and gains
combined_data_added_features['PSHprob'] = 1/combined_data_added_features['PSH']/(1/combined_data_added_features['PSH']+1/combined_data_added_features['PSD']+1/combined_data_added_features['PSA'])
combined_data_added_features['PSDprob'] = 1/combined_data_added_features['PSD']/(1/combined_data_added_features['PSH']+1/combined_data_added_features['PSD']+1/combined_data_added_features['PSA'])
combined_data_added_features['PSAprob'] = 1/combined_data_added_features['PSA']/(1/combined_data_added_features['PSH']+1/combined_data_added_features['PSD']+1/combined_data_added_features['PSA'])
combined_data_added_features['PSbookiesgain'] = (1/combined_data_added_features['PSH']+1/combined_data_added_features['PSD']+1/combined_data_added_features['PSA'])
np.mean(combined_data_added_features['PSbookiesgain'])
np.min(combined_data_added_features['PSbookiesgain'])
np.max(combined_data_added_features['PSbookiesgain'])

combined_data_added_features['BWHprob'] = 1/combined_data_added_features['BWH']/(1/combined_data_added_features['BWH']+1/combined_data_added_features['BWD']+1/combined_data_added_features['BWA'])
combined_data_added_features['BWDprob'] = 1/combined_data_added_features['BWD']/(1/combined_data_added_features['BWH']+1/combined_data_added_features['BWD']+1/combined_data_added_features['BWA'])
combined_data_added_features['BWAprob'] = 1/combined_data_added_features['BWA']/(1/combined_data_added_features['BWH']+1/combined_data_added_features['BWD']+1/combined_data_added_features['BWA'])
combined_data_added_features['BWbookiesgain'] = (1/combined_data_added_features['BWH']+1/combined_data_added_features['BWD']+1/combined_data_added_features['BWA'])
np.mean(combined_data_added_features['BWbookiesgain'])
np.min(combined_data_added_features['BWbookiesgain'])
np.max(combined_data_added_features['BWbookiesgain'])

combined_data_added_features['VCHprob'] = 1/combined_data_added_features['VCH']/(1/combined_data_added_features['VCH']+1/combined_data_added_features['VCD']+1/combined_data_added_features['VCA'])
combined_data_added_features['VCDprob'] = 1/combined_data_added_features['VCD']/(1/combined_data_added_features['VCH']+1/combined_data_added_features['VCD']+1/combined_data_added_features['VCA'])
combined_data_added_features['VCAprob'] = 1/combined_data_added_features['VCA']/(1/combined_data_added_features['VCH']+1/combined_data_added_features['VCD']+1/combined_data_added_features['VCA'])
combined_data_added_features['VCbookiesgain'] = (1/combined_data_added_features['VCH']+1/combined_data_added_features['VCD']+1/combined_data_added_features['VCA'])
np.mean(combined_data_added_features['VCbookiesgain'])
np.min(combined_data_added_features['VCbookiesgain'])
np.max(combined_data_added_features['VCbookiesgain'])




# put features into easily callable lists
homeWDLfeatures_W = [feature for feature in combined_data_added_features.columns.values if ((feature.startswith('home')) & (feature.endswith('_W')))]
homeWDLfeatures_D = [feature for feature in combined_data_added_features.columns.values if ((feature.startswith('home')) & (feature.endswith('_D')))]
homeWDLfeatures_L = [feature for feature in combined_data_added_features.columns.values if ((feature.startswith('home')) & (feature.endswith('_L')))]
homeGoalsForfeatures = [feature for feature in combined_data_added_features.columns.values if feature.startswith('homePRGoals')]
homeGoalsAgainstfeatures = [feature for feature in combined_data_added_features.columns.values if feature.startswith('homePRGoalsAgaints')]
homePROppositionPointsfeatures = [feature for feature in combined_data_added_features.columns.values if feature.startswith('homePROppositionPoints')]
homePRIsHomeFeatures = [feature for feature in combined_data_added_features.columns.values if feature.startswith('homePRIsHomeGame')]

awayWDLfeatures_W = [feature for feature in combined_data_added_features.columns.values if ((feature.startswith('away')) & (feature.endswith('_W')))]
awayWDLfeatures_D = [feature for feature in combined_data_added_features.columns.values if ((feature.startswith('away')) & (feature.endswith('_D')))]
awayWDLfeatures_L = [feature for feature in combined_data_added_features.columns.values if ((feature.startswith('away')) & (feature.endswith('_L')))]
awayGoalsForfeatures = [feature for feature in combined_data_added_features.columns.values if feature.startswith('awayPRGoals')]
awayGoalsAgainstfeatures = [feature for feature in combined_data_added_features.columns.values if feature.startswith('awayPRGoalsAgaints')]
awayPROppositionPointsfeatures = [feature for feature in combined_data_added_features.columns.values if feature.startswith('awayPROppositionPoints')]
awayPRIsHomeFeatures = [feature for feature in combined_data_added_features.columns.values if feature.startswith('awayPRIsHomeGame')]

divFeatures = ['N1', 'E1', 'SC3', 'SC2', 'D1', 'B1', 'I2', 'G1', 'E3', 'T1', 'EC','D2', 'F1', 'I1', 'P1', 'SP2', 'SP1', 'E2', 'SC0', 'E0', 'F2', 'SC1']


# remove rows which don't have sufficient past history
past_features_to_include = 20
past_features_to_include_alternative = 7
combined_data_added_features_with_history = combined_data_added_features[combined_data_added_features['homePRYear'+str(past_features_to_include)].notnull()]

combined_data_for_goals_preds = combined_data_added_features_with_history[combined_data_added_features_with_history['BbAv<2.5'].notnull()]

combined_data_for_goals_preds['BbAv>2.5prob'] = 1/combined_data_for_goals_preds['BbAv>2.5']/(1/combined_data_for_goals_preds['BbAv>2.5']+1/combined_data_for_goals_preds['BbAv<2.5'])
combined_data_for_goals_preds['BbAv<2.5prob'] = 1/combined_data_for_goals_preds['BbAv<2.5']/(1/combined_data_for_goals_preds['BbAv>2.5']+1/combined_data_for_goals_preds['BbAv<2.5'])
combined_data_for_goals_preds['BbAv2.5bookiesgain'] = (1/combined_data_for_goals_preds['BbAv>2.5']+1/combined_data_for_goals_preds['BbAv<2.5'])
np.mean(combined_data_for_goals_preds['BbAv2.5bookiesgain'])
np.min(combined_data_for_goals_preds['BbAv2.5bookiesgain'])
np.max(combined_data_for_goals_preds['BbAv2.5bookiesgain'])
np.std(combined_data_for_goals_preds['BbAv2.5bookiesgain'])

combined_data_for_goals_preds['totalGoals'] = combined_data_for_goals_preds['FTHG']+combined_data_for_goals_preds['FTAG']
combined_data_for_goals_preds['goals>2.5'] = (combined_data_for_goals_preds['totalGoals']>2.5)*1

bookiesFeatures = ['BbAv>2.5prob', 'BbAv<2.5prob', 'BbAv2.5bookiesgain']

# testing similar setup for WDL predictions
train_to_season = 2016
predictors = ['seasonWeek']+homeWDLfeatures_W[:past_features_to_include]+homeWDLfeatures_D[:past_features_to_include]+homeWDLfeatures_L[:past_features_to_include]+homeGoalsForfeatures[:past_features_to_include]+homeGoalsAgainstfeatures[:past_features_to_include]+homePROppositionPointsfeatures[:past_features_to_include]+awayWDLfeatures_W[:past_features_to_include]+awayWDLfeatures_D[:past_features_to_include]+awayWDLfeatures_L[:past_features_to_include]+awayGoalsForfeatures[:past_features_to_include]+awayGoalsAgainstfeatures[:past_features_to_include]+awayPROppositionPointsfeatures[:past_features_to_include]+divFeatures+homePRIsHomeFeatures+awayPRIsHomeFeatures+bookiesFeatures
#predictors_alt = ['seasonWeek']+homeWDLfeatures_W[:past_features_to_include_alternative]+homeWDLfeatures_D[:past_features_to_include_alternative]+homeWDLfeatures_L[:past_features_to_include_alternative]+homeGoalsForfeatures[:past_features_to_include_alternative]+homeGoalsAgainstfeatures[:past_features_to_include_alternative]+homePROppositionPointsfeatures[:past_features_to_include_alternative]+awayWDLfeatures_W[:past_features_to_include_alternative]+awayWDLfeatures_D[:past_features_to_include_alternative]+awayWDLfeatures_L[:past_features_to_include_alternative]+awayGoalsForfeatures[:past_features_to_include_alternative]+awayGoalsAgainstfeatures[:past_features_to_include_alternative]+awayPROppositionPointsfeatures[:past_features_to_include_alternative]+divFeatures+homePRIsHomeFeatures+awayPRIsHomeFeatures+bookiesFeatures
train_x = combined_data_for_goals_preds[predictors][combined_data_for_goals_preds['seasonEndYear']<=train_to_season].values
#train_x_alt = combined_data_for_goals_preds[predictors_alt][combined_data_for_goals_preds['seasonEndYear']<=train_to_season].values
train_col_means = train_x.mean(axis=0)
train_col_stds = train_x.astype(float).std(axis=0)*10
#subset = [predictor in predictors_alt for predictor in predictors]
train_x = (train_x - train_col_means)/(train_col_stds)
#train_x_alt = (train_x_alt - train_col_means[subset])/(train_col_stds[subset])

train_y = combined_data_for_goals_preds['goals>2.5'][combined_data_for_goals_preds['seasonEndYear']<=train_to_season].values
test_x = combined_data_for_goals_preds[predictors][combined_data_for_goals_preds['seasonEndYear']==(train_to_season+1)].values
#test_x_alt = combined_data_for_goals_preds[predictors_alt][combined_data_for_goals_preds['seasonEndYear']==(train_to_season+1)].values
test_x = (test_x - train_col_means)/train_col_stds
#test_x_alt = (test_x_alt - train_col_means[subset])/train_col_stds[subset]
test_y = combined_data_for_goals_preds['goals>2.5'][combined_data_for_goals_preds['seasonEndYear']==(train_to_season+1)].values


input_dimension = len(predictors)
#input_dimension_alt = len(predictors_alt)

# optimizers
optim_rmsprop = optimizers.RMSprop(lr=0.0001, rho=0.9)
optim_sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.5, nesterov=True)
optim_adagrad = optimizers.Adagrad(lr=0.1, decay=0)
optim_adam = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0)

number_epochs=10
batch_sizes=2**5
val_split=0.1
dropout = 0.2
weights = np.zeros(train_x.shape[0])+1#combined_data_for_goals_preds['seasonEndYear'][combined_data_for_goals_preds['seasonEndYear']<=train_to_season].values/max(combined_data_for_goals_preds['seasonEndYear'])
print('training model 1')
model1 = Sequential()
model1.add(Dropout(dropout, input_shape=(input_dimension,)))
model1.add(Dense(input_dimension, input_dim=input_dimension, activation='relu'))
model1.add(Dropout(dropout))
model1.add(Dense(input_dimension, activation='relu'))
model1.add(Dense(input_dimension, activation='relu'))
model1.add(Dense(1, activation='sigmoid'))
model1.compile(loss='binary_crossentropy', optimizer=optim_sgd, metrics=['accuracy'])
model1.fit(train_x,train_y,epochs=number_epochs,batch_size=batch_sizes,validation_split=val_split, sample_weight=weights)
print('training model 2')
model2 = Sequential()
model2.add(Dropout(dropout, input_shape=(input_dimension,)))
model2.add(Dense(input_dimension, input_dim=input_dimension, activation='relu'))
model2.add(Dropout(dropout))
model2.add(Dense(input_dimension, activation='relu'))
model2.add(Dense(input_dimension, activation='relu'))
model2.add(Dense(1, activation='sigmoid'))
model2.compile(loss='binary_crossentropy', optimizer=optim_sgd, metrics=['accuracy'])
model2.fit(train_x,train_y,epochs=number_epochs,batch_size=batch_sizes,validation_split=val_split, sample_weight=weights) 
print('training model 3')
model3 = Sequential()
model3.add(Dropout(dropout, input_shape=(input_dimension,)))
model3.add(Dense(input_dimension, input_dim=input_dimension, activation='relu'))
model3.add(Dropout(dropout))
model3.add(Dense(input_dimension, activation='relu'))
model3.add(Dense(input_dimension, activation='relu'))
model3.add(Dense(1, activation='sigmoid'))
model3.compile(loss='binary_crossentropy', optimizer=optim_sgd, metrics=['accuracy'])
model3.fit(train_x,train_y,epochs=number_epochs,batch_size=batch_sizes,validation_split=val_split, sample_weight=weights) 
print('training model 4')
model4 = Sequential()
model4.add(Dropout(dropout, input_shape=(input_dimension,)))
model4.add(Dense(input_dimension, input_dim=input_dimension, activation='relu'))
model4.add(Dropout(dropout))
model4.add(Dense(input_dimension, activation='relu'))
model4.add(Dense(input_dimension, activation='relu'))
model4.add(Dense(1, activation='sigmoid'))
model4.compile(loss='binary_crossentropy', optimizer=optim_sgd, metrics=['accuracy'])
model4.fit(train_x,train_y,epochs=number_epochs,batch_size=batch_sizes,validation_split=val_split, sample_weight=weights) 
print('training model 5')
model5 = Sequential()
model5.add(Dropout(dropout, input_shape=(input_dimension,)))
model5.add(Dense(input_dimension, input_dim=input_dimension, activation='relu'))
model5.add(Dropout(dropout))
model5.add(Dense(input_dimension, activation='relu'))
model5.add(Dense(input_dimension, activation='relu'))
model5.add(Dense(1, activation='sigmoid'))
model5.compile(loss='binary_crossentropy', optimizer=optim_sgd, metrics=['accuracy'])
model5.fit(train_x,train_y,epochs=number_epochs,batch_size=batch_sizes,validation_split=val_split, sample_weight=weights) 
print('training model 6')
model6 = Sequential()
model6.add(Dropout(dropout, input_shape=(input_dimension,)))
model6.add(Dense(input_dimension, input_dim=input_dimension, activation='relu'))
model6.add(Dropout(dropout))
model6.add(Dense(input_dimension, activation='relu'))
model6.add(Dense(input_dimension, activation='relu'))
model6.add(Dense(1, activation='sigmoid'))
model6.compile(loss='binary_crossentropy', optimizer=optim_sgd, metrics=['accuracy'])
model6.fit(train_x,train_y,epochs=number_epochs,batch_size=batch_sizes,validation_split=val_split, sample_weight=weights) 
print('training model 7')
model7 = Sequential()
model7.add(Dropout(dropout, input_shape=(input_dimension,)))
model7.add(Dense(input_dimension, input_dim=input_dimension, activation='relu'))
model7.add(Dropout(dropout))
model7.add(Dense(input_dimension, activation='relu'))
model7.add(Dense(input_dimension, activation='relu'))
model7.add(Dense(1, activation='sigmoid'))
model7.compile(loss='binary_crossentropy', optimizer=optim_sgd, metrics=['accuracy'])
model7.fit(train_x,train_y,epochs=number_epochs,batch_size=batch_sizes,validation_split=val_split, sample_weight=weights) 
print('training model 8')
model8 = Sequential()
model8.add(Dropout(dropout, input_shape=(input_dimension,)))
model8.add(Dense(input_dimension, input_dim=input_dimension, activation='relu'))
model8.add(Dropout(dropout))
model8.add(Dense(input_dimension, activation='relu'))
model8.add(Dense(input_dimension, activation='relu'))
model8.add(Dense(1, activation='sigmoid'))
model8.compile(loss='binary_crossentropy', optimizer=optim_sgd, metrics=['accuracy'])
model8.fit(train_x,train_y,epochs=number_epochs,batch_size=batch_sizes,validation_split=val_split, sample_weight=weights) 
print('training model 9')
model9 = Sequential()
model9.add(Dropout(dropout, input_shape=(input_dimension,)))
model9.add(Dense(input_dimension, input_dim=input_dimension, activation='relu'))
model9.add(Dropout(dropout))
model9.add(Dense(input_dimension, activation='relu'))
model9.add(Dense(input_dimension, activation='relu'))
model9.add(Dense(1, activation='sigmoid'))
model9.compile(loss='binary_crossentropy', optimizer=optim_sgd, metrics=['accuracy'])
model9.fit(train_x,train_y,epochs=number_epochs,batch_size=batch_sizes,validation_split=val_split, sample_weight=weights) 
print('training model 10')
model10 = Sequential()
model10.add(Dropout(dropout, input_shape=(input_dimension,)))
model10.add(Dense(input_dimension, input_dim=input_dimension, activation='relu'))
model10.add(Dropout(dropout))
model10.add(Dense(input_dimension, activation='relu'))
model10.add(Dense(input_dimension, activation='relu'))
model10.add(Dense(1, activation='sigmoid'))
model10.compile(loss='binary_crossentropy', optimizer=optim_sgd, metrics=['accuracy'])
model10.fit(train_x,train_y,epochs=number_epochs,batch_size=batch_sizes,validation_split=val_split, sample_weight=weights) 


dual_model_bets = combined_data_for_goals_preds[['Date','BbAv>2.5','BbAv<2.5', 'BbAv>2.5prob', 'BbAv<2.5prob', 'BbAv2.5bookiesgain','totalGoals', 'goals>2.5']][combined_data_for_goals_preds['seasonEndYear']==(train_to_season+1)]
models_list = [model1, model2, model3, model4, model5, model6, model7, model8, model9, model10]

for i in tqdm(range(len(models_list))):
    model = models_list[i]
    dual_model_bets['preds'+str(i+1)] = model.predict(test_x)

probability_cushion=0.05

dual_model_bets['preds_avg'] = 0

models_to_average=len(models_list)
for i in range(models_to_average):
    dual_model_bets['preds_avg'] = dual_model_bets['preds_avg'] + dual_model_bets['preds'+str(i+1)]/models_to_average

    dual_model_bets['bets'+str(i+1)] = (dual_model_bets['preds'+str(i+1)]>(1/dual_model_bets['BbAv>2.5']+probability_cushion)) #& (nn_multi_combined_outcomes['Apreds']>(nn_multi_combined_outcomes['B365Aprob']+probability_cushion))
    dual_model_bets['bets_less'+str(i+1)] = ((1-dual_model_bets['preds'+str(i+1)])>(1/dual_model_bets['BbAv<2.5']+probability_cushion)) #& (nn_multi_combined_outcomes['Apreds']>(nn_multi_combined_outcomes['B365Aprob']+probability_cushion))


dual_model_bets['bets_avg'] = dual_model_bets['preds_avg']>(1/dual_model_bets['BbAv>2.5']+probability_cushion)
dual_model_bets['bets_less_avg'] = (1-dual_model_bets['preds_avg'])>(1/dual_model_bets['BbAv<2.5']+probability_cushion)

dual_model_bets['bets_combined'] = 0
dual_model_bets['bets_less_combined'] = 0

for i in range(len(models_list)):
    dual_model_bets['bets_combined'] = dual_model_bets['bets_combined'] + dual_model_bets['bets'+str(i+1)]
    dual_model_bets['bets_less_combined'] = dual_model_bets['bets_less_combined'] + dual_model_bets['bets_less'+str(i+1)]

    dual_model_bets['winnings'+str(i+1)] = dual_model_bets['bets'+str(i+1)]*(dual_model_bets[['BbAv>2.5']].max(axis=1))*(dual_model_bets['goals>2.5']==1)
    dual_model_bets['winnings_less'+str(i+1)] = dual_model_bets['bets_less'+str(i+1)]*(dual_model_bets[['BbAv<2.5']].max(axis=1))*(dual_model_bets['goals>2.5']==0)

vote_cutoff=5
dual_model_bets['bets_majority'] = dual_model_bets['bets_combined']>=vote_cutoff
dual_model_bets['bets_less_majority'] = dual_model_bets['bets_less_combined']>=vote_cutoff

dual_model_bets['winnings_c'] = dual_model_bets['bets_combined']*(dual_model_bets[['BbAv>2.5']].max(axis=1))*(dual_model_bets['goals>2.5']==1)
dual_model_bets['winnings_less_c'] = dual_model_bets['bets_less_combined']*(dual_model_bets[['BbAv<2.5']].max(axis=1))*(dual_model_bets['goals>2.5']==0)

dual_model_bets['winnings_a'] = dual_model_bets['bets_avg']*(dual_model_bets[['BbAv>2.5']].max(axis=1))*(dual_model_bets['goals>2.5']==1)
dual_model_bets['winnings_less_a'] = dual_model_bets['bets_less_avg']*(dual_model_bets[['BbAv<2.5']].max(axis=1))*(dual_model_bets['goals>2.5']==0)

dual_model_bets['winnings_m'] = dual_model_bets['bets_majority']*(dual_model_bets[['BbAv>2.5']].max(axis=1))*(dual_model_bets['goals>2.5']==1)
dual_model_bets['winnings_less_m'] = dual_model_bets['bets_less_majority']*(dual_model_bets[['BbAv<2.5']].max(axis=1))*(dual_model_bets['goals>2.5']==0)

for i in range(len(models_list)):
    print('model'+str(i+1)+' bets: ', sum(dual_model_bets['bets'+str(i+1)]), 'model'+str(i+1)+' winnings: ', sum(dual_model_bets['winnings'+str(i+1)]), 'return: ', ((sum(dual_model_bets['winnings'+str(i+1)]))/(sum(dual_model_bets['bets'+str(i+1)]))-1)*100, '%')

for i in range(len(models_list)):
    print('model'+str(i+1)+' bets less: ', sum(dual_model_bets['bets_less'+str(i+1)]), 'model'+str(i+1)+' winnings: ', sum(dual_model_bets['winnings_less'+str(i+1)]), 'return: ', ((sum(dual_model_bets['winnings_less'+str(i+1)]))/(sum(dual_model_bets['bets_less'+str(i+1)]))-1)*100, '%')

print('combined bets: ', sum(dual_model_bets['bets_combined']), 'combined winnings: ', sum(dual_model_bets['winnings_c']), 'return: ', ((sum(dual_model_bets['winnings_c']))/(sum(dual_model_bets['bets_combined']))-1)*100, '%')
print('averaged bets: ', sum(dual_model_bets['bets_avg']), 'averaged winnings: ', sum(dual_model_bets['winnings_a']), 'return: ', ((sum(dual_model_bets['winnings_a']))/(sum(dual_model_bets['bets_avg']))-1)*100, '%')
print('majority bets: ', sum(dual_model_bets['bets_majority']), 'majority winnings: ', sum(dual_model_bets['winnings_m']), 'return: ', ((sum(dual_model_bets['winnings_m']))/(sum(dual_model_bets['bets_majority']))-1)*100, '%')

print('combined bets less: ', sum(dual_model_bets['bets_less_combined']), 'combined winnings: ', sum(dual_model_bets['winnings_less_c']), 'return: ', ((sum(dual_model_bets['winnings_less_c']))/(sum(dual_model_bets['bets_less_combined']))-1)*100, '%')
print('averaged bets less: ', sum(dual_model_bets['bets_less_avg']), 'averaged winnings: ', sum(dual_model_bets['winnings_less_a']), 'return: ', ((sum(dual_model_bets['winnings_less_a']))/(sum(dual_model_bets['bets_less_avg']))-1)*100, '%')
print('majority bets less: ', sum(dual_model_bets['bets_less_majority']), 'majority winnings: ', sum(dual_model_bets['winnings_less_m']), 'return: ', ((sum(dual_model_bets['winnings_less_m']))/(sum(dual_model_bets['bets_less_majority']))-1)*100, '%')



### NOT EDITED BELOW HERE
###----------------------------------------------------------------------------------------------------------------





# plot to get an idea of volatility
dual_model_bets['ProfitLoss'] = dual_model_bets[['Awinnings_m', 'Dwinnings_m', 'Hwinnings_m']].sum(axis=1) - dual_model_bets[['Abets_majority', 'Dbets_majority', 'Hbets_majority']].sum(axis=1)
dual_model_bets['cumulativeProfitLoss'] = [sum(dual_model_bets['ProfitLoss'][:(row+1)]) for row in range(dual_model_bets.shape[0])]
dual_model_bets['bets_majority'] = dual_model_bets[['Abets_majority', 'Dbets_majority', 'Hbets_majority']].sum(axis=1)
dual_model_bets['cumulativeBetsMade'] = [sum(dual_model_bets['bets_majority'][:(row+1)]) for row in range(dual_model_bets.shape[0])]

plt.plot(dual_model_bets['cumulativeProfitLoss']*1000)

# plot distribution of % returns from each 100 bets
bootstrap_samples = 10000
sample_returns = []
for i in tqdm(range(bootstrap_samples)):
    chosen_bets = np.random.choice(dual_model_bets.index.values[dual_model_bets['bets_majority']>0], 100, replace=False)
    chosen_bets_return = sum(dual_model_bets['ProfitLoss'][chosen_bets])/sum(dual_model_bets['bets_majority'][chosen_bets])*100
    sample_returns.append(chosen_bets_return)

min(sample_returns)
plt.hist(sample_returns, bins=50)

returns_from_each_100_bets = [1]
for i in range(int(sum(dual_model_bets['bets_majority'])/100)):
    returns_from_each_100_bets.append(sum((dual_model_bets[['Awinnings_m', 'Dwinnings_m', 'Hwinnings_m']].sum(axis=1))[(dual_model_bets['cumulativeBetsMade'] > i*100) & (dual_model_bets['cumulativeBetsMade'] <= (i+1)*100)])/sum(dual_model_bets['bets_majority'][(dual_model_bets['cumulativeBetsMade'] > i*100) & (dual_model_bets['cumulativeBetsMade'] <= (i+1)*100)]))

plt.plot([np.prod(returns_from_each_100_bets[:i+1]) for i in range(len(returns_from_each_100_bets))])







# Q2b: when a model generalises well on one season, does this mean it then performs better on the following season?
test_x_plus_1 = combined_data_for_goals_preds[predictors][combined_data_for_goals_preds['seasonEndYear']==(train_to_season+2)].values
test_x_plus_1 = (test_x_plus_1 - train_col_means)/train_col_stds

dual_model_bets_following_year = combined_data_for_goals_preds[['Date', 'B365A','B365D','B365H','LBA','LBD','LBH','B365Aprob','B365Dprob','B365Hprob','FTR']][combined_data_for_goals_preds['seasonEndYear']==(train_to_season+2)]
models_list = [model1, model2, model3, model4, model5, model6, model7, model8, model9, model10]
probability_cushion=0.05
for i in range(len(models_list)):
    model = models_list[i]
    dual_model_bets_following_year['Apreds'+str(i+1)] = model.predict(test_x_plus_1)[:,0]
    dual_model_bets_following_year['Dpreds'+str(i+1)] = model.predict(test_x_plus_1)[:,1]
    dual_model_bets_following_year['Hpreds'+str(i+1)] = model.predict(test_x_plus_1)[:,2]
    
    dual_model_bets_following_year['Abets'+str(i+1)] = (dual_model_bets_following_year['Apreds'+str(i+1)]>(1/dual_model_bets_following_year['B365A']+probability_cushion)) #& (nn_multi_combined_outcomes['Apreds']>(nn_multi_combined_outcomes['B365Aprob']+probability_cushion))
    dual_model_bets_following_year['Dbets'+str(i+1)] = (dual_model_bets_following_year['Dpreds'+str(i+1)]>(1/dual_model_bets_following_year['B365D']+probability_cushion)) #& (nn_multi_combined_outcomes['Dpreds']>(nn_multi_combined_outcomes['B365Dprob']+probability_cushion))
    dual_model_bets_following_year['Hbets'+str(i+1)] = (dual_model_bets_following_year['Hpreds'+str(i+1)]>(1/dual_model_bets_following_year['B365H']+probability_cushion)) #& (nn_multi_combined_outcomes['Hpreds']>(nn_multi_combined_outcomes['B365Hprob']+probability_cushion))

dual_model_bets_following_year['Abets_combined'] = 0
dual_model_bets_following_year['Dbets_combined'] = 0
dual_model_bets_following_year['Hbets_combined'] = 0

for i in range(len(models_list)):
    dual_model_bets_following_year['Abets_combined'] = dual_model_bets_following_year['Abets_combined'] + dual_model_bets_following_year['Abets'+str(i+1)]
    dual_model_bets_following_year['Dbets_combined'] = dual_model_bets_following_year['Dbets_combined'] + dual_model_bets_following_year['Dbets'+str(i+1)]
    dual_model_bets_following_year['Hbets_combined'] = dual_model_bets_following_year['Hbets_combined'] + dual_model_bets_following_year['Hbets'+str(i+1)]
    
    dual_model_bets_following_year['Awinnings'+str(i+1)] = dual_model_bets_following_year['Abets'+str(i+1)]*(dual_model_bets_following_year[['B365A','LBA']].max(axis=1))*(dual_model_bets_following_year['FTR']=='A')
    dual_model_bets_following_year['Dwinnings'+str(i+1)] = dual_model_bets_following_year['Dbets'+str(i+1)]*(dual_model_bets_following_year[['B365D','LBD']].max(axis=1))*(dual_model_bets_following_year['FTR']=='D')
    dual_model_bets_following_year['Hwinnings'+str(i+1)] = dual_model_bets_following_year['Hbets'+str(i+1)]*(dual_model_bets_following_year[['B365H','LBH']].max(axis=1))*(dual_model_bets_following_year['FTR']=='H')

vote_cutoff=5
dual_model_bets_following_year['Abets_majority'] = dual_model_bets_following_year['Abets_combined']>=vote_cutoff
dual_model_bets_following_year['Dbets_majority'] = dual_model_bets_following_year['Dbets_combined']>=vote_cutoff
dual_model_bets_following_year['Hbets_majority'] = dual_model_bets_following_year['Hbets_combined']>=vote_cutoff

dual_model_bets_following_year['Awinnings_c'] = dual_model_bets_following_year['Abets_combined']*(dual_model_bets_following_year[['B365A','LBA']].max(axis=1))*(dual_model_bets_following_year['FTR']=='A')
dual_model_bets_following_year['Dwinnings_c'] = dual_model_bets_following_year['Dbets_combined']*(dual_model_bets_following_year[['B365D','LBD']].max(axis=1))*(dual_model_bets_following_year['FTR']=='D')
dual_model_bets_following_year['Hwinnings_c'] = dual_model_bets_following_year['Hbets_combined']*(dual_model_bets_following_year[['B365H','LBH']].max(axis=1))*(dual_model_bets_following_year['FTR']=='H')

dual_model_bets_following_year['Awinnings_m'] = dual_model_bets_following_year['Abets_majority']*(dual_model_bets_following_year[['B365A','LBA']].max(axis=1))*(dual_model_bets_following_year['FTR']=='A')
dual_model_bets_following_year['Dwinnings_m'] = dual_model_bets_following_year['Dbets_majority']*(dual_model_bets_following_year[['B365D','LBD']].max(axis=1))*(dual_model_bets_following_year['FTR']=='D')
dual_model_bets_following_year['Hwinnings_m'] = dual_model_bets_following_year['Hbets_majority']*(dual_model_bets_following_year[['B365H','LBH']].max(axis=1))*(dual_model_bets_following_year['FTR']=='H')

for i in range(len(models_list)):
    print('model'+str(i+1)+' bets: ', sum(dual_model_bets_following_year['Hbets'+str(i+1)])+sum(dual_model_bets_following_year['Dbets'+str(i+1)])+sum(dual_model_bets_following_year['Abets'+str(i+1)]), 'model'+str(i+1)+' winnings: ', sum(dual_model_bets_following_year['Hwinnings'+str(i+1)])+sum(dual_model_bets_following_year['Dwinnings'+str(i+1)])+sum(dual_model_bets_following_year['Awinnings'+str(i+1)]), 'return: ', ((sum(dual_model_bets_following_year['Hwinnings'+str(i+1)])+sum(dual_model_bets_following_year['Dwinnings'+str(i+1)])+sum(dual_model_bets_following_year['Awinnings'+str(i+1)]))/(sum(dual_model_bets_following_year['Hbets'+str(i+1)])+sum(dual_model_bets_following_year['Dbets'+str(i+1)])+sum(dual_model_bets_following_year['Abets'+str(i+1)]))-1)*100, '%')

print('combined bets: ', sum(dual_model_bets_following_year['Hbets_combined'])+sum(dual_model_bets_following_year['Dbets_combined'])+sum(dual_model_bets_following_year['Abets_combined']), 'combined winnings: ', sum(dual_model_bets_following_year['Hwinnings_c'])+sum(dual_model_bets_following_year['Dwinnings_c'])+sum(dual_model_bets_following_year['Awinnings_c']), 'return: ', ((sum(dual_model_bets_following_year['Hwinnings_c'])+sum(dual_model_bets_following_year['Dwinnings_c'])+sum(dual_model_bets_following_year['Awinnings_c']))/(sum(dual_model_bets_following_year['Hbets_combined'])+sum(dual_model_bets_following_year['Dbets_combined'])+sum(dual_model_bets_following_year['Abets_combined']))-1)*100, '%')
print('majority bets: ', sum(dual_model_bets_following_year['Hbets_majority'])+sum(dual_model_bets_following_year['Dbets_majority'])+sum(dual_model_bets_following_year['Abets_majority']), 'majority winnings: ', sum(dual_model_bets_following_year['Hwinnings_m'])+sum(dual_model_bets_following_year['Dwinnings_m'])+sum(dual_model_bets_following_year['Awinnings_m']), 'return: ', ((sum(dual_model_bets_following_year['Hwinnings_m'])+sum(dual_model_bets_following_year['Dwinnings_m'])+sum(dual_model_bets_following_year['Awinnings_m']))/(sum(dual_model_bets_following_year['Hbets_majority'])+sum(dual_model_bets_following_year['Dbets_majority'])+sum(dual_model_bets_following_year['Abets_majority']))-1)*100, '%')
print('home bets: ', sum(dual_model_bets_following_year['Hbets_majority']), 'home winnings: ', sum(dual_model_bets_following_year['Hwinnings_m']), 'return: ', (sum(dual_model_bets_following_year['Hwinnings_m'])/sum(dual_model_bets_following_year['Hbets_majority'])-1)*100, '%')
print('draw bets: ', sum(dual_model_bets_following_year['Dbets_majority']), 'draw winnings: ', sum(dual_model_bets_following_year['Dwinnings_m']), 'return: ', (sum(dual_model_bets_following_year['Dwinnings_m'])/sum(dual_model_bets_following_year['Dbets_majority'])-1)*100, '%')
print('away bets: ', sum(dual_model_bets_following_year['Abets_majority']), 'away winnings: ', sum(dual_model_bets_following_year['Awinnings_m']), 'return: ', (sum(dual_model_bets_following_year['Awinnings_m'])/sum(dual_model_bets_following_year['Abets_majority'])-1)*100, '%')

# seems like they usually get worse when predicting on next year's data


# Q2c: if performance on 2b is worse, then is it worth removing the validation set?

# First sign based on training to 2016 is that it does help
# Might even be worth training every month or so



# Q2d: is it better to have lots of the same model, or slightly different models?

# best to have lots of slightly different models because of mixed performance

# NEED TO TEST THIS MORE!!!



# Q2e: do models which work well on in sample also work well on the test sample
years_ago=0 # 0 for last year in training sample, 1 for second last etc
for i in range(len(models_list)):
    model=models_list[i]
    in_home_bets = model.predict(combined_data_for_goals_preds[predictors][combined_data_for_goals_preds['seasonEndYear']==(train_to_season-years_ago)].values)[:,2]>(1/(combined_data_for_goals_preds['B365H'][combined_data_for_goals_preds['seasonEndYear']==(train_to_season-years_ago)])+probability_cushion)
    in_draw_bets = model.predict(combined_data_for_goals_preds[predictors][combined_data_for_goals_preds['seasonEndYear']==(train_to_season-years_ago)].values)[:,1]>(1/(combined_data_for_goals_preds['B365D'][combined_data_for_goals_preds['seasonEndYear']==(train_to_season-years_ago)])+probability_cushion)
    in_away_bets = model.predict(combined_data_for_goals_preds[predictors][combined_data_for_goals_preds['seasonEndYear']==(train_to_season-years_ago)].values)[:,0]>(1/(combined_data_for_goals_preds['B365A'][combined_data_for_goals_preds['seasonEndYear']==(train_to_season-years_ago)])+probability_cushion)
    in_home_winnings = in_home_bets*((combined_data_for_goals_preds[['B365H','LBH']][combined_data_for_goals_preds['seasonEndYear']==(train_to_season-years_ago)]).max(axis=1))*((combined_data_for_goals_preds['FTR']=='H')[combined_data_for_goals_preds['seasonEndYear']==(train_to_season-years_ago)])
    in_draw_winnings = in_draw_bets*((combined_data_for_goals_preds[['B365D','LBD']][combined_data_for_goals_preds['seasonEndYear']==(train_to_season-years_ago)]).max(axis=1))*((combined_data_for_goals_preds['FTR']=='D')[combined_data_for_goals_preds['seasonEndYear']==(train_to_season-years_ago)])
    in_away_winnings = in_away_bets*((combined_data_for_goals_preds[['B365A','LBA']][combined_data_for_goals_preds['seasonEndYear']==(train_to_season-years_ago)]).max(axis=1))*((combined_data_for_goals_preds['FTR']=='A')[combined_data_for_goals_preds['seasonEndYear']==(train_to_season-years_ago)])
    print('Total bet (in): ', sum(in_home_bets)+sum(in_draw_bets)+sum(in_away_bets), 'Winnings: ', sum(in_home_winnings)+sum(in_draw_winnings)+sum(in_away_winnings), 'Profit: ', round(((sum(in_home_winnings)+sum(in_draw_winnings)+sum(in_away_winnings))/(sum(in_home_bets)+sum(in_draw_bets)+sum(in_away_bets))-1)*100,2), '%')


# Q2f: train on data to 2013 and then view how performance worsens over time after that

# gets much worse over time - strongly suggests that regular retraining is beneficial



# Q2g: does a simple model, e.g. 1 or 2 hidden layers, work?



# Q2h: does combining in different ways help
# e.g. averaging will take account of any highly negative predictions


# Q2i: does it work to take, say, top 10% of possible bets?



# Q3: how does XGBoost perform with the additional features?
train_y_labels = combined_data_for_goals_preds['FTR'][combined_data_for_goals_preds['seasonEndYear']<=train_to_season]
test_y_labels = combined_data_for_goals_preds['FTR'][combined_data_for_goals_preds['seasonEndYear']==(train_to_season+1)]
label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(train_y_labels)
xgb_y_train = label_encoder.transform(train_y_labels)
label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(test_y_labels)
xgb_y_test = label_encoder.transform(test_y_labels)


xg_train = xgb.DMatrix(train_x, label=xgb_y_train)
xg_test = xgb.DMatrix(test_x, label=xgb_y_test)

param = {}
# use softmax multi-class classification
param['objective'] = 'multi:softprob'
# scale weight of positive examples
param['eta'] = 0.1
param['max_depth'] = 6
param['silent'] = 1
param['nthread'] = 4
param['num_class'] = 3
param['n_estimators'] = 1000

watchlist = [(xg_train, 'train'), (xg_test, 'test')]
num_round = 10
bst = xgb.train(param, xg_train, num_round, watchlist)

pred_prob = bst.predict(xg_test).reshape(test_y.shape[0], 3)

dual_model_bets['Apreds_xgb1'] = pred_prob[:,0]
dual_model_bets['Dpreds_xgb1'] = pred_prob[:,1]
dual_model_bets['Hpreds_xgb1'] = pred_prob[:,2]

probability_cushion=0.05
dual_model_bets['Abets_xgb1'] = (dual_model_bets['Apreds_xgb1']>(1/dual_model_bets['B365A']+probability_cushion)) #& (nn_multi_combined_outcomes['Apreds']>(nn_multi_combined_outcomes['B365Aprob']+probability_cushion))
dual_model_bets['Dbets_xgb1'] = (dual_model_bets['Dpreds_xgb1']>(1/dual_model_bets['B365D']+probability_cushion)) #& (nn_multi_combined_outcomes['Dpreds']>(nn_multi_combined_outcomes['B365Dprob']+probability_cushion))
dual_model_bets['Hbets_xgb1'] = (dual_model_bets['Hpreds_xgb1']>(1/dual_model_bets['B365H']+probability_cushion)) #& (nn_multi_combined_outcomes['Hpreds']>(nn_multi_combined_outcomes['B365Hprob']+probability_cushion))

dual_model_bets['Awinnings_xgb1'] = dual_model_bets['Abets_xgb1']*(dual_model_bets[['B365A','LBA']].max(axis=1))*(dual_model_bets['FTR']=='A')
dual_model_bets['Dwinnings_xgb1'] = dual_model_bets['Dbets_xgb1']*(dual_model_bets[['B365D','LBD']].max(axis=1))*(dual_model_bets['FTR']=='D')
dual_model_bets['Hwinnings_xgb1'] = dual_model_bets['Hbets_xgb1']*(dual_model_bets[['B365H','LBH']].max(axis=1))*(dual_model_bets['FTR']=='H')

print('home bets: ', sum(dual_model_bets['Hbets_xgb1']), 'home winnings: ', sum(dual_model_bets['Hwinnings_xgb1']), 'return: ', (sum(dual_model_bets['Hwinnings_xgb1'])/sum(dual_model_bets['Hbets_xgb1'])-1)*100, '%')
print('draw bets: ', sum(dual_model_bets['Dbets_xgb1']), 'draw winnings: ', sum(dual_model_bets['Dwinnings_xgb1']), 'return: ', (sum(dual_model_bets['Dwinnings_xgb1'])/sum(dual_model_bets['Dbets_xgb1'])-1)*100, '%')
print('away bets: ', sum(dual_model_bets['Abets_xgb1']), 'away winnings: ', sum(dual_model_bets['Awinnings_xgb1']), 'return: ', (sum(dual_model_bets['Awinnings_xgb1'])/sum(dual_model_bets['Abets_xgb1'])-1)*100, '%')




# Q4: should newer data be weighted higher? (or maybe use seasonEndYear feature?)

# initial test don't seem to help much, but worth investigating further


# Q5: How does changing the number of neurons change performance?


# Q6: How does changing the loss function change performance?
# absolute_loss

# hinge_loss



