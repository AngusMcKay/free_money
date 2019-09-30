#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 27 16:01:12 2018

@author: angus
"""

import os
os.chdir('/home/angus/projects/betting/tennis')

import numpy as np
import pandas as pd
from tqdm import tqdm
import xgboost as xgb
import sklearn as skl
from sklearn import linear_model
import datetime
import glob
import matplotlib.pyplot as plt
from keras.models import Sequential, model_from_json
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import optimizers
from keras import initializers
from keras import models
from keras import regularizers
from sklearn.preprocessing import LabelEncoder
from scipy.stats import norm





mens_data_sql_processed_with_history = pd.read_csv('all_data/mens_data_sql_processed_with_history.csv')

### create lists of features
num_past_results=10
# base features
base_features = ['best_of','clay','p1Rank','p2Rank']
bookies_features = ['b365P1prob','b365P2prob','b365bookiesgain']

# lists for basic past results features
#past_base_features = [] # THESE WILL BE SAME AS THE CURRENT GAME, SO IGNORE
past_bookies_features = []
for i in range(num_past_results):
    #past_base_features = past_base_features + ['p1'+str(i+1)+'Best_of', 'p1'+str(i+1)+'Clay','p2'+str(i+1)+'Best_of', 'p2'+str(i+1)+'Clay']
    past_bookies_features = past_bookies_features + ['p1'+str(i+1)+'Playerprob', 'p1'+str(i+1)+'Opponentprob',
                                                     'p2'+str(i+1)+'Playerprob', 'p2'+str(i+1)+'Opponentprob']

# create combined past wins crossed with opponent rank feature
pastWinRank_features = []
for i in range(num_past_results):
    pastWinRank_features = pastWinRank_features + ['p1'+str(i+1)+'WinRank', 'p2'+str(i+1)+'WinRank']

# create past set and game percentage won
pastSets_features = []
pastGames_features = []
for i in range(num_past_results):
    pastSets_features = pastSets_features + ['p1'+str(i+1)+'SetsProp', 'p2'+str(i+1)+'SetsProp']
    pastGames_features = pastSets_features + ['p1'+str(i+1)+'GamesProp', 'p2'+str(i+1)+'GamesProp']



# create several models trained on data with a range of weights to encourage higher variance where less confidence in bets
features = base_features+pastWinRank_features+pastSets_features+pastGames_features#+bookies_features+past_bookies_features

train_to_date = '2017-01-01'

train_x = mens_data_sql_processed_with_history[features][mens_data_sql_processed_with_history['date']<train_to_date].values
train_x = train_x.astype(float)
train_y = mens_data_sql_processed_with_history['player1Wins'][mens_data_sql_processed_with_history['date']<train_to_date].values
train_col_means = train_x.mean(axis=0)
train_col_stds = train_x.astype(float).std(axis=0)
train_col_stds = train_col_stds + (train_col_stds==0)*1 # adds 1 where the stdev is 0 to not break division
train_x = (train_x - train_col_means)/train_col_stds

test_x = mens_data_sql_processed_with_history[features][mens_data_sql_processed_with_history['date']>=train_to_date].values
test_x = test_x.astype(float)
test_y = mens_data_sql_processed_with_history['player1Wins'][mens_data_sql_processed_with_history['date']>=train_to_date].values
test_x = (test_x - train_col_means)/train_col_stds



# optimizers
optim_rmsprop = optimizers.RMSprop(lr=0.0001, rho=0.9)
optim_sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.5, nesterov=True)
optim_adagrad = optimizers.Adagrad(lr=0.01, decay=0)
optim_adam = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0)



# model
number_epochs=20
batch_sizes=2**7
val_split=0.1
dropout = 0.0
weights = np.zeros(train_x.shape[0])+1
weights_alt = np.zeros(train_x.shape[0])+1
weights_deviance_from_1 = 0.1
weights = (1 - weights_deviance_from_1) + np.asarray(mens_data_sql_processed_with_history['player1Wins'][mens_data_sql_processed_with_history['date']<train_to_date]*weights_deviance_from_1*2)
weights_alt = (1 + weights_deviance_from_1) - np.asarray(mens_data_sql_processed_with_history['player1Wins'][mens_data_sql_processed_with_history['date']<train_to_date]*weights_deviance_from_1*2)
#weights = np.asarray(mens_data_sql_processed_with_history['b365P1'][mens_data_sql_processed_with_history['date']<train_to_date])
#weights = np.asarray(mens_data_sql_processed_with_history['b365P2'][mens_data_sql_processed_with_history['date']<train_to_date])


input_dimension = len(features)

model1 = Sequential()
model1.add(Dense(input_dimension, input_dim=input_dimension, activation='relu',
           kernel_regularizer=regularizers.l1(0.0005),
           #activity_regularizer=regularizers.l1(0.01)
           ))
model1.add(Dense(input_dimension, activation='relu',
           kernel_regularizer=regularizers.l1(0.0005),
           #activity_regularizer=regularizers.l1(0.01)
           ))
model1.add(Dense(input_dimension, activation='relu',
           kernel_regularizer=regularizers.l1(0.0005),
           #activity_regularizer=regularizers.l1(0.01)
           ))
#model1.add(Dense(input_dimension, activation='relu',
#           #kernel_regularizer=regularizers.l1(0.0005),
#           #activity_regularizer=regularizers.l1(0.01)
#           ))
model1.add(Dense(1, activation='sigmoid'))
model1.compile(loss='binary_crossentropy', optimizer=optim_adagrad, metrics=['accuracy'])
model1.fit(train_x,train_y,epochs=number_epochs,batch_size=batch_sizes,validation_split=val_split, sample_weight=weights)

## save weights to HDF5
model1.save_weights("models/saved_model_weights.h5", overwrite=True)


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
sample_proportion = 0.8
samples = []
for i in range(number_models-1):
    samples.append(list(np.random.choice(train_x.shape[0], int(train_x.shape[0]*sample_proportion))))

weights_list = [weights_alt*(i+1)/(number_models-1) + weights*((number_models-1)-(i+1))/(number_models-1) for i in range(number_models-1)]
for i in range(len(model_list)):
    print('fitting model ', i+2)
    model_list[i].load_weights("models/saved_model_weights.h5")
    model_list[i].compile(loss='binary_crossentropy', optimizer=optim_adagrad, metrics=['accuracy'])
    model_list[i].fit(train_x[samples[i],:],train_y[samples[i]],epochs=number_epochs_refit,batch_size=batch_sizes,validation_split=val_split, sample_weight=weights_list[i][samples[i]])



# test predictions
predictions_and_bets = mens_data_sql_processed_with_history[['date', 'b365P1','b365P2','player1Wins']][mens_data_sql_processed_with_history['date']>=train_to_date]
full_model_list = [model1]+model_list

for i in tqdm(range(len(full_model_list))):
    model = full_model_list[i]
    predictions_and_bets['preds'+str(i+1)] = model.predict(test_x)

predsList = ['preds'+str(i+1) for i in range(len(full_model_list))]


# calculate mean and st dev of predictions
predictions_and_bets['preds_mean'] = np.mean(predictions_and_bets[predsList], axis=1)
predictions_and_bets['preds_sd'] = np.std(predictions_and_bets[predsList], axis=1)


# calculate lower probability for a given confidence
confidence = 0.97
predictions_and_bets['lower_confidence'] = norm.ppf(1-confidence, predictions_and_bets['preds_mean'], predictions_and_bets['preds_sd'])
predictions_and_bets['upper_confidence'] = norm.ppf(confidence, predictions_and_bets['preds_mean'], predictions_and_bets['preds_sd'])


# assign bets based on the lower confidence probability being above a threshold higher than the bookies' probability
probability_margin = 0.3
predictions_and_bets['betsP1'] = (predictions_and_bets['lower_confidence']>(1/predictions_and_bets['b365P1']+probability_margin))*1
predictions_and_bets['betsP2'] = ((1-predictions_and_bets['upper_confidence'])>(1/predictions_and_bets['b365P2']+probability_margin))*1


# calculate winnings based on lower confidence probability bets
predictions_and_bets['winningsP1'] = predictions_and_bets['betsP2']*predictions_and_bets['b365P1']*(predictions_and_bets['player1Wins']==1)
predictions_and_bets['winningsP2'] = predictions_and_bets['betsP1']*predictions_and_bets['b365P2']*(predictions_and_bets['player1Wins']==0)


print('total bets: ',sum(predictions_and_bets['betsP1'])+sum(predictions_and_bets['betsP2']),
      'total winnings: ',sum(predictions_and_bets['winningsP1'])+sum(predictions_and_bets['winningsP2']),
      'return: ', ((sum(predictions_and_bets['winningsP1'])+sum(predictions_and_bets['winningsP2']))/(sum(predictions_and_bets['betsP1'])+sum(predictions_and_bets['betsP2']))-1)*100, '%')
print('P1 bets: ',sum(predictions_and_bets['betsP1']), 'P1 winnings: ',sum(predictions_and_bets['winningsP1']))
print('P2 bets: ',sum(predictions_and_bets['betsP2']), 'P2 winnings: ',sum(predictions_and_bets['winningsP2']))

predictions_and_bets[(predictions_and_bets['betsP1'] == 1)]






### OLD CODE

# test XGBoost
dtrain = xgb.DMatrix(train_x, label=train_y, weight=weights)
dtest = xgb.DMatrix(test_x, label=test_y)
# specify parameters via map
param = {'max_depth':1, 'eta':1, 'silent':1, 'objective':'binary:logistic' }
num_round = 2
bst = xgb.train(param, dtrain, num_round)

print('train accuracy: ', sum((bst.predict(dtrain)>0.5)==train_y)/len(train_y))
print('test accuracy: ', sum((bst.predict(dtest)>0.5)==test_y)/len(test_y))

predictions_and_bets['preds_xgb'] = bst.predict(dtest)

# assign bets based on the probability being above a threshold higher than the bookies' probability
probability_margin = 0.05
predictions_and_bets['betsP1_xgb'] = ((predictions_and_bets['preds_xgb']>(1/predictions_and_bets['b365P1']+probability_margin)) & (predictions_and_bets['b365P1']<=1.5))*1
predictions_and_bets['betsP2_xgb'] = (((1-predictions_and_bets['preds_xgb'])>(1/predictions_and_bets['b365P2']+probability_margin)) & (predictions_and_bets['b365P2']<=1.5))*1

# calculate winnings based on lower confidence probability bets
predictions_and_bets['winningsP1_xgb'] = predictions_and_bets['betsP1_xgb']*predictions_and_bets['b365P1']*(predictions_and_bets['player1Wins']==1)
predictions_and_bets['winningsP2_xgb'] = predictions_and_bets['betsP2_xgb']*predictions_and_bets['b365P2']*(predictions_and_bets['player1Wins']==0)


print('total bets: ',sum(predictions_and_bets['betsP1_xgb'])+sum(predictions_and_bets['betsP2_xgb']),
      'total winnings: ',sum(predictions_and_bets['winningsP1_xgb'])+sum(predictions_and_bets['winningsP2_xgb']),
      'return: ', ((sum(predictions_and_bets['winningsP1_xgb'])+sum(predictions_and_bets['winningsP2_xgb']))/(sum(predictions_and_bets['betsP1_xgb'])+sum(predictions_and_bets['betsP2_xgb']))-1)*100, '%')
print('P1 bets: ',sum(predictions_and_bets['betsP1_xgb']), 'P1 winnings: ',sum(predictions_and_bets['winningsP1_xgb']))
print('P2 bets: ',sum(predictions_and_bets['betsP2_xgb']), 'P2 winnings: ',sum(predictions_and_bets['winningsP2_xgb']))


