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





mens_data_headers = pd.read_csv('all_data/mens_data_headers.csv')
mens_data_sql_processed = pd.read_csv('all_data/mens_data_sql_processed.csv', header=None, names=list(mens_data_headers))


# base features
base_features = ['best_of','clay','p1Rank','p2Rank']
bookies_features = ['b365P1prob','b365P2prob','b365bookiesgain']


# remove rows with missing past results
num_past_results = 10
mens_data_sql_processed_with_history = mens_data_sql_processed[(mens_data_sql_processed['p110Win']!='\\N') & (mens_data_sql_processed['p210Win']!='\\N')]


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
    mens_data_sql_processed_with_history['p1'+str(i+1)+'WinRank'] = 1/mens_data_sql_processed_with_history['p1'+str(i+1)+'OpponentRank'].astype(float) + mens_data_sql_processed_with_history['p1'+str(i+1)+'Win'].astype(float) - 1
    mens_data_sql_processed_with_history['p2'+str(i+1)+'WinRank'] = 1/mens_data_sql_processed_with_history['p2'+str(i+1)+'OpponentRank'].astype(float) + mens_data_sql_processed_with_history['p2'+str(i+1)+'Win'].astype(float) - 1
    pastWinRank_features = pastWinRank_features + ['p1'+str(i+1)+'WinRank', 'p2'+str(i+1)+'WinRank']


test = mens_data_sql_processed_with_history.iloc[:100,:]


# create past set and game percentage won
pastSets_features = []
pastGames_features = []
for i in range(num_past_results):
    
    # past sets
    mens_data_sql_processed_with_history['p1'+str(i+1)+'SetsProp'] = (mens_data_sql_processed_with_history['p1'+str(i+1)+'Playersets'].astype(float))/(mens_data_sql_processed_with_history['p1'+str(i+1)+'Playersets'].astype(float) + mens_data_sql_processed_with_history['p1'+str(i+1)+'Opponentsets'].astype(float))
    mens_data_sql_processed_with_history['p2'+str(i+1)+'SetsProp'] = (mens_data_sql_processed_with_history['p2'+str(i+1)+'Playersets'].astype(float))/(mens_data_sql_processed_with_history['p2'+str(i+1)+'Playersets'].astype(float) + mens_data_sql_processed_with_history['p2'+str(i+1)+'Opponentsets'].astype(float))
    pastSets_features = pastSets_features + ['p1'+str(i+1)+'SetsProp', 'p2'+str(i+1)+'SetsProp']
    
    # past games
    mens_data_sql_processed_with_history['p1'+str(i+1)+'GamesProp'] = (mens_data_sql_processed_with_history['p1'+str(i+1)+'Playergames1'].astype(float)+
                                        mens_data_sql_processed_with_history['p1'+str(i+1)+'Playergames2'].astype(float)+
                                        mens_data_sql_processed_with_history['p1'+str(i+1)+'Playergames3'].astype(float)+
                                        mens_data_sql_processed_with_history['p1'+str(i+1)+'Playergames4'].astype(float)+
                                        mens_data_sql_processed_with_history['p1'+str(i+1)+'Playergames5'].astype(float))/(
                                                mens_data_sql_processed_with_history['p1'+str(i+1)+'Playergames1'].astype(float)+
                                                mens_data_sql_processed_with_history['p1'+str(i+1)+'Playergames2'].astype(float)+
                                                mens_data_sql_processed_with_history['p1'+str(i+1)+'Playergames3'].astype(float)+
                                                mens_data_sql_processed_with_history['p1'+str(i+1)+'Playergames4'].astype(float)+
                                                mens_data_sql_processed_with_history['p1'+str(i+1)+'Playergames5'].astype(float)+
                                                mens_data_sql_processed_with_history['p1'+str(i+1)+'Opponentgames1'].astype(float)+
                                                mens_data_sql_processed_with_history['p1'+str(i+1)+'Opponentgames2'].astype(float)+
                                                mens_data_sql_processed_with_history['p1'+str(i+1)+'Opponentgames3'].astype(float)+
                                                mens_data_sql_processed_with_history['p1'+str(i+1)+'Opponentgames4'].astype(float)+
                                                mens_data_sql_processed_with_history['p1'+str(i+1)+'Opponentgames5'].astype(float))
    mens_data_sql_processed_with_history['p2'+str(i+1)+'GamesProp'] = (mens_data_sql_processed_with_history['p2'+str(i+1)+'Playergames1'].astype(float)+
                                        mens_data_sql_processed_with_history['p2'+str(i+1)+'Playergames2'].astype(float)+
                                        mens_data_sql_processed_with_history['p2'+str(i+1)+'Playergames3'].astype(float)+
                                        mens_data_sql_processed_with_history['p2'+str(i+1)+'Playergames4'].astype(float)+
                                        mens_data_sql_processed_with_history['p2'+str(i+1)+'Playergames5'].astype(float))/(
                                                mens_data_sql_processed_with_history['p2'+str(i+1)+'Playergames1'].astype(float)+
                                                mens_data_sql_processed_with_history['p2'+str(i+1)+'Playergames2'].astype(float)+
                                                mens_data_sql_processed_with_history['p2'+str(i+1)+'Playergames3'].astype(float)+
                                                mens_data_sql_processed_with_history['p2'+str(i+1)+'Playergames4'].astype(float)+
                                                mens_data_sql_processed_with_history['p2'+str(i+1)+'Playergames5'].astype(float)+
                                                mens_data_sql_processed_with_history['p2'+str(i+1)+'Opponentgames1'].astype(float)+
                                                mens_data_sql_processed_with_history['p2'+str(i+1)+'Opponentgames2'].astype(float)+
                                                mens_data_sql_processed_with_history['p2'+str(i+1)+'Opponentgames3'].astype(float)+
                                                mens_data_sql_processed_with_history['p2'+str(i+1)+'Opponentgames4'].astype(float)+
                                                mens_data_sql_processed_with_history['p2'+str(i+1)+'Opponentgames5'].astype(float))
    pastGames_features = pastSets_features + ['p1'+str(i+1)+'GamesProp', 'p2'+str(i+1)+'GamesProp']


test = mens_data_sql_processed_with_history.iloc[:100,:]
list(mens_data_sql_processed_with_history)


# test out features in model

features = base_features+pastWinRank_features+pastSets_features+pastGames_features#+bookies_features+past_bookies_features

# remove 34 rows which have some NaNs
sum(mens_data_sql_processed_with_history[features].isnull().sum(axis=1)>0)
mens_data_sql_processed_with_history = mens_data_sql_processed_with_history[mens_data_sql_processed_with_history[features].isnull().sum(axis=1)==0]


max(mens_data_sql_processed_with_history['date'])
train_to_date = '2018-01-01'

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
#weights = np.zeros(train_x.shape[0])+1
#weights = 0.9 + np.asarray(mens_data_sql_processed_with_history['player1Wins'][mens_data_sql_processed_with_history['date']<train_to_date]*0.2)
weights = np.asarray(mens_data_sql_processed_with_history['b365P1'][mens_data_sql_processed_with_history['date']<train_to_date])
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
#model1.add(Dense(input_dimension, activation='relu',
#           #kernel_regularizer=regularizers.l1(0.0005),
#           #activity_regularizer=regularizers.l1(0.01)
#           ))
#model1.add(Dense(input_dimension, activation='relu',
#           #kernel_regularizer=regularizers.l1(0.0005),
#           #activity_regularizer=regularizers.l1(0.01)
#           ))
model1.add(Dense(1, activation='sigmoid'))
model1.compile(loss='binary_crossentropy', optimizer=optim_adagrad, metrics=['accuracy'])
model1.fit(train_x,train_y,epochs=number_epochs,batch_size=batch_sizes,validation_split=val_split, sample_weight=weights)



# test predictions
predictions_and_bets = mens_data_sql_processed_with_history[['date', 'b365P1','b365P2','player1Wins']][mens_data_sql_processed_with_history['date']>=train_to_date]
predictions_and_bets['preds'] = model1.predict(test_x)

# assign bets based on the probability being above a threshold higher than the bookies' probability
probability_margin = 0.0
predictions_and_bets['betsP1'] = ((predictions_and_bets['preds']>(1/predictions_and_bets['b365P1']+probability_margin)) & (predictions_and_bets['b365P1']<=1.5))*1
predictions_and_bets['betsP2'] = (((1-predictions_and_bets['preds'])>(1/predictions_and_bets['b365P2']+probability_margin)) & (predictions_and_bets['b365P2']<=1.5))*1

# calculate winnings based on lower confidence probability bets
predictions_and_bets['winningsP1'] = predictions_and_bets['betsP1']*predictions_and_bets['b365P1']*(predictions_and_bets['player1Wins']==1)
predictions_and_bets['winningsP2'] = predictions_and_bets['betsP2']*predictions_and_bets['b365P2']*(predictions_and_bets['player1Wins']==0)


print('total bets: ',sum(predictions_and_bets['betsP1'])+sum(predictions_and_bets['betsP2']),
      'total winnings: ',sum(predictions_and_bets['winningsP1'])+sum(predictions_and_bets['winningsP2']),
      'return: ', ((sum(predictions_and_bets['winningsP1'])+sum(predictions_and_bets['winningsP2']))/(sum(predictions_and_bets['betsP1'])+sum(predictions_and_bets['betsP2']))-1)*100, '%')
print('P1 bets: ',sum(predictions_and_bets['betsP1']), 'P1 winnings: ',sum(predictions_and_bets['winningsP1']))
print('P2 bets: ',sum(predictions_and_bets['betsP2']), 'P2 winnings: ',sum(predictions_and_bets['winningsP2']))



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


