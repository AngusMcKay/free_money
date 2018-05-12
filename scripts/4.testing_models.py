#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 18:21:31 2018

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

# read combined data
combined_data = pd.read_csv('all_data/combined_data.csv')





### test basic xgb and linear model
train_to_season = 2016
probability_cushion = 0.00

# ['B365Hprob','seasonEndYear','seasonWeek','B365bookiesgain','LBH', 'N1', 'E1', 'SC3', 'SC2', 'D1', 'B1', 'I2', 'G1', 'E3', 'T1', 'EC','D2', 'F1', 'I1', 'P1', 'SP2', 'SP1', 'E2', 'SC0', 'E0', 'F2', 'SC1']
predictors = ['B365Hprob']#,'seasonEndYear','seasonWeek', 'N1', 'E1', 'SC3', 'SC2', 'D1', 'B1', 'I2', 'G1', 'E3', 'T1', 'EC','D2', 'F1', 'I1', 'P1', 'SP2', 'SP1', 'E2', 'SC0', 'E0', 'F2', 'SC1']

xgb_model = xgb.XGBRegressor(max_depth=2,
                             n_estimators=20,
                             objective='reg:linear').fit(combined_data[predictors][combined_data['seasonEndYear']<=train_to_season], ((combined_data['FTR']=='H')*1)[combined_data['seasonEndYear']<=train_to_season])
linRegMod=linear_model.LinearRegression()
linRegMod.fit(X=combined_data[predictors][combined_data['seasonEndYear']<=train_to_season], y=((combined_data['FTR']=='H')*1)[combined_data['seasonEndYear']<=train_to_season])

combined_outcomes = combined_data[['Date', 'B365H','B365Hprob','LBH','LBHprob','seasonEndYear','seasonWeek','FTR']][combined_data['seasonEndYear']==(train_to_season+1)]
combined_outcomes['predictions'] = xgb_model.predict(combined_data[predictors][combined_data['seasonEndYear']==(train_to_season+1)])
#combined_outcomes['predictions'] = linRegMod.predict(combined_data[predictors][combined_data['seasonEndYear']==(train_to_season+1)])
combined_outcomes['bets'] = (combined_outcomes['predictions']>(1/combined_outcomes['B365H']+probability_cushion)) #& (combined_outcomes['predictions']>(combined_outcomes['B365Hprob']+probability_cushion))
combined_outcomes['winnings'] = combined_outcomes['bets']*(combined_outcomes[['B365H','LBH']].max(axis=1))*(combined_outcomes['FTR']=='H')

match_days_combined = pd.unique(combined_data['Date'][combined_data['seasonEndYear']==(train_to_season+1)])
match_days_combined = np.sort(match_days_combined)

results_combined = pd.DataFrame(match_days_combined, columns=['match_day'])
results_combined['gamesThatDay'] = [sum(combined_outcomes['Date']==match_date) for match_date in match_days_combined]
results_combined['gamesBetOn'] = [sum((combined_outcomes['Date']==match_date) & (combined_outcomes['bets'])) for match_date in match_days_combined]
results_combined['winnings'] = [sum((combined_outcomes['Date']==match_date)*combined_outcomes['winnings']) for match_date in match_days_combined]
results_combined['profitLoss'] = results_combined['winnings']-results_combined['gamesBetOn']
results_combined['cumulativeProfitLoss'] = [sum(results_combined['profitLoss'][:(row+1)]) for row in range(len(match_days_combined))]

print('Total bet: ', sum(results_combined['gamesBetOn']), 'Winnings: ', sum(results_combined['winnings']), 'Profit: ', round((sum(results_combined['winnings'])/sum(results_combined['gamesBetOn'])-1)*100,2), '%')
#print('Total bet: ', sum(combined_outcomes['bets'][combined_outcomes['prem']==1]) + sum(combined_outcomes['bets'][combined_outcomes['champ']==1]), 'Winnings: ', sum(combined_outcomes['winnings'][combined_outcomes['prem']==1]) + sum(combined_outcomes['winnings'][combined_outcomes['champ']==1]), 'Profit: ', round(((sum(combined_outcomes['winnings'][combined_outcomes['prem']==1]) + sum(combined_outcomes['winnings'][combined_outcomes['champ']==1]))/(sum(combined_outcomes['bets'][combined_outcomes['prem']==1]) + sum(combined_outcomes['bets'][combined_outcomes['champ']==1]))-1)*100,2), '%')

#pred_cutoff=0.5
#print('Total bet: ', sum(combined_outcomes['bets'][combined_outcomes['predictions']>pred_cutoff]), 'Winnings: ', sum(combined_outcomes['winnings'][combined_outcomes['predictions']>pred_cutoff]), 'Profit: ', round((sum(combined_outcomes['winnings'][combined_outcomes['predictions']>pred_cutoff])/sum(combined_outcomes['bets'][combined_outcomes['predictions']>pred_cutoff])-1)*100,2), '%')

plt.plot(results_combined['cumulativeProfitLoss']*1000/sum(results_combined['gamesBetOn']))

linRegMod.coef_
linRegMod.intercept_

# try loss function of actual loss
# sort out analysis to more easily show results that can be copied and pasted to ss and then test features etc
# add more features like team size
# maybe concentrate on improving model before looking at winnings
# compare odds between bookies
# arbitrage?







### test simple neural net with home games
train_to_season = 2016
probability_cushion = 0.01

# ['B365Hprob','seasonEndYear','seasonWeek','B365bookiesgain','LBH', 'N1', 'E1', 'SC3', 'SC2', 'D1', 'B1', 'I2', 'G1', 'E3', 'T1', 'EC','D2', 'F1', 'I1', 'P1', 'SP2', 'SP1', 'E2', 'SC0', 'E0', 'F2', 'SC1']
predictors = ['B365Hprob','B365bookiesgain', 'N1', 'E1', 'SC3', 'SC2', 'D1', 'B1', 'I2', 'G1', 'E3', 'T1', 'EC','D2', 'F1', 'I1', 'P1', 'SP2', 'SP1', 'E2', 'SC0', 'E0', 'F2', 'SC1']#,'seasonEndYear','seasonWeek', 'N1', 'E1', 'SC3', 'SC2', 'D1', 'B1', 'I2', 'G1', 'E3', 'T1', 'EC','D2', 'F1', 'I1', 'P1', 'SP2', 'SP1', 'E2', 'SC0', 'E0', 'F2', 'SC1']
train_x = combined_data[predictors][combined_data['seasonEndYear']<=train_to_season].values
train_y = ((combined_data['FTR']=='H')*1)[combined_data['seasonEndYear']<=train_to_season].values
test_x = combined_data[predictors][combined_data['seasonEndYear']==(train_to_season+1)].values

input_dimension = len(predictors)

model = Sequential()
model.add(Dense(input_dimension*2, input_dim=input_dimension, activation='relu', init=initializers.Ones()))
model.add(Dense(input_dimension, input_dim=input_dimension, activation='relu'))
model.add(Dense(input_dimension, input_dim=input_dimension, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

optim_rmsprop = optimizers.RMSprop(lr=0.001, rho=0.9)
optim_sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.5, nesterov=True)
optim_adagrad = optimizers.Adagrad(lr=0.01)

model.compile(loss='binary_crossentropy',
              optimizer=optim_adagrad,
              metrics=['accuracy'])

model.fit(train_x,
          train_y,
          epochs=40,
          batch_size=10)


nn_combined_outcomes = combined_data[['Date', 'B365H','B365Hprob','LBH','LBHprob','seasonEndYear','seasonWeek','FTR']][combined_data['seasonEndYear']==(train_to_season+1)]
nn_combined_outcomes['predictions'] = model.predict(test_x)
nn_combined_outcomes['bets'] = (nn_combined_outcomes['predictions']>(1/nn_combined_outcomes['B365H']+probability_cushion)) #& (nn_combined_outcomes['predictions']>(nn_combined_outcomes['B365Hprob']+probability_cushion))
nn_combined_outcomes['winnings'] = nn_combined_outcomes['bets']*(nn_combined_outcomes[['B365H','LBH']].max(axis=1))*(nn_combined_outcomes['FTR']=='H')

match_days_combined = pd.unique(combined_data['Date'][combined_data['seasonEndYear']==(train_to_season+1)])
match_days_combined = np.sort(match_days_combined)

nn_results_combined = pd.DataFrame(match_days_combined, columns=['match_day'])
nn_results_combined['gamesThatDay'] = [sum(nn_combined_outcomes['Date']==match_date) for match_date in match_days_combined]
nn_results_combined['gamesBetOn'] = [sum((nn_combined_outcomes['Date']==match_date) & (nn_combined_outcomes['bets'])) for match_date in match_days_combined]
nn_results_combined['winnings'] = [sum((nn_combined_outcomes['Date']==match_date)*nn_combined_outcomes['winnings']) for match_date in match_days_combined]
nn_results_combined['profitLoss'] = nn_results_combined['winnings']-nn_results_combined['gamesBetOn']
nn_results_combined['cumulativeProfitLoss'] = [sum(nn_results_combined['profitLoss'][:(row+1)]) for row in range(len(match_days_combined))]

print('Total bet: ', sum(nn_results_combined['gamesBetOn']), 'Winnings: ', sum(nn_results_combined['winnings']), 'Profit: ', round((sum(nn_results_combined['winnings'])/sum(nn_results_combined['gamesBetOn'])-1)*100,2), '%')

in_bets = model.predict(combined_data[predictors][combined_data['seasonEndYear']<=train_to_season].values)[:,0]>(1/(combined_data['B365H'][combined_data['seasonEndYear']<=train_to_season])+probability_cushion)
in_winnings = in_bets*((combined_data[['B365H','LBH']][combined_data['seasonEndYear']<=train_to_season]).max(axis=1))*((combined_data['FTR']=='H')[combined_data['seasonEndYear']<=train_to_season])
print('Total bet (in): ', sum(in_bets), 'Winnings: ', sum(in_winnings), 'Profit: ', round((sum(in_winnings)/sum(in_bets)-1)*100,2), '%')

all_winnings = ((combined_data[['B365H','LBH']][combined_data['seasonEndYear']==(train_to_season+1)]).max(axis=1))*((combined_data['FTR']=='H')[combined_data['seasonEndYear']==(train_to_season+1)])
print('Total bet (all): ', len(all_winnings), 'Winnings: ', sum(all_winnings), 'Profit: ', round((sum(all_winnings)/len(all_winnings)-1)*100,2), '%')



### test simple neural net with just draws
train_to_season = 2016
probability_cushion = 0.01

# ['B365Hprob','seasonEndYear','seasonWeek','B365bookiesgain','LBH', 'N1', 'E1', 'SC3', 'SC2', 'D1', 'B1', 'I2', 'G1', 'E3', 'T1', 'EC','D2', 'F1', 'I1', 'P1', 'SP2', 'SP1', 'E2', 'SC0', 'E0', 'F2', 'SC1']
predictors = ['B365Dprob','B365bookiesgain', 'N1', 'E1', 'SC3', 'SC2', 'D1', 'B1', 'I2', 'G1', 'E3', 'T1', 'EC','D2', 'F1', 'I1', 'P1', 'SP2', 'SP1', 'E2', 'SC0', 'E0', 'F2', 'SC1']#,'seasonEndYear','seasonWeek', 'N1', 'E1', 'SC3', 'SC2', 'D1', 'B1', 'I2', 'G1', 'E3', 'T1', 'EC','D2', 'F1', 'I1', 'P1', 'SP2', 'SP1', 'E2', 'SC0', 'E0', 'F2', 'SC1']
train_x = combined_data[predictors][combined_data['seasonEndYear']<=train_to_season].values
train_y = ((combined_data['FTR']=='D')*1)[combined_data['seasonEndYear']<=train_to_season].values
test_x = combined_data[predictors][combined_data['seasonEndYear']==(train_to_season+1)].values

input_dimension = len(predictors)

model = Sequential()
model.add(Dense(input_dimension*2, input_dim=input_dimension, activation='relu', init=initializers.Ones()))
model.add(Dense(input_dimension, input_dim=input_dimension, activation='relu'))
model.add(Dense(input_dimension, input_dim=input_dimension, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

optim_rmsprop = optimizers.RMSprop(lr=0.001, rho=0.9)
optim_sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.5, nesterov=True)
optim_adagrad = optimizers.Adagrad(lr=0.01)

model.compile(loss='binary_crossentropy',
              optimizer=optim_adagrad,
              metrics=['accuracy'])

model.fit(train_x,
          train_y,
          epochs=40,
          batch_size=10)


nn_combined_outcomes = combined_data[['Date', 'B365D','B365Dprob','LBD','LBDprob','seasonEndYear','seasonWeek','FTR']][combined_data['seasonEndYear']==(train_to_season+1)]
nn_combined_outcomes['predictions'] = model.predict(test_x)
nn_combined_outcomes['bets'] = (nn_combined_outcomes['predictions']>(1/nn_combined_outcomes['B365D']+probability_cushion)) #& (nn_combined_outcomes['predictions']>(nn_combined_outcomes['B365Dprob']+probability_cushion))
nn_combined_outcomes['winnings'] = nn_combined_outcomes['bets']*(nn_combined_outcomes[['B365D','LBD']].max(axis=1))*(nn_combined_outcomes['FTR']=='D')

match_days_combined = pd.unique(combined_data['Date'][combined_data['seasonEndYear']==(train_to_season+1)])
match_days_combined = np.sort(match_days_combined)

nn_results_combined = pd.DataFrame(match_days_combined, columns=['match_day'])
nn_results_combined['gamesThatDay'] = [sum(nn_combined_outcomes['Date']==match_date) for match_date in match_days_combined]
nn_results_combined['gamesBetOn'] = [sum((nn_combined_outcomes['Date']==match_date) & (nn_combined_outcomes['bets'])) for match_date in match_days_combined]
nn_results_combined['winnings'] = [sum((nn_combined_outcomes['Date']==match_date)*nn_combined_outcomes['winnings']) for match_date in match_days_combined]
nn_results_combined['profitLoss'] = nn_results_combined['winnings']-nn_results_combined['gamesBetOn']
nn_results_combined['cumulativeProfitLoss'] = [sum(nn_results_combined['profitLoss'][:(row+1)]) for row in range(len(match_days_combined))]

print('Total bet: ', sum(nn_results_combined['gamesBetOn']), 'Winnings: ', sum(nn_results_combined['winnings']), 'Profit: ', round((sum(nn_results_combined['winnings'])/sum(nn_results_combined['gamesBetOn'])-1)*100,2), '%')

in_bets = model.predict(combined_data[predictors][combined_data['seasonEndYear']<=train_to_season].values)[:,0]>(1/(combined_data['B365D'][combined_data['seasonEndYear']<=train_to_season])+probability_cushion)
in_winnings = in_bets*((combined_data[['B365D','LBD']][combined_data['seasonEndYear']<=train_to_season]).max(axis=1))*((combined_data['FTR']=='D')[combined_data['seasonEndYear']<=train_to_season])
print('Total bet (in): ', sum(in_bets), 'Winnings: ', sum(in_winnings), 'Profit: ', round((sum(in_winnings)/sum(in_bets)-1)*100,2), '%')

all_winnings = ((combined_data[['B365D','LBD']][combined_data['seasonEndYear']==(train_to_season+1)]).max(axis=1))*((combined_data['FTR']=='D')[combined_data['seasonEndYear']==(train_to_season+1)])
print('Total bet (all): ', len(all_winnings), 'Winnings: ', sum(all_winnings), 'Profit: ', round((sum(all_winnings)/len(all_winnings)-1)*100,2), '%')


### test simple neural net with just away games
train_to_season = 2016
probability_cushion = 0.01

# ['B365Hprob','seasonEndYear','seasonWeek','B365bookiesgain','LBH', 'N1', 'E1', 'SC3', 'SC2', 'D1', 'B1', 'I2', 'G1', 'E3', 'T1', 'EC','D2', 'F1', 'I1', 'P1', 'SP2', 'SP1', 'E2', 'SC0', 'E0', 'F2', 'SC1']
predictors = ['B365Aprob','B365bookiesgain', 'N1', 'E1', 'SC3', 'SC2', 'D1', 'B1', 'I2', 'G1', 'E3', 'T1', 'EC','D2', 'F1', 'I1', 'P1', 'SP2', 'SP1', 'E2', 'SC0', 'E0', 'F2', 'SC1']#,'seasonEndYear','seasonWeek', 'N1', 'E1', 'SC3', 'SC2', 'D1', 'B1', 'I2', 'G1', 'E3', 'T1', 'EC','D2', 'F1', 'I1', 'P1', 'SP2', 'SP1', 'E2', 'SC0', 'E0', 'F2', 'SC1']
train_x = combined_data[predictors][combined_data['seasonEndYear']<=train_to_season].values
train_y = ((combined_data['FTR']=='A')*1)[combined_data['seasonEndYear']<=train_to_season].values
test_x = combined_data[predictors][combined_data['seasonEndYear']==(train_to_season+1)].values

input_dimension = len(predictors)

model = Sequential()
model.add(Dense(input_dimension*2, input_dim=input_dimension, activation='relu', init=initializers.Ones()))
model.add(Dense(input_dimension, input_dim=input_dimension, activation='relu'))
model.add(Dense(input_dimension, input_dim=input_dimension, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

optim_rmsprop = optimizers.RMSprop(lr=0.001, rho=0.9)
optim_sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.5, nesterov=True)
optim_adagrad = optimizers.Adagrad(lr=0.01)

model.compile(loss='binary_crossentropy',
              optimizer=optim_adagrad,
              metrics=['accuracy'])

model.fit(train_x,
          train_y,
          epochs=10,
          batch_size=10)


nn_combined_outcomes = combined_data[['Date', 'B365A','B365Aprob','LBA','LBAprob','seasonEndYear','seasonWeek','FTR']][combined_data['seasonEndYear']==(train_to_season+1)]
nn_combined_outcomes['predictions'] = model.predict(test_x)
nn_combined_outcomes['bets'] = (nn_combined_outcomes['predictions']>(1/nn_combined_outcomes['B365A']+probability_cushion)) #& (nn_combined_outcomes['predictions']>(nn_combined_outcomes['B365Aprob']+probability_cushion))
nn_combined_outcomes['winnings'] = nn_combined_outcomes['bets']*(nn_combined_outcomes[['B365A','LBA']].max(axis=1))*(nn_combined_outcomes['FTR']=='A')

match_days_combined = pd.unique(combined_data['Date'][combined_data['seasonEndYear']==(train_to_season+1)])
match_days_combined = np.sort(match_days_combined)

nn_results_combined = pd.DataFrame(match_days_combined, columns=['match_day'])
nn_results_combined['gamesThatDay'] = [sum(nn_combined_outcomes['Date']==match_date) for match_date in match_days_combined]
nn_results_combined['gamesBetOn'] = [sum((nn_combined_outcomes['Date']==match_date) & (nn_combined_outcomes['bets'])) for match_date in match_days_combined]
nn_results_combined['winnings'] = [sum((nn_combined_outcomes['Date']==match_date)*nn_combined_outcomes['winnings']) for match_date in match_days_combined]
nn_results_combined['profitLoss'] = nn_results_combined['winnings']-nn_results_combined['gamesBetOn']
nn_results_combined['cumulativeProfitLoss'] = [sum(nn_results_combined['profitLoss'][:(row+1)]) for row in range(len(match_days_combined))]

print('Total bet: ', sum(nn_results_combined['gamesBetOn']), 'Winnings: ', sum(nn_results_combined['winnings']), 'Profit: ', round((sum(nn_results_combined['winnings'])/sum(nn_results_combined['gamesBetOn'])-1)*100,2), '%')

in_bets = model.predict(combined_data[predictors][combined_data['seasonEndYear']<=train_to_season].values)[:,0]>(1/(combined_data['B365A'][combined_data['seasonEndYear']<=train_to_season])+probability_cushion)
in_winnings = in_bets*((combined_data[['B365A','LBA']][combined_data['seasonEndYear']<=train_to_season]).max(axis=1))*((combined_data['FTR']=='A')[combined_data['seasonEndYear']<=train_to_season])
print('Total bet (in): ', sum(in_bets), 'Winnings: ', sum(in_winnings), 'Profit: ', round((sum(in_winnings)/sum(in_bets)-1)*100,2), '%')

all_winnings = ((combined_data[['B365A','LBA']][combined_data['seasonEndYear']==(train_to_season+1)]).max(axis=1))*((combined_data['FTR']=='A')[combined_data['seasonEndYear']==(train_to_season+1)])
print('Total bet (all): ', len(all_winnings), 'Winnings: ', sum(all_winnings), 'Profit: ', round((sum(all_winnings)/len(all_winnings)-1)*100,2), '%')



### Neural network with H, D and A to essentially improve sample size
train_to_season = 2016
probability_cushion = 0.00

# ['B365Hprob','seasonEndYear','seasonWeek','B365bookiesgain','LBH', 'N1', 'E1', 'SC3', 'SC2', 'D1', 'B1', 'I2', 'G1', 'E3', 'T1', 'EC','D2', 'F1', 'I1', 'P1', 'SP2', 'SP1', 'E2', 'SC0', 'E0', 'F2', 'SC1']
predictors = ['B365Hprob','B365Dprob','B365Aprob','B365bookiesgain','seasonWeek', 'N1', 'E1', 'SC3', 'SC2', 'D1', 'B1', 'I2', 'G1', 'E3', 'T1', 'EC','D2', 'F1', 'I1', 'P1', 'SP2', 'SP1', 'E2', 'SC0', 'E0', 'F2', 'SC1']
train_x = combined_data[predictors][combined_data['seasonEndYear']<=train_to_season].values
train_y = (pd.get_dummies(combined_data['FTR']))[combined_data['seasonEndYear']<=train_to_season].values
test_x = combined_data[predictors][combined_data['seasonEndYear']==(train_to_season+2)].values

input_dimension = len(predictors)

model = Sequential()
model.add(Dropout(0.2, input_shape=(input_dimension,)))
model.add(Dense(input_dimension*2, input_dim=input_dimension, activation='relu', init=initializers.Constant(0.9)))
model.add(Dense(input_dimension, activation='relu'))
#model.add(Dense(input_dimension, activation='relu'))
#model.add(Dense(input_dimension, activation='relu'))
#model.add(Dense(input_dimension, input_dim=input_dimension, activation='relu'))
#model.add(Dense(input_dimension, input_dim=input_dimension, activation='relu'))
model.add(Dense(3, activation='softmax'))

epoch_for_tenth_decay = 50
input_decay=0.5**(1/epoch_for_tenth_decay)

optim_rmsprop = optimizers.RMSprop(lr=0.0001, rho=0.9)
optim_sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.5, nesterov=True)
optim_adagrad = optimizers.Adagrad(lr=0.01, decay=input_decay)
optim_adam = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=input_decay)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(train_x,
          train_y,
          epochs=50,
          batch_size=10)


nn_multi_combined_outcomes = combined_data[['Date', 'B365A','B365D','B365H','B365Aprob','B365Dprob','B365Hprob','LBA','LBD','LBH','LBAprob','LBDprob','LBHprob','seasonEndYear','seasonWeek','FTR']][combined_data['seasonEndYear']==(train_to_season+2)]
nn_multi_combined_outcomes['Apreds'] = model.predict(test_x)[:,0]
nn_multi_combined_outcomes['Dpreds'] = model.predict(test_x)[:,1]
nn_multi_combined_outcomes['Hpreds'] = model.predict(test_x)[:,2]

probability_cushion = 0.05
nn_multi_combined_outcomes['Abets'] = (nn_multi_combined_outcomes['Apreds']>(1/nn_multi_combined_outcomes['B365A']+probability_cushion)) #& (nn_multi_combined_outcomes['Apreds']>(nn_multi_combined_outcomes['B365Aprob']+probability_cushion))
nn_multi_combined_outcomes['Dbets'] = (nn_multi_combined_outcomes['Dpreds']>(1/nn_multi_combined_outcomes['B365D']+probability_cushion)) #& (nn_multi_combined_outcomes['Dpreds']>(nn_multi_combined_outcomes['B365Dprob']+probability_cushion))
nn_multi_combined_outcomes['Hbets'] = (nn_multi_combined_outcomes['Hpreds']>(1/nn_multi_combined_outcomes['B365H']+probability_cushion)) #& (nn_multi_combined_outcomes['Hpreds']>(nn_multi_combined_outcomes['B365Hprob']+probability_cushion))

nn_multi_combined_outcomes['Awinnings'] = nn_multi_combined_outcomes['Abets']*(nn_multi_combined_outcomes[['B365A','LBA']].max(axis=1))*(nn_multi_combined_outcomes['FTR']=='A')
nn_multi_combined_outcomes['Dwinnings'] = nn_multi_combined_outcomes['Dbets']*(nn_multi_combined_outcomes[['B365D','LBD']].max(axis=1))*(nn_multi_combined_outcomes['FTR']=='D')
nn_multi_combined_outcomes['Hwinnings'] = nn_multi_combined_outcomes['Hbets']*(nn_multi_combined_outcomes[['B365H','LBH']].max(axis=1))*(nn_multi_combined_outcomes['FTR']=='H')

match_days_combined = pd.unique(combined_data['Date'][combined_data['seasonEndYear']==(train_to_season+2)])
match_days_combined = np.sort(match_days_combined)


nn_multi_results_combined = pd.DataFrame(match_days_combined, columns=['match_day'])
nn_multi_results_combined['gamesThatDay'] = [sum(nn_multi_combined_outcomes['Date']==match_date) for match_date in match_days_combined]
nn_multi_results_combined['betsMade'] = [sum(((nn_multi_combined_outcomes[['Abets','Dbets','Hbets']]).sum(axis=1))*(nn_multi_combined_outcomes['Date']==match_date)) for match_date in match_days_combined]
nn_multi_results_combined['winnings'] = [sum(((nn_multi_combined_outcomes[['Awinnings','Dwinnings','Hwinnings']]).sum(axis=1))*(nn_multi_combined_outcomes['Date']==match_date)) for match_date in match_days_combined]
nn_multi_results_combined['profitLoss'] = nn_multi_results_combined['winnings']-nn_multi_results_combined['betsMade']
nn_multi_results_combined['cumulativeProfitLoss'] = [sum(nn_multi_results_combined['profitLoss'][:(row+1)]) for row in range(len(match_days_combined))]

print('Total bet: ', sum(nn_multi_results_combined['betsMade']), 'Winnings: ', sum(nn_multi_results_combined['winnings']), 'Profit: ', round((sum(nn_multi_results_combined['winnings'])/sum(nn_multi_results_combined['betsMade'])-1)*100,2), '%')
print('Home bets: ', sum(nn_multi_combined_outcomes['Hbets']), 'Winnings: ', sum(nn_multi_combined_outcomes['Hwinnings']), 'Profit: ', round((sum(nn_multi_combined_outcomes['Hwinnings'])/sum(nn_multi_combined_outcomes['Hbets'])-1)*100,2), '%')
print('Draw bets: ', sum(nn_multi_combined_outcomes['Dbets']), 'Winnings: ', sum(nn_multi_combined_outcomes['Dwinnings']), 'Profit: ', round((sum(nn_multi_combined_outcomes['Dwinnings'])/sum(nn_multi_combined_outcomes['Dbets'])-1)*100,2), '%')
print('Away bets: ', sum(nn_multi_combined_outcomes['Abets']), 'Winnings: ', sum(nn_multi_combined_outcomes['Awinnings']), 'Profit: ', round((sum(nn_multi_combined_outcomes['Awinnings'])/sum(nn_multi_combined_outcomes['Abets'])-1)*100,2), '%')

all_home_winnings = ((combined_data[['B365H','LBH']][combined_data['seasonEndYear']==(train_to_season+2)]).max(axis=1))*((combined_data['FTR']=='H')[combined_data['seasonEndYear']==(train_to_season+2)])
all_draw_winnings = ((combined_data[['B365D','LBD']][combined_data['seasonEndYear']==(train_to_season+2)]).max(axis=1))*((combined_data['FTR']=='D')[combined_data['seasonEndYear']==(train_to_season+2)])
all_away_winnings = ((combined_data[['B365A','LBA']][combined_data['seasonEndYear']==(train_to_season+2)]).max(axis=1))*((combined_data['FTR']=='A')[combined_data['seasonEndYear']==(train_to_season+2)])
print('Total bet (all home): ', len(all_home_winnings), 'Winnings: ', sum(all_home_winnings), 'Profit: ', round((sum(all_home_winnings)/len(all_home_winnings)-1)*100,2), '%')
print('Total bet (all draw): ', len(all_draw_winnings), 'Winnings: ', sum(all_draw_winnings), 'Profit: ', round((sum(all_draw_winnings)/len(all_draw_winnings)-1)*100,2), '%')
print('Total bet (all away): ', len(all_away_winnings), 'Winnings: ', sum(all_away_winnings), 'Profit: ', round((sum(all_away_winnings)/len(all_away_winnings)-1)*100,2), '%')

plt.plot(nn_multi_results_combined['cumulativeProfitLoss']*100/sum(nn_multi_results_combined['betsMade']))

# use squared loss to test for overfitting
np.mean((model.predict(combined_data[predictors][combined_data['seasonEndYear']<=(train_to_season)].values)-(pd.get_dummies(combined_data['FTR']))[combined_data['seasonEndYear']<=train_to_season].values)**2)
np.mean((model.predict(combined_data[predictors][combined_data['seasonEndYear']==(train_to_season+2)].values)-(pd.get_dummies(combined_data['FTR']))[combined_data['seasonEndYear']==(train_to_season+2)].values)**2)





### to do
# try additional features and test on basic models
# investigate whether easier to predict on some months
# could try training always on latest data (this might help overcome changes in bookies' systems each year)
# change loss function
# decide whether to work with h, d, a or just h model
# look at variability of bets






