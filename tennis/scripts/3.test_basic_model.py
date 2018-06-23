#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 21 11:56:56 2018

@author: angus
"""



import os
os.chdir('/home/angus/projects/betting/tennis')

import numpy as np
import pandas as pd
from keras.models import Sequential, model_from_json
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import optimizers
from keras import initializers
from keras import models
from keras import regularizers
from sklearn.preprocessing import LabelEncoder
from scipy.stats import norm
#import mysql.connector

# read data
mens_data_reformatted = pd.read_csv('all_data/mens_data_reformatted.csv')

features = ['Best of','Carpet','Clay','Grass','Hard',
            'ATP250','ATP500','Grand Slam','International','International Gold','Masters','Masters 1000','Masters Cup',
            'p1Rank','p2Rank']

train_x = mens_data_reformatted[features].values
train_y = mens_data_reformatted['player1Wins'].values
train_col_means = train_x.mean(axis=0)
train_col_stds = train_x.astype(float).std(axis=0)
train_col_stds = train_col_stds + (train_col_stds==0)*1 # adds 1 where the stdev is 0 to not break division
train_x = (train_x - train_col_means)/train_col_stds

input_dimension = len(features)

# optimizers
optim_rmsprop = optimizers.RMSprop(lr=0.0001, rho=0.9)
optim_sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.5, nesterov=True)
optim_adagrad = optimizers.Adagrad(lr=0.1, decay=0)
optim_adam = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0)

number_epochs_init=50
batch_sizes=2**7
val_split=0.1
dropout = 0.0
weights = np.zeros(train_x.shape[0])+1

model1 = Sequential()
model1.add(Dense(input_dimension, input_dim=input_dimension, activation='relu',
           #kernel_regularizer=regularizers.l1(0.0001),
           #activity_regularizer=regularizers.l1(0.01)
           ))
model1.add(Dense(1, activation='sigmoid'))
model1.compile(loss='binary_crossentropy', optimizer=optim_sgd, metrics=['accuracy'])
model1.fit(train_x,train_y,epochs=number_epochs_init,batch_size=batch_sizes,validation_split=val_split, sample_weight=weights)





### do similar but with sql processed data
mens_data_headers = pd.read_csv('all_data/mens_data_headers.csv')
mens_data_sql_processed = pd.read_csv('all_data/mens_data_sql_processed.csv', header=None, names=list(mens_data_headers))
mens_data_sql_processed.describe()
mens_data_sql_processed_with_history = mens_data_sql_processed[(mens_data_sql_processed['p110Win']!='\\N') & (mens_data_sql_processed['p210Win']!='\\N')]
test = mens_data_sql_processed_with_history.iloc[:100,:]
mens_data_sql_processed_with_history.describe()

features = list(mens_data_headers)
features.pop(features.index('player1Wins'))


train_x = mens_data_sql_processed_with_history[features].values
train_x = train_x.astype(float)
train_y = mens_data_sql_processed_with_history['player1Wins'].values
train_col_means = train_x.mean(axis=0)
train_col_stds = train_x.astype(float).std(axis=0)
train_col_stds = train_col_stds + (train_col_stds==0)*1 # adds 1 where the stdev is 0 to not break division
train_x = (train_x - train_col_means)/train_col_stds

input_dimension = len(features)

# optimizers
optim_rmsprop = optimizers.RMSprop(lr=0.0001, rho=0.9)
optim_sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.5, nesterov=True)
optim_adagrad = optimizers.Adagrad(lr=0.1, decay=0)
optim_adam = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0)

number_epochs=5
batch_sizes=2**7
val_split=0.1
dropout = 0.0
weights = np.zeros(train_x.shape[0])+1

model1 = Sequential()
model1.add(Dense(input_dimension, input_dim=input_dimension, activation='relu',
           kernel_regularizer=regularizers.l1(0.001),
           #activity_regularizer=regularizers.l1(0.01)
           ))
model1.add(Dense(1, activation='sigmoid'))
model1.compile(loss='binary_crossentropy', optimizer=optim_sgd, metrics=['accuracy'])
model1.fit(train_x,train_y,epochs=number_epochs,batch_size=batch_sizes,validation_split=val_split, sample_weight=weights)





















