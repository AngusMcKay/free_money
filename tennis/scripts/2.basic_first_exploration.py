#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 21 09:57:41 2018

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


# read data
mens_data = pd.read_csv('all_data/mens_data.csv')
womens_data = pd.read_csv('all_data/womens_data.csv')

# one-hot encode a couple of features
mens_data = pd.concat([mens_data, pd.get_dummies(mens_data['Surface'])], axis=1)
mens_data = pd.concat([mens_data, pd.get_dummies(mens_data['Series'])], axis=1)
womens_data = pd.concat([womens_data, pd.get_dummies(womens_data['Surface'])], axis=1)
womens_data = pd.concat([womens_data, pd.get_dummies(womens_data['Tier'])], axis=1)

# edit the data towards predicting if the higher ranked player will win
mens_data['player1_Ranking'] = np.where(mens_data['LRank'] < mens_data['WRank'])


test_data_to_inspect = mens_data.iloc[:1000,:]
# basic first models
mens_features = ['Best of','LRank','WRank',
                 'Carpet','Clay','Grass','Hard',
                 'ATP250','ATP500','Grand Slam','International','International Gold',
                 'Masters','Masters 1000','Masters Cup']

train_x = combined_data_added_features_with_history[predictors][combined_data_added_features_with_history['seasonEndYear']<=train_to_season].values










