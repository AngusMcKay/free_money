#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 12:11:38 2018

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

# clean up some of the unwanted colums
columns_to_remove = ['Unnamed: 10', 'Unnamed: 11', 'Unnamed: 12', 'Unnamed: 13',
                     'Unnamed: 14', 'Unnamed: 15', 'Unnamed: 16', 'Unnamed: 17',
                     'Unnamed: 18', 'Unnamed: 19', 'Unnamed: 20', 'Unnamed: 21',
                     'Unnamed: 22', 'Unnamed: 23', 'Unnamed: 24', 'Unnamed: 25',
                     'Unnamed: 26', 'Unnamed: 27', 'Unnamed: 28', 'Unnamed: 29',
                     'Unnamed: 30', 'Unnamed: 31', 'Unnamed: 32', 'Unnamed: 33',
                     'Unnamed: 34', 'Unnamed: 35', 'Unnamed: 36', 'Unnamed: 37',
                     'Unnamed: 38', 'Unnamed: 39', 'Unnamed: 40', 'Unnamed: 41',
                     'Unnamed: 42', 'Unnamed: 43', 'Unnamed: 44', 'Unnamed: 45',
                     'Unnamed: 46', 'Unnamed: 47', 'Unnamed: 48', 'Unnamed: 49',
                     'Unnamed: 50', 'Unnamed: 51', 'Unnamed: 52', 'Unnamed: 53',
                     'Unnamed: 54', 'Unnamed: 55', 'Unnamed: 56', 'Unnamed: 57',
                     'Unnamed: 58', 'Unnamed: 59', 'Unnamed: 60', 'Unnamed: 61',
                     'Unnamed: 62', 'Unnamed: 65', 'Unnamed: 66', 'Unnamed: 67',
                     'Unnamed: 7', 'Unnamed: 70', 'Unnamed: 71', 'Unnamed: 72',
                     'Unnamed: 8', 'Unnamed: 9']

combined_data = combined_data.drop(columns_to_remove, axis=1)


### features to add
# popularity of team (e.g. top teams like man U etc, although maybe this is reflected in points and leagues)
# last head to head between teams
# goals scored, fouls etc



# order data and get list of teams
combined_data = combined_data.sort_values(['Date'])

list_of_teams = pd.unique(pd.concat([combined_data['AwayTeam'],combined_data['HomeTeam']]))
defaultWDL = 'L'
defaultGoals = 0
defaultHomeAway = 'H'
past_results_to_include = 50

# add some columns to store additional features
for i in range(past_results_to_include):
    combined_data['homePastResults'+str(i+1)] = None
    combined_data['awayPastResults'+str(i+1)] = None
    
    combined_data['homePRYear'+str(i+1)] = None
    combined_data['awayPRYear'+str(i+1)] = None
    
    combined_data['homePRGoals'+str(i+1)] = None
    combined_data['awayPRGoals'+str(i+1)] = None
    
    combined_data['homePRHomeAway'+str(i+1)] = None
    combined_data['awayPRHomeAway'+str(i+1)] = None
    
    combined_data['homePRLeague'+str(i+1)] = None
    combined_data['awayPRLeague'+str(i+1)] = None


def win_lose_draw(result, team_in_question, homeTeam):
    if ((result == 'H') & (team_in_question==homeTeam)):
        return 'W'
    elif ((result == 'A') & (team_in_question!=homeTeam)):
        return 'W'
    elif (result == 'D'):
        return 'D'
    else:
        return 'L'


# setup insertion of extra features for first team in list (to then copy into a loop afterwards)
single_team_data = combined_data[(combined_data['HomeTeam']==list_of_teams[0]) | (combined_data['AwayTeam']==list_of_teams[0])]

single_team_data['result'] = [win_lose_draw(result, list_of_teams[0], homeTeam) for result, homeTeam in zip(single_team_data['FTR'],single_team_data['HomeTeam'])]

single_team_data['tempPastResults1'] = [defaultWDL]+list(single_team_data['result'][:(single_team_data.shape[0]-1)])
single_team_data['tempPRYear1'] = ['NA'] + list(single_team_data['seasonEndYear'][:(single_team_data.shape[0]-1)])
single_team_data['tempPRGoals1'] = [defaultGoals]+list((single_team_data['FTHG']*(single_team_data['HomeTeam']==list_of_teams[0]))[:(single_team_data.shape[0]-1)]+(single_team_data['FTAG']*(single_team_data['AwayTeam']==list_of_teams[0]))[:(single_team_data.shape[0]-1)])
single_team_data['tempPRHomeAway1'] = [defaultHomeAway] + ['H' if (HomeTeam==list_of_teams[0]) else 'A' for HomeTeam in single_team_data['HomeTeam']][:(single_team_data.shape[0]-1)]
single_team_data['tempPRLeague1'] = ['NA'] + list(single_team_data['Div'][:(single_team_data.shape[0]-1)])


for i in range(1, past_results_to_include):
    single_team_data['tempPastResults'+str(i+1)] = [defaultWDL]+list(single_team_data['tempPastResults'+str(i)][:(single_team_data.shape[0]-1)])
    single_team_data['tempPRYear'+str(i+1)] = ['NA'] + list(single_team_data['tempPRYear'+str(i)][:(single_team_data.shape[0]-1)])
    single_team_data['tempPRGoals'+str(i+1)] = [defaultGoals]+list(single_team_data['tempPRGoals'+str(i)][:(single_team_data.shape[0]-1)])
    single_team_data['tempPRHomeAway'+str(i+1)] = [defaultHomeAway] + list(single_team_data['tempPRHomeAway'+str(i)][:(single_team_data.shape[0]-1)])
    single_team_data['tempPRLeague'+str(i+1)] = ['NA'] + list(single_team_data['tempPRLeague'+str(i)][:(single_team_data.shape[0]-1)])

# reattach extra columns to original data
columns_to_join = sum([['tempPastResults'+str(i+1), 'tempPRYear'+str(i+1), 'tempPRGoals'+str(i+1), 'tempPRHomeAway'+str(i+1), 'tempPRLeague'+str(i+1)] for i in range(past_results_to_include)], [])

combined_data = combined_data.merge(single_team_data[columns_to_join], how='left', left_index=True, right_index=True)

for i in range(past_results_to_include):
    combined_data['homePastResults'+str(i+1)] = np.where(combined_data['HomeTeam']==list_of_teams[0], combined_data['tempPastResults'+str(i+1)], combined_data['homePastResults'+str(i+1)])
    combined_data['awayPastResults'+str(i+1)] = np.where(combined_data['AwayTeam']==list_of_teams[0], combined_data['tempPastResults'+str(i+1)], combined_data['awayPastResults'+str(i+1)])
    
    combined_data['homePRYear'+str(i+1)] = np.where(combined_data['HomeTeam']==list_of_teams[0], combined_data['tempPRYear'+str(i+1)], combined_data['homePRYear'+str(i+1)])
    combined_data['awayPRYear'+str(i+1)] = np.where(combined_data['AwayTeam']==list_of_teams[0], combined_data['tempPRYear'+str(i+1)], combined_data['awayPRYear'+str(i+1)])
    
    combined_data['homePRGoals'+str(i+1)] = np.where(combined_data['HomeTeam']==list_of_teams[0], combined_data['tempPRGoals'+str(i+1)], combined_data['homePRGoals'+str(i+1)])
    combined_data['awayPRGoals'+str(i+1)] = np.where(combined_data['AwayTeam']==list_of_teams[0], combined_data['tempPRGoals'+str(i+1)], combined_data['awayPRGoals'+str(i+1)])
    
    combined_data['homePRHomeAway'+str(i+1)] = np.where(combined_data['HomeTeam']==list_of_teams[0], combined_data['tempPRHomeAway'+str(i+1)], combined_data['homePRHomeAway'+str(i+1)])
    combined_data['awayPRHomeAway'+str(i+1)] = np.where(combined_data['AwayTeam']==list_of_teams[0], combined_data['tempPRHomeAway'+str(i+1)], combined_data['awayPRHomeAway'+str(i+1)])
    
    combined_data['homePRLeague'+str(i+1)] = np.where(combined_data['HomeTeam']==list_of_teams[0], combined_data['tempPRLeague'+str(i+1)], combined_data['homePRLeague'+str(i+1)])
    combined_data['awayPRLeague'+str(i+1)] = np.where(combined_data['AwayTeam']==list_of_teams[0], combined_data['tempPRLeague'+str(i+1)], combined_data['awayPRLeague'+str(i+1)])

# drop the temp columns to make way for the next teams data to be added
combined_data = combined_data.drop(columns_to_join, axis=1)


### now loop over above for remaining teams
for i in tqdm(range(1,len(list_of_teams))):
    team = list_of_teams[i]
    
    # subset data to only games including team in question and add features
    single_team_data = combined_data[(combined_data['HomeTeam']==team) | (combined_data['AwayTeam']==team)]
    single_team_data['result'] = [win_lose_draw(result, team, homeTeam) for result, homeTeam in zip(single_team_data['FTR'],single_team_data['HomeTeam'])]
    single_team_data['tempPastResults1'] = [defaultWDL]+list(single_team_data['result'][:(single_team_data.shape[0]-1)])
    single_team_data['tempPRYear1'] = ['NA'] + list(single_team_data['seasonEndYear'][:(single_team_data.shape[0]-1)])
    single_team_data['tempPRGoals1'] = [defaultGoals]+list((single_team_data['FTHG']*(single_team_data['HomeTeam']==team))[:(single_team_data.shape[0]-1)]+(single_team_data['FTAG']*(single_team_data['AwayTeam']==team))[:(single_team_data.shape[0]-1)])
    single_team_data['tempPRHomeAway1'] = [defaultHomeAway] + ['H' if (HomeTeam==team) else 'A' for HomeTeam in single_team_data['HomeTeam']][:(single_team_data.shape[0]-1)]
    single_team_data['tempPRLeague1'] = ['NA'] + list(single_team_data['Div'][:(single_team_data.shape[0]-1)])
    
    for i in range(1, past_results_to_include):
        single_team_data['tempPastResults'+str(i+1)] = [defaultWDL]+list(single_team_data['tempPastResults'+str(i)][:(single_team_data.shape[0]-1)])
        single_team_data['tempPRYear'+str(i+1)] = ['NA'] + list(single_team_data['tempPRYear'+str(i)][:(single_team_data.shape[0]-1)])
        single_team_data['tempPRGoals'+str(i+1)] = [defaultGoals]+list(single_team_data['tempPRGoals'+str(i)][:(single_team_data.shape[0]-1)])
        single_team_data['tempPRHomeAway'+str(i+1)] = [defaultHomeAway] + list(single_team_data['tempPRHomeAway'+str(i)][:(single_team_data.shape[0]-1)])
        single_team_data['tempPRLeague'+str(i+1)] = ['NA'] + list(single_team_data['tempPRLeague'+str(i)][:(single_team_data.shape[0]-1)])
    
    # reattach extra columns to original data
    columns_to_join = sum([['tempPastResults'+str(i+1), 'tempPRYear'+str(i+1), 'tempPRGoals'+str(i+1), 'tempPRHomeAway'+str(i+1), 'tempPRLeague'+str(i+1)] for i in range(past_results_to_include)], [])
    combined_data = combined_data.merge(single_team_data[columns_to_join], how='left', left_index=True, right_index=True)
    
    for i in range(past_results_to_include):
        combined_data['homePastResults'+str(i+1)] = np.where(combined_data['HomeTeam']==team, combined_data['tempPastResults'+str(i+1)], combined_data['homePastResults'+str(i+1)])
        combined_data['awayPastResults'+str(i+1)] = np.where(combined_data['AwayTeam']==team, combined_data['tempPastResults'+str(i+1)], combined_data['awayPastResults'+str(i+1)])
        
        combined_data['homePRYear'+str(i+1)] = np.where(combined_data['HomeTeam']==team, combined_data['tempPRYear'+str(i+1)], combined_data['homePRYear'+str(i+1)])
        combined_data['awayPRYear'+str(i+1)] = np.where(combined_data['AwayTeam']==team, combined_data['tempPRYear'+str(i+1)], combined_data['awayPRYear'+str(i+1)])
        
        combined_data['homePRGoals'+str(i+1)] = np.where(combined_data['HomeTeam']==team, combined_data['tempPRGoals'+str(i+1)], combined_data['homePRGoals'+str(i+1)])
        combined_data['awayPRGoals'+str(i+1)] = np.where(combined_data['AwayTeam']==team, combined_data['tempPRGoals'+str(i+1)], combined_data['awayPRGoals'+str(i+1)])
        
        combined_data['homePRHomeAway'+str(i+1)] = np.where(combined_data['HomeTeam']==team, combined_data['tempPRHomeAway'+str(i+1)], combined_data['homePRHomeAway'+str(i+1)])
        combined_data['awayPRHomeAway'+str(i+1)] = np.where(combined_data['AwayTeam']==team, combined_data['tempPRHomeAway'+str(i+1)], combined_data['awayPRHomeAway'+str(i+1)])
        
        combined_data['homePRLeague'+str(i+1)] = np.where(combined_data['HomeTeam']==team, combined_data['tempPRLeague'+str(i+1)], combined_data['homePRLeague'+str(i+1)])
        combined_data['awayPRLeague'+str(i+1)] = np.where(combined_data['AwayTeam']==team, combined_data['tempPRLeague'+str(i+1)], combined_data['awayPRLeague'+str(i+1)])
    
    # drop the temp columns to make way for the next teams data to be added
    combined_data = combined_data.drop(columns_to_join, axis=1)


# save to csv and inspect
combined_data.to_csv('all_data/combined_data_added_features.csv', index=False)
combined_data_added_features = pd.read_csv('all_data/combined_data_added_features.csv')
list_of_teams = pd.unique(pd.concat([combined_data_added_features['AwayTeam'],combined_data_added_features['HomeTeam']]))
data_to_inspect = combined_data_added_features.iloc[[12345, 99999, 32, 64335, 5525, 2234],:]
#data_to_inspect = combined_data[combined_data['HomeTeam'].isnull()]
# problem with G1 (greek games) where teams are recorded under HT and AT instead of HomeTeam and AwayTeam
combined_data_added_features = combined_data_added_features[combined_data_added_features['HomeTeam'].notnull()]

# how many rows are lost if only take games where there is 20 past games of data for each team?
sum((combined_data_added_features['homePRYear20'].isnull()) | (combined_data_added_features['awayPRYear20'].isnull()))



### create some features with the new data
data_to_inspect = combined_data_added_features.iloc[[12345, 67890, 45678],:]

# number of wins, draws and losses in last 20 and 5 matches (where different league count as draw)
combined_data_added_features['homeLongFormNumWins'] = 0
combined_data_added_features['homeLongFormNumDraws'] = 0
combined_data_added_features['homeLongFormNumLosses'] = 0
combined_data_added_features['awayLongFormNumWins'] = 0
combined_data_added_features['awayLongFormNumDraws'] = 0
combined_data_added_features['awayLongFormNumLosses'] = 0

combined_data_added_features['homeShortFormNumWins'] = 0
combined_data_added_features['homeShortFormNumDraws'] = 0
combined_data_added_features['homeShortFormNumLosses'] = 0
combined_data_added_features['awayShortFormNumWins'] = 0
combined_data_added_features['awayShortFormNumDraws'] = 0
combined_data_added_features['awayShortFormNumLosses'] = 0


long_term_form_number_matches = 20
short_term_form_number_matches = 5
for i in range(long_term_form_number_matches):
    combined_data_added_features['homeLongFormNumWins'] = combined_data_added_features['homeLongFormNumWins']+((combined_data_added_features['homePRLeague'+str(i+1)]==combined_data_added_features['Div']) & (combined_data_added_features['homePastResults'+str(i+1)]=='W'))*1
    combined_data_added_features['homeLongFormNumDraws'] = combined_data_added_features['homeLongFormNumDraws']+((combined_data_added_features['homePRLeague'+str(i+1)]!=combined_data_added_features['Div']) | (combined_data_added_features['homePastResults'+str(i+1)]=='D'))*1
    combined_data_added_features['homeLongFormNumLosses'] =combined_data_added_features['homeLongFormNumLosses']+((combined_data_added_features['homePRLeague'+str(i+1)]==combined_data_added_features['Div']) & (combined_data_added_features['homePastResults'+str(i+1)]=='L'))*1
    combined_data_added_features['awayLongFormNumWins'] = combined_data_added_features['awayLongFormNumWins']+((combined_data_added_features['awayPRLeague'+str(i+1)]==combined_data_added_features['Div']) & (combined_data_added_features['awayPastResults'+str(i+1)]=='W'))*1
    combined_data_added_features['awayLongFormNumDraws'] = combined_data_added_features['awayLongFormNumDraws']+((combined_data_added_features['awayPRLeague'+str(i+1)]!=combined_data_added_features['Div']) | (combined_data_added_features['awayPastResults'+str(i+1)]=='D'))*1
    combined_data_added_features['awayLongFormNumLosses'] =combined_data_added_features['awayLongFormNumLosses']+((combined_data_added_features['awayPRLeague'+str(i+1)]==combined_data_added_features['Div']) & (combined_data_added_features['awayPastResults'+str(i+1)]=='L'))*1


for i in range(short_term_form_number_matches):
    combined_data_added_features['homeShortFormNumWins'] = combined_data_added_features['homeShortFormNumWins']+((combined_data_added_features['homePRLeague'+str(i+1)]==combined_data_added_features['Div']) & (combined_data_added_features['homePastResults'+str(i+1)]=='W'))*1
    combined_data_added_features['homeShortFormNumDraws'] = combined_data_added_features['homeShortFormNumDraws']+((combined_data_added_features['homePRLeague'+str(i+1)]!=combined_data_added_features['Div']) | (combined_data_added_features['homePastResults'+str(i+1)]=='D'))*1
    combined_data_added_features['homeShortFormNumLosses'] =combined_data_added_features['homeShortFormNumLosses']+((combined_data_added_features['homePRLeague'+str(i+1)]==combined_data_added_features['Div']) & (combined_data_added_features['homePastResults'+str(i+1)]=='L'))*1
    combined_data_added_features['awayShortFormNumWins'] = combined_data_added_features['awayShortFormNumWins']+((combined_data_added_features['awayPRLeague'+str(i+1)]==combined_data_added_features['Div']) & (combined_data_added_features['awayPastResults'+str(i+1)]=='W'))*1
    combined_data_added_features['awayShortFormNumDraws'] = combined_data_added_features['awayShortFormNumDraws']+((combined_data_added_features['awayPRLeague'+str(i+1)]!=combined_data_added_features['Div']) | (combined_data_added_features['awayPastResults'+str(i+1)]=='D'))*1
    combined_data_added_features['awayShortFormNumLosses'] =combined_data_added_features['awayShortFormNumLosses']+((combined_data_added_features['awayPRLeague'+str(i+1)]==combined_data_added_features['Div']) & (combined_data_added_features['awayPastResults'+str(i+1)]=='L'))*1



# number of points based on last 20 matches
combined_data_added_features['homeLongFormPoints'] = combined_data_added_features['homeLongFormNumWins']*3 + combined_data_added_features['homeLongFormNumDraws']*1
combined_data_added_features['awayLongFormPoints'] = combined_data_added_features['awayLongFormNumWins']*3 + combined_data_added_features['awayLongFormNumDraws']*1

combined_data_added_features['homeShortFormPoints'] = combined_data_added_features['homeShortFormNumWins']*3 + combined_data_added_features['homeShortFormNumDraws']*1
combined_data_added_features['awayShortFormPoints'] = combined_data_added_features['awayShortFormNumWins']*3 + combined_data_added_features['awayShortFormNumDraws']*1




# Add some columns to store goals against and points of opponents
for i in range(past_results_to_include):
    combined_data_added_features['homePRGoalsAgaints'+str(i+1)] = None
    combined_data_added_features['awayPRGoalsAgaints'+str(i+1)] = None
    
    combined_data_added_features['homePROppositionPoints'+str(i+1)] = None
    combined_data_added_features['awayPROppositionPoints'+str(i+1)] = None





# loop over teams to add goals against and points of opponents' long form
for i in tqdm(range(len(list_of_teams))):
    team = list_of_teams[i]
    
    # subset data to only games including team in question and add features
    single_team_data = combined_data_added_features[(combined_data_added_features['HomeTeam']==team) | (combined_data_added_features['AwayTeam']==team)]
    single_team_data['tempPRGoalsAgaints1'] = [defaultGoals]+list((single_team_data['FTAG']*(single_team_data['HomeTeam']==team))[:(single_team_data.shape[0]-1)]+(single_team_data['FTHG']*(single_team_data['AwayTeam']==team))[:(single_team_data.shape[0]-1)])
    single_team_data['tempPROppositionPoints1'] = [20]+list((single_team_data['awayLongFormPoints']*(single_team_data['HomeTeam']==team))[:(single_team_data.shape[0]-1)]+(single_team_data['homeLongFormPoints']*(single_team_data['AwayTeam']==team))[:(single_team_data.shape[0]-1)])
    
    for i in range(1, past_results_to_include):
        single_team_data['tempPRGoalsAgaints'+str(i+1)] = [defaultGoals]+list(single_team_data['tempPRGoalsAgaints'+str(i)][:(single_team_data.shape[0]-1)])
        single_team_data['tempPROppositionPoints'+str(i+1)] = [20]+list(single_team_data['tempPROppositionPoints'+str(i)][:(single_team_data.shape[0]-1)])
    
    # reattach extra columns to original data
    columns_to_join = sum([['tempPRGoalsAgaints'+str(i+1), 'tempPROppositionPoints'+str(i+1)] for i in range(past_results_to_include)], [])
    combined_data_added_features = combined_data_added_features.merge(single_team_data[columns_to_join], how='left', left_index=True, right_index=True)
    
    for i in range(past_results_to_include):
        combined_data_added_features['homePRGoalsAgaints'+str(i+1)] = np.where(combined_data_added_features['HomeTeam']==team, combined_data_added_features['tempPRGoalsAgaints'+str(i+1)], combined_data_added_features['homePRGoalsAgaints'+str(i+1)])
        combined_data_added_features['awayPRGoalsAgaints'+str(i+1)] = np.where(combined_data_added_features['AwayTeam']==team, combined_data_added_features['tempPRGoalsAgaints'+str(i+1)], combined_data_added_features['awayPRGoalsAgaints'+str(i+1)])
        
        combined_data_added_features['homePROppositionPoints'+str(i+1)] = np.where(combined_data_added_features['HomeTeam']==team, combined_data_added_features['tempPROppositionPoints'+str(i+1)], combined_data_added_features['homePROppositionPoints'+str(i+1)])
        combined_data_added_features['awayPROppositionPoints'+str(i+1)] = np.where(combined_data_added_features['AwayTeam']==team, combined_data_added_features['tempPROppositionPoints'+str(i+1)], combined_data_added_features['awayPROppositionPoints'+str(i+1)])
    
    # drop the temp columns to make way for the next teams data to be added
    combined_data_added_features = combined_data_added_features.drop(columns_to_join, axis=1)

    

data_to_inspect = combined_data_added_features.loc[(combined_data_added_features['AwayTeam']=='Juventus') | (combined_data_added_features['HomeTeam']=='Juventus'),:]
combined_data_added_features.to_csv('all_data/combined_data_added_features.csv', index=False)
combined_data_added_features = pd.read_csv('all_data/combined_data_added_features.csv')



# goals scored
combined_data_added_features['homeLongFormGoalsScored'] = 0
combined_data_added_features['awayLongFormGoalsScored'] = 0

combined_data_added_features['homeShortFormGoalsScored'] = 0
combined_data_added_features['awayShortFormGoalsScored'] = 0

for i in range(long_term_form_number_matches):
    combined_data_added_features['homeLongFormGoalsScored'] = combined_data_added_features['homeLongFormGoalsScored'] + combined_data_added_features['homePRGoals'+str(i+1)]
    combined_data_added_features['awayLongFormGoalsScored'] = combined_data_added_features['awayLongFormGoalsScored'] + combined_data_added_features['awayPRGoals'+str(i+1)]

for i in range(short_term_form_number_matches):
    combined_data_added_features['homeShortFormGoalsScored'] = combined_data_added_features['homeShortFormGoalsScored'] + combined_data_added_features['homePRGoals'+str(i+1)]
    combined_data_added_features['awayShortFormGoalsScored'] = combined_data_added_features['awayShortFormGoalsScored'] + combined_data_added_features['awayPRGoals'+str(i+1)]

# add specific home/away wins, draws, losses for home/away team
combined_data_added_features['homeShortFormHomeWins'] = 0
combined_data_added_features['homeShortFormHomeDraws'] = 0
combined_data_added_features['homeShortFormHomeLosses'] = 0
combined_data_added_features['awayShortFormAwayWins'] = 0
combined_data_added_features['awayShortFormAwayDraws'] = 0
combined_data_added_features['awayShortFormAwayLosses'] = 0


combined_data_added_features['homePRHomeOrder1'] = (combined_data_added_features['homePRHomeAway1']=='H')*1
combined_data_added_features['awayPRAwayOrder1'] = (combined_data_added_features['awayPRHomeAway1']=='A')*1

for i in range(1, past_results_to_include):
    combined_data_added_features['homePRHomeOrder'+str(i+1)] = combined_data_added_features['homePRHomeOrder'+str(i)] + (combined_data_added_features['homePRHomeAway'+str(i+1)]=='H')*1
    combined_data_added_features['awayPRAwayOrder'+str(i+1)] = combined_data_added_features['awayPRAwayOrder'+str(i)] + (combined_data_added_features['awayPRHomeAway'+str(i+1)]=='A')*1

past_home_away_matches_to_include = 5
for i in range(past_results_to_include):
    combined_data_added_features['homeShortFormHomeWins'] = combined_data_added_features['homeShortFormHomeWins'] + ((combined_data_added_features['homePRHomeOrder'+str(i+1)] >= 1) & (combined_data_added_features['homePRHomeOrder'+str(i+1)] <= past_home_away_matches_to_include) & (combined_data_added_features['homePRHomeAway'+str(i+1)]=='H') & (combined_data_added_features['homePastResults'+str(i+1)]=='W'))*1
    combined_data_added_features['homeShortFormHomeDraws'] = combined_data_added_features['homeShortFormHomeDraws'] + ((combined_data_added_features['homePRHomeOrder'+str(i+1)] >= 1) & (combined_data_added_features['homePRHomeOrder'+str(i+1)] <= past_home_away_matches_to_include) & (combined_data_added_features['homePRHomeAway'+str(i+1)]=='H') & (combined_data_added_features['homePastResults'+str(i+1)]=='D'))*1
    combined_data_added_features['homeShortFormHomeLosses'] = combined_data_added_features['homeShortFormHomeLosses'] + ((combined_data_added_features['homePRHomeOrder'+str(i+1)] >= 1) & (combined_data_added_features['homePRHomeOrder'+str(i+1)] <= past_home_away_matches_to_include) & (combined_data_added_features['homePRHomeAway'+str(i+1)]=='H') & (combined_data_added_features['homePastResults'+str(i+1)]=='L'))*1
    
    combined_data_added_features['awayShortFormAwayWins'] = combined_data_added_features['awayShortFormAwayWins'] + ((combined_data_added_features['awayPRAwayOrder'+str(i+1)] >= 1) & (combined_data_added_features['awayPRAwayOrder'+str(i+1)] <= past_home_away_matches_to_include) & (combined_data_added_features['awayPRHomeAway'+str(i+1)]=='A') & (combined_data_added_features['homePastResults'+str(i+1)]=='W'))*1
    combined_data_added_features['awayShortFormAwayDraws'] = combined_data_added_features['awayShortFormAwayDraws'] + ((combined_data_added_features['awayPRAwayOrder'+str(i+1)] >= 1) & (combined_data_added_features['awayPRAwayOrder'+str(i+1)] <= past_home_away_matches_to_include) & (combined_data_added_features['awayPRHomeAway'+str(i+1)]=='A') & (combined_data_added_features['homePastResults'+str(i+1)]=='D'))*1
    combined_data_added_features['awayShortFormAwayLosses'] = combined_data_added_features['awayShortFormAwayLosses'] + ((combined_data_added_features['awayPRAwayOrder'+str(i+1)] >= 1) & (combined_data_added_features['awayPRAwayOrder'+str(i+1)] <= past_home_away_matches_to_include) & (combined_data_added_features['awayPRHomeAway'+str(i+1)]=='A') & (combined_data_added_features['homePastResults'+str(i+1)]=='L'))*1

data_to_inspect = combined_data_added_features.iloc[[12345, 67890, 45678],:]


# convert past WDL to dummies
for i in range(past_results_to_include):
    combined_data_added_features = pd.concat([combined_data_added_features, pd.get_dummies(combined_data_added_features['homePastResults'+str(i+1)], prefix='homePR'+str(i+1))], axis=1)
    combined_data_added_features = pd.concat([combined_data_added_features, pd.get_dummies(combined_data_added_features['awayPastResults'+str(i+1)], prefix='awayPR'+str(i+1))], axis=1)


# convert PR home/away to 1 if home, 0 if away to dummies
for i in range(past_results_to_include):
    combined_data_added_features['homePRIsHomeGame'+str(i+1)] = [1 if homeAway=='H' else 0 for homeAway in combined_data_added_features['homePRHomeAway'+str(i+1)]]
    combined_data_added_features['awayPRIsHomeGame'+str(i+1)] = [1 if homeAway=='H' else 0 for homeAway in combined_data_added_features['awayPRHomeAway'+str(i+1)]]


combined_data_added_features.to_csv('all_data/combined_data_added_features.csv', index=False)
combined_data_added_features = pd.read_csv('all_data/combined_data_added_features.csv')







### Adding extra features for past games including:
# HS = Home Team Shots, AS = Away Team Shots, HST = Home Team Shots on Target, AST = Away Team Shots on Target
# HHW = Home Team Hit Woodwork, AHW = Away Team Hit Woodwork, HC = Home Team Corners, AC = Away Team Corners
# HF = Home Team Fouls Committed, AF = Away Team Fouls Committed, HFKC = Home Team Free Kicks Conceded, AFKC = Away Team Free Kicks Conceded
# HO = Home Team Offsides, AO = Away Team Offsides, HY = Home Team Yellow Cards, AY = Away Team Yellow Cards
# HR = Home Team Red Cards, AR = Away Team Red Cards


list_of_teams = pd.unique(pd.concat([combined_data_added_features['AwayTeam'],combined_data_added_features['HomeTeam']]))
defaultShotsEtc = 'NA'
past_results_to_include = 20 # can't be bothered doing all 50 for this anymore


# add columns to store extra variables
for i in range(past_results_to_include):
    combined_data_added_features['homePRShotsFor'+str(i+1)] = None
    combined_data_added_features['homePRShotsAgainst'+str(i+1)] = None
    combined_data_added_features['homePRSOTFor'+str(i+1)] = None
    combined_data_added_features['homePRSOTAgainst'+str(i+1)] = None
    #combined_data_added_features['homePRHitWWFor'+str(i+1)] = None
    #combined_data_added_features['homePRHitWWAgainst'+str(i+1)] = None
    combined_data_added_features['homePRCornersFor'+str(i+1)] = None
    combined_data_added_features['homePRCornersAgainst'+str(i+1)] = None
    combined_data_added_features['homePRFoulsFor'+str(i+1)] = None
    combined_data_added_features['homePRFoulsAgainst'+str(i+1)] = None
    #combined_data_added_features['homePRFreeKicksFor'+str(i+1)] = None
    #combined_data_added_features['homePRFreeKicksAgainst'+str(i+1)] = None
    #combined_data_added_features['homePROffsidesFor'+str(i+1)] = None
    #combined_data_added_features['homePROffsidesAgainst'+str(i+1)] = None
    combined_data_added_features['homePRYellowsFor'+str(i+1)] = None
    combined_data_added_features['homePRYellowsAgainst'+str(i+1)] = None
    combined_data_added_features['homePRRedsFor'+str(i+1)] = None
    combined_data_added_features['homePRRedsAgainst'+str(i+1)] = None

    combined_data_added_features['awayPRShotsFor'+str(i+1)] = None
    combined_data_added_features['awayPRShotsAgainst'+str(i+1)] = None
    combined_data_added_features['awayPRSOTFor'+str(i+1)] = None
    combined_data_added_features['awayPRSOTAgainst'+str(i+1)] = None
    #combined_data_added_features['awayPRHitWWFor'+str(i+1)] = None
    #combined_data_added_features['awayPRHitWWAgainst'+str(i+1)] = None
    combined_data_added_features['awayPRCornersFor'+str(i+1)] = None
    combined_data_added_features['awayPRCornersAgainst'+str(i+1)] = None
    combined_data_added_features['awayPRFoulsFor'+str(i+1)] = None
    combined_data_added_features['awayPRFoulsAgainst'+str(i+1)] = None
    #combined_data_added_features['awayPRFreeKicksFor'+str(i+1)] = None
    #combined_data_added_features['awayPRFreeKicksAgainst'+str(i+1)] = None
    #combined_data_added_features['awayPROffsidesFor'+str(i+1)] = None
    #combined_data_added_features['awayPROffsidesAgainst'+str(i+1)] = None
    combined_data_added_features['awayPRYellowsFor'+str(i+1)] = None
    combined_data_added_features['awayPRYellowsAgainst'+str(i+1)] = None
    combined_data_added_features['awayPRRedsFor'+str(i+1)] = None
    combined_data_added_features['awayPRRedsAgainst'+str(i+1)] = None


for i in tqdm(range(len(list_of_teams))):
    team = list_of_teams[i]
    
    # subset data to only games including team in question and add features
    single_team_data = combined_data_added_features[(combined_data_added_features['HomeTeam']==team) | (combined_data_added_features['AwayTeam']==team)]
    single_team_data['tempPRShotsFor1'] = [defaultShotsEtc]+list((single_team_data['HS']*(single_team_data['HomeTeam']==team))[:(single_team_data.shape[0]-1)]+(single_team_data['AS']*(single_team_data['AwayTeam']==team))[:(single_team_data.shape[0]-1)])
    single_team_data['tempPRShotsAgainst1'] = [defaultShotsEtc]+list((single_team_data['AS']*(single_team_data['HomeTeam']==team))[:(single_team_data.shape[0]-1)]+(single_team_data['HS']*(single_team_data['AwayTeam']==team))[:(single_team_data.shape[0]-1)])
    single_team_data['tempPRSOTFor1'] = [defaultShotsEtc]+list((single_team_data['HST']*(single_team_data['HomeTeam']==team))[:(single_team_data.shape[0]-1)]+(single_team_data['AST']*(single_team_data['AwayTeam']==team))[:(single_team_data.shape[0]-1)])
    single_team_data['tempPRSOTAgainst1'] = [defaultShotsEtc]+list((single_team_data['AST']*(single_team_data['HomeTeam']==team))[:(single_team_data.shape[0]-1)]+(single_team_data['HST']*(single_team_data['AwayTeam']==team))[:(single_team_data.shape[0]-1)])
    #single_team_data['tempPRHitWWFor1'] = [defaultShotsEtc]+list((single_team_data['HHW']*(single_team_data['HomeTeam']==team))[:(single_team_data.shape[0]-1)]+(single_team_data['AHW']*(single_team_data['AwayTeam']==team))[:(single_team_data.shape[0]-1)])
    #single_team_data['tempPRHitWWAgainst1'] = [defaultShotsEtc]+list((single_team_data['AHW']*(single_team_data['HomeTeam']==team))[:(single_team_data.shape[0]-1)]+(single_team_data['HHW']*(single_team_data['AwayTeam']==team))[:(single_team_data.shape[0]-1)])
    single_team_data['tempPRCornersFor1'] = [defaultShotsEtc]+list((single_team_data['HC']*(single_team_data['HomeTeam']==team))[:(single_team_data.shape[0]-1)]+(single_team_data['AC']*(single_team_data['AwayTeam']==team))[:(single_team_data.shape[0]-1)])
    single_team_data['tempPRCornersAgainst1'] = [defaultShotsEtc]+list((single_team_data['AC']*(single_team_data['HomeTeam']==team))[:(single_team_data.shape[0]-1)]+(single_team_data['HC']*(single_team_data['AwayTeam']==team))[:(single_team_data.shape[0]-1)])
    single_team_data['tempPRFoulsFor1'] = [defaultShotsEtc]+list((single_team_data['HF']*(single_team_data['HomeTeam']==team))[:(single_team_data.shape[0]-1)]+(single_team_data['AF']*(single_team_data['AwayTeam']==team))[:(single_team_data.shape[0]-1)])
    single_team_data['tempPRFoulsAgainst1'] = [defaultShotsEtc]+list((single_team_data['AF']*(single_team_data['HomeTeam']==team))[:(single_team_data.shape[0]-1)]+(single_team_data['HF']*(single_team_data['AwayTeam']==team))[:(single_team_data.shape[0]-1)])
    #single_team_data['tempPRFreeKicksFor1'] = [defaultShotsEtc]+list((single_team_data['HFKC']*(single_team_data['HomeTeam']==team))[:(single_team_data.shape[0]-1)]+(single_team_data['AFKC']*(single_team_data['AwayTeam']==team))[:(single_team_data.shape[0]-1)])
    #single_team_data['tempPRFreeKicksAgainst1'] = [defaultShotsEtc]+list((single_team_data['AFKC']*(single_team_data['HomeTeam']==team))[:(single_team_data.shape[0]-1)]+(single_team_data['HFKC']*(single_team_data['AwayTeam']==team))[:(single_team_data.shape[0]-1)])
    #single_team_data['tempPROffsidesFor1'] = [defaultShotsEtc]+list((single_team_data['HO']*(single_team_data['HomeTeam']==team))[:(single_team_data.shape[0]-1)]+(single_team_data['AO']*(single_team_data['AwayTeam']==team))[:(single_team_data.shape[0]-1)])
    #single_team_data['tempPROffsidesAgainst1'] = [defaultShotsEtc]+list((single_team_data['AO']*(single_team_data['HomeTeam']==team))[:(single_team_data.shape[0]-1)]+(single_team_data['HO']*(single_team_data['AwayTeam']==team))[:(single_team_data.shape[0]-1)])
    single_team_data['tempPRYellowsFor1'] = [defaultShotsEtc]+list((single_team_data['HY']*(single_team_data['HomeTeam']==team))[:(single_team_data.shape[0]-1)]+(single_team_data['AY']*(single_team_data['AwayTeam']==team))[:(single_team_data.shape[0]-1)])
    single_team_data['tempPRYellowsAgainst1'] = [defaultShotsEtc]+list((single_team_data['AY']*(single_team_data['HomeTeam']==team))[:(single_team_data.shape[0]-1)]+(single_team_data['HY']*(single_team_data['AwayTeam']==team))[:(single_team_data.shape[0]-1)])
    single_team_data['tempPRRedsFor1'] = [defaultShotsEtc]+list((single_team_data['HR']*(single_team_data['HomeTeam']==team))[:(single_team_data.shape[0]-1)]+(single_team_data['AR']*(single_team_data['AwayTeam']==team))[:(single_team_data.shape[0]-1)])
    single_team_data['tempPRRedsAgainst1'] = [defaultShotsEtc]+list((single_team_data['AR']*(single_team_data['HomeTeam']==team))[:(single_team_data.shape[0]-1)]+(single_team_data['HR']*(single_team_data['AwayTeam']==team))[:(single_team_data.shape[0]-1)])


    for i in range(1, past_results_to_include):
        single_team_data['tempPRShotsFor'+str(i+1)] = [defaultShotsEtc]+list(single_team_data['tempPRShotsFor'+str(i)][:(single_team_data.shape[0]-1)])
        single_team_data['tempPRShotsAgainst'+str(i+1)] = [defaultShotsEtc]+list(single_team_data['tempPRShotsAgainst'+str(i)][:(single_team_data.shape[0]-1)])
        single_team_data['tempPRSOTFor'+str(i+1)] = [defaultShotsEtc]+list(single_team_data['tempPRSOTFor'+str(i)][:(single_team_data.shape[0]-1)])
        single_team_data['tempPRSOTAgainst'+str(i+1)] = [defaultShotsEtc]+list(single_team_data['tempPRSOTAgainst'+str(i)][:(single_team_data.shape[0]-1)])
        #single_team_data['tempPRHitWWFor'+str(i+1)] = [defaultShotsEtc]+list(single_team_data['tempPRHitWWFor'+str(i)][:(single_team_data.shape[0]-1)])
        #single_team_data['tempPRHitWWAgainst'+str(i+1)] = [defaultShotsEtc]+list(single_team_data['tempPRHitWWAgainst'+str(i)][:(single_team_data.shape[0]-1)])
        single_team_data['tempPRCornersFor'+str(i+1)] = [defaultShotsEtc]+list(single_team_data['tempPRCornersFor'+str(i)][:(single_team_data.shape[0]-1)])
        single_team_data['tempPRCornersAgainst'+str(i+1)] = [defaultShotsEtc]+list(single_team_data['tempPRCornersAgainst'+str(i)][:(single_team_data.shape[0]-1)])
        single_team_data['tempPRFoulsFor'+str(i+1)] = [defaultShotsEtc]+list(single_team_data['tempPRFoulsFor'+str(i)][:(single_team_data.shape[0]-1)])
        single_team_data['tempPRFoulsAgainst'+str(i+1)] = [defaultShotsEtc]+list(single_team_data['tempPRFoulsAgainst'+str(i)][:(single_team_data.shape[0]-1)])
        #single_team_data['tempPRFreeKicksFor'+str(i+1)] = [defaultShotsEtc]+list(single_team_data['tempPRFreeKicksFor'+str(i)][:(single_team_data.shape[0]-1)])
        #single_team_data['tempPRFreeKicksAgainst'+str(i+1)] = [defaultShotsEtc]+list(single_team_data['tempPRFreeKicksAgainst'+str(i)][:(single_team_data.shape[0]-1)])
        #single_team_data['tempPROffsidesFor'+str(i+1)] = [defaultShotsEtc]+list(single_team_data['tempPROffsidesFor'+str(i)][:(single_team_data.shape[0]-1)])
        #single_team_data['tempPROffsidesAgainst'+str(i+1)] = [defaultShotsEtc]+list(single_team_data['tempPROffsidesAgainst'+str(i)][:(single_team_data.shape[0]-1)])
        single_team_data['tempPRYellowsFor'+str(i+1)] = [defaultShotsEtc]+list(single_team_data['tempPRYellowsFor'+str(i)][:(single_team_data.shape[0]-1)])
        single_team_data['tempPRYellowsAgainst'+str(i+1)] = [defaultShotsEtc]+list(single_team_data['tempPRYellowsAgainst'+str(i)][:(single_team_data.shape[0]-1)])
        single_team_data['tempPRRedsFor'+str(i+1)] = [defaultShotsEtc]+list(single_team_data['tempPRRedsFor'+str(i)][:(single_team_data.shape[0]-1)])
        single_team_data['tempPRRedsAgainst'+str(i+1)] = [defaultShotsEtc]+list(single_team_data['tempPRRedsAgainst'+str(i)][:(single_team_data.shape[0]-1)])
        
        
    # reattach extra columns to original data
    columns_to_join = sum([['tempPRShotsFor'+str(i+1), 'tempPRShotsAgainst'+str(i+1),
                            'tempPRSOTFor'+str(i+1), 'tempPRSOTAgainst'+str(i+1),
                            #'tempPRHitWWFor'+str(i+1), 'tempPRHitWWAgainst'+str(i+1),
                            'tempPRCornersFor'+str(i+1), 'tempPRCornersAgainst'+str(i+1),
                            'tempPRFoulsFor'+str(i+1), 'tempPRFoulsAgainst'+str(i+1),
                            #'tempPRFreeKicksFor'+str(i+1), 'tempPRFreeKicksAgainst'+str(i+1),
                            #'tempPROffsidesFor'+str(i+1), 'tempPROffsidesAgainst'+str(i+1),
                            'tempPRYellowsFor'+str(i+1), 'tempPRYellowsAgainst'+str(i+1),
                            'tempPRRedsFor'+str(i+1), 'tempPRRedsAgainst'+str(i+1)] for i in range(past_results_to_include)], [])
    combined_data_added_features = combined_data_added_features.merge(single_team_data[columns_to_join], how='left', left_index=True, right_index=True)
    
    
    for i in range(past_results_to_include):
        combined_data_added_features['homePRShotsFor'+str(i+1)] = np.where(combined_data_added_features['HomeTeam']==team, combined_data_added_features['tempPRShotsFor'+str(i+1)], combined_data_added_features['homePRShotsFor'+str(i+1)])
        combined_data_added_features['homePRShotsAgainst'+str(i+1)] = np.where(combined_data_added_features['HomeTeam']==team, combined_data_added_features['tempPRShotsAgainst'+str(i+1)], combined_data_added_features['homePRShotsAgainst'+str(i+1)])
        combined_data_added_features['homePRSOTFor'+str(i+1)] = np.where(combined_data_added_features['HomeTeam']==team, combined_data_added_features['tempPRSOTFor'+str(i+1)], combined_data_added_features['homePRSOTFor'+str(i+1)])
        combined_data_added_features['homePRSOTAgainst'+str(i+1)] = np.where(combined_data_added_features['HomeTeam']==team, combined_data_added_features['tempPRSOTAgainst'+str(i+1)], combined_data_added_features['homePRSOTAgainst'+str(i+1)])
        #combined_data_added_features['homePRHitWWFor'+str(i+1)] = np.where(combined_data_added_features['HomeTeam']==team, combined_data_added_features['tempPRHitWWFor'+str(i+1)], combined_data_added_features['homePRHitWWFor'+str(i+1)])
        #combined_data_added_features['homePRHitWWAgainst'+str(i+1)] = np.where(combined_data_added_features['HomeTeam']==team, combined_data_added_features['tempPRHitWWAgainst'+str(i+1)], combined_data_added_features['homePRHitWWAgainst'+str(i+1)])
        combined_data_added_features['homePRCornersFor'+str(i+1)] = np.where(combined_data_added_features['HomeTeam']==team, combined_data_added_features['tempPRCornersFor'+str(i+1)], combined_data_added_features['homePRCornersFor'+str(i+1)])
        combined_data_added_features['homePRCornersAgainst'+str(i+1)] = np.where(combined_data_added_features['HomeTeam']==team, combined_data_added_features['tempPRCornersAgainst'+str(i+1)], combined_data_added_features['homePRCornersAgainst'+str(i+1)])
        combined_data_added_features['homePRFoulsFor'+str(i+1)] = np.where(combined_data_added_features['HomeTeam']==team, combined_data_added_features['tempPRFoulsFor'+str(i+1)], combined_data_added_features['homePRFoulsFor'+str(i+1)])
        combined_data_added_features['homePRFoulsAgainst'+str(i+1)] = np.where(combined_data_added_features['HomeTeam']==team, combined_data_added_features['tempPRFoulsAgainst'+str(i+1)], combined_data_added_features['homePRFoulsAgainst'+str(i+1)])
        #combined_data_added_features['homePRFreeKicksFor'+str(i+1)] = np.where(combined_data_added_features['HomeTeam']==team, combined_data_added_features['tempPRFreeKicksFor'+str(i+1)], combined_data_added_features['homePRFreeKicksFor'+str(i+1)])
        #combined_data_added_features['homePRFreeKicksAgainst'+str(i+1)] = np.where(combined_data_added_features['HomeTeam']==team, combined_data_added_features['tempPRFreeKicksAgainst'+str(i+1)], combined_data_added_features['homePRFreeKicksAgainst'+str(i+1)])
        #combined_data_added_features['homePROffsidesFor'+str(i+1)] = np.where(combined_data_added_features['HomeTeam']==team, combined_data_added_features['tempPROffsidesFor'+str(i+1)], combined_data_added_features['homePROffsidesFor'+str(i+1)])
        #combined_data_added_features['homePROffsidesAgainst'+str(i+1)] = np.where(combined_data_added_features['HomeTeam']==team, combined_data_added_features['tempPROffsidesAgainst'+str(i+1)], combined_data_added_features['homePROffsidesAgainst'+str(i+1)])
        combined_data_added_features['homePRYellowsFor'+str(i+1)] = np.where(combined_data_added_features['HomeTeam']==team, combined_data_added_features['tempPRYellowsFor'+str(i+1)], combined_data_added_features['homePRYellowsFor'+str(i+1)])
        combined_data_added_features['homePRYellowsAgainst'+str(i+1)] = np.where(combined_data_added_features['HomeTeam']==team, combined_data_added_features['tempPRYellowsAgainst'+str(i+1)], combined_data_added_features['homePRYellowsAgainst'+str(i+1)])
        combined_data_added_features['homePRRedsFor'+str(i+1)] = np.where(combined_data_added_features['HomeTeam']==team, combined_data_added_features['tempPRRedsFor'+str(i+1)], combined_data_added_features['homePRRedsFor'+str(i+1)])
        combined_data_added_features['homePRRedsAgainst'+str(i+1)] = np.where(combined_data_added_features['HomeTeam']==team, combined_data_added_features['tempPRRedsAgainst'+str(i+1)], combined_data_added_features['homePRRedsAgainst'+str(i+1)])

        combined_data_added_features['awayPRShotsFor'+str(i+1)] = np.where(combined_data_added_features['AwayTeam']==team, combined_data_added_features['tempPRShotsFor'+str(i+1)], combined_data_added_features['awayPRShotsFor'+str(i+1)])
        combined_data_added_features['awayPRShotsAgainst'+str(i+1)] = np.where(combined_data_added_features['AwayTeam']==team, combined_data_added_features['tempPRShotsAgainst'+str(i+1)], combined_data_added_features['awayPRShotsAgainst'+str(i+1)])
        combined_data_added_features['awayPRSOTFor'+str(i+1)] = np.where(combined_data_added_features['AwayTeam']==team, combined_data_added_features['tempPRSOTFor'+str(i+1)], combined_data_added_features['awayPRSOTFor'+str(i+1)])
        combined_data_added_features['awayPRSOTAgainst'+str(i+1)] = np.where(combined_data_added_features['AwayTeam']==team, combined_data_added_features['tempPRSOTAgainst'+str(i+1)], combined_data_added_features['awayPRSOTAgainst'+str(i+1)])
        #combined_data_added_features['awayPRHitWWFor'+str(i+1)] = np.where(combined_data_added_features['AwayTeam']==team, combined_data_added_features['tempPRHitWWFor'+str(i+1)], combined_data_added_features['awayPRHitWWFor'+str(i+1)])
        #combined_data_added_features['awayPRHitWWAgainst'+str(i+1)] = np.where(combined_data_added_features['AwayTeam']==team, combined_data_added_features['tempPRHitWWAgainst'+str(i+1)], combined_data_added_features['awayPRHitWWAgainst'+str(i+1)])
        combined_data_added_features['awayPRCornersFor'+str(i+1)] = np.where(combined_data_added_features['AwayTeam']==team, combined_data_added_features['tempPRCornersFor'+str(i+1)], combined_data_added_features['awayPRCornersFor'+str(i+1)])
        combined_data_added_features['awayPRCornersAgainst'+str(i+1)] = np.where(combined_data_added_features['AwayTeam']==team, combined_data_added_features['tempPRCornersAgainst'+str(i+1)], combined_data_added_features['awayPRCornersAgainst'+str(i+1)])
        combined_data_added_features['awayPRFoulsFor'+str(i+1)] = np.where(combined_data_added_features['AwayTeam']==team, combined_data_added_features['tempPRFoulsFor'+str(i+1)], combined_data_added_features['awayPRFoulsFor'+str(i+1)])
        combined_data_added_features['awayPRFoulsAgainst'+str(i+1)] = np.where(combined_data_added_features['AwayTeam']==team, combined_data_added_features['tempPRFoulsAgainst'+str(i+1)], combined_data_added_features['awayPRFoulsAgainst'+str(i+1)])
        #combined_data_added_features['awayPRFreeKicksFor'+str(i+1)] = np.where(combined_data_added_features['AwayTeam']==team, combined_data_added_features['tempPRFreeKicksFor'+str(i+1)], combined_data_added_features['awayPRFreeKicksFor'+str(i+1)])
        #combined_data_added_features['awayPRFreeKicksAgainst'+str(i+1)] = np.where(combined_data_added_features['AwayTeam']==team, combined_data_added_features['tempPRFreeKicksAgainst'+str(i+1)], combined_data_added_features['awayPRFreeKicksAgainst'+str(i+1)])
        #combined_data_added_features['awayPROffsidesFor'+str(i+1)] = np.where(combined_data_added_features['AwayTeam']==team, combined_data_added_features['tempPROffsidesFor'+str(i+1)], combined_data_added_features['awayPROffsidesFor'+str(i+1)])
        #combined_data_added_features['awayPROffsidesAgainst'+str(i+1)] = np.where(combined_data_added_features['AwayTeam']==team, combined_data_added_features['tempPROffsidesAgainst'+str(i+1)], combined_data_added_features['awayPROffsidesAgainst'+str(i+1)])
        combined_data_added_features['awayPRYellowsFor'+str(i+1)] = np.where(combined_data_added_features['AwayTeam']==team, combined_data_added_features['tempPRYellowsFor'+str(i+1)], combined_data_added_features['awayPRYellowsFor'+str(i+1)])
        combined_data_added_features['awayPRYellowsAgainst'+str(i+1)] = np.where(combined_data_added_features['AwayTeam']==team, combined_data_added_features['tempPRYellowsAgainst'+str(i+1)], combined_data_added_features['awayPRYellowsAgainst'+str(i+1)])
        combined_data_added_features['awayPRRedsFor'+str(i+1)] = np.where(combined_data_added_features['AwayTeam']==team, combined_data_added_features['tempPRRedsFor'+str(i+1)], combined_data_added_features['awayPRRedsFor'+str(i+1)])
        combined_data_added_features['awayPRRedsAgainst'+str(i+1)] = np.where(combined_data_added_features['AwayTeam']==team, combined_data_added_features['tempPRRedsAgainst'+str(i+1)], combined_data_added_features['awayPRRedsAgainst'+str(i+1)])


    
    # drop the temp columns to make way for the next teams data to be added
    combined_data_added_features = combined_data_added_features.drop(columns_to_join, axis=1)



data_to_inspect = combined_data_added_features.loc[(combined_data_added_features['AwayTeam']=='Kilmarnock') | (combined_data_added_features['HomeTeam']=='Kilmarnock'),:]
combined_data_added_features.to_csv('all_data/combined_data_added_features.csv', index=False)
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


combined_data_added_features.to_csv('all_data/combined_data_added_features.csv', index=False)
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
bookiesFeatures = ['B365Hprob','B365Dprob','B365Aprob','B365bookiesgain']


past_features_to_include = 20
combined_data_added_features_with_history = combined_data_added_features[combined_data_added_features['homePRYear'+str(past_features_to_include)].notnull()]


### setup basic model for testing new features
train_to_season = 2016
probability_cushion = 0.0

# ['seasonEndYear','seasonWeek','homeLongFormNumWins','homeLongFormNumDraws','awayLongFormNumWins','awayLongFormNumDraws','homeShortFormHomeWins', 'homeShortFormHomeDraws', 'homeShortFormHomeLosses', 'awayShortFormAwayWins', 'awayShortFormAwayDraws', 'awayShortFormAwayLosses']
predictors = ['seasonWeek','homeLongFormNumWins','homeLongFormNumDraws','awayLongFormNumWins','awayLongFormNumDraws']+homeWDLfeatures_W[:past_features_to_include]+homeWDLfeatures_D[:past_features_to_include]+homeWDLfeatures_L[:past_features_to_include]+homeGoalsForfeatures[:past_features_to_include]+homeGoalsAgainstfeatures[:past_features_to_include]+homePROppositionPointsfeatures[:past_features_to_include]+awayWDLfeatures_W[:past_features_to_include]+awayWDLfeatures_D[:past_features_to_include]+awayWDLfeatures_L[:past_features_to_include]+awayGoalsForfeatures[:past_features_to_include]+awayGoalsAgainstfeatures[:past_features_to_include]+awayPROppositionPointsfeatures[:past_features_to_include]+divFeatures+homePRIsHomeFeatures+awayPRIsHomeFeatures+bookiesFeatures
train_x = combined_data_added_features_with_history[predictors][combined_data_added_features_with_history['seasonEndYear']<=train_to_season].values
train_col_means = train_x.mean(axis=0)
train_col_stds = train_x.astype(float).std(axis=0)*10
train_x = (train_x - train_col_means)/(train_col_stds)
train_y = (pd.get_dummies(combined_data_added_features_with_history['FTR']))[combined_data_added_features_with_history['seasonEndYear']<=train_to_season].values
test_x = combined_data_added_features_with_history[predictors][combined_data_added_features_with_history['seasonEndYear']==(train_to_season+1)].values
test_x = (test_x - train_col_means)/train_col_stds

input_dimension = len(predictors)

model = Sequential()
#model.add(Dropout(0.2, input_shape=(input_dimension,)))
model.add(Dense(input_dimension, input_dim=input_dimension, activation='relu'))
#model.add(Dropout(0.2))
model.add(Dense(input_dimension, activation='relu'))
#model.add(Dropout(0.2))
model.add(Dense(input_dimension, activation='relu'))
#model.add(Dense(input_dimension, activation='relu'))
#model.add(Dense(input_dimension, input_dim=input_dimension, activation='relu'))
#model.add(Dense(input_dimension, input_dim=input_dimension, activation='relu'))
model.add(Dense(3, activation='softmax'))

optim_rmsprop = optimizers.RMSprop(lr=0.0001, rho=0.9)
optim_sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.5, nesterov=True)
optim_adagrad = optimizers.Adagrad(lr=0.1, decay=0)
optim_adam = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0)

model.compile(loss='categorical_crossentropy',
              optimizer=optim_sgd,
              metrics=['accuracy'])

model.fit(train_x,
          train_y,
          epochs=30,
          batch_size=100,
          validation_split=0.1) 



nn_multi_combined_outcomes = combined_data_added_features_with_history[['Date', 'B365A','B365D','B365H','B365Aprob','B365Dprob','B365Hprob','LBA','LBD','LBH','LBAprob','LBDprob','LBHprob','seasonEndYear','seasonWeek','FTR']][combined_data_added_features_with_history['seasonEndYear']==(train_to_season+1)]
nn_multi_combined_outcomes['Apreds'] = model.predict(test_x)[:,0]
nn_multi_combined_outcomes['Dpreds'] = model.predict(test_x)[:,1]
nn_multi_combined_outcomes['Hpreds'] = model.predict(test_x)[:,2]

probability_cushion = 0.05
nn_multi_combined_outcomes['Abets'] = (nn_multi_combined_outcomes['Apreds']>(1/nn_multi_combined_outcomes['B365A']+probability_cushion)) #& (nn_multi_combined_outcomes['Apreds']>(nn_multi_combined_outcomes['B365Aprob']+probability_cushion))
nn_multi_combined_outcomes['Dbets'] = (nn_multi_combined_outcomes['Dpreds']>(1/nn_multi_combined_outcomes['B365D']+probability_cushion)) #& (nn_multi_combined_outcomes['Dpreds']>(nn_multi_combined_outcomes['B365Dprob']+probability_cushion))
nn_multi_combined_outcomes['Hbets'] = (nn_multi_combined_outcomes['Hpreds']>(1/nn_multi_combined_outcomes['B365H']+probability_cushion)) #& (nn_multi_combined_outcomes['Hpreds']>(nn_multi_combined_outcomes['B365Hprob']+probability_cushion))

#nn_multi_combined_outcomes['Abets'] = (nn_multi_combined_outcomes['Apreds']/(1/nn_multi_combined_outcomes['B365A'])>1+probability_cushion) #& (nn_multi_combined_outcomes['Apreds']>(nn_multi_combined_outcomes['B365Aprob']+probability_cushion))
#nn_multi_combined_outcomes['Dbets'] = (nn_multi_combined_outcomes['Dpreds']/(1/nn_multi_combined_outcomes['B365D'])>1+probability_cushion) #& (nn_multi_combined_outcomes['Dpreds']>(nn_multi_combined_outcomes['B365Dprob']+probability_cushion))
#nn_multi_combined_outcomes['Hbets'] = (nn_multi_combined_outcomes['Hpreds']/(1/nn_multi_combined_outcomes['B365H'])>1+probability_cushion) #& (nn_multi_combined_outcomes['Hpreds']>(nn_multi_combined_outcomes['B365Hprob']+probability_cushion))

nn_multi_combined_outcomes['Awinnings'] = nn_multi_combined_outcomes['Abets']*(nn_multi_combined_outcomes[['B365A','LBA']].max(axis=1))*(nn_multi_combined_outcomes['FTR']=='A')
nn_multi_combined_outcomes['Dwinnings'] = nn_multi_combined_outcomes['Dbets']*(nn_multi_combined_outcomes[['B365D','LBD']].max(axis=1))*(nn_multi_combined_outcomes['FTR']=='D')
nn_multi_combined_outcomes['Hwinnings'] = nn_multi_combined_outcomes['Hbets']*(nn_multi_combined_outcomes[['B365H','LBH']].max(axis=1))*(nn_multi_combined_outcomes['FTR']=='H')

match_days_combined = pd.unique(combined_data_added_features_with_history['Date'][combined_data_added_features_with_history['seasonEndYear']==(train_to_season+1)])
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

nn_multi_results_combined[['Abets','Dbets','Hbets']]

all_home_winnings = ((combined_data_added_features_with_history[['B365H','LBH']][combined_data_added_features_with_history['seasonEndYear']==(train_to_season+1)]).max(axis=1))*((combined_data_added_features_with_history['FTR']=='H')[combined_data_added_features_with_history['seasonEndYear']==(train_to_season+1)])
all_draw_winnings = ((combined_data_added_features_with_history[['B365D','LBD']][combined_data_added_features_with_history['seasonEndYear']==(train_to_season+1)]).max(axis=1))*((combined_data_added_features_with_history['FTR']=='D')[combined_data_added_features_with_history['seasonEndYear']==(train_to_season+1)])
all_away_winnings = ((combined_data_added_features_with_history[['B365A','LBA']][combined_data_added_features_with_history['seasonEndYear']==(train_to_season+1)]).max(axis=1))*((combined_data_added_features_with_history['FTR']=='A')[combined_data_added_features_with_history['seasonEndYear']==(train_to_season+1)])
print('Total bet (all home): ', len(all_home_winnings), 'Winnings: ', sum(all_home_winnings), 'Profit: ', round((sum(all_home_winnings)/len(all_home_winnings)-1)*100,2), '%')
print('Total bet (all draw): ', len(all_draw_winnings), 'Winnings: ', sum(all_draw_winnings), 'Profit: ', round((sum(all_draw_winnings)/len(all_draw_winnings)-1)*100,2), '%')
print('Total bet (all away): ', len(all_away_winnings), 'Winnings: ', sum(all_away_winnings), 'Profit: ', round((sum(all_away_winnings)/len(all_away_winnings)-1)*100,2), '%')

y = 2
in_home_bets = model.predict(combined_data_added_features_with_history[predictors][combined_data_added_features_with_history['seasonEndYear']==(train_to_season-y)].values)[:,2]>(1/(combined_data_added_features_with_history['B365H'][combined_data_added_features_with_history['seasonEndYear']==(train_to_season-y)])+probability_cushion)
in_draw_bets = model.predict(combined_data_added_features_with_history[predictors][combined_data_added_features_with_history['seasonEndYear']==(train_to_season-y)].values)[:,1]>(1/(combined_data_added_features_with_history['B365D'][combined_data_added_features_with_history['seasonEndYear']==(train_to_season-y)])+probability_cushion)
in_away_bets = model.predict(combined_data_added_features_with_history[predictors][combined_data_added_features_with_history['seasonEndYear']==(train_to_season-y)].values)[:,0]>(1/(combined_data_added_features_with_history['B365A'][combined_data_added_features_with_history['seasonEndYear']==(train_to_season-y)])+probability_cushion)
in_home_winnings = in_home_bets*((combined_data_added_features_with_history[['B365H','LBH']][combined_data_added_features_with_history['seasonEndYear']==(train_to_season-y)]).max(axis=1))*((combined_data_added_features_with_history['FTR']=='H')[combined_data_added_features_with_history['seasonEndYear']==(train_to_season-y)])
in_draw_winnings = in_draw_bets*((combined_data_added_features_with_history[['B365D','LBD']][combined_data_added_features_with_history['seasonEndYear']==(train_to_season-y)]).max(axis=1))*((combined_data_added_features_with_history['FTR']=='D')[combined_data_added_features_with_history['seasonEndYear']==(train_to_season-y)])
in_away_winnings = in_away_bets*((combined_data_added_features_with_history[['B365A','LBA']][combined_data_added_features_with_history['seasonEndYear']==(train_to_season-y)]).max(axis=1))*((combined_data_added_features_with_history['FTR']=='A')[combined_data_added_features_with_history['seasonEndYear']==(train_to_season-y)])
print('Total bet (in): ', sum(in_home_bets), 'Winnings: ', sum(in_home_winnings), 'Profit: ', round((sum(in_home_winnings)/sum(in_home_bets)-1)*100,2), '%')
print('Total bet (in): ', sum(in_draw_bets), 'Winnings: ', sum(in_draw_winnings), 'Profit: ', round((sum(in_draw_winnings)/sum(in_draw_bets)-1)*100,2), '%')
print('Total bet (in): ', sum(in_away_bets), 'Winnings: ', sum(in_away_winnings), 'Profit: ', round((sum(in_away_winnings)/sum(in_away_bets)-1)*100,2), '%')


plt.plot(nn_multi_results_combined['cumulativeProfitLoss']*100/sum(nn_multi_results_combined['betsMade']))

# use squared loss to test for overfitting
np.mean((model.predict(combined_data_added_features_with_history[predictors][combined_data_added_features_with_history['seasonEndYear']<=(train_to_season)].values)-(pd.get_dummies(combined_data_added_features_with_history['FTR']))[combined_data_added_features_with_history['seasonEndYear']<=train_to_season].values)**2)
np.mean((model.predict(combined_data_added_features_with_history[predictors][combined_data_added_features_with_history['seasonEndYear']==(train_to_season+1)].values)-(pd.get_dummies(combined_data_added_features_with_history['FTR']))[combined_data_added_features_with_history['seasonEndYear']==(train_to_season+1)].values)**2)















