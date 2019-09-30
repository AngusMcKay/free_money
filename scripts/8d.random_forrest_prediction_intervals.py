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
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

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
bookiesFeatures = ['PSHprob','PSDprob','PSAprob','PSbookiesgain']

# remove rows which don't have sufficient past history
past_features_to_include = 10
combined_data_added_features_with_history = combined_data_added_features[combined_data_added_features['homePRYear'+str(past_features_to_include)].notnull()]

# optional: remove rows without corners history etc
combined_data_added_features_with_history = combined_data_added_features_with_history[combined_data_added_features_with_history[homePRShotsForFeatures+homePRShotsAgainstFeatures+homePRSOTForFeatures+homePRSOTAgainstFeatures+homePRCornersForFeatures+homePRCornersAgainstFeatures+homePRFoulsForFeatures+homePRFoulsAgainstFeatures+homePRYellowsForFeatures+homePRYellowsAgainstFeatures+homePRRedsForFeatures+homePRRedsAgainstFeatures+
                                                                                                                                awayPRShotsForFeatures+awayPRShotsAgainstFeatures+awayPRSOTForFeatures+awayPRSOTAgainstFeatures+awayPRCornersForFeatures+awayPRCornersAgainstFeatures+awayPRFoulsForFeatures+awayPRFoulsAgainstFeatures+awayPRYellowsForFeatures+awayPRYellowsAgainstFeatures+awayPRRedsForFeatures+awayPRRedsAgainstFeatures].isnull().sum(axis=1)==0]


# train fully extended random forest on a few main variables in order to get confidence intervals
train_to_season = 2017
predictors = homeWDLfeatures_W[:past_features_to_include]+homeWDLfeatures_D[:past_features_to_include]+homeWDLfeatures_L[:past_features_to_include]+homeGoalsForfeatures[:past_features_to_include]+homeGoalsAgainstfeatures[:past_features_to_include]+homePROppositionPointsfeatures[:past_features_to_include]+awayWDLfeatures_W[:past_features_to_include]+awayWDLfeatures_D[:past_features_to_include]+awayWDLfeatures_L[:past_features_to_include]+awayGoalsForfeatures[:past_features_to_include]+awayGoalsAgainstfeatures[:past_features_to_include]+awayPROppositionPointsfeatures[:past_features_to_include]#+divFeatures+homePRIsHomeFeatures+awayPRIsHomeFeatures+homePRShotsForFeatures[:past_features_to_include]+homePRShotsAgainstFeatures[:past_features_to_include]+homePRSOTForFeatures[:past_features_to_include]+homePRSOTAgainstFeatures[:past_features_to_include]+homePRFoulsForFeatures[:past_features_to_include]+homePRFoulsAgainstFeatures[:past_features_to_include]+homePRYellowsForFeatures[:past_features_to_include]+homePRYellowsAgainstFeatures[:past_features_to_include]+homePRRedsForFeatures[:past_features_to_include]+homePRRedsAgainstFeatures[:past_features_to_include]+awayPRShotsForFeatures[:past_features_to_include]+awayPRShotsAgainstFeatures[:past_features_to_include]+awayPRSOTForFeatures[:past_features_to_include]+awayPRSOTAgainstFeatures[:past_features_to_include]+awayPRFoulsForFeatures[:past_features_to_include]+awayPRFoulsAgainstFeatures[:past_features_to_include]+awayPRYellowsForFeatures[:past_features_to_include]+awayPRYellowsAgainstFeatures[:past_features_to_include]+awayPRRedsForFeatures[:past_features_to_include]+awayPRRedsAgainstFeatures[:past_features_to_include]+bookiesFeatures
train_x = combined_data_added_features_with_history[predictors][combined_data_added_features_with_history['seasonEndYear']<=train_to_season].values
train_y = combined_data_added_features_with_history['FTR'][combined_data_added_features_with_history['seasonEndYear']<=train_to_season]

test_x = combined_data_added_features_with_history[predictors][combined_data_added_features_with_history['seasonEndYear']==(train_to_season+1)].values
test_y = combined_data_added_features_with_history['FTR'][combined_data_added_features_with_history['seasonEndYear']==(train_to_season+1)].values

rfMod = RandomForestClassifier(n_estimators=100,
                               max_depth=6,
                               min_samples_leaf=10,
                               max_features='sqrt',
                               bootstrap=True,
                               random_state=123)

rfMod.fit(train_x, train_y)

predictions_and_bets = combined_data_added_features_with_history[['Date', 'B365A','B365D','B365H','LBA','LBD','LBH','PSA','PSD','PSH','BWA','BWD','BWH','VCA','VCD','VCH','B365Aprob','B365Dprob','B365Hprob','FTR']][combined_data_added_features_with_history['seasonEndYear']==(train_to_season+1)]

predictions_and_bets['Apreds'] = rfMod.predict_proba(test_x)[:,0]
predictions_and_bets['Dpreds'] = rfMod.predict_proba(test_x)[:,1]
predictions_and_bets['Hpreds'] = rfMod.predict_proba(test_x)[:,2]

# assign bets based on the probability being above a threshold higher than the bookies' probability
probability_margin = 0.05
probability_min = 0.0
probability_max = 1.01
predictions_and_bets['Abets'] = ((predictions_and_bets['Apreds']>(1/predictions_and_bets['PSA']+probability_margin)) & (predictions_and_bets['Apreds']>probability_min) & (predictions_and_bets['Apreds']<probability_max))*1
predictions_and_bets['Dbets'] = ((predictions_and_bets['Dpreds']>(1/predictions_and_bets['PSD']+probability_margin)) & (predictions_and_bets['Dpreds']>probability_min) & (predictions_and_bets['Dpreds']<probability_max))*1
predictions_and_bets['Hbets'] = ((predictions_and_bets['Hpreds']>(1/predictions_and_bets['PSH']+probability_margin)) & (predictions_and_bets['Hpreds']>probability_min) & (predictions_and_bets['Hpreds']<probability_max))*1

# calculate winnings based on lower confidence probability bets
predictions_and_bets['Awinnings'] = predictions_and_bets['Abets']*predictions_and_bets['PSA']*(predictions_and_bets['FTR']=='A')
predictions_and_bets['Dwinnings'] = predictions_and_bets['Dbets']*predictions_and_bets['PSD']*(predictions_and_bets['FTR']=='D')
predictions_and_bets['Hwinnings'] = predictions_and_bets['Hbets']*predictions_and_bets['PSH']*(predictions_and_bets['FTR']=='H')


#print('total bets: ',sum(predictions_and_bets['Abets'])+sum(predictions_and_bets['Dbets'])+sum(predictions_and_bets['Hbets']),
#      'total winnings: ',sum(predictions_and_bets['Awinnings'])+sum(predictions_and_bets['Dwinnings'])+sum(predictions_and_bets['Hwinnings']),
#      'return: ', ((sum(predictions_and_bets['Awinnings'])+sum(predictions_and_bets['Dwinnings'])+sum(predictions_and_bets['Hwinnings']))/(sum(predictions_and_bets['Abets'])+sum(predictions_and_bets['Dbets'])+sum(predictions_and_bets['Hbets']))-1)*100, '%')
print('away bets:',sum(predictions_and_bets['Abets']), 'away winnings:',round(sum(predictions_and_bets['Awinnings']),0), 'return:', round((sum(predictions_and_bets['Awinnings'])/sum(predictions_and_bets['Abets'])-1)*100,2),'%')
print('draw bets:',sum(predictions_and_bets['Dbets']), 'draw winnings:',round(sum(predictions_and_bets['Dwinnings']),0), 'return:', round((sum(predictions_and_bets['Dwinnings'])/sum(predictions_and_bets['Dbets'])-1)*100,2),'%')
print('home bets:',sum(predictions_and_bets['Hbets']), 'home winnings:',round(sum(predictions_and_bets['Hwinnings']),0), 'return:', round((sum(predictions_and_bets['Hwinnings'])/sum(predictions_and_bets['Hbets'])-1)*100,2),'%')





### betting strategies
# 1. bet the same proportion of money on every game that meets the cutoff criteria
predictions_and_bets['Apred_minus_odds_prob'] = predictions_and_bets['Alower_confidence'] - (1/predictions_and_bets['PSA']) - probability_margin
predictions_and_bets['Apred_minus_odds_prob'] = predictions_and_bets['Apred_minus_odds_prob']*(predictions_and_bets['Apred_minus_odds_prob']>0)
predictions_and_bets['Dpred_minus_odds_prob'] = predictions_and_bets['Dlower_confidence'] - (1/predictions_and_bets['PSD']) - probability_margin
predictions_and_bets['Dpred_minus_odds_prob'] = predictions_and_bets['Dpred_minus_odds_prob']*(predictions_and_bets['Dpred_minus_odds_prob']>0)
predictions_and_bets['Hpred_minus_odds_prob'] = predictions_and_bets['Hlower_confidence'] - (1/predictions_and_bets['PSH']) - probability_margin
predictions_and_bets['Hpred_minus_odds_prob'] = predictions_and_bets['Hpred_minus_odds_prob']*(predictions_and_bets['Hpred_minus_odds_prob']>0)

bets_scaling = 1/50
predictions_and_bets['Aproportion_bet'] = predictions_and_bets['Apred_minus_odds_prob']*bets_scaling/predictions_and_bets['Apred_minus_odds_prob']
predictions_and_bets['Dproportion_bet'] = predictions_and_bets['Dpred_minus_odds_prob']*bets_scaling/predictions_and_bets['Dpred_minus_odds_prob']
predictions_and_bets['Hproportion_bet'] = predictions_and_bets['Hpred_minus_odds_prob']*bets_scaling/predictions_and_bets['Hpred_minus_odds_prob']

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



# 2. bet amounts based on probability of losing (to limit drawdown), e.g. want less than 1% chance of losing 10% through consecutive losing bets
# e.g. if 0.9 chance of losing then 0.9**10 = 0.34 chance of losing 10 in a row
# so want to bet (10/x)% where x is solution to 0.9**x = 0.01, i.e. prop_bet = 10/(log(0.01)/log(0.9))
max_drawdown = 0.1
prob_max_drawdown = 0.01

predictions_and_bets['Apred_minus_odds_prob'] = predictions_and_bets['Alower_confidence'] - (1/predictions_and_bets['PSA']) - probability_margin
predictions_and_bets['Apred_minus_odds_prob'] = predictions_and_bets['Apred_minus_odds_prob']*(predictions_and_bets['Apred_minus_odds_prob']>0)
predictions_and_bets['Dpred_minus_odds_prob'] = predictions_and_bets['Dlower_confidence'] - (1/predictions_and_bets['PSD']) - probability_margin
predictions_and_bets['Dpred_minus_odds_prob'] = predictions_and_bets['Dpred_minus_odds_prob']*(predictions_and_bets['Dpred_minus_odds_prob']>0)
predictions_and_bets['Hpred_minus_odds_prob'] = predictions_and_bets['Hlower_confidence'] - (1/predictions_and_bets['PSH']) - probability_margin
predictions_and_bets['Hpred_minus_odds_prob'] = predictions_and_bets['Hpred_minus_odds_prob']*(predictions_and_bets['Hpred_minus_odds_prob']>0)

predictions_and_bets['Aproportion_bet'] = (predictions_and_bets['Apred_minus_odds_prob']>0)*max_drawdown/(np.log(prob_max_drawdown)/np.log(1-predictions_and_bets['Alower_confidence']))
predictions_and_bets['Dproportion_bet'] = (predictions_and_bets['Dpred_minus_odds_prob']>0)*max_drawdown/(np.log(prob_max_drawdown)/np.log(1-predictions_and_bets['Dlower_confidence']))
predictions_and_bets['Hproportion_bet'] = (predictions_and_bets['Hpred_minus_odds_prob']>0)*max_drawdown/(np.log(prob_max_drawdown)/np.log(1-predictions_and_bets['Hlower_confidence']))

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




# 3. bet max(0,x/c )% of total money where x = prediction - odds based probability
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



# 4. bet max(0,x/c )% of total money where x = prediction - odds based probability + probability_margin
predictions_and_bets['Apred_minus_odds_prob'] = predictions_and_bets['Alower_confidence'] - (1/predictions_and_bets['PSA']) - probability_margin
predictions_and_bets['Apred_minus_odds_prob'] = predictions_and_bets['Apred_minus_odds_prob']*(predictions_and_bets['Apred_minus_odds_prob']>0)
predictions_and_bets['Dpred_minus_odds_prob'] = predictions_and_bets['Dlower_confidence'] - (1/predictions_and_bets['PSD']) - probability_margin
predictions_and_bets['Dpred_minus_odds_prob'] = predictions_and_bets['Dpred_minus_odds_prob']*(predictions_and_bets['Dpred_minus_odds_prob']>0)
predictions_and_bets['Hpred_minus_odds_prob'] = predictions_and_bets['Hlower_confidence'] - (1/predictions_and_bets['PSH']) - probability_margin
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








