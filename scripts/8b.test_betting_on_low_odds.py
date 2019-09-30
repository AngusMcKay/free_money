#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 14:12:00 2019

@author: angus
"""

import os
os.chdir('/home/angus/projects/betting')

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from scipy.stats import binom

# read data
combined_data_added_features = pd.read_csv('all_data/combined_data_added_features.csv')


# remove rows which don't have sufficient past history
past_features_to_include = 1
past_features_to_include_alternative = 1
combined_data_added_features_with_history = combined_data_added_features[combined_data_added_features['homePRYear'+str(past_features_to_include)].notnull()]

## optional: remove rows without corners history etc
#combined_data_added_features_with_history = combined_data_added_features_with_history[combined_data_added_features_with_history[homePRShotsForFeatures+homePRShotsAgainstFeatures+homePRSOTForFeatures+homePRSOTAgainstFeatures+homePRCornersForFeatures+homePRCornersAgainstFeatures+homePRFoulsForFeatures+homePRFoulsAgainstFeatures+homePRYellowsForFeatures+homePRYellowsAgainstFeatures+homePRRedsForFeatures+homePRRedsAgainstFeatures+
#                                                                                                                                awayPRShotsForFeatures+awayPRShotsAgainstFeatures+awayPRSOTForFeatures+awayPRSOTAgainstFeatures+awayPRCornersForFeatures+awayPRCornersAgainstFeatures+awayPRFoulsForFeatures+awayPRFoulsAgainstFeatures+awayPRYellowsForFeatures+awayPRYellowsAgainstFeatures+awayPRRedsForFeatures+awayPRRedsAgainstFeatures].isnull().sum(axis=1)==0]

data_to_view = combined_data_added_features_with_history.iloc[:10,:]


# set some parameters to bet on
min_odds = 1.0001
max_odds = 1.1
max_bookies_gain = 1.1
start_date = 2000

home_bets = combined_data_added_features_with_history[(combined_data_added_features_with_history['PSH']<max_odds) &
                                                      (combined_data_added_features_with_history['PSH']>min_odds) &
                                                      (combined_data_added_features_with_history['PSbookiesgain']<max_bookies_gain) &
                                                      (combined_data_added_features_with_history['seasonEndYear']>start_date)]

print('Home bets: £',home_bets.shape[0] , ', home winnings: £',sum(home_bets['PSH'][home_bets['FTR']=='H']), sep='')


draw_bets = combined_data_added_features_with_history[(combined_data_added_features_with_history['PSD']<max_odds) &
                                                      (combined_data_added_features_with_history['PSD']>min_odds) &
                                                      (combined_data_added_features_with_history['PSbookiesgain']<max_bookies_gain) &
                                                      (combined_data_added_features_with_history['seasonEndYear']>start_date)]

print('Draw bets: £',draw_bets.shape[0] , ', draw winnings: £',sum(draw_bets['PSD'][draw_bets['FTR']=='D']), sep='')


away_bets = combined_data_added_features_with_history[(combined_data_added_features_with_history['PSA']<max_odds) &
                                                      (combined_data_added_features_with_history['PSA']>min_odds) &
                                                      (combined_data_added_features_with_history['PSbookiesgain']<max_bookies_gain) &
                                                      (combined_data_added_features_with_history['seasonEndYear']>start_date)]

print('Away bets: £',away_bets.shape[0] , ', away winnings: £',sum(away_bets['PSA'][away_bets['FTR']=='A']), sep='')




# find surfce as a function of odds band and bookies gain
odds_cutoffs = [1+x*0.05 for x in range(81)]
bookiesgain_cutoffs = [1+x*2 for x in range(2)]
output=pd.DataFrame(index=odds_cutoffs, columns=bookiesgain_cutoffs)
for i in tqdm(range(len(odds_cutoffs))):
    for j in range(len(bookiesgain_cutoffs)):
        # set params
        min_odds = odds_cutoffs[i]
        if i+1 < len(odds_cutoffs):
            max_odds = odds_cutoffs[i+1]
        else:
            max_odds = odds_cutoffs[i]*2
        
        min_bookies_gain = bookiesgain_cutoffs[j]
        if j+1 < len(bookiesgain_cutoffs):
            max_bookies_gain = bookiesgain_cutoffs[j+1]
        else:
            max_bookies_gain = bookiesgain_cutoffs[j]*2
        
        # calculate returns and populate output df
        home_bets = combined_data_added_features_with_history[(combined_data_added_features_with_history['PSH']<max_odds) &
                                                      (combined_data_added_features_with_history['PSH']>min_odds) &
                                                      (combined_data_added_features_with_history['PSbookiesgain']<max_bookies_gain) &
                                                      (combined_data_added_features_with_history['PSbookiesgain']>min_bookies_gain)]
        home_bets_placed=home_bets.shape[0]
        home_winnings=sum(home_bets['PSH'][home_bets['FTR']=='H'])
        
        away_bets = combined_data_added_features_with_history[(combined_data_added_features_with_history['PSA']<max_odds) &
                                                      (combined_data_added_features_with_history['PSA']>min_odds) &
                                                      (combined_data_added_features_with_history['PSbookiesgain']<max_bookies_gain) &
                                                      (combined_data_added_features_with_history['PSbookiesgain']>min_bookies_gain)]
        away_bets_placed=away_bets.shape[0]
        away_winnings=sum(away_bets['PSA'][away_bets['FTR']=='A'])
        
        if (home_bets_placed+away_bets_placed)==0:
            pc_profit=0
        else:
            pc_profit=(home_winnings+away_winnings)/(home_bets_placed+away_bets_placed)-1
        output.iat[i,j]=pc_profit
        
        print('odds range: ', min_odds, ' to ', max_odds, ', bookies gain range: ', min_bookies_gain, ' to ', max_bookies_gain, sep='')
        print('Home bets: £',home_bets.shape[0] , ', home winnings: £',sum(home_bets['PSH'][home_bets['FTR']=='H']), sep='')
        print('Away bets: £',away_bets.shape[0] , ', away winnings: £',sum(away_bets['PSA'][away_bets['FTR']=='A']), sep='')

sns.scatterplot(x=odds_cutoffs, y=output.iloc[:,0])



# volatility calcs
expected_return = 1.02
max_odds = 1.05
bernoulli_p = 1/max_odds
n = 20

x=np.arange(binom.ppf(0.01, n, bernoulli_p), binom.ppf(0.99, n, bernoulli_p))
x=np.arange(0, 21)
sns.scatterplot(x=x, y=binom.pmf(x, n, bernoulli_p))



# do random projections based on betting set % of winnings each time
n=10
payout=max_odds*expected_return # fix this to give average of 2% return assuminng max odds are correct (which they roughly are)
weekly_wins=binom.rvs(n, bernoulli_p, size=104)
weekly_returns=weekly_wins*payout/n
cumulative_returns = [np.prod(weekly_returns[0:(week+1)]) for week in range(len(weekly_returns))]
sns.scatterplot(x=range(len(weekly_returns)),y=cumulative_returns)




# test picking different bookies: B365, BW, LB, VC
min_odds = 1.0001
max_odds = 1.1
max_bookies_gain = 1.1
start_date = 2000

home_bets = combined_data_added_features_with_history[(combined_data_added_features_with_history['VCH']<max_odds) &
                                                      (combined_data_added_features_with_history['VCH']>min_odds) &
                                                      (combined_data_added_features_with_history['VCbookiesgain']<max_bookies_gain) &
                                                      (combined_data_added_features_with_history['seasonEndYear']>start_date)]

print('Home bets: £',home_bets.shape[0] , ', home winnings: £',sum(home_bets['VCH'][home_bets['FTR']=='H']), sep='')


draw_bets = combined_data_added_features_with_history[(combined_data_added_features_with_history['VCD']<max_odds) &
                                                      (combined_data_added_features_with_history['VCD']>min_odds) &
                                                      (combined_data_added_features_with_history['VCbookiesgain']<max_bookies_gain) &
                                                      (combined_data_added_features_with_history['seasonEndYear']>start_date)]

print('Draw bets: £',draw_bets.shape[0] , ', draw winnings: £',sum(draw_bets['VCD'][draw_bets['FTR']=='D']), sep='')


away_bets = combined_data_added_features_with_history[(combined_data_added_features_with_history['VCA']<max_odds) &
                                                      (combined_data_added_features_with_history['VCA']>min_odds) &
                                                      (combined_data_added_features_with_history['VCbookiesgain']<max_bookies_gain) &
                                                      (combined_data_added_features_with_history['seasonEndYear']>start_date)]

print('Away bets: £',away_bets.shape[0] , ', away winnings: £',sum(away_bets['VCA'][away_bets['FTR']=='A']), sep='')



# try with best odds
combined_data_added_features_with_history['MAXH']=combined_data_added_features_with_history[['B365H', 'BWH', 'LBH', 'VCH']].max(axis=1)
combined_data_added_features_with_history['MINH']=combined_data_added_features_with_history[['B365H', 'BWH', 'LBH', 'VCH']].min(axis=1)
combined_data_added_features_with_history['MAXD']=combined_data_added_features_with_history[['B365D', 'BWD', 'LBD', 'VCD']].max(axis=1)
combined_data_added_features_with_history['MIND']=combined_data_added_features_with_history[['B365D', 'BWD', 'LBD', 'VCD']].min(axis=1)
combined_data_added_features_with_history['MAXA']=combined_data_added_features_with_history[['B365A', 'BWA', 'LBA', 'VCA']].max(axis=1)
combined_data_added_features_with_history['MINA']=combined_data_added_features_with_history[['B365A', 'BWA', 'LBA', 'VCA']].min(axis=1)

min_odds = 1.0001
max_odds = 1.05
max_bookies_gain = 1.1
start_date = 2000

home_bets = combined_data_added_features_with_history[(combined_data_added_features_with_history['MINH']<max_odds) &
                                                      (combined_data_added_features_with_history['MINH']>min_odds) &
                                                      (combined_data_added_features_with_history['PSbookiesgain']<max_bookies_gain) &
                                                      (combined_data_added_features_with_history['seasonEndYear']>start_date)]

print('Home bets: £',home_bets.shape[0] , ', home winnings: £',sum(home_bets['MAXH'][home_bets['FTR']=='H']), sep='')


draw_bets = combined_data_added_features_with_history[(combined_data_added_features_with_history['MIND']<max_odds) &
                                                      (combined_data_added_features_with_history['MIND']>min_odds) &
                                                      (combined_data_added_features_with_history['PSbookiesgain']<max_bookies_gain) &
                                                      (combined_data_added_features_with_history['seasonEndYear']>start_date)]

print('Draw bets: £',draw_bets.shape[0] , ', draw winnings: £',sum(draw_bets['MAXD'][draw_bets['FTR']=='D']), sep='')


away_bets = combined_data_added_features_with_history[(combined_data_added_features_with_history['MINA']<max_odds) &
                                                      (combined_data_added_features_with_history['MINA']>min_odds) &
                                                      (combined_data_added_features_with_history['PSbookiesgain']<max_bookies_gain) &
                                                      (combined_data_added_features_with_history['seasonEndYear']>start_date)]

print('Away bets: £',away_bets.shape[0] , ', away winnings: £',sum(away_bets['MAXA'][away_bets['FTR']=='A']), sep='')

