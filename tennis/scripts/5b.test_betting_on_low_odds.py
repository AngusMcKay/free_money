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
import matplotlib.pyplot as plt



mens_data_sql_processed_with_history = pd.read_csv('all_data/mens_data_sql_processed_with_history.csv')


data_to_view = mens_data_sql_processed_with_history.iloc[:10,:]

mens_data_sql_processed_with_history['b365P1']
mens_data_sql_processed_with_history['player1Wins']


# set some parameters to bet on
min_odds = 1.0001
max_odds = 1.1
max_bookies_gain = 1.1
#start_date = '2000-01-01'

p1_bets = mens_data_sql_processed_with_history[(mens_data_sql_processed_with_history['b365P1']<max_odds) &
                                               (mens_data_sql_processed_with_history['b365P1']>min_odds) &
                                               (mens_data_sql_processed_with_history['b365bookiesgain']<max_bookies_gain)]

print('P1 bets: £',p1_bets.shape[0] , ', P1 winnings: £',sum(p1_bets['b365P1'][p1_bets['player1Wins']==1]), sep='')


p2_bets = mens_data_sql_processed_with_history[(mens_data_sql_processed_with_history['b365P2']<max_odds) &
                                               (mens_data_sql_processed_with_history['b365P2']>min_odds) &
                                               (mens_data_sql_processed_with_history['b365bookiesgain']<max_bookies_gain)]

print('P2 bets: £',p2_bets.shape[0] , ', P2 winnings: £',sum(p2_bets['b365P2'][p2_bets['player1Wins']==0]), sep='')



# find surfce as a function of odds band and bookies gain
odds_cutoffs = [1+x*0.1 for x in range(81)]
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
        p1_bets = mens_data_sql_processed_with_history[(mens_data_sql_processed_with_history['b365P1']<max_odds) &
                                                      (mens_data_sql_processed_with_history['b365P1']>min_odds) &
                                                      (mens_data_sql_processed_with_history['b365bookiesgain']<max_bookies_gain) &
                                                      (mens_data_sql_processed_with_history['b365bookiesgain']>min_bookies_gain)]
        p1_bets_placed=p1_bets.shape[0]
        p1_winnings=sum(p1_bets['b365P1'][p1_bets['player1Wins']==1])
        
        p2_bets = mens_data_sql_processed_with_history[(mens_data_sql_processed_with_history['b365P2']<max_odds) &
                                                      (mens_data_sql_processed_with_history['b365P2']>min_odds) &
                                                      (mens_data_sql_processed_with_history['b365bookiesgain']<max_bookies_gain) &
                                                      (mens_data_sql_processed_with_history['b365bookiesgain']>min_bookies_gain)]
        p2_bets_placed=p2_bets.shape[0]
        p2_winnings=sum(p2_bets['b365P2'][p2_bets['player1Wins']==0])
        
        if (p1_bets_placed+p2_bets_placed)==0:
            pc_profit=0
        else:
            pc_profit=(p1_winnings+p2_winnings)/(p1_bets_placed+p2_bets_placed)-1
        output.iat[i,j]=pc_profit
        
        print('odds range: ', min_odds, ' to ', max_odds, ', bookies gain range: ', min_bookies_gain, ' to ', max_bookies_gain, sep='')
        print('p1 bets: £',p1_bets.shape[0] , ', p1 winnings: £',sum(p1_bets['b365P1'][p1_bets['player1Wins']==1]), sep='')
        print('p2 bets: £',p2_bets.shape[0] , ', p2 winnings: £',sum(p2_bets['b365P2'][p2_bets['player1Wins']==0]), sep='')

sns.scatterplot(x=odds_cutoffs, y=output.iloc[:,0])



