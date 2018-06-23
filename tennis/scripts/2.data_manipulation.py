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


# read data
mens_data = pd.read_csv('all_data/mens_data.csv')
womens_data = pd.read_csv('all_data/womens_data.csv')

# one-hot encode a couple of features
mens_data = pd.concat([mens_data, pd.get_dummies(mens_data['Surface'])], axis=1)
mens_data = pd.concat([mens_data, pd.get_dummies(mens_data['Series'])], axis=1)
womens_data = pd.concat([womens_data, pd.get_dummies(womens_data['Surface'])], axis=1)
womens_data = pd.concat([womens_data, pd.get_dummies(womens_data['Tier'])], axis=1)

# remove rows where ranks not available
mens_data_reformatted = mens_data[mens_data['WRank'].notnull()]
mens_data_reformatted = mens_data_reformatted[mens_data_reformatted['LRank'].notnull()]
womens_data_reformatted = womens_data[womens_data['WRank'].notnull()]
womens_data_reformatted = womens_data_reformatted[womens_data_reformatted['LRank'].notnull()]
#mens_data_reformatted = mens_data_reformatted[mens_data_reformatted['WRank']!='NR']
mens_data_reformatted = mens_data_reformatted[mens_data_reformatted['LRank']!='NR']
#womens_data_reformatted = womens_data_reformatted[womens_data_reformatted['WRank']!='NR']
#womens_data_reformatted = womens_data_reformatted[womens_data_reformatted['LRank']!='NR']

mens_data_reformatted['LRank'] = pd.to_numeric(mens_data_reformatted['LRank'])

# edit the data towards predicting if the higher ranked player will win
loser_features = ['Loser', 'LRank','LPts','Lsets','L1','L2','L3','L4','L5','B365L','B365Lprob']
winner_features = ['Winner','WRank','WPts','Wsets','W1','W2','W3','W4','W5','B365W','B365Wprob']
player1features = ['player1','p1Rank','p1Pts','p1sets','p1games1','p1games2','p1games3','p1games4','p1games5','B365P1','B365P1prob']
player2features = ['player2','p2Rank','p2Pts','p2sets','p2games1','p2games2','p2games3','p2games4','p2games5','B365P2','B365P2prob']
for i in range(len(loser_features)):
    mens_data_reformatted[player1features[i]] = np.where(mens_data_reformatted['LRank'] < mens_data_reformatted['WRank'],
                         mens_data_reformatted[loser_features[i]],
                         mens_data_reformatted[winner_features[i]])
    mens_data_reformatted[player2features[i]] = np.where(mens_data_reformatted['LRank'] >= mens_data_reformatted['WRank'],
                         mens_data_reformatted[loser_features[i]],
                         mens_data_reformatted[winner_features[i]])

reformatted_features = ['ATP','Best of','Comment','Date','Location','Winner','Round',
                        'Carpet','Clay','Grass','Hard',
                        'ATP250','ATP500','Grand Slam','International','International Gold','Masters','Masters 1000','Masters Cup',
                        'player1','p1Rank','p1Pts','p1sets','p1games1','p1games2','p1games3','p1games4','p1games5','B365P1','B365P1prob',
                        'player2','p2Rank','p2Pts','p2sets','p2games1','p2games2','p2games3','p2games4','p2games5','B365P2','B365P2prob',
                        'B365bookiesgain']

mens_data_reformatted = mens_data_reformatted[reformatted_features]
test_data_to_inspect = mens_data_reformatted.iloc[:1000,:]

mens_data_reformatted['player1Wins'] = np.where(mens_data_reformatted['player1'] == mens_data_reformatted['Winner'],1,0)


mens_data_reformatted.to_csv('all_data/mens_data_reformatted.csv', index=False)









