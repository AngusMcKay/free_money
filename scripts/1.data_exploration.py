#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 18:13:25 2018

@author: angus
"""

import os
os.chdir('/home/angus/projects/betting')

import numpy as np
import pandas as pd
from tqdm import tqdm
import xgboost as xgb
import datetime
import glob
import matplotlib.pyplot as plt




#-----read data-----#
# prem
prem_files = glob.glob("premiership_data/*")
prem_files.sort(reverse=True)
rd_prem = pd.concat(pd.read_csv(file) for file in prem_files[0:13])
#nb from 0405 had to delete commas to get file to work - ignore these seasons for now (also coincides with when the Championship began)

# champ
champ_files = glob.glob("championship_data/*")
champ_files.sort(reverse=True)
rd_champ = pd.concat(pd.read_csv(file) for file in champ_files[0:13])

# league 1
league1_files = glob.glob("league1_data/*")
league1_files.sort(reverse=True)
rd_league1 = pd.concat(pd.read_csv(file) for file in league1_files[0:13])

# league 2
league2_files = glob.glob("league2_data/*")
league2_files.sort(reverse=True)
rd_league2 = pd.concat(pd.read_csv(file) for file in league2_files[0:13])

# conf
conf_files = glob.glob("conference_data/*")
conf_files.sort(reverse=True)
rd_conf1 = pd.concat(pd.read_csv(file) for file in conf_files[0:1])
rd_conf2 = pd.concat(pd.read_csv(file) for file in conf_files[2:13])
rd_conf = pd.concat((rd_conf1, rd_conf2))
del(rd_conf1, rd_conf2)
del(prem_files, champ_files, league1_files, league2_files, conf_files)






#-----data inspection-----#
### MISSING VALUES
# prem
rd_prem.loc[:,:].isnull().sum(axis=0)
sum(rd_prem.notnull().sum(axis=1)==0)
# one line is all na so remove this line
rd_prem = rd_prem.loc[rd_prem.notnull().sum(axis=1)!=0,:]
prem_missing_data = rd_prem.loc[:,:].isnull().sum(axis=0)
# note: VC bet has odds for all games, B365 has all, Ladbrokes (LB) has all but 1, Bet&Win (BW) all but 1, Interwetten (IW) all but 2, WH missin 107
# game stats seem to be complete (goals, fouls, corners etc)

# champ
rd_champ.loc[:,:].isnull().sum(axis=0)
sum(rd_champ.notnull().sum(axis=1)==0)
# one line is all na so remove this line
rd_champ = rd_champ.loc[rd_champ.notnull().sum(axis=1)!=0,:]
champ_missing_data = rd_champ.loc[:,:].isnull().sum(axis=0)
# note: VC bet missing 1 game, B365 has all, Ladbrokes (LB) has all, Bet&Win (BW) has all, Interwetten (IW) all but 16, WH missin 156
# game stats seem to be complete (goals, fouls, corners etc)

# league 1
rd_league1.loc[:,:].isnull().sum(axis=0)
sum(rd_league1.notnull().sum(axis=1)==0)
# two lines all na so remove these lines
rd_league1 = rd_league1.loc[rd_league1.notnull().sum(axis=1)!=0,:]
league1_missing_data = rd_league1.loc[:,:].isnull().sum(axis=0)
# note: VC bet missing 3 game, B365 has all, Ladbrokes (LB) has all but 1, Bet&Win (BW) missing 13, Interwetten (IW) all but 25, WH missin 156
# game stats seem to be complete (goals, fouls, corners etc)

# league 2
rd_league2.loc[:,:].isnull().sum(axis=0)
sum(rd_league2.notnull().sum(axis=1)==0)
# three lines all na so remove these lines
rd_league2 = rd_league2.loc[rd_league2.notnull().sum(axis=1)!=0,:]
league2_missing_data = rd_league2.loc[:,:].isnull().sum(axis=0)
# note: VC bet missing 2, B365 has all, Ladbrokes (LB) has all, Bet&Win (BW) missing 15, Interwetten (IW) missing 30, WH missin 158
# game stats seem to be complete (goals, fouls, corners etc)

# conference
rd_conf.loc[:,:].isnull().sum(axis=0)
sum(rd_conf.notnull().sum(axis=1)==0)
# ten lines all na so remove these lines
rd_conf = rd_conf.loc[rd_conf.notnull().sum(axis=1)!=0,:]
conf_missing_data = rd_conf.loc[:,:].isnull().sum(axis=0)
# note: VC bet missing 58, B365 missing 55, Ladbrokes (LB) missing 87, Bet&Win (BW) missing 68, Interwetten (IW) missing 166, WH missin 266
# results all there but missing some 419 corners, fouls etc


### check values sensible
# prem stats
pd.unique(rd_prem['HomeTeam']), pd.unique(rd_prem['AwayTeam']), pd.unique(rd_prem['FTR']), pd.unique(rd_prem['HTR'])
pd.unique(rd_prem['FTHG']), pd.unique(rd_prem['FTAG']), pd.unique(rd_prem['HTHG']), pd.unique(rd_prem['HTAG'])
pd.unique(rd_prem['HS']), pd.unique(rd_prem['AS']), pd.unique(rd_prem['HST']), pd.unique(rd_prem['AST'])
pd.unique(rd_prem['HC']), pd.unique(rd_prem['AC']), pd.unique(rd_prem['HF']), pd.unique(rd_prem['AF'])
pd.unique(rd_prem['HY']), pd.unique(rd_prem['AY']), pd.unique(rd_prem['HR']), pd.unique(rd_prem['AR'])

# champ stats
pd.unique(rd_champ['HomeTeam']), pd.unique(rd_champ['AwayTeam']), pd.unique(rd_champ['FTR']), pd.unique(rd_champ['HTR'])
pd.unique(rd_champ['FTHG']), pd.unique(rd_champ['FTAG']), pd.unique(rd_champ['HTHG']), pd.unique(rd_champ['HTAG'])
pd.unique(rd_champ['HS']), pd.unique(rd_champ['AS']), pd.unique(rd_champ['HST']), pd.unique(rd_champ['AST'])
pd.unique(rd_champ['HC']), pd.unique(rd_champ['AC']), pd.unique(rd_champ['HF']), pd.unique(rd_champ['AF'])
pd.unique(rd_champ['HY']), pd.unique(rd_champ['AY']), pd.unique(rd_champ['HR']), pd.unique(rd_champ['AR'])






#-----imputation-----#
## set LB odds as B365 odds where it is missing
rd_prem['LBA']=np.where(rd_prem['LBA'].isnull(), rd_prem['B365A'], rd_prem['LBA'])
rd_prem['LBD']=np.where(rd_prem['LBD'].isnull(), rd_prem['B365D'], rd_prem['LBD'])
rd_prem['LBH']=np.where(rd_prem['LBH'].isnull(), rd_prem['B365H'], rd_prem['LBH'])

# one 0 return for B365 home result, impute ladbrokes odds here
rd_champ['B365H']=np.where(rd_champ['B365H']==0, rd_champ['LBH'], rd_champ['B365H'])




#-----comparing odds-----#
# prem
rd_prem.loc[:,['B365H','B365A', 'LBH','LBA','VCH','VCA']].max(axis=0), rd_prem.loc[:,['B365H','B365A', 'LBH','LBA','VCH','VCA']].min(axis=0)
np.mean(rd_prem['B365H']/rd_prem['LBH']), np.mean(rd_prem['B365H']/rd_prem['VCH']), np.mean(rd_prem['LBH']/rd_prem['VCH'])
np.max(rd_prem['B365H']/rd_prem['LBH']), np.max(rd_prem['B365H']/rd_prem['VCH']), np.max(rd_prem['LBH']/rd_prem['VCH'])
np.min(rd_prem['B365H']/rd_prem['LBH']), np.min(rd_prem['B365H']/rd_prem['VCH']), np.min(rd_prem['LBH']/rd_prem['VCH'])
# some biggish difference but on the whole quite similar
# is it possible to take advantage of arbitrage???

np.mean(rd_champ['B365H']/rd_champ['LBH']), np.mean(rd_champ['B365H']/rd_champ['VCH']), np.mean(rd_champ['LBH']/rd_champ['VCH'])
np.max(rd_champ['B365H']/rd_champ['LBH']), np.max(rd_champ['B365H']/rd_champ['VCH']), np.max(rd_champ['LBH']/rd_champ['VCH'])
np.min(rd_champ['B365H']/rd_champ['LBH']), np.min(rd_champ['B365H']/rd_champ['VCH']), np.min(rd_champ['LBH']/rd_champ['VCH'])






#----- adding useful columns -----#
rd_prem['Date']=rd_prem['Date'].map(lambda x: datetime.datetime.strptime(x, '%d/%m/%y'))
rd_prem['seasonEndYear']=rd_prem['Date'].map(lambda x: x.year + (x.month>=7)*1)

rd_champ['Date']=rd_champ['Date'].map(lambda x: datetime.datetime.strptime(x, '%d/%m/%y'))
rd_champ['seasonEndYear']=rd_champ['Date'].map(lambda x: x.year + (x.month>=7)*1)








#-----adding probabilities based on odds-----#
# prem
rd_prem['B365Hprob'] = 1/rd_prem['B365H']/(1/rd_prem['B365H']+1/rd_prem['B365D']+1/rd_prem['B365A'])
rd_prem['B365Dprob'] = 1/rd_prem['B365D']/(1/rd_prem['B365H']+1/rd_prem['B365D']+1/rd_prem['B365A'])
rd_prem['B365Aprob'] = 1/rd_prem['B365A']/(1/rd_prem['B365H']+1/rd_prem['B365D']+1/rd_prem['B365A'])
rd_prem['B365bookiesgain'] = (1/rd_prem['B365H']+1/rd_prem['B365D']+1/rd_prem['B365A'])
np.mean(rd_prem['B365bookiesgain'])
np.min(rd_prem['B365bookiesgain'])
np.max(rd_prem['B365bookiesgain'])
np.mean(rd_prem['B365bookiesgain'][rd_prem['seasonEndYear']==2017])
np.min(rd_prem['B365bookiesgain'][rd_prem['seasonEndYear']==2017])
np.max(rd_prem['B365bookiesgain'][rd_prem['seasonEndYear']==2017])


rd_prem['LBHprob'] = 1/rd_prem['LBH']/(1/rd_prem['LBH']+1/rd_prem['LBD']+1/rd_prem['LBA'])
rd_prem['LBDprob'] = 1/rd_prem['LBD']/(1/rd_prem['LBH']+1/rd_prem['LBD']+1/rd_prem['LBA'])
rd_prem['LBAprob'] = 1/rd_prem['LBA']/(1/rd_prem['LBH']+1/rd_prem['LBD']+1/rd_prem['LBA'])
rd_prem['LBbookiesgain'] = (1/rd_prem['LBH']+1/rd_prem['LBD']+1/rd_prem['LBA'])
np.mean(rd_prem['LBbookiesgain'])
np.min(rd_prem['LBbookiesgain'])
np.max(rd_prem['LBbookiesgain'])
np.mean(rd_prem['LBbookiesgain'][rd_prem['seasonEndYear']==2017])
np.min(rd_prem['LBbookiesgain'][rd_prem['seasonEndYear']==2017])
np.max(rd_prem['LBbookiesgain'][rd_prem['seasonEndYear']==2017])


# champ
rd_champ['B365Hprob'] = 1/rd_champ['B365H']/(1/rd_champ['B365H']+1/rd_champ['B365D']+1/rd_champ['B365A'])
rd_champ['B365Dprob'] = 1/rd_champ['B365D']/(1/rd_champ['B365H']+1/rd_champ['B365D']+1/rd_champ['B365A'])
rd_champ['B365Aprob'] = 1/rd_champ['B365A']/(1/rd_champ['B365H']+1/rd_champ['B365D']+1/rd_champ['B365A'])
rd_champ['B365bookiesgain'] = (1/rd_champ['B365H']+1/rd_champ['B365D']+1/rd_champ['B365A'])
np.mean(rd_champ['B365bookiesgain'])
np.min(rd_champ['B365bookiesgain'])
np.max(rd_champ['B365bookiesgain'])

rd_champ['LBHprob'] = 1/rd_champ['LBH']/(1/rd_champ['LBH']+1/rd_champ['LBD']+1/rd_champ['LBA'])
rd_champ['LBDprob'] = 1/rd_champ['LBD']/(1/rd_champ['LBH']+1/rd_champ['LBD']+1/rd_champ['LBA'])
rd_champ['LBAprob'] = 1/rd_champ['LBA']/(1/rd_champ['LBH']+1/rd_champ['LBD']+1/rd_champ['LBA'])
rd_champ['LBbookiesgain'] = (1/rd_champ['LBH']+1/rd_champ['LBD']+1/rd_champ['LBA'])
np.mean(rd_champ['LBbookiesgain'])
np.min(rd_champ['LBbookiesgain'])
np.max(rd_champ['LBbookiesgain'])












#----- compare expected vs actual home/away/long odds -----#
def exp_vs_act_wins(probs, results):
    print('expected: ', sum(probs), ', actual: ', sum(results))

exp_vs_act_wins(rd_prem['B365Hprob'], rd_prem['FTR']=='H')
exp_vs_act_wins(rd_prem['B365Aprob'], rd_prem['FTR']=='A')
exp_vs_act_wins(rd_prem['B365Dprob'], rd_prem['FTR']=='D')
exp_vs_act_wins(rd_prem['LBHprob'], rd_prem['FTR']=='H')
exp_vs_act_wins(rd_prem['LBAprob'], rd_prem['FTR']=='A')
exp_vs_act_wins(rd_prem['LBDprob'], rd_prem['FTR']=='D')

# compare long odds expected vs actual
long_odds_cutoff = 0.2
exp_vs_act_wins(rd_prem['B365Hprob'][rd_prem['B365Hprob']<long_odds_cutoff], rd_prem['FTR'][rd_prem['B365Hprob']<long_odds_cutoff]=='H')
exp_vs_act_wins(rd_prem['B365Aprob'][rd_prem['B365Aprob']<long_odds_cutoff], rd_prem['FTR'][rd_prem['B365Aprob']<long_odds_cutoff]=='A')
exp_vs_act_wins(rd_prem['B365Dprob'][rd_prem['B365Dprob']<long_odds_cutoff], rd_prem['FTR'][rd_prem['B365Dprob']<long_odds_cutoff]=='D')
exp_vs_act_wins(rd_prem['LBHprob'][rd_prem['LBHprob']<long_odds_cutoff], rd_prem['FTR'][rd_prem['LBHprob']<long_odds_cutoff]=='H')
exp_vs_act_wins(rd_prem['LBAprob'][rd_prem['LBAprob']<long_odds_cutoff], rd_prem['FTR'][rd_prem['LBAprob']<long_odds_cutoff]=='A')
exp_vs_act_wins(rd_prem['LBDprob'][rd_prem['LBDprob']<long_odds_cutoff], rd_prem['FTR'][rd_prem['LBDprob']<long_odds_cutoff]=='D')
# first signs - betting against away games with long odds seems to give an advantage

exp_vs_act_wins(rd_champ['B365Hprob'], rd_champ['FTR']=='H')
exp_vs_act_wins(rd_champ['B365Aprob'], rd_champ['FTR']=='A')
exp_vs_act_wins(rd_champ['B365Dprob'], rd_champ['FTR']=='D')
exp_vs_act_wins(rd_champ['LBHprob'], rd_champ['FTR']=='H')
exp_vs_act_wins(rd_champ['LBAprob'], rd_champ['FTR']=='A')
exp_vs_act_wins(rd_champ['LBDprob'], rd_champ['FTR']=='D')

# compare long odds expected vs actual
exp_vs_act_wins(rd_champ['B365Hprob'][rd_champ['B365Hprob']<long_odds_cutoff], rd_champ['FTR'][rd_champ['B365Hprob']<long_odds_cutoff]=='H')
exp_vs_act_wins(rd_champ['B365Aprob'][rd_champ['B365Aprob']<long_odds_cutoff], rd_champ['FTR'][rd_champ['B365Aprob']<long_odds_cutoff]=='A')
exp_vs_act_wins(rd_champ['B365Dprob'][rd_champ['B365Dprob']<long_odds_cutoff], rd_champ['FTR'][rd_champ['B365Dprob']<long_odds_cutoff]=='D')
exp_vs_act_wins(rd_champ['LBHprob'][rd_champ['LBHprob']<long_odds_cutoff], rd_champ['FTR'][rd_champ['LBHprob']<long_odds_cutoff]=='H')
exp_vs_act_wins(rd_champ['LBAprob'][rd_champ['LBAprob']<long_odds_cutoff], rd_champ['FTR'][rd_champ['LBAprob']<long_odds_cutoff]=='A')
exp_vs_act_wins(rd_champ['LBDprob'][rd_champ['LBDprob']<long_odds_cutoff], rd_champ['FTR'][rd_champ['LBDprob']<long_odds_cutoff]=='D')




#-----testing scenarios-----#
### random picks
random_picks = np.random.choice(pd.unique(rd_prem['FTR']), len(rd_prem['FTR']))
sum((random_picks==rd_prem['FTR']) * ((rd_prem['FTR']=='H')*1*rd_prem['B365H']+(rd_prem['FTR']=='D')*1*rd_prem['B365D']+(rd_prem['FTR']=='A')*1*rd_prem['B365A']))
len(random_picks)

sum((random_picks==rd_prem['FTR']) * ((rd_prem['FTR']=='H')*1*rd_prem['LBH']+(rd_prem['FTR']=='D')*1*rd_prem['LBD']+(rd_prem['FTR']=='A')*1*rd_prem['LBA']))
len(random_picks)

random_picks_champ = np.random.choice(pd.unique(rd_champ['FTR']), len(rd_champ['FTR']))
sum((random_picks_champ==rd_champ['FTR']) * ((rd_champ['FTR']=='H')*1*rd_champ['B365H']+(rd_champ['FTR']=='D')*1*rd_champ['B365D']+(rd_champ['FTR']=='A')*1*rd_champ['B365A']))
len(random_picks_champ)

sum((random_picks_champ==rd_champ['FTR']) * ((rd_champ['FTR']=='H')*1*rd_champ['LBH']+(rd_champ['FTR']=='D')*1*rd_champ['LBD']+(rd_champ['FTR']=='A')*1*rd_champ['LBA']))
len(random_picks_champ)


### home team every time
sum((rd_prem['FTR']=='H')*1*rd_prem['B365H'])
sum((rd_prem['FTR']=='H')*1*rd_prem['LBH'])

sum((rd_champ['FTR']=='H')*1*rd_champ['B365H'])
sum((rd_champ['FTR']=='H')*1*rd_champ['LBH'])


### away team every time
sum((rd_prem['FTR']=='A')*1*rd_prem['B365A'])
sum((rd_prem['FTR']=='A')*1*rd_prem['LBA'])

sum((rd_champ['FTR']=='A')*1*rd_champ['B365A'])
sum((rd_champ['FTR']=='A')*1*rd_champ['LBA'])


### draw every time
sum((rd_prem['FTR']=='D')*1*rd_prem['B365D'])
sum((rd_prem['FTR']=='D')*1*rd_prem['LBD'])

sum((rd_champ['FTR']=='D')*1*rd_champ['B365D'])
sum((rd_champ['FTR']=='D')*1*rd_champ['LBD'])


### against long odds away wins
long_odds_cutoff=0.2
sum(((rd_prem['FTR']=='H')*1*rd_prem['B365H'])[rd_prem['B365Aprob']<long_odds_cutoff])
sum(rd_prem['B365Aprob']<long_odds_cutoff)
sum(((rd_prem['FTR']=='H')*1*rd_prem['LBH'])[rd_prem['LBAprob']<long_odds_cutoff])
sum(rd_prem['LBAprob']<long_odds_cutoff)

sum(((rd_champ['FTR']=='H')*1*rd_champ['B365H'])[rd_champ['B365Aprob']<long_odds_cutoff])
sum(rd_champ['B365Aprob']<long_odds_cutoff)
sum(((rd_champ['FTR']=='H')*1*rd_champ['LBH'])[rd_champ['LBAprob']<long_odds_cutoff])
sum(rd_champ['LBAprob']<long_odds_cutoff)



### data frame of odds cutoffs and wins etc
# prem
# Bet 365
odds_cutoffs = np.arange(20)/20
odds_cutoffs_df = pd.DataFrame(odds_cutoffs, columns=['odds_cutoff'])
odds_cutoffs_df['longerAwayOdds'] = [sum(rd_prem['B365Aprob']<prob) for prob in odds_cutoffs]
odds_cutoffs_df['homeWinningsAgainstLongerAwayOdds'] = [sum(((rd_prem['FTR']=='H')*1*rd_prem['B365H'])[rd_prem['B365Aprob']<prob]) for prob in odds_cutoffs]
odds_cutoffs_df['awayWinningsForLongerAwayOdds'] = [sum(((rd_prem['FTR']=='A')*1*rd_prem['B365A'])[rd_prem['B365Aprob']<prob]) for prob in odds_cutoffs]
odds_cutoffs_df['drawWinningsForLongerAwayOdds'] = [sum(((rd_prem['FTR']=='D')*1*rd_prem['B365D'])[rd_prem['B365Aprob']<prob]) for prob in odds_cutoffs]
odds_cutoffs_df['longerHomeOdds'] = [sum(rd_prem['B365Hprob']<prob) for prob in odds_cutoffs]
odds_cutoffs_df['homeWinningsForLongerHomeOdds'] = [sum(((rd_prem['FTR']=='H')*1*rd_prem['B365H'])[rd_prem['B365Hprob']<prob]) for prob in odds_cutoffs]
odds_cutoffs_df['awayWinningsAgainstLongerHomeOdds'] = [sum(((rd_prem['FTR']=='A')*1*rd_prem['B365A'])[rd_prem['B365Hprob']<prob]) for prob in odds_cutoffs]
odds_cutoffs_df['drawWinningsAgainstLongerHomeOdds'] = [sum(((rd_prem['FTR']=='D')*1*rd_prem['B365D'])[rd_prem['B365Hprob']<prob]) for prob in odds_cutoffs]

odds_cutoffs_df['shorterAwayOdds'] = [sum(rd_prem['B365Aprob']>prob) for prob in odds_cutoffs]
odds_cutoffs_df['homeWinningsAgainstShorterAwayOdds'] = [sum(((rd_prem['FTR']=='H')*1*rd_prem['B365H'])[rd_prem['B365Aprob']>prob]) for prob in odds_cutoffs]
odds_cutoffs_df['awayWinningsForShorterAwayOdds'] = [sum(((rd_prem['FTR']=='A')*1*rd_prem['B365A'])[rd_prem['B365Aprob']>prob]) for prob in odds_cutoffs]
odds_cutoffs_df['drawWinningsForShorterAwayOdds'] = [sum(((rd_prem['FTR']=='D')*1*rd_prem['B365D'])[rd_prem['B365Aprob']>prob]) for prob in odds_cutoffs]
odds_cutoffs_df['shorterHomeOdds'] = [sum(rd_prem['B365Hprob']>prob) for prob in odds_cutoffs]
odds_cutoffs_df['homeWinningsForShorterHomeOdds'] = [sum(((rd_prem['FTR']=='H')*1*rd_prem['B365H'])[rd_prem['B365Hprob']>prob]) for prob in odds_cutoffs]
odds_cutoffs_df['awayWinningsAgainstShorterHomeOdds'] = [sum(((rd_prem['FTR']=='A')*1*rd_prem['B365A'])[rd_prem['B365Hprob']>prob]) for prob in odds_cutoffs]
odds_cutoffs_df['drawWinningsAgainstShorterHomeOdds'] = [sum(((rd_prem['FTR']=='D')*1*rd_prem['B365D'])[rd_prem['B365Hprob']>prob]) for prob in odds_cutoffs]


# Ladbrokes
odds_cutoffs_df_lb = pd.DataFrame(odds_cutoffs, columns=['odds_cutoff'])
odds_cutoffs_df_lb['longerAwayOdds'] = [sum(rd_prem['LBAprob']<prob) for prob in odds_cutoffs]
odds_cutoffs_df_lb['homeWinningsAgainstLongerAwayOdds'] = [sum(((rd_prem['FTR']=='H')*1*rd_prem['LBH'])[rd_prem['LBAprob']<prob]) for prob in odds_cutoffs]
odds_cutoffs_df_lb['awayWinningsForLongerAwayOdds'] = [sum(((rd_prem['FTR']=='A')*1*rd_prem['LBA'])[rd_prem['LBAprob']<prob]) for prob in odds_cutoffs]
odds_cutoffs_df_lb['drawWinningsForLongerAwayOdds'] = [sum(((rd_prem['FTR']=='D')*1*rd_prem['LBD'])[rd_prem['LBAprob']<prob]) for prob in odds_cutoffs]
odds_cutoffs_df_lb['longerHomeOdds'] = [sum(rd_prem['LBHprob']<prob) for prob in odds_cutoffs]
odds_cutoffs_df_lb['homeWinningsForLongerHomeOdds'] = [sum(((rd_prem['FTR']=='H')*1*rd_prem['LBH'])[rd_prem['LBHprob']<prob]) for prob in odds_cutoffs]
odds_cutoffs_df_lb['awayWinningsAgainstLongerHomeOdds'] = [sum(((rd_prem['FTR']=='A')*1*rd_prem['LBA'])[rd_prem['LBHprob']<prob]) for prob in odds_cutoffs]
odds_cutoffs_df_lb['drawWinningsAgainstLongerHomeOdds'] = [sum(((rd_prem['FTR']=='D')*1*rd_prem['LBD'])[rd_prem['LBHprob']<prob]) for prob in odds_cutoffs]

odds_cutoffs_df_lb['shorterAwayOdds'] = [sum(rd_prem['LBAprob']>prob) for prob in odds_cutoffs]
odds_cutoffs_df_lb['homeWinningsAgainstShorterAwayOdds'] = [sum(((rd_prem['FTR']=='H')*1*rd_prem['LBH'])[rd_prem['LBAprob']>prob]) for prob in odds_cutoffs]
odds_cutoffs_df_lb['awayWinningsForShorterAwayOdds'] = [sum(((rd_prem['FTR']=='A')*1*rd_prem['LBA'])[rd_prem['LBAprob']>prob]) for prob in odds_cutoffs]
odds_cutoffs_df_lb['drawWinningsForShorterAwayOdds'] = [sum(((rd_prem['FTR']=='D')*1*rd_prem['LBD'])[rd_prem['LBAprob']>prob]) for prob in odds_cutoffs]
odds_cutoffs_df_lb['shorterHomeOdds'] = [sum(rd_prem['LBHprob']>prob) for prob in odds_cutoffs]
odds_cutoffs_df_lb['homeWinningsForShorterHomeOdds'] = [sum(((rd_prem['FTR']=='H')*1*rd_prem['LBH'])[rd_prem['LBHprob']>prob]) for prob in odds_cutoffs]
odds_cutoffs_df_lb['awayWinningsAgainstShorterHomeOdds'] = [sum(((rd_prem['FTR']=='A')*1*rd_prem['LBA'])[rd_prem['LBHprob']>prob]) for prob in odds_cutoffs]
odds_cutoffs_df_lb['drawWinningsAgainstShorterHomeOdds'] = [sum(((rd_prem['FTR']=='D')*1*rd_prem['LBD'])[rd_prem['LBHprob']>prob]) for prob in odds_cutoffs]


### ODDS MORE OFTEN LONGER THAN THEY ARE SHORT - CONFIRMED ON THE INTERNET

# champ
# Bet 365
odds_cutoffs_df_champ_b365 = pd.DataFrame(odds_cutoffs, columns=['odds_cutoff'])
odds_cutoffs_df_champ_b365['longerAwayOdds'] = [sum(rd_champ['B365Aprob']<prob) for prob in odds_cutoffs]
odds_cutoffs_df_champ_b365['homeWinningsAgainstLongerAwayOdds'] = [sum(((rd_champ['FTR']=='H')*1*rd_champ['B365H'])[rd_champ['B365Aprob']<prob]) for prob in odds_cutoffs]
odds_cutoffs_df_champ_b365['awayWinningsForLongerAwayOdds'] = [sum(((rd_champ['FTR']=='A')*1*rd_champ['B365A'])[rd_champ['B365Aprob']<prob]) for prob in odds_cutoffs]
odds_cutoffs_df_champ_b365['drawWinningsForLongerAwayOdds'] = [sum(((rd_champ['FTR']=='D')*1*rd_champ['B365D'])[rd_champ['B365Aprob']<prob]) for prob in odds_cutoffs]
odds_cutoffs_df_champ_b365['longerHomeOdds'] = [sum(rd_champ['B365Hprob']<prob) for prob in odds_cutoffs]
odds_cutoffs_df_champ_b365['homeWinningsForLongerHomeOdds'] = [sum(((rd_champ['FTR']=='H')*1*rd_champ['B365H'])[rd_champ['B365Hprob']<prob]) for prob in odds_cutoffs]
odds_cutoffs_df_champ_b365['awayWinningsAgainstLongerHomeOdds'] = [sum(((rd_champ['FTR']=='A')*1*rd_champ['B365A'])[rd_champ['B365Hprob']<prob]) for prob in odds_cutoffs]
odds_cutoffs_df_champ_b365['drawWinningsAgainstLongerHomeOdds'] = [sum(((rd_champ['FTR']=='D')*1*rd_champ['B365D'])[rd_champ['B365Hprob']<prob]) for prob in odds_cutoffs]

odds_cutoffs_df_champ_b365['shorterAwayOdds'] = [sum(rd_champ['B365Aprob']>prob) for prob in odds_cutoffs]
odds_cutoffs_df_champ_b365['homeWinningsAgainstShorterAwayOdds'] = [sum(((rd_champ['FTR']=='H')*1*rd_champ['B365H'])[rd_champ['B365Aprob']>prob]) for prob in odds_cutoffs]
odds_cutoffs_df_champ_b365['awayWinningsForShorterAwayOdds'] = [sum(((rd_champ['FTR']=='A')*1*rd_champ['B365A'])[rd_champ['B365Aprob']>prob]) for prob in odds_cutoffs]
odds_cutoffs_df_champ_b365['drawWinningsForShorterAwayOdds'] = [sum(((rd_champ['FTR']=='D')*1*rd_champ['B365D'])[rd_champ['B365Aprob']>prob]) for prob in odds_cutoffs]
odds_cutoffs_df_champ_b365['shorterHomeOdds'] = [sum(rd_champ['B365Hprob']>prob) for prob in odds_cutoffs]
odds_cutoffs_df_champ_b365['homeWinningsForShorterHomeOdds'] = [sum(((rd_champ['FTR']=='H')*1*rd_champ['B365H'])[rd_champ['B365Hprob']>prob]) for prob in odds_cutoffs]
odds_cutoffs_df_champ_b365['awayWinningsAgainstShorterHomeOdds'] = [sum(((rd_champ['FTR']=='A')*1*rd_champ['B365A'])[rd_champ['B365Hprob']>prob]) for prob in odds_cutoffs]
odds_cutoffs_df_champ_b365['drawWinningsAgainstShorterHomeOdds'] = [sum(((rd_champ['FTR']=='D')*1*rd_champ['B365D'])[rd_champ['B365Hprob']>prob]) for prob in odds_cutoffs]


# Ladbrokes
odds_cutoffs_df_champ_lb = pd.DataFrame(odds_cutoffs, columns=['odds_cutoff'])
odds_cutoffs_df_champ_lb['longerAwayOdds'] = [sum(rd_champ['LBAprob']<prob) for prob in odds_cutoffs]
odds_cutoffs_df_champ_lb['homeWinningsAgainstLongerAwayOdds'] = [sum(((rd_champ['FTR']=='H')*1*rd_champ['LBH'])[rd_champ['LBAprob']<prob]) for prob in odds_cutoffs]
odds_cutoffs_df_champ_lb['awayWinningsForLongerAwayOdds'] = [sum(((rd_champ['FTR']=='A')*1*rd_champ['LBA'])[rd_champ['LBAprob']<prob]) for prob in odds_cutoffs]
odds_cutoffs_df_champ_lb['drawWinningsForLongerAwayOdds'] = [sum(((rd_champ['FTR']=='D')*1*rd_champ['LBD'])[rd_champ['LBAprob']<prob]) for prob in odds_cutoffs]
odds_cutoffs_df_champ_lb['longerHomeOdds'] = [sum(rd_champ['LBHprob']<prob) for prob in odds_cutoffs]
odds_cutoffs_df_champ_lb['homeWinningsForLongerHomeOdds'] = [sum(((rd_champ['FTR']=='H')*1*rd_champ['LBH'])[rd_champ['LBHprob']<prob]) for prob in odds_cutoffs]
odds_cutoffs_df_champ_lb['awayWinningsAgainstLongerHomeOdds'] = [sum(((rd_champ['FTR']=='A')*1*rd_champ['LBA'])[rd_champ['LBHprob']<prob]) for prob in odds_cutoffs]
odds_cutoffs_df_champ_lb['drawWinningsAgainstLongerHomeOdds'] = [sum(((rd_champ['FTR']=='D')*1*rd_champ['LBD'])[rd_champ['LBHprob']<prob]) for prob in odds_cutoffs]

odds_cutoffs_df_champ_lb['shorterAwayOdds'] = [sum(rd_champ['LBAprob']>prob) for prob in odds_cutoffs]
odds_cutoffs_df_champ_lb['homeWinningsAgainstShorterAwayOdds'] = [sum(((rd_champ['FTR']=='H')*1*rd_champ['LBH'])[rd_champ['LBAprob']>prob]) for prob in odds_cutoffs]
odds_cutoffs_df_champ_lb['awayWinningsForShorterAwayOdds'] = [sum(((rd_champ['FTR']=='A')*1*rd_champ['LBA'])[rd_champ['LBAprob']>prob]) for prob in odds_cutoffs]
odds_cutoffs_df_champ_lb['drawWinningsForShorterAwayOdds'] = [sum(((rd_champ['FTR']=='D')*1*rd_champ['LBD'])[rd_champ['LBAprob']>prob]) for prob in odds_cutoffs]
odds_cutoffs_df_champ_lb['shorterHomeOdds'] = [sum(rd_champ['LBHprob']>prob) for prob in odds_cutoffs]
odds_cutoffs_df_champ_lb['homeWinningsForShorterHomeOdds'] = [sum(((rd_champ['FTR']=='H')*1*rd_champ['LBH'])[rd_champ['LBHprob']>prob]) for prob in odds_cutoffs]
odds_cutoffs_df_champ_lb['awayWinningsAgainstShorterHomeOdds'] = [sum(((rd_champ['FTR']=='A')*1*rd_champ['LBA'])[rd_champ['LBHprob']>prob]) for prob in odds_cutoffs]
odds_cutoffs_df_champ_lb['drawWinningsAgainstShorterHomeOdds'] = [sum(((rd_champ['FTR']=='D')*1*rd_champ['LBD'])[rd_champ['LBHprob']>prob]) for prob in odds_cutoffs]





# prem B365 2017
odds_cutoffs_df_prem_b365_2017 = pd.DataFrame(odds_cutoffs, columns=['odds_cutoff'])
odds_cutoffs_df_prem_b365_2017['longerAwayOdds'] = [sum((rd_prem['B365Aprob']<prob) & (rd_prem['seasonEndYear']==2017)) for prob in odds_cutoffs]
odds_cutoffs_df_prem_b365_2017['homeWinningsAgainstLongerAwayOdds'] = [sum(((rd_prem['FTR']=='H')*1*rd_prem['B365H'])[(rd_prem['B365Aprob']<prob) & (rd_prem['seasonEndYear']==2017)]) for prob in odds_cutoffs]
odds_cutoffs_df_prem_b365_2017['awayWinningsForLongerAwayOdds'] = [sum(((rd_prem['FTR']=='A')*1*rd_prem['B365A'])[(rd_prem['B365Aprob']<prob) & (rd_prem['seasonEndYear']==2017)]) for prob in odds_cutoffs]
odds_cutoffs_df_prem_b365_2017['drawWinningsForLongerAwayOdds'] = [sum(((rd_prem['FTR']=='D')*1*rd_prem['B365D'])[(rd_prem['B365Aprob']<prob) & (rd_prem['seasonEndYear']==2017)]) for prob in odds_cutoffs]
odds_cutoffs_df_prem_b365_2017['longerHomeOdds'] = [sum((rd_prem['B365Hprob']<prob) & (rd_prem['seasonEndYear']==2017)) for prob in odds_cutoffs]
odds_cutoffs_df_prem_b365_2017['homeWinningsForLongerHomeOdds'] = [sum(((rd_prem['FTR']=='H')*1*rd_prem['B365H'])[(rd_prem['B365Hprob']<prob) & (rd_prem['seasonEndYear']==2017)]) for prob in odds_cutoffs]
odds_cutoffs_df_prem_b365_2017['awayWinningsAgainstLongerHomeOdds'] = [sum(((rd_prem['FTR']=='A')*1*rd_prem['B365A'])[(rd_prem['B365Hprob']<prob) & (rd_prem['seasonEndYear']==2017)]) for prob in odds_cutoffs]
odds_cutoffs_df_prem_b365_2017['drawWinningsAgainstLongerHomeOdds'] = [sum(((rd_prem['FTR']=='D')*1*rd_prem['B365D'])[(rd_prem['B365Hprob']<prob) & (rd_prem['seasonEndYear']==2017)]) for prob in odds_cutoffs]

odds_cutoffs_df_prem_b365_2017['shorterAwayOdds'] = [sum((rd_prem['B365Aprob']>prob) & (rd_prem['seasonEndYear']==2017)) for prob in odds_cutoffs]
odds_cutoffs_df_prem_b365_2017['homeWinningsAgainstShorterAwayOdds'] = [sum(((rd_prem['FTR']=='H')*1*rd_prem['B365H'])[(rd_prem['B365Aprob']>prob) & (rd_prem['seasonEndYear']==2017)]) for prob in odds_cutoffs]
odds_cutoffs_df_prem_b365_2017['awayWinningsForShorterAwayOdds'] = [sum(((rd_prem['FTR']=='A')*1*rd_prem['B365A'])[(rd_prem['B365Aprob']>prob) & (rd_prem['seasonEndYear']==2017)]) for prob in odds_cutoffs]
odds_cutoffs_df_prem_b365_2017['drawWinningsForShorterAwayOdds'] = [sum(((rd_prem['FTR']=='D')*1*rd_prem['B365D'])[(rd_prem['B365Aprob']>prob) & (rd_prem['seasonEndYear']==2017)]) for prob in odds_cutoffs]
odds_cutoffs_df_prem_b365_2017['shorterHomeOdds'] = [sum((rd_prem['B365Hprob']>prob) & (rd_prem['seasonEndYear']==2017)) for prob in odds_cutoffs]
odds_cutoffs_df_prem_b365_2017['homeWinningsForShorterHomeOdds'] = [sum(((rd_prem['FTR']=='H')*1*rd_prem['B365H'])[(rd_prem['B365Hprob']>prob) & (rd_prem['seasonEndYear']==2017)]) for prob in odds_cutoffs]
odds_cutoffs_df_prem_b365_2017['awayWinningsAgainstShorterHomeOdds'] = [sum(((rd_prem['FTR']=='A')*1*rd_prem['B365A'])[(rd_prem['B365Hprob']>prob) & (rd_prem['seasonEndYear']==2017)]) for prob in odds_cutoffs]
odds_cutoffs_df_prem_b365_2017['drawWinningsAgainstShorterHomeOdds'] = [sum(((rd_prem['FTR']=='D')*1*rd_prem['B365D'])[(rd_prem['B365Hprob']>prob) & (rd_prem['seasonEndYear']==2017)]) for prob in odds_cutoffs]




#----- test income if betting £1 on certain games -----#
match_days = pd.unique(rd_prem['Date'])
match_days = np.sort(match_days)

# betting on home games

odds_cap = 1/0.001
odds_floor = 1/1
bets_home = (rd_prem['B365H']>odds_floor) & (rd_prem['B365H']<odds_cap)

results_home = pd.DataFrame(match_days, columns=['match_day'])
results_home['gamesThatDay'] = [sum(rd_prem['Date']==match_date) for match_date in match_days]
results_home['gamesBetOn'] = [sum((rd_prem['Date']==match_date) & bets_home) for match_date in match_days]
results_home['winnings'] = [sum(((rd_prem['Date']==match_date) & (rd_prem['FTR']=='H'))*1*rd_prem['B365H']*bets_home) for match_date in match_days]
results_home['profitLoss'] = results_home['winnings']-results_home['gamesBetOn']
results_home['cumulativeProfitLoss'] = [sum(results_home['profitLoss'][:(row+1)]) for row in range(len(match_days))]

print('Total bet: ', sum(results_home['gamesBetOn']), 'Winnings: ', sum(results_home['winnings']))

plt.plot(results_home['cumulativeProfitLoss']*1000/sum(results_home['gamesBetOn']))





odds_cap = 1/0.5
odds_floor = 1/1
bets = (rd_prem['B365H']>odds_floor) & (rd_prem['B365H']<odds_cap)

results = pd.DataFrame(match_days, columns=['match_day'])
results['gamesThatDay'] = [sum(rd_prem['Date']==match_date) for match_date in match_days]
results['gamesBetOn'] = [sum((rd_prem['Date']==match_date) & bets) for match_date in match_days]
results['winnings'] = [sum(((rd_prem['Date']==match_date) & (rd_prem['FTR']=='H'))*1*rd_prem['B365H']*bets) for match_date in match_days]
results['profitLoss'] = results['winnings']-results['gamesBetOn']
results['cumulativeProfitLoss'] = [sum(results['profitLoss'][:(row+1)]) for row in range(len(match_days))]

print('Total bet: ', sum(results['gamesBetOn']), 'Winnings: ', sum(results['winnings']))

plt.plot(results['cumulativeProfitLoss']*1000/sum(results['gamesBetOn']))




# betting on away games


odds_cap = 1/0.001
odds_floor = 1/1
bets_away = (rd_prem['B365A']>odds_floor) & (rd_prem['B365A']<odds_cap)

results_away = pd.DataFrame(match_days, columns=['match_day'])
results_away['gamesThatDay'] = [sum(rd_prem['Date']==match_date) for match_date in match_days]
results_away['gamesBetOn'] = [sum((rd_prem['Date']==match_date) & bets_away) for match_date in match_days]
results_away['winnings'] = [sum(((rd_prem['Date']==match_date) & (rd_prem['FTR']=='A'))*1*rd_prem['B365A']*bets_away) for match_date in match_days]
results_away['profitLoss'] = results_away['winnings']-results_away['gamesBetOn']
results_away['cumulativeProfitLoss'] = [sum(results_away['profitLoss'][:(row+1)]) for row in range(len(match_days))]

print('Total bet: ', sum(results_away['gamesBetOn']), 'Winnings: ', sum(results_away['winnings']))

plt.plot(results_away['cumulativeProfitLoss']*1000/sum(results_away['gamesBetOn']))



### TRY BASIC MODEL WITH ODDS AS FEATURE TO LEARN ODDS BIASES BETTER - HAVE FEATURES LIKE HOME, AWAY ETC











