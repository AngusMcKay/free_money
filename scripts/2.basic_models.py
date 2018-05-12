#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 20:06:22 2018

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







#-----imputation-----#
## set LB odds as B365 odds where it is missing
rd_prem['LBA']=np.where(rd_prem['LBA'].isnull(), rd_prem['B365A'], rd_prem['LBA'])
rd_prem['LBD']=np.where(rd_prem['LBD'].isnull(), rd_prem['B365D'], rd_prem['LBD'])
rd_prem['LBH']=np.where(rd_prem['LBH'].isnull(), rd_prem['B365H'], rd_prem['LBH'])

# one 0 return for B365 home result, impute ladbrokes odds here
rd_champ['B365H']=np.where(rd_champ['B365H']==0, rd_champ['LBH'], rd_champ['B365H'])
rd_league1['B365H']=np.where(rd_league1['B365H']==0, rd_league1['LBH'], rd_league1['B365H'])
rd_league2['B365H']=np.where(rd_league2['B365H']==0, rd_league2['LBH'], rd_league2['B365H'])







#----- adding useful columns -----#
rd_prem['Date']=rd_prem['Date'].map(lambda x: datetime.datetime.strptime(x, '%d/%m/%y'))
rd_prem['seasonEndYear']=rd_prem['Date'].map(lambda x: x.year + (x.month>=7)*1)
rd_prem['seasonWeek']=(53+rd_prem['Date'].map(lambda x: x.week)-31) % 53
rd_prem['prem']=1

rd_champ['Date']=rd_champ['Date'].map(lambda x: datetime.datetime.strptime(x, '%d/%m/%y'))
rd_champ['seasonEndYear']=rd_champ['Date'].map(lambda x: x.year + (x.month>=7)*1)
rd_champ['seasonWeek']=(53+rd_champ['Date'].map(lambda x: x.week)-31) % 53
rd_champ['champ']=1

rd_league1['Date']=rd_league1['Date'].map(lambda x: datetime.datetime.strptime(x, '%d/%m/%y'))
rd_league1['seasonEndYear']=rd_league1['Date'].map(lambda x: x.year + (x.month>=7)*1)
rd_league1['seasonWeek']=(53+rd_league1['Date'].map(lambda x: x.week)-31) % 53
rd_league1['league1']=1

rd_league2['Date']=rd_league2['Date'].map(lambda x: datetime.datetime.strptime(x, '%d/%m/%y'))
rd_league2['seasonEndYear']=rd_league2['Date'].map(lambda x: x.year + (x.month>=7)*1)
rd_league2['seasonWeek']=(53+rd_league2['Date'].map(lambda x: x.week)-31) % 53
rd_league2['league2']=1

rd_conf['Date']=rd_conf['Date'].map(lambda x: datetime.datetime.strptime(x, '%d/%m/%y'))
rd_conf['seasonEndYear']=rd_conf['Date'].map(lambda x: x.year + (x.month>=7)*1)
rd_conf['seasonWeek']=(53+rd_conf['Date'].map(lambda x: x.week)-31) % 53
rd_conf['conf']=1

rd_combined=pd.concat([rd_prem, rd_champ, rd_league1, rd_league2, rd_conf])
# remove where B365 is missing for ease for now, only removes 55 rows
rd_combined = rd_combined[rd_combined['B365H'].notnull()]

# 2 rows where B365 home is 0, impute as LB data here
rd_combined['B365H'] = np.where(rd_combined['B365H']==0, rd_combined['LBH'], rd_combined['B365H'])

# change nan to 0 for league classifications
rd_combined['prem']=np.where(rd_combined['prem'].isnull(), 0, rd_combined['prem'])
rd_combined['champ']=np.where(rd_combined['champ'].isnull(), 0, rd_combined['champ'])
rd_combined['league1']=np.where(rd_combined['league1'].isnull(), 0, rd_combined['league1'])
rd_combined['league2']=np.where(rd_combined['league2'].isnull(), 0, rd_combined['league2'])
rd_combined['conf']=np.where(rd_combined['conf'].isnull(), 0, rd_combined['conf'])

# impute missing lb data with B365 data
rd_combined['LBH'] = np.where(rd_combined['LBH'].isnull(), rd_combined['B365H'], rd_combined['LBH'])
rd_combined['LBD'] = np.where(rd_combined['LBD'].isnull(), rd_combined['B365D'], rd_combined['LBD'])
rd_combined['LBA'] = np.where(rd_combined['LBA'].isnull(), rd_combined['B365A'], rd_combined['LBA'])

np.mean(rd_combined['LBH']/rd_combined['B365H']), np.min(rd_combined['LBH']/rd_combined['B365H']), np.max(rd_combined['LBH']/rd_combined['B365H'])




#-----adding probabilities based on odds-----#
# prem
rd_prem['B365Hprob'] = 1/rd_prem['B365H']#/(1/rd_prem['B365H']+1/rd_prem['B365D']+1/rd_prem['B365A'])
rd_prem['B365Dprob'] = 1/rd_prem['B365D']#/(1/rd_prem['B365H']+1/rd_prem['B365D']+1/rd_prem['B365A'])
rd_prem['B365Aprob'] = 1/rd_prem['B365A']#/(1/rd_prem['B365H']+1/rd_prem['B365D']+1/rd_prem['B365A'])
rd_prem['B365bookiesgain'] = (1/rd_prem['B365H']+1/rd_prem['B365D']+1/rd_prem['B365A'])
np.mean(rd_prem['B365bookiesgain'])
np.min(rd_prem['B365bookiesgain'])
np.max(rd_prem['B365bookiesgain'])
np.mean(rd_prem['B365bookiesgain'][rd_prem['seasonEndYear']==2017])
np.min(rd_prem['B365bookiesgain'][rd_prem['seasonEndYear']==2017])
np.max(rd_prem['B365bookiesgain'][rd_prem['seasonEndYear']==2017])


rd_prem['LBHprob'] = 1/rd_prem['LBH']#/(1/rd_prem['LBH']+1/rd_prem['LBD']+1/rd_prem['LBA'])
rd_prem['LBDprob'] = 1/rd_prem['LBD']#/(1/rd_prem['LBH']+1/rd_prem['LBD']+1/rd_prem['LBA'])
rd_prem['LBAprob'] = 1/rd_prem['LBA']#/(1/rd_prem['LBH']+1/rd_prem['LBD']+1/rd_prem['LBA'])
rd_prem['LBbookiesgain'] = (1/rd_prem['LBH']+1/rd_prem['LBD']+1/rd_prem['LBA'])
np.mean(rd_prem['LBbookiesgain'])
np.min(rd_prem['LBbookiesgain'])
np.max(rd_prem['LBbookiesgain'])
np.mean(rd_prem['LBbookiesgain'][rd_prem['seasonEndYear']==2017])
np.min(rd_prem['LBbookiesgain'][rd_prem['seasonEndYear']==2017])
np.max(rd_prem['LBbookiesgain'][rd_prem['seasonEndYear']==2017])


# champ
rd_champ['B365Hprob'] = 1/rd_champ['B365H']#/(1/rd_champ['B365H']+1/rd_champ['B365D']+1/rd_champ['B365A'])
rd_champ['B365Dprob'] = 1/rd_champ['B365D']#/(1/rd_champ['B365H']+1/rd_champ['B365D']+1/rd_champ['B365A'])
rd_champ['B365Aprob'] = 1/rd_champ['B365A']#/(1/rd_champ['B365H']+1/rd_champ['B365D']+1/rd_champ['B365A'])
rd_champ['B365bookiesgain'] = (1/rd_champ['B365H']+1/rd_champ['B365D']+1/rd_champ['B365A'])
np.mean(rd_champ['B365bookiesgain'])
np.min(rd_champ['B365bookiesgain'])
np.max(rd_champ['B365bookiesgain'])

rd_champ['LBHprob'] = 1/rd_champ['LBH']#/(1/rd_champ['LBH']+1/rd_champ['LBD']+1/rd_champ['LBA'])
rd_champ['LBDprob'] = 1/rd_champ['LBD']#/(1/rd_champ['LBH']+1/rd_champ['LBD']+1/rd_champ['LBA'])
rd_champ['LBAprob'] = 1/rd_champ['LBA']#/(1/rd_champ['LBH']+1/rd_champ['LBD']+1/rd_champ['LBA'])
rd_champ['LBbookiesgain'] = (1/rd_champ['LBH']+1/rd_champ['LBD']+1/rd_champ['LBA'])
np.mean(rd_champ['LBbookiesgain'])
np.min(rd_champ['LBbookiesgain'])
np.max(rd_champ['LBbookiesgain'])


# combined data
rd_combined['B365Hprob'] = 1/rd_combined['B365H']#/(1/rd_combined['B365H']+1/rd_combined['B365D']+1/rd_combined['B365A'])
rd_combined['B365Dprob'] = 1/rd_combined['B365D']#/(1/rd_combined['B365H']+1/rd_combined['B365D']+1/rd_combined['B365A'])
rd_combined['B365Aprob'] = 1/rd_combined['B365A']#/(1/rd_combined['B365H']+1/rd_combined['B365D']+1/rd_combined['B365A'])
rd_combined['B365bookiesgain'] = (1/rd_combined['B365H']+1/rd_combined['B365D']+1/rd_combined['B365A'])
np.mean(rd_combined['B365bookiesgain'])
np.min(rd_combined['B365bookiesgain'])
np.max(rd_combined['B365bookiesgain'])

rd_combined['LBHprob'] = 1/rd_combined['LBH']#/(1/rd_combined['LBH']+1/rd_combined['LBD']+1/rd_combined['LBA'])
rd_combined['LBDprob'] = 1/rd_combined['LBD']#/(1/rd_combined['LBH']+1/rd_combined['LBD']+1/rd_combined['LBA'])
rd_combined['LBAprob'] = 1/rd_combined['LBA']#/(1/rd_combined['LBH']+1/rd_combined['LBD']+1/rd_combined['LBA'])
rd_combined['LBbookiesgain'] = (1/rd_combined['LBH']+1/rd_combined['LBD']+1/rd_combined['LBA'])
np.mean(rd_combined['LBbookiesgain'])
np.min(rd_combined['LBbookiesgain'])
np.max(rd_combined['LBbookiesgain'])




#----- fit basic model -----#
### betting on home matches...

# in sample model to test if method is potentially profitable
xgb_train_features = rd_prem[['B365Hprob', 'B365Dprob', 'B365Aprob']]
xgb_train_labels = rd_prem['FTR']

xgb_model = xgb.XGBRegressor().fit(rd_prem['B365Hprob'].values.reshape(len(rd_prem['B365Hprob']),1), ((rd_prem['FTR']=='H')*1))
xgb_in_predictions = xgb_model.predict(rd_prem['B365Hprob'].values.reshape(len(rd_prem['B365Hprob']),1))
xgb_bets = xgb_in_predictions>(rd_prem['B365Hprob'])

match_days = pd.unique(rd_prem['Date'])
match_days = np.sort(match_days)

results_xgb = pd.DataFrame(match_days, columns=['match_day'])
results_xgb['gamesThatDay'] = [sum(rd_prem['Date']==match_date) for match_date in match_days]
results_xgb['gamesBetOn'] = [sum((rd_prem['Date']==match_date) & xgb_bets) for match_date in match_days]
results_xgb['winnings'] = [sum(((rd_prem['Date']==match_date) & (rd_prem['FTR']=='H'))*1*rd_prem['B365H']*xgb_bets) for match_date in match_days]
results_xgb['profitLoss'] = results_xgb['winnings']-results_xgb['gamesBetOn']
results_xgb['cumulativeProfitLoss'] = [sum(results_xgb['profitLoss'][:(row+1)]) for row in range(len(match_days))]

print('Total bet: ', sum(results_xgb['gamesBetOn']), 'Winnings: ', sum(results_xgb['winnings']))

plt.plot(results_xgb['cumulativeProfitLoss']*1000/sum(results_xgb['gamesBetOn']))



# out sample based on prem up to season cutoff
train_to_season = 2011
xgb_train_features = rd_prem.loc[rd_prem['seasonEndYear']<=train_to_season,['B365Hprob', 'B365Dprob', 'B365Aprob']]
xgb_train_labels = rd_prem['FTR'][rd_prem['seasonEndYear']<=train_to_season]

xgb_model = xgb.XGBRegressor().fit(rd_prem['B365Hprob'].values.reshape(len(rd_prem['B365Hprob']),1)[rd_prem['seasonEndYear']<=train_to_season], ((rd_prem['FTR']=='H')*1)[rd_prem['seasonEndYear']<=train_to_season])
xgb_2017_outcomes = rd_prem[['Date', 'B365H','B365Hprob','FTR']][rd_prem['seasonEndYear']==(train_to_season+1)]
xgb_2017_outcomes['predictions'] = xgb_model.predict(rd_prem['B365Hprob'].values.reshape(len(rd_prem['B365Hprob']),1)[rd_prem['seasonEndYear']==(train_to_season+1)])
xgb_2017_outcomes['bets'] = xgb_2017_outcomes['predictions']>(xgb_2017_outcomes['B365Hprob']+0.0)
xgb_2017_outcomes['winnings'] = xgb_2017_outcomes['bets']*xgb_2017_outcomes['B365H']*(xgb_2017_outcomes['FTR']=='H')

match_days_2017 = pd.unique(rd_prem['Date'][rd_prem['seasonEndYear']==(train_to_season+1)])
match_days_2017 = np.sort(match_days_2017)

results_xgb_2017 = pd.DataFrame(match_days_2017, columns=['match_day'])
results_xgb_2017['gamesThatDay'] = [sum(xgb_2017_outcomes['Date']==match_date) for match_date in match_days_2017]
results_xgb_2017['gamesBetOn'] = [sum((xgb_2017_outcomes['Date']==match_date) & (xgb_2017_outcomes['bets'])) for match_date in match_days_2017]
results_xgb_2017['winnings'] = [sum((xgb_2017_outcomes['Date']==match_date)*xgb_2017_outcomes['winnings']) for match_date in match_days_2017]
results_xgb_2017['profitLoss'] = results_xgb_2017['winnings']-results_xgb_2017['gamesBetOn']
results_xgb_2017['cumulativeProfitLoss'] = [sum(results_xgb_2017['profitLoss'][:(row+1)]) for row in range(len(match_days_2017))]

print('Total bet: ', sum(results_xgb_2017['gamesBetOn']), 'Winnings: ', sum(results_xgb_2017['winnings']), 'Profit: ', round((sum(results_xgb_2017['winnings'])/sum(results_xgb_2017['gamesBetOn'])-1)*100,2), '%')

plt.plot(results_xgb_2017['cumulativeProfitLoss']*1000/sum(results_xgb_2017['gamesBetOn']))


# https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
# http://xgboost.readthedocs.io/en/latest/parameter.html





# combined data
# out sample based on combined data up to season cutoff
train_to_season = 2015
probability_cushion = 0.00
xgb_train_features = rd_combined.loc[rd_combined['seasonEndYear']<=train_to_season,['B365Hprob', 'B365Dprob', 'B365Aprob']]
xgb_train_labels = rd_combined['FTR'][rd_combined['seasonEndYear']<=train_to_season]

# ['B365Hprob', 'prem', 'champ','league1','league2','conf','seasonEndYear','seasonWeek','B365bookiesgain','LBH']
predictors = ['B365Hprob', 'prem', 'champ','league1','league2','conf','B365bookiesgain','seasonEndYear','LBH','seasonWeek']

xgb_model = xgb.XGBRegressor(max_depth=1,
                             n_estimators=20,
                             objective='reg:linear').fit(rd_combined[predictors][rd_combined['seasonEndYear']<=train_to_season], ((rd_combined['FTR']=='H')*1)[rd_combined['seasonEndYear']<=train_to_season])
linRegMod=linear_model.LinearRegression()
linRegMod.fit(X=rd_combined[predictors][rd_combined['seasonEndYear']<=train_to_season], y=((rd_combined['FTR']=='H')*1)[rd_combined['seasonEndYear']<=train_to_season])

xgb_combined_outcomes = rd_combined[['Date', 'B365H','B365Hprob','LBH','LBHprob', 'prem', 'champ','league1','league2','conf','seasonEndYear','seasonWeek','FTR']][rd_combined['seasonEndYear']==(train_to_season+1)]
#xgb_combined_outcomes['predictions'] = xgb_model.predict(rd_combined[predictors][rd_combined['seasonEndYear']==(train_to_season+1)])
xgb_combined_outcomes['predictions'] = linRegMod.predict(rd_combined[predictors][rd_combined['seasonEndYear']==(train_to_season+1)])
xgb_combined_outcomes['bets'] = (xgb_combined_outcomes['predictions']>(xgb_combined_outcomes['B365Hprob']+probability_cushion)) #& (xgb_combined_outcomes['predictions']>(xgb_combined_outcomes['B365Hprob']+probability_cushion))
xgb_combined_outcomes['winnings'] = xgb_combined_outcomes['bets']*(xgb_combined_outcomes[['B365H','LBH']].max(axis=1))*(xgb_combined_outcomes['FTR']=='H')

match_days_combined = pd.unique(rd_combined['Date'][rd_combined['seasonEndYear']==(train_to_season+1)])
match_days_combined = np.sort(match_days_combined)

results_xgb_combined = pd.DataFrame(match_days_combined, columns=['match_day'])
results_xgb_combined['gamesThatDay'] = [sum(xgb_combined_outcomes['Date']==match_date) for match_date in match_days_combined]
results_xgb_combined['gamesBetOn'] = [sum((xgb_combined_outcomes['Date']==match_date) & (xgb_combined_outcomes['bets'])) for match_date in match_days_combined]
results_xgb_combined['winnings'] = [sum((xgb_combined_outcomes['Date']==match_date)*xgb_combined_outcomes['winnings']) for match_date in match_days_combined]
results_xgb_combined['profitLoss'] = results_xgb_combined['winnings']-results_xgb_combined['gamesBetOn']
results_xgb_combined['cumulativeProfitLoss'] = [sum(results_xgb_combined['profitLoss'][:(row+1)]) for row in range(len(match_days_combined))]

print('Total bet: ', sum(results_xgb_combined['gamesBetOn']), 'Winnings: ', sum(results_xgb_combined['winnings']), 'Profit: ', round((sum(results_xgb_combined['winnings'])/sum(results_xgb_combined['gamesBetOn'])-1)*100,2), '%')
print('Total bet: ', sum(xgb_combined_outcomes['bets'][xgb_combined_outcomes['prem']==1]) + sum(xgb_combined_outcomes['bets'][xgb_combined_outcomes['champ']==1]), 'Winnings: ', sum(xgb_combined_outcomes['winnings'][xgb_combined_outcomes['prem']==1]) + sum(xgb_combined_outcomes['winnings'][xgb_combined_outcomes['champ']==1]), 'Profit: ', round(((sum(xgb_combined_outcomes['winnings'][xgb_combined_outcomes['prem']==1]) + sum(xgb_combined_outcomes['winnings'][xgb_combined_outcomes['champ']==1]))/(sum(xgb_combined_outcomes['bets'][xgb_combined_outcomes['prem']==1]) + sum(xgb_combined_outcomes['bets'][xgb_combined_outcomes['champ']==1]))-1)*100,2), '%')

pred_cutoff=0.5
print('Total bet: ', sum(xgb_combined_outcomes['bets'][xgb_combined_outcomes['predictions']>pred_cutoff]), 'Winnings: ', sum(xgb_combined_outcomes['winnings'][xgb_combined_outcomes['predictions']>pred_cutoff]), 'Profit: ', round((sum(xgb_combined_outcomes['winnings'][xgb_combined_outcomes['predictions']>pred_cutoff])/sum(xgb_combined_outcomes['bets'][xgb_combined_outcomes['predictions']>pred_cutoff])-1)*100,2), '%')

plt.plot(results_xgb_combined['cumulativeProfitLoss']*1000/sum(results_xgb_combined['gamesBetOn']))

linRegMod.coef_
linRegMod.intercept_
# note: small samples making it volatile - get other countries' data
# need to control model better - adding a feature shouldn't dramatically change it (maybe try lin reg)
# try loss function of actual loss
# sort out analysis to more easily show results that can be copied and pasted to ss and then test features etc
# add more features like team size
# maybe concentrate on improving model before looking at winnings
# compare odds between bookies
# arbitrage?

