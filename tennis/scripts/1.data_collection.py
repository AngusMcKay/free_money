#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 09:03:28 2018

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
from urllib.request import urlopen
from zipfile import ZipFile


### import data from http://www.football-data.co.uk/downloadm.php
years_for_search_and_extract = ['1718','1617','1516','1415','1314','1213','1112',
                                '1011','0910','0809','0708','0607','0506','0405',
                                '0304','0203','0102','0001','9900','9899','9798'
                                ]

#for year in tqdm(years_for_search_and_extract):
#    
#    # Download the file from the URL
#    zipresp = urlopen('http://www.football-data.co.uk/mmz4281/'+year+'/data.zip')
#    
#    # Create a new file on the hard drive
#    tempzip = open("/tmp/tempfile.zip", "wb")
#    
#    # Write the contents of the downloaded file into the new file
#    tempzip.write(zipresp.read())
#    
#    # Close the newly-created file
#    tempzip.close()
#    
#    # Re-open the newly-created file with ZipFile()
#    zf = ZipFile("/tmp/tempfile.zip")
#    
#    # Extract contents
#    zf.extractall(path = 'all_data')
#    
#    # close the ZipFile instance
#    zf.close()



#### test which files have unreadable stuff in them
#for year in tqdm(years_for_search_and_extract):
#    
#    files = glob.glob('all_data/'+year+'/*')
#    
#    for file in files:
#        try:
#            pd.read_csv(file, error_bad_lines=False)
#        except UnicodeDecodeError:
#            print(file+year)



### extract content from each folder
data={}
for year in tqdm(years_for_search_and_extract):
    
    files = glob.glob('all_data/'+year+'/*')
    
    # read data and skip lines in cases when there is extra commas at the end of the
    data[year] = pd.concat(pd.read_csv(file, error_bad_lines=False, encoding='latin1') for file in files)
    

combined_data = pd.concat(data[year] for year in years_for_search_and_extract)



### data inspection
sum(combined_data.notnull().sum(axis=1)<=1)
# 8933 lines are all na so remove these
combined_data = combined_data.loc[combined_data.notnull().sum(axis=1)>1,:]

# checks on variables
pd.unique(combined_data['Div'])
sum(combined_data['Div'].isnull())
sum(combined_data['Date'].isnull())
sum(combined_data['B365H'].isnull())
sum((combined_data['B365H'].isnull()) & (combined_data['LBH'].isnull()) & 
    (combined_data['WHH'].isnull()) & (combined_data['VCH'].isnull()) & 
    (combined_data['IWH'].isnull()) & (combined_data['BWH'].isnull()))
sum(combined_data['LBH'].isnull())
pd.unique(combined_data['Div'][(combined_data['B365H'].isnull()) & (combined_data['LBH'].isnull())])
sum(combined_data['B365H']==0)
sum(combined_data['LBH']==0)



### imputation
# where B365 data is missing impute LB data and vice versa
combined_data['B365H']=np.where(combined_data['B365H'].isnull(), combined_data['LBH'], combined_data['B365H'])
combined_data['B365D']=np.where(combined_data['B365D'].isnull(), combined_data['LBD'], combined_data['B365D'])
combined_data['B365A']=np.where(combined_data['B365A'].isnull(), combined_data['LBA'], combined_data['B365A'])

combined_data['LBH']=np.where(combined_data['LBH'].isnull(), combined_data['B365H'], combined_data['LBH'])
combined_data['LBD']=np.where(combined_data['LBD'].isnull(), combined_data['B365D'], combined_data['LBD'])
combined_data['LBA']=np.where(combined_data['LBA'].isnull(), combined_data['B365A'], combined_data['LBA'])

# where B365H is 0 impute LBH
combined_data['B365H']=np.where(combined_data['B365H']==0, combined_data['LBH'], combined_data['B365H'])

# remove rows where B365 data is null
combined_data = combined_data.loc[combined_data['B365H'].notnull(),:]

np.mean(combined_data['B365D']/combined_data['LBD'])




### data processing
combined_data['Date']=combined_data['Date'].map(lambda x: datetime.datetime.strptime(x, '%d/%m/%y') if len(x)==8 else datetime.datetime.strptime(x, '%d/%m/%Y'))
combined_data['seasonEndYear']=combined_data['Date'].map(lambda x: x.year + (x.month>=7)*1)
combined_data['seasonWeek']=(53+combined_data['Date'].map(lambda x: x.week)-31) % 53


# add bookies probabilities and gains
combined_data['B365Hprob'] = 1/combined_data['B365H']/(1/combined_data['B365H']+1/combined_data['B365D']+1/combined_data['B365A'])
combined_data['B365Dprob'] = 1/combined_data['B365D']/(1/combined_data['B365H']+1/combined_data['B365D']+1/combined_data['B365A'])
combined_data['B365Aprob'] = 1/combined_data['B365A']/(1/combined_data['B365H']+1/combined_data['B365D']+1/combined_data['B365A'])
combined_data['B365bookiesgain'] = (1/combined_data['B365H']+1/combined_data['B365D']+1/combined_data['B365A'])
np.mean(combined_data['B365bookiesgain'])
np.min(combined_data['B365bookiesgain'])
np.max(combined_data['B365bookiesgain'])


combined_data['LBHprob'] = 1/combined_data['LBH']/(1/combined_data['LBH']+1/combined_data['LBD']+1/combined_data['LBA'])
combined_data['LBDprob'] = 1/combined_data['LBD']/(1/combined_data['LBH']+1/combined_data['LBD']+1/combined_data['LBA'])
combined_data['LBAprob'] = 1/combined_data['LBA']/(1/combined_data['LBH']+1/combined_data['LBD']+1/combined_data['LBA'])
combined_data['LBbookiesgain'] = (1/combined_data['LBH']+1/combined_data['LBD']+1/combined_data['LBA'])
np.mean(combined_data['LBbookiesgain'])
np.min(combined_data['LBbookiesgain'])
np.max(combined_data['LBbookiesgain'])


# change Div to one-hot encoding
combined_data = pd.concat([combined_data, pd.get_dummies(combined_data['Div'])], axis=1)

combined_data.to_csv('all_data/combined_data.csv', index=False)



