#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 09:03:28 2018

@author: angus
"""

import os
os.chdir('/home/angus/projects/betting/tennis')

import numpy as np
import pandas as pd
from tqdm import tqdm
import glob
from urllib.request import urlopen
from zipfile import ZipFile


### import data from http://www.football-data.co.uk/downloadm.php
years_for_search_and_extract = ['2018','2017','2016','2015','2014','2013',
                                '2012','2011','2010','2009','2008','2007',
                                '2006','2005','2004','2003','2002','2001'
                                ]

# mens data
for year in tqdm(years_for_search_and_extract):
    
    # Download the file from the URL
    zipresp = urlopen('http://www.tennis-data.co.uk/'+year+'/'+year+'.zip')
    
    # Create a new file on the hard drive
    tempzip = open("/tmp/tempfile.zip", "wb")
    
    # Write the contents of the downloaded file into the new file
    tempzip.write(zipresp.read())
    
    # Close the newly-created file
    tempzip.close()
    
    # Re-open the newly-created file with ZipFile()
    zf = ZipFile("/tmp/tempfile.zip")
    
    # Extract contents
    zf.extractall(path = 'all_data/mens')
    
    # close the ZipFile instance
    zf.close()

# womens data
for year in tqdm(years_for_search_and_extract[:12]):
    
    # Download the file from the URL
    if int(year) >= 2016:
        zipresp = urlopen('http://www.tennis-data.co.uk/'+year+'w/'+year+'zip')
    else:
        zipresp = urlopen('http://www.tennis-data.co.uk/'+year+'w/'+year+'.zip')
    
    # Create a new file on the hard drive
    tempzip = open("/tmp/tempfile.zip", "wb")
    
    # Write the contents of the downloaded file into the new file
    tempzip.write(zipresp.read())
    
    # Close the newly-created file
    tempzip.close()
    
    # Re-open the newly-created file with ZipFile()
    zf = ZipFile("/tmp/tempfile.zip")
    
    # Extract contents
    zf.extractall(path = 'all_data/womens')
    
    # close the ZipFile instance
    zf.close()





### test which files have unreadable stuff in them
#mens_files = glob.glob('all_data/mens/*')
#womens_files = glob.glob('all_data/womens/*')
#for file in mens_files:
#    try:
#        pd.read_csv(file, error_bad_lines=False)
#    except UnicodeDecodeError:
#        print(file)


test = pd.read_excel('all_data/mens/2014.xlsx')
### extract content from each folder
mens_files = glob.glob('all_data/mens/*')
womens_files = glob.glob('all_data/womens/*')
# read data and skip lines in cases when there is extra commas at the end of the
mens_data = pd.concat(pd.read_excel(file, error_bad_lines=False, encoding='latin1') for file in mens_files)
womens_data = pd.concat(pd.read_excel(file, error_bad_lines=False, encoding='latin1') for file in womens_files)

del(test, mens_files, womens_files, year)

### data inspection
sum(mens_data.notnull().sum(axis=1)<=1)
sum(womens_data.notnull().sum(axis=1)<=1)
# 0 lines are all na
#combined_data = combined_data.loc[combined_data.notnull().sum(axis=1)>1,:]

test_data_to_inspect = mens_data.iloc[:1000,:]

# checks on variables
pd.unique(mens_data['Series'])
pd.unique(mens_data['Comment'])
pd.unique(mens_data['Court'])
pd.unique(mens_data['Surface'])
pd.unique(mens_data['W1'])
pd.unique(mens_data['L1'])
pd.unique(mens_data['W2'])
data_to_inspect = mens_data.loc[mens_data['W2']==' '] # only 2 retired matches have this, remove retired
pd.unique(mens_data['L2'])
data_to_inspect = mens_data.loc[mens_data['L2']==' '] # only 2 retired matches have this, remove retired
mens_data = mens_data.loc[mens_data['Comment']!='Retired']
mens_data = mens_data.loc[mens_data['Comment']!='Walkover']
mens_data = mens_data.loc[mens_data['Comment']!='Sched']
mens_data = mens_data.loc[mens_data['Comment']!='Disqualified']
pd.unique(mens_data['W3'])
data_to_inspect = mens_data.loc[mens_data['W3']==' '] # looks like ' '  is sometimes entered when match only goes to 2 sets
pd.unique(mens_data['L3'])
pd.unique(mens_data['W4'])
pd.unique(mens_data['L4'])
pd.unique(mens_data['W5'])
pd.unique(mens_data['L5'])
pd.unique(mens_data['WRank'])
pd.unique(mens_data['Wsets'])
data_to_inspect = mens_data.loc[mens_data['Wsets'].isnull()] # remove one row with missing sets info
mens_data = mens_data.loc[mens_data['Wsets'].notnull()] # remove one row with missing sets info
pd.unique(mens_data['LRank'])
pd.unique(mens_data['Lsets']) # switch the '`1' to number 1
mens_data['Lsets']=np.where(mens_data['Lsets']=='`1', 1, mens_data['Lsets'])

sum(mens_data['WRank'].isnull())
sum(mens_data['WPts'].isnull())
sum(mens_data['LRank'].isnull())
sum(mens_data['LPts'].isnull())

sum(womens_data['WRank'].isnull())
sum(womens_data['WPts'].isnull())
sum(womens_data['LRank'].isnull())
sum(womens_data['LPts'].isnull())



### imputation
# where B365 data is missing impute LB data and vice versa
mens_data.isnull().sum(axis=0)

sum(mens_data['B365L'].isnull() &
    #mens_data['EXW'].isnull() &
    #mens_data['PSW'].isnull() &
    #mens_data['LBW'].isnull() &
    mens_data['CBL'].isnull() &
    #mens_data['B&WW'].isnull() &
    mens_data['IWL'].isnull()
    )
# impute to B365 the IW and CB odds

mens_data['B365W']=np.where(mens_data['B365W'].isnull(), mens_data['IWW'], mens_data['B365W'])
mens_data['B365W']=np.where(mens_data['B365W'].isnull(), mens_data['CBW'], mens_data['B365W'])
mens_data['B365L']=np.where(mens_data['B365L'].isnull(), mens_data['IWL'], mens_data['B365L'])
mens_data['B365L']=np.where(mens_data['B365L'].isnull(), mens_data['CBL'], mens_data['B365L'])

### data processing
#combined_data['Date']=combined_data['Date'].map(lambda x: datetime.datetime.strptime(x, '%d/%m/%y') if len(x)==8 else datetime.datetime.strptime(x, '%d/%m/%Y'))
#combined_data['seasonEndYear']=combined_data['Date'].map(lambda x: x.year + (x.month>=7)*1)
#combined_data['seasonWeek']=(53+combined_data['Date'].map(lambda x: x.week)-31) % 53


# add bookies probabilities and gains
mens_data['B365Wprob'] = 1/mens_data['B365W']/(1/mens_data['B365W']+1/mens_data['B365L'])
mens_data['B365Lprob'] = 1/mens_data['B365L']/(1/mens_data['B365W']+1/mens_data['B365L'])
mens_data['B365bookiesgain'] = (1/mens_data['B365W']+1/mens_data['B365L'])
np.mean(mens_data['B365bookiesgain'])
np.min(mens_data['B365bookiesgain'])
np.max(mens_data['B365bookiesgain'])



# change Div to one-hot encoding
#combined_data = pd.concat([combined_data, pd.get_dummies(combined_data['Div'])], axis=1)

# save mens data
mens_data.to_csv('all_data/mens_data.csv', index=False)







test_data_to_inspect = womens_data.iloc[:1000,:]

# checks on variables
pd.unique(womens_data['Series'])
pd.unique(womens_data['Comment'])
womens_data = womens_data.loc[womens_data['Comment']!='Retired']
womens_data = womens_data.loc[womens_data['Comment']!='Walkover']
womens_data = womens_data.loc[womens_data['Comment']!='Sched']
womens_data = womens_data.loc[womens_data['Comment']!='Disqualified']

pd.unique(womens_data['Court'])
pd.unique(womens_data['Surface'])
pd.unique(womens_data['W1'])
pd.unique(womens_data['L1'])
pd.unique(womens_data['W2'])
pd.unique(womens_data['L2'])
data_to_inspect = womens_data.loc[womens_data['L2'].isnull()] # remove this one as looks dodgy
womens_data = womens_data.loc[womens_data['L2'].notnull()] # remove this one as looks dodgy
pd.unique(womens_data['W3'])
data_to_inspect = womens_data.loc[womens_data['W3'].isnull()] # looks like ' '  is sometimes entered when match only goes to 2 sets
pd.unique(womens_data['L3'])
pd.unique(womens_data['WRank'])
pd.unique(womens_data['Wsets'])
data_to_inspect = womens_data.loc[womens_data['Wsets'].isnull()] # remove one row with missing sets info
womens_data = womens_data.loc[womens_data['Wsets'].notnull()] # remove one row with missing sets info
pd.unique(womens_data['LRank'])
pd.unique(womens_data['Lsets'])


### imputation
# where B365 data is missing impute LB data and vice versa
womens_data.isnull().sum(axis=0)

sum(womens_data['B365L'].isnull() &
    womens_data['EXW'].isnull() &
    #womens_data['PSW'].isnull() &
    #womens_data['LBW'].isnull() &
    womens_data['CBL'].isnull()
    )
# impute to B365 the EX and CB odds

womens_data['B365W']=np.where(womens_data['B365W'].isnull(), womens_data['EXW'], womens_data['B365W'])
womens_data['B365W']=np.where(womens_data['B365W'].isnull(), womens_data['CBW'], womens_data['B365W'])
womens_data['B365L']=np.where(womens_data['B365L'].isnull(), womens_data['EXL'], womens_data['B365L'])
womens_data['B365L']=np.where(womens_data['B365L'].isnull(), womens_data['CBL'], womens_data['B365L'])
womens_data['B365L']=np.where(womens_data['B365L']=='5..5', 5.5, womens_data['B365L'])
### data processing
#combined_data['Date']=combined_data['Date'].map(lambda x: datetime.datetime.strptime(x, '%d/%m/%y') if len(x)==8 else datetime.datetime.strptime(x, '%d/%m/%Y'))
#combined_data['seasonEndYear']=combined_data['Date'].map(lambda x: x.year + (x.month>=7)*1)
#combined_data['seasonWeek']=(53+combined_data['Date'].map(lambda x: x.week)-31) % 53


# add bookies probabilities and gains
womens_data['B365Wprob'] = 1/womens_data['B365W']/(1/womens_data['B365W']+1/womens_data['B365L'])
womens_data['B365Lprob'] = 1/womens_data['B365L']/(1/womens_data['B365W']+1/womens_data['B365L'])
womens_data['B365bookiesgain'] = (1/womens_data['B365W']+1/womens_data['B365L'])
np.mean(womens_data['B365bookiesgain'])
np.min(womens_data['B365bookiesgain'])
np.max(womens_data['B365bookiesgain'])



# change Div to one-hot encoding
#combined_data = pd.concat([combined_data, pd.get_dummies(combined_data['Div'])], axis=1)

# save womens data
womens_data.to_csv('all_data/womens_data.csv', index=False)








