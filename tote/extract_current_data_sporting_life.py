"""
Data from sportinglife.com which provides
date, race type, number of runners, course, going, class, distance, horse name, age, weight, odds (sp and bsp), race value, finish position, distance won/lost by, OR

get raceIds and general race info from:
https://www.sportinglife.com/api/horse-racing/racing/racecards/2019-02-15

get specific horse data for eac race from:
https://www.sportinglife.com/api/horse-racing/race/512693
"""

import urllib
import requests
import json
import time
import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm
import pymysql
import sqlalchemy
import sys
import threading
from multiprocessing import Queue, Pool
import tote.extract_current_data_sporting_life_helper_functions as hf
# import importlib
# importlib.reload(hf)

# look up latest date in database to start from
connect_string = 'mysql+pymysql://root:angunix1@localhost/horses'
sql_engine = sqlalchemy.create_engine(connect_string)
latest_date_in_db = pd.read_sql("SELECT MAX(race_date) FROM races_data", con=sql_engine).iloc[0, 0]
todays_date = datetime.date.today().strftime("%Y-%m-%d")

latest_date_in_db_plus_one = hf.add_days_to_date(latest_date_in_db, 1)
yesterday = hf.add_days_to_date(todays_date, -1)

dates_to_get = pd.date_range(start=latest_date_in_db_plus_one, end=yesterday)
dates_to_get = pd.date_range(start='2019-09-23', end='2020-04-01')
races_data = []
for d in tqdm(dates_to_get):
    try:
        yyyymmdd = d.strftime('%Y')+'-'+d.strftime('%m')+'-'+d.strftime('%d')
        dateurl = 'https://www.sportinglife.com/api/horse-racing/racing/racecards/'+yyyymmdd
        datejson = urllib.request.urlopen(dateurl).read()
        datedict = json.loads(datejson)
        races_data = races_data + hf.get_days_races(datedict)
    except:
        pass

# convert to pandas df
races_df = pd.DataFrame(races_data, columns=['race_date', 'course', 'country', 'feed_source', 'surface', 'going', 'weather', 'meeting_id',
                                             'meeting_order', 'race_time', 'age', 'distance', 'has_handicap', 'name', 'off_time',
                                             'race_class', 'race_id', 'runners', 'winning_time'])

# bit of data manipulation to make things easier later on
races_df['yards'] = [hf.convert_distance(dis) for dis in races_df['distance']]
races_df['winning_time_seconds'] = [hf.convert_time(tim) for tim in races_df['winning_time']]

# send to db
#races_df.to_sql(name='races_data', con=sql_engine, schema='horses', if_exists='append', index=False)
races_df.to_sql(name='races_data_new', con=sql_engine, schema='horses', if_exists='append', index=False)


'''
now using the race ids, get info for each race
note that some of the race info is duplicated in this table, but this is done in case data is missing from either source
'''

### NOTE: ONLY GET HORSE DATA FOR LATEST DATES (i.e. not for missing older dates) SO THAT FUTURE JOINS TO CREATE
### PAST RESULTS DATA IS NOT AFFECTED

# columns for df
race_columns = ['race_date', 'race_id', 'course', 'surface', 'going', 'race_time', 'age', 'distance',
                'has_handicap', 'off_time', 'race_class', 'runners', 'winning_time',
                'prize1', 'prize2', 'prize3', 'stewards']

bet_columns = ['number_of_placed_rides', 'tote_win', 'place_win', 'exacta_win',
               'trifecta', 'place_pot_pool', 'place_pot_pot', 'place_pot_winstakes',
               'quad_pot_pool', 'quad_pot_pot', 'quad_pot_winstakes', 'straight_forecast',
               'tricast', 'swingers', 'on_course_book_percentage']

horse_columns = ['horse_name', 'horse_id', 'ride_status', 'horse_age', 'horse_sex',
                 'horse_last_ran_days', 'horse_form', 'finish_position', 'finish_distance',
                 'cloth_number', 'draw_number', 'handicap', 'betting_odds', 'historical_odds',
                 'jockey_name', 'jockey_id', 'trainer_name', 'trainer_id', 'owner_name',
                 'race_history_stats', 'insights', 'medication']

past_results_columns_base = ['pr_date', 'pr_distance', 'pr_going', 'pr_odds', 'pr_position', 'pr_race_class',
                             'pr_race_id', 'pr_course_name','pr_run_type', 'pr_runner_count', 'pr_time', 'pr_weight']
past_results_columns = []
for i in range(6):
    past_results_columns = past_results_columns + ['pr_'+str(i+1)+col_name[2:] for col_name in past_results_columns_base]

horse_data_columns = race_columns + bet_columns + horse_columns + past_results_columns


# loop over race ids to get horse data for each
connect_string = 'mysql+pymysql://root:angunix1@localhost/horses'
sql_engine = sqlalchemy.create_engine(connect_string)

raceids = pd.read_sql("SELECT race_id FROM races_data", con=sql_engine)
raceids_already_got = pd.read_sql("SELECT DISTINCT race_id FROM horses_data", con=sql_engine)
raceids_to_get = pd.read_sql("SELECT race_id FROM races_data WHERE race_id NOT IN (SELECT race_id FROM horses_data)", con=sql_engine)

raceids_batch = raceids_to_get
raceurls = ['https://www.sportinglife.com/api/horse-racing/race/'+str(i) for i in raceids_batch['race_id']]

p = Pool(4)
#racedicts = p.map(hf.read_url, raceurls)
racedicts = list(tqdm(p.imap(hf.read_url, raceurls), total=len(raceurls)))

horses_data = []
for i in tqdm(racedicts):
    horses_data = horses_data + hf.get_horses_from_race(i)

horses_df = pd.DataFrame(horses_data, columns=horse_data_columns)

# add some useful columns (actually not that useful)
horses_df['yards'] = [hf.convert_distance(dis) for dis in horses_df['distance']]
horses_df['pr_1_yards'] = [hf.convert_distance(dis) for dis in horses_df['pr_1_distance']]
horses_df['pr_2_yards'] = [hf.convert_distance(dis) for dis in horses_df['pr_2_distance']]
horses_df['pr_3_yards'] = [hf.convert_distance(dis) for dis in horses_df['pr_3_distance']]
horses_df['pr_4_yards'] = [hf.convert_distance(dis) for dis in horses_df['pr_4_distance']]
horses_df['pr_5_yards'] = [hf.convert_distance(dis) for dis in horses_df['pr_5_distance']]
horses_df['pr_6_yards'] = [hf.convert_distance(dis) for dis in horses_df['pr_6_distance']]
horses_df['winning_time_seconds'] = [hf.convert_time(tim) for tim in horses_df['winning_time']]


# send to db
connect_string = 'mysql+pymysql://root:angunix1@localhost/horses'
sql_engine = sqlalchemy.create_engine(connect_string)

#horses_df.to_sql(name='horses_data', con=sql_engine, schema='horses', if_exists='append', index=False)
horses_df.to_sql(name='horses_data_new', con=sql_engine, schema='horses', if_exists='append', index=False)

