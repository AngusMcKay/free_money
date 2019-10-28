"""
Historic data from sportinglife.com which provides
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
import pandas as pd
import numpy as np
from tqdm import tqdm
import pymysql
import sqlalchemy
import sys
import threading
from multiprocessing import Queue, Pool



'''
first loop over past dates and get race details for each date
the race ids can then be used after to get horse info
'''
# functions to parse json for each date
def get_meeting_races(meeting_data):
    try:
        output = []
        # note: should only fail if meeting_data['races'] doesn't exist, in which case returns empty list
        # if races data exists but is empty, will also return empty list
        try:
            race_date = meeting_data['meeting_summary']['date']
        except:
            race_date = None
        
        try:
            course = meeting_data['meeting_summary']['course']['name']
        except:
            course = None
        
        try:
            country = meeting_data['meeting_summary']['course']['country']['long_name'] # not in race details
        except:
            country = None
        
        try:
            feed_source = meeting_data['meeting_summary']['course']['feed_source'] # not in race details
        except:
            feed_source = None
        
        try:
            surface = meeting_data['meeting_summary']['surface_summary']
        except:
            surface = None
        
        try:
            going = meeting_data['meeting_summary']['going']
        except:
            going = None
        
        try:
            weather = meeting_data['meeting_summary']['weather'] # not in race details
        except:
            weather = None
        
        try:
            meeting_id = meeting_data['meeting_summary']['meeting_reference']['id'] # not in race details
        except:
            meeting_id = None
        
        for i, r in enumerate(meeting_data['races']):
            try:
                meeting_order = i+1
            except:
                meeting_order = None
            try:
                race_time = r['time']
            except:
                race_time = None
            
            try:
                age = r['age']
            except:
                age = None
            
            try:
                distance = r['distance']
            except:
                distance = None
            
            try:
                has_handicap = r['has_handicap']
            except:
                has_handicap = None
            
            try:
                name = r['name']
            except:
                name = None
            
            try:
                off_time = r['off_time']
            except:
                off_time = None
            
            try:
                race_class = r['race_class']
            except:
                race_class = None
            
            try:
                race_id = r['race_summary_reference']['id']
            except:
                race_id = None
            
            try:
                runners = r['ride_count']
            except:
                runners = None
            
            try:
                winning_time = r['winning_time']
            except:
                winning_time = None
            
            race_details = [race_date, course, country, feed_source, surface, going, weather, meeting_id,
                            meeting_order, race_time, age, distance, has_handicap, name, off_time, race_class, race_id, runners, winning_time]
            
            output.append(race_details)
        
        return output
    
    except:
        return []

#get_meeting_races(getdict[0])
        
def get_days_races(day_data):
    try:
        day_output = []
        for m in day_data:
            day_output = day_output + get_meeting_races(m)
        
        return day_output
    except:
        return []

#get_days_races(getdict)

# loop over dates to get data for each date
past_dates = pd.date_range(start='2016-01-01', end='2016-12-31')
races_data = []
for d in tqdm(past_dates):
    try:
        yyyymmdd = d.strftime('%Y')+'-'+d.strftime('%m')+'-'+d.strftime('%d')
        dateurl = 'https://www.sportinglife.com/api/horse-racing/racing/racecards/'+yyyymmdd
        datejson = urllib.request.urlopen(dateurl).read()
        datedict = json.loads(datejson)
        races_data = races_data + get_days_races(datedict)
    except:
        pass

# convert to pandas df
races_df = pd.DataFrame(races_data, columns=['race_date', 'course', 'country', 'feed_source', 'surface', 'going', 'weather', 'meeting_id',
                                             'meeting_order', 'race_time', 'age', 'distance', 'has_handicap', 'name', 'off_time',
                                             'race_class', 'race_id', 'runners', 'winning_time'])

# bit of data manipulation to make things easier later on
def convert_to_yards(distance_measure):
    try:
        if 'm' in distance_measure:
            return int(distance_measure[:-1])*220*8
        elif 'f' in distance_measure:
            return int(distance_measure[:-1])*220
        elif 'y' in distance_measure:
            return int(distance_measure[:-1])
        else:
            return 0
    except:
        return 0

def convert_distance(distance_string):
    try:
        split_string = distance_string.split()
        distance_in_yards = [convert_to_yards(s) for s in split_string]
        return sum(distance_in_yards)
    except:
        return None

races_df['yards'] = [convert_distance(dis) for dis in races_df['distance']]

def convert_to_seconds(time_measure):
    try:
        if 'm' in time_measure:
            return int(time_measure[:-1])*60
        elif 's' in time_measure:
            return float(time_measure[:-1])
        else:
            return 0
    except:
        return 0

def convert_time(time_string):
    try:
        split_string = time_string.split()
        times_in_seconds = [convert_to_seconds(t) for t in split_string]
        return sum(times_in_seconds)
    except:
        return None

races_df['winning_time_seconds'] = [convert_time(tim) for tim in races_df['winning_time']]

# send to db
connect_string = 'mysql+pymysql://root:angunix1@localhost/horses'
sql_engine = sqlalchemy.create_engine(connect_string)

races_df.to_sql(name='races_data', con=sql_engine, schema='horses', if_exists='append', index=False)

data_to_inspect = races_df[:1000]




'''
now using the race ids, get info for each race
note that some of the race info is duplicated in this table, but this is done in case data is missing from either source
'''

# function to process race and horse data
def get_horses_from_race(race_data):
    try:
        output = []
        
        # race details
        try:
            race_date = race_data['race_summary']['date']
        except:
            race_date = None
        
        try:
            race_id = race_data['race_summary']['race_summary_reference']['id']
        except:
            race_id = None
        
        try:
            course = race_data['race_summary']['course_name']
        except:
            course = None
        
        try:
            surface = race_data['race_summary']['course_surface']['surface']
        except:
            surface = None
        
        try:
            going = race_data['race_summary']['going']
        except:
            going = None
        
        try:
            race_time = race_data['race_summary']['time']
        except:
            race_time = None
        
        try:
            age = race_data['race_summary']['age']
        except:
            age = None
        
        try:
            distance = race_data['race_summary']['distance']
        except:
            distance = None
        
        try:
            has_handicap = race_data['race_summary']['has_handicap']
        except:
            has_handicap = None
        
        try:
            off_time = race_data['race_summary']['off_time']
        except:
            off_time = None
        
        try:
            race_class = race_data['race_summary']['race_class']
        except:
            race_class = None
        
        try:
            runners = race_data['race_summary']['ride_count']
        except:
            runners = None
        
        try:
            winning_time = race_data['race_summary']['winning_time']
        except:
            winning_time = None
        
        try:
            prize1 = race_data['prizes']['prize'][0]['prize']
        except:
            prize1 = None
        
        try:
            prize2 = race_data['prizes']['prize'][1]['prize']
        except:
            prize2 = None
        
        try:
            prize3 = race_data['prizes']['prize'][2]['prize']
        except:
            prize3 = None
        
        try:
            stewards = race_data['stewards']
        except:
            stewards = None
        
        race_details = [race_date, race_id, course, surface, going, race_time, age, distance,
                        has_handicap, off_time, race_class, runners, winning_time,
                        prize1, prize2, prize3, stewards]
        
        # bet details
        try:
            number_of_placed_rides = race_data['number_of_placed_rides']
        except:
            number_of_placed_rides = None
        
        try:
            tote_win = race_data['tote_win']
        except:
            tote_win = None
        
        try:
            place_win = race_data['place_win']
        except:
            place_win = None
        
        try:
            exacta_win = race_data['exacta_win']
        except:
            exacta_win = None
        
        try:
            trifecta = race_data['trifecta']
        except:
            trifecta = None
        
        try:
            place_pot_pool = race_data['place_pot']['pool']
        except:
            place_pot_pool = None
        
        try:
            place_pot_pot = race_data['place_pot']['pot']
        except:
            place_pot_pot = None
        
        try:
            place_pot_winstakes = race_data['place_pot']['winStakes']
        except:
            place_pot_winstakes = None
        
        try:
            quad_pot_pool = race_data['quad_pot']['pool']
        except:
            quad_pot_pool = None
        
        try:
            quad_pot_pot = race_data['quad_pot']['pot']
        except:
            quad_pot_pot = None
        
        try:
            quad_pot_winstakes = race_data['quad_pot']['winStakes']
        except:
            quad_pot_winstakes = None
        
        try:
            straight_forecast = race_data['straight_forecast']
        except:
            straight_forecast = None
        
        try:
            tricast = race_data['tricast']
        except:
            tricast = None
        
        try:
            swingers = race_data['swingers']
        except:
            swingers = None
        
        try:
            on_course_book_percentage = race_data['on_course_book_percentage']
        except:
            on_course_book_percentage = None
        
        bet_details = [number_of_placed_rides, tote_win, place_win, exacta_win,
                       trifecta, place_pot_pool, place_pot_pot, place_pot_winstakes,
                       quad_pot_pool, quad_pot_pot, quad_pot_winstakes, straight_forecast,
                       tricast, swingers, on_course_book_percentage]
        
        # horses
        for h in race_data['rides']:
            try:
                horse_name = h['horse']['name']
            except:
                horse_name = None
            
            try:
                horse_id = h['horse']['horse_reference']['id']
            except:
                horse_id = None
            
            try:
                ride_status = h['ride_status']
            except:
                ride_status = None
            
            try:
                horse_age = h['horse']['age']
            except:
                horse_age = None
            
            try:
                horse_sex = h['horse']['sex']['type']
            except:
                horse_sex = None
            
            try:
                horse_last_ran_days = h['horse']['last_ran_days']
            except:
                horse_last_ran_days = None
            
            try:
                horse_form = h['horse']['formsummary']['display_text']
            except:
                horse_form = None
            
            try:
                finish_position = h['finish_position']
            except:
                finish_position = None
            
            try:
                if h['finsh_position']==1:
                    finish_distance = 0
                else:
                    finish_distance = h['finish_distance']
            except:
                try:
                    finish_distance = h['finish_distance']
                except:
                    finish_distance = None
            
            try:
                cloth_number = h['cloth_number']
            except:
                cloth_number = None
            
            try:
                draw_number = h['draw_number']
            except:
                draw_number = None
            
            try:
                handicap = h['handicap']
            except:
                handicap = None
            
            try:
                betting_odds = h['betting']['current_odds']
            except:
                betting_odds = None
            
            try:
                historical_odds = str(h['betting']['historical_odds'])
            except:
                historical_odds = None
            
            try:
                jockey_name = h['jockey']['name']
            except:
                jockey_name = None
            
            try:
                jockey_id = h['jockey']['person_reference']['id']
            except:
                jockey_id = None
            
            try:
                trainer_name = h['trainer']['name']
            except:
                trainer_name = None
            
            try:
                trainer_id = h['trainer']['business_reference']['id']
            except:
                trainer_id = None
            
            try:
                owner_name = h['owner']['name']
            except:
                owner_name = None
            
            try:
                race_history_stats = str(h['race_history_stats'])
            except:
                race_history_stats = None
            
            try:
                insights = str(h['insights'])
            except:
                insights = None
            
            try:
                medication = str(h['medication'])
            except:
                medication = None
            
            past_results = []
            for pr in range(6):
                try:
                    pr_date = h['horse']['previous_results'][pr]['date']
                except:
                    pr_date = None
                
                try:
                    pr_distance = h['horse']['previous_results'][pr]['distance']
                except:
                    pr_distance = None
                
                try:
                    pr_going = h['horse']['previous_results'][pr]['going']
                except:
                    pr_going = None
                
                try:
                    pr_odds = h['horse']['previous_results'][pr]['odds']
                except:
                    pr_odds = None
                
                try:
                    pr_position = h['horse']['previous_results'][pr]['position']
                except:
                    pr_position = None
                
                try:
                    pr_race_class = h['horse']['previous_results'][pr]['race_class']
                except:
                    pr_race_class = None
                
                try:
                    pr_race_id = h['horse']['previous_results'][pr]['race_id']
                except:
                    pr_race_id = None
                
                try:
                    pr_course_name = h['horse']['previous_results'][pr]['course_name']
                except:
                    pr_course_name = None
                
                try:
                    pr_run_type = h['horse']['previous_results'][pr]['run_type']
                except:
                    pr_run_type = None
                
                try:
                    pr_runner_count = h['horse']['previous_results'][pr]['runner_count']
                except:
                    pr_runner_count = None
                
                try:
                    pr_time = h['horse']['previous_results'][pr]['time']
                except:
                    pr_time = None
                
                try:
                    pr_weight = h['horse']['previous_results'][pr]['weight']
                except:
                    pr_weight = None
                
                past_results = past_results + [pr_date, pr_distance, pr_going, pr_odds,
                                               pr_position, pr_race_class, pr_race_id, pr_course_name,
                                               pr_run_type, pr_runner_count, pr_time, pr_weight]
            
            horse_details = [horse_name, horse_id, ride_status, horse_age, horse_sex,
                             horse_last_ran_days, horse_form, finish_position, finish_distance,
                             cloth_number, draw_number, handicap, betting_odds, historical_odds,
                             jockey_name, jockey_id, trainer_name, trainer_id, owner_name,
                             race_history_stats, insights, medication] + past_results
            
            runner_details = race_details + bet_details + horse_details
            
            output.append(runner_details)
        
        return output
    
    except:
        return []


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
#raceids = races_df['race_id']
raceids = pd.read_sql("SELECT race_id FROM races_data", con=sql_engine)
raceids_already_got = pd.read_sql("SELECT DISTINCT race_id FROM horses_data", con=sql_engine)
raceids_to_get = pd.read_sql("SELECT race_id FROM races_data WHERE race_id NOT IN (SELECT race_id FROM horses_data)", con=sql_engine)

raceids_batch = raceids_to_get[:]
#time_0 = time.time()
#horses_data = []
#for i in tqdm(raceids_batch['race_id']):
#    try:
#        raceurl = 'https://www.sportinglife.com/api/horse-racing/race/'+str(i)
#        racejson = urllib.request.urlopen(raceurl).read()
#        racedict = json.loads(racejson)
#        horses_data = horses_data + get_horses_from_race(racedict)
#    except:
#        pass
#
### multiprocessing attempt
#time_1 = time.time()
#time_1 - time_0
time_2 = time.time()
raceurls = ['https://www.sportinglife.com/api/horse-racing/race/'+str(i) for i in raceids_batch['race_id']]

def read_url2(url):
    racejson = urllib.request.urlopen(url).read()
    racedict = json.loads(racejson)
    return racedict
p = Pool(4)
racedicts = p.map(read_url2, raceurls)

horses_data = []
for i in racedicts:
    horses_data = horses_data + get_horses_from_race(i)

time_3 = time.time()
time_3 - time_2

horses_df = pd.DataFrame(horses_data, columns = horse_data_columns)
#data_to_inspect = horses_df[:100]

# add some useful columns
horses_df['yards'] = [convert_distance(dis) for dis in horses_df['distance']]
horses_df['pr_1_yards'] = [convert_distance(dis) for dis in horses_df['pr_1_distance']]
horses_df['pr_2_yards'] = [convert_distance(dis) for dis in horses_df['pr_2_distance']]
horses_df['pr_3_yards'] = [convert_distance(dis) for dis in horses_df['pr_3_distance']]
horses_df['pr_4_yards'] = [convert_distance(dis) for dis in horses_df['pr_4_distance']]
horses_df['pr_5_yards'] = [convert_distance(dis) for dis in horses_df['pr_5_distance']]
horses_df['pr_6_yards'] = [convert_distance(dis) for dis in horses_df['pr_6_distance']]
horses_df['winning_time_seconds'] = [convert_time(tim) for tim in horses_df['winning_time']]


# send to db
connect_string = 'mysql+pymysql://root:angunix1@localhost/horses'
sql_engine = sqlalchemy.create_engine(connect_string)

horses_df.to_sql(name='horses_data', con=sql_engine, schema='horses', if_exists='append', index=False)

sys.getsizeof(horses_df)/(1024*1024*1024)
#test_transfer = horses_df.iloc[:2]
#test_transfer['historical_odds'] = [str(ho) for ho in test_transfer['historical_odds']]
#test_transfer.to_sql(name='horses_data', con=sql_engine, schema='horses', if_exists='append', index=False)

data_to_inspect = horses_df[:1000]
data_to_inspect2['finish_distance'].iloc[0][-1]=='½'
