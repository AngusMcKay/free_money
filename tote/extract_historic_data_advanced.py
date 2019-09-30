"""
Historic data from sportinglife.com which provides
date, race type, number of runners, course, going, class, distance, horse name, age, weight, odds (sp and bsp), race value, finish position, distance won/lost by, OR

get raceIds and general race info from:
https://www.sportinglife.com/api/horse-racing/racing/racecards/2019-02-15

get specific horse data for eac race from:
https://www.sportinglife.com/api/horse-racing/race/512693
"""

import urllib
import json
import time
import pandas as pd
import numpy as np
from tqdm import tqdm


past_dates = pd.date_range(start='2018-01-01', end='2019-09-18')


d = past_dates[410]
yyyymmdd = d.strftime('%Y')+'-'+d.strftime('%m')+'-'+d.strftime('%d')
dateurl = 'https://www.sportinglife.com/api/horse-racing/racing/racecards/'+yyyymmdd
getjson = urllib.request.urlopen(dateurl).read()
getdict = json.loads(getjson)


# meeting summary
getdict[0]['meeting_summary']['date']
getdict[0]['meeting_summary']['course']['name']
getdict[0]['meeting_summary']['course']['country']['long_name'] # not in race details
getdict[0]['meeting_summary']['course']['feed_source'] # not in race details
getdict[0]['meeting_summary']['surface_summary']
getdict[0]['meeting_summary']['going']
getdict[0]['meeting_summary']['weather'] # not in race details
getdict[0]['meeting_summary']['meeting_reference']['id'] # not in race details

# race summary
getdict[0]['races'][0]['time']
getdict[0]['races'][0]['age']
getdict[0]['races'][0]['distance']
getdict[0]['races'][0]['has_handicap']
getdict[0]['races'][0]['name']
getdict[0]['races'][0]['off_time']
getdict[0]['races'][0]['race_class']
getdict[0]['races'][0]['race_summary_reference']['id']
getdict[0]['races'][0]['ride_count']
getdict[0]['races'][0]['winning_time']


def get_races_data(date_dict):
    
    pass


raceid = getdict[0]['races'][0]['race_summary_reference']['id']

raceurl = 'https://www.sportinglife.com/api/horse-racing/race/'+str(raceid)
racejson = urllib.request.urlopen(raceurl).read()
racedict = json.loads(racejson)

# race summary 2
racedict['race_summary']['age']
racedict['race_summary']['course_name']
racedict['race_summary']['course_surface']['surface']
racedict['race_summary']['date']
racedict['race_summary']['distance']
racedict['race_summary']['going']
racedict['race_summary']['has_handicap']
racedict['race_summary']['off_time']
racedict['race_summary']['race_class']
racedict['race_summary']['ride_count']
racedict['race_summary']['time']
racedict['race_summary']['winning_time']

# betting summary
racedict['exacta_win']
racedict['number_of_placed_rides']
racedict['on_course_book_percentage']
racedict['tote_win']
racedict['straight_forecast']
racedict['swingers']
racedict['trifecta']

# horses
racedict['rides'][0]['betting']['current_odds']
racedict['rides'][0]['betting']['historical_odds']
racedict['rides'][0]['cloth_number']
racedict['rides'][0]['draw_number']
racedict['rides'][0]['finish_position']
racedict['rides'][0]['finish_distance']
racedict['rides'][0]['handicap']
racedict['rides'][0]['horse']['age']
racedict['rides'][0]['horse']['formsummary']
racedict['rides'][0]['horse']['horse_reference']['id']
racedict['rides'][0]['horse']['last_ran_days']
racedict['rides'][0]['horse']['sex']['type']
racedict['rides'][0]['jockey']['name']
racedict['rides'][0]['jockey']['person_reference']['id']
racedict['rides'][0]['owner']['name']
racedict['rides'][0]['race_history_stats'][0]['type'] # NEED TO LOOK INTO THIS
racedict['rides'][0]['race_history_stats'][0]['value'] # NEED TO LOOK INTO THIS
racedict['rides'][0]['trainer']['name']
racedict['rides'][0]['trainer']['business_reference']['id']
racedict['rides'][0]['ride_status']



# horse prev results
racedict['rides'][0]['horse']['previous_results'][0]['course_name']
racedict['rides'][0]['horse']['previous_results'][0]['date']
racedict['rides'][0]['horse']['previous_results'][0]['distance']
racedict['rides'][0]['horse']['previous_results'][0]['going']
racedict['rides'][0]['horse']['previous_results'][0]['odds']
racedict['rides'][0]['horse']['previous_results'][0]['position']
racedict['rides'][0]['horse']['previous_results'][0]['race_class']
racedict['rides'][0]['horse']['previous_results'][0]['race_id']
racedict['rides'][0]['horse']['previous_results'][0]['run_type']
racedict['rides'][0]['horse']['previous_results'][0]['runner_count']
racedict['rides'][0]['horse']['previous_results'][0]['time']
racedict['rides'][0]['horse']['previous_results'][0]['weight']

def get_extra_race_data(race_ids):
    pass

def get_horse_data(race_ids):
    pass
