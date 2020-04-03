

import urllib
import pandas as pd
import json
from tqdm import tqdm

# set some dates
past_dates = pd.date_range(start='2018-01-01', end='2019-09-18')



# function to get jackpot data
def get_jackpot_from_meeting(meeting):
    meeting_pools = meeting['meetingPools']
    
    # try to find jackpot in pools
    for pool in meeting_pools:
        try:
            if pool['name']==2:
                jackpot_details = pool
                try:
                    dividend_amount = jackpot_details['dividend']['amount']
                except:
                    dividend_amount = 0
                try:
                    dividend_winning_units = jackpot_details['dividend']['winningUnits']
                except:
                    dividend_winning_units = 0
                output = [meeting['course'],
                          jackpot_details['grossTotal'],
                          jackpot_details['guarantee'],
                          jackpot_details['netTotal'],
                          dividend_amount,
                          dividend_winning_units
                          ]
                race_results = []
                for r in jackpot_details['legs']:
                    try:
                        remainingstake = r['breakdown']['remainingStake']
                    except:
                        remainingstake = 0
                    race_results = race_results + [remainingstake,
                                                   r['winningHorses'][0]
                                                   ]
                output = output + race_results
                return output
        except:
            pass
    
    # return nothing if doesn't exist
    return None



# function to get placepot data
def get_placepot_from_meeting(meeting):
    meeting_pools = meeting['meetingPools']
    
    # try to find placepot in pools
    for pool in meeting_pools:
        try:
            if pool['name']==1:
                placepot_details = pool
                try:
                    dividend_amount = placepot_details['dividend']['amount']
                except:
                    dividend_amount = 0
                try:
                    dividend_winning_units = placepot_details['dividend']['winningUnits']
                except:
                    dividend_winning_units = 0
                output = [meeting['course'],
                          placepot_details['grossTotal'],
                          placepot_details['netTotal'],
                          dividend_amount,
                          dividend_winning_units
                          ]
                race_results = []
                for r in placepot_details['legs']:
                    try:
                        remainingstake = r['breakdown']['remainingStake']
                    except:
                        remainingstake = 0
                    try:
                        winner = r['winningHorses'][0]
                    except:
                        winner = None
                    try:
                        second = r['winningHorses'][1]
                    except:
                        second = None
                    try:
                        third = r['winningHorses'][2]
                    except:
                        third = None
                    race_results = race_results + [remainingstake,winner,second,third]
                output = output + race_results
                return output
        except:
            pass
    
    # return nothing if doesn't exist
    return None



# function to get quadpot data
def get_quadpot_from_meeting(meeting):
    meeting_pools = meeting['meetingPools']
    
    # try to find quadpot in pools
    for pool in meeting_pools:
        try:
            quadpots_dont_have_names = pool['name']
            pass
                
        except:
            try:
                if len(pool['legs'])==4:
                    quadpot_details = pool
                    try:
                        dividend_amount = quadpot_details['dividend']['amount']
                    except:
                        dividend_amount = 0
                    try:
                        dividend_winning_units = quadpot_details['dividend']['winningUnits']
                    except:
                        dividend_winning_units = 0
                    output = [meeting['course'],
                              quadpot_details['grossTotal'],
                              quadpot_details['netTotal'],
                              dividend_amount,
                              dividend_winning_units
                              ]
                    race_results = []
                    for r in quadpot_details['legs']:
                        try:
                            remainingstake = r['breakdown']['remainingStake']
                        except:
                            remainingstake = 0
                        try:
                            winner = r['winningHorses'][0]
                        except:
                            winner = None
                        try:
                            second = r['winningHorses'][1]
                        except:
                            second = None
                        try:
                            third = r['winningHorses'][2]
                        except:
                            third = None
                        race_results = race_results + [remainingstake,winner,second,third]
                    output = output + race_results
                    return output
            except:
                pass
    
    # return nothing if doesn't exist
    return None



# function to get race details
def get_races_from_meeting(meeting):
    meeting_races = meeting['races']
    course = meeting['course']
    
    try:
        output = []
        for i, r in enumerate(meeting_races):
            try:
                race_number = i
                try:
                    time = r['time']
                except:
                    time = None
                try:
                    name = r['distance']
                except:
                    name = None
                try:
                    distance = r['distance']
                except:
                    distance = None
                try:
                    runners = len(r['horses'])
                except:
                    runners = None
                try:
                    places = r['result']['numberOfTotePlaces']
                except:
                    places = None
                try:
                    if r['result']['placedHorses'][0]['position']==1:
                        winner = r['result']['placedHorses'][0]['number']
                    else:
                        print('WARNING: placedHorses 0 IS NOT WINNER')
                except:
                    winner = None
                try:
                    if r['result']['placedHorses'][1]['position'] in [1,2]:
                        second = r['result']['placedHorses'][1]['number']
                    else:
                        print('WARNING: placedHorses 1 IS NOT SECOND')
                except:
                    second = None
                try:
                    if r['result']['placedHorses'][2]['position'] in [1,2,3]:
                        third = r['result']['placedHorses'][2]['number']
                    else:
                        print('WARNING: placedHorses 2 IS NOT THIRD')
                except:
                    third = None
                tote_win_gross, tote_win_net, tote_win_dividend, tote_win_units, tote_win_dividend_guide = [None]*5
                tote_place_gross, tote_place_net, tote_place_dividend_1, tote_place_units_1, tote_place_dividend_2, tote_place_units_2, tote_place_dividend_3, tote_place_units_3, tote_place_dividend_guide = [None]*9
                tote_exacta_gross, tote_exacta_net, tote_exacta_dividend, tote_exacta_units, tote_exacta_first, tote_exacta_second = [None]*6
                tote_trifecta_gross, tote_trifecta_net, tote_trifecta_dividend, tote_trifecta_units, tote_trifecta_first, tote_trifecta_second, tote_trifecta_third = [None]*7
                for j, p in enumerate(r['racePools']):
                    try:
                        if p['name']==1:
                            tote_place_gross = p['grossTotal']
                            tote_place_net = p['netTotal']
                            try:
                                tote_place_dividend_1 = p['dividends'][0]['amount']
                                tote_place_units_1 = p['dividends'][0]['winningUnits']
                            except:
                                pass
                            try:
                                tote_place_dividend_2 = p['dividends'][1]['amount']
                                tote_place_units_2 = p['dividends'][1]['winningUnits']
                            except:
                                pass
                            try:
                                tote_place_dividend_3 = p['dividends'][2]['amount']
                                tote_place_units_3 = p['dividends'][2]['winningUnits']
                            except:
                                pass
                            tote_place_dividend_guide = p['dividendGuides']
                        elif p['name']==3: # this is following order that appears in json
                            tote_exacta_gross = p['grossTotal']
                            tote_exacta_net = p['netTotal']
                            tote_exacta_dividend = p['dividends'][0]['amount']
                            tote_exacta_units = p['dividends'][0]['winningUnits']
                            tote_exacta_dividend_guide = p['dividendGuides']
                            try:
                                tote_exacta_first = p['dividends'][0]['winningHorses'][0]['number']
                            except:
                                pass
                            try:
                                tote_exacta_second = p['dividends'][0]['winningHorses'][1]['number']
                            except:
                                pass
                        elif p['name']==4:
                            tote_trifecta_gross = p['grossTotal']
                            tote_trifecta_net = p['netTotal']
                            tote_trifecta_dividend = p['dividends'][0]['amount']
                            tote_trifecta_units = p['dividends'][0]['winningUnits']
                            try:
                                tote_trifecta_first = p['dividends'][0]['winningHorses'][0]['number']
                            except:
                                pass
                            try:
                                tote_trifecta_second = p['dividends'][0]['winningHorses'][1]['number']
                            except:
                                pass
                            try:
                                tote_trifecta_third = p['dividends'][0]['winningHorses'][2]['number']
                            except:
                                pass
                    except:
                        try:
                            tote_win_gross = p['grossTotal']
                        except:
                            tote_win_gross = None
                        try:
                            tote_win_net = p['netTotal']
                        except:
                            tote_win_net = None
                        try:
                            tote_win_dividend = p['dividends'][0]['amount']
                        except:
                            tote_win_dividend = None
                        try:
                            tote_win_units = p['dividends'][0]['winningUnits']
                        except:
                            tote_win_units = None
                        try:
                            tote_win_dividend_guide = p['dividendGuides']
                        except:
                            tote_win_dividend_guide = None
                
                output.append([
                        time[:10], course, race_number, time, name, distance, runners, places, winner, second, third,
                        tote_win_gross, tote_win_net, tote_win_dividend, tote_win_units, tote_win_dividend_guide,
                        tote_place_gross, tote_place_net, tote_place_dividend_1, tote_place_units_1, tote_place_dividend_2, tote_place_units_2, tote_place_dividend_3, tote_place_units_3, tote_place_dividend_guide,
                        tote_exacta_gross, tote_exacta_net, tote_exacta_dividend, tote_exacta_units, tote_exacta_first, tote_exacta_second, tote_exacta_dividend_guide,
                        tote_trifecta_gross, tote_trifecta_net, tote_trifecta_dividend, tote_trifecta_units, tote_trifecta_first, tote_trifecta_second, tote_trifecta_third
                        ])
            except:
                pass
        
        if len(output)>0:
            return output
        else:
            return None
    
    except:
        return None



# loop over dates to obtain data for each date
jackpot_data = []
placepot_data = []
quadpot_data = []
races_data = []
for d in tqdm(past_dates):
    # get data and test if data from correct date (and not default which seems to be today)
    yyyymmdd = d.strftime('%Y')+'-'+d.strftime('%m')+'-'+d.strftime('%d')
    geturl = 'https://api.totepoolliveinfo.com/racecard/getHistoricalRacecard?racecardDate='+yyyymmdd
    getjson = urllib.request.urlopen(geturl).read()
    getdict = json.loads(getjson)
    if getdict['racecardDate'][:10]!=yyyymmdd:
        continue
    
    for m in getdict['meetings']:
        jackpot = get_jackpot_from_meeting(m)
        if jackpot==None:
            pass
        else:
            jackpot_data.append([yyyymmdd]+jackpot)
        
        placepot = get_placepot_from_meeting(m)
        if placepot==None:
            pass
        else:
            placepot_data.append([yyyymmdd]+placepot)
        
        quadpot = get_quadpot_from_meeting(m)
        if quadpot==None:
            pass
        else:
            quadpot_data.append([yyyymmdd]+quadpot)
        
        races = get_races_from_meeting(m)
        if races==None:
            pass
        else:
            races_data = races_data + races



# convert lists into pandas dataframes
jackpot_df = pd.DataFrame(jackpot_data, columns=['date','course','gross_total','guarantee',
                                                 'net_total','dividend_amount','dividend_winning_units',
                                                 'race_1_remaining_stake','race_1_winning_horse',
                                                 'race_2_remaining_stake','race_2_winning_horse',
                                                 'race_3_remaining_stake','race_3_winning_horse',
                                                 'race_4_remaining_stake','race_4_winning_horse',
                                                 'race_5_remaining_stake','race_5_winning_horse',
                                                 'race_6_remaining_stake','race_6_winning_horse'])

placepot_df = pd.DataFrame(placepot_data, columns=['date','course','gross_total',
                                                 'net_total','dividend_amount','dividend_winning_units',
                                                 'race_1_remaining_stake','race_1_winner','race_1_second','race_1_third',
                                                 'race_2_remaining_stake','race_2_winner','race_2_second','race_2_third',
                                                 'race_3_remaining_stake','race_3_winner','race_3_second','race_3_third',
                                                 'race_4_remaining_stake','race_4_winner','race_4_second','race_4_third',
                                                 'race_5_remaining_stake','race_5_winner','race_5_second','race_5_third',
                                                 'race_6_remaining_stake','race_6_winner','race_6_second','race_6_third'])

quadpot_df = pd.DataFrame(quadpot_data, columns=['date','course','gross_total',
                                                 'net_total','dividend_amount','dividend_winning_units',
                                                 'race_1_remaining_stake','race_1_winner','race_1_second','race_1_third',
                                                 'race_2_remaining_stake','race_2_winner','race_2_second','race_2_third',
                                                 'race_3_remaining_stake','race_3_winner','race_3_second','race_3_third',
                                                 'race_4_remaining_stake','race_4_winner','race_4_second','race_4_third'])

races_df = pd.DataFrame(races_data, columns=['date','course','race_number','time','name','distance',
                                             'runners','places','winner','second','third',
                                             'tote_win_gross','tote_win_net','tote_win_dividend','tote_win_units','tote_win_dividend_guide',
                                             'tote_place_gross','tote_place_net','tote_place_dividend_1','tote_place_units_1','tote_place_dividend_2','tote_place_units_2','tote_place_dividend_3','tote_place_units_3','tote_place_dividend_guide',
                                             'tote_exacta_gross','tote_exacta_net','tote_exacta_dividend','tote_exacta_units','tote_exacta_first','tote_exacta_second','tote_exacta_dividend_guide',
                                             'tote_trifecta_gross','tote_trifecta_net','tote_trifecta_dividend','tote_trifecta_units','tote_trifecta_first','tote_trifecta_second','tote_trifecta_third'])


# re-get races data
races_data = []
for d in tqdm(past_dates):
    # get data and test if data from correct date (and not default which seems to be today)
    yyyymmdd = d.strftime('%Y')+'-'+d.strftime('%m')+'-'+d.strftime('%d')
    geturl = 'https://api.totepoolliveinfo.com/racecard/getHistoricalRacecard?racecardDate='+yyyymmdd
    getjson = urllib.request.urlopen(geturl).read()
    getdict = json.loads(getjson)
    if getdict['racecardDate'][:10]!=yyyymmdd:
        continue
    
    for m in getdict['meetings']:
        races = get_races_from_meeting(m)
        if races==None:
            pass
        else:
            races_data = races_data + races

races_df = pd.DataFrame(races_data, columns=['date','course','race_number','time','name','distance',
                                             'runners','places','winner','second','third',
                                             'tote_win_gross','tote_win_net','tote_win_dividend','tote_win_units','tote_win_dividend_guide',
                                             'tote_place_gross','tote_place_net','tote_place_dividend_1','tote_place_units_1','tote_place_dividend_2','tote_place_units_2','tote_place_dividend_3','tote_place_units_3','tote_place_dividend_guide',
                                             'tote_exacta_gross','tote_exacta_net','tote_exacta_dividend','tote_exacta_units','tote_exacta_first','tote_exacta_second','tote_exacta_dividend_guide',
                                             'tote_trifecta_gross','tote_trifecta_net','tote_trifecta_dividend','tote_trifecta_units','tote_trifecta_first','tote_trifecta_second','tote_trifecta_third'])


jackpot_df.to_csv('/home/angus/analytics/product/dev_work/tote/historic_data/jackpot_data.csv')
placepot_df.to_csv('/home/angus/analytics/product/dev_work/tote/historic_data/placepot_data.csv')
quadpot_df.to_csv('/home/angus/analytics/product/dev_work/tote/historic_data/quadpot_data.csv')
races_df.to_csv('/home/angus/projects/betting/tote/historic_data/races_data.csv')





# extract current data in run up to event
import time
import urllib
import json
import numpy as np
geturl = 'https://api.totepoolliveinfo.com/racecard/getCurrentRacecard'
getjson = urllib.request.urlopen(geturl).read()
getdict = json.loads(getjson)


meeting=3
race=3
minutes=45
output = []
for i in tqdm(range(minutes)):
    geturl = 'https://api.totepoolliveinfo.com/racecard/getCurrentRacecard'
    getjson = urllib.request.urlopen(geturl).read()
    getdict = json.loads(getjson)
    output.append(getdict['meetings'][meeting]['races'][race]['racePools'][0]['dividendGuides'])
    time.sleep(60)

musselburgh_0 = pd.DataFrame({'horses':[o['horseNumbers'][0] for o in output[0]]})
for i in range(len(output)):
    minute_output = output[i]
    time_left = len(output)-i
    musselburgh_0['t_'+str(time_left)] = [o['amount'] for o in minute_output]

musselburgh_1 = pd.DataFrame({'horses':[o['horseNumbers'][0] for o in output[0]]})
for i in range(len(output)):
    minute_output = output[i]
    time_left = len(output)-i
    musselburgh_1['t_'+str(time_left)] = [o['amount'] for o in minute_output]

curragh_2 = pd.DataFrame({'horses':[o['horseNumbers'][0] for o in output[0]]})
for i in range(len(output)):
    minute_output = output[i]
    time_left = len(output)-i
    curragh_2['t_'+str(time_left)] = [o['amount'] for o in minute_output]

epsom_3 = pd.DataFrame({'horses':[o['horseNumbers'][0] for o in output[0]]})
for i in range(len(output)):
    minute_output = output[i]
    time_left = len(output)-i
    epsom_3['t_'+str(time_left)] = [o['amount'] for o in minute_output]

mus0_start_over_end = musselburgh_0['t_7']/musselburgh_0['t_5']
mus1_start_over_end = musselburgh_1['t_6']/musselburgh_1['t_4']
cur2_start_over_end = curragh_2['t_3']/curragh_2['t_1']
eps3_start_over_end = epsom_3['t_3']/epsom_3['t_1']

combined_start_over_end = list(mus0_start_over_end)+list(mus1_start_over_end)+list(cur2_start_over_end)+list(eps3_start_over_end)
np.mean(combined_start_over_end)
np.std(combined_start_over_end)
# roughly normal(1, 0.5) in line with expectations
# test with only long and short odds
odds_cutoff = 10
combined_start_over_end_short = (list(mus0_start_over_end[musselburgh_0['t_5']<odds_cutoff])+
                                 list(mus1_start_over_end[musselburgh_1['t_4']<odds_cutoff])+
                                 list(cur2_start_over_end[curragh_2['t_1']<odds_cutoff])+
                                 list(eps3_start_over_end[epsom_3['t_1']<odds_cutoff]))
combined_start_over_end_long = (list(mus0_start_over_end[musselburgh_0['t_5']>odds_cutoff])+
                                 list(mus1_start_over_end[musselburgh_1['t_4']>odds_cutoff])+
                                 list(cur2_start_over_end[curragh_2['t_1']>odds_cutoff])+
                                 list(eps3_start_over_end[epsom_3['t_1']>odds_cutoff]))
np.mean(combined_start_over_end_short)
np.std(combined_start_over_end_short)
np.mean(combined_start_over_end_long)
np.std(combined_start_over_end_long)



