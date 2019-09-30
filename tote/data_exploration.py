

import pandas as pd
import numpy as np
import scipy.stats as stats
import json
import ast
import itertools
import sys
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from tqdm import tqdm

jackpot_df = pd.read_csv('/home/angus/projects/betting/tote/historic_data/jackpot_data.csv', index_col=0)
placepot_df = pd.read_csv('/home/angus/projects/betting/tote/historic_data/placepot_data.csv', index_col=0)
quadpot_df = pd.read_csv('/home/angus/projects/betting/tote/historic_data/quadpot_data.csv', index_col=0)
races_df = pd.read_csv('/home/angus/projects/betting/tote/historic_data/races_data.csv', index_col=0)



'''
clean races data
'''
# remove null rows
data_to_inspect = races_df[races_df['tote_win_dividend_guide'].isnull()]
races_df = races_df[races_df['tote_win_gross'].notnull()]
data_to_inspect = races_df[:100]

# convert distance to yards
def convert_to_yards(distance_measure):
    try:
        if 'm' in distance_measure:
            return int(distance_measure[:-1])*220*8
        elif 'f' in distance_measure:
            return int(distance_measure[:-1])*220
        else:
            return int(distance_measure[:-1])
    except:
        return None

def convert_distance(distance_string):
    split_string = distance_string.split()
    distance_in_yards = [convert_to_yards(s) for s in split_string]
    return sum(distance_in_yards)

races_df['yards'] = [convert_distance(d) for d in races_df['distance']]
sum(races_df['yards'].isnull())

# break dividend guide into list of horse numbers and list of dividends
dividend_guides = [ast.literal_eval(d) for d in races_df['tote_win_dividend_guide']]
def get_horses(dividend_guide):
    try:
        return [d['horseNumbers'][0] for d in dividend_guide]
    except:
        return []
horse_lists = [get_horses(dg) for dg in dividend_guides]
def get_dividend_amount(dividend_guide_item):
    try:
        return dividend_guide_item['amount']
    except:
        return None
def get_dividends(dividend_guide):
    try:
        return [get_dividend_amount(d) for d in dividend_guide]
    except:
        return []
dividend_lists = [get_dividends(dg) for dg in dividend_guides]
def check_winners_and_dividend(winner, dividend, horse_list, dividend_list):
    try:
        return (winner in horse_list) and (dividend_list[horse_list.index(winner)]==dividend)
    except:
        return False
winners = list(races_df['winner'])
dividends = list(races_df['tote_win_dividend'])
winner_and_dividend_agree_list = [check_winners_and_dividend(w, d, hl, dl) for w, d, hl, dl in zip(winners, dividends, horse_lists, dividend_lists)]
races_df['tote_win_dividend_guide'] = dividend_guides
races_df['horses'] = horse_lists
races_df['dividends'] = dividend_lists
races_df['winner_and_dividend_agree'] = winner_and_dividend_agree_list

data_to_inspect = races_df[:100]

win_df = races_df[races_df['winner_and_dividend_agree']]


# break place pool dividend guide into list of horse numbers and list of dividends
place_df = races_df[races_df['tote_place_gross'].notnull()]
place_dividend_guides = [ast.literal_eval(d) for d in place_df['tote_place_dividend_guide']]
place_horse_lists = [get_horses(dg) for dg in place_dividend_guides]
place_dividend_lists = [get_dividends(dg) for dg in place_dividend_guides]
places_list = place_df['places']
place_winners = list(place_df['winner'])
place_winner_divs = list(place_df['tote_place_dividend_1'])
seconds = list(place_df['second'])
second_divs = list(place_df['tote_place_dividend_2'])
thirds = list(place_df['third'])
third_divs = list(place_df['tote_place_dividend_3'])
def check_places_and_dividends(winner, winner_dividend, second, second_dividend, third, third_dividend, horse_list, dividend_list, places):
    try:
        winner_check = winner in horse_list
        winner_dividend_check = dividend_list[horse_list.index(winner)] in [winner_dividend, second_dividend, third_dividend]
        if places>=2:
            second_check = second in horse_list
            second_dividend_check = dividend_list[horse_list.index(second)] in [winner_dividend, second_dividend, third_dividend]
        else:
            second_check = True
            second_dividend_check = True
        if places>=3:
            third_check = third in horse_list
            third_dividend_check = dividend_list[horse_list.index(third)] in [winner_dividend, second_dividend, third_dividend]
        else:
            third_check = True
            third_dividend_check = True
        
        return winner_check and winner_dividend_check and second_check and second_dividend_check and third_check and third_dividend_check
    except:
        return False

place_df['tote_place_dividend_guide'] = place_dividend_guides
place_df['place_horses'] = place_horse_lists
place_df['place_dividends'] = place_dividend_lists
place_df['places_and_dividends_agree'] = [check_places_and_dividends(w, wd, s, sd, t, td, hl, dl, p) for w, wd, s, sd, t, td, hl, dl, p in zip(place_winners,place_winner_divs,seconds,second_divs,thirds,third_divs,place_horse_lists,place_dividend_lists,places_list)]

data_to_inspect = place_df[:100]

place_df = place_df[place_df['places_and_dividends_agree']]


# break place pool dividend guide into list of horse numbers and list of dividends
exacta_df = races_df[races_df['tote_exacta_gross'].notnull()]
exacta_dividend_guides = [ast.literal_eval(d) for d in exacta_df['tote_exacta_dividend_guide']]
def get_exacta_horses(dividend_guide):
    try:
        return [d['horseNumbers'] for d in dividend_guide]
    except:
        return []
exacta_horse_lists = [get_exacta_horses(dg) for dg in exacta_dividend_guides]
exacta_dividend_lists = [get_dividends(dg) for dg in exacta_dividend_guides]
exacta_result = [[f,s] for f,s in zip(exacta_df['tote_exacta_first'],exacta_df['tote_exacta_second'])]
exacta_winner_divs = list(exacta_df['tote_exacta_dividend'])
def check_result_and_dividends(result, result_dividend, exacta_horse_list, dividend_list):
    try:
        result_check = result in exacta_horse_list
        result_dividend_check = dividend_list[exacta_horse_list.index(result)]==result_dividend
        return result_check and result_dividend_check
    except:
        return False

exacta_df['tote_exacta_dividend_guide'] = exacta_dividend_guides
exacta_df['exacta_horses'] = exacta_horse_lists
exacta_df['exacta_dividends'] = exacta_dividend_lists
exacta_df['exactas_and_dividends_agree'] = [check_result_and_dividends(r, rd, hl, dl) for r, rd, hl, dl in zip(exacta_result,exacta_winner_divs,exacta_horse_lists,exacta_dividend_lists)]

data_to_inspect = exacta_df[:100]

exacta_df = exacta_df[exacta_df['exactas_and_dividends_agree']]



'''
first look at some basic race data
'''

# Q1: in race win pool, is there a bias based on any ranges of dividends?
horses = [item for sublist in win_df['tote_win_dividend_guide'] for item in sublist]
horses_df = pd.DataFrame(horses)

horses_df['amount'].hist()
horses_df['amount'][horses_df['amount']<100].hist()

dividends_df = pd.DataFrame({'min':range(1,500),
                             'max':range(2,501)})
dividends_df['total_runners'] = [sum((horses_df['amount']>=i) & (horses_df['amount']<i+1)) for i in dividends_df['min']]
dividends_df['winners'] = [sum((win_df['tote_win_dividend']>=i) & (win_df['tote_win_dividend']<i+1)) for i in dividends_df['min']]
dividends_df['potential_payout_min'] = dividends_df['min']*dividends_df['winners']
dividends_df['potential_payout_max'] = dividends_df['max']*dividends_df['winners']
dividends_df['potential_return_min'] = dividends_df['potential_payout_min']/dividends_df['total_runners']
dividends_df['potential_return_max'] = dividends_df['potential_payout_max']/dividends_df['total_runners']
dividends_df['dodgy_ones'] = (dividends_df['winners']>dividends_df['total_runners'])*1
# Note: seems pretty fairly priced based on this

sys.getsizeof(horses_df)/(1024*1024*1024)
del(horses, horses_df)
sys.getsizeof(win_df)/(1024*1024*1024) 
del(win_df)


# Q2: test the same as Q1 but split by various factors


# Q3: test same for place pool
place_horses = [item for sublist in place_df['tote_place_dividend_guide'] for item in sublist]
place_horses_df = pd.DataFrame(place_horses)

place_horses_df['amount'].hist()
place_horses_df['amount'][place_horses_df['amount']<100].hist()

dividends_df['place_runners'] = [sum((place_horses_df['amount']>=i) & (place_horses_df['amount']<i+1)) for i in dividends_df['min']]
dividends_df['place_winners'] = [
        sum((place_df['tote_place_dividend_1']>=i) & (place_df['tote_place_dividend_1']<i+1)) +
        sum((place_df['tote_place_dividend_2']>=i) & (place_df['tote_place_dividend_2']<i+1)) +
        sum((place_df['tote_place_dividend_3']>=i) & (place_df['tote_place_dividend_3']<i+1))
        for i in dividends_df['min']
        ]
dividends_df['place_potential_payout_min'] = dividends_df['min']*dividends_df['place_winners']
dividends_df['place_potential_payout_max'] = dividends_df['max']*dividends_df['place_winners']
dividends_df['place_potential_return_min'] = dividends_df['place_potential_payout_min']/dividends_df['place_runners']
dividends_df['place_potential_return_max'] = dividends_df['place_potential_payout_max']/dividends_df['place_runners']
dividends_df['dodgy_place_ones'] = (dividends_df['place_winners']>dividends_df['place_runners'])*1
# Need to see in more detail at lower end:
place_dividends_df = pd.DataFrame({'min':[x/50 for x in range(50,500)],
                                   'max':[x/50 for x in range(51,501)]})
place_dividends_df['place_runners'] = [sum((place_horses_df['amount']>=i) & (place_horses_df['amount']<i+1)) for i in place_dividends_df['min']]
place_dividends_df['place_winners'] = [
        sum((place_df['tote_place_dividend_1']>=i) & (place_df['tote_place_dividend_1']<i+1)) +
        sum((place_df['tote_place_dividend_2']>=i) & (place_df['tote_place_dividend_2']<i+1)) +
        sum((place_df['tote_place_dividend_3']>=i) & (place_df['tote_place_dividend_3']<i+1))
        for i in place_dividends_df['min']
        ]
place_dividends_df['place_potential_payout_min'] = place_dividends_df['min']*place_dividends_df['place_winners']
place_dividends_df['place_potential_payout_max'] = place_dividends_df['max']*place_dividends_df['place_winners']
place_dividends_df['place_potential_return_min'] = place_dividends_df['place_potential_payout_min']/place_dividends_df['place_runners']
place_dividends_df['place_potential_return_max'] = place_dividends_df['place_potential_payout_max']/place_dividends_df['place_runners']

sys.getsizeof(place_horses_df)/(1024*1024*1024)
del(place_horses, place_horses_df)


# Q4: test same for exactas
exacta_horses = [item for sublist in exacta_df['tote_exacta_dividend_guide'] for item in sublist]
exacta_horses_df = pd.DataFrame(exacta_horses)

exacta_horses_df['amount'].hist()
exacta_horses_df['amount'][exacta_horses_df['amount']<300].hist()

dividends_df['exacta_selections'] = [sum((exacta_horses_df['amount']>=i) & (exacta_horses_df['amount']<i+1)) for i in dividends_df['min']]
dividends_df['exacta_winners'] = [sum((exacta_df['tote_exacta_dividend']>=i) & (exacta_df['tote_exacta_dividend']<i+1)) for i in dividends_df['min']]
dividends_df['exacta_potential_payout_min'] = dividends_df['min']*dividends_df['exacta_winners']
dividends_df['exacta_potential_payout_max'] = dividends_df['max']*dividends_df['exacta_winners']
dividends_df['exacta_potential_return_min'] = dividends_df['exacta_potential_payout_min']/dividends_df['exacta_selections']
dividends_df['exacta_potential_return_max'] = dividends_df['exacta_potential_payout_max']/dividends_df['exacta_selections']
dividends_df['dodgy_exacta_ones'] = (dividends_df['exacta_winners']>dividends_df['exacta_selections'])*1

sys.getsizeof(exacta_horses_df)/(1024*1024*1024)
del(exacta_horses, exacta_horses_df)



'''
exactas using tote win payouts
'''
# Q1: do exacta and triefects dividends seem reasonable based on tote win and tote place dividends
data_to_inspect = exacta_df[:100]
exacta_vs_win_place_data = exacta_df[exacta_df['places']==2]
exacta_vs_win_place_data = exacta_vs_win_place_data[exacta_vs_win_place_data['tote_place_gross'].notnull()]
exacta_vs_win_place_data['tote_place_dividend_guide'] = [ast.literal_eval(d) for d in exacta_vs_win_place_data['tote_place_dividend_guide']]
exacta_vs_win_place_data['place_horses'] = [get_horses(dg) for dg in exacta_vs_win_place_data['tote_place_dividend_guide']]
exacta_vs_win_place_data['place_dividends'] = [get_dividends(dg) for dg in exacta_vs_win_place_data['tote_place_dividend_guide']]
def get_winner_win_dividend(winner, horse_list, dividend_list):
    try:
        return dividend_list[horse_list.index(winner)]
    except:
        return None
exacta_vs_win_place_data['winner_win_dividend'] = [get_winner_win_dividend(w, hl, dl) for dl, hl, w in zip(exacta_vs_win_place_data['dividends'],exacta_vs_win_place_data['horses'],exacta_vs_win_place_data['winner'])]
exacta_vs_win_place_data['second_win_dividend'] = [get_winner_win_dividend(s, hl, dl) for dl, hl, s in zip(exacta_vs_win_place_data['dividends'],exacta_vs_win_place_data['horses'],exacta_vs_win_place_data['second'])]
exacta_vs_win_place_data['second_place_dividend'] = [get_winner_win_dividend(s, hl, dl) for dl, hl, s in zip(exacta_vs_win_place_data['place_dividends'],exacta_vs_win_place_data['place_horses'],exacta_vs_win_place_data['second'])]
exacta_vs_win_place_data = exacta_vs_win_place_data[exacta_vs_win_place_data['winner_win_dividend'].notnull()]
exacta_vs_win_place_data = exacta_vs_win_place_data[exacta_vs_win_place_data['second_win_dividend'].notnull()]
exacta_vs_win_place_data = exacta_vs_win_place_data[exacta_vs_win_place_data['second_place_dividend'].notnull()]
exacta_vs_win_place_data['prob_exacta_result'] = (1/exacta_vs_win_place_data['winner_win_dividend'])*(1/exacta_vs_win_place_data['second_place_dividend']-1/exacta_vs_win_place_data['second_win_dividend'])
exacta_vs_win_place_data['prob_exacta_result_implied'] = 1/exacta_vs_win_place_data['tote_exacta_dividend']

# Note: looks like some discrepancy between win/place and exacta dividends, where value could be found

# Q2: When exacta dividends are higher than those expected based on win and place odds, does this enhance returns?
def get_exacta_win_place_implied_prob(exacta_selection, win_horse_list, win_dividends, place_horse_list, place_dividends):
    try:
        win_div = win_dividends[win_horse_list.index(exacta_selection[0])]
        second_win_div = win_dividends[win_horse_list.index(exacta_selection[1])]
        second_place_div = place_dividends[place_horse_list.index(exacta_selection[1])]
        return round((1/win_div) * (1/second_place_div - 1/second_win_div),5)
    except:
        return 0
    
def get_exacta_win_place_implied_probs(exacta_horse_list, win_horse_list, win_dividends, place_horse_list, place_dividends):
    return [get_exacta_win_place_implied_prob(es, win_horse_list, win_dividends, place_horse_list, place_dividends) for es in exacta_horse_list]

exacta_vs_win_place_data['exacta_win_place_implied_probs'] = [get_exacta_win_place_implied_probs(ehl, whl, wd, phl, pd) for ehl, whl, wd, phl, pd in zip(exacta_vs_win_place_data['exacta_horses'], exacta_vs_win_place_data['horses'], exacta_vs_win_place_data['dividends'], exacta_vs_win_place_data['place_horses'], exacta_vs_win_place_data['place_dividends'])]
exacta_dividend_margin = 1.2*1.2 # 1.2*1.2 is based on tote win and place fee deductions
def convert_implied_prob_to_dividend_threshold(implied_prob, dividend_margin=1):
    try:
        return round(dividend_margin/implied_prob,2)
    except:
        return 1000000
def convert_implied_probs_list_to_dividend_thresholds(implied_probs_list, dividend_margin=1):
    return [convert_implied_prob_to_dividend_threshold(p, dividend_margin) for p in implied_probs_list]
exacta_vs_win_place_data['exacta_dividend_thresholds'] = [convert_implied_probs_list_to_dividend_thresholds(probs_list, exacta_dividend_margin) for probs_list in exacta_vs_win_place_data['exacta_win_place_implied_probs']]
def get_exacta_selections_above_threshold(exacta_horses_list, exacta_divs, thresholds):
    try:
        return [eh for eh, d, t in zip(exacta_horses_list, exacta_divs, thresholds) if d>t]
    except:
        return []
exacta_vs_win_place_data['exacta_selections_above_threshold'] = [get_exacta_selections_above_threshold(ehl, eds, ths) for ehl, eds, ths in zip(exacta_vs_win_place_data['exacta_horses'], exacta_vs_win_place_data['exacta_dividends'], exacta_vs_win_place_data['exacta_dividend_thresholds'])]
exacta_vs_win_place_data['selections'] = [len(s) for s in exacta_vs_win_place_data['exacta_selections_above_threshold']]
def getting_exacta_winnings_from_selections(exacta_div, winner, second, exacta_selections):
    try:
        if [winner, second] in exacta_selections:
            return exacta_div
        else:
            return 0
    except:
        return 0
exacta_vs_win_place_data['selections_winnings'] = [getting_exacta_winnings_from_selections(d, w, s, es) for d, w, s, es in zip(exacta_vs_win_place_data['tote_exacta_dividend'], exacta_vs_win_place_data['tote_exacta_first'], exacta_vs_win_place_data['tote_exacta_second'], exacta_vs_win_place_data['exacta_selections_above_threshold'])]
sum(exacta_vs_win_place_data['selections'])
sum(exacta_vs_win_place_data['selections_winnings'])

# Q3: based on above, is there a difference at different dividend ranges
def get_dividends_for_selections(exacta_horse_list, exacta_dividend_list, exacta_selections):
    return [exacta_dividend_list[exacta_horse_list.index(s)] for s in exacta_selections]

exacta_vs_win_place_data['dividends_for_exacta_selections_above_threshold'] = [get_dividends_for_selections(ehl, edl, es) for ehl, edl, es in zip(exacta_vs_win_place_data['exacta_horses'],exacta_vs_win_place_data['exacta_dividends'],exacta_vs_win_place_data['exacta_selections_above_threshold'])]

exacta_selection_dividends = [item for sublist in exacta_vs_win_place_data['dividends_for_exacta_selections_above_threshold'] for item in sublist]
exacta_selection_dividends_df = pd.DataFrame(exacta_selection_dividends)

dividends_df['exacta_enhanced_selections'] = [sum((exacta_selection_dividends_df[0]>=i) & (exacta_selection_dividends_df[0]<i+1)) for i in dividends_df['min']]
dividends_df['exacta_enhanced_winners'] = [sum((exacta_vs_win_place_data['selections_winnings']>=i) & (exacta_vs_win_place_data['selections_winnings']<i+1)) for i in dividends_df['min']]
dividends_df['exacta_enhanced_potential_payout_min'] = dividends_df['min']*dividends_df['exacta_enhanced_winners']
dividends_df['exacta_enhanced_potential_payout_max'] = dividends_df['max']*dividends_df['exacta_enhanced_winners']
dividends_df['exacta_enhanced_potential_return_min'] = dividends_df['exacta_enhanced_potential_payout_min']/dividends_df['exacta_enhanced_selections']
dividends_df['exacta_enhanced_potential_return_max'] = dividends_df['exacta_enhanced_potential_payout_max']/dividends_df['exacta_enhanced_selections']

# Quantify additional return compared to random selection
exacta_start_div = 15
exacta_end_div = 100
sum(dividends_df['exacta_enhanced_potential_payout_max'][(dividends_df['min']>=exacta_start_div) & (dividends_df['max']<=exacta_end_div)])/sum(dividends_df['exacta_enhanced_selections'][(dividends_df['min']>=exacta_start_div) & (dividends_df['max']<=exacta_end_div)])
sum(dividends_df['exacta_potential_payout_max'][(dividends_df['min']>=exacta_start_div) & (dividends_df['max']<=exacta_end_div)])/sum(dividends_df['exacta_selections'][(dividends_df['min']>=exacta_start_div) & (dividends_df['max']<=exacta_end_div)])
# conclusion: seems to be some benefit to this, could make good feature in prediction model

# Q4: is the inverse true, i.e. beneficial win and place odds can be found based on exacta odds?

data_to_inspect = exacta_vs_win_place_data[:100]



'''
triectas using tote win payouts
'''
# Q1: how does trifecta dividend implied odds compare to odds implied by win and place dividends
trifecta_df = races_df[races_df['tote_trifecta_gross'].notnull()]
trifecta_df = trifecta_df[trifecta_df['places']==3]
trifecta_df = trifecta_df[trifecta_df['tote_place_gross'].notnull()]
trifecta_df['tote_place_dividend_guide'] = [ast.literal_eval(d) for d in trifecta_df['tote_place_dividend_guide']]
trifecta_df['place_horses'] = [get_horses(dg) for dg in trifecta_df['tote_place_dividend_guide']]
trifecta_df['place_dividends'] = [get_dividends(dg) for dg in trifecta_df['tote_place_dividend_guide']]
trifecta_df['winner_win_dividend'] = [get_winner_win_dividend(w, hl, dl) for dl, hl, w in zip(trifecta_df['dividends'],trifecta_df['horses'],trifecta_df['winner'])]
trifecta_df['second_win_dividend'] = [get_winner_win_dividend(s, hl, dl) for dl, hl, s in zip(trifecta_df['dividends'],trifecta_df['horses'],trifecta_df['second'])]
trifecta_df['second_place_dividend'] = [get_winner_win_dividend(s, hl, dl) for dl, hl, s in zip(trifecta_df['place_dividends'],trifecta_df['place_horses'],trifecta_df['second'])]
trifecta_df['third_win_dividend'] = [get_winner_win_dividend(s, hl, dl) for dl, hl, s in zip(trifecta_df['dividends'],trifecta_df['horses'],trifecta_df['third'])]
trifecta_df['third_place_dividend'] = [get_winner_win_dividend(s, hl, dl) for dl, hl, s in zip(trifecta_df['place_dividends'],trifecta_df['place_horses'],trifecta_df['third'])]

trifecta_df = trifecta_df[trifecta_df['winner_win_dividend'].notnull()]
trifecta_df = trifecta_df[trifecta_df['second_win_dividend'].notnull()]
trifecta_df = trifecta_df[trifecta_df['second_place_dividend'].notnull()]
trifecta_df = trifecta_df[trifecta_df['third_win_dividend'].notnull()]
trifecta_df = trifecta_df[trifecta_df['third_place_dividend'].notnull()]

trifecta_df['prob_trifecta_result'] = (1/trifecta_df['winner_win_dividend'])*(1/trifecta_df['second_place_dividend']-1/trifecta_df['second_win_dividend'])*(1/trifecta_df['third_place_dividend']-1/trifecta_df['third_win_dividend'])*0.5 # multiple by 0.5 because is in order, this assumes horses equally likely to come 2nd or 3rd, should maybe be some ratio of probabilities of 2nd and 3rd place winning
trifecta_df['prob_trifecta_result_implied'] = 1/trifecta_df['tote_trifecta_dividend']
# at first inspection, seems like at longer odds there is potential for beating odds

# Q2: get win and place implied odds for all trifecta combinations and check for bias
def get_trifecta_horse_combos(horses_list):
    try:
        combos_1 = list(itertools.combinations(horses_list,3))
        combos_2 = [(c[0],c[2],c[1]) for c in combos_1]
        combos_3 = [(c[1],c[0],c[2]) for c in combos_1]
        combos_4 = [(c[1],c[2],c[0]) for c in combos_1]
        combos_5 = [(c[2],c[0],c[1]) for c in combos_1]
        combos_6 = [(c[2],c[1],c[0]) for c in combos_1]
        return combos_1 + combos_2 + combos_3 + combos_4 + combos_5 + combos_6
    except:
        return []
trifecta_df['trifecta_horse_combos'] = [get_trifecta_horse_combos(h) for h in trifecta_df['horses']]

def get_trifecta_win_place_implied_prob(trifecta_selection, win_horse_list, win_dividends, place_horse_list, place_dividends):
    try:
        win_div = win_dividends[win_horse_list.index(trifecta_selection[0])]
        second_win_div = win_dividends[win_horse_list.index(trifecta_selection[1])]
        second_place_div = place_dividends[place_horse_list.index(trifecta_selection[1])]
        third_win_div = win_dividends[win_horse_list.index(trifecta_selection[2])]
        third_place_div = place_dividends[place_horse_list.index(trifecta_selection[2])]
        return round((1/win_div) * (1/second_place_div - 1/second_win_div) * (1/third_place_div - 1/third_win_div)*0.5,5) # 0.5 adjustment for 2nd, 3rd ordering, but in reality is a constant so should affect this analysis two much
    except:
        return 0
    
def get_trifecta_win_place_implied_probs(trifecta_horse_list, win_horse_list, win_dividends, place_horse_list, place_dividends):
    return [get_trifecta_win_place_implied_prob(ts, win_horse_list, win_dividends, place_horse_list, place_dividends) for ts in trifecta_horse_list]

trifecta_df['trifecta_win_place_implied_probs'] = [get_trifecta_win_place_implied_probs(thl, whl, wd, phl, pd) for thl, whl, wd, phl, pd in zip(trifecta_df['trifecta_horse_combos'], trifecta_df['horses'], trifecta_df['dividends'], trifecta_df['place_horses'], trifecta_df['place_dividends'])]

def convert_implied_prob_to_dividend_threshold(implied_prob, dividend_margin=1):
    try:
        return round(dividend_margin/implied_prob,2)
    except:
        return 1000000
def convert_implied_probs_list_to_dividend_thresholds(implied_probs_list, dividend_margin=1):
    return [convert_implied_prob_to_dividend_threshold(p, dividend_margin) for p in implied_probs_list]
trifecta_dividend_margin = 1 # doesn't really matter for this analysis as it's all relative
trifecta_df['trifecta_expected_dividends'] = [convert_implied_probs_list_to_dividend_thresholds(probs_list, trifecta_dividend_margin) for probs_list in trifecta_df['trifecta_win_place_implied_probs']]
def get_winning_combo_implied_dividend(winning_combo, trifecta_horse_combos, trifecta_expected_dividends):
    try:
        return trifecta_expected_dividends[trifecta_horse_combos.index(winning_combo)]
    except:
        return 0
trifecta_df['winning_combo_implied_dividend'] = [get_winning_combo_implied_dividend((f,s,t),thc,ted) for f,s,t,thc,ted in zip(trifecta_df['tote_trifecta_first'],trifecta_df['tote_trifecta_second'],trifecta_df['tote_trifecta_third'],trifecta_df['trifecta_horse_combos'],trifecta_df['trifecta_expected_dividends'])]

trifecta_dividend_list = [item for sublist in trifecta_df['trifecta_expected_dividends'] for item in sublist]
trifecta_dividend_list_df = pd.DataFrame(trifecta_dividend_list)

trifecta_div_mins = [round(1.02**x,2) for x in list(range(0,500))]
trifecta_div_maxs = [round(1.02**x,2) for x in list(range(1,501))]
trifecta_dividends_df = pd.DataFrame({'min':trifecta_div_mins,
                                      'max':trifecta_div_maxs})

trifecta_dividends_df['trifecta_combos'] = [sum((trifecta_dividend_list_df[0]>=i) & (trifecta_dividend_list_df[0]<j)) for i,j in zip(trifecta_dividends_df['min'],trifecta_dividends_df['max'])]
trifecta_dividends_df['trifecta_winners'] = [sum((trifecta_df['winning_combo_implied_dividend']>=i) & (trifecta_df['winning_combo_implied_dividend']<j)) for i,j in zip(trifecta_dividends_df['min'],trifecta_dividends_df['max'])]
trifecta_dividends_df['sum_actual_dividends'] = [sum(trifecta_df['tote_trifecta_dividend'][(trifecta_df['winning_combo_implied_dividend']>=i) & (trifecta_df['winning_combo_implied_dividend']<j)]) for i,j in zip(trifecta_dividends_df['min'],trifecta_dividends_df['max'])]
trifecta_dividends_df['return'] = trifecta_dividends_df['sum_actual_dividends']/trifecta_dividends_df['trifecta_combos']


# seems like definitely some good ranges to work in, but expensive ones, plot moving average returns
window = 100
def window_return_calc(dividends, selections):
    try:
        return 100*(sum(dividends)/sum(selections)-1)
    except:
        return None
plt.plot([sum(trifecta_dividends_df['min'][i:(i+window)])/window for i in range(trifecta_dividends_df.shape[0]-window)],
         [window_return_calc(trifecta_dividends_df['sum_actual_dividends'][i:(i+window)],trifecta_dividends_df['trifecta_combos'][i:(i+window)]) for i in range(trifecta_dividends_df.shape[0]-window)])
lose_lines = 350
plt.plot([sum(trifecta_dividends_df['min'][i:(i+window)])/window for i in range(trifecta_dividends_df.shape[0]-window-lose_lines)],
         [window_return_calc(trifecta_dividends_df['sum_actual_dividends'][i:(i+window)],trifecta_dividends_df['trifecta_combos'][i:(i+window)]) for i in range(trifecta_dividends_df.shape[0]-window-lose_lines)])
window = 25
lose_first_rows = 76
lose_last_rows = 150
plt.plot([sum(trifecta_dividends_df['min'][i:(i+window)])/window for i in range(lose_first_rows, trifecta_dividends_df.shape[0]-window-lose_last_rows)],
         [window_return_calc(trifecta_dividends_df['sum_actual_dividends'][i:(i+window)],trifecta_dividends_df['trifecta_combos'][i:(i+window)]) for i in range(lose_first_rows, trifecta_dividends_df.shape[0]-window-lose_last_rows)])

sum(trifecta_dividends_df['sum_actual_dividends'][trifecta_dividends_df['min']<=7.5])
sum(trifecta_dividends_df['trifecta_combos'][trifecta_dividends_df['min']<=7.5])
# seems to suggest best value is odds of around 200-500, and a small window of very low odds
sys.getsizeof(trifecta_dividend_list)/(1024*1024*1024)
del(trifecta_dividend_list, trifecta_dividend_list_df)
sys.getsizeof(trifecta_df)/(1024*1024*1024) # Leave for now but may want to delete as pretty big


# Q3: is order important that makes up the probability (e.g. unlikely horse first or last etc), can test with model, features will be:
# horse 1,2 and 3 win odds and place odds, and combined odds, number runners, and leave it at that for now, output is 0 if lose otherwise actual dividend
def get_trifecta_pred_features_for_selection(trifecta_selection, win_horse_list, win_dividends, place_horse_list, place_dividends, runners, winning_selection, race_dividend):
    try:
        first_win_div = win_dividends[win_horse_list.index(trifecta_selection[0])]
        first_place_div = place_dividends[place_horse_list.index(trifecta_selection[0])]
        second_win_div = win_dividends[win_horse_list.index(trifecta_selection[1])]
        second_place_div = place_dividends[place_horse_list.index(trifecta_selection[1])]
        third_win_div = win_dividends[win_horse_list.index(trifecta_selection[2])]
        third_place_div = place_dividends[place_horse_list.index(trifecta_selection[2])]
        win_place_implied_prob = round((1/first_win_div) * (1/second_place_div - 1/second_win_div) * (1/third_place_div - 1/third_win_div)*0.5,5)
        excta_expected_dividend = 1/win_place_implied_prob
        if trifecta_selection==winning_selection:
            payout = race_dividend
        else:
            payout = 0
        return [first_win_div, first_place_div, second_win_div, second_place_div, third_win_div, third_place_div, excta_expected_dividend, runners, payout]
    except:
        return []

def get_trifecta_pred_features_for_all_selections(trifecta_selections, win_horse_list, win_dividends, place_horse_list, place_dividends, runners, winning_selection, race_dividend):
    return [get_trifecta_pred_features_for_selection(ts, win_horse_list, win_dividends, place_horse_list, place_dividends, runners, winning_selection, race_dividend) for ts in trifecta_selections]

trifecta_df = trifecta_df[trifecta_df['winner_and_dividend_agree']==True]
trifecta_df = trifecta_df[[len(dg[0]['horseNumbers'])==1 for dg in trifecta_df['tote_win_dividend_guide']]]
trifecta_df = trifecta_df[trifecta_df['yards'].notnull()]
trifecta_pred_df = [get_trifecta_pred_features_for_all_selections(tss, whl, wd, phl, pd, r, (f,s,t), rd) for tss, whl, wd, phl, pd, r, f, s, t, rd in zip(trifecta_df['trifecta_horse_combos'], trifecta_df['horses'], trifecta_df['dividends'], trifecta_df['place_horses'], trifecta_df['place_dividends'], trifecta_df['runners'], trifecta_df['tote_trifecta_first'], trifecta_df['tote_trifecta_second'], trifecta_df['tote_trifecta_third'], trifecta_df['tote_trifecta_dividend'])]
trifecta_pred_df = [item for sublist in trifecta_pred_df for item in sublist]
#data_to_inspect = [f for f in trifecta_pred_df[:1000] if f!=[]]
#data_to_inspect = pd.DataFrame(data_to_inspect, columns=['first_win_div','first_place_div','second_win_div','second_place_div','third_win_div','third_place_div','trifecta_expected_dividend','runners','payout'])
trifecta_pred_df = [f for f in trifecta_pred_df if f!=[]]
trifecta_pred_df = pd.DataFrame(trifecta_pred_df, columns=['first_win_div','first_place_div','second_win_div','second_place_div','third_win_div','third_place_div','trifecta_expected_dividend','runners','payout'])

#data_to_inspect = get_trifecta_pred_features_for_all_selections(trifecta_df['trifecta_horse_combos'][0], trifecta_df['horses'][0], trifecta_df['dividends'][0], trifecta_df['place_horses'][0], trifecta_df['place_dividends'][0], trifecta_df['runners'][0], (trifecta_df['tote_trifecta_first'][0], trifecta_df['tote_trifecta_second'][0], trifecta_df['tote_trifecta_third'][0]), trifecta_df['tote_trifecta_dividend'][0])
#data_to_inspect = [d for d in data_to_inspect if d!=[]]
#data_to_inspect = pd.DataFrame(data_to_inspect)
trifecta_pred_df_winners = trifecta_pred_df[trifecta_pred_df['payout']>0]
data_to_inspect = trifecta_pred_df_winners[:2000]
data_to_inspect = trifecta_df[[(44.8 in d) & (8.6 in d) & (62.2 in d) for d in trifecta_df['dividends']]]


expected_dividend_cutoff_min = 000 # to reduce size and also expensive strategies
expected_dividend_cutoff_max = 300 # to reduce size and also expensive strategies
trifecta_train_df = trifecta_pred_df[(trifecta_pred_df['trifecta_expected_dividend']>=expected_dividend_cutoff_min) & (trifecta_pred_df['trifecta_expected_dividend']<expected_dividend_cutoff_max)]
features = ['first_win_div','first_place_div','second_win_div','second_place_div','third_win_div','third_place_div','trifecta_expected_dividend','runners']
X_train, X_test, y_train, y_test = train_test_split(trifecta_train_df[features], trifecta_train_df['payout'], test_size=0.5, random_state=12)
jitter_inputs = True
if jitter_inputs:
    np.random.seed(seed=123)
    jitter_mean = 1
    jitter_std = 0.1
    jitter_table = stats.truncnorm.rvs(-1, 1, loc=jitter_mean, scale=jitter_std, size=X_train.shape)
    X_train.iloc[:,:6] = X_train.iloc[:,:6]*jitter_table[:,:6]
    X_train.iloc[:,:6] = X_train.iloc[:,:6].clip(lower=1.001)
    X_train.iloc[:,:6] = X_train.iloc[:,:6].round(1)
    jitter_table = stats.truncnorm.rvs(-1, 1, loc=jitter_mean, scale=jitter_std, size=X_test.shape)
    X_test.iloc[:,:6] = X_test.iloc[:,:6]*jitter_table[:,:6]
    X_test.iloc[:,:6] = X_test.iloc[:,:6].clip(lower=1.001)
    X_test.iloc[:,:6] = X_test.iloc[:,:6].round(1)

#check1 = X_test.iloc[:1000,:]
#check2 = X_test.iloc[:1000,:]
#check3 = jitter_table[:1000,:]
#check4 = X_train.iloc[:1000,:]
#check5 = X_train.iloc[:1000,:]
#del(check1,check2,check3,check4,check5)
#
#data_to_inspect = X_test.copy()
#(data_to_inspect < X_test*0.9999999).sum(axis=0)

#X_train.iloc[:,:6] = X_train.iloc[:,:6].clip(lower=1.001)
#X_test.iloc[:,:6] = X_test.iloc[:,:6].clip(lower=1.001)
#(X_test.iloc[:,:6]<1.001).sum(axis=0)
#data_to_inspect = X_train.iloc[:1000,]

xgb_trifecta_model = xgb.XGBRegressor(max_depth=6,n_estimators=30,random_state=123)
xgb_trifecta_model.fit(X_train, y_train)

train_sample_preds = xgb_trifecta_model.predict(X_train)
test_sample_preds = xgb_trifecta_model.predict(X_test)

pred_mins = [0,1,1.2,1.5,2,3,5,10,15,20]
trifecta_output_df = pd.DataFrame({'pred_min:':pred_mins})
trifecta_output_df['combos'] = [sum(test_sample_preds>x) for x in pred_mins]
trifecta_output_df['winners'] = [sum(y_test[test_sample_preds>x]>0) for x in pred_mins]
trifecta_output_df['winnings'] = [sum(y_test[test_sample_preds>x]) for x in pred_mins]
trifecta_output_df['return'] = trifecta_output_df['winnings']/trifecta_output_df['combos']

pred_mins = [0,1,1.2,1.5,2,3,5,10,15,20]
trifecta_output_df = pd.DataFrame({'pred_min:':pred_mins})
trifecta_output_df['combos'] = [sum(train_sample_preds>x) for x in pred_mins]
trifecta_output_df['winners'] = [sum(y_train[train_sample_preds>x]>0) for x in pred_mins]
trifecta_output_df['winnings'] = [sum(y_train[train_sample_preds>x]) for x in pred_mins]
trifecta_output_df['return'] = trifecta_output_df['winnings']/trifecta_output_df['combos']


# notes on results:
# seems like when prediction is high, over 5 or so, get good results, but need to run simulations with drawdown etc
# also need to check variation in dividends before races and apply some sort of distortion to data for training and testing
# if proves viable then check costs of running model on Azure or AWS, and set up first implementation
# next steps:
# need to build own prediction model and see results with that too
# and test other features like location, length of race, size of pot
# should maybe explore some different models and parameters too?

pred_cutoff=5
rough_combos_per_day = int(len(test_sample_preds)/500)
spend_per_day = [sum((test_sample_preds>pred_cutoff)[x:x+rough_combos_per_day]) for x in range(0,len(test_sample_preds),rough_combos_per_day)]
cumulative_spend = [sum(spend_per_day[:x]) for x in range(len(spend_per_day))]
winnings = [y*(p>pred_cutoff) for y, p in zip(y_test, test_sample_preds)]
winnings_per_day = [sum(winnings[x:x+rough_combos_per_day]) for x in range(0,len(test_sample_preds),rough_combos_per_day)]
cumulative_winnings = [sum(winnings_per_day[:x]) for x in range(len(spend_per_day))]

plt.plot(cumulative_spend)
plt.plot(cumulative_winnings)
sum(y_test>0)
sum(winnings)/sum([w>0 for w in winnings])

X_test_positive_preds = X_test[test_sample_preds>1]
X_test_positive_preds['preds'] = [p for p in test_sample_preds if p>1]
X_test_positive_preds['winnings'] = [w for w, p in zip(winnings, test_sample_preds) if p>1]

data_to_inspect = X_test[:1000]
data_to_inspect = trifecta_df[:100]

    

'''
Jackpot analysis
'''
# Q1: Do horses that make up 50% of implied prob of winning, actually win jackpot 0.5^6 = 1.5% of the time? Similarly for 0.25^6, 0.75^6 etc



# Q2: When the jackpot guarantee kicks in, can this give enough of a boost to push odds into favour?



'''
Other questions and thoughts
'''
# Can actually get past data for each horse's races and use that in predictions
# But to rpedict what, probability win? Expected payout? Would need separate models for different number of runners?



