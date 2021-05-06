'''
using data from sportinglife to model horse race times
'''

import pymysql
import sqlalchemy
import pandas as pd
import numpy as np
import sys
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
import statsmodels.api as sm
import itertools



'''
get data
'''
connect_string = 'mysql+pymysql://root:angunix1@localhost/horses'
sql_engine = sqlalchemy.create_engine(connect_string)
horses_data = pd.read_sql('''
                          SELECT * FROM training_data_6_relative_pr_no_nrs
                            WHERE 1
                            #AND pr_1_horse_time IS NOT NULL
                            #AND pr_2_horse_time IS NOT NULL
                            #AND pr_3_horse_time IS NOT NULL
                            #AND pr_4_horse_time IS NOT NULL
                            #AND pr_5_horse_time IS NOT NULL
                            #AND pr_6_horse_time IS NOT NULL
                            #AND pr_7_horse_time IS NOT NULL
                            #AND pr_8_horse_time IS NOT NULL
                            #AND pr_9_horse_time IS NOT NULL
                            #AND pr_10_horse_time IS NOT NULL
                            #AND did_not_finish = 0
                            #AND pr_1_did_not_finish =0
                            #AND pr_2_did_not_finish =0
                            #AND pr_3_did_not_finish =0
                            #AND pr_4_did_not_finish =0
                            #AND pr_5_did_not_finish =0
                            #AND pr_6_did_not_finish =0
                            #AND pr_7_did_not_finish =0
                            #AND pr_8_did_not_finish =0
                            #AND pr_9_did_not_finish =0
                            #AND pr_10_did_not_finish =0
                          ''',
                          con=sql_engine)
len(horses_data)
sys.getsizeof(horses_data)/(1024*1024*1024)


'''
data manipulation
'''
horses_data.columns[-60:]
ft_pr_details = ['horse_time', 'course',
       'surface', 'going', 'yards', 'runners',
       'prize1', 'number_of_placed_rides', 'handicap_pounds',
       'horse_age', 'horse_sex', 'horse_last_ran_days',
       'horse_form', 'country', 'weather', 'race_class',
       'going_main', 'going_grouped', 'race_type',
       'race_type_devised']

#features_to_copy = ['pr_'+str(1)+'_'+c for c in ft_pr_details]
#features_to_impute = ['pr_'+str(2)+'_'+c for c in ft_pr_details]
#horses_data.loc[:,features_to_impute][horses_data['pr_2_horse_time'].isnull()] = horses_data.loc[:,features_to_copy][horses_data['pr_2_horse_time'].isnull()].values
#horses_data.loc[:,features_to_impute] = np.where(horses_data['pr_2_horse_time'].isnull()[:,None], horses_data.loc[:,features_to_copy], horses_data.loc[:,features_to_impute])
for i in list(range(5,11)):
    features_to_copy = ['pr_'+str(i-1)+'_'+c for c in ft_pr_details]
    features_to_impute = ['pr_'+str(i)+'_'+c for c in ft_pr_details]
    column_to_check = 'pr_'+str(i)+'_horse_time'
    horses_data.loc[:,features_to_impute] = np.where(horses_data[column_to_check].isnull()[:,None], horses_data.loc[:,features_to_copy], horses_data.loc[:,features_to_impute])


## one hot encoding - omit for now as response coding works well with boosted trees
#horses_data = horses_data.merge(pd.get_dummies(horses_data['going_grouped']), left_index=True, right_index=True)
#horses_data = horses_data.merge(pd.get_dummies(horses_data['race_type']), left_index=True, right_index=True)
#
#going_grouped_types = ['heavy','soft','good','standard','fast','firm','sloppy','slow','yielding']
#for t in going_grouped_types:
#    horses_data[t] = (horses_data['going_grouped']==t)*1
#
#past_results_to_get_dummies = 5
#for i in range(past_results_to_get_dummies):
#    for t in going_grouped_types:
#        horses_data['pr_'+str(i+1)+'_'+t] = (horses_data['pr_'+str(i+1)+'_going_grouped']==t)*1
horse_sexes = ['g','m','f','c','h','r']
for t in horse_sexes:
    horses_data['horse_sex_'+t] = (horses_data['horse_sex']==t)*1


# add decimal odds and win payouts
def odds_parser(odds_string):
    try:
        odds_split = odds_string.split('/')
        decimal_odds = (int(odds_split[0])+int(odds_split[1]))/int(odds_split[1])
        return decimal_odds
    except:
        return 1

horses_data['decimal_odds'] = [odds_parser(o) for o in horses_data['betting_odds']]
for i in range(10):
    horses_data['pr_'+str(i+1)+'_decimal_odds'] = [odds_parser(o) for o in horses_data['pr_'+str(i+1)+'_betting_odds']]


# calculate horse speed and expected time based on past results
def horse_speed(race_length, race_time):
    try:
        return race_length/race_time
    except:
        return None


for i in range(10):
    horses_data['pr_'+str(i+1)+'_speed'] = [horse_speed(rl, rt) for rl, rt
                in zip(horses_data['pr_'+str(i+1)+'_yards'], horses_data['pr_'+str(i+1)+'_horse_time'])]
    horses_data['pr_'+str(i+1)+'_implied_time'] = [y/s for y, s in zip(
        horses_data['yards'], horses_data['pr_'+str(i+1)+'_speed'])]


def get_average_speed(speeds):
    speeds_list = np.array(speeds)[np.array(speeds)!=None]
    return sum(speeds_list)/len(speeds_list)

speeds_cols = ['pr_'+str(i+1)+'_speed' for i in range(10)]
horses_data['average_past_speed'] = [get_average_speed(horses_data.loc[i,speeds_cols]) for i in tqdm(horses_data.index)]
horses_data['expected_time'] = horses_data['yards']/horses_data['average_past_speed']


def convert_prize(p):
    try:
        return float(str(p).replace(' GBP', '').replace(' EUR', ''))
    except:
        return 0
horses_data['prize1'] = [convert_prize(p) for p in tqdm(horses_data['prize1'])]
for i in tqdm(range(10)):
    horses_data['pr_'+str(i+1)+'_prize1'] = [convert_prize(p) for p in horses_data['pr_'+str(i+1)+'_prize1']]


def convert_race_class(c):
    try:
        return int(c)
    except:
        return 99
horses_data['race_class'] = [convert_race_class(c) for c in tqdm(horses_data['race_class'])]
for i in tqdm(range(10)):
    horses_data['pr_'+str(i+1)+'_race_class'] = [convert_race_class(c) for c in horses_data['pr_'+str(i+1)+'_race_class']]


# make more relative past results
for i in tqdm(range(6)):
    horses_data['pr_'+str(i+1)+'_relative_runners'] = (
            horses_data['pr_'+str(i+1)+'_runners'] - horses_data['runners'])
    horses_data['pr_'+str(i+1)+'_relative_handicap'] = (
            horses_data['pr_'+str(i+1)+'_handicap_pounds'] - horses_data['handicap_pounds'])
    horses_data['pr_'+str(i+1)+'_relative_going'] = (
            horses_data['pr_'+str(i+1)+'_going_numerical'] - horses_data['going_numerical'])
    horses_data['pr_'+str(i+1)+'_relative_race_type'] = (
            horses_data['pr_'+str(i+1)+'_race_type_numerical'] - horses_data['race_type_numerical'])
    horses_data['pr_'+str(i+1)+'_relative_weather'] = (
            horses_data['pr_'+str(i+1)+'_weather_numerical'] - horses_data['weather_numerical'])


## add win percentage
past_places_cols = ['pr_'+str(i+1)+'_finish_position_for_ordering' for i in range(10)]
horses_data['past_win_pc'] = (
        (horses_data.loc[:, past_places_cols]==1).sum(axis=1)
        / (horses_data.loc[:, past_places_cols]).notnull().sum(axis=1)
)

## add average finish position ratio
runners_cols = ['pr_'+str(i+1)+'_runners' for i in range(10)]
horses_data['past_avg_fp_ratio'] = (
        pd.DataFrame(np.array(horses_data.loc[:, past_places_cols])
                     / np.array(horses_data.loc[:, runners_cols])).apply(np.nanmean, axis=1)
)

horses_data['winner'] = (horses_data['finish_position_for_ordering']==1)*1

data_to_inspect = horses_data[:1000]

horses_data['payout'] = horses_data['winner'] * horses_data['decimal_odds']

list(horses_data.columns)


'''
horse-race pairings
'''

race_horse_lists = horses_data.groupby(['race_id'])['horse_id'].apply(
    lambda x: list(np.random.permutation(x))).reset_index()
race_horse_lists.columns = ['race_id', 'horses_list']

race_horse_pairs = []
for i in tqdm(range(len(race_horse_lists))):
    race_horse_pairs = race_horse_pairs + (
        [[race_horse_lists['race_id'].iloc[i]] + list(h)
         for h in list(itertools.combinations(race_horse_lists['horses_list'].iloc[i], 2))])

race_horse_pairs = pd.DataFrame(race_horse_pairs, columns=['race_id', 'horse_1', 'horse_2'])

data_to_inspect = race_horse_pairs.iloc[:1000, :]



'''
get features and subset data
'''

# select features
features_extra_info_for_results = ['race_id', 'horse_id', 'race_date', 'betting_odds',
                                   'finish_position', 'did_not_finish', #'jockey_id',
                                   'payout', 'winner', 'horse_time']
features_current_race = ['yards', 'runners', 'handicap_pounds', 'horse_age', 'race_class', 'prize1',
                         'past_win_pc', 'past_avg_fp_ratio',
                         'horse_sex_g','horse_sex_m','horse_sex_f','horse_sex_c','horse_sex_h','horse_sex_r',
                         'horse_last_ran_days',
                         #'going_grouped_horse_time_rc', 'race_type_horse_time_rc', 'weather_horse_time_rc',
                         'going_numerical', 'race_type_numerical', 'weather_numerical',
                         'average_past_speed', 'expected_time',
                         'decimal_odds',
                         #'course', 'race_type_devised' # for response coding
                         ]# + going_grouped_types
features_pr = ['implied_time', 'relative_yards', 'speed', 'horse_time', 'finish_position_for_ordering', #'yards',
               'runners', 'handicap_pounds',
               #'relative_runners', 'relative_handicap',
               'going_numerical', 'race_type_numerical', 'weather_numerical',
               #'relative_going', 'relative_race_type', 'relative_weather',
               'race_class', 'prize1',
               #'going_grouped_horse_time_rc', 'race_type_horse_time_rc', 'weather_horse_time_rc',
               'decimal_odds',
               #'course', 'race_type_devised' # for response coding
               ]# + going_grouped_types
number_past_results = 3
features_prs = []
for i in range(1, number_past_results+1):
    features_prs = features_prs + ['pr_'+str(i)+'_'+pr for pr in features_pr]

features = features_extra_info_for_results + features_current_race + features_prs

train_data = horses_data[features]

data_to_inspect = train_data.isnull().sum(axis=0)
train_data = train_data[train_data['yards'].notnull()]
train_data = train_data[train_data['horse_last_ran_days'].notnull()]
train_data = train_data[train_data['average_past_speed'].notnull()]
train_data = train_data[train_data['pr_1_relative_yards'].notnull()]
train_data = train_data[train_data['pr_2_relative_yards'].notnull()]
train_data = train_data[train_data['pr_3_relative_yards'].notnull()]
train_data = train_data[train_data['pr_4_relative_yards'].notnull()]
train_data = train_data[train_data['pr_5_relative_yards'].notnull()]
train_data = train_data[train_data['pr_6_relative_yards'].notnull()]
# train_data = train_data[train_data['pr_7_relative_yards'].notnull()]
# train_data = train_data[train_data['pr_8_relative_yards'].notnull()]
# train_data = train_data[train_data['pr_9_relative_yards'].notnull()]
# train_data = train_data[train_data['pr_10_relative_yards'].notnull()]
#train_data = train_data[train_data['pr_6_finish_position_for_ordering'].notnull()]
data_to_inspect = train_data.isnull().sum(axis=0)


# add features to race_horse_pairs
race_horse_pairs_train_data = race_horse_pairs.merge(
    train_data, how='inner', left_on=['race_id', 'horse_1'], right_on=['race_id', 'horse_id'])

race_horse_pairs_train_data = race_horse_pairs_train_data.merge(
    train_data, how='inner', left_on=['race_id', 'horse_2'], right_on=['race_id', 'horse_id'],
    suffixes=('_1', '_2'))

#data_to_inspect = race_horse_pairs_train_data.iloc[:100, :]

# create y and features
race_horse_pairs_train_data['y'] = (
    race_horse_pairs_train_data['finish_position_1'] < race_horse_pairs_train_data['finish_position_2'])*1

features_y = ['y']
features_x_1 = [f+'_1' for f in features_current_race + features_prs]
features_x_2 = [f+'_2' for f in features_current_race + features_prs]
features_x = features_x_1 + features_x_2

test_races = race_horse_pairs_train_data.loc[
    (race_horse_pairs_train_data['race_date_1'] > '2019-09-22'), 'race_id'].unique()
train_races = race_horse_pairs_train_data.loc[
    ~race_horse_pairs_train_data['race_id'].isin(test_races), 'race_id'].unique()
train_mask = race_horse_pairs_train_data['race_id'].isin(train_races)
test_mask = race_horse_pairs_train_data['race_id'].isin(test_races)

train_X = race_horse_pairs_train_data.loc[train_mask, features_x]
train_y = race_horse_pairs_train_data.loc[train_mask, features_y]
test_X = race_horse_pairs_train_data.loc[test_mask, features_x]
test_y = race_horse_pairs_train_data.loc[test_mask, features_y]

# num_non_pred_features = len(features_y + features_extra_info_for_results)
# train_extra_info = train_data.loc[train_idx, features[1:num_non_pred_features]]
# test_extra_info = train_data.loc[test_idx, features[1:num_non_pred_features]]


'''
train models
'''
# len(train_y)
# # lin mod
# linMod = sm.OLS(train_y, sm.add_constant(train_X))
# linModFit = linMod.fit()
#
# linModPreds_train = linModFit.predict(sm.add_constant(train_X))
# linModPreds_test = linModFit.predict(sm.add_constant(test_X))
# sum((linModPreds_train - train_y)**2)/len(train_y)
# sum((linModPreds_test - test_y)**2)/len(test_y)
# linModFit.summary()


# xgb mod
params = {
    'max_depth':2,
    'min_child_weight': 50,
    'eta':.2,
#    'subsample': 1,
#    'colsample_bytree': 1,
    'objective': 'binary:logistic',  # 'reg:linear',  #
    'eval_metric': 'auc'  # 'rmse',  #
}
num_boost_round = 5
early_stopping = 10

dtrain = xgb.DMatrix(train_X, label=train_y)
dtest = xgb.DMatrix(test_X, label=test_y)
xgbMod = xgb.train(params,
                   dtrain,
                   num_boost_round=num_boost_round,
                   early_stopping_rounds=early_stopping,
                   evals=[(dtest, "Test")]
                   )

xgbModPreds_train = xgbMod.predict(dtrain)
xgbModPreds_test = xgbMod.predict(dtest)

# % right
(sum((xgbModPreds_train > 0.5) & (train_y['y'] == 1)) + sum((xgbModPreds_train <= 0.5) & (train_y['y'] == 0)))/len(train_y)
(sum((xgbModPreds_test > 0.5) & (test_y['y'] == 1)) + sum((xgbModPreds_test <= 0.5) & (test_y['y'] == 0)))/len(test_y)


'''
per race analysis
'''
race_horse_pairs_train_data['pred'] = None
race_horse_pairs_train_data.loc[train_mask, 'pred'] = xgbModPreds_train
race_horse_pairs_train_data.loc[test_mask, 'pred'] = xgbModPreds_test

data_to_inspect = race_horse_pairs_train_data.iloc[-100:, :]

race_horse_pairs_train_data['pred_1'] = race_horse_pairs_train_data['pred']
race_horse_pairs_train_data['pred_2'] = 1- race_horse_pairs_train_data['pred']

race_horse_probs_1 = race_horse_pairs_train_data.groupby(
    ['race_id', 'horse_1', 'finish_position_1', 'decimal_odds_1', 'runners_1'])['pred_1'].apply(lambda x: list(x)).reset_index()
race_horse_probs_1.columns = ['race_id', 'horse_id', 'finish_position', 'decimal_odds', 'number_runners', 'probs_1']
race_horse_probs_2 = race_horse_pairs_train_data.groupby(
    ['race_id', 'horse_2', 'finish_position_2', 'decimal_odds_2', 'runners_2'])['pred_2'].apply(lambda x: list(x)).reset_index()
race_horse_probs_2.columns = ['race_id', 'horse_id', 'finish_position', 'decimal_odds', 'number_runners', 'probs_2']

race_horse_probs = race_horse_probs_1.merge(race_horse_probs_2, how='outer',
                                            on=['race_id', 'horse_id', 'finish_position', 'decimal_odds', 'number_runners'])
def combine_prob_lists(list_1, list_2):
    if list_1 != list_1:
        list_1 = []
    if list_2 != list_2:
        list_2 = []
    return list_1 + list_2
race_horse_probs['probs'] = [combine_prob_lists(l1, l2) for l1, l2
                             in zip(race_horse_probs['probs_1'], race_horse_probs['probs_2'])]
race_horse_probs = race_horse_probs.drop(columns=['probs_1', 'probs_2'])
race_horse_probs = race_horse_probs.sort_values(['race_id', 'finish_position'])

race_horse_probs['min_prob'] = [min(p) for p in race_horse_probs['probs']]
race_horse_probs['prod_prob'] = [np.prod(p) for p in race_horse_probs['probs']]

race_horse_probs_runners_checks = race_horse_probs.groupby('race_id').aggregate(
    {'horse_id': len, 'finish_position': lambda x: sum(np.array(x) == 1) > 0, 'prod_prob': sum}).reset_index()
race_horse_probs_runners_checks.columns = ['race_id', 'horses_in_data', 'contains_first', 'sum_probs']

race_horse_probs_with_all_runners = race_horse_probs.merge(race_horse_probs_runners_checks, how='left', on='race_id')
race_horse_probs_with_all_runners['all_runners_check'] = (
        race_horse_probs_with_all_runners['number_runners'] == race_horse_probs_with_all_runners['horses_in_data'])
race_horse_probs_with_all_runners['pc_runners'] = (
        race_horse_probs_with_all_runners['horses_in_data'] / race_horse_probs_with_all_runners['number_runners'])
pc_runners_cutoff = 1
race_horse_probs_with_all_runners['pc_runners_check'] = race_horse_probs_with_all_runners['pc_runners'] >= pc_runners_cutoff

race_horse_probs_with_all_runners = race_horse_probs_with_all_runners[
    race_horse_probs_with_all_runners['contains_first'] == 1]
race_horse_probs_with_all_runners = race_horse_probs_with_all_runners[
    race_horse_probs_with_all_runners['pc_runners_check'] == 1]

prob_win_cutoff = 0.5
# win %
sum((race_horse_probs_with_all_runners['min_prob'] > prob_win_cutoff) &
    (race_horse_probs_with_all_runners['finish_position'] == 1))/sum(
    race_horse_probs_with_all_runners['min_prob'] > prob_win_cutoff)
# % return
sum(((race_horse_probs_with_all_runners['min_prob'] > prob_win_cutoff) &
     (race_horse_probs_with_all_runners['finish_position'] == 1))*race_horse_probs_with_all_runners['decimal_odds'])/sum(
    race_horse_probs_with_all_runners['min_prob'] > prob_win_cutoff)

race_horse_probs_with_all_runners_test = race_horse_probs_with_all_runners[
    race_horse_probs_with_all_runners['race_id'].isin(test_races)]
# win %
sum((race_horse_probs_with_all_runners_test['min_prob'] > prob_win_cutoff) &
    (race_horse_probs_with_all_runners_test['finish_position'] == 1))/sum(
    race_horse_probs_with_all_runners_test['min_prob'] > prob_win_cutoff)
# % return
sum(((race_horse_probs_with_all_runners_test['min_prob'] > prob_win_cutoff) &
     (race_horse_probs_with_all_runners_test['finish_position'] == 1))*race_horse_probs_with_all_runners_test['decimal_odds'])/sum(
    race_horse_probs_with_all_runners_test['min_prob'] > prob_win_cutoff)

# random return
sum((race_horse_probs_with_all_runners_test['finish_position'] == 1)*race_horse_probs_with_all_runners_test['decimal_odds'])/len(
    race_horse_probs_with_all_runners_test)

# bookies favourite win %
bookies_favourites = race_horse_probs_with_all_runners.groupby('race_id')['decimal_odds'].min().reset_index()
bookies_favourites.columns = ['race_id', 'favourite_odds']
race_horse_probs_with_all_runners = race_horse_probs_with_all_runners.merge(
    bookies_favourites, how='left', on='race_id')
race_horse_probs_with_all_runners_test = race_horse_probs_with_all_runners_test.merge(
    bookies_favourites, how='left', on='race_id')
race_horse_probs_with_all_runners['bookies_favourite'] = (
    race_horse_probs_with_all_runners['decimal_odds'] == race_horse_probs_with_all_runners['favourite_odds'])*1
race_horse_probs_with_all_runners_test['bookies_favourite'] = (
    race_horse_probs_with_all_runners_test['decimal_odds'] == race_horse_probs_with_all_runners_test['favourite_odds'])*1

bet_on_races = race_horse_probs_with_all_runners.loc[
    race_horse_probs_with_all_runners['min_prob'] > prob_win_cutoff, 'race_id']

sum(race_horse_probs_with_all_runners['bookies_favourite'] *
    race_horse_probs_with_all_runners['race_id'].isin(bet_on_races) *
    (race_horse_probs_with_all_runners['finish_position'] == 1)) / sum(
    race_horse_probs_with_all_runners['bookies_favourite'] *
    race_horse_probs_with_all_runners['race_id'].isin(bet_on_races))

sum(race_horse_probs_with_all_runners_test['bookies_favourite'] *
    race_horse_probs_with_all_runners_test['race_id'].isin(bet_on_races) *
    (race_horse_probs_with_all_runners_test['finish_position'] == 1)) / sum(
    race_horse_probs_with_all_runners_test['bookies_favourite'] *
    race_horse_probs_with_all_runners_test['race_id'].isin(bet_on_races))

# value bets (do prod_prob / sum_prob)
race_horse_probs_with_all_runners['scaled_prob'] = (
    race_horse_probs_with_all_runners['prod_prob'] / race_horse_probs_with_all_runners['sum_probs'])
race_horse_probs_with_all_runners_test['scaled_prob'] = (
    race_horse_probs_with_all_runners_test['prod_prob'] / race_horse_probs_with_all_runners_test['sum_probs'])
race_horse_probs_with_all_runners['my_odds'] = (1 / race_horse_probs_with_all_runners['scaled_prob'])
race_horse_probs_with_all_runners_test['my_odds'] = (1 / race_horse_probs_with_all_runners_test['scaled_prob'])
race_horse_probs_with_all_runners['bookies_prob'] = (1 / race_horse_probs_with_all_runners['decimal_odds'])
race_horse_probs_with_all_runners_test['bookies_prob'] = (1 / race_horse_probs_with_all_runners_test['decimal_odds'])

margin = 0.2
sum(((race_horse_probs_with_all_runners['bookies_prob'] + margin) < race_horse_probs_with_all_runners['scaled_prob']) *
    (race_horse_probs_with_all_runners['finish_position'] == 1) *
    race_horse_probs_with_all_runners['decimal_odds']) / sum(
    ((race_horse_probs_with_all_runners['bookies_prob'] + margin) < race_horse_probs_with_all_runners['scaled_prob']))

sum(((race_horse_probs_with_all_runners_test['bookies_prob'] + margin) < race_horse_probs_with_all_runners_test['scaled_prob']) *
    (race_horse_probs_with_all_runners_test['finish_position'] == 1) *
    race_horse_probs_with_all_runners_test['decimal_odds']) / sum(
    ((race_horse_probs_with_all_runners_test['bookies_prob'] + margin) < race_horse_probs_with_all_runners_test['scaled_prob']))

# win %s
sum(((race_horse_probs_with_all_runners_test['bookies_prob'] + margin) < race_horse_probs_with_all_runners_test['scaled_prob']) *
    (race_horse_probs_with_all_runners_test['finish_position'] == 1)) / sum(
    ((race_horse_probs_with_all_runners_test['bookies_prob'] + margin) < race_horse_probs_with_all_runners_test['scaled_prob']))

# third place bets?
# use bookies odds in model?
# model to predict winnings (can work out which horses are underrated more)?