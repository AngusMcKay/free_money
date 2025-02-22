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
import pickle



'''
get data
'''
# horse data first
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
jockey data to add on
'''
connect_string = 'mysql+pymysql://root:angunix1@localhost/horses'
sql_engine = sqlalchemy.create_engine(connect_string)
jockeys_data = pd.read_sql('''
                          SELECT    horse_id, race_id,
                                    pr_1_finish_position_for_ordering,
                                    pr_2_finish_position_for_ordering,
                                    pr_3_finish_position_for_ordering,
                                    pr_4_finish_position_for_ordering,
                                    pr_5_finish_position_for_ordering,
                                    pr_6_finish_position_for_ordering,
                                    pr_7_finish_position_for_ordering,
                                    pr_8_finish_position_for_ordering,
                                    pr_9_finish_position_for_ordering,
                                    pr_10_finish_position_for_ordering,
                                    pr_1_runners,
                                    pr_2_runners,
                                    pr_3_runners,
                                    pr_4_runners,
                                    pr_5_runners,
                                    pr_6_runners,
                                    pr_7_runners,
                                    pr_8_runners,
                                    pr_9_runners,
                                    pr_10_runners
                          FROM jockeys_data_combined_no_nrs_with_past_results
                            WHERE 1
                          ''',
                          con=sql_engine)

## add win percentage
past_places_cols = ['pr_'+str(i+1)+'_finish_position_for_ordering' for i in range(10)]
jockeys_data['jockey_past_win_pc'] = (
        (jockeys_data.loc[:, past_places_cols]==1).sum(axis=1)
        / (jockeys_data.loc[:, past_places_cols]).notnull().sum(axis=1)
)

## add average finish position ratio
runners_cols = ['pr_'+str(i+1)+'_runners' for i in range(10)]
jockeys_data['jockey_past_avg_fp_ratio'] = (
        pd.DataFrame(np.array(jockeys_data.loc[:, past_places_cols])
                     / np.array(jockeys_data.loc[:, runners_cols])).apply(np.nanmean, axis=1)
)

# add features to horses data
horses_data = horses_data.merge(jockeys_data[['horse_id', 'race_id', 'jockey_past_win_pc', 'jockey_past_avg_fp_ratio']],
                                how='left', on=['horse_id', 'race_id'])

len(horses_data)



'''
get best comparison pr using race similarity model
'''
with open('tote/models/race_similarity_20200412.pickle', 'rb') as f:
    xgbMod_rs = pickle.load(f)

features_current_race_rs = [
    'yards', 'runners', 'handicap_pounds', 'horse_age', 'race_class', 'prize1',
    'going_numerical', 'race_type_numerical', 'weather_numerical']
features_pr_rs = [
    'pr_speed', 'pr_horse_time', 'pr_implied_time', 'pr_yards', 'pr_runners', 'pr_handicap_pounds', 'pr_horse_age',
    'pr_race_class', 'pr_prize1', 'pr_going_numerical', 'pr_race_type_numerical', 'pr_weather_numerical']

for i in tqdm(range(6)):
    features_pr_rs_i = [f.replace('pr_', 'pr_'+str(i+1)+'_') for f in features_pr_rs]
    horses_data_subset = horses_data[features_current_race_rs + features_pr_rs_i]
    horses_data_subset.columns = features_current_race_rs + features_pr_rs
    horses_data_subset['pr_number'] = i+1

    # Not done imputation, instead set races will null yards to pr number (so will take more recent pr as most similar

    horses_data_subset['yards_ratio'] = horses_data_subset['yards'] / horses_data_subset['pr_yards']
    horses_data_subset['runners_ratio'] = horses_data_subset['runners'] / horses_data_subset['pr_runners']
    horses_data_subset['handicap_diff'] = horses_data_subset['handicap_pounds'] - horses_data_subset[
        'pr_handicap_pounds']
    horses_data_subset['same_course'] = (horses_data['course'] == horses_data['pr_'+str(i+1)+'_course']) * 1

    features_x = [
        'yards', 'runners', 'handicap_pounds', 'horse_age', 'race_class', 'prize1',
        'going_numerical', 'race_type_numerical', 'weather_numerical',
        'pr_speed', 'pr_horse_time', 'pr_implied_time', 'pr_yards',
        'pr_runners', 'pr_handicap_pounds', 'pr_horse_age', 'pr_race_class', 'pr_prize1',
        'pr_going_numerical', 'pr_race_type_numerical', 'pr_weather_numerical', 'pr_number',
        'yards_ratio', 'runners_ratio', 'handicap_diff', 'same_course']
    dpred = xgb.DMatrix(horses_data_subset[features_x])

    race_similarities = xgbMod_rs.predict(dpred)

    horses_data['pr_'+str(i+1)+'_race_similarity'] = race_similarities

    horses_data.loc[horses_data['pr_'+str(i+1)+'_yards'].isnull(), 'pr_'+str(i+1)+'_race_similarity'] = i+1

pr_rs1_columns = [f.replace('pr_1_', 'pr_rs1_') for f in horses_data.columns if 'pr_1_' in f]
pr_rs2_columns = [f.replace('pr_1_', 'pr_rs2_') for f in horses_data.columns if 'pr_1_' in f]
pr_rs3_columns = [f.replace('pr_1_', 'pr_rs3_') for f in horses_data.columns if 'pr_1_' in f]

for f in tqdm(range(len(pr_rs1_columns))):
    horses_data[pr_rs1_columns[f]] = None
    horses_data[pr_rs2_columns[f]] = None
    horses_data[pr_rs3_columns[f]] = None

rs_columns = ['pr_'+str(i+1)+'_race_similarity' for i in range(6)]
horses_data['pr_rs1'] = horses_data[rs_columns].apply(np.argmin, axis=1, raw=True) + 1
horses_data['pr_rs2'] = horses_data[rs_columns].apply(lambda x: np.argsort(x)[1], axis=1, raw=True) + 1
horses_data['pr_rs3'] = horses_data[rs_columns].apply(lambda x: np.argsort(x)[2], axis=1, raw=True) + 1

for i in tqdm(range(6)):
    pr_cols_to_copy = [f for f in horses_data.columns if 'pr_'+str(i+1)+'_' in f]
    for f in range(len(pr_rs1_columns)):
        horses_data.loc[horses_data['pr_rs1'] == i + 1, pr_rs1_columns[f]] = horses_data.loc[
            horses_data['pr_rs1'] == i + 1, pr_cols_to_copy[f]]
        horses_data.loc[horses_data['pr_rs2'] == i + 1, pr_rs2_columns[f]] = horses_data.loc[
            horses_data['pr_rs2'] == i + 1, pr_cols_to_copy[f]]
        horses_data.loc[horses_data['pr_rs3'] == i + 1, pr_rs3_columns[f]] = horses_data.loc[
            horses_data['pr_rs3'] == i + 1, pr_cols_to_copy[f]]


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


'''
get features and subset data
'''

# select features
features_extra_info_for_results = ['race_id', 'horse_id', 'race_date', 'betting_odds',
                                   'finish_position', 'did_not_finish', #'jockey_id',
                                   'payout', 'winner', 'horse_time', 'decimal_odds']
features_current_race = ['yards', 'runners', 'handicap_pounds', 'horse_age', 'race_class', 'prize1',
                         'past_win_pc', 'past_avg_fp_ratio', 'jockey_past_win_pc', 'jockey_past_avg_fp_ratio',
                         'horse_sex_g','horse_sex_m','horse_sex_f','horse_sex_c','horse_sex_h','horse_sex_r',
                         'horse_last_ran_days',
                         #'going_grouped_horse_time_rc', 'race_type_horse_time_rc', 'weather_horse_time_rc',
                         'going_numerical', 'race_type_numerical', 'weather_numerical',
                         'average_past_speed', #'expected_time',
                         #'decimal_odds',
                         #'course', 'race_type_devised' # for response coding
                         ]# + going_grouped_types
features_pr = ['implied_time', 'relative_yards', 'speed', 'horse_time', 'finish_position_for_ordering', #'yards',
               'runners', 'handicap_pounds',
               #'relative_runners', 'relative_handicap',
               'going_numerical', 'race_type_numerical', 'weather_numerical',
               #'relative_going', 'relative_race_type', 'relative_weather',
               'race_class', 'prize1',
               #'going_grouped_horse_time_rc', 'race_type_horse_time_rc', 'weather_horse_time_rc',
               'decimal_odds', 'race_similarity'
               #'course', 'race_type_devised' # for response coding
               ]# + going_grouped_types

number_past_results = 3
use_rs_prs = False
features_prs = []
for i in range(1, number_past_results+1):
    if use_rs_prs:
        features_prs = features_prs + ['pr_rs' + str(i) + '_' + pr for pr in features_pr]
    else:
        features_prs = features_prs + ['pr_'+str(i)+'_'+pr for pr in features_pr]

features = features_extra_info_for_results + features_current_race + features_prs

train_data = horses_data[features]

train_data.columns[train_data.isnull().sum(axis=0)>0]
train_data = train_data[train_data['yards'].notnull()]
train_data = train_data[train_data['pr_rs3_relative_yards'].notnull()]
train_data = train_data[train_data['jockey_past_win_pc'].notnull()]
train_data = train_data[train_data['jockey_past_avg_fp_ratio'].notnull()]
train_data = train_data[train_data['horse_last_ran_days'].notnull()]
train_data = train_data[train_data['average_past_speed'].notnull()]

train_data = train_data[train_data['horse_last_ran_days'].notnull()]
train_data = train_data[train_data['average_past_speed'].notnull()]
train_data = train_data[train_data['pr_1_relative_yards'].notnull()]
train_data = train_data[train_data['pr_2_relative_yards'].notnull()]
train_data = train_data[train_data['pr_3_relative_yards'].notnull()]
# train_data = train_data[train_data['pr_4_relative_yards'].notnull()]
# train_data = train_data[train_data['pr_5_relative_yards'].notnull()]
# train_data = train_data[train_data['pr_6_relative_yards'].notnull()]
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

# convert dtypes if need be
current_dtypes = train_X.dtypes
for i in tqdm(range(len(current_dtypes))):
    if current_dtypes[i] == 'object':
        train_X[current_dtypes.index[i]] = train_X[current_dtypes.index[i]].astype(float)

current_dtypes = test_X.dtypes
for i in tqdm(range(len(current_dtypes))):
    if current_dtypes[i] == 'object':
        test_X[current_dtypes.index[i]] = test_X[current_dtypes.index[i]].astype(float)

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
    'max_depth':3,
    'min_child_weight': 20,
    'eta':.2,
#    'subsample': 1,
#    'colsample_bytree': 1,
    'objective': 'binary:logistic',  # 'reg:linear',  #
    'eval_metric': 'auc'  # 'rmse',  #
}
num_boost_round = 100
early_stopping = 50

dtrain = xgb.DMatrix(train_X, label=train_y)
dtest = xgb.DMatrix(test_X, label=test_y)
xgbMod = xgb.train(params,
                   dtrain,
                   num_boost_round=num_boost_round,
                   early_stopping_rounds=early_stopping,
                   evals=[(dtest, "Test")]
                   )

xgbModPreds_train = xgbMod.predict(dtrain, ntree_limit=xgbMod.best_ntree_limit)
xgbModPreds_test = xgbMod.predict(dtest, ntree_limit=xgbMod.best_ntree_limit)

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

# random win %
1 / (sum(race_horse_probs_with_all_runners['number_runners']/race_horse_probs_with_all_runners['horses_in_data']) /
     sum(1/race_horse_probs_with_all_runners['horses_in_data']))
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

margin = 0.4
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



feature_importances_dict = xgbMod.get_score(importance_type='gain')
feature_importances_df = pd.DataFrame({'feature':list(feature_importances_dict.keys()),
                                       'importance':list(feature_importances_dict.values())})
feature_importances_df = feature_importances_df.sort_values(by='importance', ascending=False)
feature_importances_df['importance'] = feature_importances_df['importance']/sum(feature_importances_df['importance'])

feature_importances_df
feature_importances_df[feature_importances_df['feature'] == 'jockey_past_win_pc_2']
feature_importances_df[feature_importances_df['feature'] == 'jockey_past_avg_fp_ratio_2']

list(feature_importances_df['feature'])
list(feature_importances_df['importance'])

# third place bets?
# use bookies odds in model?
# model to predict winnings (can work out which horses are underrated more)?


'''
predict winnings
'''



