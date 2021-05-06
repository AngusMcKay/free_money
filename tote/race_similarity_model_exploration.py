'''
take difference in variables and use to predict absolute % difference in speed or other similar outcome
use model to find most similar race(s) for each horse
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
import seaborn as sns
import shap



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


# add horse speed for current race
horses_data['speed'] = horses_data['yards'] / horses_data['horse_time']


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

horses_data['payout'] = horses_data['winner'] * horses_data['decimal_odds']




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




'''
horse-pr pairings
'''
list(horses_data.columns)
current_race_outcomes = ['speed', 'horse_time']
pr_outcome_variables = ['speed', 'horse_time', 'implied_time']
numerical_comparison_variables = ['yards', 'runners', 'handicap_pounds', 'horse_age', 'race_class', 'prize1',
                                  'horse_last_ran_days', 'going_numerical', 'race_type_numerical', 'weather_numerical']
categorical_comparison_variables = ['course']

horse_pr_pair_data = []
for i in tqdm(range(6)):
    pr_outcome_variables_pr = ['pr_' + str(i+1) + '_' + v for v in pr_outcome_variables]
    numerical_comparison_variables_pr = ['pr_' + str(i+1) + '_' + v for v in numerical_comparison_variables]
    categorical_comparison_variables_pr = ['pr_' + str(i+1) + '_' + v for v in categorical_comparison_variables]
    horse_pr_pair_df = horses_data[current_race_outcomes + pr_outcome_variables_pr +
                                   numerical_comparison_variables + categorical_comparison_variables +
                                   numerical_comparison_variables_pr + categorical_comparison_variables_pr]
    horse_pr_pair_df.columns = [c.replace('pr_' + str(i+1), 'pr') for c in horse_pr_pair_df.columns]
    horse_pr_pair_df['pr_number'] = i+1
    horse_pr_pair_data.append(horse_pr_pair_df)

horse_pr_pair_data = pd.concat(horse_pr_pair_data)

null_counts = horse_pr_pair_data.isnull().sum(axis=0)
null_counts

horse_pr_pair_data = horse_pr_pair_data[horse_pr_pair_data['pr_speed'].notnull()]
horse_pr_pair_data = horse_pr_pair_data[horse_pr_pair_data['speed'].notnull()]
horse_pr_pair_data = horse_pr_pair_data[horse_pr_pair_data['horse_time'] != 0]


data_describe = horse_pr_pair_data.describe(
    percentiles=[0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 0.9, 0.925, 0.95, 0.975, 0.99, 0.995])


'''
view data
'''
features_to_plot = ['speed', 'horse_time', 'yards', #'runners', 'handicap_pounds',
                    #'horse_age', 'race_class', 'prize1',
                    'horse_last_ran_days', 'going_numerical', 'race_type_numerical', 'weather_numerical'
                    ]
sns.pairplot(horse_pr_pair_data.loc[horse_pr_pair_data.index[:1000], features_to_plot])



100*sum(horses_data['speed']<9)/len(horses_data)


### NOTES ###
## SLOW HORSES
# horses unusually slow at times (i.e. < 10 yards/second). Reviewing data it seems like they sometimes just come in...
# ... real slow in last place, maybe cruising or after a fall.
# only really interested in horses who are challenging, otherwise past times and speeds are unreliable, so there is...
# ... two parts, 1) horse strength *if it is challening for win*, and 2) likelihood of being in the running during race
# could take out slow speed horses from current race and include past horse speed as variable (and past speed >10)

# remove extreme values for speed, horse time, and pr equivalents, other variable all look fine
horse_pr_pair_data = horse_pr_pair_data[horse_pr_pair_data['speed'] <= 50]

data_describe = horse_pr_pair_data.describe(
    percentiles=[0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 0.9, 0.925, 0.95, 0.975, 0.99, 0.995])

'''
generate race comparison features
'''
horse_pr_pair_data['yards_ratio'] = horse_pr_pair_data['yards'] / horse_pr_pair_data['pr_yards']
horse_pr_pair_data['runners_ratio'] = horse_pr_pair_data['runners'] / horse_pr_pair_data['pr_runners']
horse_pr_pair_data['handicap_diff'] = horse_pr_pair_data['handicap_pounds'] - horse_pr_pair_data['pr_handicap_pounds']
horse_pr_pair_data['same_course'] = (horse_pr_pair_data['course'] == horse_pr_pair_data['pr_course'])*1

features_x = [
    'yards', 'runners', 'handicap_pounds', 'horse_age', 'race_class','prize1', #'horse_last_ran_days',
    'going_numerical', 'race_type_numerical', 'weather_numerical',
    'pr_speed', 'pr_horse_time', 'pr_implied_time', 'pr_yards',
    'pr_runners', 'pr_handicap_pounds', 'pr_horse_age', 'pr_race_class',
    'pr_prize1', #'pr_horse_last_ran_days',
    'pr_going_numerical', 'pr_race_type_numerical', 'pr_weather_numerical', 'pr_number',
    'yards_ratio', 'runners_ratio', 'handicap_diff', 'same_course']
outcome = 'speed'  # 'horse_time'

'''
subset data
'''

train_data = horse_pr_pair_data[features_x + [outcome]]
train_data.isnull().sum(axis=0)

# split data into three, 40% to train predict time model, 40% to then train error model, and remaining 20% to test
train, test = train_test_split(horse_pr_pair_data, test_size=0.2, random_state=123)
train_1, train_2 = train_test_split(train, test_size=0.5, random_state=123)

'''
predict time model
'''
train_1_X1 = train_1[features_x]
train_1_y1 = train_1[outcome]

train_2_X1 = train_2[features_x]
train_2_y1 = train_2[outcome]

test_X1 = test[features_x]
test_y1 = test[outcome]


# lin mod
linMod1 = sm.OLS(train_1_y1, sm.add_constant(train_1_X1))
linModFit1 = linMod1.fit()

linModPreds1_train_1 = linModFit1.predict(sm.add_constant(train_1_X1))
linModPreds1_train_2 = linModFit1.predict(sm.add_constant(train_2_X1))
sum((linModPreds1_train_1 - train_1_y1)**2)/len(train_1_y1)
sum((linModPreds1_train_2 - train_2_y1)**2)/len(train_2_y1)


# xgb mod
params = {
    'max_depth': 5,
    'min_child_weight': 20,
    'eta': .2,
#    'subsample': 1,
#    'colsample_bytree': 1,
    'objective': 'reg:linear',  # 'binary:logistic',  #
    'eval_metric': 'rmse',  # 'auc'  #
}
num_boost_round = 100
early_stopping = 50

dtrain1 = xgb.DMatrix(train_1_X1, label=train_1_y1)
dtrain2 = xgb.DMatrix(train_2_X1, label=train_2_y1)
dtest= xgb.DMatrix(test_X1, label=test_y1)
xgbMod1 = xgb.train(params,
                   dtrain1,
                   num_boost_round=num_boost_round,
                   early_stopping_rounds=early_stopping,
                   evals=[(dtrain2, "Test")]
                   )

xgbModPreds_train_1 = xgbMod1.predict(dtrain1)
xgbModPreds_train_2 = xgbMod1.predict(dtrain2)
sum((xgbModPreds_train_1 - train_1_y1)**2)/len(train_1_y1)
sum((xgbModPreds_train_2 - train_2_y1)**2)/len(train_2_y1)


'''
predict error model
'''
train_1['prediction_1'] = xgbMod1.predict(dtrain1)
train_2['prediction_1'] = xgbMod1.predict(dtrain2)
test['prediction_1'] = xgbMod1.predict(dtest)

train_1['pc_error_1'] = abs(train_1['prediction_1']/train_1['speed'] - 1)
train_2['pc_error_1'] = abs(train_2['prediction_1']/train_2['speed'] - 1)
test['pc_error_1'] = abs(test['prediction_1']/test['speed'] - 1)

error = 'pc_error_1'

train_1_X2 = train_1[features_x]
train_1_y2 = train_1[error]

train_2_X2 = train_2[features_x]
train_2_y2 = train_2[error]

test_X2 = test[features_x]
test_y2 = test[error]


# lin mod
linMod2 = sm.OLS(train_2_y2, sm.add_constant(train_2_X2))
linModFit2 = linMod2.fit()

linModPreds2_train_2 = linModFit2.predict(sm.add_constant(train_2_X2))
linModPreds2_test_2 = linModFit2.predict(sm.add_constant(test_X2))
sum((linModPreds2_train_2 - train_2_y2)**2)/len(train_2_y2)
sum((linModPreds2_test_2 - test_y2)**2)/len(test_y2)


# xgb mod
params = {
    'max_depth': 6,
    'min_child_weight': 20,
    'eta': .2,
#    'subsample': 1,
#    'colsample_bytree': 1,
    'objective': 'reg:linear',  # 'binary:logistic',  #
    'eval_metric': 'rmse',  # 'auc'  #
}
num_boost_round = 100
early_stopping = 50

dtrain1_2 = xgb.DMatrix(train_1_X2, label=train_1_y2)
dtrain2_2 = xgb.DMatrix(train_2_X2, label=train_2_y2)
dtest_2 = xgb.DMatrix(test_X2, label=test_y2)
xgbMod2 = xgb.train(params,
                   dtrain2_2,
                   num_boost_round=num_boost_round,
                   early_stopping_rounds=early_stopping,
                   evals=[(dtest_2, "Test")]
                   )

xgbModPreds2_train_2 = xgbMod2.predict(dtrain2_2)
xgbModPreds2_test = xgbMod2.predict(dtest_2)
sum((xgbModPreds2_train_2 - train_2_y2)**2)/len(train_2_y2)
sum((xgbModPreds2_test - test_y2)**2)/len(test_y2)





'''
evaluation
'''

linModFit1.summary()
linModFit2.summary()

feature_importances_dict1 = xgbMod1.get_score(importance_type='gain')
feature_importances_df1 = pd.DataFrame({'feature':list(feature_importances_dict1.keys()),
                                       'importance':list(feature_importances_dict1.values())})
feature_importances_df1 = feature_importances_df1.sort_values(by='importance', ascending=False)
feature_importances_df1['importance'] = feature_importances_df1['importance']/sum(feature_importances_df1['importance'])

feature_importances_dict2 = xgbMod2.get_score(importance_type='gain')
feature_importances_df2 = pd.DataFrame({'feature':list(feature_importances_dict2.keys()),
                                       'importance':list(feature_importances_dict2.values())})
feature_importances_df2 = feature_importances_df2.sort_values(by='importance', ascending=False)
feature_importances_df2['importance'] = feature_importances_df2['importance']/sum(feature_importances_df2['importance'])

upper_limit = 1
test_y2_subset = test_y2[test_y2 < upper_limit]
xgbModPreds2_test_subset = xgbModPreds2_test[test_y2 < upper_limit]

points_to_plot = 1000
sns.scatterplot(x=test_y2_subset[:points_to_plot], y=xgbModPreds2_test_subset[:points_to_plot])


# plot best fit line
linMod_eval = sm.OLS(test_y2, sm.add_constant(xgbModPreds2_test))
linModFit_eval = linMod_eval.fit()

linModFit_eval.summary()

# SHAP
explainer = shap.TreeExplainer(xgbMod2)
shap_values = explainer.shap_values(train_2_X2)

shap.summary_plot(shap_values, train_2_X2)
shap_values.shape

# pickle model to be used in later models
import pickle
with open('tote/models/race_similarity_20200412.pickle', 'wb') as f:
    pickle.dump(xgbMod2, f)


