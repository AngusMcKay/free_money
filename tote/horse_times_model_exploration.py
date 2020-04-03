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

from keras.models import Sequential, model_from_json
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import optimizers
from keras import initializers
from keras import models
from keras import regularizers



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
get features and subset data
'''
horses_data.columns[-60:]

# select features
features_y = ['payout'] # ['winner'] # ['horse_time'] #
features_extra_info_for_results = ['race_id','race_date','betting_odds','finish_position','did_not_finish','jockey_id']
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
number_past_results = 6
features_prs = []
for i in range(1, number_past_results+1):
    features_prs = features_prs + ['pr_'+str(i)+'_'+pr for pr in features_pr]

features = features_y + features_extra_info_for_results + features_current_race + features_prs

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

num_non_pred_features = len(features_y + features_extra_info_for_results)
#train_races, test_races = train_test_split(train_data['race_id'].unique(), test_size=0.2, random_state=12)
test_races = train_data.loc[train_data['race_date'] > '2018-12-01', 'race_id'].unique()
train_races = train_data.loc[~train_data['race_id'].isin(test_races), 'race_id'].unique()
#len(test_races)/(len(test_races)+len(train_races))
train_idx = [r in train_races for r in train_data['race_id']]
test_idx = [r in test_races for r in train_data['race_id']]
train_X = train_data.loc[train_idx, features[num_non_pred_features:]]
train_y = train_data.loc[train_idx, features[0]]
train_extra_info = train_data.loc[train_idx, features[1:num_non_pred_features]]
test_X = train_data.loc[test_idx, features[num_non_pred_features:]]
test_y = train_data.loc[test_idx, features[0]]
test_extra_info = train_data.loc[test_idx, features[1:num_non_pred_features]]

# ## add response coded variables
# race_course_response_coding = pd.DataFrame({'course': train_X['course'],
#                                             'race_type_devised': train_X['race_type_devised'],
#                                             'finish_time': train_y})
# race_course_response_coding = race_course_response_coding.groupby(
#     ['course', 'race_type_devised'])['finish_time'].mean().reset_index()
# race_course_response_coding.columns = ['course', 'race_type_devised', 'race_course_rc']
# race_course_response_coding_default = np.mean(race_course_response_coding['race_course_rc'])
#
# train_X = train_X.merge(race_course_response_coding, on=['course', 'race_type_devised'], how='left')
# train_X.loc[train_X['race_course_rc'].isnull(), 'race_course_rc'] = race_course_response_coding_default
# train_X = train_X.drop(columns=['course', 'race_type_devised'])
#
# test_X = test_X.merge(race_course_response_coding, on=['course', 'race_type_devised'], how='left')
# test_X.loc[test_X['race_course_rc'].isnull(), 'race_course_rc'] = race_course_response_coding_default
# test_X = test_X.drop(columns=['course', 'race_type_devised'])
#
# for i in tqdm(range(number_past_results)):
#     race_course_response_coding.columns = ['course', 'race_type_devised', 'pr_'+str(i+1)+'_race_course_rc']
#     train_X = train_X.merge(race_course_response_coding, how='left',
#                             left_on=['pr_'+str(i+1)+'_course', 'pr_'+str(i+1)+'_race_type_devised'],
#                             right_on=['course', 'race_type_devised'])
#     train_X.loc[train_X['pr_'+str(i+1)+'_race_course_rc'].isnull(), 'pr_'+str(i+1)+'_race_course_rc'] = race_course_response_coding_default
#     train_X = train_X.drop(columns=['pr_'+str(i+1)+'_course', 'pr_'+str(i+1)+'_race_type_devised',
#                                     'course', 'race_type_devised'])
#
# for i in tqdm(range(number_past_results)):
#     race_course_response_coding.columns = ['course', 'race_type_devised', 'pr_'+str(i+1)+'_race_course_rc']
#     test_X = test_X.merge(race_course_response_coding, how='left',
#                             left_on=['pr_'+str(i+1)+'_course', 'pr_'+str(i+1)+'_race_type_devised'],
#                             right_on=['course', 'race_type_devised'])
#     test_X.loc[test_X['pr_'+str(i+1)+'_race_course_rc'].isnull(), 'pr_'+str(i+1)+'_race_course_rc'] = race_course_response_coding_default
#     test_X = test_X.drop(columns=['pr_'+str(i+1)+'_course', 'pr_'+str(i+1)+'_race_type_devised',
#                                     'course', 'race_type_devised'])
#
# race_course_response_coding.columns = ['course', 'race_type_devised', 'race_course_rc']




#data_to_inspect = train_X.iloc[:1000,]

'''
train models
'''
len(train_y)
# lin mod
linMod = sm.OLS(train_y, sm.add_constant(train_X))
linModFit = linMod.fit()

linModPreds_train = linModFit.predict(sm.add_constant(train_X))
linModPreds_test = linModFit.predict(sm.add_constant(test_X))
sum((linModPreds_train - train_y)**2)/len(train_y)
sum((linModPreds_test - test_y)**2)/len(test_y)
linModFit.summary()


# xgb mod
params = {
    'max_depth':3,
    'min_child_weight': 5,
    'eta':.3,
#    'subsample': 1,
#    'colsample_bytree': 1,
    'objective': 'reg:linear', # 'binary:logistic',
    'eval_metric': 'rmse', #'auc'
}
num_boost_round = 100
early_stopping = 10
#xgbMod = xgb.XGBRegressor(max_depth=3, learning_rate=0.1, n_estimators=100)
##params_contrained = xgbMod.get_params()
##cur_race_cols = list(range(len(features_current_race)))
##past_race_group_size = int(len(features_prs)/number_past_results)
##past_race_group_cols = [list(range(i*past_race_group_size,i*past_race_group_size+past_race_group_size)) for i in range(number_past_results)]
##params_contrained['interaction_constraints'] = [cur_race_cols] + past_race_group_cols
##if False:
##    xgbMod.set_params(params_contrained)
#xgbMod.fit(train_X, train_y)
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
sum((xgbModPreds_train - train_y)**2)/len(train_y)
sum((xgbModPreds_test - test_y)**2)/len(test_y)
sum(abs(xgbModPreds_test - test_y))/len(test_y)

feature_importances_dict = xgbMod.get_score(importance_type='gain')
feature_importances_df = pd.DataFrame({'feature':list(feature_importances_dict.keys()),
                                       'importance':list(feature_importances_dict.values())})
feature_importances_df = feature_importances_df.sort_values(by='importance', ascending=False)
feature_importances_df['importance'] = feature_importances_df['importance']/sum(feature_importances_df['importance'])


'''
jockey analysis and feature engineering
'''
jockey_pred_time_vs_actual = pd.DataFrame({'actual_times': train_y,
                                           'pred_times': xgbModPreds_train,
                                           'jockey_id': train_data.loc[train_idx, 'jockey_id']})
jockey_pred_time_vs_actual['time_diff'] = jockey_pred_time_vs_actual['actual_times'] - jockey_pred_time_vs_actual['pred_times']
jockey_analysis = jockey_pred_time_vs_actual.groupby('jockey_id').aggregate(
    {'time_diff': [len, min, max, np.mean, np.std, np.median, lambda x: np.quantile(x, 0.75),
                   lambda x: sum(np.array(x) < 0) / len(x)]}).reset_index()
jockey_analysis.columns = [
    'jockey_id', 'jockey_races', 'jockey_best_diff', 'jockey_worst_diff', 'jockey_mean_diff', 'jockey_std_diff',
    'jockey_median_diff', 'jockey_q_75_diff', 'jockey_prop_faster']

# create default values for jockeys that only have a few races
jockey_min_races = 6
inexperienced_median_avg = np.mean(jockey_analysis.loc[jockey_analysis['jockey_races'] < 100, 'jockey_median_diff'])
inexperienced_q_75_avg = np.mean(jockey_analysis.loc[jockey_analysis['jockey_races'] < 100, 'jockey_q_75_diff'])
jockey_analysis['jockey_median_diff_adjusted'] = jockey_analysis['jockey_median_diff']
jockey_analysis['jockey_q_75_diff_adjusted'] = jockey_analysis['jockey_q_75_diff']
jockey_analysis.loc[jockey_analysis['jockey_races'] < jockey_min_races, 'jockey_median_diff_adjusted'] = inexperienced_median_avg
jockey_analysis.loc[jockey_analysis['jockey_races'] < jockey_min_races, 'jockey_q_75_diff_adjusted'] = inexperienced_q_75_avg

# add jockey stats back on to data
train_data = train_data.merge(jockey_analysis, how='left', on='jockey_id')
train_data[['jockey_races', 'jockey_std_diff', 'jockey_prop_faster']] = train_data[
    ['jockey_races', 'jockey_std_diff', 'jockey_prop_faster']].fillna(value=0)
train_data[['jockey_best_diff', 'jockey_worst_diff', 'jockey_mean_diff', 'jockey_median_diff', 'jockey_median_diff_adjusted']] = train_data[
    ['jockey_best_diff', 'jockey_worst_diff', 'jockey_mean_diff', 'jockey_median_diff', 'jockey_median_diff_adjusted']].fillna(value=inexperienced_median_avg)
train_data[['jockey_q_75_diff', 'jockey_q_75_diff_adjusted']] = train_data[
    ['jockey_q_75_diff', 'jockey_q_75_diff_adjusted']].fillna(value=inexperienced_q_75_avg)

data_to_inspect = train_data.isnull().sum(axis=0)


'''
retrain with jockey features
'''

# add jockey features to input data
features_jockey_stats = ['jockey_races', 'jockey_median_diff_adjusted', 'jockey_q_75_diff_adjusted']
features = features_y + features_extra_info_for_results + features_current_race + features_prs + features_jockey_stats

train_X = train_data.loc[train_idx, features[num_non_pred_features:]]
train_y = train_data.loc[train_idx, features[0]]
train_extra_info = train_data.loc[train_idx, features[1:num_non_pred_features]]
test_X = train_data.loc[test_idx, features[num_non_pred_features:]]
test_y = train_data.loc[test_idx, features[0]]
test_extra_info = train_data.loc[test_idx, features[1:num_non_pred_features]]

# lin mod
linMod = sm.OLS(train_y, sm.add_constant(train_X))
linModFit = linMod.fit()

linModPreds_train = linModFit.predict(sm.add_constant(train_X))
linModPreds_test = linModFit.predict(sm.add_constant(test_X))
sum((linModPreds_train - train_y)**2)/len(train_y)
sum((linModPreds_test - test_y)**2)/len(test_y)
linModFit.summary()


# xgb mod
params = {
    'max_depth':3,
    'min_child_weight': 5,
    'eta':.3,
#    'subsample': 1,
#    'colsample_bytree': 1,
    'objective': 'binary:logistic', # 'reg:linear',
    'eval_metric':'auc' #'rmse'
}
num_boost_round = 100
early_stopping = 10
#xgbMod = xgb.XGBRegressor(max_depth=3, learning_rate=0.1, n_estimators=100)
##params_contrained = xgbMod.get_params()
##cur_race_cols = list(range(len(features_current_race)))
##past_race_group_size = int(len(features_prs)/number_past_results)
##past_race_group_cols = [list(range(i*past_race_group_size,i*past_race_group_size+past_race_group_size)) for i in range(number_past_results)]
##params_contrained['interaction_constraints'] = [cur_race_cols] + past_race_group_cols
##if False:
##    xgbMod.set_params(params_contrained)
#xgbMod.fit(train_X, train_y)
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
sum((xgbModPreds_train - train_y)**2)/len(train_y)
sum((xgbModPreds_test - test_y)**2)/len(test_y)

feature_importances_dict = xgbMod.get_score(importance_type='gain')
feature_importances_df = pd.DataFrame({'feature':list(feature_importances_dict.keys()),
                                       'importance':list(feature_importances_dict.values())})
feature_importances_df = feature_importances_df.sort_values(by='importance', ascending=False)
feature_importances_df['importance'] = feature_importances_df['importance']/sum(feature_importances_df['importance'])





# SHAP
import shap
explainer = shap.TreeExplainer(xgbMod)
shap_values = explainer.shap_values(train_X)
shap.force_plot(explainer.expected_value, shap_values[0,:], train_X.iloc[0,:], matplotlib=True)
#shap.force_plot(explainer.expected_value, shap_values, train_X)
shap.summary_plot(shap_values[:100], train_X[:100])




# do some param testing
gridsearch_params = [
    (max_depth, min_child_weight, colsample_bytree, subsample)
    for max_depth in range(3,4)
    for min_child_weight in [1, 4, 6]
    for colsample_bytree in [1]
    for subsample in [1]
]
min_rmse = float("Inf")
best_params = None
for max_depth, min_child_weight, colsample_bytree, subsample in gridsearch_params:
    print("CV with max_depth={}, min_child_weight={}, colsample_bytree={}, subsample={}".format(
                             max_depth,
                             min_child_weight,
                             colsample_bytree,
                             subsample
                             ))
    # Update our parameters
    params['max_depth'] = max_depth
    params['min_child_weight'] = min_child_weight
    params['colsample_bytree'] = colsample_bytree
    params['subsample'] = subsample
    # Run CV
    cv_results = xgb.cv(params, dtrain, num_boost_round=num_boost_round, early_stopping_rounds=early_stopping,
                        seed=123, nfold=5, metrics={'rmse'}
                        )
    # Update best MAE
    mean_rmse = cv_results['test-rmse-mean'].min()
    boost_rounds = cv_results['test-rmse-mean'].argmin()
    print("\tRMSE {} for {} rounds".format(mean_rmse, boost_rounds))
    if mean_rmse < min_rmse:
        min_rmse = mean_rmse
        best_params = (max_depth,min_child_weight,colsample_bytree,subsample)
print("Best params: {}, {}, RMSE: {}".format(best_params[0], best_params[1], min_rmse))

# seems like max depth and min child weight doesn't make too much difference

# test sklearn to see if can control it better
from sklearn.ensemble import RandomForestRegressor

rfMod = RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_leaf=50, random_state=123)
rfMod.fit(train_X, train_y)

rfModPreds_train = rfMod.predict(train_X)
rfModPreds_test = rfMod.predict(test_X)
sum((rfModPreds_train - train_y)**2)/len(train_y)
sum((rfModPreds_test - test_y)**2)/len(test_y)
sum(abs(rfModPreds_test - test_y))/len(test_y)

feature_importances_df = pd.DataFrame({'feature':list(train_X.columns),
                                       'importance':list(rfMod.feature_importances_)})
feature_importances_df = feature_importances_df.sort_values(by='importance', ascending=False)
feature_importances_df['importance'] = feature_importances_df['importance']/sum(feature_importances_df['importance'])




# neural network
input_dimension = train_X.shape[1]

# optimizers
optim_rmsprop = optimizers.RMSprop(lr=0.0001, rho=0.9)
optim_sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.5, nesterov=True)
optim_adagrad = optimizers.Adagrad(lr=0.1, decay=0)
optim_adam = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0)

number_epochs=50
batch_sizes=2**7
val_split=0.1
dropout = 0.0
weights = np.zeros(train_X.shape[0])+1

nnMod = Sequential()
nnMod.add(Dense(input_dimension, input_dim=input_dimension, kernel_initializer='normal', activation='relu',
           #kernel_regularizer=regularizers.l1(0.001),
           #activity_regularizer=regularizers.l1(0.01)
           ))
nnMod.add(Dense(2*input_dimension, kernel_initializer='normal', activation='relu'))
#nnMod.add(Dense(input_dimension, kernel_initializer='normal', activation='relu'))
#nnMod.add(Dense(input_dimension, kernel_initializer='normal', activation='relu'))
nnMod.add(Dense(1, kernel_initializer='normal'))
nnMod.compile(loss='mean_squared_error', optimizer=optim_adam)
nnMod.fit(train_X,train_y,epochs=number_epochs,batch_size=batch_sizes,validation_split=val_split, sample_weight=weights)

nnModPreds_train = np.array(nnMod.predict(train_X))[:,0]
nnModPreds_test = np.array(nnMod.predict(test_X))[:,0]
sum((nnModPreds_train - train_y)**2)/len(train_y)
sum((nnModPreds_test - test_y)**2)/len(test_y)


'''
analyse output
'''
output = pd.DataFrame({'actual':test_y,
                       #'linModPreds':linModPreds_test,
                       'xgbModPreds':xgbModPreds_test,
                       #'nnModPreds':nnModPreds_test,
                       'race_id':test_extra_info['race_id'],
                       'race_date':test_extra_info['race_date'],
                       'betting_odds':test_extra_info['betting_odds'],
                       'finish_position':test_extra_info['finish_position'],
                       'runners':test_X['runners']
                       })
len(test_y)
# add checks that top runner or all runners are in the prediction sets for each race
top_3_in_preds = output.groupby('race_id')['finish_position'].apply(lambda x: (1 in list(x)) and (2 in list(x)) and (3 in list(x))).reset_index()
top_3_in_preds.columns = ['race_id','top_3_check']
output = output.merge(top_3_in_preds, how='left', on='race_id')

number_preds_per_race = output.groupby('race_id')['runners'].count().reset_index()
number_preds_per_race.columns = ['race_id','number_preds']
output = output.merge(number_preds_per_race, how='left', on='race_id')
output['number_preds_check'] = (output['runners']<=output['number_preds'])

preds_to_use = 'xgbModPreds'
pred_times = output.groupby('race_id')[preds_to_use].apply(lambda x: list(x)).reset_index()
for timelist in pred_times[preds_to_use]:
    timelist.sort(reverse=True)
def get_time(lis, pos):
    try:
        return lis[pos]
    except:
        return None
pred_times['pred_time_1'] = [get_time(l,0) for l in pred_times[preds_to_use]]
pred_times['pred_time_2'] = [get_time(l,1) for l in pred_times[preds_to_use]]
pred_times['pred_time_3'] = [get_time(l,2) for l in pred_times[preds_to_use]]
pred_times = pred_times[['race_id','pred_time_1','pred_time_2','pred_time_3']]
output = output.merge(pred_times, how='left', on='race_id')
output['pred_place_1'] = output[preds_to_use]==output['pred_time_1']
output['pred_place_2'] = output[preds_to_use]==output['pred_time_2']
output['pred_place_3'] = output[preds_to_use]==output['pred_time_3']

output = output.sort_values(['race_id',preds_to_use])

output_with_top_3_horses = output[output['number_preds_check']==True]
output_with_top_3_horses = output_with_top_3_horses[output_with_top_3_horses['top_3_check']==True]

# % right
sum((output_with_top_3_horses['pred_place_1']) & (output_with_top_3_horses['finish_position']==1))/sum(output_with_top_3_horses['pred_place_1'])
# this has gone up from 19% with initial models, to 23.1% with model with most features and 5 past results
# now at 27% with relative pr features and jockey stats. 27.4% with expected time for each past result
output_with_top_3_horses['decimal_odds'] = [odds_parser(o) for o in output_with_top_3_horses['betting_odds']]
sum((output_with_top_3_horses[preds_to_use]>1)*output_with_top_3_horses['decimal_odds']*(output_with_top_3_horses['finish_position']==1))
sum(output_with_top_3_horses[preds_to_use]>1)

first_place_bets = output_with_top_3_horses[['race_id','betting_odds','pred_place_1','finish_position']]
first_place_bets = first_place_bets[first_place_bets['pred_place_1']]
first_place_bets['win'] = (first_place_bets['finish_position']==1)

first_place_bets['decimal_odds'] = [odds_parser(o) for o in first_place_bets['betting_odds']]
first_place_bets['payouts'] = first_place_bets['decimal_odds']*first_place_bets['win']
sum(first_place_bets['payouts'][first_place_bets['payouts'].notnull()])/sum(first_place_bets['payouts'].notnull())
odds_cutoff = 3
sum(first_place_bets['payouts'][first_place_bets['decimal_odds']<odds_cutoff])/sum(first_place_bets['decimal_odds']<odds_cutoff)

# what is random return
output_with_top_3_horses['decimal_odds'] = [odds_parser(o) for o in output_with_top_3_horses['betting_odds']]
sum(output_with_top_3_horses.loc[output_with_top_3_horses['finish_position'] == 1, 'decimal_odds']) / len(output_with_top_3_horses)

# how many times does bookies favourite win
lowest_odds_per_race = output_with_top_3_horses.groupby('race_id')['decimal_odds'].min().reset_index()
lowest_odds_per_race.columns = ['race_id', 'lowest_odds']
output_with_top_3_horses = output_with_top_3_horses.merge(lowest_odds_per_race, how='left', on='race_id')
output_with_top_3_horses['bookies_fave'] = (output_with_top_3_horses['lowest_odds'] == output_with_top_3_horses['decimal_odds'])*1
sum((output_with_top_3_horses['bookies_fave']) & (output_with_top_3_horses['finish_position']==1))/sum(output_with_top_3_horses['bookies_fave'])


'''
do preds per race
'''
len(output_with_top_3_horses)
all_preds_per_race = output_with_top_3_horses.groupby('race_id')[preds_to_use].apply(lambda x: list(x)[::-1])
all_preds_per_race = pd.DataFrame([l for l in all_preds_per_race], index=all_preds_per_race.index)
all_preds_per_race = all_preds_per_race.fillna(0)
all_preds_per_race.columns = ['horse_pred_' + str(c) for c in all_preds_per_race.columns]

preds_per_race_data = output_with_top_3_horses.copy()
preds_per_race_data = preds_per_race_data.merge(all_preds_per_race, how='left',
                                                left_on='race_id', right_index=True)

preds_per_race_data['payout'] = preds_per_race_data['actual'] * preds_per_race_data['decimal_odds']

race_pred_y = ['actual']
race_pred_X = [preds_to_use, 'runners', 'decimal_odds'] + list(all_preds_per_race.columns)[:10]
race_pred_train_races = preds_per_race_data.loc[preds_per_race_data['race_date'] < '2017-09-01', 'race_id']
race_pred_test_races = preds_per_race_data.loc[~preds_per_race_data['race_id'].isin(race_pred_train_races), 'race_id']

race_pred_train_data_y = preds_per_race_data.loc[preds_per_race_data['race_id'].isin(race_pred_train_races), race_pred_y]
race_pred_train_data_X = preds_per_race_data.loc[preds_per_race_data['race_id'].isin(race_pred_train_races), race_pred_X]
race_pred_test_data_y = preds_per_race_data.loc[preds_per_race_data['race_id'].isin(race_pred_test_races), race_pred_y]
race_pred_test_data_X = preds_per_race_data.loc[preds_per_race_data['race_id'].isin(race_pred_test_races), race_pred_X]



# xgb mod
params = {
    'max_depth':1,
    'min_child_weight': 5,
    'eta':.3,
#    'subsample': 1,
#    'colsample_bytree': 1,
    'objective': 'reg:linear', # 'binary:logistic', #
    'eval_metric': 'rmse' # 'auc' #
}
num_boost_round = 100
early_stopping = 10

dtrain = xgb.DMatrix(race_pred_train_data_X, label=race_pred_train_data_y)
dtest = xgb.DMatrix(race_pred_test_data_X, label=race_pred_test_data_y)
xgbMod = xgb.train(params,
                   dtrain,
                   num_boost_round=num_boost_round,
                   early_stopping_rounds=early_stopping,
                   evals=[(dtest, "Test")]
                   )

xgbModPreds_train = xgbMod.predict(dtrain)
xgbModPreds_test = xgbMod.predict(dtest)

all_preds = list(xgbModPreds_train) + list(xgbModPreds_test)
output_with_top_3_horses['reworked_preds'] = all_preds

sum(output_with_top_3_horses['decimal_odds'] * output_with_top_3_horses['actual'] * (
        (output_with_top_3_horses['decimal_odds'] > 1/output_with_top_3_horses['reworked_preds']) &
        #(output_with_top_3_horses['reworked_preds'] > 1) &
        #(output_with_top_3_horses['reworked_preds'] > 0.5) &
        (output_with_top_3_horses['pred_place_1']==1) &
        (output_with_top_3_horses['race_id'].isin(race_pred_test_races)))
    )

sum((output_with_top_3_horses['decimal_odds'] > 1/output_with_top_3_horses['reworked_preds']) &
    #(output_with_top_3_horses['reworked_preds'] > 1) &
    #(output_with_top_3_horses['reworked_preds'] > 0.5) &
    (output_with_top_3_horses['pred_place_1']==1) &
    (output_with_top_3_horses['race_id'].isin(race_pred_test_races)))


'''
get prediction variances per prediction point
'''
#train_y_var_lin = (linModPreds_train - train_y)**2
train_y_var_xgb = (xgbModPreds_train - train_y)**2
test_y_var_xgb = (xgbModPreds_test - test_y)**2
#train_y_var_nn = (nnModPreds_train - train_y)**2

# train model for lin mod variances
#linMod_var = LinearRegression()
#linMod_var.fit(train_X, train_y_var_lin)
#linModVars_train = linMod_var.predict(train_X)
#linModVars_test = linMod_var.predict(test_X)

# train model for xgb mod variances
# xgb mod
params_var = {
    'max_depth':5,
    'min_child_weight': 1,
    'eta':.01,
#    'subsample': 1,
#    'colsample_bytree': 1,
    'objective':'reg:linear',
    'eval_metric':'mae'
}

dtrain_var = xgb.DMatrix(train_X, label=train_y_var_xgb)
dtest_var = xgb.DMatrix(test_X, label=test_y_var_xgb)
xgbMod_var = xgb.train(params_var,
                   dtrain_var,
                   num_boost_round=num_boost_round,
                   early_stopping_rounds=early_stopping,
                   evals=[(dtest_var, "Test")]
                   )
xgbModVars_train = xgbMod_var.predict(dtrain)
xgbModVars_test = xgbMod_var.predict(dtest)
np.mean(abs(xgbModPreds_train - train_y))
np.mean(np.sqrt(xgbModVars_train))
np.sqrt(np.mean(xgbModVars_train))
np.sqrt(np.mean((xgbModPreds_train - train_y)**2))
np.mean(abs(xgbModPreds_test - test_y))
np.mean(np.sqrt(xgbModVars_test))
np.sqrt(np.mean(xgbModVars_test))
np.sqrt(np.mean((xgbModPreds_test - test_y)**2))


#xgbMod_var = xgb.XGBRegressor(max_depth=3, learning_rate=0.1, n_estimators=100)
#xgbMod_var.fit(train_X, train_y_var_xgb)
#xgbModVars_train = xgbMod_var.predict(train_X)
#xgbModVars_test = xgbMod_var.predict(test_X)

# train model for nn mod variances
input_dimension = train_X.shape[1]

# optimizers
optim_rmsprop = optimizers.RMSprop(lr=0.0001, rho=0.9)
optim_sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.5, nesterov=True)
optim_adagrad = optimizers.Adagrad(lr=0.1, decay=0)
optim_adam = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0)

number_epochs=50
batch_sizes=2**7
val_split=0.1
dropout = 0.0
weights = np.zeros(train_X.shape[0])+1

nnMod_var = Sequential()
nnMod_var.add(Dense(input_dimension, input_dim=input_dimension, kernel_initializer='normal', activation='relu',
           #kernel_regularizer=regularizers.l1(0.001),
           #activity_regularizer=regularizers.l1(0.01)
           ))
nnMod_var.add(Dense(2*input_dimension, kernel_initializer='normal', activation='relu'))
#nnMod_var.add(Dense(input_dimension, kernel_initializer='normal', activation='relu'))
#nnMod_var.add(Dense(input_dimension, kernel_initializer='normal', activation='relu'))
nnMod_var.add(Dense(1, kernel_initializer='normal'))
nnMod_var.compile(loss='mean_squared_error', optimizer=optim_adam)
nnMod_var.fit(train_X,train_y_var_nn,epochs=number_epochs,batch_size=batch_sizes,validation_split=val_split, sample_weight=weights)

nnModVars_train = nnMod_var.predict(train_X)
nnModVars_test = nnMod_var.predict(test_X)


np.mean(np.sqrt(linModVars_train))
np.mean(np.sqrt(xgbModVars_train))
np.mean(np.sqrt(nnModVars_train))
np.std(np.sqrt(xgbModVars_train))
np.std(np.sqrt(nnModVars_train))
min(np.sqrt(xgbModVars_train))
max(np.sqrt(xgbModVars_train))
min(np.sqrt(nnModVars_train))
max(np.sqrt(nnModVars_train))



output['xgbModVars'] = xgbModVars_test
#output['nnModVars'] = nnModVars_test

#nn_within_range = sum((output['actual'] > (output['nnModPreds'] - np.sqrt(output['nnModVars']))) & (output['actual'] < (output['nnModPreds'] + np.sqrt(output['nnModVars']))))
#nn_within_range/output.shape[0] # nb: 72% > 68% expected, either due to randomness, or because not completely normal distribution

output_with_all_runners = output[output['number_preds_check']]
output_with_all_runners = output_with_all_runners[output_with_all_runners['top_3_check']]

race_ids_with_all_runners = output_with_all_runners['race_id'].unique()
output_with_all_runners['pred_odds'] = None
output_with_all_runners['xgb_pred_odds'] = None
#output_with_all_runners['nn_pred_odds'] = None
def get_probs_from_time_list(timelist, std_list, simulations=1000):
    try:
        draw = np.random.normal(timelist, std_list)
        win_counter = (draw==min(draw))*1
        for i in range(simulations-1):
            draw = np.random.normal(timelist, std_list)
            win_counter = win_counter + (draw==min(draw))*1
        return 1/(win_counter/simulations)
    except:
        return 1

for i in tqdm(race_ids_with_all_runners):
    subset = output_with_all_runners[output_with_all_runners['race_id']==i]
    timelist = list(subset['xgbModPreds'])
    stdlist = list(subset['xgbModVars'].map(lambda x: np.sqrt(x)))
    output_with_all_runners['xgb_pred_odds'][output_with_all_runners['race_id']==i] = get_probs_from_time_list(timelist, stdlist, 1000)
#    timelist = list(subset['nnModPreds'])
#    stdlist = list(subset['nnModVars'].map(lambda x: np.sqrt(x)))
#    output_with_all_runners['nn_pred_odds'][output_with_all_runners['race_id']==i] = get_probs_from_time_list(timelist, stdlist, 1000)


output_with_all_runners['decimal_odds'] = [odds_parser(x) for x in output_with_all_runners['betting_odds']]
np.mean((1/output_with_all_runners['decimal_odds'] - 1/output_with_all_runners['xgb_pred_odds']))
np.std((1/output_with_all_runners['decimal_odds'] - 1/output_with_all_runners['xgb_pred_odds']))
odds_ratios = ((1/output_with_all_runners['decimal_odds']) / (1/output_with_all_runners['xgb_pred_odds'].replace([np.inf, -np.inf], np.nan)))
np.std(odds_ratios)



'''
use winning prob and horse features to predict winnings
'''
# create new training data with prob win and variance using above models
train_data_winnings = train_data.copy()
dtrain_winnings = xgb.DMatrix(train_data_winnings[features[num_non_pred_features:]])
train_data_winnings['pred_time'] = xgbMod.predict(dtrain_winnings)
train_data_winnings['pred_var'] = xgbMod_var.predict(dtrain_winnings)

train_data_winnings['pred_odds'] = 0.0
race_ids = train_data_winnings['race_id'].unique()
for i in tqdm(race_ids):
    subset = train_data_winnings[train_data_winnings['race_id']==i]
    timelist = list(subset['pred_time'])
    stdlist = list(subset['pred_var'].map(lambda x: np.sqrt(x)))
    train_data_winnings['pred_odds'][train_data_winnings['race_id']==i] = get_probs_from_time_list(timelist, stdlist, 1000)


# add checks that top runner or all runners are in the prediction sets for each race
top_3_in_preds = train_data_winnings.groupby('race_id')['finish_position'].apply(lambda x: (1 in list(x)) and (2 in list(x)) and (3 in list(x))).reset_index()
top_3_in_preds.columns = ['race_id','top_3_check']
train_data_winnings = train_data_winnings.merge(top_3_in_preds, how='left', on='race_id')

number_preds_per_race = train_data_winnings.groupby('race_id')['runners'].count().reset_index()
number_preds_per_race.columns = ['race_id','number_preds']
train_data_winnings = train_data_winnings.merge(number_preds_per_race, how='left', on='race_id')
train_data_winnings['number_preds_check'] = train_data_winnings['runners']==train_data_winnings['number_preds']

train_data_winnings['decimal_odds'] = [odds_parser(x) for x in train_data_winnings['betting_odds']]
train_data_winnings['win_bet_winnings'] = (train_data_winnings['decimal_odds'])*(train_data_winnings['finish_position']==1)

data_to_inspect = train_data_winnings[:100]

# split data
features_y_win = ['win_bet_winnings']
features_additional = ['pred_odds','pred_time','pred_var']
features_win = features_y_win + features_extra_info_for_results + features_current_race + features_prs + features_jockey_stats + features_additional

train_data_winnings_to_split = train_data_winnings.copy()
train_data_winnings_to_split = train_data_winnings_to_split[train_data_winnings_to_split['number_preds_check']]
train_data_winnings_to_split = train_data_winnings_to_split[train_data_winnings_to_split['top_3_check']]
train_data_winnings_to_split = train_data_winnings_to_split[features_win]

train_data_winnings_to_split.isnull().sum(axis=0)
train_data_winnings_to_split = train_data_winnings_to_split[train_data_winnings_to_split['betting_odds'].notnull()]

train_idx_winnings = [r in train_races for r in train_data_winnings_to_split['race_id']]
test_idx_winnings = [r in test_races for r in train_data_winnings_to_split['race_id']]

train_X_winnings = train_data_winnings_to_split.loc[train_idx_winnings, features_win[num_non_pred_features:]]
train_y_winnings = train_data_winnings_to_split.loc[train_idx_winnings, features_win[0]]
train_extra_info_winnings = train_data_winnings_to_split.loc[train_idx_winnings, features_win[1:num_non_pred_features]]
test_X_winnings = train_data_winnings_to_split.loc[test_idx_winnings, features_win[num_non_pred_features:]]
test_y_winnings = train_data_winnings_to_split.loc[test_idx_winnings, features_win[0]]
test_extra_info_winnings = train_data_winnings_to_split.loc[test_idx_winnings, features_win[1:num_non_pred_features]]

data_to_inspect = train_X_winnings[:100]

# train model
#xgbMod_win = xgb.XGBRegressor(max_depth=2, learning_rate=0.1, n_estimators=10)
#xgbMod_win.fit(train_X_winnings[['pred_odds','pred_var']], train_y_winnings)
#xgbModWins_train = xgbMod_win.predict(train_X_winnings[['pred_odds','pred_var']])
#xgbModWins_test = xgbMod_win.predict(test_X_winnings[['pred_odds','pred_var']])

params_win = {
    'max_depth':4,
    'min_child_weight': 50,
    'eta':.1,
#    'subsample': 1,
#    'colsample_bytree': 1,
    'objective':'reg:linear',
    'eval_metric':'rmse'
}

dtrain_win = xgb.DMatrix(train_X_winnings, label=train_y_winnings)
dtest_win = xgb.DMatrix(test_X_winnings, label=test_y_winnings)
xgbMod_win = xgb.train(params_win,
                   dtrain_win,
                   num_boost_round=100,
                   early_stopping_rounds=10,
                   evals=[(dtest_win, "Test")]
                   )
xgbModWins_train = xgbMod_win.predict(dtrain_win)
xgbModWins_test = xgbMod_win.predict(dtest_win)


pred_win_cutoff = 1.2
sum(test_y_winnings[xgbModWins_test>pred_win_cutoff])/sum(xgbModWins_test>pred_win_cutoff)

sum(xgbModWins_train>pred_win_cutoff)
sum(xgbModWins_test>pred_win_cutoff)

sum(train_y_winnings[xgbModWins_train>pred_win_cutoff])
sum(test_y_winnings[xgbModWins_test>pred_win_cutoff])

len(train_data_winnings_to_split['race_id'].unique())
