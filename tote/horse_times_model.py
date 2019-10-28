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
#from keras.models import Sequential, model_from_json
#from keras.layers import Activation, Dropout, Flatten, Dense
#from keras import optimizers
#from keras import initializers
#from keras import models
#from keras import regularizers



'''
get data
'''
connect_string = 'mysql+pymysql://root:angunix1@localhost/horses'
sql_engine = sqlalchemy.create_engine(connect_string)
horses_data = pd.read_sql('''
                          SELECT * FROM training_data_10_pr_no_nrs
                            WHERE 1
                            AND pr_1_horse_time IS NOT NULL
                            AND pr_2_horse_time IS NOT NULL
                            AND pr_3_horse_time IS NOT NULL
                            AND pr_4_horse_time IS NOT NULL
                            AND pr_5_horse_time IS NOT NULL
                            AND pr_6_horse_time IS NOT NULL
                            AND pr_7_horse_time IS NOT NULL
                            AND pr_8_horse_time IS NOT NULL
                            AND pr_9_horse_time IS NOT NULL
                            AND pr_10_horse_time IS NOT NULL
                            AND did_not_finish = 0
                            AND pr_1_did_not_finish =0
                            AND pr_2_did_not_finish =0
                            AND pr_3_did_not_finish =0
                            AND pr_4_did_not_finish =0
                            AND pr_5_did_not_finish =0
                            AND pr_6_did_not_finish =0
                            AND pr_7_did_not_finish =0
                            AND pr_8_did_not_finish =0
                            AND pr_9_did_not_finish =0
                            AND pr_10_did_not_finish =0
                          ''',
                          con=sql_engine)

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

#parsed_odds = [odds_parser(x) for x in horses_data['betting_odds']]
#winning_ods = [p for p,w in zip(parsed_odds, horses_data['finish_position']) if w==1]
#sum(winning_ods)/len(parsed_odds)

data_to_inspect = horses_data[:1000]


'''
get features and subset data
'''
horses_data.columns[-60:]

# select features
features_y = ['horse_time']
features_extra_info_for_results = ['race_id','race_date','betting_odds','finish_position','did_not_finish']
features_current_race = ['yards', 'runners', 'handicap_pounds', 'horse_age',
                         'horse_sex_g','horse_sex_m','horse_sex_f','horse_sex_c','horse_sex_h','horse_sex_r',
                         'horse_last_ran_days', 'going_grouped_horse_time_rc',
                         'race_type_horse_time_rc', 'weather_horse_time_rc']# + going_grouped_types
features_pr = ['horse_time', 'finish_position_for_ordering', 'yards',
               'runners', 'handicap_pounds', 'going_grouped_horse_time_rc',
               'race_type_horse_time_rc', 'weather_horse_time_rc']# + going_grouped_types
number_past_results = 10
features_prs = []
for i in range(1, number_past_results+1):
    features_prs = features_prs + ['pr_'+str(i)+'_'+pr for pr in features_pr]

features = features_y + features_extra_info_for_results + features_current_race + features_prs

train_data = horses_data[features]

data_to_inspect = train_data.isnull().sum(axis=0)
train_data = train_data[train_data['yards'].notnull()]
train_data = train_data[train_data['horse_last_ran_days'].notnull()]
train_data = train_data[train_data['pr_1_yards'].notnull()]
train_data = train_data[train_data['pr_2_yards'].notnull()]
train_data = train_data[train_data['pr_3_yards'].notnull()]
train_data = train_data[train_data['pr_4_yards'].notnull()]
train_data = train_data[train_data['pr_5_yards'].notnull()]
train_data = train_data[train_data['pr_6_yards'].notnull()]
train_data = train_data[train_data['pr_7_yards'].notnull()]
train_data = train_data[train_data['pr_8_yards'].notnull()]
train_data = train_data[train_data['pr_9_yards'].notnull()]
train_data = train_data[train_data['pr_10_yards'].notnull()]
train_data.isnull().sum(axis=0)


train_races, test_races = train_test_split(train_data['race_id'].unique(), test_size=0.2)
train_idx = [r in train_races for r in train_data['race_id']]
test_idx = [r in test_races for r in train_data['race_id']]
train_X = train_data.loc[train_idx, features[6:]]
train_y = train_data.loc[train_idx, features[0]]
train_extra_info = train_data.loc[train_idx, features[1:6]]
test_X = train_data.loc[test_idx, features[6:]]
test_y = train_data.loc[test_idx, features[0]]
test_extra_info = train_data.loc[test_idx, features[1:6]]


'''
train models
'''
# lin mod
linMod = LinearRegression()
linMod.fit(train_X, train_y)
linMod.score(train_X, train_y) # this is r**2 value
linMod.score(test_X, test_y)

linModPreds_train = linMod.predict(train_X)
linModPreds_test = linMod.predict(test_X)
sum((linModPreds_train - train_y)**2)/len(train_y)
sum((linModPreds_test - test_y)**2)/len(test_y)


# xgb mod
params = {
    'max_depth':3,
    'min_child_weight': 5,
    'eta':.3,
#    'subsample': 1,
#    'colsample_bytree': 1,
    'objective':'reg:linear',
    'eval_metric':'rmse'
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
sum((xgbModPreds_train - train_y)**2)/len(train_y) # 8819 with max depth=5, 7433 with almost all features and 5 past results
sum((xgbModPreds_test - test_y)**2)/len(test_y) # 9370 with max depth=5, 9166 with almost all features and 5 past results, with 10 past results and not all features gives 8182, 10 past results and all features is 7423


# do some param testing
gridsearch_params = [
    (max_depth, min_child_weight, colsample_bytree, subsample)
    for max_depth in range(2,6)
    for min_child_weight in range(3,4)
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

# seems like max depth 5 and min child weight doesn't make too much difference



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
    timelist.sort()
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

first_place_bets = output_with_top_3_horses[['race_id','betting_odds','pred_place_1','finish_position']]
first_place_bets = first_place_bets[first_place_bets['pred_place_1']]
first_place_bets['win'] = (first_place_bets['finish_position']==1)

first_place_bets['decimal_odds'] = [odds_parser(o) for o in first_place_bets['betting_odds']]
first_place_bets['payouts'] = first_place_bets['decimal_odds']*first_place_bets['win']
sum(first_place_bets['payouts'][first_place_bets['payouts'].notnull()])/sum(first_place_bets['payouts'].notnull())
odds_cutoff = 3
sum(first_place_bets['payouts'][first_place_bets['decimal_odds']<odds_cutoff])/sum(first_place_bets['decimal_odds']<odds_cutoff)



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
dtrain_winnings = xgb.DMatrix(train_data_winnings[features[6:]])
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
features_win = features_y_win + features_extra_info_for_results + features_current_race + features_prs + features_additional

train_data_winnings_to_split = train_data_winnings.copy()
train_data_winnings_to_split = train_data_winnings_to_split[train_data_winnings_to_split['number_preds_check']]
train_data_winnings_to_split = train_data_winnings_to_split[train_data_winnings_to_split['top_3_check']]
train_data_winnings_to_split = train_data_winnings_to_split[features_win]

train_data_winnings_to_split.isnull().sum(axis=0)
train_data_winnings_to_split = train_data_winnings_to_split[train_data_winnings_to_split['betting_odds'].notnull()]

train_idx_winnings = [r in train_races for r in train_data_winnings_to_split['race_id']]
test_idx_winnings = [r in test_races for r in train_data_winnings_to_split['race_id']]

train_X_winnings = train_data_winnings_to_split.loc[train_idx_winnings, features_win[6:]]
train_y_winnings = train_data_winnings_to_split.loc[train_idx_winnings, features_win[0]]
train_extra_info_winnings = train_data_winnings_to_split.loc[train_idx_winnings, features_win[1:6]]
test_X_winnings = train_data_winnings_to_split.loc[test_idx_winnings, features_win[6:]]
test_y_winnings = train_data_winnings_to_split.loc[test_idx_winnings, features_win[0]]
test_extra_info_winnings = train_data_winnings_to_split.loc[test_idx_winnings, features_win[1:6]]

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


pred_win_cutoff = 1
sum(test_y_winnings[xgbModWins_test>pred_win_cutoff])/sum(xgbModWins_test>pred_win_cutoff)

sum(xgbModWins_train>pred_win_cutoff)
sum(xgbModWins_test>pred_win_cutoff)

sum(train_y_winnings[xgbModWins_train>pred_win_cutoff])
sum(test_y_winnings[xgbModWins_test>pred_win_cutoff])

len(train_data_winnings_to_split['race_id'].unique())
