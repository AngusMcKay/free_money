

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
import itertools

'''
get data
'''
connect_string = 'mysql+pymysql://root:angunix1@localhost/horses'
sql_engine = sqlalchemy.create_engine(connect_string)
horses_data = pd.read_sql('''
                          SELECT * FROM training_data_10_pr_no_nrs
                          ''',
                          con=sql_engine)

data_to_inspect = horses_data.isnull().sum(axis=0)



'''
data manipulation
'''
horse_sexes = ['g','m','f','c','h','r']
for t in horse_sexes:
    horses_data['horse_sex_'+t] = (horses_data['horse_sex']==t)*1

horses_data['horse_last_ran_days'][horses_data['horse_last_ran_days'].isnull()] = 0


# add decimal odds
def odds_parser(odds_string):
    try:
        odds_split = odds_string.split('/')
        decimal_odds = (int(odds_split[0])+int(odds_split[1]))/int(odds_split[1])
        return decimal_odds
    except:
        return 1

for i in range(10):
    horses_data['pr_'+str(i+1)+'_decimal_odds'] = [odds_parser(o) for o in horses_data['pr_'+str(i+1)+'_betting_odds']]

odds_cols = ['pr_'+str(i+1)+'_decimal_odds' for i in range(10)]
horses_data['average_past_odds'] = horses_data[odds_cols].apply(np.nanmean, axis=1)


# calculate horse speed and expected time based on past results
def horse_speed(race_length, race_time):
    try:
        return race_length/race_time
    except:
        return None

for i in range(10):
    horses_data['pr_'+str(i+1)+'_speed'] = [horse_speed(rl, rt) for rl, rt
                in zip(horses_data['pr_'+str(i+1)+'_yards'], horses_data['pr_'+str(i+1)+'_horse_time'])]

speeds_cols = ['pr_'+str(i+1)+'_speed' for i in range(10)]
horses_data['average_past_speed'] = horses_data[speeds_cols].apply(np.nanmean, axis=1)
horses_data['expected_time'] = horses_data['yards']/horses_data['average_past_speed']


# add # past wins and past win %
finish_position_cols = ['pr_'+str(i+1)+'_finish_position' for i in range(10)]
horses_data['number_past_wins'] = (horses_data[finish_position_cols]==1).apply(np.nansum, axis=1)
horses_data['percentage_past_wins'] = horses_data['number_past_wins']/horses_data[finish_position_cols].notnull().sum(axis=1)

data_to_inspect = horses_data[:100]


'''
training function
'''
def model_training(train_data, feature_y, feature_cur, feature_pr, number_past_results):
    # set up data
    features_prs = []
    if number_past_results>0:
        for i in range(1, number_past_results+1):
            features_prs = features_prs + ['pr_'+str(i)+'_'+pr for pr in features_pr]
    
    features = feature_y + feature_cur + features_prs
    train_data = train_data[train_data[features].isnull().sum(axis=1)==0]
    train_split, test_split = train_test_split(train_data, test_size=0.1)
    train_X = train_split[feature_cur + features_prs]
    train_y = train_split[feature_y]
    test_X = test_split[feature_cur + features_prs]
    test_y = test_split[feature_y]
    
    # create model
    params = {
        'max_depth':3,
        'min_child_weight': 5,
        'eta':0.5,
        'objective':'reg:linear',
        'eval_metric':'rmse'
    }
    num_boost_round = 1000
    early_stopping = 10
    
    dtrain = xgb.DMatrix(train_X, label=train_y)
    dtest = xgb.DMatrix(test_X, label=test_y)

    model = xgb.train(params, dtrain, num_boost_round=num_boost_round,
                      early_stopping_rounds=early_stopping,
                      evals=[(dtest, "Test")])
    
    # train variance model
    preds_train = model.predict(dtrain)
    preds_test = model.predict(dtest)
    train_y_var = (preds_train - np.array(train_y).flatten())**2
    test_y_var = (preds_test - np.array(test_y).flatten())**2
    
    params_var = {
        'max_depth':5,
        'min_child_weight': 5,
        'eta':.01,
        'objective':'reg:linear',
        'eval_metric':'mae'
    }
    
    dtrain_var = xgb.DMatrix(train_X, label=train_y_var)
    dtest_var = xgb.DMatrix(test_X, label=test_y_var)
    model_var = xgb.train(params_var, dtrain_var, num_boost_round=num_boost_round,
                          early_stopping_rounds=early_stopping,
                          evals=[(dtest_var, "Test")])
    
    return model, model_var









'''
train horse times models
'''
train_data_up_to = pd.to_datetime('2019-03-01')
train_idx = [d < train_data_up_to for d in pd.to_datetime(horses_data['race_date'])]
test_idx = [d >= train_data_up_to for d in pd.to_datetime(horses_data['race_date'])]
train_data = horses_data[train_idx]

features_y = ['horse_time']
features_current_race = ['yards', 'runners', 'handicap_pounds', 'horse_age',
                         'horse_sex_g','horse_sex_m','horse_sex_f','horse_sex_c','horse_sex_h','horse_sex_r',
                         'horse_last_ran_days', 'going_grouped_horse_time_rc',
                         'race_type_horse_time_rc', 'weather_horse_time_rc',
                         'average_past_speed', 'expected_time', 'average_past_odds']
features_pr = ['horse_time', 'finish_position_for_ordering', 'yards',
               'runners', 'handicap_pounds', 'going_grouped_horse_time_rc',
               'race_type_horse_time_rc', 'weather_horse_time_rc', 'decimal_odds', 'speed']

pr10Mod, pr10Mod_var = model_training(train_data=train_data, feature_y=features_y,
                                      feature_cur=features_current_race,
                                      feature_pr=features_pr, number_past_results=10)

pr09Mod, pr09Mod_var = model_training(train_data=train_data, feature_y=features_y,
                                      feature_cur=features_current_race,
                                      feature_pr=features_pr, number_past_results=9)

pr08Mod, pr08Mod_var = model_training(train_data=train_data, feature_y=features_y,
                                      feature_cur=features_current_race,
                                      feature_pr=features_pr, number_past_results=8)

pr07Mod, pr07Mod_var = model_training(train_data=train_data, feature_y=features_y,
                                      feature_cur=features_current_race,
                                      feature_pr=features_pr, number_past_results=7)

pr06Mod, pr06Mod_var = model_training(train_data=train_data, feature_y=features_y,
                                      feature_cur=features_current_race,
                                      feature_pr=features_pr, number_past_results=6)

pr05Mod, pr05Mod_var = model_training(train_data=train_data, feature_y=features_y,
                                      feature_cur=features_current_race,
                                      feature_pr=features_pr, number_past_results=5)

pr04Mod, pr04Mod_var = model_training(train_data=train_data, feature_y=features_y,
                                      feature_cur=features_current_race,
                                      feature_pr=features_pr, number_past_results=4)

pr03Mod, pr03Mod_var = model_training(train_data=train_data, feature_y=features_y,
                                      feature_cur=features_current_race,
                                      feature_pr=features_pr, number_past_results=3)

pr02Mod, pr02Mod_var = model_training(train_data=train_data, feature_y=features_y,
                                      feature_cur=features_current_race,
                                      feature_pr=features_pr, number_past_results=2)

pr01Mod, pr01Mod_var = model_training(train_data=train_data, feature_y=features_y,
                                      feature_cur=features_current_race,
                                      feature_pr=features_pr, number_past_results=1)

pr00Mod, pr00Mod_var = model_training(train_data=train_data, feature_y=features_y,
                                      feature_cur=features_current_race,
                                      feature_pr=features_pr, number_past_results=0)



'''
add prediction times and variances to data
'''
def get_time_predictions(data, feature_cur, feature_pr, number_past_results,
                         horse_time_model, variance_model, overwrite_preds=False):
    features_prs = []
    if number_past_results>0:
        for i in range(1, number_past_results+1):
            features_prs = features_prs + ['pr_'+str(i)+'_'+pr for pr in features_pr]
    
    features = feature_cur + features_prs
    
    pred_data = data.copy()
    if not overwrite_preds:
        if 'pred_time' in pred_data.columns:
            pred_data = pred_data[pred_data['pred_time'].isnull()]
        if 'pred_var' in data.columns:
            pred_data = pred_data[pred_data['pred_var'].isnull()]
    
    pred_data = pred_data[pred_data[features].isnull().sum(axis=1)==0]
    pred_data = pred_data[features]
    dpred = xgb.DMatrix(pred_data)
    
    pred_times = horse_time_model.predict(dpred)
    pred_vars = variance_model.predict(dpred)
    pred_data['pred_time'] = pred_times
    pred_data['pred_var'] = pred_vars
    pred_data['pred_prs'] = number_past_results
    
    if 'pred_time' not in data.columns:
        data['pred_time'] = None
    if 'pred_var' not in data.columns:
        data['pred_var'] = None
    if 'pred_prs' not in data.columns:
        data['pred_prs'] = None
    
    data.loc[pred_data.index,'pred_time'] = pred_data['pred_time']
    data.loc[pred_data.index,'pred_var'] = pred_data['pred_var']
    data.loc[pred_data.index,'pred_prs'] = pred_data['pred_prs']
    
    return data


train_data_with_time_preds = train_data.copy()
train_data_with_time_preds = get_time_predictions(train_data_with_time_preds, feature_cur=features_current_race,
                                             feature_pr=features_pr, number_past_results=10,
                                             horse_time_model=pr10Mod, variance_model=pr10Mod_var)

train_data_with_time_preds = get_time_predictions(train_data_with_time_preds, feature_cur=features_current_race,
                                             feature_pr=features_pr, number_past_results=9,
                                             horse_time_model=pr09Mod, variance_model=pr09Mod_var)

train_data_with_time_preds = get_time_predictions(train_data_with_time_preds, feature_cur=features_current_race,
                                             feature_pr=features_pr, number_past_results=8,
                                             horse_time_model=pr08Mod, variance_model=pr08Mod_var)

train_data_with_time_preds = get_time_predictions(train_data_with_time_preds, feature_cur=features_current_race,
                                             feature_pr=features_pr, number_past_results=7,
                                             horse_time_model=pr07Mod, variance_model=pr07Mod_var)

train_data_with_time_preds = get_time_predictions(train_data_with_time_preds, feature_cur=features_current_race,
                                             feature_pr=features_pr, number_past_results=6,
                                             horse_time_model=pr06Mod, variance_model=pr06Mod_var)

train_data_with_time_preds = get_time_predictions(train_data_with_time_preds, feature_cur=features_current_race,
                                             feature_pr=features_pr, number_past_results=5,
                                             horse_time_model=pr05Mod, variance_model=pr05Mod_var)

train_data_with_time_preds = get_time_predictions(train_data_with_time_preds, feature_cur=features_current_race,
                                             feature_pr=features_pr, number_past_results=4,
                                             horse_time_model=pr04Mod, variance_model=pr04Mod_var)

train_data_with_time_preds = get_time_predictions(train_data_with_time_preds, feature_cur=features_current_race,
                                             feature_pr=features_pr, number_past_results=3,
                                             horse_time_model=pr03Mod, variance_model=pr03Mod_var)

train_data_with_time_preds = get_time_predictions(train_data_with_time_preds, feature_cur=features_current_race,
                                             feature_pr=features_pr, number_past_results=2,
                                             horse_time_model=pr02Mod, variance_model=pr02Mod_var)

train_data_with_time_preds = get_time_predictions(train_data_with_time_preds, feature_cur=features_current_race,
                                             feature_pr=features_pr, number_past_results=1,
                                             horse_time_model=pr01Mod, variance_model=pr01Mod_var)

train_data_with_time_preds = get_time_predictions(train_data_with_time_preds, feature_cur=features_current_race,
                                             feature_pr=features_pr, number_past_results=0,
                                             horse_time_model=pr00Mod, variance_model=pr00Mod_var)



'''
create model to predict places based on time and variance predictions
'''
def top_3_check(positions):
    positions_list = list(positions)
    return (1 in positions_list) & (2 in positions_list) & (3 in positions_list)
def first_item(x):
    return list(x)[0]


## note: non-finishers (incl. non-runners) discluded from training data
## NEED TO INCLUDE NON-FINISHERS WHO WERE RUNNERS
train_data_with_time_preds_complete = train_data_with_time_preds[train_data_with_time_preds['pred_time'].notnull()]
train_data_with_time_preds_complete['decimal_odds'] = [odds_parser(o) for o in train_data_with_time_preds_complete['betting_odds']]

races_data = train_data_with_time_preds_complete.groupby('race_id').aggregate(
        {
                'race_date': first_item,
                'course': first_item,
                'country': first_item,
                'yards': first_item,
                'going_grouped': first_item,
                'race_type_devised': first_item,
                'weather_grouped': first_item,
                'horse_id': list,
                'pred_time': list,
                'pred_var': list,
                'pred_prs': list,
                'average_past_odds': list,
                'number_past_wins': list,
                'percentage_past_wins': list,
                'finish_position_for_ordering': [list, top_3_check],
                'handicap_pounds': list,
                'horse_age': list,
                'horse_sex': list,
                'horse_last_ran_days': list,
                'did_not_finish': list,
                'runners': [len, max],
                'decimal_odds': list
                }
        ).reset_index()

rename_columns = ['race_id','race_date','course','country','yards','going_grouped',
                  'race_type_devised','weather_grouped','horse_id_list','pred_time_list',
                  'pred_var_list','pred_prs_list','average_past_odds_list',
                  'number_past_wins_list','percentage_past_wins_list','finish_position_for_ordering_list',
                  'top_3_check','handicap_pounds_list','horse_age_list','horse_sex_list',
                  'horse_last_ran_days_list','did_not_finish_list','runners_in_data','runners_given',
                  'decimal_odds_list']
races_data.columns = rename_columns

races_data['num_runners_check'] = races_data['runners_in_data']>=races_data['runners_given']

def create_win_place_train_and_pred_data(races_data, number_runners_min, number_runners_max,
                                         train_or_pred='train', pred_with_result=False,
                                         order_inputs=True, rounding=5):
    '''
    data input is df with lists containing features for each horse (as created above)
    it is intended that a separate model is created for each number of runners
    '''
    subset_data = races_data[races_data['runners_in_data'].isin(range(number_runners_min, number_runners_max+1))]
    subset_data = subset_data[subset_data['top_3_check'] & subset_data['num_runners_check']]
    
    if order_inputs:
        horse_pred_time_orders = [np.argsort(l) for l in subset_data['pred_time_list']]
    else:
        horse_pred_time_orders = [list(range(len(l))) for l in subset_data['pred_time_list']]
    
    def order_and_add_defaults(to_be_ordered, orders, default, length, rounding):
        if rounding==None:
            ordered_list = list(np.array(to_be_ordered)[orders])
        else:
            ordered_list = list(np.around(np.array(to_be_ordered)[orders],rounding))
        
        defaults = [default]*(length - len(ordered_list))
        return ordered_list + defaults
    
    # get arrays of items, in order if ordering inputs
    pred_times_lists_ordered = np.array([order_and_add_defaults(tbo, o, 999999, number_runners_max, rounding) for tbo, o in zip(subset_data['pred_time_list'], horse_pred_time_orders)])
    pred_vars_lists_ordered = np.array([order_and_add_defaults(tbo, o, 999999, number_runners_max, rounding) for tbo, o in zip(subset_data['pred_var_list'], horse_pred_time_orders)])
    pred_prs_lists_ordered = np.array([order_and_add_defaults(tbo, o, 0, number_runners_max, rounding) for tbo, o in zip(subset_data['pred_prs_list'], horse_pred_time_orders)])
    average_past_odds_lists_ordered = np.array([order_and_add_defaults(tbo, o, 999, number_runners_max, 2) for tbo, o in zip(subset_data['average_past_odds_list'], horse_pred_time_orders)])
    number_past_wins_lists_ordered = np.array([order_and_add_defaults(tbo, o, 999, number_runners_max, 2) for tbo, o in zip(subset_data['number_past_wins_list'], horse_pred_time_orders)])
    percentage_past_wins_lists_ordered = np.array([order_and_add_defaults(tbo, o, 999, number_runners_max, 2) for tbo, o in zip(subset_data['percentage_past_wins_list'], horse_pred_time_orders)])
    odds_lists_ordered = np.array([order_and_add_defaults(tbo, o, 999, number_runners_max, 2) for tbo, o in zip(subset_data['decimal_odds_list'], horse_pred_time_orders)])
    
    pred_time_cols = ['cp_'+str(i)+'_pred_time' for i in range(number_runners_max)]
    pred_var_cols = ['cp_'+str(i)+'_pred_var' for i in range(number_runners_max)]
    pred_prs_cols = ['cp_'+str(i)+'_pred_prs' for i in range(number_runners_max)]
    avg_odds_cols = ['cp_'+str(i)+'_average_past_odds' for i in range(number_runners_max)]
    num_pwin_cols = ['cp_'+str(i)+'_number_past_wins' for i in range(number_runners_max)]
    pct_pwin_cols = ['cp_'+str(i)+'_percentage_past_wins' for i in range(number_runners_max)]
    odds_cols = ['cp_'+str(i)+'_odds' for i in range(number_runners_max)]
    
    features = pred_time_cols + pred_var_cols + pred_prs_cols + avg_odds_cols + num_pwin_cols + pct_pwin_cols #+ odds_cols
    
    if train_or_pred.lower()=='train':
        finish_position_lists_ordered = np.array([order_and_add_defaults(tbo, o, 999, number_runners_max, rounding) for tbo, o in zip(subset_data['finish_position_for_ordering_list'], horse_pred_time_orders)])
        df = subset_data[['race_id','yards']].reset_index(drop=True)
        df[pred_time_cols] = pd.DataFrame(pred_times_lists_ordered, columns=pred_time_cols)
        df[pred_var_cols] = pd.DataFrame(pred_vars_lists_ordered, columns=pred_var_cols)
        df[pred_prs_cols] = pd.DataFrame(pred_prs_lists_ordered, columns=pred_prs_cols)
        df[avg_odds_cols] = pd.DataFrame(average_past_odds_lists_ordered, columns=avg_odds_cols)
        df[num_pwin_cols] = pd.DataFrame(number_past_wins_lists_ordered, columns=num_pwin_cols)
        df[pct_pwin_cols] = pd.DataFrame(percentage_past_wins_lists_ordered, columns=pct_pwin_cols)
        df[odds_cols] = pd.DataFrame(odds_lists_ordered, columns=odds_cols)
        df['winner'] = [list(x).index(1) for x in list(finish_position_lists_ordered)]
        df['second'] = [list(x).index(2) for x in list(finish_position_lists_ordered)]
        df['third'] = [list(x).index(3) for x in list(finish_position_lists_ordered)]
        
        train_data_winner = df[df[features].isnull().sum(axis=1)==0]
        race_ids = train_data_winner['race_id'].unique()
        
        train_races, test_races = train_test_split(race_ids, test_size=0.2)
        train_set = train_data_winner[train_data_winner['race_id'].isin(train_races)]
        test_set = train_data_winner[train_data_winner['race_id'].isin(test_races)]
        train_X = train_set[features]
        test_X = test_set[features]
        
        return train_set, train_X, test_set, test_X
    
    else:
        original_index = subset_data.index
        df = subset_data.copy()
        df[pred_time_cols] = pd.DataFrame(pred_times_lists_ordered, columns=pred_time_cols, index=original_index)
        df[pred_var_cols] = pd.DataFrame(pred_vars_lists_ordered, columns=pred_var_cols, index=original_index)
        df[pred_prs_cols] = pd.DataFrame(pred_prs_lists_ordered, columns=pred_prs_cols, index=original_index)
        df[avg_odds_cols] = pd.DataFrame(average_past_odds_lists_ordered, columns=avg_odds_cols, index=original_index)
        df[num_pwin_cols] = pd.DataFrame(number_past_wins_lists_ordered, columns=num_pwin_cols, index=original_index)
        df[pct_pwin_cols] = pd.DataFrame(percentage_past_wins_lists_ordered, columns=pct_pwin_cols, index=original_index)
        df[odds_cols] = pd.DataFrame(odds_lists_ordered, columns=odds_cols, index=original_index)
        if order_inputs:
            additional_columns_to_reorder = ['horse_id_list', 'pred_time_list', 'pred_var_list',
                                             'pred_prs_list','finish_position_for_ordering_list',
                                             'handicap_pounds_list', 'horse_age_list', 'horse_sex_list',
                                             'horse_last_ran_days_list', 'did_not_finish_list', 'decimal_odds_list'
                                             ]
            for c in additional_columns_to_reorder:
                df[c] = [list(np.array(l)[o]) for l, o in zip(df[c], horse_pred_time_orders)]
            
        if pred_with_result:
            finish_position_lists_ordered = np.array([order_and_add_defaults(tbo, o, 999, number_runners_max, None) for tbo, o in zip(subset_data['finish_position_for_ordering_list'], horse_pred_time_orders)])
            df['winner'] = [list(x).index(1) for x in list(finish_position_lists_ordered)]
            df['second'] = [list(x).index(2) for x in list(finish_position_lists_ordered)]
            df['third'] = [list(x).index(3) for x in list(finish_position_lists_ordered)]
        
        pred_data = df[df[features].isnull().sum(axis=1)==0]
        pred_X = pred_data[features]
        
        return pred_data, pred_X




def win_and_place_model_training(races_data, number_runners_min, number_runners_max, order_training_inputs=True, rounding=0):
    '''
    data input is df with lists containing features for each horse (as created above)
    it is intended that a separate model is created for each number of runners
    '''
    train_set, train_X, test_set, test_X = create_win_place_train_and_pred_data(
            races_data=races_data, number_runners_min=number_runners_min,
            number_runners_max=number_runners_max, order_inputs=order_training_inputs,
            rounding=rounding
            )
    
    # train win model
    params = {
        'max_depth':2,
        'min_child_weight': 5,
        'eta':1, # set to 1 for rf
        'col_sample_by_tree':0.5, # set to < 1 for rf
        'subsample':0.5, # set to < 1 for rf
        'num_parallel_tree':30, # set to >1 for rf (remove for gbt)
        'objective':'multi:softprob',
        'eval_metric':'mlogloss',
        'num_class':number_runners_max
    }
    num_boost_round = 1
    early_stopping = 1
    
    dtrain = xgb.DMatrix(train_X, label=train_set[['winner']])
    dtest = xgb.DMatrix(test_X, label=test_set[['winner']])
    
    model_winner = xgb.train(params, dtrain, num_boost_round=num_boost_round,
                             early_stopping_rounds=early_stopping,
                             evals=[(dtest, "Test")])
    
    dtrain = xgb.DMatrix(train_X, label=train_set[['second']])
    dtest = xgb.DMatrix(test_X, label=test_set[['second']])
    
    model_second = xgb.train(params, dtrain, num_boost_round=num_boost_round,
                             early_stopping_rounds=early_stopping,
                             evals=[(dtest, "Test")])
    
    dtrain = xgb.DMatrix(train_X, label=train_set[['third']])
    dtest = xgb.DMatrix(test_X, label=test_set[['third']])
    
    model_third = xgb.train(params, dtrain, num_boost_round=num_boost_round,
                             early_stopping_rounds=early_stopping,
                             evals=[(dtest, "Test")])
    
    return [model_winner, model_second, model_third]



# train models
races_data_train_win_place, races_data_train_trifecta = train_test_split(races_data, test_size=0.5, random_state=123)
runners_in_data = races_data_train_win_place['runners_in_data'].unique()
runners_in_data.sort()
runners_in_data
win_place_models_number_runners_list = list(range(3,25))
win_place_models_dict = {}
for r in win_place_models_number_runners_list[:-2]:
    win_place_models_dict[r] = win_and_place_model_training(
            races_data=races_data_train_win_place,
            number_runners_min=r, number_runners_max=r+2)





'''
predict places and train trifecta model
'''
def get_win_and_place_predictions(races_data, models_dict, order_prediction_inputs=True, rounding=0, padding=0):
    num_racers_list = list(models_dict.keys())
    
    output_data_columns = list(races_data.columns)+['pred_winner','pred_second','pred_third']
    output_df = pd.DataFrame(columns=output_data_columns)
    for n in num_racers_list:
        pred_data, pred_X = create_win_place_train_and_pred_data(
                races_data=races_data, number_runners_min=n, number_runners_max=n+padding,
                train_or_pred='prediction', pred_with_result=True,
                order_inputs=order_prediction_inputs, rounding=rounding
                )
        win_mod, sec_mod, thi_mod = models_dict[n]
        win_preds = win_mod.predict(xgb.DMatrix(pred_X))
        win_preds_lists = [list(l) for l in win_preds]
        pred_data['pred_winner'] = win_preds_lists
        
        sec_preds = sec_mod.predict(xgb.DMatrix(pred_X))
        sec_preds_lists = [list(l) for l in sec_preds]
        pred_data['pred_second'] = sec_preds_lists
        
        thi_preds = thi_mod.predict(xgb.DMatrix(pred_X))
        thi_preds_lists = [list(l) for l in thi_preds]
        pred_data['pred_third'] = thi_preds_lists
        
        output_df = output_df.append(pred_data[output_data_columns])
    
    return output_df


races_data_train_win_place_with_preds = get_win_and_place_predictions(
        races_data=races_data_train_win_place, models_dict=win_place_models_dict,
        order_prediction_inputs=True, rounding=0, padding=2)

races_data_train_trifecta_with_preds = get_win_and_place_predictions(
        races_data=races_data_train_trifecta, models_dict=win_place_models_dict,
        order_prediction_inputs=True, rounding=0, padding=2)

data_to_inspect=races_data_train_trifecta_with_preds[:100]
# get trifecta data
connect_string = 'mysql+pymysql://root:angunix1@localhost/horses'
sql_engine = sqlalchemy.create_engine(connect_string)
trifecta_payouts = pd.read_sql('''
                          SELECT DISTINCT race_id, SUBSTRING_INDEX(trifecta, ' ', 1) AS trifecta_payout FROM horses_data
                          WHERE race_id IN (SELECT race_id FROM races_data WHERE country IN ('scotland', 'england', 'northern ireland', 'eire', 'wales'))
                          ''',
                          con=sql_engine)

#races_data_with_predictions = races_data_with_predictions.merge(trifecta_payouts,
#                                                                how='left', on='race_id')


races_data_train_trifecta_with_preds = races_data_train_trifecta_with_preds.merge(trifecta_payouts,
                                                                how='left', on='race_id')

trifecta_data = races_data_train_trifecta_with_preds[races_data_train_trifecta_with_preds['trifecta_payout'].notnull()]

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

trifecta_data['trifecta_horse_combos'] = [get_trifecta_horse_combos(h) for h in tqdm(trifecta_data['horse_id_list'])]

def get_trifecta_pred_prob(trifecta_selection, horse_list, win_probs, second_probs, third_probs):
    try:
        winner_win = win_probs[horse_list.index(trifecta_selection[0])]
        second_win = win_probs[horse_list.index(trifecta_selection[1])]
        second_2nd = second_probs[horse_list.index(trifecta_selection[1])]
        third_win = win_probs[horse_list.index(trifecta_selection[2])]
        third_2nd = second_probs[horse_list.index(trifecta_selection[2])]
        third_3rd = third_probs[horse_list.index(trifecta_selection[2])]
        return round(winner_win * (second_2nd / (1-second_win)) * (third_3rd / (1 - third_win - third_2nd)),5)
    except:
        return 0
    
def get_trifecta_pred_probs(trifecta_horse_combos, horse_list, win_probs, second_probs, third_probs):
    return [get_trifecta_pred_prob(ts, horse_list, win_probs, second_probs, third_probs) for ts in trifecta_horse_combos]

trifecta_data['trifect_combo_probs'] = [get_trifecta_pred_probs(thc, hl, wp, sp, tp) for thc, hl, wp, sp, tp in tqdm(zip(trifecta_data['trifecta_horse_combos'], trifecta_data['horse_id_list'], trifecta_data['pred_winner'], trifecta_data['pred_second'], trifecta_data['pred_third']), total=len(trifecta_data))]

def get_trifecta_winning_combo(horse_list, finish_positions):
    return (horse_list[finish_positions.index(1)], horse_list[finish_positions.index(2)], horse_list[finish_positions.index(3)])

trifecta_data['winning_selection'] = [get_trifecta_winning_combo(hl, fp) for hl, fp in tqdm(zip(trifecta_data['horse_id_list'], trifecta_data['finish_position_for_ordering_list']), total=len(trifecta_data))]

def get_trifecta_pred_features_for_selection(trifecta_selection, horse_list, win_probs, second_probs, third_probs, runners, winning_selection, trifecta_payout):
    try:
        winner_win = win_probs[horse_list.index(trifecta_selection[0])]
        winner_2nd = second_probs[horse_list.index(trifecta_selection[0])]
        winner_3rd = third_probs[horse_list.index(trifecta_selection[0])]
        second_win = win_probs[horse_list.index(trifecta_selection[1])]
        second_2nd = second_probs[horse_list.index(trifecta_selection[1])]
        second_3rd = third_probs[horse_list.index(trifecta_selection[1])]
        third_win = win_probs[horse_list.index(trifecta_selection[2])]
        third_2nd = second_probs[horse_list.index(trifecta_selection[2])]
        third_3rd = third_probs[horse_list.index(trifecta_selection[2])]
        trifecta_prob = round(winner_win * (second_2nd / (1-second_win)) * (third_3rd / (1 - third_win - third_2nd)),5)
        if trifecta_selection==winning_selection:
            payout = trifecta_payout
        else:
            payout = 0
        return [winner_win, winner_2nd, winner_3rd, second_win, second_2nd, second_3rd, third_win, third_2nd, third_3rd, trifecta_prob, runners, payout]
    except:
        return []

def get_trifecta_pred_features_for_all_selections(trifecta_selections, horse_list, win_probs, second_probs, third_probs, runners, winning_selection, trifecta_payout):
    return [get_trifecta_pred_features_for_selection(ts, horse_list, win_probs, second_probs, third_probs, runners, winning_selection, trifecta_payout) for ts in trifecta_selections]

trifecta_pred_df = [get_trifecta_pred_features_for_all_selections(tss, hl, wp, sp, tp, r, ws, p) for tss, hl, wp, sp, tp, r, ws, p in tqdm(zip(trifecta_data['trifecta_horse_combos'], trifecta_data['horse_id_list'], trifecta_data['pred_winner'], trifecta_data['pred_second'], trifecta_data['pred_third'], trifecta_data['runners_in_data'], trifecta_data['winning_selection'], trifecta_data['trifecta_payout']), total=len(trifecta_data))]
trifecta_pred_df = [item for sublist in trifecta_pred_df for item in sublist]
trifecta_pred_df = [f for f in trifecta_pred_df if f!=[]]
trifecta_pred_df = pd.DataFrame(trifecta_pred_df, columns=['winner_win', 'winner_2nd', 'winner_3rd', 'second_win', 'second_2nd', 'second_3rd', 'third_win', 'third_2nd', 'third_3rd', 'trifecta_prob', 'runners', 'payout'])
trifecta_pred_df['payout'] = trifecta_pred_df['payout'].astype(float)
sys.getsizeof(trifecta_pred_df)/(1024*1024*1024)

# delete trifecta data to free up some space
del(trifecta_data)

# train trifecta model
min_runners = 7
max_runners = 7
trifecta_pred_df_subset = trifecta_pred_df[(trifecta_pred_df['runners']>=min_runners) & (trifecta_pred_df['runners']<=max_runners)]
len(trifecta_pred_df_subset)

data_to_inspect = trifecta_pred_df_subset[:1000]

# NOTE: WILL NEED TO ADD INFO LIKE RACE AND HORSE IDS TO DATA
trifecta_train_data, trifecta_test_data = train_test_split(trifecta_pred_df_subset, test_size=0.2, random_state=123)
trifecta_features = trifecta_train_data.columns[:-1]
trifecta_train_X = trifecta_train_data[trifecta_features]
trifecta_train_y = trifecta_train_data['payout']
trifecta_test_X = trifecta_test_data[trifecta_features]
trifecta_test_y = trifecta_test_data['payout']

params = {
    'max_depth':1,
    'min_child_weight': 10,
    'eta':0.11, # set to 1 for rf
    'col_sample_by_tree':0.8, # set to < 1 for rf
    'subsample':0.8, # set to < 1 for rf
    #'num_parallel_tree':50, # set to >1 for rf (remove for gbt)
    'objective':'reg:squarederror',
    'eval_metric':'rmse',
}
num_boost_round = 100
early_stopping = 10

dtrain_trifecta = xgb.DMatrix(trifecta_train_X, label=trifecta_train_y)
dtest_trifecta = xgb.DMatrix(trifecta_test_X, label=trifecta_test_y)

model_trifecta = xgb.train(params, dtrain_trifecta, num_boost_round=num_boost_round,
                         early_stopping_rounds=early_stopping,
                         evals=[(dtest_trifecta, "Test")])

trifecta_train_preds = model_trifecta.predict(dtrain_trifecta)
trifecta_test_preds = model_trifecta.predict(dtest_trifecta)

trifecta_bet_cutoff = 1
sum(trifecta_train_preds>trifecta_bet_cutoff)
sum(trifecta_train_y[trifecta_train_preds>trifecta_bet_cutoff])
sum(trifecta_test_preds>trifecta_bet_cutoff)
sum(trifecta_test_y[trifecta_test_preds>trifecta_bet_cutoff])
data_to_inspect = trifecta_test_y[trifecta_test_preds>trifecta_bet_cutoff]


import statsmodels.api as sm
linMod = sm.OLS(trifecta_train_y, sm.add_constant(trifecta_train_X))
linModFit = linMod.fit()
linModFit.summary()
linMod_train_preds = linModFit.predict(sm.add_constant(trifecta_train_X))
linMod_test_preds = linModFit.predict(sm.add_constant(trifecta_test_X))

trifecta_bet_cutoff = 1
sum(linMod_train_preds>trifecta_bet_cutoff)
sum(trifecta_train_y[linMod_train_preds>trifecta_bet_cutoff])
sum(linMod_test_preds>trifecta_bet_cutoff)
sum(trifecta_test_y[linMod_test_preds>trifecta_bet_cutoff])





'''
run full process on held out testing data
'''
test_data = horses_data[test_idx]
features_y = ['horse_time']
features_current_race = ['yards', 'runners', 'handicap_pounds', 'horse_age',
                         'horse_sex_g','horse_sex_m','horse_sex_f','horse_sex_c','horse_sex_h','horse_sex_r',
                         'horse_last_ran_days', 'going_grouped_horse_time_rc',
                         'race_type_horse_time_rc', 'weather_horse_time_rc',
                         'average_past_speed', 'expected_time', 'average_past_odds']
features_pr = ['horse_time', 'finish_position_for_ordering', 'yards',
               'runners', 'handicap_pounds', 'going_grouped_horse_time_rc',
               'race_type_horse_time_rc', 'weather_horse_time_rc', 'decimal_odds', 'speed']

# horse time preds
test_data_with_time_preds = test_data.copy()
test_data_with_time_preds = get_time_predictions(test_data_with_time_preds, feature_cur=features_current_race,
                                             feature_pr=features_pr, number_past_results=10,
                                             horse_time_model=pr10Mod, variance_model=pr10Mod_var)

test_data_with_time_preds = get_time_predictions(test_data_with_time_preds, feature_cur=features_current_race,
                                             feature_pr=features_pr, number_past_results=9,
                                             horse_time_model=pr09Mod, variance_model=pr09Mod_var)

test_data_with_time_preds = get_time_predictions(test_data_with_time_preds, feature_cur=features_current_race,
                                             feature_pr=features_pr, number_past_results=8,
                                             horse_time_model=pr08Mod, variance_model=pr08Mod_var)

test_data_with_time_preds = get_time_predictions(test_data_with_time_preds, feature_cur=features_current_race,
                                             feature_pr=features_pr, number_past_results=7,
                                             horse_time_model=pr07Mod, variance_model=pr07Mod_var)

test_data_with_time_preds = get_time_predictions(test_data_with_time_preds, feature_cur=features_current_race,
                                             feature_pr=features_pr, number_past_results=6,
                                             horse_time_model=pr06Mod, variance_model=pr06Mod_var)

test_data_with_time_preds = get_time_predictions(test_data_with_time_preds, feature_cur=features_current_race,
                                             feature_pr=features_pr, number_past_results=5,
                                             horse_time_model=pr05Mod, variance_model=pr05Mod_var)

test_data_with_time_preds = get_time_predictions(test_data_with_time_preds, feature_cur=features_current_race,
                                             feature_pr=features_pr, number_past_results=4,
                                             horse_time_model=pr04Mod, variance_model=pr04Mod_var)

test_data_with_time_preds = get_time_predictions(test_data_with_time_preds, feature_cur=features_current_race,
                                             feature_pr=features_pr, number_past_results=3,
                                             horse_time_model=pr03Mod, variance_model=pr03Mod_var)

test_data_with_time_preds = get_time_predictions(test_data_with_time_preds, feature_cur=features_current_race,
                                             feature_pr=features_pr, number_past_results=2,
                                             horse_time_model=pr02Mod, variance_model=pr02Mod_var)

test_data_with_time_preds = get_time_predictions(test_data_with_time_preds, feature_cur=features_current_race,
                                             feature_pr=features_pr, number_past_results=1,
                                             horse_time_model=pr01Mod, variance_model=pr01Mod_var)

test_data_with_time_preds = get_time_predictions(test_data_with_time_preds, feature_cur=features_current_race,
                                             feature_pr=features_pr, number_past_results=0,
                                             horse_time_model=pr00Mod, variance_model=pr00Mod_var)


# carry out win prob preds for each race
test_data_with_time_preds_complete = test_data_with_time_preds[test_data_with_time_preds['pred_time'].notnull()]
test_data_with_time_preds_complete['decimal_odds'] = [odds_parser(o) for o in test_data_with_time_preds_complete['betting_odds']]

races_test_data = test_data_with_time_preds_complete.groupby('race_id').aggregate(
        {
                'race_date': first_item,
                'course': first_item,
                'country': first_item,
                'yards': first_item,
                'going_grouped': first_item,
                'race_type_devised': first_item,
                'weather_grouped': first_item,
                'horse_id': list,
                'pred_time': list,
                'pred_var': list,
                'pred_prs': list,
                'average_past_odds': list,
                'number_past_wins': list,
                'percentage_past_wins': list,
                'finish_position_for_ordering': [list, top_3_check],
                'handicap_pounds': list,
                'horse_age': list,
                'horse_sex': list,
                'horse_last_ran_days': list,
                'did_not_finish': list,
                'runners': [len, max],
                'decimal_odds': list
                }
        ).reset_index()

rename_columns = ['race_id','race_date','course','country','yards','going_grouped',
                  'race_type_devised','weather_grouped','horse_id_list','pred_time_list',
                  'pred_var_list','pred_prs_list','average_past_odds_list',
                  'number_past_wins_list','percentage_past_wins_list','finish_position_for_ordering_list',
                  'top_3_check','handicap_pounds_list','horse_age_list','horse_sex_list',
                  'horse_last_ran_days_list','did_not_finish_list','runners_in_data','runners_given',
                  'decimal_odds_list']
races_test_data.columns = rename_columns

races_test_data['num_runners_check'] = races_test_data['runners_in_data']>=races_test_data['runners_given']


win_place_preds_number_runners_list = list(range(3,25))
win_place_preds_number_runners_list = [n for n in win_place_preds_number_runners_list if n in races_test_data['runners_in_data'].unique()]

races_test_data_with_predictions = get_win_and_place_predictions(
        races_data=races_test_data, models_dict=win_place_models_dict,
        order_prediction_inputs=True, rounding=0, padding=2)



# get trifecta data and preds
connect_string = 'mysql+pymysql://root:angunix1@localhost/horses'
sql_engine = sqlalchemy.create_engine(connect_string)
trifecta_payouts = pd.read_sql('''
                          SELECT DISTINCT race_id, SUBSTRING_INDEX(trifecta, ' ', 1) AS trifecta_payout FROM horses_data
                          WHERE race_id IN (SELECT race_id FROM races_data WHERE country IN ('scotland', 'england', 'northern ireland', 'eire', 'wales'))
                          ''',
                          con=sql_engine)

races_test_data_with_predictions = races_test_data_with_predictions.merge(trifecta_payouts,
                                                                how='left', on='race_id')

trifecta_test_data = races_test_data_with_predictions[races_test_data_with_predictions['trifecta_payout'].notnull()]

trifecta_test_data['trifecta_horse_combos'] = [get_trifecta_horse_combos(h) for h in tqdm(trifecta_test_data['horse_id_list'])]

trifecta_test_data['trifect_combo_probs'] = [get_trifecta_pred_probs(thc, hl, wp, sp, tp) for thc, hl, wp, sp, tp in tqdm(zip(trifecta_test_data['trifecta_horse_combos'], trifecta_test_data['horse_id_list'], trifecta_test_data['pred_winner'], trifecta_test_data['pred_second'], trifecta_test_data['pred_third']), total=len(trifecta_test_data))]

trifecta_test_data['winning_selection'] = [get_trifecta_winning_combo(hl, fp) for hl, fp in tqdm(zip(trifecta_test_data['horse_id_list'], trifecta_test_data['finish_position_for_ordering_list']), total=len(trifecta_test_data))]

trifecta_test_pred_df = [get_trifecta_pred_features_for_all_selections(tss, hl, wp, sp, tp, r, ws, p) for tss, hl, wp, sp, tp, r, ws, p in tqdm(zip(trifecta_test_data['trifecta_horse_combos'], trifecta_test_data['horse_id_list'], trifecta_test_data['pred_winner'], trifecta_test_data['pred_second'], trifecta_test_data['pred_third'], trifecta_test_data['runners_in_data'], trifecta_test_data['winning_selection'], trifecta_test_data['trifecta_payout']), total=len(trifecta_test_data))]
# need to free up some memory
del(horses_data)
del(trifecta_test_data)
trifecta_test_pred_df = [item for sublist in trifecta_test_pred_df for item in sublist]
trifecta_test_pred_df = [f for f in trifecta_test_pred_df if f!=[]]
trifecta_test_pred_df = pd.DataFrame(trifecta_test_pred_df, columns=['winner_win', 'winner_2nd', 'winner_3rd', 'second_win', 'second_2nd', 'second_3rd', 'third_win', 'third_2nd', 'third_3rd', 'trifecta_prob', 'runners', 'payout'])
trifecta_test_pred_df['payout'] = trifecta_test_pred_df['payout'].astype(float)
sys.getsizeof(trifecta_test_pred_df)/(1024*1024*1024)

#6951 took ages

min_runners = 7
max_runners = 7
trifecta_test_pred_df_subset = trifecta_test_pred_df[(trifecta_test_pred_df['runners']>=min_runners) & (trifecta_test_pred_df['runners']<=max_runners)]
len(trifecta_test_pred_df_subset)

# NOTE: WILL NEED TO ADD INFO LIKE RACE AND HORSE IDS TO DATA
trifecta_features = trifecta_train_data.columns[:-1]
trifecta_test_pred_X = trifecta_test_pred_df_subset[trifecta_features]
trifecta_test_pred_y = trifecta_test_pred_df_subset['payout']

dtrifecta_test = xgb.DMatrix(trifecta_test_pred_X, label=trifecta_test_pred_y)

trifecta_preds = model_trifecta.predict(dtrifecta_test)

trifecta_bet_cutoff = 1
sum(trifecta_preds>trifecta_bet_cutoff)
sum(trifecta_test_pred_y[trifecta_preds>trifecta_bet_cutoff])

# lin mod
trifecta_linmod_preds = linModFit.predict(sm.add_constant(trifecta_test_pred_X))
trifecta_bet_cutoff = 1.2
sum(trifecta_linmod_preds>trifecta_bet_cutoff)
sum(trifecta_test_pred_y[trifecta_linmod_preds>trifecta_bet_cutoff])







### test winner predictions with held out data
data_to_inspect = races_test_data_with_predictions[:100]
number_runners = 7
horse_number = 0
margin = 2
races_test_data_with_predictions_subset = races_test_data_with_predictions[races_test_data_with_predictions['runners_in_data']==number_runners]
horse_win_pred = [p[horse_number] for p in races_test_data_with_predictions_subset['pred_winner']]
horse_win_pred_odds = 1/np.array(horse_win_pred)
horse_odds = [o[horse_number] for o in races_test_data_with_predictions_subset['decimal_odds_list']]
horses_bet_on = horse_odds > horse_win_pred_odds*margin
horse_wins = [(f[horse_number]==1)*1 for f in races_test_data_with_predictions_subset['finish_position_for_ordering_list']]
sum(horses_bet_on)
sum(horses_bet_on*horse_odds*horse_wins)














'''
testing models
'''
# rearrange data to predict whether each horse will win or not, for races with given number of runners
select_runners_min = 7
select_runners_max = 9
rounding = 0
subset_data = races_data[races_data['runners_in_data'].isin(range(select_runners_min,select_runners_max+1))]
subset_data = subset_data[subset_data['top_3_check'] & subset_data['num_runners_check']]
if False:
    min_prs = 5
    max_prs_under_min = 1
    subset_data = subset_data[[sum(np.array(l)<min_prs) <= max_prs_under_min for l in subset_data['pred_prs_list']]]
len(subset_data)

order_training_inputs = True
if order_training_inputs:
    horse_pred_time_orders = [np.argsort(l) for l in subset_data['pred_time_list']]
else:
    horse_pred_time_orders = [list(range(len(l))) for l in subset_data['pred_time_list']]

def order_and_add_defaults(to_be_ordered, orders, default, length, rounding):
    ordered_list = list(np.around(np.array(to_be_ordered)[orders],rounding))
    defaults = [default]*(length - len(ordered_list))
    return ordered_list + defaults

pred_times_lists_ordered = np.array([order_and_add_defaults(tbo, o, 999999, select_runners_max, rounding) for tbo, o in zip(subset_data['pred_time_list'], horse_pred_time_orders)])

pred_vars_lists_ordered = np.array([order_and_add_defaults(tbo, o, 999999, select_runners_max, rounding) for tbo, o in zip(subset_data['pred_var_list'], horse_pred_time_orders)])

pred_prs_lists_ordered = np.array([order_and_add_defaults(tbo, o, 0, select_runners_max, rounding) for tbo, o in zip(subset_data['pred_prs_list'], horse_pred_time_orders)])

average_past_odds_lists_ordered = np.array([order_and_add_defaults(tbo, o, 999, select_runners_max, 2) for tbo, o in zip(subset_data['average_past_odds_list'], horse_pred_time_orders)])

number_past_wins_lists_ordered = np.array([order_and_add_defaults(tbo, o, 999, select_runners_max, 2) for tbo, o in zip(subset_data['number_past_wins_list'], horse_pred_time_orders)])

percentage_past_wins_lists_ordered = np.array([order_and_add_defaults(tbo, o, 999, select_runners_max, 2) for tbo, o in zip(subset_data['percentage_past_wins_list'], horse_pred_time_orders)])

finish_position_lists_ordered = np.array([order_and_add_defaults(tbo, o, 999, select_runners_max, rounding) for tbo, o in zip(subset_data['finish_position_for_ordering_list'], horse_pred_time_orders)])

horse_id_lists_ordered = np.array([order_and_add_defaults(tbo, o, -1, select_runners_max, rounding) for tbo, o in zip(subset_data['horse_id_list'], horse_pred_time_orders)])

odds_lists_ordered = np.array([order_and_add_defaults(tbo, o, 999, select_runners_max, 2) for tbo, o in zip(subset_data['decimal_odds_list'], horse_pred_time_orders)])



# multi-class model
col_names = ['race_id','yards']
pred_time_cols = ['cp_'+str(i)+'_pred_time' for i in range(0,select_runners_max)]
pred_var_cols = ['cp_'+str(i)+'_pred_var' for i in range(0,select_runners_max)]
pred_prs_cols = ['cp_'+str(i)+'_pred_prs' for i in range(0,select_runners_max)]
avg_odds_cols = ['cp_'+str(i)+'_average_past_odds' for i in range(0,select_runners_max)]
num_pwin_cols = ['cp_'+str(i)+'_number_past_wins' for i in range(0,select_runners_max)]
pct_pwin_cols = ['cp_'+str(i)+'_percentage_past_wins' for i in range(0,select_runners_max)]
fin_pos_cols = ['cp_'+str(i)+'_finish_position' for i in range(0,select_runners_max)]
horse_id_cols = ['cp_'+str(i)+'_horse_id' for i in range(0,select_runners_max)]
odds_cols = ['cp_'+str(i)+'_odds' for i in range(0,select_runners_max)]

df = subset_data[['race_id','yards','runners_in_data']].reset_index(drop=True)
df[pred_time_cols] = pd.DataFrame(pred_times_lists_ordered,columns=pred_time_cols)
df[pred_var_cols] = pd.DataFrame(pred_vars_lists_ordered,columns=pred_var_cols)
df[pred_prs_cols] = pd.DataFrame(pred_prs_lists_ordered,columns=pred_prs_cols)
df[avg_odds_cols] = pd.DataFrame(average_past_odds_lists_ordered,columns=avg_odds_cols)
df[num_pwin_cols] = pd.DataFrame(number_past_wins_lists_ordered ,columns=num_pwin_cols)
df[pct_pwin_cols] = pd.DataFrame(percentage_past_wins_lists_ordered ,columns=pct_pwin_cols)
df[fin_pos_cols] = pd.DataFrame(finish_position_lists_ordered,columns=fin_pos_cols)
df[horse_id_cols] = pd.DataFrame(horse_id_lists_ordered,columns=horse_id_cols)
df[odds_cols] = pd.DataFrame(odds_lists_ordered,columns=odds_cols)
df['winner'] = [list(x).index(1) for x in list(finish_position_lists_ordered)]
df['second'] = [list(x).index(2) for x in list(finish_position_lists_ordered)]
df['third'] = [list(x).index(3) for x in list(finish_position_lists_ordered)]

feature_y = ['winner']
features_x =  pred_time_cols + pred_var_cols + avg_odds_cols + pct_pwin_cols + num_pwin_cols + pred_prs_cols #+ odds_cols
features = feature_y + features_x
features_extra_info = ['race_id', 'yards', 'runners_in_data'] + fin_pos_cols + horse_id_cols + odds_cols
train_data_winner = df[df[features].isnull().sum(axis=1)==0]
race_ids = train_data_winner['race_id'].unique()
train_races, test_races = train_test_split(race_ids, test_size=0.2, random_state=12)
train_set = train_data_winner[train_data_winner['race_id'].isin(train_races)]
test_set = train_data_winner[train_data_winner['race_id'].isin(test_races)]
train_X = train_set[features_x]
train_y = train_set[feature_y]
train_extra_info = train_set[features_extra_info]
test_X = test_set[features_x]
test_y = test_set[feature_y]
test_extra_info = test_set[features_extra_info]


# create model # NOTE TO SELF: NEED TO ALTER THIS TO BE MORE HEAVILY FOCUSSED ON PRECISION (I.E. GETTING WINNERS RIGHT)
params = {
    'max_depth':2,
    'min_child_weight': 5,
    'eta':1, # set to 1 for rf
    'col_sample_by_tree':0.5, # set to < 1 for rf
    'subsample':0.5, # set to < 1 for rf
    'num_parallel_tree':30, # set to >1 for rf (remove for gbt)
    'objective':'multi:softprob',
    'eval_metric':'mlogloss',
    'num_class':select_runners_max
}
num_boost_round = 1 # set to 1 for rf
early_stopping = 1

dtrain = xgb.DMatrix(train_X, label=train_y)
dtest = xgb.DMatrix(test_X, label=test_y)

model_winner = xgb.train(params, dtrain, num_boost_round=num_boost_round,
                         early_stopping_rounds=early_stopping,
                         evals=[(dtest, "Test")])

preds_train = model_winner.predict(dtrain)
preds_test = model_winner.predict(dtest)


## softmax analysis
#sum(preds_train == np.array(train_y).flatten())/len(preds_train)
#sum(preds_test == np.array(test_y).flatten())/len(preds_test)
#max(preds_test)
#
#test_set_num_runners_min = 7
#test_set_num_runners_max = 7
#train_idx = train_extra_info['runners_in_data'].isin(range(test_set_num_runners_min, test_set_num_runners_max+1))
#test_idx = test_extra_info['runners_in_data'].isin(range(test_set_num_runners_min, test_set_num_runners_max+1))
#sum(preds_train[train_idx] == np.array(train_y[train_idx]).flatten())/len(preds_train[train_idx])
#sum(preds_test[test_idx] == np.array(test_y[test_idx]).flatten())/len(preds_test[test_idx])
#sum(0 == np.array(test_y[test_idx]).flatten())/len(preds_test[test_idx])
#
#feature_importances_dict = model_winner.get_score(importance_type='gain')
#feature_importances_df = pd.DataFrame({'feature':list(feature_importances_dict.keys()),
#                                       'importance':list(feature_importances_dict.values())})
#feature_importances_df = feature_importances_df.sort_values(by='importance', ascending=False)
#feature_importances_df['importance'] = feature_importances_df['importance']/sum(feature_importances_df['importance'])
#
#
#min_prs_cutoff = 3
#
#sum(preds_test[(test_X[pred_prs_cols]<min_prs_cutoff).sum(axis=1)==0] == np.array(test_y).flatten()[(test_X[pred_prs_cols]<min_prs_cutoff).sum(axis=1)==0])
#sum((test_X[pred_prs_cols]<min_prs_cutoff).sum(axis=1)==0)
#
#
## test betting on predicted winner
#subset_test_data = subset_data[subset_data['race_id'].isin(test_races)]
#subset_test_data['winner_idx'] = [fpl.index(1) for fpl in subset_test_data['finish_position_for_ordering_list']]
#subset_test_data['winner_odds'] = [dol[w] for dol, w in zip(subset_test_data['decimal_odds_list'], subset_test_data['winner_idx'])]
#subset_test_data['win_preds'] = (preds_test == np.array(test_y).flatten())
#sum(subset_test_data['winner_odds']*subset_test_data['win_preds'])/len(subset_test_data)
#
#data_to_inspect = subset_data[:100]



# softprob analysis
preds_test_df = pd.DataFrame(preds_test, index=test_X.index)
preds_test_df = 1/preds_test_df

horse_no = 1
margin = 1
bets_placed_on = test_extra_info['cp_'+str(horse_no)+'_odds']>(preds_test_df.iloc[:,horse_no])*margin
winning_odds = test_extra_info['cp_'+str(horse_no)+'_odds']*(np.array(test_y).flatten()==horse_no)
sum(bets_placed_on)
sum(bets_placed_on*winning_odds)

sum(np.array(test_y).flatten()==0)







# do some param testing
gridsearch_params = [
    (max_depth, min_child_weight, colsample_bytree, subsample)
    for max_depth in range(1,6)
    for min_child_weight in range(1,6)
    for colsample_bytree in [0.5, 0.8, 1]
    for subsample in [0.5, 0.8, 1]
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
                        seed=123, nfold=5, metrics={'merror'}
                        )
    # Update best MAE
    mean_rmse = cv_results['test-merror-mean'].min()
    boost_rounds = cv_results['test-merror-mean'].argmin()
    print("\tRMSE {} for {} rounds".format(mean_rmse, boost_rounds))
    if mean_rmse < min_rmse:
        min_rmse = mean_rmse
        best_params = (max_depth,min_child_weight,colsample_bytree,subsample)
print("Best params: {}, {}, RMSE: {}".format(best_params[0], best_params[1], min_rmse))

best_params
















# winner model per individual horse
select_runners = 8
subset_data = races_data[races_data['runners_in_data']==select_runners]
subset_data = subset_data[subset_data['top_3_check'] & subset_data['num_runners_check']]
horse_pred_time_orders = [np.argsort(l) for l in subset_data['pred_time_list']]

pred_times_lists = np.array(list(subset_data['pred_time_list']))
pred_times_lists_ordered = np.array([l[o] for l, o in zip(pred_times_lists, horse_pred_time_orders)])

pred_vars_lists = np.array(list(subset_data['pred_var_list']))
pred_vars_lists_ordered = np.array([l[o] for l, o in zip(pred_vars_lists, horse_pred_time_orders)])

pred_prs_lists = np.array(list(subset_data['pred_prs_list']))
pred_prs_lists_ordered = np.array([l[o] for l, o in zip(pred_prs_lists, horse_pred_time_orders)])

finish_position_lists = np.array(list(subset_data['finish_position_for_ordering_list']))
finish_position_lists_ordered = np.array([l[o] for l, o in zip(finish_position_lists, horse_pred_time_orders)])

horse_id_lists = np.array(list(subset_data['horse_id_list']))
horse_id_lists_ordered = np.array([l[o] for l, o in zip(horse_id_lists, horse_pred_time_orders)])

odds_lists = np.array(list(subset_data['decimal_odds_list']))
odds_lists_ordered = np.array([l[o] for l, o in zip(odds_lists, horse_pred_time_orders)])


col_names = ['race_id','yards','horse_id','pred_time','pred_var','pred_prs','finish_position']
cp_pred_time_cols = ['cp_'+str(i)+'_pred_time' for i in range(1,select_runners)]
cp_pred_var_cols = ['cp_'+str(i)+'_pred_var' for i in range(1,select_runners)]
cp_pred_prs_cols = ['cp_'+str(i)+'_pred_prs' for i in range(1,select_runners)]
train_data_win_preds = pd.DataFrame(columns=col_names+cp_pred_time_cols+cp_pred_var_cols+cp_pred_prs_cols)

for i in range(select_runners):
    df = subset_data[['race_id','yards']].reset_index(drop=True)
    df['horse_id'] = horse_id_lists_ordered[:,i]
    df['pred_time'] = pred_times_lists_ordered[:,i]
    df['pred_var'] = pred_vars_lists_ordered[:,i]
    df['pred_prs'] = pred_prs_lists_ordered[:,i]
    df['finish_position'] = finish_position_lists_ordered[:,i]
    df[cp_pred_time_cols] = pd.DataFrame(np.delete(pred_times_lists_ordered, i, axis=1),columns=cp_pred_time_cols)
    df[cp_pred_var_cols] = pd.DataFrame(np.delete(pred_vars_lists_ordered, i, axis=1),columns=cp_pred_var_cols)
    df[cp_pred_prs_cols] = pd.DataFrame(np.delete(pred_prs_lists_ordered, i, axis=1),columns=cp_pred_prs_cols)
    
    train_data_win_preds = train_data_win_preds.append(df)

train_data_win_preds['winner'] = (train_data_win_preds['finish_position']==1)*1
train_data_win_preds['second'] = (train_data_win_preds['finish_position']==2)*1
train_data_win_preds['third'] = (train_data_win_preds['finish_position']==3)*1

sum(train_data_win_preds['winner'])/len(train_data_win_preds)
sum(train_data_win_preds['second'])/len(train_data_win_preds)
sum(train_data_win_preds['third'])/len(train_data_win_preds)

train_data_win_preds[['pred_prs'] + cp_pred_prs_cols] = train_data_win_preds[['pred_prs'] + cp_pred_prs_cols].astype(int)

feature_y = ['winner']
features_x = ['pred_time'] + cp_pred_time_cols#,'pred_var','pred_prs'] + cp_pred_time_cols + cp_pred_var_cols + cp_pred_prs_cols
features = feature_y + features_x
train_data_winner = train_data_win_preds[train_data_win_preds[features].isnull().sum(axis=1)==0]
train_split, test_split = train_test_split(train_data_winner, test_size=0.2)
train_X = train_split[features_x]
train_y = train_split[feature_y]
test_X = test_split[features_x]
test_y = test_split[feature_y]

# create model # NOTE TO SELF: NEED TO ALTER THIS TO BE MORE HEAVILY FOCUSSED ON PRECISION (I.E. GETTING WINNERS RIGHT)
params = {
    'max_depth':2,
    'min_child_weight': 5,
    'eta':0.5,
    'objective':'binary:logistic',
    'eval_metric':'logloss'
}
num_boost_round = 1000
early_stopping = 100

dtrain = xgb.DMatrix(train_X, label=train_y)
dtest = xgb.DMatrix(test_X, label=test_y)

model_winner = xgb.train(params, dtrain, num_boost_round=num_boost_round,
                         early_stopping_rounds=early_stopping,
                         evals=[(dtest, "Test")])

preds_train = model_winner.predict(dtrain)
preds_test = model_winner.predict(dtest)

sum((preds_test>0.1) & (np.array(test_y).flatten()==1))/sum(preds_test>0.1)

data_to_inspect = train_data_win_preds[:100]








'''
testing code
'''
train_data=train_data
feature_y=features_y
feature_cur=features_current_race
feature_pr=features_pr
number_past_results=10

features_prs = []
if number_past_results>0:
    for i in range(1, number_past_results+1):
        features_prs = features_prs + ['pr_'+str(i)+'_'+pr for pr in features_pr]

features = feature_y + feature_cur + features_prs
train_data = train_data[train_data[features].isnull().sum(axis=1)==0]
train_split, test_split = train_test_split(train_data, test_size=0.1)
train_X = train_split[feature_cur + features_prs]
train_y = train_split[feature_y]
test_X = test_split[feature_cur + features_prs]
test_y = test_split.loc[:,feature_y]

# create model
params = {
    'max_depth':3,
    'min_child_weight': 5,
    'eta':0.5,
    'objective':'reg:linear',
    'eval_metric':'rmse'
}
num_boost_round = 1000
early_stopping = 10

dtrain = xgb.DMatrix(train_X, label=train_y)
dtest = xgb.DMatrix(test_X, label=test_y)

model = xgb.train(params, dtrain, num_boost_round=num_boost_round,
                  early_stopping_rounds=early_stopping,
                  evals=[(dtest, "Test")])

# train variance model
preds_train = model.predict(dtrain)
preds_test = model.predict(dtest)
train_y_var = (pd.Series(preds_train) - train_y_s)**2
test_y_var = (preds_test - test_y_s)**2
train_y_s = np.array(train_y).flatten()
params_var = {
    'max_depth':5,
    'min_child_weight': 5,
    'eta':.01,
    'objective':'reg:linear',
    'eval_metric':'mae'
}

dtrain_var = xgb.DMatrix(train_X, label=train_y_var)
dtest_var = xgb.DMatrix(test_X, label=test_y_var)
model_var = xgb.train(params_var, dtrain_var, num_boost_round=num_boost_round,
                      early_stopping_rounds=early_stopping,
                      evals=[(dtest_var, "Test")])
    







data=train_data.copy()
feature_y=features_y
feature_cur=features_current_race
feature_pr=features_pr
number_past_results=9
horse_time_model=pr09Mod
variance_model=pr09Mod_var
overwrite_preds=False

features_prs = []
if number_past_results>0:
    for i in range(1, number_past_results+1):
        features_prs = features_prs + ['pr_'+str(i)+'_'+pr for pr in features_pr]

features = feature_cur + features_prs

pred_data = data.copy()
if not overwrite_preds:
    if 'pred_time' in pred_data.columns:
        pred_data = pred_data[pred_data['pred_time'].isnull()]
    if 'pred_var' in data.columns:
        pred_data = pred_data[pred_data['pred_var'].isnull()]

pred_data = pred_data[pred_data[features].isnull().sum(axis=1)==0]
pred_data = pred_data[features]
dpred = xgb.DMatrix(pred_data)

pred_times = horse_time_model.predict(dpred)
pred_vars = horse_time_model.predict(dpred)
pred_data['pred_time'] = pred_times
pred_data['pred_var'] = pred_vars

if 'pred_time' not in data.columns:
    data['pred_time'] = None
if 'pred_var' not in data.columns:
    data['pred_var'] = None

data.loc[pred_data.index,'pred_time'] = pred_data['pred_time']
data.loc[pred_data.index,'pred_var'] = pred_data['pred_var']
max(pred_data.index)
sum(data['pred_time'].isnull())


