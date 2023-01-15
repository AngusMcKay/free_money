use sl_bf_late_goals;

select * from testing_events_df;

select * from testing_matches_at_prediction_times;

select * from testing_matches_pre_prediction_times;

select * from testing_matches_df;

select * from testing_model_data order by datetime_utc desc;

select team_a_score, team_b_score, actual_odds_over_back_1, actual_odds_over_lay_1
from testing_viable_matches where next_prediction_time = 80;


select min(minutes_to_following_prediction_time) from testing_matches_at_prediction_times;

select min(minutes_to_next_prediction_time) from testing_matches_pre_prediction_times;


select count(1) from testing_events_df;

select count(1) from testing_matches_at_prediction_times;

select count(1) from testing_matches_pre_prediction_times;

select count(1) from testing_matches_df;

select count(1) from testing_model_data;

select count(1) from testing_viable_matches;

select count(1) from testing3_viable_matches_with_unmatched_goals;


select total_available, count(1) from testing3_model_data_with_preds group by 1 order by 2 desc limit 100;

select * from testing3_model_data_with_preds where total_available = 7351.86;

select min(datetime_utc) from testing_viable_matches;

select * from testing_live_order_fails;

select * from testing_live_order_results;

SELECT * FROM testing_live_model_data_with_preds where team_a_name like '%Westerlo%';

select * from testing_live_model_data_with_preds where market_id in ('1.208431813', '1.208431854', '1.208431858') and action <> 'None';

select 
	o.error_code, o.market_id, selection_id, action, price, average_price_matched, actual_odds_over_back_1, actual_odds_over_back_3, actual_odds_under_back_1, actual_odds_under_back_3,
    placed_date, size_matched, order_status, o.datetime_utc,
    team_a_name, team_b_name, team_a_score, team_b_score, clock, next_prediction_time
from testing_live_order_results o
left join testing_live_model_data_with_preds m
on o.market_id = m.market_id and o.datetime_utc = m.datetime_utc;

select delay_time, count(1) from testing_live_model_data_with_preds group by 1;

