use betfair;

#drop table football_market_definitions_EXTRA;
#drop table football_runners_EXTRA;
#drop table football_runner_changes_EXTRA;

###################
### Review data ###
###################
# and compare to UK data
select * from football_market_definitions_EXTRA limit 10;
select count(1) from football_market_definitions_EXTRA;
select COUnt(1), market_type from football_market_definitions_EXTRA group by market_type;
select count(distinct event_id) from football_market_definitions_EXTRA;
select count(distinct event_id) from football_market_definitions;

select * from football_runners_EXTRA limit 1000;
# NOTE: Not all correct scores have the same options
select count(1) from football_runners;
select count(1) from football_runners_EXTRA;

select * from football_runner_changes_EXTRA limit 10;
select count(1) from football_runner_changes;
select count(1) from football_runner_changes_EXTRA;


###################
### add indices ###
###################
ALTER TABLE football_market_definitions_EXTRA
MODIFY COLUMN market_id varchar(255);
CREATE INDEX market_id ON football_market_definitions_EXTRA(market_id);

ALTER TABLE football_runner_changes_EXTRA
MODIFY COLUMN market_id varchar(255);
CREATE INDEX market_id ON football_runner_changes_EXTRA(market_id);

ALTER TABLE football_runners_EXTRA
MODIFY COLUMN market_id varchar(255);
CREATE INDEX market_id ON football_runners_EXTRA(market_id);

SHOW FULL PROCESSLIST;
#####################################
### create some additional tables ###
#####################################

# starting price times (in order to be able to select historic odds that are pre-bsp)
drop table if exists football_market_pre_inplay_time_EXTRA;
create table football_market_pre_inplay_time_EXTRA as (
	with football_market_pre_inplay as (
		select m.*, row_number() over (partition by market_id order by datetime desc) as rn
        from (select * from football_market_definitions_EXTRA where in_play = 0) m
        )
	select * from football_market_pre_inplay where rn = 1
);
ALTER TABLE football_market_pre_inplay_time_EXTRA
MODIFY COLUMN market_id varchar(255);
CREATE INDEX market_id ON football_market_pre_inplay_time_EXTRA(market_id);

# check all exist
select count(distinct market_id) from football_market_definitions_EXTRA;
select count(distinct market_id) from football_market_pre_inplay_time_EXTRA;
# almost all exist - likely all significant events exist


# table with one line for every single market
drop table if exists football_markets_EXTRA;
create table football_markets_tmp as (
	select min(venue) as venue, min(event_name) as event_name, min(event_id) as event_id_min, max(event_id) as event_id_max,
			country_code, timezone, event_type_id, market_type, market_id, betting_type,
            min(number_of_winners) as number_of_winners_min, max(open_date) as open_date_max, 
            max(market_time) as market_time_max, max(suspend_time) as suspend_time_max, max(settled_time) as settled_time_max,
            SUBSTRING_INDEX(event_name,' v ',1) as home, SUBSTRING_INDEX(event_name,' v ',-1) as away
    from football_market_definitions_EXTRA
    group by country_code, timezone, event_type_id, market_type, market_id, betting_type, home, away
);

CREATE TABLE football_markets_EXTRA AS (
	WITH ordered_markets AS (
		SELECT *, row_number() over
					(partition by market_id order by country_code asc, timezone DESC, market_type DESC) as rn
		FROM football_markets_tmp)
	SELECT * FROM ordered_markets WHERE rn=1
);
DROP TABLE football_markets_tmp;

select count(distinct market_id) from football_market_definitions_EXTRA;
select count(distinct market_id) from football_markets_EXTRA;
select count(1) from football_markets_EXTRA;

ALTER TABLE football_markets_EXTRA
MODIFY COLUMN market_id varchar(255);
CREATE INDEX market_id ON football_markets_EXTRA(market_id);


# create market mapping
# NOTE: Not sure this is needed for football if can just join on event_ids

# get last runner_changes before every x mins pre event
select * from football_markets_EXTRA limit 10;
select * from football_runner_changes_EXTRA limit 10;
drop table if exists football_runner_changes_with_start_time_EXTRA;
create table football_runner_changes_with_start_time_EXTRA as (
	select r.runner_id, r.ltp, r.market_id, r.datetime, m.market_time_max,
			TIMESTAMPDIFF(MINUTE, r.datetime, cast(REPLACE(REPLACE(m.market_time_max, 'Z', ''), 'T', ' ') as datetime)) as time_to_event
    from football_runner_changes_EXTRA r
    left join football_markets_EXTRA m on r.market_id = m.market_id
    
);
select * from football_runner_changes_with_start_time_EXTRA limit 10;
select count(1) from football_runner_changes_EXTRA;
select count(1) from football_runner_changes_with_start_time_EXTRA;

CREATE INDEX runner_id ON football_runner_changes_with_start_time_EXTRA(runner_id);
CREATE INDEX market_id ON football_runner_changes_with_start_time_EXTRA(market_id);
CREATE INDEX time_to_event ON football_runner_changes_with_start_time_EXTRA(time_to_event);
CREATE INDEX datetime ON football_runner_changes_with_start_time_EXTRA(datetime);

drop table if exists football_runner_changes_1m_before_EXTRA;
create table football_runner_changes_1m_before_EXTRA as (
	with ranked_changes as (
		select *, ROW_NUMBER() OVER (PARTITION BY runner_id, market_id ORDER BY datetime DESC) AS rn
        from football_runner_changes_with_start_time_EXTRA where time_to_event >= 1
        )
	select * from ranked_changes where rn=1
);

drop table if exists football_runner_changes_60m_before_EXTRA;
create table football_runner_changes_60m_before_EXTRA as (
	with ranked_changes as (
		select *, ROW_NUMBER() OVER (PARTITION BY runner_id, market_id ORDER BY datetime DESC) AS rn
        from football_runner_changes_with_start_time_EXTRA where time_to_event >= 60
        )
	select * from ranked_changes where rn=1
);

drop table if exists football_runner_changes_120m_before_EXTRA;
create table football_runner_changes_120m_before_EXTRA as (
	with ranked_changes as (
		select *, ROW_NUMBER() OVER (PARTITION BY runner_id, market_id ORDER BY datetime DESC) AS rn
        from football_runner_changes_with_start_time_EXTRA where time_to_event >= 120
        )
	select * from ranked_changes where rn=1
);

drop table if exists football_runner_changes_180m_before_EXTRA;
create table football_runner_changes_180m_before_EXTRA as (
	with ranked_changes as (
		select *, ROW_NUMBER() OVER (PARTITION BY runner_id, market_id ORDER BY datetime DESC) AS rn
        from football_runner_changes_with_start_time_EXTRA where time_to_event >= 180
        )
	select * from ranked_changes where rn=1
);

drop table if exists football_runner_changes_360m_before_EXTRA;
create table football_runner_changes_360m_before_EXTRA as (
	with ranked_changes as (
		select *, ROW_NUMBER() OVER (PARTITION BY runner_id, market_id ORDER BY datetime DESC) AS rn
        from football_runner_changes_with_start_time_EXTRA where time_to_event >= 360
        )
	select * from ranked_changes where rn=1
);

drop table if exists football_runner_changes_720m_before_EXTRA;
create table football_runner_changes_720m_before_EXTRA as (
	with ranked_changes as (
		select *, ROW_NUMBER() OVER (PARTITION BY runner_id, market_id ORDER BY datetime DESC) AS rn
        from football_runner_changes_with_start_time_EXTRA where time_to_event >= 720
        )
	select * from ranked_changes where rn=1
);

drop table if exists football_runner_changes_1080m_before_EXTRA;
create table football_runner_changes_1080m_before_EXTRA as (
	with ranked_changes as (
		select *, ROW_NUMBER() OVER (PARTITION BY runner_id, market_id ORDER BY datetime DESC) AS rn
        from football_runner_changes_with_start_time_EXTRA where time_to_event >= 1080
        )
	select * from ranked_changes where rn=1
);

drop table if exists football_runner_changes_1440m_before_EXTRA;
create table football_runner_changes_1440m_before_EXTRA as (
	with ranked_changes as (
		select *, ROW_NUMBER() OVER (PARTITION BY runner_id, market_id ORDER BY datetime DESC) AS rn
        from football_runner_changes_with_start_time_EXTRA where time_to_event >= 1440
        )
	select * from ranked_changes where rn=1
);

drop table if exists football_runner_changes_xm_before_EXTRA;
create table football_runner_changes_xm_before_EXTRA as (
	select * from football_runner_changes_1m_before_EXTRA
    union
	select * from football_runner_changes_60m_before_EXTRA
    union
    select * from football_runner_changes_120m_before_EXTRA
    union
    select * from football_runner_changes_180m_before_EXTRA
    union
    select * from football_runner_changes_360m_before_EXTRA
    union
    select * from football_runner_changes_720m_before_EXTRA
    union
    select * from football_runner_changes_1080m_before_EXTRA
    union
    select * from football_runner_changes_1440m_before_EXTRA
);

select count(1) from football_runner_changes_60m_before_EXTRA;
select count(distinct runner_id, market_id) from football_runner_changes_EXTRA;

SHOW FULL PROCESSLIST;



# get final outcomes with consistent names
select count(distinct runner_id, market_id) from football_runners_EXTRA;
select count(distinct runner_id, market_id) from football_runners_EXTRA where status in ('WINNER', 'LOSER', 'REMOVED');
select * from football_markets_EXTRA where event_id_min <> event_id_max limit 100;

drop table if exists football_runner_outcomes_EXTRA;
create table football_runner_outcomes_EXTRA as (
	with football_runners_last_entry as (
		select * from (
			select *, ROW_NUMBER() OVER (PARTITION BY runner_id, market_id ORDER BY datetime DESC) AS rn
			from football_runners_EXTRA) t
		where rn = 1
	)
	select r.*, m.event_name, m.event_id_min as event_id, m.home, m.away, m.country_code, m.timezone, m.market_type,
			m.betting_type, m.number_of_winners_min as number_of_winners,
			m.open_date_max as open_date, m.market_time_max as market_time,
			m.suspend_time_max as suspend_time, m.settled_time_max as settled_time,
            replace(replace(replace(replace(runner_name, home, 'Home'), away, 'Away'), ' - ','-'), '-', ' - ') as runner_name_general
	from football_runners_last_entry r
	left join football_markets_EXTRA m on r.market_id = m.market_id
);

SHOW FULL PROCESSLIST;

select runner_name_general, COUNT(1) as count from football_runner_outcomes_EXTRA group by runner_name_general order by count desc;

####################
### data queries ###
####################

select count(1)
from football_runner_outcomes_EXTRA o 
left join football_runner_changes_1m_before_EXTRA c
on o.runner_id = c.runner_id and o.market_id = c.market_id
#where market_type in ('CORRECT_SCORE', 'MATCH_ODDS', 'OVER_UNDER_05', 'OVER_UNDER_15',
#						'OVER_UNDER_25', 'OVER_UNDER_35', 'OVER_UNDER_45')
limit 10;

SHOW FULL PROCESSLIST;
select count(1) from football_runner_changes_60m_before limit 10;
select count(1) from football_runner_changes_1m_before limit 10;
select count(1) from football_runner_outcomes;

select count(distinct runner_id, market_id) from football_runner_changes;
select count(distinct runner_id, market_id) from football_runner_outcomes;

###############
### TESTING ###
###############

select * from football_market_definitions where event_id = 29641970;
select * from football_runner_changes where market_id = 1.167066927;
select * from football_runners where market_id = 1.167066927;
select * from football_market_definitions where market_id = 1.167066927;
select * from football_runner_changes_with_start_time where market_id = 1.167066927;

select * from football_runners where market_id = 1.160967411;

select * from
	(
	select b.*, o.market_start_time, o.api_call_time_utc, o.minutes_to_event,
			o.event_name, o.competition_name, o.market_name, o.market_type, o.country_code,
            o.timezone, o.total_matched_market, o.total_available, o.runner_name, o.ltp,
            o.back_price_1, o.back_price_2, o.back_price_3, o.back_size_1, o.back_size_2, o.back_size_3,
            o.lay_price_1, o.lay_price_2, o.lay_price_3, o.lay_size_1, o.lay_size_2, o.lay_size_3,
            o.runner_name_general, o.market_runner, o.input_odds, o.pred, o.pred_odds, o.bet,
            o.correct_score_overround, o.match_odds_overround, o.over_under_overround,
            ROW_NUMBER() OVER (PARTITION BY b.event_id, b.market_id, b.selection_id ORDER BY api_call_time_utc DESC) AS rn
	from football_bet_outcomes_live b
	left join football_output_live o on b.event_id = o.event_id
									and b.market_id = o.market_id
									and b.selection_id = o.runner_id
									and o.api_call_time_utc <= b.placed_date
	) t
where rn = 1
limit 10;

select * from football_order_results_live o
left join football_bet_outcomes_live b
on o.bet_id = b.bet_id and o.market_id = b.market_id and o.selection_id = b.selection_id
where b.bet_id is null
limit 10;

use betfair;
select * from football_order_results_live where placed_date > '2021-08-07' limit 100;

select * from football_bet_outcomes_live limit 10;
select * from football_order_results_live limit 10;
select * from football_output_live limit 10;
select * from football_predictions_live limit 10;

select * from football_output_live
where back_price_1/pred_odds >= 1.25
and bet = 1
and api_call_time_utc > '2021-08-05'
limit 100
;

select case when back_price_1/pred_odds < 1.25 then 'Under 1.25' else 'Over 1.25' end as odds_ratio, count(1)
from football_output_live
where bet = 1
and api_call_time_utc > '2021-08-05'
#and market_start_time > '2021-08-05'
group by 1
;

SELECT table_schema "DB Name",
        ROUND(SUM(data_length + index_length) / 1024 / 1024 / 1024, 1) "DB Size in GB" 
FROM information_schema.tables 
GROUP BY table_schema;

SELECT 
     table_schema as `Database`, 
     table_name AS `Table`, 
     round(((data_length + index_length) / 1024 / 1024 / 1024), 2) `Size in GB` 
FROM information_schema.TABLES 
WHERE table_schema = 'betfair'
AND table_name like '%_live%'
ORDER BY (data_length + index_length) DESC;

select *
from football_order_results_live
#'30850663'
limit 100
;

select *
from football_predictions_live
where event_name like '%Liverpool%'
and event_name like '%Palace%'
;

select *
from football_output_live
where event_id = '30850663'
limit 10
;

select *
from football_bet_outcomes_live
limit 10;

SHOW FULL PROCESSLIST;