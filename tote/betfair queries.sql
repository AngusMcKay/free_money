use betfair;

###################
### Review data ###
###################
select * from market_definitions limit 10;
select count(1) from market_definitions;

select * from runners limit 1000;
select count(1) from runners;
select * from runners where market_id = '1.130081731' limit 1000;

select * from runner_changes limit 10;
select count(1) from runner_changes;

###################
### add indices ###
###################
ALTER TABLE market_definitions
MODIFY COLUMN market_id varchar(255);
CREATE INDEX market_id ON market_definitions(market_id);

ALTER TABLE runner_changes
MODIFY COLUMN market_id varchar(255);
CREATE INDEX market_id ON runner_changes(market_id);

ALTER TABLE runners
MODIFY COLUMN market_id varchar(255);
CREATE INDEX market_id ON runners(market_id);

#####################################
### create some additional tables ###
#####################################

# starting price times (in order to be able to select historic odds that are pre-bsp)
drop table if exists market_bsp_times;
create table market_bsp_times as (
	with market_ordered as (
		select m.*, row_number() over (partition by market_id order by datetime asc) as rn
		from (select * from market_definitions where bsp_reconciled = 1) m
        )
	select * from market_ordered where rn = 1
);
ALTER TABLE market_bsp_times
MODIFY COLUMN market_id varchar(255);
CREATE INDEX market_id ON market_bsp_times(market_id);

# check all exist
select count(distinct market_id) from market_definitions;
select count(distinct market_id) from market_bsp_times;
# they don't all exist


# table with one line per race
drop table if exists markets;
create table markets_tmp as (
	select min(venue) as venue, min(event_name) as event_name, min(event_id) as event_id_min, max(event_id) as event_id_max,
			country_code, timezone, event_type_id, market_type, market_id, betting_type,
            min(number_of_winners) as number_of_winners_min, max(open_date) as open_date_max, 
            max(market_time) as market_time_max, max(suspend_time) as suspend_time_max, max(settled_time) as settled_time_max
    from market_definitions
    group by country_code, timezone, event_type_id, market_type, market_id, betting_type
);

CREATE TABLE markets AS (
	WITH ordered_markets AS (
		SELECT *, row_number() over
					(partition by market_id order by country_code asc, timezone DESC, market_type DESC) as rn
		FROM markets_tmp)
	SELECT * FROM ordered_markets WHERE rn=1
);

select count(distinct market_id) from market_definitions;
select count(distinct market_id) from markets;
select count(1) from markets;

ALTER TABLE markets
MODIFY COLUMN market_id varchar(255);
CREATE INDEX market_id ON markets(market_id);
select * from markets where market_id in (select market_id from markets group by market_id having count(1) > 1) limit 1000;



# combine win and place markets
drop table if exists win_place_map;
create table win_place_map as (
	(
    select case when w.venue is null then p.venue else w.venue end as venue,
			case when w.event_name is null then p.event_name else w.event_name end as event_name,
			case when w.country_code is null then p.country_code else w.country_code end as country_code,
			w.market_id as market_id_win, p.market_id as market_id_place
	from (select * from markets where market_type = 'WIN') w
	left join (select * from markets where market_type = 'PLACE') p
	on (w.event_id_min = p.event_id_min or w.event_id_max = p.event_id_max) and w.market_time_max = p.market_time_max
    )
    
    UNION
    
    (
    select case when w.venue is null then p.venue else w.venue end as venue,
			case when w.event_name is null then p.event_name else w.event_name end as event_name,
			case when w.country_code is null then p.country_code else w.country_code end as country_code,
			w.market_id as market_id_win, p.market_id as market_id_place
	from (select * from markets where market_type = 'WIN') w
	right join (select * from markets where market_type = 'PLACE') p
	on (w.event_id_min = p.event_id_min or w.event_id_max = p.event_id_max) and w.market_time_max = p.market_time_max
    )
);
select * from win_place_map limit 10;
select count(distinct market_id_win) from win_place_map;
select count(distinct market_id) from markets where market_type = 'WIN';
select count(distinct market_id) from market_definitions where market_type = 'WIN';
select count(distinct market_id_place) from win_place_map;
select count(distinct market_id) from markets where market_type = 'PLACE';
select count(distinct market_id) from market_definitions where market_type = 'PLACE';


# get last runner_changes before every 30 mins pre event
select * from markets limit 10;
select * from runner_changes limit 10;
drop table runner_changes_with_start_time;
create table runner_changes_with_start_time as (
	select r.runner_id, r.ltp, r.market_id, r.datetime, m.market_time_max,
			TIMESTAMPDIFF(MINUTE, r.datetime, cast(REPLACE(REPLACE(m.market_time_max, 'Z', ''), 'T', ' ') as datetime)) as time_to_event
    from runner_changes r
    left join markets m on r.market_id = m.market_id
    
);
select * from runner_changes_with_start_time limit 10;
select count(1) from runner_changes;
select count(1) from runner_changes_with_start_time;

CREATE INDEX runner_id ON runner_changes_with_start_time(runner_id);
CREATE INDEX market_id ON runner_changes_with_start_time(market_id);
CREATE INDEX time_to_event ON runner_changes_with_start_time(time_to_event);
CREATE INDEX datetime ON runner_changes_with_start_time(datetime);

drop table runner_changes_30m_before;
create table runner_changes_30m_before as (
	with ranked_changes as (
		select *, ROW_NUMBER() OVER (PARTITION BY runner_id, market_id ORDER BY datetime DESC) AS rn
        from runner_changes_with_start_time where time_to_event >= 30
        )
	select * from ranked_changes where rn=1
);

drop table runner_changes_60m_before;
create table runner_changes_60m_before as (
	with ranked_changes as (
		select *, ROW_NUMBER() OVER (PARTITION BY runner_id, market_id ORDER BY datetime DESC) AS rn
        from runner_changes_with_start_time where time_to_event >= 60
        )
	select * from ranked_changes where rn=1
);

drop table runner_changes_90m_before;
create table runner_changes_90m_before as (
	with ranked_changes as (
		select *, ROW_NUMBER() OVER (PARTITION BY runner_id, market_id ORDER BY datetime DESC) AS rn
        from runner_changes_with_start_time where time_to_event >= 90
        )
	select * from ranked_changes where rn=1
);

drop table runner_changes_120m_before;
create table runner_changes_120m_before as (
	with ranked_changes as (
		select *, ROW_NUMBER() OVER (PARTITION BY runner_id, market_id ORDER BY datetime DESC) AS rn
        from runner_changes_with_start_time where time_to_event >= 120
        )
	select * from ranked_changes where rn=1
);

drop table runner_changes_150m_before;
create table runner_changes_150m_before as (
	with ranked_changes as (
		select *, ROW_NUMBER() OVER (PARTITION BY runner_id, market_id ORDER BY datetime DESC) AS rn
        from runner_changes_with_start_time where time_to_event >= 150
        )
	select * from ranked_changes where rn=1
);

drop table runner_changes_180m_before;
create table runner_changes_180m_before as (
	with ranked_changes as (
		select *, ROW_NUMBER() OVER (PARTITION BY runner_id, market_id ORDER BY datetime DESC) AS rn
        from runner_changes_with_start_time where time_to_event >= 180
        )
	select * from ranked_changes where rn=1
);

drop table runner_changes_210m_before;
create table runner_changes_210m_before as (
	with ranked_changes as (
		select *, ROW_NUMBER() OVER (PARTITION BY runner_id, market_id ORDER BY datetime DESC) AS rn
        from runner_changes_with_start_time where time_to_event >= 210
        )
	select * from ranked_changes where rn=1
);

drop table runner_changes_240m_before;
create table runner_changes_240m_before as (
	with ranked_changes as (
		select *, ROW_NUMBER() OVER (PARTITION BY runner_id, market_id ORDER BY datetime DESC) AS rn
        from runner_changes_with_start_time where time_to_event >= 240
        )
	select * from ranked_changes where rn=1
);

drop table runner_changes_270m_before;
create table runner_changes_270m_before as (
	with ranked_changes as (
		select *, ROW_NUMBER() OVER (PARTITION BY runner_id, market_id ORDER BY datetime DESC) AS rn
        from runner_changes_with_start_time where time_to_event >= 270
        )
	select * from ranked_changes where rn=1
);

drop table runner_changes_300m_before;
create table runner_changes_300m_before as (
	with ranked_changes as (
		select *, ROW_NUMBER() OVER (PARTITION BY runner_id, market_id ORDER BY datetime DESC) AS rn
        from runner_changes_with_start_time where time_to_event >= 300
        )
	select * from ranked_changes where rn=1
);

drop table runner_changes_xm_before;
create table runner_changes_xm_before as (
	select * from runner_changes_30m_before
    union
    select * from runner_changes_60m_before
    union
    select * from runner_changes_90m_before
    union
    select * from runner_changes_120m_before
    union
    select * from runner_changes_150m_before
    union
    select * from runner_changes_180m_before
    union
    select * from runner_changes_210m_before
    union
    select * from runner_changes_240m_before
    union
    select * from runner_changes_270m_before
    union
    select * from runner_changes_300m_before
);

select count(1) from runner_changes_xm_before;
select count(distinct runner_id, market_id) from runner_changes;
select * from runner_changes_xm_before where runner_id = 163 limit 1000;


SELECT * FROM runner_changes_with_start_time where market_id = 1.130081731;;
select * from market_definitions where market_id = 1.130025335;
select * from runner_changes where market_id = 1.130025335;
select * from runners where market_id = 1.130025335;
SHOW FULL PROCESSLIST;

select * from market_definitions where datetime between '2019-12-31' and '2020-02-29' limit 10;
####################
### data queries ###
####################
# get unique horses per race from runners, with bsps
WITH ordered_runners AS (
  SELECT r.*, ROW_NUMBER() OVER (PARTITION BY runner_id, market_id ORDER BY datetime DESC) AS rn
  FROM runners r
)
SELECT pt, runner_name, runner_id, sort_priority, status, adjustment_factor, bsp, market_id, datetime
FROM ordered_runners WHERE rn = 1
limit 39;

# pre bsp prices
select count(1) from runner_changes;
select count(1) from (
select r.*, m.venue, m.event_name, m.name, m.country_code, m.open_date, m.event_id, m.market_type,
		m.number_of_winners, m.number_active_runners, m.datetime as bsp_datetime
from runner_changes r
left join market_bsp_times m
on r.market_id = m.market_id
where r.datetime <= m.datetime
and r.datetime >= DATE_SUB(m.datetime, INTERVAL 3 HOUR)) t
;

# get result, bsp and runner statuses
with ordered_runners as (
		select *, ROW_NUMBER() OVER (PARTITION BY runner_id, market_id ORDER BY datetime DESC) AS rn
        from runners
        )
	select * from ordered_runners where rn=1
;

# get datetime when horses removed
select r.runner_name, r.runner_id, r.status, r.market_id, m.timezone as removed_timezone, min(r.datetime) as removed_datetime
from runners r
left join markets m on r.market_id = m.market_id
where status = 'REMOVED'
group by 1, 2, 3, 4, 5
limit 10;

select *
from runners
where market_id = 1.135703000
;

select *
from markets
where market_id = 1.135703000;

select *
from market_definitions
where market_id = 1.135703000;

######################################
### Investigate adjustment factors ###
######################################

select *
from runners
#where market_id = 1.130081731
where status = 'REMOVED'
limit 100;

select *
from runners
where market_id = 1.130280390
limit 1000;

select *
from runner_changes
where market_id = 1.130280390
limit 1000;

-- 11/3/2017 17:12:05, Ulls De Vassy (5882807) removed with af 5.83
-- 12/3/2017 08.58.06, Breath Of Blighty (10460777) removed with af 17.13
-- 12/3/2017 08.58.06, 105: 4.48, 694: 2.52, 842: 2.34
-- 12/3/2017 09.11.05, 841: 2.1, 921: 1.83
-- 12/3/2017 prev odds, 105: 5.2, 694: 2.84, 842: 2.62
-- CONCLUSION: Odds are as at the time, and then adjusted at the point horses are removed
-- so pretty tricky to get the correct adjustment factors as need to only include horses
-- that were removed after the point at which odds are taken


ALTER TABLE predictions_output_live
ADD COLUMN odds_21_win double AFTER odds_20_win;

ALTER TABLE predictions_output_live
ADD COLUMN odds_22_win double AFTER odds_21_win;

select * from predictions_output_live where odds_22_win is not null;

###############
### TESTING ###
###############
select * from order_fails_live limit 10;

select * from order_results_live order by placed_date desc limit 10;

select curdate() ;

select * from predictions_output_live where back = 1 or lay = 1 order by api_call_time_utc desc limit 10;
select * from predictions_output_live where runner_id = 39439922 order by api_call_time_utc desc limit 10;

select * from runners_and_odds_live limit 10;

select * from market_details_combined_live limit 10;

select * from bet_outcomes_live limit 10;

select * from order_results_live where market_id = 1.183192401;

select * from predictions_output_live where market_id = 1.183192401 and runner_id = 39422448;

#drop table bet_outcomes_live;
select * from bet_outcomes_live where market_id = 1.183192401;

select * from predictions_output_live limit 10;

WITH last_predictions AS (
	SELECT *, CASE WHEN back=1 THEN 'BACK' WHEN lay=1 THEN 'LAY' ELSE NULL END AS side,
			ROW_NUMBER() OVER (PARTITION BY runner_id, market_id, back, lay ORDER BY api_call_time_utc DESC) AS rn
    FROM predictions_output_live
    WHERE back = 1 OR lay = 1
    )
select o.market_id, o.selection_id, o.size, o.price, o.side, o.average_price_matched,
		b.placed_date, b.bet_outcome, b.settled_date, b.price_matched, b.price_reduced,
        b.size_settled, b.size_cancelled, b.profit,
        p.runner_name, p.api_call_time_utc, p.back_price_1, p.lay_price_1, p.venue, p.event_name,
        p.race_type, p.number_of_runners_orig, p.number_of_runners, p.p_sum, p.minutes_to_event_rounded,
        p.preds, p.pred_odds, p.back, p.lay
from order_results_live o
LEFT JOIN bet_outcomes_live b
		ON o.market_id = b.market_id
		AND o.selection_id = b.selection_id
		AND o.side = b.side
		AND o.bet_id = b.bet_id
LEFT JOIN (SELECT * FROM last_predictions WHERE rn=1) p
		ON o.market_id = p.market_id
        AND o.selection_id = p.runner_id
        AND o.side = p.side
WHERE o.order_status = 'EXECUTION_COMPLETE'
LIMIT 10;


select * FROM predictions_output_live limit 10;
select * from order_results_live limit 10;