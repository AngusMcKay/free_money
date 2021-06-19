use betfair;

#drop table football_market_definitions_FIGS;
#drop table football_runners_FIGS;
#drop table football_runner_changes_FIGS;

###################
### Review data ###
###################
select * from football_market_definitions limit 10;
select count(1) from market_definitions;
select * from football_market_definitions_FIGS limit 10;
select COUnt(1), market_type from football_market_definitions_FIGS group by market_type;
select count(distinct event_id) from football_market_definitions_FIGS;
select count(distinct event_id) from football_market_definitions;
select DISTINCT CAST(market_time as date) from football_market_definitions;
select * from football_market_definitions where CAST(market_time as date) = '2017-12-20' limit 10;


select * from football_runners limit 1000;
# NOTE: Not all correct scores have the same options
select count(1) from football_runners;
select count(1) from football_runners_FIGS;

select * from football_runner_changes limit 10;
select count(1) from football_runner_changes;
select count(1) from football_runner_changes_FIGS;

select * from football_market_definitions limit 100;

select distinct event_name, name, market_type, market_id from football_market_definitions where event_id = '29641959';
select distinct market_type from football_market_definitions;
select distinct market_type, runner_id, runner_name
from football_market_definitions m
left join football_runners r on m.market_id = r.market_id
limit 1000
;

# Notes:
# - Need to make sure the selections have the same names in each event and market_types are always the same
# - market "names" will differ as they are things like 'Wolves Clean Sheet'
# - smaller games have less market types
# - sometimes runner_names and ids are the same (like when Yes or No), but...
# - ...sometimes has team names as runner_name and id, so need to convert these to Home/Away etc

select * from football_runners where market_id = 1.167067357; # dumbarton clean sheet market (type is 'CLEAN_SHEET')
# runner_names are 'Yes' and 'No' which is a good sign! Also looks like the runner_ids are the same between markets, even better

select * from football_runners where runner_id = '22428984' limit 1000;

select * from football_market_definitions where market_id = '1.167340559';

###################
### add indices ###
###################
ALTER TABLE football_market_definitions
MODIFY COLUMN market_id varchar(255);
CREATE INDEX market_id ON football_market_definitions(market_id);

ALTER TABLE football_runner_changes
MODIFY COLUMN market_id varchar(255);
CREATE INDEX market_id ON football_runner_changes(market_id);

ALTER TABLE football_runners
MODIFY COLUMN market_id varchar(255);
CREATE INDEX market_id ON football_runners(market_id);

#####################################
### create some additional tables ###
#####################################

# starting price times (in order to be able to select historic odds that are pre-bsp)
drop table if exists football_market_pre_inplay_time;
create table football_market_pre_inplay_time as (
	with football_market_pre_inplay as (
		select m.*, row_number() over (partition by market_id order by datetime desc) as rn
        from (select * from football_market_definitions where in_play = 0) m
        )
	select * from football_market_pre_inplay where rn = 1
);
ALTER TABLE football_market_pre_inplay_time
MODIFY COLUMN market_id varchar(255);
CREATE INDEX market_id ON football_market_pre_inplay_time(market_id);

# check all exist
select count(distinct market_id) from football_market_definitions;
select count(distinct market_id) from football_market_pre_inplay_time;
# almost all exist - likely all significant events exist


# table with one line for every single market
drop table if exists football_markets;
create table football_markets_tmp as (
	select min(venue) as venue, min(event_name) as event_name, min(event_id) as event_id_min, max(event_id) as event_id_max,
			country_code, timezone, event_type_id, market_type, market_id, betting_type,
            min(number_of_winners) as number_of_winners_min, max(open_date) as open_date_max, 
            max(market_time) as market_time_max, max(suspend_time) as suspend_time_max, max(settled_time) as settled_time_max,
            SUBSTRING_INDEX(event_name,' v ',1) as home, SUBSTRING_INDEX(event_name,' v ',-1) as away
    from football_market_definitions
    group by country_code, timezone, event_type_id, market_type, market_id, betting_type, home, away
);

CREATE TABLE football_markets AS (
	WITH ordered_markets AS (
		SELECT *, row_number() over
					(partition by market_id order by country_code asc, timezone DESC, market_type DESC) as rn
		FROM football_markets_tmp)
	SELECT * FROM ordered_markets WHERE rn=1
);
DROP TABLE football_markets_tmp;

select count(distinct market_id) from football_market_definitions;
select count(distinct market_id) from football_markets;
select count(1) from football_markets;

ALTER TABLE football_markets
MODIFY COLUMN market_id varchar(255);
CREATE INDEX market_id ON football_markets(market_id);


# create market mapping
# NOTE: Not sure this is needed for football if can just join on event_ids

# get last runner_changes before every x mins pre event
select * from football_markets limit 10;
select * from football_runner_changes limit 10;
drop table football_runner_changes_with_start_time;
create table football_runner_changes_with_start_time as (
	select r.runner_id, r.ltp, r.market_id, r.datetime, m.market_time_max,
			TIMESTAMPDIFF(MINUTE, r.datetime, cast(REPLACE(REPLACE(m.market_time_max, 'Z', ''), 'T', ' ') as datetime)) as time_to_event
    from football_runner_changes r
    left join football_markets m on r.market_id = m.market_id
    
);
select * from football_runner_changes_with_start_time limit 10;
select count(1) from football_runner_changes;
select count(1) from football_runner_changes_with_start_time;

CREATE INDEX runner_id ON football_runner_changes_with_start_time(runner_id);
CREATE INDEX market_id ON football_runner_changes_with_start_time(market_id);
CREATE INDEX time_to_event ON football_runner_changes_with_start_time(time_to_event);
CREATE INDEX datetime ON football_runner_changes_with_start_time(datetime);

drop table if exists football_runner_changes_1m_before;
create table football_runner_changes_1m_before as (
	with ranked_changes as (
		select *, ROW_NUMBER() OVER (PARTITION BY runner_id, market_id ORDER BY datetime DESC) AS rn
        from football_runner_changes_with_start_time where time_to_event >= 1
        )
	select * from ranked_changes where rn=1
);

drop table if exists football_runner_changes_60m_before;
create table football_runner_changes_60m_before as (
	with ranked_changes as (
		select *, ROW_NUMBER() OVER (PARTITION BY runner_id, market_id ORDER BY datetime DESC) AS rn
        from football_runner_changes_with_start_time where time_to_event >= 60
        )
	select * from ranked_changes where rn=1
);

drop table if exists football_runner_changes_120m_before;
create table football_runner_changes_120m_before as (
	with ranked_changes as (
		select *, ROW_NUMBER() OVER (PARTITION BY runner_id, market_id ORDER BY datetime DESC) AS rn
        from football_runner_changes_with_start_time where time_to_event >= 120
        )
	select * from ranked_changes where rn=1
);

drop table if exists football_runner_changes_180m_before;
create table football_runner_changes_180m_before as (
	with ranked_changes as (
		select *, ROW_NUMBER() OVER (PARTITION BY runner_id, market_id ORDER BY datetime DESC) AS rn
        from football_runner_changes_with_start_time where time_to_event >= 180
        )
	select * from ranked_changes where rn=1
);

drop table if exists football_runner_changes_360m_before;
create table football_runner_changes_360m_before as (
	with ranked_changes as (
		select *, ROW_NUMBER() OVER (PARTITION BY runner_id, market_id ORDER BY datetime DESC) AS rn
        from football_runner_changes_with_start_time where time_to_event >= 360
        )
	select * from ranked_changes where rn=1
);

drop table if exists football_runner_changes_720m_before;
create table football_runner_changes_720m_before as (
	with ranked_changes as (
		select *, ROW_NUMBER() OVER (PARTITION BY runner_id, market_id ORDER BY datetime DESC) AS rn
        from football_runner_changes_with_start_time where time_to_event >= 720
        )
	select * from ranked_changes where rn=1
);

drop table if exists football_runner_changes_1080m_before;
create table football_runner_changes_1080m_before as (
	with ranked_changes as (
		select *, ROW_NUMBER() OVER (PARTITION BY runner_id, market_id ORDER BY datetime DESC) AS rn
        from football_runner_changes_with_start_time where time_to_event >= 1080
        )
	select * from ranked_changes where rn=1
);

drop table if exists football_runner_changes_1440m_before;
create table football_runner_changes_1440m_before as (
	with ranked_changes as (
		select *, ROW_NUMBER() OVER (PARTITION BY runner_id, market_id ORDER BY datetime DESC) AS rn
        from football_runner_changes_with_start_time where time_to_event >= 1440
        )
	select * from ranked_changes where rn=1
);

drop table if exists football_runner_changes_xm_before;
create table football_runner_changes_xm_before as (
	select * from football_runner_changes_1m_before
    union
	select * from football_runner_changes_60m_before
    union
    select * from football_runner_changes_120m_before
    union
    select * from football_runner_changes_180m_before
    union
    select * from football_runner_changes_360m_before
    union
    select * from football_runner_changes_720m_before
    union
    select * from football_runner_changes_1080m_before
    union
    select * from football_runner_changes_1440m_before
);

select count(1) from football_runner_changes_60m_before;
select count(distinct runner_id, market_id) from football_runner_changes;

SHOW FULL PROCESSLIST;



# get final outcomes with consistent names
select count(distinct runner_id, market_id) from football_runners;
select count(distinct runner_id, market_id) from football_runners where status in ('WINNER', 'LOSER', 'REMOVED');
select * from football_markets where event_id_min <> event_id_max limit 100;

drop table if exists football_runner_outcomes;
create table football_runner_outcomes as (
	with football_runners_last_entry as (
		select * from (
			select *, ROW_NUMBER() OVER (PARTITION BY runner_id, market_id ORDER BY datetime DESC) AS rn
			from football_runners) t
		where rn = 1
	)
	select r.*, m.event_name, m.event_id_min as event_id, m.home, m.away, m.country_code, m.timezone, m.market_type,
			m.betting_type, m.number_of_winners_min as number_of_winners,
			m.open_date_max as open_date, m.market_time_max as market_time,
			m.suspend_time_max as suspend_time, m.settled_time_max as settled_time,
            replace(replace(replace(replace(runner_name, home, 'Home'), away, 'Away'), ' - ','-'), '-', ' - ') as runner_name_general
	from football_runners_last_entry r
	left join football_markets m on r.market_id = m.market_id
);

SHOW FULL PROCESSLIST;

select runner_name_general, COUNT(1) as count from football_runner_outcomes group by runner_name_general order by count desc;
select * from football_runner_outcomes where event_id = 29634101 limit 1000;

select * from football_runners r
left join football_markets m on r.market_id = m.market_id
limit 1000;

select * from football_runners where market_id = 1.167426555 order by datetime;

####################
### data queries ###
####################

select count(1)
from football_runner_outcomes o 
left join football_runner_changes_1m_before c
on o.runner_id = c.runner_id and o.market_id = c.market_id
limit 10;

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

select * from predictions_output_live where back = 1 or lay = 1 order by api_call_time_utc desc limit 10;