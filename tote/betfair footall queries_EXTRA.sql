use betfair;

#drop table football_market_definitions_FIGS;
#drop table football_runners_FIGS;
#drop table football_runner_changes_FIGS;

###################
### Review data ###
###################
# and compare to UK data
select * from football_market_definitions_FIGS limit 10;
select count(1) from market_definitions;
select * from football_market_definitions_FIGS limit 10;
select COUnt(1), market_type from football_market_definitions_FIGS group by market_type;
select count(distinct event_id) from football_market_definitions_FIGS;
select count(distinct event_id) from football_market_definitions;

select * from football_runners_FIGS limit 1000;
# NOTE: Not all correct scores have the same options
select count(1) from football_runners;
select count(1) from football_runners_FIGS;

select * from football_runner_changes limit 10;
select count(1) from football_runner_changes;
select count(1) from football_runner_changes_FIGS;


###################
### add indices ###
###################
ALTER TABLE football_market_definitions_FIGS
MODIFY COLUMN market_id varchar(255);
CREATE INDEX market_id ON football_market_definitions_FIGS(market_id);

ALTER TABLE football_runner_changes_FIGS
MODIFY COLUMN market_id varchar(255);
CREATE INDEX market_id ON football_runner_changes_FIGS(market_id);

ALTER TABLE football_runners_FIGS
MODIFY COLUMN market_id varchar(255);
CREATE INDEX market_id ON football_runners_FIGS(market_id);

SHOW FULL PROCESSLIST;
#####################################
### create some additional tables ###
#####################################

# starting price times (in order to be able to select historic odds that are pre-bsp)
drop table if exists football_market_pre_inplay_time_FIGS;
create table football_market_pre_inplay_time_FIGS as (
	with football_market_pre_inplay as (
		select m.*, row_number() over (partition by market_id order by datetime desc) as rn
        from (select * from football_market_definitions_FIGS where in_play = 0) m
        )
	select * from football_market_pre_inplay where rn = 1
);
ALTER TABLE football_market_pre_inplay_time_FIGS
MODIFY COLUMN market_id varchar(255);
CREATE INDEX market_id ON football_market_pre_inplay_time_FIGS(market_id);

# check all exist
select count(distinct market_id) from football_market_definitions_FIGS;
select count(distinct market_id) from football_market_pre_inplay_time_FIGS;
# almost all exist - likely all significant events exist


# table with one line for every single market
drop table if exists football_markets_FIGS;
create table football_markets_tmp as (
	select min(venue) as venue, min(event_name) as event_name, min(event_id) as event_id_min, max(event_id) as event_id_max,
			country_code, timezone, event_type_id, market_type, market_id, betting_type,
            min(number_of_winners) as number_of_winners_min, max(open_date) as open_date_max, 
            max(market_time) as market_time_max, max(suspend_time) as suspend_time_max, max(settled_time) as settled_time_max,
            SUBSTRING_INDEX(event_name,' v ',1) as home, SUBSTRING_INDEX(event_name,' v ',-1) as away
    from football_market_definitions_FIGS
    group by country_code, timezone, event_type_id, market_type, market_id, betting_type, home, away
);

CREATE TABLE football_markets_FIGS AS (
	WITH ordered_markets AS (
		SELECT *, row_number() over
					(partition by market_id order by country_code asc, timezone DESC, market_type DESC) as rn
		FROM football_markets_tmp)
	SELECT * FROM ordered_markets WHERE rn=1
);
DROP TABLE football_markets_tmp;

select count(distinct market_id) from football_market_definitions_FIGS;
select count(distinct market_id) from football_markets_FIGS;
select count(1) from football_markets_FIGS;

ALTER TABLE football_markets_FIGS
MODIFY COLUMN market_id varchar(255);
CREATE INDEX market_id ON football_markets_FIGS(market_id);


# create market mapping
# NOTE: Not sure this is needed for football if can just join on event_ids

# get last runner_changes before every x mins pre event
select * from football_markets_FIGS limit 10;
select * from football_runner_changes_FIGS limit 10;
drop table if exists football_runner_changes_with_start_time_FIGS;
create table football_runner_changes_with_start_time_FIGS as (
	select r.runner_id, r.ltp, r.market_id, r.datetime, m.market_time_max,
			TIMESTAMPDIFF(MINUTE, r.datetime, cast(REPLACE(REPLACE(m.market_time_max, 'Z', ''), 'T', ' ') as datetime)) as time_to_event
    from football_runner_changes_FIGS r
    left join football_markets_FIGS m on r.market_id = m.market_id
    
);
select * from football_runner_changes_with_start_time_FIGS limit 10;
select count(1) from football_runner_changes_FIGS;
select count(1) from football_runner_changes_with_start_time_FIGS;

CREATE INDEX runner_id ON football_runner_changes_with_start_time_FIGS(runner_id);
CREATE INDEX market_id ON football_runner_changes_with_start_time_FIGS(market_id);
CREATE INDEX time_to_event ON football_runner_changes_with_start_time_FIGS(time_to_event);
CREATE INDEX datetime ON football_runner_changes_with_start_time_FIGS(datetime);

drop table if exists football_runner_changes_1m_before_FIGS;
create table football_runner_changes_1m_before_FIGS as (
	with ranked_changes as (
		select *, ROW_NUMBER() OVER (PARTITION BY runner_id, market_id ORDER BY datetime DESC) AS rn
        from football_runner_changes_with_start_time_FIGS where time_to_event >= 1
        )
	select * from ranked_changes where rn=1
);

drop table if exists football_runner_changes_60m_before_FIGS;
create table football_runner_changes_60m_before_FIGS as (
	with ranked_changes as (
		select *, ROW_NUMBER() OVER (PARTITION BY runner_id, market_id ORDER BY datetime DESC) AS rn
        from football_runner_changes_with_start_time_FIGS where time_to_event >= 60
        )
	select * from ranked_changes where rn=1
);

drop table if exists football_runner_changes_120m_before_FIGS;
create table football_runner_changes_120m_before_FIGS as (
	with ranked_changes as (
		select *, ROW_NUMBER() OVER (PARTITION BY runner_id, market_id ORDER BY datetime DESC) AS rn
        from football_runner_changes_with_start_time_FIGS where time_to_event >= 120
        )
	select * from ranked_changes where rn=1
);

drop table if exists football_runner_changes_180m_before_FIGS;
create table football_runner_changes_180m_before_FIGS as (
	with ranked_changes as (
		select *, ROW_NUMBER() OVER (PARTITION BY runner_id, market_id ORDER BY datetime DESC) AS rn
        from football_runner_changes_with_start_time_FIGS where time_to_event >= 180
        )
	select * from ranked_changes where rn=1
);

drop table if exists football_runner_changes_360m_before_FIGS;
create table football_runner_changes_360m_before_FIGS as (
	with ranked_changes as (
		select *, ROW_NUMBER() OVER (PARTITION BY runner_id, market_id ORDER BY datetime DESC) AS rn
        from football_runner_changes_with_start_time_FIGS where time_to_event >= 360
        )
	select * from ranked_changes where rn=1
);

drop table if exists football_runner_changes_720m_before_FIGS;
create table football_runner_changes_720m_before_FIGS as (
	with ranked_changes as (
		select *, ROW_NUMBER() OVER (PARTITION BY runner_id, market_id ORDER BY datetime DESC) AS rn
        from football_runner_changes_with_start_time_FIGS where time_to_event >= 720
        )
	select * from ranked_changes where rn=1
);

drop table if exists football_runner_changes_1080m_before_FIGS;
create table football_runner_changes_1080m_before_FIGS as (
	with ranked_changes as (
		select *, ROW_NUMBER() OVER (PARTITION BY runner_id, market_id ORDER BY datetime DESC) AS rn
        from football_runner_changes_with_start_time_FIGS where time_to_event >= 1080
        )
	select * from ranked_changes where rn=1
);

drop table if exists football_runner_changes_1440m_before_FIGS;
create table football_runner_changes_1440m_before_FIGS as (
	with ranked_changes as (
		select *, ROW_NUMBER() OVER (PARTITION BY runner_id, market_id ORDER BY datetime DESC) AS rn
        from football_runner_changes_with_start_time_FIGS where time_to_event >= 1440
        )
	select * from ranked_changes where rn=1
);

drop table if exists football_runner_changes_xm_before_FIGS;
create table football_runner_changes_xm_before_FIGS as (
	select * from football_runner_changes_1m_before_FIGS
    union
	select * from football_runner_changes_60m_before_FIGS
    union
    select * from football_runner_changes_120m_before_FIGS
    union
    select * from football_runner_changes_180m_before_FIGS
    union
    select * from football_runner_changes_360m_before_FIGS
    union
    select * from football_runner_changes_720m_before_FIGS
    union
    select * from football_runner_changes_1080m_before_FIGS
    union
    select * from football_runner_changes_1440m_before_FIGS
);

select count(1) from football_runner_changes_60m_before_FIGS;
select count(distinct runner_id, market_id) from football_runner_changes_FIGS;

SHOW FULL PROCESSLIST;



# get final outcomes with consistent names
select count(distinct runner_id, market_id) from football_runners_FIGS;
select count(distinct runner_id, market_id) from football_runners_FIGS where status in ('WINNER', 'LOSER', 'REMOVED');
select * from football_markets_FIGS where event_id_min <> event_id_max limit 100;

drop table if exists football_runner_outcomes_FIGS;
create table football_runner_outcomes_FIGS as (
	with football_runners_last_entry as (
		select * from (
			select *, ROW_NUMBER() OVER (PARTITION BY runner_id, market_id ORDER BY datetime DESC) AS rn
			from football_runners_FIGS) t
		where rn = 1
	)
	select r.*, m.event_name, m.event_id_min as event_id, m.home, m.away, m.country_code, m.timezone, m.market_type,
			m.betting_type, m.number_of_winners_min as number_of_winners,
			m.open_date_max as open_date, m.market_time_max as market_time,
			m.suspend_time_max as suspend_time, m.settled_time_max as settled_time,
            replace(replace(replace(replace(runner_name, home, 'Home'), away, 'Away'), ' - ','-'), '-', ' - ') as runner_name_general
	from football_runners_last_entry r
	left join football_markets_FIGS m on r.market_id = m.market_id
);

SHOW FULL PROCESSLIST;

select runner_name_general, COUNT(1) as count from football_runner_outcomes_FIGS group by runner_name_general order by count desc;

####################
### data queries ###
####################

select count(1)
from football_runner_outcomes_FIGS o 
left join football_runner_changes_1m_before_FIGS c
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

select * from predictions_output_live where back = 1 or lay = 1 order by api_call_time_utc desc limit 10;