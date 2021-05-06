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
create table markets as (
	select min(venue) as venue, min(event_name) as event_name, min(event_id) as event_id_min, max(event_id) as event_id_max,
			country_code, timezone, event_type_id, market_type, market_id, betting_type,
            min(number_of_winners) as number_of_winners_min, max(open_date) as open_date_max, 
            max(market_time) as market_time_max, max(suspend_time) as suspend_time_max, max(settled_time) as settled_time_max
    from market_definitions
    group by country_code, timezone, event_type_id, market_type, market_id, betting_type
);
# check all exist
select count(distinct market_id) from market_definitions;
select count(distinct market_id) from markets;
select count(1) from markets;

ALTER TABLE markets
MODIFY COLUMN market_id varchar(255);

CREATE INDEX market_id ON markets(market_id);
select * from markets where market_id in (select market_id from markets group by market_id having count(1) > 1);


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
create table runner_changes_with_start_time as (
	select r.runner_id, r.ltp, r.market_id, r.datetime, m.market_time_max,
			TIMESTAMPDIFF(MINUTE, r.datetime, cast(REPLACE(REPLACE(m.market_time_max, 'Z', ''), 'T', ' ') as datetime)) as time_to_event
    from runner_changes r
    left join markets m on r.market_id = m.market_id
    
);
select * from runner_changes_with_start_time limit 10;

CREATE INDEX runner_id ON runner_changes_with_start_time(runner_id);
CREATE INDEX market_id ON runner_changes_with_start_time(market_id);
CREATE INDEX time_to_event ON runner_changes_with_start_time(time_to_event);
CREATE INDEX datetime ON runner_changes_with_start_time(datetime);

create table runner_changes_30m_before as (
	with ranked_changes as (
		select *, ROW_NUMBER() OVER (PARTITION BY runner_id, market_id ORDER BY datetime DESC) AS rn
        from runner_changes_with_start_time where time_to_event >= 30
        )
	select * from ranked_changes where rn=1
);

create table runner_changes_60m_before as (
	with ranked_changes as (
		select *, ROW_NUMBER() OVER (PARTITION BY runner_id, market_id ORDER BY datetime DESC) AS rn
        from runner_changes_with_start_time where time_to_event >= 60
        )
	select * from ranked_changes where rn=1
);

create table runner_changes_90m_before as (
	with ranked_changes as (
		select *, ROW_NUMBER() OVER (PARTITION BY runner_id, market_id ORDER BY datetime DESC) AS rn
        from runner_changes_with_start_time where time_to_event >= 90
        )
	select * from ranked_changes where rn=1
);

create table runner_changes_120m_before as (
	with ranked_changes as (
		select *, ROW_NUMBER() OVER (PARTITION BY runner_id, market_id ORDER BY datetime DESC) AS rn
        from runner_changes_with_start_time where time_to_event >= 120
        )
	select * from ranked_changes where rn=1
);

create table runner_changes_150m_before as (
	with ranked_changes as (
		select *, ROW_NUMBER() OVER (PARTITION BY runner_id, market_id ORDER BY datetime DESC) AS rn
        from runner_changes_with_start_time where time_to_event >= 150
        )
	select * from ranked_changes where rn=1
);

create table runner_changes_180m_before as (
	with ranked_changes as (
		select *, ROW_NUMBER() OVER (PARTITION BY runner_id, market_id ORDER BY datetime DESC) AS rn
        from runner_changes_with_start_time where time_to_event >= 180
        )
	select * from ranked_changes where rn=1
);

create table runner_changes_210m_before as (
	with ranked_changes as (
		select *, ROW_NUMBER() OVER (PARTITION BY runner_id, market_id ORDER BY datetime DESC) AS rn
        from runner_changes_with_start_time where time_to_event >= 210
        )
	select * from ranked_changes where rn=1
);

create table runner_changes_240m_before as (
	with ranked_changes as (
		select *, ROW_NUMBER() OVER (PARTITION BY runner_id, market_id ORDER BY datetime DESC) AS rn
        from runner_changes_with_start_time where time_to_event >= 240
        )
	select * from ranked_changes where rn=1
);

create table runner_changes_270m_before as (
	with ranked_changes as (
		select *, ROW_NUMBER() OVER (PARTITION BY runner_id, market_id ORDER BY datetime DESC) AS rn
        from runner_changes_with_start_time where time_to_event >= 270
        )
	select * from ranked_changes where rn=1
);

create table runner_changes_300m_before as (
	with ranked_changes as (
		select *, ROW_NUMBER() OVER (PARTITION BY runner_id, market_id ORDER BY datetime DESC) AS rn
        from runner_changes_with_start_time where time_to_event >= 300
        )
	select * from ranked_changes where rn=1
);

SELECT * FROM runner_changes_with_start_time where market_id = 1.130081731;;
select * from market_definitions where market_id = 1.130025335;
select * from runner_changes where market_id = 1.130025335;
select * from runners where market_id = 1.130025335;
SHOW FULL PROCESSLIST;
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

select * from market_bsp_times Limit 10;
