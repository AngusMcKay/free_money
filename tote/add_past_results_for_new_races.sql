
### QUERIES TO BE EXECUTED IN PYTHON SCRIPT TO ALLOW MORE CONTROL

CREATE INDEX race_id ON horses_data_new(race_id);
CREATE INDEX horse_id ON horses_data_new(horse_id);
CREATE INDEX race_id ON races_data_new(race_id);
### ADD HORSE TIMES ###
# change finish distance to estimated yards, then use to create estimated finish time
# see https://en.wikipedia.org/wiki/Horse_length which says in british horse racing the distance is lengths-per-second
# use conversion of 5 lengths-per-second https://www.britishhorseracing.com/press_releases/british-horseracing-authority-introduces-new-finishing-distances-and-alteration-to-their-calculation/
CREATE TABLE horses_times_pretmp AS
SELECT *, REPLACE(REPLACE(finish_distance, ' Lengths',''),' Length', '') AS finish_distance_wo_length
FROM horses_data_new
;

# do try except of following in python in order to return error if new finish distance string which not seen before
CREATE TABLE horses_times_tmp AS
SELECT race_id, horse_id, horse_name, finish_position, winning_time_seconds, finish_distance, finish_distance_wo_length,
        CASE 	WHEN finish_position = 1 THEN 0
				WHEN RIGHT(finish_distance_wo_length, 1) = '¼' THEN 0.25
				WHEN RIGHT(finish_distance_wo_length, 1) = '½' THEN 0.5
				WHEN RIGHT(finish_distance_wo_length, 1) = '¾' THEN 0.75
				ELSE 0.0 END AS finish_distance_fractional_part,
		CASE	WHEN finish_position = 1 THEN 0
				WHEN finish_distance_wo_length IN ('nse', 'ns', '¼', '½', '¾') THEN 0.0
				WHEN finish_distance_wo_length IN ('sh', 's.h', 'hd', 'dh', 'd.h', 'snk', 'nk') THEN 0.1
				WHEN finish_distance_wo_length IN ('dst', 'DIS', 'dist') THEN 30
				WHEN (LENGTH(finish_distance_wo_length)>1 AND RIGHT(finish_distance_wo_length, 1) IN ('¼', '½', '¾'))
				THEN CAST(LEFT(finish_distance_wo_length, LENGTH(finish_distance_wo_length)-1) AS DECIMAL(10,2))
				ELSE CAST(finish_distance_wo_length AS DECIMAL(10,2)) END AS finish_distance_non_fractional_part,
		CASE WHEN finish_position = 0 THEN 1 ELSE 0 END AS did_not_finish,
        CASE WHEN finish_position = 0 THEN 999 ELSE finish_position END AS finish_position_for_ordering
FROM horses_times_pretmp
;

CREATE TABLE horses_times_new AS
SELECT *, winning_time_seconds + 5*SUM(finish_distance_fractional_part + finish_distance_non_fractional_part)
			OVER (PARTITION BY race_id ORDER BY finish_position_for_ordering ROWS UNBOUNDED PRECEDING) AS horse_time
FROM horses_times_tmp
;

CREATE INDEX race_id ON horses_times_new(race_id);
CREATE INDEX horse_id ON horses_times_new(horse_id);

DROP TABLE horses_times_pretmp;
DROP TABLE horses_times_tmp;


### CREATE GOINGS FOR EACH RACE ID ###
CREATE TABLE races_goings_new AS
SELECT DISTINCT race_id,
	CASE	WHEN LEFT(LOWER(going), 12) = 'good to firm' THEN 'good to firm'
			WHEN LEFT(LOWER(going), 12) = 'good to soft' THEN 'good to soft'
            WHEN LEFT(LOWER(going), 16) = 'good to yielding' THEN 'good to yielding'
            WHEN LEFT(LOWER(going), 16) = 'yielding to soft' THEN 'yielding to soft'
            WHEN LEFT(LOWER(going), 8) = 'standard' THEN 'standard'
            WHEN LEFT(LOWER(going), 8) = 'yielding' THEN 'yielding'
            WHEN LEFT(LOWER(going), 6) = 'sloppy' THEN 'sloppy'
            WHEN LEFT(LOWER(going), 5) = 'heavy' THEN 'heavy'
            WHEN LEFT(LOWER(going), 4) = 'fast' THEN 'fast'
            WHEN LEFT(LOWER(going), 4) = 'firm' THEN 'firm'
            WHEN LEFT(LOWER(going), 4) = 'good' THEN 'good'
            WHEN LEFT(LOWER(going), 4) = 'slow' THEN 'slow'
            WHEN LEFT(LOWER(going), 4) = 'soft' THEN 'soft'
            WHEN LEFT(LOWER(going), 9) = 'very soft' THEN 'soft' # only 0.1% are 'very soft'
            ELSE 'good'
            END AS going_main,
	CASE 	WHEN LEFT(LOWER(going), 8) = 'standard' THEN 'standard'
			WHEN LEFT(LOWER(going), 8) = 'yielding' THEN 'yielding'
            WHEN LEFT(LOWER(going), 6) = 'sloppy' THEN 'sloppy'
            WHEN LEFT(LOWER(going), 5) = 'heavy' THEN 'heavy'
            WHEN LEFT(LOWER(going), 4) = 'fast' THEN 'fast'
            WHEN LEFT(LOWER(going), 4) = 'firm' THEN 'firm'
            WHEN LEFT(LOWER(going), 4) = 'good' THEN 'good'
            WHEN LEFT(LOWER(going), 4) = 'slow' THEN 'slow'
            WHEN LEFT(LOWER(going), 4) = 'soft' THEN 'soft'
            WHEN LEFT(LOWER(going), 9) = 'very soft' THEN 'soft'
            ELSE 'good'
            END AS going_grouped
FROM horses_data_new
;

CREATE INDEX race_id ON races_goings_new(race_id);


### HADICAPS ###
SELECT DISTINCT handicap FROM horses_data_new;
#DROP TABLE horses_handicaps;
CREATE TABLE horses_handicaps_new AS
SELECT race_id, horse_id,
		handicap, 14*SUBSTRING_INDEX(handicap, '-', 1) + SUBSTRING_INDEX(handicap, '-', -1) AS handicap_pounds,
        pr_1_weight, 14*SUBSTRING_INDEX(pr_1_weight, '-', 1) + SUBSTRING_INDEX(pr_1_weight, '-', -1) AS pr_1_weight_pounds,
        pr_2_weight, 14*SUBSTRING_INDEX(pr_2_weight, '-', 1) + SUBSTRING_INDEX(pr_2_weight, '-', -1) AS pr_2_weight_pounds,
        pr_3_weight, 14*SUBSTRING_INDEX(pr_3_weight, '-', 1) + SUBSTRING_INDEX(pr_3_weight, '-', -1) AS pr_3_weight_pounds,
        pr_4_weight, 14*SUBSTRING_INDEX(pr_4_weight, '-', 1) + SUBSTRING_INDEX(pr_4_weight, '-', -1) AS pr_4_weight_pounds,
        pr_5_weight, 14*SUBSTRING_INDEX(pr_5_weight, '-', 1) + SUBSTRING_INDEX(pr_5_weight, '-', -1) AS pr_5_weight_pounds,
        pr_6_weight, 14*SUBSTRING_INDEX(pr_6_weight, '-', 1) + SUBSTRING_INDEX(pr_6_weight, '-', -1) AS pr_6_weight_pounds
FROM horses_data_new
;

CREATE INDEX race_id ON horses_handicaps_new(race_id);
CREATE INDEX horse_id ON horses_handicaps_new(horse_id);


### RACE TYPES ###
# categorise some UK races based on age, distance and weight according to https://www.keithprowse.co.uk/news-and-blog/2019/04/05/the-differences-between-flat-racing-and-jump-racing/
# i.e. 2YO race means flat, <3520 yards means flat, weight under 140 means flat
# although seems like the weight data doesn't actually agree with this based on data available
CREATE TABLE races_types_new AS
SELECT race_date, course, country, rd.race_id, age, yards, name,
		CASE 	WHEN LOWER(name) LIKE '%hurdle%' THEN 'hurdle'
				WHEN LOWER(name) LIKE '%chase%' THEN 'chase'
                WHEN LOWER(name) LIKE '%flat%' THEN 'flat'
                WHEN (yards>4820 AND lower(country) IN ('england','eire','wales','scotland')) THEN 'jumps'
                WHEN (yards>3300 AND lower(country) IN ('england','eire','wales','scotland')) THEN 'flat'
                ELSE 'unknown' END AS race_type,
		CASE 	WHEN LOWER(name) LIKE '%hurdle%' THEN 'hurdle'
				WHEN LOWER(name) LIKE '%chase%' THEN 'chase'
                WHEN LOWER(name) LIKE '%flat%' THEN 'flat'
				WHEN (yards>4820 AND lower(country) IN ('england','eire','wales','scotland')) THEN 'jumps'
                WHEN (yards>3300 AND lower(country) IN ('england','eire','wales','scotland')) THEN 'flat'
                WHEN LEFT(LOWER(age),3) IN ('1yo','2yo') THEN 'devisedflat'
                ELSE 'unknown' END AS race_type_devised
FROM races_data_new rd
LEFT JOIN (SELECT race_id, MIN(handicap_pounds) AS min_handicap_pounds, MAX(handicap_pounds) AS max_handicap_pounds FROM horses_handicaps_new GROUP BY race_id) hh
ON rd.race_id = hh.race_id
;

CREATE INDEX race_id ON races_types_new(race_id);

# create table based on past results to see if it sheds more light
#DROP TABLE races_pr_types;
CREATE TABLE races_pr_types_new AS
SELECT DISTINCT race_id, race_type
FROM
(
SELECT DISTINCT pr_1_race_id AS race_id, pr_1_run_type AS race_type FROM horses_data_new
UNION
SELECT DISTINCT pr_2_race_id AS race_id, pr_2_run_type AS race_type FROM horses_data_new
UNION
SELECT DISTINCT pr_3_race_id AS race_id, pr_3_run_type AS race_type FROM horses_data_new
UNION
SELECT DISTINCT pr_4_race_id AS race_id, pr_4_run_type AS race_type FROM horses_data_new
UNION
SELECT DISTINCT pr_5_race_id AS race_id, pr_5_run_type AS race_type FROM horses_data_new
UNION
SELECT DISTINCT pr_6_race_id AS race_id, pr_6_run_type AS race_type FROM horses_data_new
) t
;

CREATE INDEX race_id ON races_pr_types_new(race_id);

#DROP TABLE races_types_combined;
CREATE TABLE races_types_combined_new AS
SELECT race_date, course, country, rt.race_id, age, yards, name,
		CASE 	WHEN rpt.race_type IN ('FLAT', 'N_H_FLAT') THEN 'flat'
				WHEN rpt.race_type IS NULL THEN rt.race_type
                ELSE LOWER(rpt.race_type) END AS race_type,
		CASE 	WHEN rpt.race_type IN ('FLAT', 'N_H_FLAT') THEN 'flat'
				WHEN rpt.race_type IS NULL THEN rt.race_type_devised
                ELSE LOWER(rt.race_type) END AS race_type_devised
FROM races_types_new rt
LEFT JOIN races_pr_types_new rpt ON rt.race_id = rpt.race_id
;

CREATE INDEX race_id ON races_types_combined_new(race_id);


### WEATHER ###
CREATE TABLE races_weather_new AS
SELECT race_id,
		CASE 	WHEN LOWER(weather) LIKE '%fine%' THEN 'fine'
				WHEN LOWER(weather) LIKE '%rain%' THEN 'rain'
				WHEN (LOWER(weather) LIKE '%snow%' OR LOWER(weather) LIKE '%hail%' OR LOWER(weather) LIKE '%sleet%') THEN 'snowandhail'
                WHEN (LOWER(weather) LIKE '%shower%' OR LOWER(weather) LIKE '%wet%' OR LOWER(weather) LIKE '%drizzle%') THEN 'wet'
                WHEN LOWER(weather) LIKE '%overcast%' THEN 'overcast'
                WHEN LOWER(weather) LIKE '%cloudy%' THEN 'cloudy'
                WHEN LOWER(weather) LIKE '%sunny%' THEN 'sunny'
                WHEN LOWER(weather) LIKE '%bright%' THEN 'bright'
                WHEN LOWER(weather) LIKE '%dry%' THEN 'dry'
				ELSE 'other'
		END AS weather_grouped,
        
		CASE WHEN LOWER(weather) LIKE '%fine%' THEN 1 ELSE 0 END AS fine,
        CASE WHEN LOWER(weather) LIKE '%overcast%' THEN 1 ELSE 0 END AS overcast,
        CASE WHEN LOWER(weather) LIKE '%cold%' THEN 1 ELSE 0 END AS cold,
        CASE WHEN LOWER(weather) LIKE '%cloudy%' THEN 1 ELSE 0 END AS cloudy,
        CASE WHEN LOWER(weather) LIKE '%bright%' THEN 1 ELSE 0 END AS bright,
        CASE WHEN LOWER(weather) LIKE '%rain%' THEN 1 ELSE 0 END AS rain,
        CASE WHEN LOWER(weather) LIKE '%showers%' THEN 1 ELSE 0 END AS showers,
        CASE WHEN LOWER(weather) LIKE '%drizzle%' THEN 1 ELSE 0 END AS drizzle,
        CASE WHEN LOWER(weather) LIKE '%sunny%' THEN 1 ELSE 0 END AS sunny,
        CASE WHEN LOWER(weather) LIKE '%windy%' THEN 1 ELSE 0 END AS windy,
        CASE WHEN LOWER(weather) LIKE '%dry%' THEN 1 ELSE 0 END AS dry,
        CASE WHEN LOWER(weather) LIKE '%breezy%' THEN 1 ELSE 0 END AS breezy,
        CASE WHEN LOWER(weather) LIKE '%unsettled%' THEN 1 ELSE 0 END AS unsettled,
        CASE WHEN LOWER(weather) LIKE '%snow%' THEN 1 ELSE 0 END AS snow,
        CASE WHEN LOWER(weather) LIKE '%foggy%' THEN 1 ELSE 0 END AS foggy,
		CASE WHEN LOWER(weather) LIKE '%misty%' THEN 1 ELSE 0 END AS misty,
        CASE WHEN LOWER(weather) LIKE '%warm%' THEN 1 ELSE 0 END AS warm
FROM races_data_new
;        
CREATE INDEX race_id ON races_weather_new(race_id);


### COMBINED DATA ###
CREATE TABLE horses_data_combined_stg1 AS
SELECT hd.horse_id, hd.race_date, hd.race_id, hd.betting_odds, hd.finish_position, ht.horse_time, ht.did_not_finish, ht.finish_position_for_ordering,
		hd.course, hd.surface, hd.going, hd.yards, hd.runners, hd.prize1, hd.number_of_placed_rides,
		hh.handicap_pounds, hd.horse_age, hd.horse_sex, hd.horse_last_ran_days, hd.horse_form,
        rd.country, rd.weather, rd.race_class, rg.going_main, rg.going_grouped, rtc.race_type, rtc.race_type_devised,
        rw.weather_grouped#,
        #ROW_NUMBER() OVER(PARTITION BY hd.horse_id ORDER BY hd.race_date, hd.race_time) AS horse_race_number
FROM horses_data_new hd
LEFT JOIN horses_times_new ht ON hd.race_id = ht.race_id AND hd.horse_id = ht.horse_id
LEFT JOIN horses_handicaps_new hh ON hd.race_id = hh.race_id AND hd.horse_id = hh.horse_id
LEFT JOIN races_data_new rd ON hd.race_id = rd.race_id
LEFT JOIN races_goings_new rg ON hd.race_id = rg.race_id
LEFT JOIN races_types_combined_new rtc ON hd.race_id = rtc.race_id
LEFT JOIN races_weather_new rw ON hd.race_id = rw.race_id
;

CREATE INDEX horse_id ON horses_data_combined_stg1(horse_id);
CREATE INDEX race_id ON horses_data_combined_stg1(race_id);

#DROP TABLE horses_data_combined_new;
CREATE TABLE horses_data_combined_new AS
SELECT hd.*, rn.horse_race_number + IFNULL(hdc.prev_horse_race_number, 0) AS horse_race_number
FROM horses_data_combined_stg1 hd
LEFT JOIN (SELECT horse_id, race_id, ROW_NUMBER() OVER(PARTITION BY horse_id ORDER BY race_date, race_time) AS horse_race_number FROM horses_data_new) rn
ON hd.horse_id = rn.horse_id AND hd.race_id = rn.race_id
LEFT JOIN (SELECT horse_id, MAX(horse_race_number) AS prev_horse_race_number FROM horses_data_combined GROUP BY horse_id) hdc
ON hd.horse_id = hdc.horse_id
;

CREATE INDEX horse_id ON horses_data_combined_new(horse_id);
CREATE INDEX race_id ON horses_data_combined_new(race_id);
CREATE INDEX horse_race_number ON horses_data_combined_new(horse_race_number);

DROP TABLE horses_data_combined_stg1;


### SAME FOR NO NON-RUNNERS DATA ###
CREATE TABLE horses_data_combined_no_non_runners_stg1 AS
SELECT * FROM horses_data_combined_new WHERE horse_time IS NOT NULL AND did_not_finish=0
;
ALTER TABLE horses_data_combined_no_non_runners_stg1 DROP horse_race_number
;
CREATE INDEX horse_id ON horses_data_combined_no_non_runners_stg1(horse_id);
CREATE INDEX race_id ON horses_data_combined_no_non_runners_stg1(race_id);
;
CREATE TABLE horses_data_combined_no_non_runners_stg2 AS
SELECT t1.*, t2.race_time
FROM horses_data_combined_no_non_runners_stg1 t1
LEFT JOIN horses_data_new t2 ON t1.race_id = t2.race_id AND t1.horse_id = t2.horse_id
;
CREATE INDEX horse_id ON horses_data_combined_no_non_runners_stg2(horse_id);
CREATE INDEX race_id ON horses_data_combined_no_non_runners_stg2(race_id);
;
CREATE TABLE horses_data_combined_no_non_runners_new AS
SELECT hd.*, rn.horse_race_number + IFNULL(hdc.prev_horse_race_number, 0) AS horse_race_number
FROM horses_data_combined_no_non_runners_stg2 hd
LEFT JOIN (SELECT horse_id, race_id, ROW_NUMBER() OVER(PARTITION BY horse_id ORDER BY race_date, race_time) AS horse_race_number FROM horses_data_combined_no_non_runners_stg2) rn
ON hd.horse_id = rn.horse_id AND hd.race_id = rn.race_id
LEFT JOIN (SELECT horse_id, MAX(horse_race_number) AS prev_horse_race_number FROM horses_data_combined_no_non_runners GROUP BY horse_id) hdc
ON hd.horse_id = hdc.horse_id
;

CREATE INDEX horse_id ON horses_data_combined_no_non_runners_new(horse_id);
CREATE INDEX race_id ON horses_data_combined_no_non_runners_new(race_id);
CREATE INDEX horse_race_number ON horses_data_combined_no_non_runners_new(horse_race_number);

DROP TABLE horses_data_combined_no_non_runners_stg1;
DROP TABLE horses_data_combined_no_non_runners_stg2;


### ADD TO EXISTING DATA ###
INSERT INTO horses_data_combined SELECT * FROM horses_data_combined_new;
INSERT INTO horses_data_combined_no_non_runners SELECT * FROM horses_data_combined_no_non_runners_new;
INSERT INTO horses_handicaps SELECT * FROM horses_handicaps_new;
INSERT INTO horses_times SELECT * FROM horses_times_new;
INSERT INTO races_goings SELECT * FROM races_goings_new;
INSERT INTO races_pr_types SELECT * FROM races_pr_types_new;
INSERT INTO races_types SELECT * FROM races_types_new;
INSERT INTO races_types_combined SELECT * FROM races_types_combined_new;
INSERT INTO races_weather SELECT * FROM races_weather_new;


### NEW TEN PAST RESULTS DATA ###
CREATE TABLE horses_data_combined_with_past_results_new AS
SELECT 
		# current race details
		h1.horse_id, h1.race_date, h1.race_id, h1.betting_odds, h1.finish_position, h1.horse_time, h1.did_not_finish, h1.finish_position_for_ordering,
        h1.course, h1.surface, h1.going, h1.yards, h1.runners, h1.prize1, h1.number_of_placed_rides,
		h1.handicap_pounds, h1.horse_age, h1.horse_sex, h1.horse_last_ran_days, h1.horse_form,
        h1.country, h1.weather, h1.race_class, h1.going_main, h1.going_grouped, h1.race_type, h1.race_type_devised, h1.weather_grouped,
        
        # past result 1 details
        h2.horse_time AS pr_1_horse_time, h2.betting_odds AS pr_1_betting_odds, h2.finish_position AS pr_1_finish_position, h2.did_not_finish AS pr_1_did_not_finish, h2.finish_position_for_ordering AS pr_1_finish_position_for_ordering,
        h2.course AS pr_1_course, h2.surface AS pr_1_surface, h2.going AS pr_1_going, h2.yards AS pr_1_yards, h2.runners AS pr_1_runners, h2.prize1 AS pr_1_prize1, h2.number_of_placed_rides AS pr_1_number_of_placed_rides,
		h2.handicap_pounds AS pr_1_handicap_pounds, h2.horse_age AS pr_1_horse_age, h2.horse_sex AS pr_1_horse_sex, h2.horse_last_ran_days AS pr_1_horse_last_ran_days, h2.horse_form AS pr_1_horse_form,
        h2.country AS pr_1_country, h2.weather AS pr_1_weather, h2.race_class AS pr_1_race_class, h2.going_main AS pr_1_going_main, h2.going_grouped AS pr_1_going_grouped, h2.race_type AS pr_1_race_type, h2.race_type_devised AS pr_1_race_type_devised,
        h2.weather_grouped AS pr_1_weather_grouped,
        
        # past result 1 details
        h3.horse_time AS pr_2_horse_time, h3.betting_odds AS pr_2_betting_odds, h3.finish_position AS pr_2_finish_position, h3.did_not_finish AS pr_2_did_not_finish, h3.finish_position_for_ordering AS pr_2_finish_position_for_ordering,
        h3.course AS pr_2_course, h3.surface AS pr_2_surface, h3.going AS pr_2_going, h3.yards AS pr_2_yards, h3.runners AS pr_2_runners, h3.prize1 AS pr_2_prize1, h3.number_of_placed_rides AS pr_2_number_of_placed_rides,
		h3.handicap_pounds AS pr_2_handicap_pounds, h3.horse_age AS pr_2_horse_age, h3.horse_sex AS pr_2_horse_sex, h3.horse_last_ran_days AS pr_2_horse_last_ran_days, h3.horse_form AS pr_2_horse_form,
        h3.country AS pr_2_country, h3.weather AS pr_2_weather, h3.race_class AS pr_2_race_class, h3.going_main AS pr_2_going_main, h3.going_grouped AS pr_2_going_grouped, h3.race_type AS pr_2_race_type, h3.race_type_devised AS pr_2_race_type_devised,
        h3.weather_grouped AS pr_2_weather_grouped,
        
        # past result 1 details
        h4.horse_time AS pr_3_horse_time, h4.betting_odds AS pr_3_betting_odds, h4.finish_position AS pr_3_finish_position, h4.did_not_finish AS pr_3_did_not_finish, h4.finish_position_for_ordering AS pr_3_finish_position_for_ordering,
        h4.course AS pr_3_course, h4.surface AS pr_3_surface, h4.going AS pr_3_going, h4.yards AS pr_3_yards, h4.runners AS pr_3_runners, h4.prize1 AS pr_3_prize1, h4.number_of_placed_rides AS pr_3_number_of_placed_rides,
		h4.handicap_pounds AS pr_3_handicap_pounds, h4.horse_age AS pr_3_horse_age, h4.horse_sex AS pr_3_horse_sex, h4.horse_last_ran_days AS pr_3_horse_last_ran_days, h4.horse_form AS pr_3_horse_form,
        h4.country AS pr_3_country, h4.weather AS pr_3_weather, h4.race_class AS pr_3_race_class, h4.going_main AS pr_3_going_main, h4.going_grouped AS pr_3_going_grouped, h4.race_type AS pr_3_race_type, h4.race_type_devised AS pr_3_race_type_devised,
        h4.weather_grouped AS pr_3_weather_grouped,
        
        # past result 1 details
        h5.horse_time AS pr_4_horse_time, h5.betting_odds AS pr_4_betting_odds, h5.finish_position AS pr_4_finish_position, h5.did_not_finish AS pr_4_did_not_finish, h5.finish_position_for_ordering AS pr_4_finish_position_for_ordering,
        h5.course AS pr_4_course, h5.surface AS pr_4_surface, h5.going AS pr_4_going, h5.yards AS pr_4_yards, h5.runners AS pr_4_runners, h5.prize1 AS pr_4_prize1, h5.number_of_placed_rides AS pr_4_number_of_placed_rides,
		h5.handicap_pounds AS pr_4_handicap_pounds, h5.horse_age AS pr_4_horse_age, h5.horse_sex AS pr_4_horse_sex, h5.horse_last_ran_days AS pr_4_horse_last_ran_days, h5.horse_form AS pr_4_horse_form,
        h5.country AS pr_4_country, h5.weather AS pr_4_weather, h5.race_class AS pr_4_race_class, h5.going_main AS pr_4_going_main, h5.going_grouped AS pr_4_going_grouped, h5.race_type AS pr_4_race_type, h5.race_type_devised AS pr_4_race_type_devised,
        h5.weather_grouped AS pr_4_weather_grouped,
        
        # past result 1 details
        h6.horse_time AS pr_5_horse_time, h6.betting_odds AS pr_5_betting_odds, h6.finish_position AS pr_5_finish_position, h6.did_not_finish AS pr_5_did_not_finish, h6.finish_position_for_ordering AS pr_5_finish_position_for_ordering,
        h6.course AS pr_5_course, h6.surface AS pr_5_surface, h6.going AS pr_5_going, h6.yards AS pr_5_yards, h6.runners AS pr_5_runners, h6.prize1 AS pr_5_prize1, h6.number_of_placed_rides AS pr_5_number_of_placed_rides,
		h6.handicap_pounds AS pr_5_handicap_pounds, h6.horse_age AS pr_5_horse_age, h6.horse_sex AS pr_5_horse_sex, h6.horse_last_ran_days AS pr_5_horse_last_ran_days, h6.horse_form AS pr_5_horse_form,
        h6.country AS pr_5_country, h6.weather AS pr_5_weather, h6.race_class AS pr_5_race_class, h6.going_main AS pr_5_going_main, h6.going_grouped AS pr_5_going_grouped, h6.race_type AS pr_5_race_type, h6.race_type_devised AS pr_5_race_type_devised,
        h6.weather_grouped AS pr_5_weather_grouped,
        
        # past result 1 details
        h7.horse_time AS pr_6_horse_time, h7.betting_odds AS pr_6_betting_odds, h7.finish_position AS pr_6_finish_position, h7.did_not_finish AS pr_6_did_not_finish, h7.finish_position_for_ordering AS pr_6_finish_position_for_ordering,
        h7.course AS pr_6_course, h7.surface AS pr_6_surface, h7.going AS pr_6_going, h7.yards AS pr_6_yards, h7.runners AS pr_6_runners, h7.prize1 AS pr_6_prize1, h7.number_of_placed_rides AS pr_6_number_of_placed_rides,
		h7.handicap_pounds AS pr_6_handicap_pounds, h7.horse_age AS pr_6_horse_age, h7.horse_sex AS pr_6_horse_sex, h7.horse_last_ran_days AS pr_6_horse_last_ran_days, h7.horse_form AS pr_6_horse_form,
        h7.country AS pr_6_country, h7.weather AS pr_6_weather, h7.race_class AS pr_6_race_class, h7.going_main AS pr_6_going_main, h7.going_grouped AS pr_6_going_grouped, h7.race_type AS pr_6_race_type, h7.race_type_devised AS pr_6_race_type_devised,
        h7.weather_grouped AS pr_6_weather_grouped,
        
        # past result 1 details
        h8.horse_time AS pr_7_horse_time, h8.betting_odds AS pr_7_betting_odds, h8.finish_position AS pr_7_finish_position, h8.did_not_finish AS pr_7_did_not_finish, h8.finish_position_for_ordering AS pr_7_finish_position_for_ordering,
        h8.course AS pr_7_course, h8.surface AS pr_7_surface, h8.going AS pr_7_going, h8.yards AS pr_7_yards, h8.runners AS pr_7_runners, h8.prize1 AS pr_7_prize1, h8.number_of_placed_rides AS pr_7_number_of_placed_rides,
		h8.handicap_pounds AS pr_7_handicap_pounds, h8.horse_age AS pr_7_horse_age, h8.horse_sex AS pr_7_horse_sex, h8.horse_last_ran_days AS pr_7_horse_last_ran_days, h8.horse_form AS pr_7_horse_form,
        h8.country AS pr_7_country, h8.weather AS pr_7_weather, h8.race_class AS pr_7_race_class, h8.going_main AS pr_7_going_main, h8.going_grouped AS pr_7_going_grouped, h8.race_type AS pr_7_race_type, h8.race_type_devised AS pr_7_race_type_devised,
        h8.weather_grouped AS pr_7_weather_grouped,
        
        # past result 1 details
        h9.horse_time AS pr_8_horse_time, h9.betting_odds AS pr_8_betting_odds, h9.finish_position AS pr_8_finish_position, h9.did_not_finish AS pr_8_did_not_finish, h9.finish_position_for_ordering AS pr_8_finish_position_for_ordering,
        h9.course AS pr_8_course, h9.surface AS pr_8_surface, h9.going AS pr_8_going, h9.yards AS pr_8_yards, h9.runners AS pr_8_runners, h9.prize1 AS pr_8_prize1, h9.number_of_placed_rides AS pr_8_number_of_placed_rides,
		h9.handicap_pounds AS pr_8_handicap_pounds, h9.horse_age AS pr_8_horse_age, h9.horse_sex AS pr_8_horse_sex, h9.horse_last_ran_days AS pr_8_horse_last_ran_days, h9.horse_form AS pr_8_horse_form,
        h9.country AS pr_8_country, h9.weather AS pr_8_weather, h9.race_class AS pr_8_race_class, h9.going_main AS pr_8_going_main, h9.going_grouped AS pr_8_going_grouped, h9.race_type AS pr_8_race_type, h9.race_type_devised AS pr_8_race_type_devised,
        h9.weather_grouped AS pr_8_weather_grouped,
        
        # past result 1 details
        h10.horse_time AS pr_9_horse_time, h10.betting_odds AS pr_9_betting_odds, h10.finish_position AS pr_9_finish_position, h10.did_not_finish AS pr_9_did_not_finish, h10.finish_position_for_ordering AS pr_9_finish_position_for_ordering,
        h10.course AS pr_9_course, h10.surface AS pr_9_surface, h10.going AS pr_9_going, h10.yards AS pr_9_yards, h10.runners AS pr_9_runners, h10.prize1 AS pr_9_prize1, h10.number_of_placed_rides AS pr_9_number_of_placed_rides,
		h10.handicap_pounds AS pr_9_handicap_pounds, h10.horse_age AS pr_9_horse_age, h10.horse_sex AS pr_9_horse_sex, h10.horse_last_ran_days AS pr_9_horse_last_ran_days, h10.horse_form AS pr_9_horse_form,
        h10.country AS pr_9_country, h10.weather AS pr_9_weather, h10.race_class AS pr_9_race_class, h10.going_main AS pr_9_going_main, h10.going_grouped AS pr_9_going_grouped, h10.race_type AS pr_9_race_type, h10.race_type_devised AS pr_9_race_type_devised,
        h10.weather_grouped AS pr_9_weather_grouped,
        
        # past result 1 details
        h11.horse_time AS pr_10_horse_time, h11.betting_odds AS pr_10_betting_odds, h11.finish_position AS pr_10_finish_position, h11.did_not_finish AS pr_10_did_not_finish, h11.finish_position_for_ordering AS pr_10_finish_position_for_ordering,
        h11.course AS pr_10_course, h11.surface AS pr_10_surface, h11.going AS pr_10_going, h11.yards AS pr_10_yards, h11.runners AS pr_10_runners, h11.prize1 AS pr_10_prize1, h11.number_of_placed_rides AS pr_10_number_of_placed_rides,
		h11.handicap_pounds AS pr_10_handicap_pounds, h11.horse_age AS pr_10_horse_age, h11.horse_sex AS pr_10_horse_sex, h11.horse_last_ran_days AS pr_10_horse_last_ran_days, h11.horse_form AS pr_10_horse_form,
        h11.country AS pr_10_country, h11.weather AS pr_10_weather, h11.race_class AS pr_10_race_class, h11.going_main AS pr_10_going_main, h11.going_grouped AS pr_10_going_grouped, h11.race_type AS pr_10_race_type, h11.race_type_devised AS pr_10_race_type_devised,
        h11.weather_grouped AS pr_10_weather_grouped
        
FROM horses_data_combined_new h1
LEFT JOIN horses_data_combined h2 ON h1.horse_id = h2.horse_id AND h1.horse_race_number = h2.horse_race_number+1
LEFT JOIN horses_data_combined h3 ON h1.horse_id = h3.horse_id AND h1.horse_race_number = h3.horse_race_number+2
LEFT JOIN horses_data_combined h4 ON h1.horse_id = h4.horse_id AND h1.horse_race_number = h4.horse_race_number+3
LEFT JOIN horses_data_combined h5 ON h1.horse_id = h5.horse_id AND h1.horse_race_number = h5.horse_race_number+4
LEFT JOIN horses_data_combined h6 ON h1.horse_id = h6.horse_id AND h1.horse_race_number = h6.horse_race_number+5
LEFT JOIN horses_data_combined h7 ON h1.horse_id = h7.horse_id AND h1.horse_race_number = h7.horse_race_number+6
LEFT JOIN horses_data_combined h8 ON h1.horse_id = h8.horse_id AND h1.horse_race_number = h8.horse_race_number+7
LEFT JOIN horses_data_combined h9 ON h1.horse_id = h9.horse_id AND h1.horse_race_number = h9.horse_race_number+8
LEFT JOIN horses_data_combined h10 ON h1.horse_id = h10.horse_id AND h1.horse_race_number = h10.horse_race_number+9
LEFT JOIN horses_data_combined h11 ON h1.horse_id = h11.horse_id AND h1.horse_race_number = h11.horse_race_number+10
;

CREATE INDEX horse_id ON horses_data_combined_with_past_results_new(horse_id);
CREATE INDEX race_id ON horses_data_combined_with_past_results_new(race_id);


### AND FOR NO NON RUNNERS DATA ###
CREATE TABLE horses_data_combined_no_nrs_with_past_results_new AS
SELECT 
		# current race details
		h1.horse_id, h1.race_date, h1.race_id, h1.betting_odds, h1.finish_position, h1.horse_time, h1.did_not_finish, h1.finish_position_for_ordering,
        h1.course, h1.surface, h1.going, h1.yards, h1.runners, h1.prize1, h1.number_of_placed_rides,
		h1.handicap_pounds, h1.horse_age, h1.horse_sex, h1.horse_last_ran_days, h1.horse_form,
        h1.country, h1.weather, h1.race_class, h1.going_main, h1.going_grouped, h1.race_type, h1.race_type_devised, h1.weather_grouped,
        
        # past result 1 details
        h2.horse_time AS pr_1_horse_time, h2.betting_odds AS pr_1_betting_odds, h2.finish_position AS pr_1_finish_position, h2.did_not_finish AS pr_1_did_not_finish, h2.finish_position_for_ordering AS pr_1_finish_position_for_ordering,
        h2.course AS pr_1_course, h2.surface AS pr_1_surface, h2.going AS pr_1_going, h2.yards AS pr_1_yards, h2.runners AS pr_1_runners, h2.prize1 AS pr_1_prize1, h2.number_of_placed_rides AS pr_1_number_of_placed_rides,
		h2.handicap_pounds AS pr_1_handicap_pounds, h2.horse_age AS pr_1_horse_age, h2.horse_sex AS pr_1_horse_sex, h2.horse_last_ran_days AS pr_1_horse_last_ran_days, h2.horse_form AS pr_1_horse_form,
        h2.country AS pr_1_country, h2.weather AS pr_1_weather, h2.race_class AS pr_1_race_class, h2.going_main AS pr_1_going_main, h2.going_grouped AS pr_1_going_grouped, h2.race_type AS pr_1_race_type, h2.race_type_devised AS pr_1_race_type_devised,
        h2.weather_grouped AS pr_1_weather_grouped,
        
        # past result 1 details
        h3.horse_time AS pr_2_horse_time, h3.betting_odds AS pr_2_betting_odds, h3.finish_position AS pr_2_finish_position, h3.did_not_finish AS pr_2_did_not_finish, h3.finish_position_for_ordering AS pr_2_finish_position_for_ordering,
        h3.course AS pr_2_course, h3.surface AS pr_2_surface, h3.going AS pr_2_going, h3.yards AS pr_2_yards, h3.runners AS pr_2_runners, h3.prize1 AS pr_2_prize1, h3.number_of_placed_rides AS pr_2_number_of_placed_rides,
		h3.handicap_pounds AS pr_2_handicap_pounds, h3.horse_age AS pr_2_horse_age, h3.horse_sex AS pr_2_horse_sex, h3.horse_last_ran_days AS pr_2_horse_last_ran_days, h3.horse_form AS pr_2_horse_form,
        h3.country AS pr_2_country, h3.weather AS pr_2_weather, h3.race_class AS pr_2_race_class, h3.going_main AS pr_2_going_main, h3.going_grouped AS pr_2_going_grouped, h3.race_type AS pr_2_race_type, h3.race_type_devised AS pr_2_race_type_devised,
        h3.weather_grouped AS pr_2_weather_grouped,
        
        # past result 1 details
        h4.horse_time AS pr_3_horse_time, h4.betting_odds AS pr_3_betting_odds, h4.finish_position AS pr_3_finish_position, h4.did_not_finish AS pr_3_did_not_finish, h4.finish_position_for_ordering AS pr_3_finish_position_for_ordering,
        h4.course AS pr_3_course, h4.surface AS pr_3_surface, h4.going AS pr_3_going, h4.yards AS pr_3_yards, h4.runners AS pr_3_runners, h4.prize1 AS pr_3_prize1, h4.number_of_placed_rides AS pr_3_number_of_placed_rides,
		h4.handicap_pounds AS pr_3_handicap_pounds, h4.horse_age AS pr_3_horse_age, h4.horse_sex AS pr_3_horse_sex, h4.horse_last_ran_days AS pr_3_horse_last_ran_days, h4.horse_form AS pr_3_horse_form,
        h4.country AS pr_3_country, h4.weather AS pr_3_weather, h4.race_class AS pr_3_race_class, h4.going_main AS pr_3_going_main, h4.going_grouped AS pr_3_going_grouped, h4.race_type AS pr_3_race_type, h4.race_type_devised AS pr_3_race_type_devised,
        h4.weather_grouped AS pr_3_weather_grouped,
        
        # past result 1 details
        h5.horse_time AS pr_4_horse_time, h5.betting_odds AS pr_4_betting_odds, h5.finish_position AS pr_4_finish_position, h5.did_not_finish AS pr_4_did_not_finish, h5.finish_position_for_ordering AS pr_4_finish_position_for_ordering,
        h5.course AS pr_4_course, h5.surface AS pr_4_surface, h5.going AS pr_4_going, h5.yards AS pr_4_yards, h5.runners AS pr_4_runners, h5.prize1 AS pr_4_prize1, h5.number_of_placed_rides AS pr_4_number_of_placed_rides,
		h5.handicap_pounds AS pr_4_handicap_pounds, h5.horse_age AS pr_4_horse_age, h5.horse_sex AS pr_4_horse_sex, h5.horse_last_ran_days AS pr_4_horse_last_ran_days, h5.horse_form AS pr_4_horse_form,
        h5.country AS pr_4_country, h5.weather AS pr_4_weather, h5.race_class AS pr_4_race_class, h5.going_main AS pr_4_going_main, h5.going_grouped AS pr_4_going_grouped, h5.race_type AS pr_4_race_type, h5.race_type_devised AS pr_4_race_type_devised,
        h5.weather_grouped AS pr_4_weather_grouped,
        
        # past result 1 details
        h6.horse_time AS pr_5_horse_time, h6.betting_odds AS pr_5_betting_odds, h6.finish_position AS pr_5_finish_position, h6.did_not_finish AS pr_5_did_not_finish, h6.finish_position_for_ordering AS pr_5_finish_position_for_ordering,
        h6.course AS pr_5_course, h6.surface AS pr_5_surface, h6.going AS pr_5_going, h6.yards AS pr_5_yards, h6.runners AS pr_5_runners, h6.prize1 AS pr_5_prize1, h6.number_of_placed_rides AS pr_5_number_of_placed_rides,
		h6.handicap_pounds AS pr_5_handicap_pounds, h6.horse_age AS pr_5_horse_age, h6.horse_sex AS pr_5_horse_sex, h6.horse_last_ran_days AS pr_5_horse_last_ran_days, h6.horse_form AS pr_5_horse_form,
        h6.country AS pr_5_country, h6.weather AS pr_5_weather, h6.race_class AS pr_5_race_class, h6.going_main AS pr_5_going_main, h6.going_grouped AS pr_5_going_grouped, h6.race_type AS pr_5_race_type, h6.race_type_devised AS pr_5_race_type_devised,
        h6.weather_grouped AS pr_5_weather_grouped,
        
        # past result 1 details
        h7.horse_time AS pr_6_horse_time, h7.betting_odds AS pr_6_betting_odds, h7.finish_position AS pr_6_finish_position, h7.did_not_finish AS pr_6_did_not_finish, h7.finish_position_for_ordering AS pr_6_finish_position_for_ordering,
        h7.course AS pr_6_course, h7.surface AS pr_6_surface, h7.going AS pr_6_going, h7.yards AS pr_6_yards, h7.runners AS pr_6_runners, h7.prize1 AS pr_6_prize1, h7.number_of_placed_rides AS pr_6_number_of_placed_rides,
		h7.handicap_pounds AS pr_6_handicap_pounds, h7.horse_age AS pr_6_horse_age, h7.horse_sex AS pr_6_horse_sex, h7.horse_last_ran_days AS pr_6_horse_last_ran_days, h7.horse_form AS pr_6_horse_form,
        h7.country AS pr_6_country, h7.weather AS pr_6_weather, h7.race_class AS pr_6_race_class, h7.going_main AS pr_6_going_main, h7.going_grouped AS pr_6_going_grouped, h7.race_type AS pr_6_race_type, h7.race_type_devised AS pr_6_race_type_devised,
        h7.weather_grouped AS pr_6_weather_grouped,
        
        # past result 1 details
        h8.horse_time AS pr_7_horse_time, h8.betting_odds AS pr_7_betting_odds, h8.finish_position AS pr_7_finish_position, h8.did_not_finish AS pr_7_did_not_finish, h8.finish_position_for_ordering AS pr_7_finish_position_for_ordering,
        h8.course AS pr_7_course, h8.surface AS pr_7_surface, h8.going AS pr_7_going, h8.yards AS pr_7_yards, h8.runners AS pr_7_runners, h8.prize1 AS pr_7_prize1, h8.number_of_placed_rides AS pr_7_number_of_placed_rides,
		h8.handicap_pounds AS pr_7_handicap_pounds, h8.horse_age AS pr_7_horse_age, h8.horse_sex AS pr_7_horse_sex, h8.horse_last_ran_days AS pr_7_horse_last_ran_days, h8.horse_form AS pr_7_horse_form,
        h8.country AS pr_7_country, h8.weather AS pr_7_weather, h8.race_class AS pr_7_race_class, h8.going_main AS pr_7_going_main, h8.going_grouped AS pr_7_going_grouped, h8.race_type AS pr_7_race_type, h8.race_type_devised AS pr_7_race_type_devised,
        h8.weather_grouped AS pr_7_weather_grouped,
        
        # past result 1 details
        h9.horse_time AS pr_8_horse_time, h9.betting_odds AS pr_8_betting_odds, h9.finish_position AS pr_8_finish_position, h9.did_not_finish AS pr_8_did_not_finish, h9.finish_position_for_ordering AS pr_8_finish_position_for_ordering,
        h9.course AS pr_8_course, h9.surface AS pr_8_surface, h9.going AS pr_8_going, h9.yards AS pr_8_yards, h9.runners AS pr_8_runners, h9.prize1 AS pr_8_prize1, h9.number_of_placed_rides AS pr_8_number_of_placed_rides,
		h9.handicap_pounds AS pr_8_handicap_pounds, h9.horse_age AS pr_8_horse_age, h9.horse_sex AS pr_8_horse_sex, h9.horse_last_ran_days AS pr_8_horse_last_ran_days, h9.horse_form AS pr_8_horse_form,
        h9.country AS pr_8_country, h9.weather AS pr_8_weather, h9.race_class AS pr_8_race_class, h9.going_main AS pr_8_going_main, h9.going_grouped AS pr_8_going_grouped, h9.race_type AS pr_8_race_type, h9.race_type_devised AS pr_8_race_type_devised,
        h9.weather_grouped AS pr_8_weather_grouped,
        
        # past result 1 details
        h10.horse_time AS pr_9_horse_time, h10.betting_odds AS pr_9_betting_odds, h10.finish_position AS pr_9_finish_position, h10.did_not_finish AS pr_9_did_not_finish, h10.finish_position_for_ordering AS pr_9_finish_position_for_ordering,
        h10.course AS pr_9_course, h10.surface AS pr_9_surface, h10.going AS pr_9_going, h10.yards AS pr_9_yards, h10.runners AS pr_9_runners, h10.prize1 AS pr_9_prize1, h10.number_of_placed_rides AS pr_9_number_of_placed_rides,
		h10.handicap_pounds AS pr_9_handicap_pounds, h10.horse_age AS pr_9_horse_age, h10.horse_sex AS pr_9_horse_sex, h10.horse_last_ran_days AS pr_9_horse_last_ran_days, h10.horse_form AS pr_9_horse_form,
        h10.country AS pr_9_country, h10.weather AS pr_9_weather, h10.race_class AS pr_9_race_class, h10.going_main AS pr_9_going_main, h10.going_grouped AS pr_9_going_grouped, h10.race_type AS pr_9_race_type, h10.race_type_devised AS pr_9_race_type_devised,
        h10.weather_grouped AS pr_9_weather_grouped,
        
        # past result 1 details
        h11.horse_time AS pr_10_horse_time, h11.betting_odds AS pr_10_betting_odds, h11.finish_position AS pr_10_finish_position, h11.did_not_finish AS pr_10_did_not_finish, h11.finish_position_for_ordering AS pr_10_finish_position_for_ordering,
        h11.course AS pr_10_course, h11.surface AS pr_10_surface, h11.going AS pr_10_going, h11.yards AS pr_10_yards, h11.runners AS pr_10_runners, h11.prize1 AS pr_10_prize1, h11.number_of_placed_rides AS pr_10_number_of_placed_rides,
		h11.handicap_pounds AS pr_10_handicap_pounds, h11.horse_age AS pr_10_horse_age, h11.horse_sex AS pr_10_horse_sex, h11.horse_last_ran_days AS pr_10_horse_last_ran_days, h11.horse_form AS pr_10_horse_form,
        h11.country AS pr_10_country, h11.weather AS pr_10_weather, h11.race_class AS pr_10_race_class, h11.going_main AS pr_10_going_main, h11.going_grouped AS pr_10_going_grouped, h11.race_type AS pr_10_race_type, h11.race_type_devised AS pr_10_race_type_devised,
        h11.weather_grouped AS pr_10_weather_grouped
        
FROM horses_data_combined_no_non_runners_new h1
LEFT JOIN horses_data_combined_no_non_runners h2 ON h1.horse_id = h2.horse_id AND h1.horse_race_number = h2.horse_race_number+1
LEFT JOIN horses_data_combined_no_non_runners h3 ON h1.horse_id = h3.horse_id AND h1.horse_race_number = h3.horse_race_number+2
LEFT JOIN horses_data_combined_no_non_runners h4 ON h1.horse_id = h4.horse_id AND h1.horse_race_number = h4.horse_race_number+3
LEFT JOIN horses_data_combined_no_non_runners h5 ON h1.horse_id = h5.horse_id AND h1.horse_race_number = h5.horse_race_number+4
LEFT JOIN horses_data_combined_no_non_runners h6 ON h1.horse_id = h6.horse_id AND h1.horse_race_number = h6.horse_race_number+5
LEFT JOIN horses_data_combined_no_non_runners h7 ON h1.horse_id = h7.horse_id AND h1.horse_race_number = h7.horse_race_number+6
LEFT JOIN horses_data_combined_no_non_runners h8 ON h1.horse_id = h8.horse_id AND h1.horse_race_number = h8.horse_race_number+7
LEFT JOIN horses_data_combined_no_non_runners h9 ON h1.horse_id = h9.horse_id AND h1.horse_race_number = h9.horse_race_number+8
LEFT JOIN horses_data_combined_no_non_runners h10 ON h1.horse_id = h10.horse_id AND h1.horse_race_number = h10.horse_race_number+9
LEFT JOIN horses_data_combined_no_non_runners h11 ON h1.horse_id = h11.horse_id AND h1.horse_race_number = h11.horse_race_number+10
;

CREATE INDEX horse_id ON horses_data_combined_no_nrs_with_past_results_new(horse_id);
CREATE INDEX race_id ON horses_data_combined_no_nrs_with_past_results_new(race_id);


### ADD INTO EXISTING TABLES ###
INSERT INTO horses_data_combined_with_past_results SELECT * FROM horses_data_combined_with_past_results_new;
INSERT INTO horses_data_combined_no_nrs_with_past_results SELECT * FROM horses_data_combined_no_nrs_with_past_results_new;
INSERT INTO horses_data SELECT * FROM horses_data_new;
INSERT INTO races_data SELECT * FROM races_data_new;

SELECT horse_id, race_id, COUNT(1) AS count FROM horses_data_combined_with_past_results GROUP BY horse_id, race_id ORDER BY count DESC LIMIT 10;


SELECT COUNT(1), SUM(CASE WHEN runners = horses_in_data THEN 1 ELSE 0 END)
FROM (
SELECT race_id, runners, COUNT(1) AS horses_in_data
FROM horses_data_combined
WHERE race_date > '2019-09-22'
GROUP BY race_id, runners) t;

SELECT COUNT(1) FROM horses_data_combined WHERE horse_time IS NOT NULL AND did_not_finish=0;
SELECT COUNT(1) FROM horses_data WHERE horse_time IS NOT NULL;
