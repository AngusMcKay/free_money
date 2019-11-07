USE horses;

SHOW FULL PROCESSLIST;
KILL 13;

# get basic data for initial analysis
SELECT AVG(ratio), STD(ratio)
FROM (
SELECT hd1.race_date rd1, hd2.race_date rd2, ht1.horse_time ht1, ht2.horse_time ht2, ht1.horse_time/ht2.horse_time AS ratio
FROM (SELECT * FROM horses_data WHERE winning_time_seconds IS NOT NULL LIMIT 10000) hd1
LEFT JOIN horses_data hd2 ON hd1.horse_id = hd2.horse_id AND hd1.race_id <> hd2.race_id
LEFT JOIN races_goings rg1 ON hd1.race_id = rg1.race_id
LEFT JOIN races_goings rg2 ON hd2.race_id = rg2.race_id
LEFT JOIN horses_times ht1 ON hd1.horse_id = ht1.horse_id AND hd1.race_id = ht1.race_id
LEFT JOIN horses_times ht2 ON hd2.horse_id = ht2.horse_id AND hd2.race_id = ht2.race_id
LEFT JOIN horses_handicaps hh1 ON hd1.horse_id = hh1.horse_id AND hd1.race_id = hh1.race_id
LEFT JOIN horses_handicaps hh2 ON hd2.horse_id = hh2.horse_id AND hd2.race_id = hh2.race_id
WHERE 1
AND hd1.course = hd2.course
AND hd1.surface = hd2.surface
AND rg1.going_grouped = rg2.going_grouped
AND hd1.yards = hd2.yards
#AND hd1.yards BETWEEN 0.9*hd2.yards AND 1.1*hd2.yards
AND hd1.runners = hd2.runners
AND hh1.handicap_pounds = hh2.handicap_pounds
#AND hh1.handicap_pounds BETWEEN 0.9*hh2.handicap_pounds AND 1.1*hh2.handicap_pounds
LIMIT 1000
) t
;

# check if enough horses per race - is below because of duplicates
SELECT *
FROM
(
SELECT race_id, COUNT(1) c, MAX(runners) r
FROM horses_data
WHERE ride_status IN ('RUNNER','DOUBTFUL')
AND race_id IN (SELECT race_id FROM races_data WHERE country IN ('scotland', 'england', 'northern ireland', 'eire', 'wales'))
GROUP BY race_id
) t
WHERE c < r
LIMIT 1000
;

SELECT * FROM horses_data WHERE race_id = 947
;

SELECT COUNT(1)
FROM horses_data_combined_with_past_results
WHERE 1
AND horse_time IS NOT NULL
;

SELECT COUNT(DISTINCT race_id, horse_id)
FROM horses_data_combined_with_past_results
;



# times prediction data, first model
CREATE TABLE training_data_5_pr AS
SELECT h.*,
		weather_horse_time_rc, pr_1_weather_horse_time_rc, pr_2_weather_horse_time_rc, pr_3_weather_horse_time_rc, pr_4_weather_horse_time_rc, pr_5_weather_horse_time_rc,
        going_grouped_horse_time_rc, pr_1_going_grouped_horse_time_rc, pr_2_going_grouped_horse_time_rc, pr_3_going_grouped_horse_time_rc, pr_4_going_grouped_horse_time_rc, pr_5_going_grouped_horse_time_rc,
        race_type_horse_time_rc, pr_1_race_type_horse_time_rc, pr_2_race_type_horse_time_rc, pr_3_race_type_horse_time_rc, pr_4_race_type_horse_time_rc, pr_5_race_type_horse_time_rc
FROM horses_data_combined_with_past_results h
# join response coded variables
LEFT JOIN (SELECT weather_grouped, AVG(horse_time) AS weather_horse_time_rc FROM horses_data_combined_with_past_results GROUP BY weather_grouped) w ON h.weather_grouped = w.weather_grouped
LEFT JOIN (SELECT pr_1_weather_grouped, AVG(horse_time) AS pr_1_weather_horse_time_rc FROM horses_data_combined_with_past_results GROUP BY pr_1_weather_grouped) w1 ON h.pr_1_weather_grouped = w1.pr_1_weather_grouped
LEFT JOIN (SELECT pr_2_weather_grouped, AVG(horse_time) AS pr_2_weather_horse_time_rc FROM horses_data_combined_with_past_results GROUP BY pr_2_weather_grouped) w2 ON h.pr_2_weather_grouped = w2.pr_2_weather_grouped
LEFT JOIN (SELECT pr_3_weather_grouped, AVG(horse_time) AS pr_3_weather_horse_time_rc FROM horses_data_combined_with_past_results GROUP BY pr_3_weather_grouped) w3 ON h.pr_3_weather_grouped = w3.pr_3_weather_grouped
LEFT JOIN (SELECT pr_4_weather_grouped, AVG(horse_time) AS pr_4_weather_horse_time_rc FROM horses_data_combined_with_past_results GROUP BY pr_4_weather_grouped) w4 ON h.pr_4_weather_grouped = w4.pr_4_weather_grouped
LEFT JOIN (SELECT pr_5_weather_grouped, AVG(horse_time) AS pr_5_weather_horse_time_rc FROM horses_data_combined_with_past_results GROUP BY pr_5_weather_grouped) w5 ON h.pr_5_weather_grouped = w5.pr_5_weather_grouped

LEFT JOIN (SELECT going_grouped, AVG(horse_time) AS going_grouped_horse_time_rc FROM horses_data_combined_with_past_results GROUP BY going_grouped) g ON h.going_grouped = g.going_grouped
LEFT JOIN (SELECT pr_1_going_grouped, AVG(horse_time) AS pr_1_going_grouped_horse_time_rc FROM horses_data_combined_with_past_results GROUP BY pr_1_going_grouped) g1 ON h.pr_1_going_grouped = g1.pr_1_going_grouped
LEFT JOIN (SELECT pr_2_going_grouped, AVG(horse_time) AS pr_2_going_grouped_horse_time_rc FROM horses_data_combined_with_past_results GROUP BY pr_2_going_grouped) g2 ON h.pr_2_going_grouped = g2.pr_2_going_grouped
LEFT JOIN (SELECT pr_3_going_grouped, AVG(horse_time) AS pr_3_going_grouped_horse_time_rc FROM horses_data_combined_with_past_results GROUP BY pr_3_going_grouped) g3 ON h.pr_3_going_grouped = g3.pr_3_going_grouped
LEFT JOIN (SELECT pr_4_going_grouped, AVG(horse_time) AS pr_4_going_grouped_horse_time_rc FROM horses_data_combined_with_past_results GROUP BY pr_4_going_grouped) g4 ON h.pr_4_going_grouped = g4.pr_4_going_grouped
LEFT JOIN (SELECT pr_5_going_grouped, AVG(horse_time) AS pr_5_going_grouped_horse_time_rc FROM horses_data_combined_with_past_results GROUP BY pr_5_going_grouped) g5 ON h.pr_5_going_grouped = g5.pr_5_going_grouped

LEFT JOIN (SELECT race_type, AVG(horse_time) AS race_type_horse_time_rc FROM horses_data_combined_with_past_results GROUP BY race_type) rt ON h.race_type = rt.race_type
LEFT JOIN (SELECT pr_1_race_type, AVG(horse_time) AS pr_1_race_type_horse_time_rc FROM horses_data_combined_with_past_results GROUP BY pr_1_race_type) rt1 ON h.pr_1_race_type = rt1.pr_1_race_type
LEFT JOIN (SELECT pr_2_race_type, AVG(horse_time) AS pr_2_race_type_horse_time_rc FROM horses_data_combined_with_past_results GROUP BY pr_2_race_type) rt2 ON h.pr_2_race_type = rt2.pr_2_race_type
LEFT JOIN (SELECT pr_3_race_type, AVG(horse_time) AS pr_3_race_type_horse_time_rc FROM horses_data_combined_with_past_results GROUP BY pr_3_race_type) rt3 ON h.pr_3_race_type = rt3.pr_3_race_type
LEFT JOIN (SELECT pr_4_race_type, AVG(horse_time) AS pr_4_race_type_horse_time_rc FROM horses_data_combined_with_past_results GROUP BY pr_4_race_type) rt4 ON h.pr_4_race_type = rt4.pr_4_race_type
LEFT JOIN (SELECT pr_5_race_type, AVG(horse_time) AS pr_5_race_type_horse_time_rc FROM horses_data_combined_with_past_results GROUP BY pr_5_race_type) rt5 ON h.pr_5_race_type = rt5.pr_5_race_type

WHERE 1
AND horse_time IS NOT NULL
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
AND LOWER(country) IN ('scotland', 'england', 'northern ireland', 'eire', 'wales')
#AND race_date > '2018-06-01'
#LIMIT 100
;


CREATE TABLE training_data_10_pr AS
SELECT t1.*,
		pr_6_weather_horse_time_rc, pr_7_weather_horse_time_rc, pr_8_weather_horse_time_rc, pr_9_weather_horse_time_rc, pr_10_weather_horse_time_rc,
        pr_6_going_grouped_horse_time_rc, pr_7_going_grouped_horse_time_rc, pr_8_going_grouped_horse_time_rc, pr_9_going_grouped_horse_time_rc, pr_10_going_grouped_horse_time_rc,
        pr_6_race_type_horse_time_rc, pr_7_race_type_horse_time_rc, pr_8_race_type_horse_time_rc, pr_9_race_type_horse_time_rc, pr_10_race_type_horse_time_rc

FROM training_data_5_pr t1
# join response coded variables for prs 6 to 10
LEFT JOIN (SELECT pr_6_weather_grouped, AVG(horse_time) AS pr_6_weather_horse_time_rc FROM horses_data_combined_with_past_results GROUP BY pr_6_weather_grouped) w1 ON t1.pr_6_weather_grouped = w1.pr_6_weather_grouped
LEFT JOIN (SELECT pr_7_weather_grouped, AVG(horse_time) AS pr_7_weather_horse_time_rc FROM horses_data_combined_with_past_results GROUP BY pr_7_weather_grouped) w2 ON t1.pr_7_weather_grouped = w2.pr_7_weather_grouped
LEFT JOIN (SELECT pr_8_weather_grouped, AVG(horse_time) AS pr_8_weather_horse_time_rc FROM horses_data_combined_with_past_results GROUP BY pr_8_weather_grouped) w3 ON t1.pr_8_weather_grouped = w3.pr_8_weather_grouped
LEFT JOIN (SELECT pr_9_weather_grouped, AVG(horse_time) AS pr_9_weather_horse_time_rc FROM horses_data_combined_with_past_results GROUP BY pr_9_weather_grouped) w4 ON t1.pr_9_weather_grouped = w4.pr_9_weather_grouped
LEFT JOIN (SELECT pr_10_weather_grouped, AVG(horse_time) AS pr_10_weather_horse_time_rc FROM horses_data_combined_with_past_results GROUP BY pr_10_weather_grouped) w5 ON t1.pr_10_weather_grouped = w5.pr_10_weather_grouped

LEFT JOIN (SELECT pr_6_going_grouped, AVG(horse_time) AS pr_6_going_grouped_horse_time_rc FROM horses_data_combined_with_past_results GROUP BY pr_6_going_grouped) g1 ON t1.pr_6_going_grouped = g1.pr_6_going_grouped
LEFT JOIN (SELECT pr_7_going_grouped, AVG(horse_time) AS pr_7_going_grouped_horse_time_rc FROM horses_data_combined_with_past_results GROUP BY pr_7_going_grouped) g2 ON t1.pr_7_going_grouped = g2.pr_7_going_grouped
LEFT JOIN (SELECT pr_8_going_grouped, AVG(horse_time) AS pr_8_going_grouped_horse_time_rc FROM horses_data_combined_with_past_results GROUP BY pr_8_going_grouped) g3 ON t1.pr_8_going_grouped = g3.pr_8_going_grouped
LEFT JOIN (SELECT pr_9_going_grouped, AVG(horse_time) AS pr_9_going_grouped_horse_time_rc FROM horses_data_combined_with_past_results GROUP BY pr_9_going_grouped) g4 ON t1.pr_9_going_grouped = g4.pr_9_going_grouped
LEFT JOIN (SELECT pr_10_going_grouped, AVG(horse_time) AS pr_10_going_grouped_horse_time_rc FROM horses_data_combined_with_past_results GROUP BY pr_10_going_grouped) g5 ON t1.pr_10_going_grouped = g5.pr_10_going_grouped

LEFT JOIN (SELECT pr_6_race_type, AVG(horse_time) AS pr_6_race_type_horse_time_rc FROM horses_data_combined_with_past_results GROUP BY pr_6_race_type) rt1 ON t1.pr_6_race_type = rt1.pr_6_race_type
LEFT JOIN (SELECT pr_7_race_type, AVG(horse_time) AS pr_7_race_type_horse_time_rc FROM horses_data_combined_with_past_results GROUP BY pr_7_race_type) rt2 ON t1.pr_7_race_type = rt2.pr_7_race_type
LEFT JOIN (SELECT pr_8_race_type, AVG(horse_time) AS pr_8_race_type_horse_time_rc FROM horses_data_combined_with_past_results GROUP BY pr_8_race_type) rt3 ON t1.pr_8_race_type = rt3.pr_8_race_type
LEFT JOIN (SELECT pr_9_race_type, AVG(horse_time) AS pr_9_race_type_horse_time_rc FROM horses_data_combined_with_past_results GROUP BY pr_9_race_type) rt4 ON t1.pr_9_race_type = rt4.pr_9_race_type
LEFT JOIN (SELECT pr_10_race_type, AVG(horse_time) AS pr_10_race_type_horse_time_rc FROM horses_data_combined_with_past_results GROUP BY pr_10_race_type) rt5 ON t1.pr_10_race_type = rt5.pr_10_race_type

WHERE 1
AND LOWER(country) IN ('scotland', 'england', 'northern ireland', 'eire', 'wales')
#AND race_date > '2018-06-01'
#LIMIT 100
;


SELECT COUNT(1) FROM training_data_10_pr
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
;



# no non-runners training data
CREATE TABLE training_data_5_pr_no_nrs AS
SELECT h.*,
		weather_horse_time_rc, pr_1_weather_horse_time_rc, pr_2_weather_horse_time_rc, pr_3_weather_horse_time_rc, pr_4_weather_horse_time_rc, pr_5_weather_horse_time_rc,
        going_grouped_horse_time_rc, pr_1_going_grouped_horse_time_rc, pr_2_going_grouped_horse_time_rc, pr_3_going_grouped_horse_time_rc, pr_4_going_grouped_horse_time_rc, pr_5_going_grouped_horse_time_rc,
        race_type_horse_time_rc, pr_1_race_type_horse_time_rc, pr_2_race_type_horse_time_rc, pr_3_race_type_horse_time_rc, pr_4_race_type_horse_time_rc, pr_5_race_type_horse_time_rc
FROM horses_data_combined_no_nrs_with_past_results h
# join response coded variables
LEFT JOIN (SELECT weather_grouped, AVG(horse_time) AS weather_horse_time_rc FROM horses_data_combined_no_nrs_with_past_results GROUP BY weather_grouped) w ON h.weather_grouped = w.weather_grouped
LEFT JOIN (SELECT pr_1_weather_grouped, AVG(horse_time) AS pr_1_weather_horse_time_rc FROM horses_data_combined_no_nrs_with_past_results GROUP BY pr_1_weather_grouped) w1 ON h.pr_1_weather_grouped = w1.pr_1_weather_grouped
LEFT JOIN (SELECT pr_2_weather_grouped, AVG(horse_time) AS pr_2_weather_horse_time_rc FROM horses_data_combined_no_nrs_with_past_results GROUP BY pr_2_weather_grouped) w2 ON h.pr_2_weather_grouped = w2.pr_2_weather_grouped
LEFT JOIN (SELECT pr_3_weather_grouped, AVG(horse_time) AS pr_3_weather_horse_time_rc FROM horses_data_combined_no_nrs_with_past_results GROUP BY pr_3_weather_grouped) w3 ON h.pr_3_weather_grouped = w3.pr_3_weather_grouped
LEFT JOIN (SELECT pr_4_weather_grouped, AVG(horse_time) AS pr_4_weather_horse_time_rc FROM horses_data_combined_no_nrs_with_past_results GROUP BY pr_4_weather_grouped) w4 ON h.pr_4_weather_grouped = w4.pr_4_weather_grouped
LEFT JOIN (SELECT pr_5_weather_grouped, AVG(horse_time) AS pr_5_weather_horse_time_rc FROM horses_data_combined_no_nrs_with_past_results GROUP BY pr_5_weather_grouped) w5 ON h.pr_5_weather_grouped = w5.pr_5_weather_grouped

LEFT JOIN (SELECT going_grouped, AVG(horse_time) AS going_grouped_horse_time_rc FROM horses_data_combined_no_nrs_with_past_results GROUP BY going_grouped) g ON h.going_grouped = g.going_grouped
LEFT JOIN (SELECT pr_1_going_grouped, AVG(horse_time) AS pr_1_going_grouped_horse_time_rc FROM horses_data_combined_no_nrs_with_past_results GROUP BY pr_1_going_grouped) g1 ON h.pr_1_going_grouped = g1.pr_1_going_grouped
LEFT JOIN (SELECT pr_2_going_grouped, AVG(horse_time) AS pr_2_going_grouped_horse_time_rc FROM horses_data_combined_no_nrs_with_past_results GROUP BY pr_2_going_grouped) g2 ON h.pr_2_going_grouped = g2.pr_2_going_grouped
LEFT JOIN (SELECT pr_3_going_grouped, AVG(horse_time) AS pr_3_going_grouped_horse_time_rc FROM horses_data_combined_no_nrs_with_past_results GROUP BY pr_3_going_grouped) g3 ON h.pr_3_going_grouped = g3.pr_3_going_grouped
LEFT JOIN (SELECT pr_4_going_grouped, AVG(horse_time) AS pr_4_going_grouped_horse_time_rc FROM horses_data_combined_no_nrs_with_past_results GROUP BY pr_4_going_grouped) g4 ON h.pr_4_going_grouped = g4.pr_4_going_grouped
LEFT JOIN (SELECT pr_5_going_grouped, AVG(horse_time) AS pr_5_going_grouped_horse_time_rc FROM horses_data_combined_no_nrs_with_past_results GROUP BY pr_5_going_grouped) g5 ON h.pr_5_going_grouped = g5.pr_5_going_grouped

LEFT JOIN (SELECT race_type, AVG(horse_time) AS race_type_horse_time_rc FROM horses_data_combined_no_nrs_with_past_results GROUP BY race_type) rt ON h.race_type = rt.race_type
LEFT JOIN (SELECT pr_1_race_type, AVG(horse_time) AS pr_1_race_type_horse_time_rc FROM horses_data_combined_no_nrs_with_past_results GROUP BY pr_1_race_type) rt1 ON h.pr_1_race_type = rt1.pr_1_race_type
LEFT JOIN (SELECT pr_2_race_type, AVG(horse_time) AS pr_2_race_type_horse_time_rc FROM horses_data_combined_no_nrs_with_past_results GROUP BY pr_2_race_type) rt2 ON h.pr_2_race_type = rt2.pr_2_race_type
LEFT JOIN (SELECT pr_3_race_type, AVG(horse_time) AS pr_3_race_type_horse_time_rc FROM horses_data_combined_no_nrs_with_past_results GROUP BY pr_3_race_type) rt3 ON h.pr_3_race_type = rt3.pr_3_race_type
LEFT JOIN (SELECT pr_4_race_type, AVG(horse_time) AS pr_4_race_type_horse_time_rc FROM horses_data_combined_no_nrs_with_past_results GROUP BY pr_4_race_type) rt4 ON h.pr_4_race_type = rt4.pr_4_race_type
LEFT JOIN (SELECT pr_5_race_type, AVG(horse_time) AS pr_5_race_type_horse_time_rc FROM horses_data_combined_no_nrs_with_past_results GROUP BY pr_5_race_type) rt5 ON h.pr_5_race_type = rt5.pr_5_race_type

WHERE 1
AND horse_time IS NOT NULL
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
AND LOWER(country) IN ('scotland', 'england', 'northern ireland', 'eire', 'wales')
#AND race_date > '2018-06-01'
#LIMIT 100
;

DROP TABLE training_data_10_pr_no_nrs;
CREATE TABLE training_data_10_pr_no_nrs AS
SELECT t1.*,
		pr_6_weather_horse_time_rc, pr_7_weather_horse_time_rc, pr_8_weather_horse_time_rc, pr_9_weather_horse_time_rc, pr_10_weather_horse_time_rc,
        pr_6_going_grouped_horse_time_rc, pr_7_going_grouped_horse_time_rc, pr_8_going_grouped_horse_time_rc, pr_9_going_grouped_horse_time_rc, pr_10_going_grouped_horse_time_rc,
        pr_6_race_type_horse_time_rc, pr_7_race_type_horse_time_rc, pr_8_race_type_horse_time_rc, pr_9_race_type_horse_time_rc, pr_10_race_type_horse_time_rc

FROM training_data_5_pr_no_nrs t1
# join response coded variables for prs 6 to 10
LEFT JOIN (SELECT pr_6_weather_grouped, AVG(horse_time) AS pr_6_weather_horse_time_rc FROM horses_data_combined_no_nrs_with_past_results GROUP BY pr_6_weather_grouped) w1 ON t1.pr_6_weather_grouped = w1.pr_6_weather_grouped
LEFT JOIN (SELECT pr_7_weather_grouped, AVG(horse_time) AS pr_7_weather_horse_time_rc FROM horses_data_combined_no_nrs_with_past_results GROUP BY pr_7_weather_grouped) w2 ON t1.pr_7_weather_grouped = w2.pr_7_weather_grouped
LEFT JOIN (SELECT pr_8_weather_grouped, AVG(horse_time) AS pr_8_weather_horse_time_rc FROM horses_data_combined_no_nrs_with_past_results GROUP BY pr_8_weather_grouped) w3 ON t1.pr_8_weather_grouped = w3.pr_8_weather_grouped
LEFT JOIN (SELECT pr_9_weather_grouped, AVG(horse_time) AS pr_9_weather_horse_time_rc FROM horses_data_combined_no_nrs_with_past_results GROUP BY pr_9_weather_grouped) w4 ON t1.pr_9_weather_grouped = w4.pr_9_weather_grouped
LEFT JOIN (SELECT pr_10_weather_grouped, AVG(horse_time) AS pr_10_weather_horse_time_rc FROM horses_data_combined_no_nrs_with_past_results GROUP BY pr_10_weather_grouped) w5 ON t1.pr_10_weather_grouped = w5.pr_10_weather_grouped

LEFT JOIN (SELECT pr_6_going_grouped, AVG(horse_time) AS pr_6_going_grouped_horse_time_rc FROM horses_data_combined_no_nrs_with_past_results GROUP BY pr_6_going_grouped) g1 ON t1.pr_6_going_grouped = g1.pr_6_going_grouped
LEFT JOIN (SELECT pr_7_going_grouped, AVG(horse_time) AS pr_7_going_grouped_horse_time_rc FROM horses_data_combined_no_nrs_with_past_results GROUP BY pr_7_going_grouped) g2 ON t1.pr_7_going_grouped = g2.pr_7_going_grouped
LEFT JOIN (SELECT pr_8_going_grouped, AVG(horse_time) AS pr_8_going_grouped_horse_time_rc FROM horses_data_combined_no_nrs_with_past_results GROUP BY pr_8_going_grouped) g3 ON t1.pr_8_going_grouped = g3.pr_8_going_grouped
LEFT JOIN (SELECT pr_9_going_grouped, AVG(horse_time) AS pr_9_going_grouped_horse_time_rc FROM horses_data_combined_no_nrs_with_past_results GROUP BY pr_9_going_grouped) g4 ON t1.pr_9_going_grouped = g4.pr_9_going_grouped
LEFT JOIN (SELECT pr_10_going_grouped, AVG(horse_time) AS pr_10_going_grouped_horse_time_rc FROM horses_data_combined_no_nrs_with_past_results GROUP BY pr_10_going_grouped) g5 ON t1.pr_10_going_grouped = g5.pr_10_going_grouped

LEFT JOIN (SELECT pr_6_race_type, AVG(horse_time) AS pr_6_race_type_horse_time_rc FROM horses_data_combined_no_nrs_with_past_results GROUP BY pr_6_race_type) rt1 ON t1.pr_6_race_type = rt1.pr_6_race_type
LEFT JOIN (SELECT pr_7_race_type, AVG(horse_time) AS pr_7_race_type_horse_time_rc FROM horses_data_combined_no_nrs_with_past_results GROUP BY pr_7_race_type) rt2 ON t1.pr_7_race_type = rt2.pr_7_race_type
LEFT JOIN (SELECT pr_8_race_type, AVG(horse_time) AS pr_8_race_type_horse_time_rc FROM horses_data_combined_no_nrs_with_past_results GROUP BY pr_8_race_type) rt3 ON t1.pr_8_race_type = rt3.pr_8_race_type
LEFT JOIN (SELECT pr_9_race_type, AVG(horse_time) AS pr_9_race_type_horse_time_rc FROM horses_data_combined_no_nrs_with_past_results GROUP BY pr_9_race_type) rt4 ON t1.pr_9_race_type = rt4.pr_9_race_type
LEFT JOIN (SELECT pr_10_race_type, AVG(horse_time) AS pr_10_race_type_horse_time_rc FROM horses_data_combined_no_nrs_with_past_results GROUP BY pr_10_race_type) rt5 ON t1.pr_10_race_type = rt5.pr_10_race_type

WHERE 1
AND LOWER(country) IN ('scotland', 'england', 'northern ireland', 'eire', 'wales')
#AND race_date > '2018-06-01'
#LIMIT 100
;


SELECT COUNT(1) FROM training_data_5_pr;
SELECT COUNT(1) FROM training_data_10_pr;
SELECT COUNT(1) FROM training_data_5_pr_no_nrs;
SELECT COUNT(1) FROM training_data_10_pr_no_nrs;

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
;


SELECT DISTINCT race_id, SUBSTRING_INDEX(trifecta, ' ', 1)
FROM horses_data
LIMIT 1000;
SELECT DISTINCT SUBSTRING_INDEX(tricast, ' ', 1) FROM horses_data LIMIT 1000;
