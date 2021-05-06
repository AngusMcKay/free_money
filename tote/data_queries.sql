USE horses;

SHOW FULL PROCESSLIST;
KILL 47;

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
SELECT * FROM horses_data_combined_no_nrs_with_past_results LIMIT 10;
DROP TABLE IF EXISTS training_data_5_pr_no_nrs;
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

DROP TABLE IF EXISTS training_data_10_pr_no_nrs;
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

SELECT COUNT(DISTINCT jockey_id) FROM training_data_10_pr_no_nrs
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

SELECT * FROM training_data_10_pr_no_nrs LIMIT 10;

# Add 'relative' pr variables, e.g. current race yards divided by past race yards, or indicator if going and weather is similar
DROP TABLE training_data_6_relative_pr_no_nrs;
CREATE TABLE training_data_6_relative_pr_no_nrs AS
SELECT h.*, weather_horse_time_rc, going_grouped_horse_time_rc, race_type_horse_time_rc,
		ROUND(yards*1.0/pr_1_yards, 3) AS pr_1_relative_yards,
        ROUND(yards*1.0/pr_2_yards, 3) AS pr_2_relative_yards,
        ROUND(yards*1.0/pr_3_yards, 3) AS pr_3_relative_yards,
        ROUND(yards*1.0/pr_4_yards, 3) AS pr_4_relative_yards,
        ROUND(yards*1.0/pr_5_yards, 3) AS pr_5_relative_yards,
        ROUND(yards*1.0/pr_6_yards, 3) AS pr_6_relative_yards, 
		
        CASE	WHEN going_main IN ('fast', 'firm') THEN 4
				WHEN going_main IN ('good to firm') THEN 3.5
				WHEN going_main IN ('standard', 'good') THEN 3
				WHEN going_main IN ('good to soft', 'good to yielding') THEN 2.5
                WHEN going_main IN ('yielding to soft', 'yielding') THEN 2
                WHEN going_main IN ('slow', 'soft') THEN 1
                WHEN going_main IN ('sloppy', 'heavy', 'very soft') THEN 1
                ELSE 2.5
		END AS going_numerical,
        CASE	WHEN pr_1_going_main IN ('fast', 'firm') THEN 4
				WHEN pr_1_going_main IN ('good to firm') THEN 3.5
				WHEN pr_1_going_main IN ('standard', 'good') THEN 3
				WHEN pr_1_going_main IN ('good to soft', 'good to yielding') THEN 2.5
                WHEN pr_1_going_main IN ('yielding to soft', 'yielding') THEN 2
                WHEN pr_1_going_main IN ('slow', 'soft') THEN 1
                WHEN pr_1_going_main IN ('sloppy', 'heavy', 'very soft') THEN 1
                ELSE 2.5
		END AS pr_1_going_numerical,
        CASE	WHEN pr_2_going_main IN ('fast', 'firm') THEN 4
				WHEN pr_2_going_main IN ('good to firm') THEN 3.5
				WHEN pr_2_going_main IN ('standard', 'good') THEN 3
				WHEN pr_2_going_main IN ('good to soft', 'good to yielding') THEN 2.5
                WHEN pr_2_going_main IN ('yielding to soft', 'yielding') THEN 2
                WHEN pr_2_going_main IN ('slow', 'soft') THEN 1
                WHEN pr_2_going_main IN ('sloppy', 'heavy', 'very soft') THEN 1
                ELSE 2.5
		END AS pr_2_going_numerical,
        CASE	WHEN pr_3_going_main IN ('fast', 'firm') THEN 4
				WHEN pr_3_going_main IN ('good to firm') THEN 3.5
				WHEN pr_3_going_main IN ('standard', 'good') THEN 3
				WHEN pr_3_going_main IN ('good to soft', 'good to yielding') THEN 2.5
                WHEN pr_3_going_main IN ('yielding to soft', 'yielding') THEN 2
                WHEN pr_3_going_main IN ('slow', 'soft') THEN 1
                WHEN pr_3_going_main IN ('sloppy', 'heavy', 'very soft') THEN 1
                ELSE 2.5
		END AS pr_3_going_numerical,
        CASE	WHEN pr_4_going_main IN ('fast', 'firm') THEN 4
				WHEN pr_4_going_main IN ('good to firm') THEN 3.5
				WHEN pr_4_going_main IN ('standard', 'good') THEN 3
				WHEN pr_4_going_main IN ('good to soft', 'good to yielding') THEN 2.5
                WHEN pr_4_going_main IN ('yielding to soft', 'yielding') THEN 2
                WHEN pr_4_going_main IN ('slow', 'soft') THEN 1
                WHEN pr_4_going_main IN ('sloppy', 'heavy', 'very soft') THEN 1
                ELSE 2.5
		END AS pr_4_going_numerical,
        CASE	WHEN pr_5_going_main IN ('fast', 'firm') THEN 4
				WHEN pr_5_going_main IN ('good to firm') THEN 3.5
				WHEN pr_5_going_main IN ('standard', 'good') THEN 3
				WHEN pr_5_going_main IN ('good to soft', 'good to yielding') THEN 2.5
                WHEN pr_5_going_main IN ('yielding to soft', 'yielding') THEN 2
                WHEN pr_5_going_main IN ('slow', 'soft') THEN 1
                WHEN pr_5_going_main IN ('sloppy', 'heavy', 'very soft') THEN 1
                ELSE 2.5
		END AS pr_5_going_numerical,
        CASE	WHEN pr_6_going_main IN ('fast', 'firm') THEN 4
				WHEN pr_6_going_main IN ('good to firm') THEN 3.5
				WHEN pr_6_going_main IN ('standard', 'good') THEN 3
				WHEN pr_6_going_main IN ('good to soft', 'good to yielding') THEN 2.5
                WHEN pr_6_going_main IN ('yielding to soft', 'yielding') THEN 2
                WHEN pr_6_going_main IN ('slow', 'soft') THEN 1
                WHEN pr_6_going_main IN ('sloppy', 'heavy', 'very soft') THEN 1
                ELSE 2.5
		END AS pr_6_going_numerical,
        
        CASE	WHEN race_type_devised IN ('chase') THEN 4
				WHEN race_type_devised IN ('hurdle', 'jumps') THEN 3
                WHEN race_type_devised IN ('flat', 'devisedflat') THEN 1
                ELSE 2
		END AS race_type_numerical,
        CASE	WHEN pr_1_race_type_devised IN ('chase') THEN 4
				WHEN pr_1_race_type_devised IN ('hurdle', 'jumps') THEN 3
                WHEN pr_1_race_type_devised IN ('flat', 'devisedflat') THEN 1
                ELSE 2
		END AS pr_1_race_type_numerical,
        CASE	WHEN pr_2_race_type_devised IN ('chase') THEN 4
				WHEN pr_2_race_type_devised IN ('hurdle', 'jumps') THEN 3
                WHEN pr_2_race_type_devised IN ('flat', 'devisedflat') THEN 1
                ELSE 2
		END AS pr_2_race_type_numerical,
        CASE	WHEN pr_3_race_type_devised IN ('chase') THEN 4
				WHEN pr_3_race_type_devised IN ('hurdle', 'jumps') THEN 3
                WHEN pr_3_race_type_devised IN ('flat', 'devisedflat') THEN 1
                ELSE 2
		END AS pr_3_race_type_numerical,
        CASE	WHEN pr_4_race_type_devised IN ('chase') THEN 4
				WHEN pr_4_race_type_devised IN ('hurdle', 'jumps') THEN 3
                WHEN pr_4_race_type_devised IN ('flat', 'devisedflat') THEN 1
                ELSE 2
		END AS pr_4_race_type_numerical,
        CASE	WHEN pr_5_race_type_devised IN ('chase') THEN 4
				WHEN pr_5_race_type_devised IN ('hurdle', 'jumps') THEN 3
                WHEN pr_5_race_type_devised IN ('flat', 'devisedflat') THEN 1
                ELSE 2
		END AS pr_5_race_type_numerical,
        CASE	WHEN pr_6_race_type_devised IN ('chase') THEN 4
				WHEN pr_6_race_type_devised IN ('hurdle', 'jumps') THEN 3
                WHEN pr_6_race_type_devised IN ('flat', 'devisedflat') THEN 1
                ELSE 2
		END AS pr_6_race_type_numerical,
        
        CASE	WHEN h.weather_grouped IN ('snowandhail') THEN 1
				WHEN h.weather_grouped IN ('wet', 'rain') THEN 2
                WHEN h.weather_grouped IN ('overcast', 'cloudy') THEN 4
                WHEN h.weather_grouped IN ('fine', 'dry') THEN 5
                WHEN h.weather_grouped IN ('sunny', 'bright') THEN 6
                ELSE 3
		END AS weather_numerical,
        CASE	WHEN pr_1_weather_grouped IN ('snowandhail') THEN 1
				WHEN pr_1_weather_grouped IN ('wet', 'rain') THEN 2
                WHEN pr_1_weather_grouped IN ('overcast', 'cloudy') THEN 4
                WHEN pr_1_weather_grouped IN ('fine', 'dry') THEN 5
                WHEN pr_1_weather_grouped IN ('sunny', 'bright') THEN 6
                ELSE 3
		END AS pr_1_weather_numerical,
        CASE	WHEN pr_2_weather_grouped IN ('snowandhail') THEN 1
				WHEN pr_2_weather_grouped IN ('wet', 'rain') THEN 2
                WHEN pr_2_weather_grouped IN ('overcast', 'cloudy') THEN 4
                WHEN pr_2_weather_grouped IN ('fine', 'dry') THEN 5
                WHEN pr_2_weather_grouped IN ('sunny', 'bright') THEN 6
                ELSE 3
		END AS pr_2_weather_numerical,
        CASE	WHEN pr_3_weather_grouped IN ('snowandhail') THEN 1
				WHEN pr_3_weather_grouped IN ('wet', 'rain') THEN 2
                WHEN pr_3_weather_grouped IN ('overcast', 'cloudy') THEN 4
                WHEN pr_3_weather_grouped IN ('fine', 'dry') THEN 5
                WHEN pr_3_weather_grouped IN ('sunny', 'bright') THEN 6
                ELSE 3
		END AS pr_3_weather_numerical,
        CASE	WHEN pr_4_weather_grouped IN ('snowandhail') THEN 1
				WHEN pr_4_weather_grouped IN ('wet', 'rain') THEN 2
                WHEN pr_4_weather_grouped IN ('overcast', 'cloudy') THEN 4
                WHEN pr_4_weather_grouped IN ('fine', 'dry') THEN 5
                WHEN pr_4_weather_grouped IN ('sunny', 'bright') THEN 6
                ELSE 3
		END AS pr_4_weather_numerical,
        CASE	WHEN pr_5_weather_grouped IN ('snowandhail') THEN 1
				WHEN pr_5_weather_grouped IN ('wet', 'rain') THEN 2
                WHEN pr_5_weather_grouped IN ('overcast', 'cloudy') THEN 4
                WHEN pr_5_weather_grouped IN ('fine', 'dry') THEN 5
                WHEN pr_5_weather_grouped IN ('sunny', 'bright') THEN 6
                ELSE 3
		END AS pr_5_weather_numerical,
        CASE	WHEN pr_6_weather_grouped IN ('snowandhail') THEN 1
				WHEN pr_6_weather_grouped IN ('wet', 'rain') THEN 2
                WHEN pr_6_weather_grouped IN ('overcast', 'cloudy') THEN 4
                WHEN pr_6_weather_grouped IN ('fine', 'dry') THEN 5
                WHEN pr_6_weather_grouped IN ('sunny', 'bright') THEN 6
                ELSE 3
		END AS pr_6_weather_numerical

FROM horses_data_combined_no_nrs_with_past_results h
LEFT JOIN (SELECT weather_grouped, AVG(horse_time) AS weather_horse_time_rc FROM horses_data_combined_no_nrs_with_past_results GROUP BY weather_grouped) w ON h.weather_grouped = w.weather_grouped
LEFT JOIN (SELECT going_grouped, AVG(horse_time) AS going_grouped_horse_time_rc FROM horses_data_combined_no_nrs_with_past_results GROUP BY going_grouped) g ON h.going_grouped = g.going_grouped
LEFT JOIN (SELECT race_type, AVG(horse_time) AS race_type_horse_time_rc FROM horses_data_combined_no_nrs_with_past_results GROUP BY race_type) rt ON h.race_type = rt.race_type
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

SELECT COUNT(1) FROM training_data_6_relative_pr_no_nrs;

# Add 'relative' pr variables, e.g. current race yards divided by past race yards, or indicator if going and weather is similar
DROP TABLE training_data_6_relative_pr;
CREATE TABLE training_data_6_relative_pr AS
SELECT h.*, weather_horse_time_rc, going_grouped_horse_time_rc, race_type_horse_time_rc,
		ROUND(yards*1.0/pr_1_yards, 3) AS pr_1_relative_yards,
        ROUND(yards*1.0/pr_2_yards, 3) AS pr_2_relative_yards,
        ROUND(yards*1.0/pr_3_yards, 3) AS pr_3_relative_yards,
        ROUND(yards*1.0/pr_4_yards, 3) AS pr_4_relative_yards,
        ROUND(yards*1.0/pr_5_yards, 3) AS pr_5_relative_yards,
        ROUND(yards*1.0/pr_6_yards, 3) AS pr_6_relative_yards, 
		
        CASE	WHEN going_main IN ('fast', 'firm') THEN 4
				WHEN going_main IN ('good to firm') THEN 3.5
				WHEN going_main IN ('standard', 'good') THEN 3
				WHEN going_main IN ('good to soft', 'good to yielding') THEN 2.5
                WHEN going_main IN ('yielding to soft', 'yielding') THEN 2
                WHEN going_main IN ('slow', 'soft') THEN 1
                WHEN going_main IN ('sloppy', 'heavy', 'very soft') THEN 1
                ELSE 2.5
		END AS going_numerical,
        CASE	WHEN pr_1_going_main IN ('fast', 'firm') THEN 4
				WHEN pr_1_going_main IN ('good to firm') THEN 3.5
				WHEN pr_1_going_main IN ('standard', 'good') THEN 3
				WHEN pr_1_going_main IN ('good to soft', 'good to yielding') THEN 2.5
                WHEN pr_1_going_main IN ('yielding to soft', 'yielding') THEN 2
                WHEN pr_1_going_main IN ('slow', 'soft') THEN 1
                WHEN pr_1_going_main IN ('sloppy', 'heavy', 'very soft') THEN 1
                ELSE 2.5
		END AS pr_1_going_numerical,
        CASE	WHEN pr_2_going_main IN ('fast', 'firm') THEN 4
				WHEN pr_2_going_main IN ('good to firm') THEN 3.5
				WHEN pr_2_going_main IN ('standard', 'good') THEN 3
				WHEN pr_2_going_main IN ('good to soft', 'good to yielding') THEN 2.5
                WHEN pr_2_going_main IN ('yielding to soft', 'yielding') THEN 2
                WHEN pr_2_going_main IN ('slow', 'soft') THEN 1
                WHEN pr_2_going_main IN ('sloppy', 'heavy', 'very soft') THEN 1
                ELSE 2.5
		END AS pr_2_going_numerical,
        CASE	WHEN pr_3_going_main IN ('fast', 'firm') THEN 4
				WHEN pr_3_going_main IN ('good to firm') THEN 3.5
				WHEN pr_3_going_main IN ('standard', 'good') THEN 3
				WHEN pr_3_going_main IN ('good to soft', 'good to yielding') THEN 2.5
                WHEN pr_3_going_main IN ('yielding to soft', 'yielding') THEN 2
                WHEN pr_3_going_main IN ('slow', 'soft') THEN 1
                WHEN pr_3_going_main IN ('sloppy', 'heavy', 'very soft') THEN 1
                ELSE 2.5
		END AS pr_3_going_numerical,
        CASE	WHEN pr_4_going_main IN ('fast', 'firm') THEN 4
				WHEN pr_4_going_main IN ('good to firm') THEN 3.5
				WHEN pr_4_going_main IN ('standard', 'good') THEN 3
				WHEN pr_4_going_main IN ('good to soft', 'good to yielding') THEN 2.5
                WHEN pr_4_going_main IN ('yielding to soft', 'yielding') THEN 2
                WHEN pr_4_going_main IN ('slow', 'soft') THEN 1
                WHEN pr_4_going_main IN ('sloppy', 'heavy', 'very soft') THEN 1
                ELSE 2.5
		END AS pr_4_going_numerical,
        CASE	WHEN pr_5_going_main IN ('fast', 'firm') THEN 4
				WHEN pr_5_going_main IN ('good to firm') THEN 3.5
				WHEN pr_5_going_main IN ('standard', 'good') THEN 3
				WHEN pr_5_going_main IN ('good to soft', 'good to yielding') THEN 2.5
                WHEN pr_5_going_main IN ('yielding to soft', 'yielding') THEN 2
                WHEN pr_5_going_main IN ('slow', 'soft') THEN 1
                WHEN pr_5_going_main IN ('sloppy', 'heavy', 'very soft') THEN 1
                ELSE 2.5
		END AS pr_5_going_numerical,
        CASE	WHEN pr_6_going_main IN ('fast', 'firm') THEN 4
				WHEN pr_6_going_main IN ('good to firm') THEN 3.5
				WHEN pr_6_going_main IN ('standard', 'good') THEN 3
				WHEN pr_6_going_main IN ('good to soft', 'good to yielding') THEN 2.5
                WHEN pr_6_going_main IN ('yielding to soft', 'yielding') THEN 2
                WHEN pr_6_going_main IN ('slow', 'soft') THEN 1
                WHEN pr_6_going_main IN ('sloppy', 'heavy', 'very soft') THEN 1
                ELSE 2.5
		END AS pr_6_going_numerical,
        
        CASE	WHEN race_type_devised IN ('chase') THEN 4
				WHEN race_type_devised IN ('hurdle', 'jumps') THEN 3
                WHEN race_type_devised IN ('flat', 'devisedflat') THEN 1
                ELSE 2
		END AS race_type_numerical,
        CASE	WHEN pr_1_race_type_devised IN ('chase') THEN 4
				WHEN pr_1_race_type_devised IN ('hurdle', 'jumps') THEN 3
                WHEN pr_1_race_type_devised IN ('flat', 'devisedflat') THEN 1
                ELSE 2
		END AS pr_1_race_type_numerical,
        CASE	WHEN pr_2_race_type_devised IN ('chase') THEN 4
				WHEN pr_2_race_type_devised IN ('hurdle', 'jumps') THEN 3
                WHEN pr_2_race_type_devised IN ('flat', 'devisedflat') THEN 1
                ELSE 2
		END AS pr_2_race_type_numerical,
        CASE	WHEN pr_3_race_type_devised IN ('chase') THEN 4
				WHEN pr_3_race_type_devised IN ('hurdle', 'jumps') THEN 3
                WHEN pr_3_race_type_devised IN ('flat', 'devisedflat') THEN 1
                ELSE 2
		END AS pr_3_race_type_numerical,
        CASE	WHEN pr_4_race_type_devised IN ('chase') THEN 4
				WHEN pr_4_race_type_devised IN ('hurdle', 'jumps') THEN 3
                WHEN pr_4_race_type_devised IN ('flat', 'devisedflat') THEN 1
                ELSE 2
		END AS pr_4_race_type_numerical,
        CASE	WHEN pr_5_race_type_devised IN ('chase') THEN 4
				WHEN pr_5_race_type_devised IN ('hurdle', 'jumps') THEN 3
                WHEN pr_5_race_type_devised IN ('flat', 'devisedflat') THEN 1
                ELSE 2
		END AS pr_5_race_type_numerical,
        CASE	WHEN pr_6_race_type_devised IN ('chase') THEN 4
				WHEN pr_6_race_type_devised IN ('hurdle', 'jumps') THEN 3
                WHEN pr_6_race_type_devised IN ('flat', 'devisedflat') THEN 1
                ELSE 2
		END AS pr_6_race_type_numerical,
        
        CASE	WHEN h.weather_grouped IN ('snowandhail') THEN 1
				WHEN h.weather_grouped IN ('wet', 'rain') THEN 2
                WHEN h.weather_grouped IN ('overcast', 'cloudy') THEN 4
                WHEN h.weather_grouped IN ('fine', 'dry') THEN 5
                WHEN h.weather_grouped IN ('sunny', 'bright') THEN 6
                ELSE 3
		END AS weather_numerical,
        CASE	WHEN pr_1_weather_grouped IN ('snowandhail') THEN 1
				WHEN pr_1_weather_grouped IN ('wet', 'rain') THEN 2
                WHEN pr_1_weather_grouped IN ('overcast', 'cloudy') THEN 4
                WHEN pr_1_weather_grouped IN ('fine', 'dry') THEN 5
                WHEN pr_1_weather_grouped IN ('sunny', 'bright') THEN 6
                ELSE 3
		END AS pr_1_weather_numerical,
        CASE	WHEN pr_2_weather_grouped IN ('snowandhail') THEN 1
				WHEN pr_2_weather_grouped IN ('wet', 'rain') THEN 2
                WHEN pr_2_weather_grouped IN ('overcast', 'cloudy') THEN 4
                WHEN pr_2_weather_grouped IN ('fine', 'dry') THEN 5
                WHEN pr_2_weather_grouped IN ('sunny', 'bright') THEN 6
                ELSE 3
		END AS pr_2_weather_numerical,
        CASE	WHEN pr_3_weather_grouped IN ('snowandhail') THEN 1
				WHEN pr_3_weather_grouped IN ('wet', 'rain') THEN 2
                WHEN pr_3_weather_grouped IN ('overcast', 'cloudy') THEN 4
                WHEN pr_3_weather_grouped IN ('fine', 'dry') THEN 5
                WHEN pr_3_weather_grouped IN ('sunny', 'bright') THEN 6
                ELSE 3
		END AS pr_3_weather_numerical,
        CASE	WHEN pr_4_weather_grouped IN ('snowandhail') THEN 1
				WHEN pr_4_weather_grouped IN ('wet', 'rain') THEN 2
                WHEN pr_4_weather_grouped IN ('overcast', 'cloudy') THEN 4
                WHEN pr_4_weather_grouped IN ('fine', 'dry') THEN 5
                WHEN pr_4_weather_grouped IN ('sunny', 'bright') THEN 6
                ELSE 3
		END AS pr_4_weather_numerical,
        CASE	WHEN pr_5_weather_grouped IN ('snowandhail') THEN 1
				WHEN pr_5_weather_grouped IN ('wet', 'rain') THEN 2
                WHEN pr_5_weather_grouped IN ('overcast', 'cloudy') THEN 4
                WHEN pr_5_weather_grouped IN ('fine', 'dry') THEN 5
                WHEN pr_5_weather_grouped IN ('sunny', 'bright') THEN 6
                ELSE 3
		END AS pr_5_weather_numerical,
        CASE	WHEN pr_6_weather_grouped IN ('snowandhail') THEN 1
				WHEN pr_6_weather_grouped IN ('wet', 'rain') THEN 2
                WHEN pr_6_weather_grouped IN ('overcast', 'cloudy') THEN 4
                WHEN pr_6_weather_grouped IN ('fine', 'dry') THEN 5
                WHEN pr_6_weather_grouped IN ('sunny', 'bright') THEN 6
                ELSE 3
		END AS pr_6_weather_numerical

FROM horses_data_combined_with_past_results h
LEFT JOIN (SELECT weather_grouped, AVG(horse_time) AS weather_horse_time_rc FROM horses_data_combined_with_past_results GROUP BY weather_grouped) w ON h.weather_grouped = w.weather_grouped
LEFT JOIN (SELECT going_grouped, AVG(horse_time) AS going_grouped_horse_time_rc FROM horses_data_combined_with_past_results GROUP BY going_grouped) g ON h.going_grouped = g.going_grouped
LEFT JOIN (SELECT race_type, AVG(horse_time) AS race_type_horse_time_rc FROM horses_data_combined_with_past_results GROUP BY race_type) rt ON h.race_type = rt.race_type
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


# relevant features from jockey data to use in horse vs horse model
SELECT *
FROM jockeys_data_combined_no_nrs_with_past_results
WHERE horse_time IS NOT NULL
AND pr_1_finish_position_for_ordering = 999
LIMIT 30;

SELECT * FROM jockeys_data_combined_no_non_runners LIMIT 20;

SELECT * FROM training_data_6_relative_pr LIMIT 100;

SELECT * FROM horses_data_combined_no_non_runners hd
INNER JOIN (SELECT * FROM (SELECT horse_id, race_id, yards/horse_time as speed FROM horses_data_combined) sp1 WHERE speed < 5 LIMIT 100) sp
ON hd.horse_id = sp.horse_id AND hd.race_id = sp.race_id
LIMIT 100
;
