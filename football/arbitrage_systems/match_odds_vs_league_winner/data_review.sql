USE league_arb;

SELECT * FROM league_odds ORDER BY datetime_extracted LIMIT 10;

select * FROM match_odds LIMIT 10;

SELECT * FROM league_odds
WHERE runner_name LIKE '%Man%Utd%' AND market_type = 'WINNER'
ORDER BY datetime_extracted DESC LIMIT 10;

